# ---------------------------------------------------
# ---------------LCM-LoRA Studio---------------------
# -----------------Version 1.5b----------------------
# ---------------------------------------------------
# main application
# Filename: lcm-lora-studio.py
# ---------------------------------------------------
# Copyright (C) 2025 Rock Stevens
# ---------------------------------------------------

# ---------------------------------
# base libraries, imports for the app, GUI, etc...
import gradio as gr
import time 
import os 
import sys
import random
import subprocess
import ctypes
os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'

# ---------------------------------
# base libraries for torch, need for Diffusion
import torch

# -------------------------------
# base libraries, imports for SD Pipelines and Scheduler
import string
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline, LCMScheduler)

# --------------------------------
# for ControlNet only
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ---------------------------------
# imports for seed - I like this method better than generate.
from diffusers.training_utils import set_seed

# ---------------------------------
# general image import
# and for the Image Processing section
from PIL import Image, ImageEnhance, ImageOps

# ------------------------------------------------------------
# Image gallery
import re
import pathlib
from pathlib import Path
import html
import base64

# ------------------------------------------------------------
# copy last prompt and last image to gallery 'Send to Gallery'
# file copy
import shutil

# -------------------------------------------------
# InstructPix2Pix imports
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionXLInstructPix2PixPipeline

# -------------------------------------------------
# GUI Image as an Input
from diffusers.utils import load_image


# -------------------------------------------------
# Custom imports - LCM-LoRA Studio model conversion routines
from utils.lcm_convert_diffusers_to_original_stable_diffusion import convert_sd_to_safetensors
from utils.lcm_convert_diffusers_to_original_sdxl import convert_sdxl_to_safetensors

# -------------------------------------------------
# imports, settings, read model config settings, etc...
import json

# ----------------------------------------
# LLSTUDIO import a few starting variables to seed the app.
# although most are built from multiple STUDIO variables later
# See 'config.py' that's where they are.
from config import LLSTUDIO

# ----------------------------------------
# imports, faster math
import pandas as pd

# ------------------------------------------
# imports, raw image processing
import numpy as np

# ------------------------------------------
# diffusers verbose output control inports
from diffusers import logging

# ---------------------------------
# context manager to ignore the UserWarning during model loading
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------
# get processor info and OS info, memory garbage collection etc...
import platform
import gc

# ---------------------------------------------------------
# download huggingface repository/model, auto find cache dir
from huggingface_hub import snapshot_download, constants

# ---------------------------------
# get memory info 
import psutil

# ---------------------------------
# image sd upscale
from diffusers import StableDiffusionLatentUpscalePipeline

# ---------------------------------
# load separate text encoder
import transformers
from transformers import CLIPTextModel

# ----------------------------------------
# mainly for date/time in image filename
from datetime import datetime

# ----------------------------------------
# OpenCV for image processing section, to do edge detection.
# although already an apparent 'requirement' for 
# another library already used/installed
# but we still need to import it ! :)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not found. So, Canny edge detection will be forced to use a slower processing method")
    print("If running headless, did you forget to install some drivers? Namely 'libgl1-mesa-dev' ?")
    print("Running a 'headless' Linux (Pi5), Type: 'sudo apt-get install libgl1-mesa-dev'")
    print("This will install the drivers needed for headless operation. Then restart LCM-LoRA Studio.")


# ---------------------------------
# 'config.py' imports
#----------------------------------
# ===========DICT-BASED-APP-SETTINGS-LOAD-SAVE-FUNCTIONS=======================
# App settings are a dict named 'STUDIO' in 'config.py'
# Already populated with default settings there.
# The rest of the settings are loaded from the default settings file into the dict
# when the app starts, if exists. If not, the dict in 'config.py' is written and saved.
# So, to default the app settings, delete the JSON settings file and restart.

# main app settings, linked to the 'settings' tab in ui
from config import STUDIO, load_settings, save_settings, update_settings, create_settings_ui

# simple sd pipeline class to model type, generation/pipeline mode - lookup table
from config import PIPECLASSES

# status of our single pipeline, model loaded?, which one?, text encoder?, type?, etc...
from config import SDPIPELINE

# class list for pipeline class dropdown boxes
from config import PIPELINE_CLASSES

# -------------------------------------------------
# NOTE: This is the ONLY PIPELINE on CPU ONLY !
from config import pipeline
pipeline = None
# -------------------------------------------------

# imports for COMPEL prompt parsing
# this import is conditional on COMPEL version
# the program can run without the prompt 'weighting' this library provides
# yet your image output will gain some from using it.
# the application will run without it.
LLSTUDIO["compel_installed"] = 0      # default - not installed
try:
    from compel import Compel, ReturnedEmbeddingsType
    LLSTUDIO["compel_installed"] = 1
except ImportError:
    LLSTUDIO["compel_installed"] = 0


# ----------end of imports-------------------------



# ----------------------start of settings---------------------------------------
# we do settings first, then set up all the variables


# load the app settings from the settings JSON file and update 'STUDIO'
STUDIO.update(load_settings())

# ----------end of settings-------------------------



# ----------start of setup variables from settings-------------------------------
# these variables are 'built' from the 'STUDIO' settings.
# some 'STUDIO' settings are used directly, some are used combined with 
# other's to build some of the variables. ie... fullpath = root_dir/dir


# -------------------------------------------------------------------------------
# flag to halt multi image generation, after current inference is finished
LLSTUDIO["halt_gen"] = 0

# ====================================================
# find out which OS, Windows/Linux by (proper cased) OS name
LLSTUDIO["current_os"] = platform.system()

# ====================================================
# debug levels
# # 0=nothing 1=app 2=important info 3=superflurious info 4=ALL info (TMI)
# debug level - turn on/off printing info to stdout
# 0 = Nothing out from app controled print output
# 1 = app print outputs important info
# 2 = app print outputs important info + superflurious model loading output... TMI

# ---------------------------------------
# show the Debug Level when App starts, if greater than 0
if int(STUDIO["app_debug"]["value"]) > 0: print("Debug Level: " + STUDIO["app_debug"]["value"])


# ---------------------------------------
# used for help, should be '.'
LLSTUDIO["root_dir"] = STUDIO["root_dir"]["value"]


# --------------------------------------------------------------
LLSTUDIO["lcm_model_list"]=['NO MODEL', 'NO MODEL']
LLSTUDIO["lcm_sdonly_model_list"] = ['NO MODEL', 'NO MODEL']

LLSTUDIO["lcm_model_dir"] = os.path.join(STUDIO["lcm_model_rootdir"]["value"], STUDIO["lcm_model_dir"]["value"])


# ----------------------------------------------------------------------
# Auto/Manual (app settings) Huggingface Cache location and model list
LLSTUDIO["hub_model_list"]=['NO MODEL', 'NO MODEL']
if STUDIO["hub_model_dir"]["value"] != "":
    LLSTUDIO["hub_model_dir"] = STUDIO["hub_model_dir"]["value"]
else:
    LLSTUDIO["hub_model_dir"] = constants.HUGGINGFACE_HUB_CACHE


# ----------------------------------------------------------------------
# Huggingface Hub ONLINE or OFFLINE by default on app start
if STUDIO["hub_online"]["value"]:
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
else:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


# ----------------------------------------------------------------------
# LCM-LoRA Models 
# ALL models YOU SAVE by this app models will go here
LLSTUDIO["lcm_model_image_list"]=['NO MODEL', 'NO MODEL']
LLSTUDIO["lcm_model_image_dir"] = os.path.join(STUDIO["lcm_model_image_rootdir"]["value"], STUDIO["lcm_model_image_dir"]["value"])
data_lcmdir_path =  Path(LLSTUDIO["lcm_model_image_dir"])

# ----------------------------------------------------------------------
# Safetensors Models 
# ALL Single File Safetensors type Base models you get go here
LLSTUDIO["safe_model_list"] = ['NO MODEL', 'NO MODEL']
LLSTUDIO["safe_model_dir"] = os.path.join(STUDIO["safe_model_rootdir"]["value"], STUDIO["safe_model_dir"]["value"])

# ----------------------------------------------------------------------
# Safetensors Model Image Gallery
# ALL Single File Safetensors type Base models Image Gallery
LLSTUDIO["safe_model_image_list"] = ['NO MODEL', 'NO MODEL']
LLSTUDIO["safe_model_image_dir"] = os.path.join(STUDIO["safe_model_image_rootdir"]["value"], STUDIO["safe_model_image_dir"]["value"])
data_safedir_path =  Path(LLSTUDIO["safe_model_image_dir"])

# ----------------------------------------------------------------------
# LoRA Models 
# ALL Single File Safetensors LoRA type models you get go here
LLSTUDIO["lora_model_list"]=['NO MODEL', 'NO MODEL']

LLSTUDIO["lora_model_dir"] = os.path.join(STUDIO["lora_model_rootdir"]["value"], STUDIO["lora_model_dir"]["value"])

# ----------------------------------------------------------------------
# LoRA Models Image Gallery
# ALL Single File Safetensors LoRA type models Image Gallery
LLSTUDIO["lora_model_image_list"]=['NO MODEL', 'NO MODEL']
LLSTUDIO["lora_model_image_dir"] = os.path.join(STUDIO["lora_model_image_rootdir"]["value"], STUDIO["lora_model_image_dir"]["value"])
data_loradir_path =  Path(LLSTUDIO["lora_model_image_dir"])


# ----------------------------------------------------------------------
# Output Image Gallery
# ALL Generated images Image Gallery
LLSTUDIO["output_image_dir"] = os.path.join(STUDIO["output_image_rootdir"]["value"], STUDIO["output_image_dir"]["value"])
data_outputdir_path =  Path(LLSTUDIO["output_image_dir"])


# ----------------------------------------------------------------------
# HELP - EX URL: http://127.0.0.1:7860/gradio_api/file/help/index.html
LLSTUDIO["help_path"] = os.path.join(LLSTUDIO["root_dir"], "help")
help_path =  Path(LLSTUDIO["help_path"])


# ----------------------------------------------------------------------
# Allowed Paths for Browser file access, for help and Image galleries
all_allowed_file_paths = [str(help_path.absolute()),str(data_safedir_path.absolute()),str(data_lcmdir_path.absolute()),str(data_loradir_path.absolute()),str(data_outputdir_path.absolute())]


# ----------------------------------------------------------------------
# Image genration and Image gallery settings
LLSTUDIO['last_prompt_filename'] = ""
LLSTUDIO['last_negative_prompt_filename'] = ""
LLSTUDIO['last_image_filename'] = ""


# ----------------------------------------------------------------------
# LoRA model settings - Used to keep up with LoRas loaded
# There is a good reason for doing this 
# 'outside' of the diffusers lib :)
# ----------------------------------------------------------------------
LLSTUDIO["lora_adapter_numb"] = 0
LLSTUDIO["loaded_lora_model_value"] = []
LLSTUDIO["loaded_lora_model_name"] = []
LLSTUDIO["loaded_lora_model_adapter"] = []
LLSTUDIO["loaded_lora_model_list"] = []

# ====================================================================================

# -----------generic stuff--------------
if int(STUDIO["app_debug"]["value"]) < 2: logging.set_verbosity_error() 

# ------------------------------------------------------------------
# holds a global page number to remember for each image gallery
LLSTUDIO["output_page_num"]=1
LLSTUDIO["lcm_page_num"]=1
LLSTUDIO["safe_page_num"]=1
LLSTUDIO["lora_page_num"]=1

# ------------------------------------------------------------------
# our device name
LLSTUDIO["device"] = "cpu"
LLSTUDIO["friendly_device_name"] = "CPU"

# ------------------------------------------------------------------
# enables/disables hidden image to visible image
# on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
# it's a FLAG, don't touch it !
LLSTUDIO["hidden_image_flag"] = 0

# ------------------------------------------------------
# advanced image gallery
LLSTUDIO["advanced_gallery_dir"] = os.path.join(STUDIO["advanced_gallery_root"]["value"], STUDIO["advanced_gallery_dir"]["value"])
LLSTUDIO["gallery_selected_image"] = ""

# ------------------------------------------------------
# ControlNet defines
LLSTUDIO["cnet_model_name_list"] = ["MLSD Line Detection","HED Edge Detection","Depth Estimation","Scribble","Canny", "Normal Map Estimation", "Image Segmentation", "OpenPose"]

CNETMODELS = {
    "MLSD Line Detection": "lllyasviel/sd-controlnet-mlsd",
    "HED Edge Detection": "lllyasviel/sd-controlnet-hed",
    "Depth Estimation": "lllyasviel/sd-controlnet-depth",
    "Scribble": "lllyasviel/sd-controlnet-scribble",
    "Canny": "lllyasviel/sd-controlnet-canny",
    "Normal Map Estimation": "lllyasviel/sd-controlnet-normal",
    "Image Segmentation": "lllyasviel/sd-controlnet-seg",
    "OpenPose": "lllyasviel/sd-controlnet-openpose"
}


# -------------------------------------------------------
# Image Processing Defines
EDGEFILTERS = ["Canny", "Laplacian", "Scharr", "Sobel", "Simple Gradient", "Canny (Numpy)", "Laplacian (Numpy)", "Prewitt (Numpy)", "Roberts Cross (Numpy)", "Sobel (Numpy)"]

# ------------------------------------
# Folders for Image Processing
LLSTUDIO["imgp_file_dir"] = os.path.join(STUDIO["imgp_files_root"]["value"], STUDIO["imgp_files_dir"]["value"])

# ------------------------------------------------------
# List of files - Image Processing
LLSTUDIO["imgp_file_list"] = []

#----------------------------------------
# NO IMAGE image, and list of filters for the dropdown
LLSTUDIO["no_image"] = "no_image.png"



#----------------------------------------
# model merge setup
PROFILES_ROOT = "merge_profiles"
LLSTUDIO["profiles_list"]=['NO PROFILES']
LLSTUDIO["profiles_dir"] = os.path.join(".", PROFILES_ROOT)
os.makedirs(LLSTUDIO["profiles_dir"], exist_ok=True)
is_sdxl_merge = False



# ----------end of defines--------------------

# ----------end of setup variables from settings---------------------------

#----------------------------------------
# this flag, Keeps from starting a second download before last one stops
DOWNLOAD_MODELS_FLAG = False


# ==========================================================================================
# Merge Model Static data
# SD1.5 slider labels
SD15_BLOCK_LABELS = ["00-IN00","01-IN01","02-IN02","03-IN03","04-IN04","05-IN05","06-IN06","07-IN07","08-IN08","09-IN09","10-IN10","11-IN11","12-IN12","13-MID","14-OUT00","15-OUT01","16-OUT02","17-OUT03","18-OUT04","19-OUT05","20-OUT06","21-OUT07","22-OUT08","23-OUT09","24-OUT10","25-OUT11"]
# SDXL1.0 slider labels
SDXL_BLOCK_LABELS = ["00-BASE","01-DOWN0-RES0","02-DOWN0-ATTN0","03-DOWN0-RES1","04-DOWN0-ATTN1","05-DOWN1-RES0","06-DOWN1-ATTN0","07-DOWN1-RES1","08-DOWN1-ATTN1","09-DOWN2-RES0","10-DOWN2-ATTN0","11-DOWN2-RES1","12-DOWN2-ATTN1","13-MID-RES0","14-MID-ATTN0","15-MID-RES1","16-UP0-RES0","17-UP0-ATTN0","18-UP0-RES1","19-UP0-ATTN1","20-UP0-RES2","21-UP0-ATTN2","22-UP1-RES0","23-UP1-ATTN0","24-UP1-RES1","25-UP1-ATTN1","26-UP1-RES2","27-UP1-ATTN2","28-UP2-RES0","29-UP2-ATTN0","30-UP2-RES1","31-UP2-ATTN1","32-UP2-RES2","33-UP2-ATTN2","34-OUT","35-OUT-NORM","36-OUT-CONV","37-FINAL-REFINE","38-FINAL-SHARPEN","39-FINAL"]

# ----------------------
# built-in presets
PRESETS = {
    "Balanced":
        [0.5] * 40,
    "Model A Dominant":
        [0.2] * 40,
    "Model B Dominant":
        [0.8] * 40,
    "Structure from A / Detail from B":
        (
            [0.2] * 14 +
            [0.7] * 26
        ),
    "Style Transfer":
        (
            [0.1] * 12 +
            [0.85] * 8 +
            [0.5] * 20
        ),
    "Detail Booster":
        (
            [0.3] * 20 +
            [0.85] * 20
        ),
    "Composition Keeper":
        (
            [0.05] * 15 +
            [0.65] * 25
        )
}

# ----------end of setup variables---------------------------


# # ====================================================================================
# # ======START========FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====
# # ====================================================================================


# ==========================================================================================
# Merge model functions


# ---------------------------------
def get_block_weight(key, weights, is_sdxl):

    if is_sdxl:
        if "conv_in" in key:
            return weights[0]
        if "down_blocks" in key:
            block_num = int(key.split("down_blocks.")[1].split(".")[0])
            if "resnets.0" in key:
                return weights[1 + (block_num * 4)]
            if "attentions.0" in key:
                return weights[2 + (block_num * 4)]
            if "resnets.1" in key:
                return weights[3 + (block_num * 4)]
            if "attentions.1" in key:
                return weights[4 + (block_num * 4)]
        if "mid_block" in key:
            if "resnets.0" in key:
                return weights[13]
            if "attentions.0" in key:
                return weights[14]
            if "resnets.1" in key:
                return weights[15]
        if "up_blocks" in key:
            block_num = int(key.split("up_blocks.")[1].split(".")[0])
            base = 16 + (block_num * 6)
            if "resnets.0" in key:
                return weights[base]
            if "attentions.0" in key:
                return weights[base + 1]
            if "resnets.1" in key:
                return weights[base + 2]
            if "attentions.1" in key:
                return weights[base + 3]
            if "resnets.2" in key:
                return weights[base + 4]
            if "attentions.2" in key:
                return weights[base + 5]
        if "conv_norm_out" in key:
            return weights[35]
        if "conv_out" in key:
            return weights[36]
        return weights[39]
    else:
        if "time_embed" in key or "conv_in" in key:
            return weights[0]
        if "down_blocks" in key:
            block_num = int(key.split("down_blocks.")[1].split(".")[0])
            sub = 0
            if "resnets" in key:
                sub = int(key.split("resnets.")[1].split(".")[0])
            elif "attentions" in key:
                sub = int(key.split("attentions.")[1].split(".")[0])
            idx = 1 + (block_num * 3) + sub
            return weights[min(idx, 12)]
        if "mid_block" in key:
            return weights[13]
        if "up_blocks" in key:
            block_num = int(key.split("up_blocks.")[1].split(".")[0])
            sub = 0
            if "resnets" in key:
                sub = int(key.split("resnets.")[1].split(".")[0])
            elif "attentions" in key:
                sub = int(key.split("attentions.")[1].split(".")[0])
            idx = 14 + (block_num * 3) + sub
            return weights[min(idx, 25)]
        return 0.5


# ---------------------------------
# simple lookup table
def get_block_description(idx, is_sdxl):

    if is_sdxl:
        if idx == 0:
            return (
                "SDXL Input Block.\n\n"
                "Controls foundational image composition.\n"
                "Strong influence on framing, pose, structure, and prompt adherence.\n\n"
                "Lower values preserve Model A layout.\n"
                "Higher values inject Model B composition behavior."
            )
        elif 1 <= idx <= 9:
            return (
                f"SDXL Down Block {idx}.\n\n"
                "Early feature extraction stage.\n"
                "Influences:\n"
                "- anatomy\n"
                "- perspective\n"
                "- scene structure\n"
                "- object placement\n"
                "- prompt interpretation\n\n"
                "Higher values move toward Model B structure."
            )
        elif idx == 10:
            return (
                "SDXL Mid Block.\n\n"
                "One of the MOST influential layers.\n"
                "Strongly affects:\n"
                "- artistic style\n"
                "- coherence\n"
                "- lighting\n"
                "- overall image identity\n\n"
                "This is often the 'style transfer core' of SDXL merges."
            )
        elif 11 <= idx <= 30:
            return (
                f"SDXL Up Block {idx}.\n\n"
                "Reconstruction and refinement layers.\n"
                "Affects:\n"
                "- textures\n"
                "- detail density\n"
                "- realism\n"
                "- sharpness\n"
                "- skin detail\n"
                "- material appearance\n\n"
                "Higher values favor Model B detailing."
            )
        else:
            return (
                f"SDXL Final Output Block {idx}.\n\n"
                "Late denoising and cleanup stage.\n"
                "Controls:\n"
                "- micro-detail\n"
                "- final sharpness\n"
                "- edge cleanup\n"
                "- noise characteristics\n"
                "- image polish"
            )
    else:
        if idx == 0:
            return (
                "SD1.5 Input Block.\n\n"
                "Controls core composition and image planning.\n"
                "Influences layout, framing, and general scene structure."
            )
        elif 1 <= idx <= 12:
            return (
                f"SD1.5 Down Block {idx}.\n\n"
                "Early U-Net encoding stage.\n"
                "Affects:\n"
                "- shape language\n"
                "- anatomy\n"
                "- perspective\n"
                "- composition\n"
                "- concept interpretation"
            )
        elif idx == 13:
            return (
                "SD1.5 Mid Block.\n\n"
                "Critical style synthesis region.\n"
                "Controls:\n"
                "- artistic style\n"
                "- contrast\n"
                "- lighting\n"
                "- image cohesion"
            )
        elif 14 <= idx <= 24:
            return (
                f"SD1.5 Up Block {idx}.\n\n"
                "Decoder reconstruction layers.\n"
                "Strong effect on:\n"
                "- detail quality\n"
                "- textures\n"
                "- realism\n"
                "- edge quality\n"
                "- rendering precision"
            )
        else:
            return (
                "Unused for SD1.5.\n\n"
                "SD1.5 uses approximately 25 structural merge regions.\n"
                "This slider is only active for SDXL."
            )


# ---------------------------------
def export_profile(save_dir, profile_name, description, text_alpha, vae_alpha, model_type, weights):

    data = {
        "model_type": model_type,
        "description": description,
        "text_alpha": text_alpha,
        "vae_alpha": vae_alpha,
        "weights": list(weights)
    }

    path = os.path.join(save_dir, f"{profile_name}.json")
        
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        return f"<h3>Error Saving Profile Name - {profile_name}</h3>"

    return f"<h3>Saved profile: {profile_name}</h3>"

# ---------------------------------
def save_profile(profile_name, description, text_alpha, vae_alpha, model_type, *weights):

    data = {
        "model_type": model_type,
        "description": description,
        "text_alpha": text_alpha,
        "vae_alpha": vae_alpha,
        "weights": list(weights)
    }

    path = os.path.join(LLSTUDIO["profiles_dir"], f"{profile_name}.json")
        
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        return f"<h3>Error Saving Profile Name - {profile_name}</h3>"

    return f"<h3>Saved profile: {profile_name}</h3>"


# -----------------------------------------------------
def load_profile(profile_name):

    if not profile_name or profile_name == None:
        weights = []
        while len(weights) < 40:
            weights.append(0.5)
        return ["SDXL", f"<h3>Error: Bad Profile Name: {profile_name}</h3>", "", "", 0.5, 0.5] + weights

    path = os.path.join(LLSTUDIO["profiles_dir"], f"{profile_name}.json")
   
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        weights = []
        while len(weights) < 40:
            weights.append(0.5)
        return ["SDXL", f"<h3>Error: Loading Profile: {profile_name}</h3>", "", "", 0.5, 0.5] + weights

    weights = data["weights"]

    while len(weights) < 40:
        weights.append(0.5)

    return [data["model_type"], f"<h3>Loaded Profile: {profile_name}<br>Description: {data['description']}</h3>", profile_name, data['description'], data["text_alpha"], data["vae_alpha"]] + weights


# ---------------------------------
# load merge model preset
def apply_preset(preset_name):
    values = PRESETS[preset_name]
    while len(values) < 40:
        values.append(0.5)
    return values


# ---------------------------------
# merge model - long function
def block_merge(model_a, model_b, fp16, out_fp16, out_safe, model_type, merged_model_name, text_alpha, vae_alpha, *block_weights):

    global is_sdxl_merge
    if model_type == "SDXL":
        is_sdxl_merge = True
    else:
        is_sdxl_merge = False
    
    weights = list(block_weights)
    global_text_alpha = float(text_alpha)
    global_vae_alpha = float(vae_alpha)

    tempout = ""

    if model_a == model_b:
        tempout = "<h3>Error - The same models for both Model A and Model B were selected. You can not merge the same model.</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    if not model_a:
        tempout = "<h3>Error - No Model Name for Model A. Try selecting another model.</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    if not model_b:
        tempout = "<h3>Error - No Model Name for Model B. Try selecting another model.</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    output_dir = os.path.join(LLSTUDIO["lcm_model_dir"], merged_model_name)
    if os.path.exists(output_dir):
        tempout = f"<h3>Error - Merge Model Name '{output_dir}' Already Exists.<br>Try selecting another name for your merged model.</h3>"
        yield gr.update(value=tempout)
        return tempout

    pipeline_cls = (
        StableDiffusionXLPipeline
        if is_sdxl_merge
        else StableDiffusionPipeline
    )

    dtype = torch.float16 if fp16 else torch.float32

    # Init a dict for arguments
    pipeline_args = {}
    if fp16:
        pipeline_args["variant"] = "fp16"
        pipeline_args["torch_dtype"] = dtype
    pipeline_args["safety_checker"] = None
    pipeline_args["requires_safety_checker"] = False
    pipeline_args["feature_extractor"] = None
    pipeline_args["local_files_only"] = True
    pipeline_args["low_cpu_mem_usage"] = True

    tempout = "<h3>Loading Model A...</h3>"
    yield gr.update(value=tempout)

    try:
        pipe_a = pipeline_cls.from_pretrained(get_lcm_model_path_file(model_a), **pipeline_args)
    except Exception as e:
        tempout = f"<h3>Error: Loading Model A<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    tempout = "<h3>Loading Model B...</h3>"
    yield gr.update(value=tempout)

    try:
        pipe_b = pipeline_cls.from_pretrained(get_lcm_model_path_file(model_b), **pipeline_args)
    except Exception as e:
        tempout = f"<h3>Error: Loading Model B<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    tempout = "<h3>Merging U-Net...</h3>"
    yield gr.update(value=tempout)

    try:
        state_dict_b = pipe_b.unet.state_dict()
        total_layers = len(list(pipe_a.unet.named_parameters()))
        for i, (key, param_a) in enumerate(pipe_a.unet.named_parameters()):
            if key in state_dict_b:
                alpha = get_block_weight(key, weights, is_sdxl_merge)
                param_a.data.copy_((1.0 - alpha) * param_a.data + alpha * state_dict_b[key].data)
                if i % 100 == 0:
                    tempout = f"<h3>Processed [{i}/{total_layers}] U-Net layers</h3>"
                    yield gr.update(value=tempout)
    except Exception as e:
        tempout = f"<h3>Error: Merging U-Net<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    del state_dict_b
    pipe_b.unet = None
    gc.collect()
    rkmalloc_trim()
    
    tempout = "<h3>Merging Text Encoders...</h3>"
    yield gr.update(value=tempout)

    try:
        encoders = ["text_encoder"]
        if is_sdxl_merge:
            encoders.append("text_encoder_2")
        for encoder_name in encoders:
            tempout = f"<h3>Merging - {encoder_name}...</h3>"
            yield gr.update(value=tempout)
            if hasattr(pipe_a, encoder_name):
                mod_a = getattr(pipe_a, encoder_name)
                mod_b = getattr(pipe_b, encoder_name)
                if mod_a is None or mod_b is None:
                    continue
                state_dict_b = mod_b.state_dict()
                for key, param_a in mod_a.named_parameters():
                    if key in state_dict_b:
                        param_a.data.copy_((1.0 - global_text_alpha) * param_a.data + global_text_alpha * state_dict_b[key].data)

                del state_dict_b
                setattr(pipe_b, encoder_name, None)
                gc.collect()
                rkmalloc_trim()
    except Exception as e:
        tempout = f"<h3>Error: Merging Text Encoders<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    tempout = "<h3>Merging VAE...</h3>"
    yield gr.update(value=tempout)

    try:
        state_dict_b = pipe_b.vae.state_dict()
        for key, param_a in pipe_a.vae.named_parameters():
            if key in state_dict_b:
                param_a.data.copy_((1.0 - global_vae_alpha) * param_a.data + global_vae_alpha * state_dict_b[key].data)
        del state_dict_b
        gc.collect()
        rkmalloc_trim()
    except Exception as e:
        tempout = f"<h3>Error: Merging VAE<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    del pipe_b
    pipe_b = None
    gc.collect()
    rkmalloc_trim()

    if out_fp16:
        fp16_tempout = "fp16"
    else:
        fp16_tempout = "fp32"
    
    tempout = "<h3>Converting Merged Model to: " + fp16_tempout + "...</h3>"
    yield gr.update(value=tempout)
    
    try:
        if out_fp16:
            pipe_a = pipe_a.to(dtype=torch.float16)
        else:
            pipe_a = pipe_a.to(dtype=torch.float32)
    except Exception as e:
        tempout = f"<h3>Error: Converting Merged Model to: {fp16_tempout}.<br>{e}</h3>"
        yield gr.update(value=tempout)
        return tempout

    # Init a dict with the common arguments  **save_pipeline_args
    save_pipeline_args = { }
    
    if out_fp16:
        save_pipeline_args["variant"] = "fp16"
        
    if out_safe:
        save_pipeline_args["safe_serialization"] = True
    
    tempout = f"<h3>Saving Merged Model to: {merged_model_name}...</h3>"
    yield gr.update(value=tempout)

    try:
        pipe_a.save_pretrained(f"{output_dir}", **save_pipeline_args)
    except Exception as e:
        del pipe_a
        pipe_a = None
        gc.collect()
        rkmalloc_trim()
        tempout = f"<h3>Error: Saving Merged Model to: {merged_model_name} <br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    del pipe_a
    pipe_a = None
    gc.collect()
    rkmalloc_trim()

    # create a model card (*.md) for this 'merged' model 
    # and put it in the image gallery for this specific LCM-LoRA Model
    in_fp16 = ("fp16" if fp16 else "fp32" )
    model_image_path_file = get_lcm_model_image_path_file(merged_model_name)
    if not os.path.exists(model_image_path_file):
        os.makedirs(model_image_path_file)
    file1 = open(os.path.join(model_image_path_file, merged_model_name) + ".md", 'w')
    content = "## Merged Model: " + merged_model_name + "    \n\n"
    content = content + "## Original Model A: " + model_a + "    \n"
    content = content + "## Original Model B: " + model_b + "    \n\n"
    content = content + "### Models Type: " + model_type + "    \n"
    content = content + "### Models Precision: " + in_fp16 + "    \n\n"
    content = content + "### -----------------------------------    \n\n"
    content = content + "### Merged Model Precision: " + fp16_tempout + "    \n\n"
    content = content + "### Merged Model Text Encoder Weights: " + str(text_alpha) + "    \n"
    content = content + "### Merged Model VAE Weights: " + str(vae_alpha) + "    \n"
    content = content + f"### Merged Model Profile Name: '{merged_model_name}.json'    \n\n"
    content = content + f"*Merged using {LLSTUDIO['app_title']} - {LLSTUDIO['app_version']}*    \n\n\n"
    file1.write(content)
    file1.close()    

    # export/saves a 'merged_model_name.json' - merge profile
    # in the model card directory to recreate with same weights later
    export_profile(model_image_path_file, merged_model_name, content, text_alpha, vae_alpha, model_type, weights)


    tempout = f"<h3>Merge completed successfully!<br>Saved Merged Model to: {merged_model_name}<br>Saved Model Card to: '{merged_model_name}.md'<br>Saved Merge Profile to: '{merged_model_name}.json'</h3>"
    yield gr.update(value=tempout)
    return f"<h3>Merge completed successfully!<br>Saved Merged Model to: {merged_model_name}<br>Saved Model Card to: '{merged_model_name}.md'<br>Saved Merge Profile to: '{merged_model_name}.json'</h3>"


# ------------------------------------------------------------------
# init ui sliders for block weights - runs once
def build_block_slider_ui():
    global is_sdxl_merge

    sliders = []
    for i in range(0, 40, 4):
        with gr.Row(equal_height=True):
            for j in range(4):
                idx = i + j
                if idx >= 40:
                    continue
                if idx >= 25:
                    visible = False
                else:
                    visible = True
                with gr.Column(scale=1, min_width=100):
                    slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.01,
                        label=(
                            SDXL_BLOCK_LABELS[idx]
                            if idx < len(SDXL_BLOCK_LABELS)
                            else f"Block {idx}"
                        ),
                        info=get_block_description(idx, is_sdxl_merge),
                        visible=visible
                    )
                sliders.append(slider)

    return sliders


# ------------------------------------------------------------------
def update_slider_visibility(model_type):
    global is_sdxl_merge

    if model_type == "SDXL":
        is_sdxl_merge = True
    else:
        is_sdxl_merge = False

    updates = []

    for idx in range(40):
        if is_sdxl_merge:
            updates.append(
                gr.update(
                    visible=True,
                    info=get_block_description(idx, is_sdxl_merge)
                )
            )
        else:
            visible = idx <= 24
            updates.append(
                gr.update(
                    visible=visible,
                    info=get_block_description(idx, is_sdxl_merge)
                )
            )

    return updates
    
 

# ==========================================================================================


# # ==============================================================
# # START Image Processing Functions
# # ==============================================================

# -------------------------
# transformers depthmap
# -------------------------

def do_depth_map(img):
    if img is None:
        return None

    # convert to PIL, that what transformers wants to see
    depth_image = numpy_to_pil(img)

    # load depth_estimator, we use the transformers.pipeline()
    # to not get it confused with the global 'pipeline' for our main model pipeline
    depth_estimator = transformers.pipeline('depth-estimation')
 
    # run depth_estimator
    depth_output = depth_estimator(depth_image)['depth']

    del depth_estimator
    depth_estimator = None
    del depth_image
    depth_image = None
    gc.collect()
    
    # return the image as it is.
    return depth_output



# ====================================================================
# START - NUMPY ONLY IMAGE PROCESSING UTILS
# ====================================================================

# -------------------------
# Utility: 2D Convolution
# -------------------------
def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)

    return result


# -------------------------
# Edge Detection Functions
# -------------------------
def sobel_edge_detection(image):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    Ix = convolve2d(image, Gx)
    Iy = convolve2d(image, Gy)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def prewitt_edge_detection(image):
    Gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]], dtype=np.float32)

    Ix = convolve2d(image, Gx)
    Iy = convolve2d(image, Gy)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def laplacian_edge_detection(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    result = convolve2d(image, kernel)
    result = np.abs(result)
    return np.clip(result, 0, 255).astype(np.uint8)


def roberts_cross_edge_detection(image):
    Gx = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)

    Gy = np.array([[0, 1],
                   [-1, 0]], dtype=np.float32)

    Ix = convolve2d(image, Gx)
    Iy = convolve2d(image, Gy)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def canny_edge_detection(image, low_threshold=50, high_threshold=100):
    #Gaussian blur
    gaussian_kernel = (1/273) * np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])
    blurred = convolve2d(image, gaussian_kernel)

    #Sobel gradient
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    Ix = convolve2d(blurred, Gx)
    Iy = convolve2d(blurred, Gy)

    magnitude = np.hypot(Ix, Iy)
    angle = np.arctan2(Iy, Ix)
    angle = np.degrees(angle) % 180

    #Non-maximum suppression
    nms = np.zeros_like(magnitude)
    h, w = image.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            a = angle[i, j]
            mag = magnitude[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                before = magnitude[i, j-1]
                after = magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                before = magnitude[i-1, j+1]
                after = magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                before = magnitude[i-1, j]
                after = magnitude[i+1, j]
            else:
                before = magnitude[i-1, j-1]
                after = magnitude[i+1, j+1]

            if mag >= before and mag >= after:
                nms[i, j] = mag

    #Double threshold
    strong, weak = 255, 75
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms >= low_threshold) & (nms < high_threshold))
    result = np.zeros_like(nms, dtype=np.uint8)
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    #Hysteresis
    for i in range(1, h-1):
        for j in range(1, w-1):
            if result[i, j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result


# -------------------------------------------------------------------------------------------------
def simple_gradient_detection(image: np.ndarray) -> np.ndarray:
    # Ensure the input image is of float type for gradient calculations
    image_float = image.astype(float)

    # Compute the gradient using np.gradient
    # This returns two arrays: grad_y (for rows) and grad_x (for columns)
    grad_y, grad_x = np.gradient(image_float)

    # Calculate the gradient magnitude (edge strength)
    # The magnitude is the hypotenuse of the horizontal and vertical gradients
    edge_magnitude = np.hypot(grad_x, grad_y)

    # Normalize the output to be in the valid image intensity range (0-255)
    # This ensures the full dynamic range is used for better visualization
    if np.max(edge_magnitude) > 0:
        edge_magnitude *= 255.0 / np.max(edge_magnitude)

    # Convert the array back to the unsigned 8-bit integer type
    return edge_magnitude.astype(np.uint8)


# -------------------------------------------------------------------------------------------------
# Post-processing filters (numpy)
def apply_post_filters_numpy(img, apply_sharpen, apply_edges, filter_name, canny_low_threshold, canny_high_threshold):
    
    # Check for input image
    if img is None:
        return None

    output = img.copy()
        
    if (apply_sharpen and OPENCV_AVAILABLE):
        # Sharpen kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        output = cv2.filter2D(output, -1, kernel)


    if apply_edges:
        if filter_name == "Laplacian (Numpy)":
            output = laplacian_edge_detection(output)
        elif filter_name == "Roberts Cross (Numpy)":
            output = roberts_cross_edge_detection(output)
        elif filter_name == "Simple Gradient":
            output = simple_gradient_detection(output)
        elif filter_name == "Prewitt (Numpy)":
            output = prewitt_edge_detection(output)
        elif filter_name == "Canny (Numpy)":
            output = canny_edge_detection(output, canny_low_threshold, canny_high_threshold)
        elif filter_name == "Sobel (Numpy)":
            output = sobel_edge_detection(output)
    
    return output

# -------------------------------------------------------------------------------------------------

# ====================================================================
# END - NUMPY ONLY IMAGE PROCESSING UTILS
# ====================================================================

# -------------------------------------------------------------------------------------------------
# Converts a PIL Type Image to a Numpy Array Type Image
def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    if not isinstance(pil_img, Image.Image):
        raise TypeError("Input must be a PIL.Image.Image object.")

    mode = pil_img.mode

    if mode == "1":
        # Convert 1-bit pixels to 0 and 255 in uint8
        return np.array(pil_img.convert("L")) > 127  # Binary mask as bool
    elif mode in ("L", "RGB", "RGBA"):
        return np.array(pil_img)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")



# -------------------------------------------------------------------------------------------------
# Converts a Numpy Array Type Image to a PIL Type Image
def numpy_to_pil(np_image: np.ndarray) -> Image.Image:
    if not isinstance(np_image, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")

    # Handle monochrome: bool or uint8 with only 0 and 255
    if np_image.dtype == bool:
        return Image.fromarray(np_image.astype("uint8") * 255).convert("1")
    
    if np_image.dtype == np.uint8 and np_image.ndim == 2:
        unique_vals = np.unique(np_image)
        if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
            return Image.fromarray(np_image).convert("1")

    # Normalize if not uint8
    if np_image.dtype != np.uint8:
        np_image = (255 * (np_image / np_image.max())).astype(np.uint8)

    # Try and determine mode from shape
    if np_image.ndim == 2:
        return Image.fromarray(np_image, mode="L")  # Grayscale
    elif np_image.ndim == 3:
        if np_image.shape[2] == 3:
            return Image.fromarray(np_image, mode="RGB")
        elif np_image.shape[2] == 4:
            return Image.fromarray(np_image, mode="RGBA")
        else:
            raise ValueError("Unsupported channel number: expected 3 (RGB) or 4 (RGBA).")
    else:
        raise ValueError("Unsupported array shape for image conversion.")


# --------------------------------------------------------

# ==================================================
# NO NUMPY below here....
# OPENCV is below here...
# ==================================================

# --------------------------------------------------------
# edge detecion and sharpening.
def process_image(image: np.ndarray, 
                  method: str = 'canny', 
                  sharpen: bool = False,
                  **kwargs) -> np.ndarray:

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy ndarray.")
    if image.ndim != 2:
        raise ValueError("Image must be a 2D monochrome (grayscale) array.")

    # Clone the image to work on
    result = image.copy()

    # Apply edge detection
    if method == 'canny':
        low = kwargs.get('low_threshold', 100)
        high = kwargs.get('high_threshold', 200)
        result = cv2.Canny(result, low, high)

    elif method == 'sobel':
        ksize = kwargs.get('ksize', 3)
        sobelx = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=ksize)
        result = cv2.magnitude(sobelx, sobely)
        result = np.uint8(np.clip(result, 0, 255))

    elif method == 'laplacian':
        lap = cv2.Laplacian(result, cv2.CV_64F)
        result = np.uint8(np.clip(np.absolute(lap), 0, 255))

    elif method == 'scharr':
        scharrx = cv2.Scharr(result, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(result, cv2.CV_64F, 0, 1)
        result = cv2.magnitude(scharrx, scharry)
        result = np.uint8(np.clip(result, 0, 255))

    elif method == 'prewitt':
        # Define Prewitt kernels
        kernel_prewitt_x = np.array([[-1, 0, 1],
                                     [-1, 0, 1],
                                     [-1, 0, 1]])
        kernel_prewitt_y = np.array([[-1, -1, -1],
                                     [ 0,  0,  0],
                                     [ 1,  1,  1]])
        # Apply the Prewitt kernels using cv2.filter2D
        # The ddepth=-1 means the output image will have the same depth as the source.
        gradient_x = cv2.filter2D(result, -1, kernel_prewitt_x)
        gradient_y = cv2.filter2D(result, -1, kernel_prewitt_y)
        # Combine the gradients to get the final edge magnitude
        # take the absolute values and then convert to an 8-bit unsigned integer
        prewitt_edges = np.sqrt(gradient_x**2 + gradient_y**2)
        result = cv2.normalize(prewitt_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    elif method == 'none':
        pass  # No edge detection

    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'canny', 'sobel', 'laplacian', 'scharr', 'prewitt' or 'none'.")

    # Apply sharpening
    if sharpen:
        # Sharpening kernel
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
        result = cv2.filter2D(result, -1, kernel)

    return result


# --------------------------------------------------------
# conversion from color to adjusted color image (pre-process)
def adjust_image(img, brightness=1.0, contrast=1.0, color=1.0, r_weight=1.0, g_weight=1.0, b_weight=1.0):
    # Check for input image
    if img is None:
        return None
    
    # do 3 color RGB adjustment, then pass on to rest of function as numpy array    
    # Ensure image is in float for calculations
    image_float = img.astype(np.float32) / 255.0

    # Apply color adjustments
    image_float[:, :, 0] *= r_weight  # Red channel
    image_float[:, :, 1] *= g_weight # Green channel
    image_float[:, :, 2] *= b_weight # Blue channel

    # Clip values to valid range [0, 1] and convert back to uint8
    adjusted_image_np = (np.clip(image_float, 0, 1) * 255).astype(np.uint8)

    adjusted_image_np = Image.fromarray(adjusted_image_np)
    adjusted_image_np = ImageEnhance.Brightness(adjusted_image_np).enhance(brightness)
    adjusted_image_np = ImageEnhance.Contrast(adjusted_image_np).enhance(contrast)
    adjusted_image_np = ImageEnhance.Color(adjusted_image_np).enhance(color)
    return np.array(adjusted_image_np)


# -------------------------------------------------------------------------------------------------
# Convert to grayscale
def convert_to_grayscale(img, r_weight=0.2989, g_weight=0.5870, b_weight=0.1140):
    if img is None:
        return None
    img = np.array(img).astype(np.float32)
    grayscale = (img[:, :, 0] * r_weight + img[:, :, 1] * g_weight + img[:, :, 2] * b_weight).astype(np.uint8)
    return grayscale


# -------------------------------------------------------------------------------------------------
# Convert to monochrome
def convert_to_monochrome(img, lower_thresh=100, upper_thresh=200, invert=False):
    if img is None:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    mono = np.where((gray >= lower_thresh) & (gray <= upper_thresh), 255, 0).astype(np.uint8)
    if invert:
        mono = 255 - mono
    return mono


# -------------------------------------------------------------------------------------------------
# Apply multiple blur types in sequence
def apply_blurs(img, apply_gaussian, gaussian_amount,
                      apply_motion_h, motion_h_amount,
                      apply_motion_v, motion_v_amount):
   
    # Check for input image
    if img is None:
        return None

    output = img.copy()

    def apply_motion_blur(image, amount, vertical=False):
        k = max(1, int(amount))
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k))
        if vertical:
            kernel[:, k // 2] = np.ones(k)
        else:
            kernel[k // 2, :] = np.ones(k)
        kernel /= k
        return cv2.filter2D(image, -1, kernel)

    if apply_gaussian and gaussian_amount > 0:
        k = max(1, int(gaussian_amount))
        if k % 2 == 0:
            k += 1
        output = cv2.GaussianBlur(output, (k, k), 0)

    if apply_motion_h and motion_h_amount > 0:
        output = apply_motion_blur(output, motion_h_amount, vertical=False)

    if apply_motion_v and motion_v_amount > 0:
        output = apply_motion_blur(output, motion_v_amount, vertical=True)

    return output


# -------------------------------------------------------------------------------------------------
# Post-processing filters
def apply_post_filters(img, apply_sharpen, apply_edges, edge_type, low_threshold, high_threshold):

    # Check for input image
    if img is None:
        return None

    output = img.copy()

    edge_kwargs = { }

    if apply_sharpen:
        edge_kwargs["sharpen"] = True

    if apply_edges:
        if edge_type == "Sobel":
            # Sobel edge detection
            edge_kwargs["method"] = "sobel"
            edge_kwargs["ksize"] = 3
            output = process_image(output, **edge_kwargs)

        if edge_type == "Canny":
            # Perform Canny edge detection
            edge_kwargs["method"] = "canny"
            edge_kwargs["low_threshold"] = low_threshold
            edge_kwargs["high_threshold"] = high_threshold
            edge_kwargs["ksize"] = 3
            output = process_image(output, **edge_kwargs)
            
        if edge_type == "Laplacian":
            # Perform Laplacian edge detection
            edge_kwargs["method"] = "laplacian"
            output = process_image(output, **edge_kwargs)
        

        if edge_type == "Scharr":
            # Perform Scharr edge detection
            edge_kwargs["method"] = "scharr"
            output = process_image(output, **edge_kwargs)


        if edge_type == "Prewitt":
            # Perform Prewitt edge detection
            edge_kwargs["method"] = "prewitt"
            output = process_image(output, **edge_kwargs)
            

        
    return output


# -------------------------------------------------------------------------------------------------
# invert final output image
def invert_colors_numpy_io(image_array):
    if image_array is None:
        return None
    
    # Convert the input NumPy array to a Pillow Image object
    # The image is already a NumPy array because we didn't specify `type="pil"`
    pil_image = Image.fromarray(image_array.astype('uint8'))
    
    # Invert the colors using Pillow's ImageOps.invert()
    inverted_pil_image = ImageOps.invert(pil_image)
    
    # Convert the inverted Pillow image back to a NumPy array
    inverted_image_array = np.array(inverted_pil_image)
    
    return inverted_image_array
    
    
# -------------------------------------------------------------------------------------------------
# open an image an return to rest of app as a numpy array
def load_image_as_numpy(image_path):
    try:
        # Open the image file using Pillow
        pil_image = Image.open(image_path)
        # Convert the Pillow image to a NumPy array
        numpy_array = np.array(pil_image)
        return numpy_array

    except FileNotFoundError:
        gr.Warning(f"The file '{image_path}' was not found.")
        return None

    except Exception as e:
        gr.Error(f"An error occurred: {e}")
        return None



# -------------------------------------------------------------------------------------------------
# Post-Process -  Stage 3 only - image processing pipeline
def post_process_pipeline(img,
                  lower_thresh, upper_thresh, invert_grayscale, invert_final,
                  lower_canny_thresh, upper_canny_thresh,
                  # Blur settings stage 3
                  s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt,

                  # Post-processing
                  sharpen, apply_edges, edge_filters):

    # Check for input image
    if img is None:
        gr.Info("No Valid Input Image !!<br>Please Load an Input Image.", duration=5.0, title="Input Image")
        no_image = load_image_as_numpy(LLSTUDIO["no_image"])
        return (no_image)

    # Monochrome
    monochrome = convert_to_monochrome(img, lower_thresh, upper_thresh, invert_grayscale)
    monochrome = apply_blurs(monochrome, s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt)

    # Post-filters
    # short OpenCV edge detection list
    cvedges = ["Canny", "Sobel", "Laplacian", "Prewitt", "Scharr"]
    # check if edge detection filter is in the OpenCV list
    # if not must be a numpy only edge detection algo
    if edge_filters in cvedges:
        # OpenCV edge detectors
        final_output = apply_post_filters(monochrome, sharpen, apply_edges, edge_filters, lower_canny_thresh, upper_canny_thresh)
    else:
        # Numpy ONLY edge detectors 
        # (although OpenCV is used for 'sharpen') 
        # so not pure numpy, but numpy only edges
        final_output = apply_post_filters_numpy(monochrome, sharpen, apply_edges, edge_filters, lower_canny_thresh, upper_canny_thresh)


    # invert final b/w monochrome image
    if invert_final:
        final_output = invert_colors_numpy_io(final_output)

    return final_output


# -------------------------------------------------------------------------------------------------
# main image processing 'pipeline'
def image_pipeline(img,
                  brightness, contrast, color,
                  r_weight, g_weight, b_weight,
                  r_gray_weight, g_gray_weight, b_gray_weight,
                  lower_thresh, upper_thresh, invert_grayscale, invert_final,
                  lower_canny_thresh, upper_canny_thresh,
                  # Blur settings stage 1
                  s1_g, s1_g_amt, s1_h, s1_h_amt, s1_v, s1_v_amt,
                  # Blur settings stage 2
                  s2_g, s2_g_amt, s2_h, s2_h_amt, s2_v, s2_v_amt,
                  # Blur settings stage 3
                  s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt,
                  # Post-processing
                  sharpen, apply_edges, edge_filters):

    # Check for input image
    if img is None:
        gr.Info("No Valid Input Image !!<br>Please Load an Input Image.", duration=5.0, title="Input Image")
        no_image = load_image_as_numpy(LLSTUDIO["no_image"])
        return (no_image, no_image, no_image)

    # Adjust
    adjusted = adjust_image(img, brightness, contrast, color, r_weight, g_weight, b_weight)
    adjusted = apply_blurs(adjusted, s1_g, s1_g_amt, s1_h, s1_h_amt, s1_v, s1_v_amt)

    # Grayscale
    grayscale = convert_to_grayscale(adjusted, r_gray_weight, g_gray_weight, b_gray_weight)
    grayscale = apply_blurs(grayscale, s2_g, s2_g_amt, s2_h, s2_h_amt, s2_v, s2_v_amt)

    # Monochrome
    monochrome = convert_to_monochrome(grayscale, lower_thresh, upper_thresh, invert_grayscale)
    monochrome = apply_blurs(monochrome, s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt)

    # Post-filters
    # short OpenCV edge detection list
    cvedges = cvedges = ["Canny", "Sobel", "Laplacian", "Prewitt", "Scharr"]
    # check if edge detection filter is in the OpenCV list
    # if not must be a numpy only edge detection algo
    if edge_filters in cvedges:
        # OpenCV edge detectors
        final_output = apply_post_filters(monochrome, sharpen, apply_edges, edge_filters, lower_canny_thresh, upper_canny_thresh)
    else:
        # Numpy ONLY edge detectors 
        # (although OpenCV is used for 'sharpen') 
        # so not pure numpy, but numpy only edges
        final_output = apply_post_filters_numpy(monochrome, sharpen, apply_edges, edge_filters, lower_canny_thresh, upper_canny_thresh)


    # invert final b/w monochrome image
    if invert_final:
        final_output = invert_colors_numpy_io(final_output)

    return (
        adjusted, grayscale, final_output
    )


# -------------------------------------------------------------------------------------------------
# resets the gradio ui for ALL settings, and closes the gr.Accordions too.
def reset_config():

    # go get the no image, IMAGE !
    no_image = load_image_as_numpy(LLSTUDIO["no_image"])
    
    # return all the default values for the ui controls
    # and close the gr.Accordions too.
    return (1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            0.2989, 0.5870, 0.1140,
            100, 200, False, False,
            100, 200,
            False, 0.0, False, 0.0, False, 0.0,
            False, 0.0, False, 0.0, False, 0.0,
            False, 0.0, False, 0.0, False, 0.0,
            False, False, (gr.Dropdown(choices=EDGEFILTERS, interactive=True)),
            no_image, no_image, no_image,
            (gr.Accordion(open=False)),(gr.Accordion(open=False)),(gr.Accordion(open=False)),(gr.Accordion(open=False))
    )

    
# -------------------------------------------------------------------------------------------------

# =======================================================================
# Functions called by the gradio ui in Image Processing
# =======================================================================

# -------------------------------------------------------
def send_to_controlnet(img):
    if img is None:
        gr.Info("No Valid Image to Send to ControlNet!!<br>Please Process an Image.", duration=5.0, title="Send to ControlNet")
        no_image = load_image_as_numpy(LLSTUDIO["no_image"])
        return no_image, "No Valid Image to Send to ControlNet!! Please Process an Image."

    return numpy_to_pil(img), "Sucessfully Sent Image to ControlNet."


# -------------------------------------------------------------------------------------------------
# save the numpy type image to a png file
def imgp_load_file(image_path, image_input_loc, input_img, adjusted_img, grayscale_img, output_img):

    filepathname = os.path.join(LLSTUDIO["imgp_file_dir"],image_path)
   
    try:
        # Open the image file using Pillow
        pil_image = Image.open(filepathname)
        # Convert the Pillow image to a NumPy array
        numpy_array = np.array(pil_image)
    except Exception as e:
        return f"Error loading Image '{image_path}': {e}", input_img, adjusted_img, grayscale_img, output_img

    if image_input_loc == "Input":
        return "", numpy_array, adjusted_img, grayscale_img, output_img
    if image_input_loc == "Adjusted":
        return "", input_img, numpy_array, grayscale_img, output_img
    if image_input_loc == "Grayscale":
        return "", input_img, adjusted_img, numpy_array, output_img
    if image_input_loc == "Output":
        return "", input_img, adjusted_img, grayscale_img, numpy_array
    

# -------------------------------------------------------------------------------------------------
# save the numpy type image to a png file
def imgp_save_file(image_Input: np.ndarray, image_Adjusted: np.ndarray, image_Grayscale: np.ndarray, image_Output: np.ndarray, image_input_loc: str, filename: str):

    filepathname = os.path.join(LLSTUDIO["imgp_file_dir"],filename)
    
    if image_input_loc == "Input":
        image = image_Input
    if image_input_loc == "Adjusted":
        image = image_Adjusted
    if image_input_loc == "Grayscale":
        image = image_Grayscale
    if image_input_loc == "Output":
        image = image_Output

    if image is None:
        return f"Error no valid {image_input_loc} to Send to ControlNet!!"

    try:
        pil_image = Image.fromarray(image.astype(np.uint8))
        if not filepathname.lower().endswith(".png"):
            filepathname += ".png"
        pil_image.save(filepathname)
        return f"Saved {image_input_loc} Image: {filename}"
    except Exception as e:
        return f"Error saving {image_input_loc} Image '{filename}': {e}"
    


# -----------------------------------------------------------------------------
# just reloads imgp_file_list[] - called to refresh imgp_file_list[] items
def imgp_get_file_list():
    LLSTUDIO["imgp_file_list"] = []
    entries = sorted([f for f in os.listdir(LLSTUDIO["imgp_file_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["imgp_file_dir"], f))])
    for i in range(len(entries)):
        tmp_text = entries[i]
        if (tmp_text.lower().endswith('.png') or tmp_text.lower().endswith('.jpg') or tmp_text.lower().endswith('.jpeg')):
            LLSTUDIO["imgp_file_list"].append(tmp_text)

    return LLSTUDIO["imgp_file_list"]


# ------------------------------------------------------
def imgp_refresh_file_list_dropdown():
    imgp_get_file_list()
    return gr.Dropdown(choices=LLSTUDIO["imgp_file_list"], interactive=True)
    

# ------------------------------------------------------


# # ==============================================================
# # END Image Processing Functions
# # ==============================================================


# ------------------------------------------------------  
def get_system_stats(mem_opt=0):
    if mem_opt:
        rkmalloc_trim()
    sysstats_output = ""
    mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()
    process = psutil.Process(os.getpid())
    font_color_mem = "#00FF00"
    font_color_swap = "#00FF00"
    if mem.percent > 80.0:
        font_color_mem = "#FF0000"
    if swap_mem.percent > 35.0:
        font_color_swap = "#FF0000"

    sysstats_output = sysstats_output + f"<div style='color: {font_color_mem};'>Memory: {mem.total // (1024**2):,}M total, {mem.used // (1024**2):,}M used, {mem.available // (1024**2):,}M available - Usage: [{mem.percent:.1f}%]</div>"
    
    if swap_mem.total > 0:
        sysstats_output = sysstats_output + f"<div style='color: {font_color_swap};'>Swap: {swap_mem.total // (1024**2):,}M total, {swap_mem.used // (1024**2):,}M used, {swap_mem.free // (1024**2):,}M free - Usage: [{swap_mem.percent:.1f}%]</div>"
    else:
        sysstats_output = sysstats_output + "Swap: Not used"
   
    return f"<p>{sysstats_output}</p>"



# =======================================================================  
def set_freeu_values(ins1, ins2, inb1, inb2):
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return ins1, ins2, inb1, inb2
 
    if (SDPIPELINE['pipeline_model_type']=="SDXL"):
        return LLSTUDIO["freeu_sdxl_s1"], LLSTUDIO["freeu_sdxl_s2"], LLSTUDIO["freeu_sdxl_b1"], LLSTUDIO["freeu_sdxl_b2"]
    else:
        return LLSTUDIO["freeu_sd_s1"], LLSTUDIO["freeu_sd_s2"], LLSTUDIO["freeu_sd_b1"], LLSTUDIO["freeu_sd_b2"]


# ------------------------------------------------------------
# refixed5 func diff vers from RKv0.5
def do_prompt_embeds(device, pipeline, prompt, negative_prompt):

    # 'feel' it needs modification to handle no negative prompt, Hmmmm
    # althougth seems to work ok so far...
    max_length = pipeline.tokenizer.model_max_length

    # determine length of tokens
    input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
    negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)

    # create the tensor based on which prompt is longer
    # prompt is equal or longer than negative prompt.
    if input_ids.shape[-1] >= negative_ids.shape[-1]:
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    # negative prompt is longer than prompt.
    else:
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, truncation=False, padding="max_length", max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    max_length = None
    input_ids = None
    negative_ids = None
    shape_max_length = None
    del max_length
    del input_ids
    del negative_ids
    del shape_max_length
    gc.collect()

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
	

# ------------------------------------------------------------
# SDXL tokenize and encode prompt
# modification for diffusers SDXL with padding + pooled + embeds
def get_prompt_and_pooled_embeddings(device, pipeline, text):

    # Tokenizer and encoder used by the pipeline
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    # Tokenize with correct padding and max_length
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)

    # Encode text and extract prompt_embeds and pooled_prompt_embeds
    # faster this way...
    with torch.no_grad():
        encoder_output = text_encoder(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # SDXL uses second-to-last hidden state for token embeddings
        prompt_embeds = encoder_output.hidden_states[-2]
        pooled_prompt_embeds = encoder_output.pooler_output
    
    tokenizer = None
    text_encoder = None
    text_inputs = None
    input_ids = None
    encoder_output = None
    del tokenizer
    del text_encoder
    del text_inputs
    del input_ids
    del encoder_output
    gc.collect()


    return prompt_embeds, pooled_prompt_embeds


# ------------------------------------------------------
def halt_generation():
    LLSTUDIO["halt_gen"] = 1
    gr.Info("Generation Halted</br>Please wait for current inference to complete...", duration=5.0, title="Halt Generation")


# ------------------------------------------------------
def format_seconds_strftime(seconds):
    time_tuple = time.gmtime(seconds)
    formatted_time = time.strftime("%M minutes, %S seconds", time_tuple)
    return formatted_time



# ------------------------------------------------------
# read the model file information from the 'modelfilename.txt'
def preview_get_model_info_file(file):
    file = open(file, "r")
    content = file.read()
    file.close()
    return content


# ------------------------------------------------------------
# read TEXT generation paramaters from the 'image-filename.txt'
def preview_create_text_code(file):
    file = open(file, "r")
    content = ""
    idx=0
    while True:
        line = file.readline()
        if not line:
            break
        idx=idx+1
        if idx==1:
            content = content + "<code>" + line.strip() + "</code>"
        else:
            content = content + "<br>" + line.strip()
    file.close()
    text_code = f'{content}'
    return text_code


# ------------------------------------------------------------
# create HTML code to display the image 
# as a link to open image in new window
def preview_create_html_code(file):
    html_enc_file = file.replace(" ","%20")
    html_img_code = f"""<a href=#top>Go to TOP</a></br><img src="/gradio_api/file/{html_enc_file}" width="{STUDIO["img_view_img_width"]["value"]}%" height="auto" style="cursor:pointer" onclick="window.open(this.src)"></img>"""
    return html_img_code


# ---------------------------------------------------------------------------------------
# output viewer
def get_output_image_list():
    output_image_list = []
    entries = [f for f in os.listdir(LLSTUDIO["output_image_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["output_image_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get png files only
        if tmp_text.endswith('.png'):
            output_image_list.append(os.path.join(LLSTUDIO["output_image_dir"], tmp_text))

    # modified time 'm'
    # output_image_list.sort(key=os.path.getmtime, reverse=True)
    # created time 'c'
    output_image_list.sort(key=os.path.getctime, reverse=True)
    return output_image_list
 

# ---------------------------------------------------------------------------------------
# lcm model images viewer
def get_lcm_image_list(modelname):
    output_image_list = []
    image_dir = os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname)
    entries = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get png files only
        if (tmp_text.endswith(('.jpg', '.jpeg', '.png', '.webp'))):
            output_image_list.append(os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname, tmp_text))
    # modified time 'm'
    # output_image_list.sort(key=os.path.getmtime, reverse=True)
    # created time 'c'
    output_image_list.sort(key=os.path.getctime, reverse=True)
    return output_image_list
 


# ---------------------------------------------------------------------------------------


# safetensors model images viewer
def get_safe_image_list(modelname):
    output_image_list = []
    image_dir = os.path.join(LLSTUDIO["safe_model_image_dir"],modelname)
    entries = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get png files only
        if (tmp_text.endswith(('.jpg', '.jpeg', '.png', '.webp'))):
            output_image_list.append(os.path.join(LLSTUDIO["safe_model_image_dir"],modelname, tmp_text))
    # modified time 'm'
    # output_image_list.sort(key=os.path.getmtime, reverse=True)
    # created time 'c'
    output_image_list.sort(key=os.path.getctime, reverse=True)
    return output_image_list
 


# ---------------------------------------------------------------------------------------
# lora model images viewer
def get_lora_image_list(modelname):
    output_image_list = []
    image_dir = os.path.join(LLSTUDIO["lora_model_image_dir"],modelname)
    entries = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get png files only
        if (tmp_text.endswith(('.jpg', '.jpeg', '.png', '.webp'))):
            output_image_list.append(os.path.join(LLSTUDIO["lora_model_image_dir"],modelname, tmp_text))
    # modified time 'm'
    # output_image_list.sort(key=os.path.getmtime, reverse=True)
    # created time 'c'
    output_image_list.sort(key=os.path.getctime, reverse=True)
    return output_image_list
 
# ------------------------------------------------------------------------------------------------------------------
# Returns a subset of items for a specific page
def paginate_list(items, page_number, page_size):
    start_index = (page_number - 1) * page_size
    end_index = start_index + page_size
    return items[start_index:end_index]


# ---------------------------------------------------------------
# the actual viewer - page/tab - output
def show_output_preview(input_cmd):
    if input_cmd == 0:
        return "", ""
    html_code_list = ""
    html_header_list = ""
    output_png_list = []
    output_image_list = get_output_image_list()

    # --- Start Page logic ---
    # first
    if input_cmd == 2:
        LLSTUDIO["output_page_num"] = 1
    # previous
    if input_cmd == 3:  
        LLSTUDIO["output_page_num"] = min(max(LLSTUDIO["output_page_num"] - 1, 1), len(output_image_list))
    # next
    if input_cmd == 4:
        LLSTUDIO["output_page_num"] = min(max(LLSTUDIO["output_page_num"] + 1, 1), len(output_image_list))
    # last
    if input_cmd == 5:
        LLSTUDIO["output_page_num"] = int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)
            
    current_page = LLSTUDIO["output_page_num"]
    output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # make sure 'last' page is not empty, if it is, go back one page.
    if len(output_png_list) < 1:
        LLSTUDIO["output_page_num"] = LLSTUDIO["output_page_num"] - 1
        current_page = LLSTUDIO["output_page_num"]
        output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # --- End page logic ---

    # page header above images
    page_header_output = "<h3>Page " + str(LLSTUDIO["output_page_num"]) + " of " + str(int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)) + " - (" + str(len(output_image_list)) + " : Images Total)</h3>"

    for i in range(len(output_png_list)):
        png_filename_ext = output_png_list[i]
        png_filename = os.path.splitext(os.path.basename(png_filename_ext))[0]
        if os.path.isfile(png_filename_ext):
            html_code_list = html_code_list + '<table cellpadding=10 cellspacing=10 border=0>'
            html_code_list = html_code_list + '<tr>'
            html_code_list = html_code_list + '<td width=50%>'
            html_code_list = html_code_list + preview_create_html_code(png_filename_ext)
            html_code_list = html_code_list + '</td>'
            txtinfofile = os.path.join(LLSTUDIO["output_image_dir"], png_filename + ".txt")
            if os.path.isfile(txtinfofile):
                html_code_list = html_code_list + '<td width=45%>'
                html_code_list = html_code_list + preview_create_text_code(txtinfofile)
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
            else:
                html_code_list = html_code_list + '<td width=50%>'
                html_code_list = html_code_list + 'No Image information found.'
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
        
    if len(output_image_list) < 1:
        html_header_list = "</br><font size=+1>No Images found.</font></br>"
    else:
        html_header_list = "</br><font size=+2>IMAGE GALLERY</font></br>"
    
    html_code_list = html_header_list + html_code_list
    
    return page_header_output, html_code_list


# --------------------------------------------------------------------
# read/write/view modelcard functions
# --------------------------------------------------------------------

# -----------------------------------------------------------
# swaps which is visible, markdown or code, in that order, for the modelcard
# ie... yield markdown, code
def set_modelcard_editmode(view_content, edit_content):
    yield gr.update(visible=False), gr.update(value=view_content, visible=True)

# -----------------------------------------------------------
# swaps which is visible, markdown or code, in that order, for the modelcard
# ie... yield markdown, code
def set_modelcard_viewmode(view_content, edit_content):
    yield gr.update(value=edit_content, visible=True), gr.update(visible=False)


# -----------------------------------------------------------
# collapes the Model Information Accordian
def set_modelcard_collapse():
    yield gr.update(open=False)

#-----------------------------------------------------------
# send view back to gr.Code code window for the modelcard after loading using .then()
# after loading model info
def set_modelcard_setcode(view_content):
    yield gr.update(visible=True), gr.update(value=view_content, visible=False)

# -----------------------------------------------------------
# send view back to gr.Code code window for the modelcard after loading using .then()
# after loading model info
def set_modelcard_hideedit_buttons():
    yield gr.update(visible=False), gr.update(visible=False)

# -----------------------------------------------------------
# send view back to gr.Code code window for the modelcard after loading using .then()
# after loading model info
def set_modelcard_showedit_buttons():
    yield gr.update(visible=True), gr.update(visible=True)

# -----------------------------------------------------------
# the actual viewer - page/tab - lcm_model images
def show_lcm_model_preview(modelname, input_cmd):
    if input_cmd == 0:
        return "", ""
    if not modelname:
        return "", ""
    html_code_list = ""
    html_header_list = ""
    output_png_list = []
    output_image_list = get_lcm_image_list(modelname)

    # --- Start Page logic ---
    # first
    if input_cmd == 2:
        LLSTUDIO["lcm_page_num"] = 1
    # previous
    if input_cmd == 3:  
        LLSTUDIO["lcm_page_num"] = min(max(LLSTUDIO["lcm_page_num"] - 1, 1), len(output_image_list))
    # next
    if input_cmd == 4:
        LLSTUDIO["lcm_page_num"] = min(max(LLSTUDIO["lcm_page_num"] + 1, 1), len(output_image_list))
    # last
    if input_cmd == 5:
        LLSTUDIO["lcm_page_num"] = int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)

    current_page = LLSTUDIO["lcm_page_num"]
    output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # make sure 'last' page is not empty, if it is, go back one page.
    if len(output_png_list) < 1:
        LLSTUDIO["lcm_page_num"] = LLSTUDIO["lcm_page_num"] - 1
        current_page = LLSTUDIO["lcm_page_num"]
        output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # --- End page logic ---

    mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
    
    if os.path.isfile(mdl_filename):
        model_info = preview_get_model_info_file(mdl_filename)
    else:
        model_info = "No model information found."

    # page header above images
    page_header_output = "<h3>Page " + str(LLSTUDIO["lcm_page_num"]) + " of " + str(int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)) + " - (" + str(len(output_image_list)) + " : Images Total)</h3>"

    
    for i in range(len(output_png_list)):
        png_filename_ext = output_png_list[i]
        png_filename = os.path.splitext(os.path.basename(png_filename_ext))[0]
        if os.path.isfile(png_filename_ext):
            html_code_list = html_code_list + '<table cellpadding=10 cellspacing=10 border=0>'
            html_code_list = html_code_list + '<tr>'
            html_code_list = html_code_list + '<td width=50%>'
            html_code_list = html_code_list + preview_create_html_code(png_filename_ext)
            html_code_list = html_code_list + '</td>'
            txtinfofile = os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname, png_filename + ".txt")
            if os.path.isfile(txtinfofile):
                html_code_list = html_code_list + '<td width=45%>'
                html_code_list = html_code_list + preview_create_text_code(txtinfofile)
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
            else:
                html_code_list = html_code_list + '<td width=50%>'
                html_code_list = html_code_list + 'No Image information found.'
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
        
    if len(output_image_list) < 1:
        html_header_list = "</br><font size=+1>No Images found.</font></br>"
    else:
        html_header_list = "</br><font size=+2>IMAGE GALLERY</font></br>"
    
    html_code_list = html_header_list + page_header_output + html_code_list
    
    return model_info, html_code_list



# -----------------------------------------------------------
# the actual viewer - page/tab - safetensors model images
def show_safe_model_preview(modelname, input_cmd):
    
    if input_cmd == 0:
        return "", ""
    if not modelname:
        return "", ""
    html_code_list = ""
    html_header_list = ""
    output_png_list = []
    output_image_list = get_safe_image_list(modelname)

    # --- Start Page logic ---
    # first
    if input_cmd == 2:
        LLSTUDIO["safe_page_num"] = 1
    # previous
    if input_cmd == 3:  
        LLSTUDIO["safe_page_num"] = min(max(LLSTUDIO["safe_page_num"] - 1, 1), len(output_image_list))
    # next
    if input_cmd == 4:
        LLSTUDIO["safe_page_num"] = min(max(LLSTUDIO["safe_page_num"] + 1, 1), len(output_image_list))
    # last
    if input_cmd == 5:
        LLSTUDIO["safe_page_num"] = int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)

    current_page = LLSTUDIO["safe_page_num"]
    output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # make sure 'last' page is not empty, if it is, go back one page.
    if len(output_png_list) < 1:
        LLSTUDIO["safe_page_num"] = LLSTUDIO["safe_page_num"] - 1
        current_page = LLSTUDIO["safe_page_num"]
        output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # --- End page logic ---

    mdl_filename = (os.path.join(LLSTUDIO["safe_model_image_dir"],modelname,modelname + '.md'))
    
    if os.path.isfile(mdl_filename):
        model_info = preview_get_model_info_file(mdl_filename)
    else:
        model_info = "No model information found."

    # page header above images
    page_header_output = "<h3>Page " + str(LLSTUDIO["safe_page_num"]) + " of " + str(int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)) + " - (" + str(len(output_image_list)) + " : Images Total)</h3>"

    
    for i in range(len(output_png_list)):
        png_filename_ext = output_png_list[i]
        png_filename = os.path.splitext(os.path.basename(png_filename_ext))[0]
        if os.path.isfile(png_filename_ext):
            html_code_list = html_code_list + '<table cellpadding=10 cellspacing=10 border=0>'
            html_code_list = html_code_list + '<tr>'
            html_code_list = html_code_list + '<td width=50%>'
            html_code_list = html_code_list + preview_create_html_code(png_filename_ext)
            html_code_list = html_code_list + '</td>'
            txtinfofile = os.path.join(LLSTUDIO["safe_model_image_dir"],modelname, png_filename + ".txt")
            if os.path.isfile(txtinfofile):
                html_code_list = html_code_list + '<td width=45%>'
                html_code_list = html_code_list + preview_create_text_code(txtinfofile)
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
            else:
                html_code_list = html_code_list + '<td width=50%>'
                html_code_list = html_code_list + 'No Image information found.'
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
        
    if len(output_image_list) < 1:
        html_header_list = "</br><font size=+1>No Images found.</font></br>"
    else:
        html_header_list = "</br><font size=+2>IMAGE GALLERY</font></br>"
    
    html_code_list = html_header_list + page_header_output + html_code_list
    
    return model_info, html_code_list


# -----------------------------------------------------------
# the actual viewer - page/tab - LoRa model images
def show_lora_model_preview(modelname, input_cmd):
    if input_cmd == 0:
        return "", ""
    if not modelname:
        return "", ""
    html_code_list = ""
    html_header_list = ""
    output_png_list = []
    output_image_list = get_lora_image_list(modelname)

    # --- Start Page logic ---
    # first
    if input_cmd == 2:
        LLSTUDIO["lora_page_num"] = 1
    # previous
    if input_cmd == 3:  
        LLSTUDIO["lora_page_num"] = min(max(LLSTUDIO["lora_page_num"] - 1, 1), len(output_image_list))
    # next
    if input_cmd == 4:
        LLSTUDIO["lora_page_num"] = min(max(LLSTUDIO["lora_page_num"] + 1, 1), len(output_image_list))
    # last
    if input_cmd == 5:
        LLSTUDIO["lora_page_num"] = int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)

    current_page = LLSTUDIO["lora_page_num"]
    output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # make sure 'last' page is not empty, if it is, go back one page.
    if len(output_png_list) < 1:
        LLSTUDIO["lora_page_num"] = LLSTUDIO["lora_page_num"] - 1
        current_page = LLSTUDIO["lora_page_num"]
        output_png_list = paginate_list(output_image_list, current_page, int(STUDIO["img_view_img_per_page"]["value"]))
    # --- End page logic ---

    mdl_filename = (os.path.join(LLSTUDIO["lora_model_image_dir"],modelname,modelname + '.md'))
    
    if os.path.isfile(mdl_filename):
        model_info = preview_get_model_info_file(mdl_filename)
    else:
        model_info = "No model information found."

    # page header above images
    page_header_output = "<h3>Page " + str(LLSTUDIO["lora_page_num"]) + " of " + str(int(len(output_image_list)/int(STUDIO["img_view_img_per_page"]["value"])+1)) + " - (" + str(len(output_image_list)) + " : Images Total)</h3>"

    
    for i in range(len(output_png_list)):
        png_filename_ext = output_png_list[i]
        png_filename = os.path.splitext(os.path.basename(png_filename_ext))[0]
        if os.path.isfile(png_filename_ext):
            html_code_list = html_code_list + '<table cellpadding=10 cellspacing=10 border=0>'
            html_code_list = html_code_list + '<tr>'
            html_code_list = html_code_list + '<td width=50%>'
            html_code_list = html_code_list + preview_create_html_code(png_filename_ext)
            html_code_list = html_code_list + '</td>'
            txtinfofile = os.path.join(LLSTUDIO["lora_model_image_dir"],modelname, png_filename + ".txt")
            if os.path.isfile(txtinfofile):
                html_code_list = html_code_list + '<td width=45%>'
                html_code_list = html_code_list + preview_create_text_code(txtinfofile)
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
            else:
                html_code_list = html_code_list + '<td width=50%>'
                html_code_list = html_code_list + 'No Image information found.'
                html_code_list = html_code_list + '</td>'
                html_code_list = html_code_list + '</tr>'
                html_code_list = html_code_list + '</table>'
        
    if len(output_image_list) < 1:
        html_header_list = "</br><font size=+1>No Images found.</font></br>"
    else:
        html_header_list = "</br><font size=+2>IMAGE GALLERY</font></br>"
    
    html_code_list = html_header_list + page_header_output + html_code_list
    
    return model_info, html_code_list




# ------------------------------------------------------------
# get contents of file and return
def get_file_content(file):
    
    file = open(file, "r")
    content = file.read()
    file.close()
    return content



# --------------------------------------------------------------
def gen_random_seed():
    seed = random.randint(0, 2**32 - 1)
    return seed


# ------------------------------------------------------
# Text 2 Image - Image Generation
def t2igen_LCM_images(
    prompt, 
    negative_prompt, 
    width, height, 
    guidance_scale, 
    num_inference_steps, 
    numimgs, 
    rseed, usesameseed, incrementseed, incseedamount, 
    freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2, 
    clip_skip,
    progress=gr.Progress()):

    
    global pipeline             # where the model is loaded to
    
    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)

    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] == "StableDiffusionLatentUpscalePipeline":
        gr.Info("UpScaler2X Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
    
    
    # reset Halt generation flag
    LLSTUDIO["halt_gen"] = 0
    # clear last image and last prompt, need to add last_negative_prompt
    LLSTUDIO['last_image_filename'] = ""
    LLSTUDIO['last_prompt_filename'] = ""
    # enables/disables hidden image to visible image 
    # onchange copy from oimage to oimage2
    # 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 1
    
    # # start setting up the inference arguments/parameters

    # setup our inference arguments dictionary
    inference_args = {}

    # PROMPTS - start -----------------------------------------------------------------
    # STUDIO["use_prompt_embeds"]["value"]
    # Use Normal Prompts, Prompt Embeddings or Prompt Weighting (using Compel).
    # ---------------------------------------------------------------------------------
    # 0=Normal Prompts (76 Max Prompt Tokens)                           BOTH SD/SDXL
    # 1=Prompt Embeddings and Padding                                   BOTH SD/SDXL    
    # 2=Prompt Weighting (Compel) and Prompt Embeddings                 BOTH SD/SDXL    
    # 3=Prompt Weighting (Compel) and Prompt Embeddings and Padding     BOTH SD/SDXL   
    #
    # use_prompt_embeds (always pad)
    # if int(STUDIO["use_prompt_embeds"]["value"]) == 1:
    #
    #
    # ---------------------------------------------------------------------------------
    # # # SDXL-PROMPTS
    # # # PADDING + POOLED + EMBEDS
    # prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
    # negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
    # ---------------------------------------------------------------------------------
    
    # we update the 'progress bar' to 0% at the beginning 
    # of each (use_prompt_embeds/pipeline_model_type) section
    # then we update to 100% when finished
    progress(0.0, desc=f"Creating Prompt Embeds...")

    if int(STUDIO["use_prompt_embeds"]["value"]) == 0:
        # No prompt embeds, No prompt weighting, just the plain prompts
        # check the prompt length for SD15, which can not be longer than 76 tokens
        # this is needed for plain prompts for SD15
        # Diffusers library seems to indicate SDXL is different anyway
        # so we do not check SDXL prompt length
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompts...")
            plen = get_prompt_length(prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Prompt Too Long." + "</br>Prompt Length = " + str(plen), duration=5.0, title="Prompt Length > 76")    
                return
            progress(.40, desc=f"Creating Prompts...")
            plen = get_prompt_length(negative_prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Negative Prompt Too Long." + "</br>Negative Prompt Length = " + str(plen), duration=5.0, title="Negative Prompt Length > 76")    
                return
            progress(.95, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SD15)"
            progress(1.0, desc=f"Finished Creating Prompts.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SDXL)"
            progress(1.0, desc=f"Finished Creating Prompts.")

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 1:
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SD15 embedded prompts - diffusers
            prompt_embeds, negative_prompt_embeds = do_prompt_embeds(LLSTUDIO["device"], pipeline, prompt, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # embedded prompts
            inference_args["prompt_embeds"] = prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SDXL embedded prompts - diffusers
            # # SDXL-PROMPTS
            # # PADDING + POOLED + EMBEDS
            prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
            progress(.45, desc=f"Creating Prompt Embeds...")
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # set inference arguments
            inference_args["prompt_embeds"] = prompt_embeds
            inference_args["pooled_prompt_embeds"] = pooled_prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            inference_args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
            
    elif int(STUDIO["use_prompt_embeds"]["value"]) == 2:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel embedded prompts
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # embedded prompts
                inference_args["prompt_embeds"] = compel_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = compel_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel pooled + embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True]
                )
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings from the first text encoder
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 3:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel padded embeds
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                prompt_embeds = compel_proc.build_conditioning_tensor(prompt)
                progress(.5, desc=f"Creating Weighted Prompt Embeds...")
                negative_prompt_embeds = compel_proc.build_conditioning_tensor(negative_prompt)
                progress(.75, desc=f"Creating Weighted Prompt Embeds...")
                [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
                # embedded prompts
                inference_args["prompt_embeds"] = prompt_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = negative_prompt_embeds
                prompt_type = "Compel Embedded Prompts Pad Same Length"
                progress(1, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel padded + pooled embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel with padding enabled
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False   # don't truncate, pad instead
                )
                progress(.1, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # Pad to same length using Compel helper
                [pos_prompt_embeds, neg_prompt_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_prompt_embeds, neg_prompt_embeds])
                # progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # # NO REAL EXAMPLE FOUND, but doing it anyway, then check for errors, and image output
                # [pos_pooled_embeds, neg_pooled_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_pooled_embeds, neg_pooled_embeds])
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts Pad Length"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    # PROMPTS - end -------------------------------------------------------------------

    
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs

    inference_args["width"] = width
    inference_args["height"] = height
    # Guidance scale is enabled when `guidance_scale > 1
    inference_args["guidance_scale"] = guidance_scale
    inference_args["num_inference_steps"] = num_inference_steps
    inference_args["callback_on_step_end"] = callback_on_step_end

    # clip_skip can only be use on SD15, not SDXL. 
    if SDPIPELINE["pipeline_model_type"]=="SD15":
        # Number of layers to be skipped from CLIP while computing the prompt embeddings. 
        # A value of 1 means that
        # the output of the pre-final layer will be used for computing the prompt embeddings.
        if clip_skip > 0:
            inference_args["clip_skip"] = clip_skip

    # input seed to local seed variable that we manipulate after each generation
    myseed=rseed
    # LOOP for multiple image generation
    for i in range(0, numimgs):
        imgnumb = i+1
        # Decide how to handle the seed.
        # two checkboxes, 'incrementseed' and 'usesameseed'
        # if the 'incrementseed' is checked, no randomization
        # and seed is incremented by 'x' amount 'after' first image
        # therefore uses sent seed as starting seed.
        # if the 'incrementseed' is UNchecked, USES randomization
        # if the 'usesameseed' is also checked, uses sent seed 
        # as starting seed. elsewise it starts on a random seed
        # and sent seed is not used
        if incrementseed:
            if imgnumb > 1:
                myseed = myseed + incseedamount
        else:
            if not usesameseed:
                myseed=gen_random_seed()    # change to  random start seed rnd_start_seed check
            else:
                if imgnumb > 1:
                    myseed=gen_random_seed()
    
        # set the seed for inference  
        # we use 'diffusers.training_utils.set_seed' instead of 'torch generator'
        # may switch to 'torch generator' later -or- provide 'setting' to switch
        set_seed(myseed)
        
        
        if len(str(STUDIO["output_image_datetime"]["value"])) > 0:
            # Get the current date and time
            now = datetime.now()
            # Get the current local time as a struct_time object
            timestamp_str = now.strftime(str(STUDIO["output_image_datetime"]["value"]))
            # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
            formatted_time = timestamp_str
        else:
            formatted_time = ""
            
        # go ahead and set the image and txt filename now, so we can display it to user while running inference
        imagebasename = STUDIO["output_image_prefix"]["value"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + STUDIO["output_image_suffix"]["value"] 
        imagefilename = imagebasename + ".png"
        textfilename = imagebasename + ".txt"
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

        # we init the progress bar, rknote needs to be below check model loaded...
        progress(0, desc=f"Starting Inference. Step 1 of {num_inference_steps} - Image# {imgnumb} of {numimgs}")

        # mark start time
        pstart = time.time()
        
        # check if using FreeU or not
        if freeu: 
            pipeline.enable_freeu(s1=float(freeu_s1), s2=float(freeu_s2), b1=float(freeu_b1), b2=float(freeu_b2))
        else:
            pipeline.disable_freeu()

        
        with torch.no_grad():
            # run inference
            image2 = pipeline(**inference_args).images[0]


        # save the image generated
        image2.save(os.path.join(LLSTUDIO["output_image_dir"], imagefilename), "png")
        
        
        image2 = None
        del image2
        gc.collect()

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filename
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["output_image_dir"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["output_image_dir"], imagefilename)
        
        # create text for image generation parameters image'.txt' file
        text_output = prompt + "\n\n"
        if negative_prompt:
            text_output = text_output + negative_prompt + "\n\n"
        text_output = text_output + "Steps: " + str(num_inference_steps) + ", "
        text_output = text_output + "CFG scale: " + str(guidance_scale) + ", "
        text_output = text_output + "Seed: " + str(myseed) + ", "
        text_output = text_output + "Size: " + str(width) + "x"  + str(height)+ "\n"
        text_output = text_output + "Pipeline: " + str(SDPIPELINE['pipeline_class']) + "\n"
        text_output = text_output + "Model Loaded From: " + str(SDPIPELINE['pipeline_source']) + "\n"
        text_output = text_output + "Model Type: " + str(SDPIPELINE['pipeline_model_type']) + "\n"
        text_output = text_output + "Model: " + str(SDPIPELINE['pipeline_model_name']) + "\n"
        if SDPIPELINE["pipeline_text_encoder"] > 0:
            text_output = text_output + "Used Text Encoder from: " + SDPIPELINE["pipeline_text_encoder_name"] + "\n"
            text_output = text_output + "ClipSkip Value: " + str(clip_skip) + "\n"
        text_output = text_output + get_loaded_lora_models_text()
        text_output = text_output + "Image Filename: " + imagefilename + "\n"
        text_output = text_output + "Inference Time: " + format_seconds_strftime(pelapsed) + "\n"
        text_output = text_output + "Generation Method: " + SDPIPELINE["pipeline_gen_mode"] + "\n"
        text_output = text_output + "Prompt Type: " + prompt_type + "\n"
        if freeu: 
            text_output = text_output + "FreeU Enabled:\n"
            text_output = text_output + "FreeU Values: s1=" + freeu_s1 + ", s2=" + freeu_s2 + ", b1=" + freeu_b1 + ", b2=" + freeu_b2 + "\n"


        # write image generation parameters image'.txt' file
        file1 = open(LLSTUDIO['last_prompt_filename'], 'w')
        file1.write(text_output)
        file1.close()
        
        # write image generation parameters to 'last_prompt.txt' file
        file1 = open(os.path.join(".", "last_prompt.txt"), 'w')
        file1.write(text_output)
        file1.close()
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Generating Image# " + str(imgnumb) + " of " + str(numimgs))
        
        
        # # let's try and reduce the size of the font on the output 'label '
        a1 = "Finished Saving: " + str(imagefilename) + "<br>"
        a1 = a1 + "Image " + str(imgnumb) + " of " + str(numimgs)

        # # yield the data to both gradio outputs [progress/text,img]
        yield gr.update(value=a1), gr.update(value=LLSTUDIO['last_image_filename'])
        
        # update header/title - memory stats usage - plus apply rkmemopt
        update_grapptitle_mem()
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            
            
            prompt_embeds = None
            negative_prompt_embeds = None
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            compel_proc = None
            compel_sdxl_proc = None
            pos_prompt_embeds = None
            pos_pooled_embeds = None
            neg_prompt_embeds = None
            neg_pooled_embeds = None
            del prompt_embeds
            del negative_prompt_embeds
            del pooled_prompt_embeds
            del negative_pooled_prompt_embeds
            del compel_proc
            del compel_sdxl_proc
            del pos_prompt_embeds
            del pos_pooled_embeds
            del neg_prompt_embeds
            del neg_pooled_embeds
            gc.collect()
            
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']
            
    
    prompt_embeds = None
    negative_prompt_embeds = None
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    compel_proc = None
    compel_sdxl_proc = None
    pos_prompt_embeds = None
    pos_pooled_embeds = None
    neg_prompt_embeds = None
    neg_pooled_embeds = None
    del prompt_embeds
    del negative_prompt_embeds
    del pooled_prompt_embeds
    del negative_pooled_prompt_embeds
    del compel_proc
    del compel_sdxl_proc
    del pos_prompt_embeds
    del pos_pooled_embeds
    del neg_prompt_embeds
    del neg_pooled_embeds
    gc.collect()

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



    
# ------------------------------------------------------
# Image 2 Image - Image Generation
def i2igen_LCM_images(
    prompt, 
    negative_prompt, 
    width, height, 
    guidance_scale, 
    num_inference_steps, 
    rseed, 
    numimgs, 
    incrementseed, 
    incseedamount, 
    image, 
    resizeimage, 
    freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2, 
    clip_skip,
    strength,
    progress=gr.Progress()
    ):
    
    
    
    global pipeline             # where the model is loaded to
    
    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)

    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] == "StableDiffusionLatentUpscalePipeline":
        gr.Info("UpScaler2X Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
    
        

    # reset Halt generation flag
    LLSTUDIO["halt_gen"] = 0
    # clear last image and last prompt, need to add last_negative_prompt
    LLSTUDIO['last_image_filename'] = ""
    LLSTUDIO['last_prompt_filename'] = ""
    # enables/disables hidden image to visible image 
    # onchange copy from oimage to oimage2
    # 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 1
    
    # # start setting up the inference arguments/parameters

    # setup our inference arguments dictionary
    inference_args = {}


    # PROMPTS - start -----------------------------------------------------------------
    # STUDIO["use_prompt_embeds"]["value"]
    # Use Normal Prompts, Prompt Embeddings or Prompt Weighting (using Compel).
    # ---------------------------------------------------------------------------------
    # 0=Normal Prompts (76 Max Prompt Tokens)                           BOTH SD/SDXL
    # 1=Prompt Embeddings and Padding                                   BOTH SD/SDXL    
    # 2=Prompt Weighting (Compel) and Prompt Embeddings                 BOTH SD/SDXL    
    # 3=Prompt Weighting (Compel) and Prompt Embeddings and Padding     BOTH SD/SDXL   
    #
    # use_prompt_embeds (always pad)
    # if int(STUDIO["use_prompt_embeds"]["value"]) == 1:
    #
    #
    # ---------------------------------------------------------------------------------
    # # # SDXL-PROMPTS
    # # # PADDING + POOLED + EMBEDS
    # prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
    # negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
    # ---------------------------------------------------------------------------------
    
    # we update the 'progress bar' to 0% at the beginning 
    # of each (use_prompt_embeds/pipeline_model_type) section
    # then we update to 100% when finished
    progress(0.0, desc=f"Creating Prompt Embeds...")

    if int(STUDIO["use_prompt_embeds"]["value"]) == 0:
        # DONE
        # No prompt embeds, No prompt weighting, just the plain prompts
        # check the prompt length for SD15, which can not be longer than 76 tokens
        # this is needed for plain prompts for SD15
        # Diffusers library seems to indicate SDXL is different anyway
        # so we do not check SDXL prompt length
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompts...")
            plen = get_prompt_length(prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Prompt Too Long." + "</br>Prompt Length = " + str(plen), duration=5.0, title="Prompt Length > 76")    
                return
            progress(.40, desc=f"Creating Prompts...")
            plen = get_prompt_length(negative_prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Negative Prompt Too Long." + "</br>Negative Prompt Length = " + str(plen), duration=5.0, title="Negative Prompt Length > 76")    
                return
            progress(.95, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SD15)"
            progress(1.0, desc=f"Finished Creating Prompts.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SDXL)"
            progress(1.0, desc=f"Finished Creating Prompts.")

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 1:
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SD15 embedded prompts - diffusers
            prompt_embeds, negative_prompt_embeds = do_prompt_embeds(LLSTUDIO["device"], pipeline, prompt, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # embedded prompts
            inference_args["prompt_embeds"] = prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SDXL embedded prompts - diffusers
            # # SDXL-PROMPTS
            # # PADDING + POOLED + EMBEDS
            prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
            progress(.45, desc=f"Creating Prompt Embeds...")
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # set inference arguments
            inference_args["prompt_embeds"] = prompt_embeds
            inference_args["pooled_prompt_embeds"] = pooled_prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            inference_args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
            
    elif int(STUDIO["use_prompt_embeds"]["value"]) == 2:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel embedded prompts
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # embedded prompts
                inference_args["prompt_embeds"] = compel_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = compel_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel pooled + embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True]
                )
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings from the first text encoder
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 3:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel padded embeds
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                prompt_embeds = compel_proc.build_conditioning_tensor(prompt)
                progress(.5, desc=f"Creating Weighted Prompt Embeds...")
                negative_prompt_embeds = compel_proc.build_conditioning_tensor(negative_prompt)
                progress(.75, desc=f"Creating Weighted Prompt Embeds...")
                [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
                # embedded prompts
                inference_args["prompt_embeds"] = prompt_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = negative_prompt_embeds
                prompt_type = "Compel Embedded Prompts Pad Same Length"
                progress(1, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel padded + pooled embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel with padding enabled
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False   # don't truncate, pad instead
                )
                progress(.1, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # Pad to same length using Compel helper
                [pos_prompt_embeds, neg_prompt_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_prompt_embeds, neg_prompt_embeds])
                # progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # # NO REAL EXAMPLE FOUND, but doing it anyway, then check for errors, and image output
                # [pos_pooled_embeds, neg_pooled_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_pooled_embeds, neg_pooled_embeds])
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts Pad Length"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    # PROMPTS - end -------------------------------------------------------------------

    
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs
    

    # resize input image to 512x512
    if resizeimage:
        new_width = 512
        new_height = 512
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_img = image


    inference_args["width"] = width
    inference_args["height"] = height
    # we resize input image to 512x512
    inference_args["image"] = resized_img

    # strength Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
    # starting point and more noise is added the higher the `strength`. The number of denoising steps depends
    # on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
    # process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
    # essentially ignores `image`.
    inference_args["strength"] = strength
    # Guidance scale is enabled when `guidance_scale > 1
    inference_args["guidance_scale"] = guidance_scale
    inference_args["num_inference_steps"] = num_inference_steps
    inference_args["callback_on_step_end"] = callback_on_step_end

    # clip_skip can only be use on SD15, not SDXL. 
    if SDPIPELINE["pipeline_model_type"]=="SD15":
        # Number of layers to be skipped from CLIP while computing the prompt embeddings. 
        # A value of 1 means that
        # the output of the pre-final layer will be used for computing the prompt embeddings.
        if clip_skip > 0:
            inference_args["clip_skip"] = clip_skip



    # input seed to local seed variable that we manipulate after each generation
    myseed=rseed
    # LOOP for multiple image generation
    for i in range(0, numimgs):
        imgnumb = i+1
        # Decide how to handle the seed.
        # two checkboxes, 'incrementseed' and 'usesameseed'
        # if the 'incrementseed' is checked, no randomization
        # and seed is incremented by 'x' amount 'after' first image
        # therefore uses sent seed as starting seed.
        # if the 'incrementseed' is UNchecked, USES randomization
        # if the 'usesameseed' is also checked, uses sent seed 
        # as starting seed. elsewise it starts on a random seed
        # and sent seed is not used
        usesameseed=False
        
        if incrementseed:
            if imgnumb > 1:
                myseed = myseed + incseedamount
        else:
            if not usesameseed:
                myseed=gen_random_seed()    # change to  random start seed rnd_start_seed check
            else:
                if imgnumb > 1:
                    myseed=gen_random_seed()
    
        # set the seed for inference  
        # we use 'diffusers.training_utils.set_seed' instead of 'torch generator'
        # may switch to 'torch generator' later -or- provide 'setting' to switch
        set_seed(myseed)
        
        if len(str(STUDIO["output_image_datetime"]["value"])) > 0:
            # Get the current date and time
            now = datetime.now()
            # Get the current local time as a struct_time object
            timestamp_str = now.strftime(str(STUDIO["output_image_datetime"]["value"]))
            # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
            formatted_time = timestamp_str
        else:
            formatted_time = ""
             
        # go ahead and set the image and txt filename now, so we can display it to user while running inference
        imagebasename = STUDIO["output_image_prefix"]["value"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + STUDIO["output_image_suffix"]["value"] 
        imagefilename = imagebasename + ".png"
        textfilename = imagebasename + ".txt"
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

        # we init the progress bar, rknote needs to be below check model loaded...
        progress(0, desc=f"Starting Inference. Step 1 of {num_inference_steps} - Image# {imgnumb} of {numimgs}")


        # Run inference
        pstart = time.time()

        # check if using FreeU or not
        if freeu: 
            pipeline.enable_freeu(s1=float(freeu_s1), s2=float(freeu_s2), b1=float(freeu_b1), b2=float(freeu_b2))
        else:
            pipeline.disable_freeu()
            
            
        with torch.no_grad():
            # run inference
            image2 = pipeline(**inference_args).images[0]


        # save the image generated
        image2.save(os.path.join(LLSTUDIO["output_image_dir"], imagefilename), "png")
        
        
        image2 = None
        del image2
        gc.collect()

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["output_image_dir"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["output_image_dir"], imagefilename)
        
        # create text for image generation parameters image'.txt' file
        text_output = prompt + "\n\n"
        if negative_prompt:
            text_output = text_output + negative_prompt + "\n\n"
        text_output = text_output + "Steps: " + str(num_inference_steps) + ", "
        text_output = text_output + "CFG scale: " + str(guidance_scale) + ", "
        text_output = text_output + "Seed: " + str(myseed) + ", "
        text_output = text_output + "Size: " + str(width) + "x"  + str(height)+ "\n"
        text_output = text_output + "Pipeline: " + str(SDPIPELINE['pipeline_class']) + "\n"
        text_output = text_output + "Model Loaded From: " + str(SDPIPELINE['pipeline_source']) + "\n"
        text_output = text_output + "Model Type: " + str(SDPIPELINE['pipeline_model_type']) + "\n"
        text_output = text_output + "Model: " + str(SDPIPELINE['pipeline_model_name']) + "\n"
        if SDPIPELINE["pipeline_text_encoder"] > 0:
            text_output = text_output + "Used Text Encoder from: " + SDPIPELINE["pipeline_text_encoder_name"] + "\n"
            text_output = text_output + "ClipSkip Value: " + str(clip_skip) + "\n"
        text_output = text_output + get_loaded_lora_models_text()
        text_output = text_output + "Image Filename: " + imagefilename + "\n"
        text_output = text_output + "Inference Time: " + format_seconds_strftime(pelapsed) + "\n"
        text_output = text_output + "Generation Method: " + SDPIPELINE["pipeline_gen_mode"] + "\n"
        text_output = text_output + "Prompt Type: " + prompt_type + "\n"
        if freeu: 
            text_output = text_output + "FreeU Enabled:\n"
            text_output = text_output + "FreeU Values: s1=" + freeu_s1 + ", s2=" + freeu_s2 + ", b1=" + freeu_b1 + ", b2=" + freeu_b2 + "\n"


        # write image generation parameters image'.txt' file
        file1 = open(LLSTUDIO['last_prompt_filename'], 'w')
        file1.write(text_output)
        file1.close()
        
        # write image generation parameters to 'last_prompt.txt' file
        file1 = open(os.path.join(".", "last_prompt.txt"), 'w')
        file1.write(text_output)
        file1.close()
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Generating Image# " + str(imgnumb) + " of " + str(numimgs))
        
        
        # # let's try and reduce the size of the font on the output 'label '
        a1 = "Finished Saving: " + str(imagefilename) + "<br>"
        a1 = a1 + "Image " + str(imgnumb) + " of " + str(numimgs)

        # # yield the data to both gradio outputs [progress/text,img]
        yield gr.update(value=a1), gr.update(value=LLSTUDIO['last_image_filename'])
        
        # update header/title - memory stats usage - plus apply rkmemopt
        update_grapptitle_mem()
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# Image Inpainting - Image Generation
def inpgen_LCM_images(
    prompt, 
    negative_prompt,
    width, 
    height, 
    guidance_scale, 
    num_inference_steps, 
    rseed, 
    numimgs, 
    incrementseed, 
    incseedamount, 
    image, 
    resizeimage, 
    maskimage, 
    freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2, 
    clip_skip,
    strength,
    progress=gr.Progress()
    ):
    
    
    global pipeline             # where the model is loaded to
    
    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)

    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] == "StableDiffusionLatentUpscalePipeline":
        gr.Info("UpScaler2X Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
        

    # reset Halt generation flag
    LLSTUDIO["halt_gen"] = 0
    # clear last image and last prompt, need to add last_negative_prompt
    LLSTUDIO['last_image_filename'] = ""
    LLSTUDIO['last_prompt_filename'] = ""
    # enables/disables hidden image to visible image 
    # onchange copy from oimage to oimage2
    # 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 1
    
    # # start setting up the inference arguments/parameters

    # setup our inference arguments dictionary
    inference_args = {}


    # PROMPTS - start -----------------------------------------------------------------
    # STUDIO["use_prompt_embeds"]["value"]
    # Use Normal Prompts, Prompt Embeddings or Prompt Weighting (using Compel).
    # ---------------------------------------------------------------------------------
    # 0=Normal Prompts (76 Max Prompt Tokens)                           BOTH SD/SDXL
    # 1=Prompt Embeddings and Padding                                   BOTH SD/SDXL    
    # 2=Prompt Weighting (Compel) and Prompt Embeddings                 BOTH SD/SDXL    
    # 3=Prompt Weighting (Compel) and Prompt Embeddings and Padding     BOTH SD/SDXL   
    #
    # use_prompt_embeds (always pad)
    # if int(STUDIO["use_prompt_embeds"]["value"]) == 1:
    #
    #
    # ---------------------------------------------------------------------------------
    # # # SDXL-PROMPTS
    # # # PADDING + POOLED + EMBEDS
    # prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
    # negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
    # ---------------------------------------------------------------------------------
    
    # we update the 'progress bar' to 0% at the beginning 
    # of each (use_prompt_embeds/pipeline_model_type) section
    # then we update to 100% when finished
    progress(0.0, desc=f"Creating Prompt Embeds...")

    if int(STUDIO["use_prompt_embeds"]["value"]) == 0:
        # DONE
        # No prompt embeds, No prompt weighting, just the plain prompts
        # check the prompt length for SD15, which can not be longer than 76 tokens
        # this is needed for plain prompts for SD15
        # Diffusers library seems to indicate SDXL is different anyway
        # so we do not check SDXL prompt length
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompts...")
            plen = get_prompt_length(prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Prompt Too Long." + "</br>Prompt Length = " + str(plen), duration=5.0, title="Prompt Length > 76")    
                return
            progress(.40, desc=f"Creating Prompts...")
            plen = get_prompt_length(negative_prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Negative Prompt Too Long." + "</br>Negative Prompt Length = " + str(plen), duration=5.0, title="Negative Prompt Length > 76")    
                return
            progress(.95, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SD15)"
            progress(1.0, desc=f"Finished Creating Prompts.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SDXL)"
            progress(1.0, desc=f"Finished Creating Prompts.")

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 1:
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SD15 embedded prompts - diffusers
            prompt_embeds, negative_prompt_embeds = do_prompt_embeds(LLSTUDIO["device"], pipeline, prompt, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # embedded prompts
            inference_args["prompt_embeds"] = prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SDXL embedded prompts - diffusers
            # # SDXL-PROMPTS
            # # PADDING + POOLED + EMBEDS
            prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
            progress(.45, desc=f"Creating Prompt Embeds...")
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # set inference arguments
            inference_args["prompt_embeds"] = prompt_embeds
            inference_args["pooled_prompt_embeds"] = pooled_prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            inference_args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
            
    elif int(STUDIO["use_prompt_embeds"]["value"]) == 2:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel embedded prompts
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # embedded prompts
                inference_args["prompt_embeds"] = compel_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = compel_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel pooled + embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True]
                )
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings from the first text encoder
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 3:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel padded embeds
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                prompt_embeds = compel_proc.build_conditioning_tensor(prompt)
                progress(.5, desc=f"Creating Weighted Prompt Embeds...")
                negative_prompt_embeds = compel_proc.build_conditioning_tensor(negative_prompt)
                progress(.75, desc=f"Creating Weighted Prompt Embeds...")
                [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
                # embedded prompts
                inference_args["prompt_embeds"] = prompt_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = negative_prompt_embeds
                prompt_type = "Compel Embedded Prompts Pad Same Length"
                progress(1, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel padded + pooled embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel with padding enabled
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False   # don't truncate, pad instead
                )
                progress(.1, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # Pad to same length using Compel helper
                [pos_prompt_embeds, neg_prompt_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_prompt_embeds, neg_prompt_embeds])
                # progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # # NO REAL EXAMPLE FOUND, but doing it anyway, then check for errors, and image output
                # [pos_pooled_embeds, neg_pooled_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_pooled_embeds, neg_pooled_embeds])
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts Pad Length"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    # PROMPTS - end -------------------------------------------------------------------
        
    
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs


    # resize input image to 512x512, and mask image to 512x512
    # if both images are same size to begin with, and mask is correctly
    # aligned, should resize ok, with the exception of width/height distortion
    # ie: the aspect ratio
    if resizeimage:
        new_width = 512
        new_height = 512
        # we init the progress bar, rknote needs to be below check model loaded...
        progress(0, desc=f"Resizing Input Image...")
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        # we init the progress bar, rknote needs to be below check model loaded...
        progress(50, desc=f"Resizing Input Mask Image...")
        resized_maskimg = maskimage.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_img = image
        resized_maskimg = maskimage


    inference_args["width"] = width
    inference_args["height"] = height
    # we resize input image to 512x512
    inference_args["image"] = resized_img
    # we resize input image mask to 512x512
    inference_args["mask_image"] = resized_maskimg

    # strength Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
    # starting point and more noise is added the higher the `strength`. The number of denoising steps depends
    # on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
    # process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
    # essentially ignores `image`.
    inference_args["strength"] = strength  # Indicates extent to transform the reference `image`. Must be between 0 and 1.
    # Guidance scale is enabled when `guidance_scale > 1
    inference_args["guidance_scale"] = guidance_scale
    inference_args["num_inference_steps"] = num_inference_steps
    inference_args["callback_on_step_end"] = callback_on_step_end

    # clip_skip can only be use on SD15, not SDXL. 
    if SDPIPELINE["pipeline_model_type"]=="SD15":
        # Number of layers to be skipped from CLIP while computing the prompt embeddings. 
        # A value of 1 means that
        # the output of the pre-final layer will be used for computing the prompt embeddings.
        if clip_skip > 0:
            inference_args["clip_skip"] = clip_skip



    # input seed to local seed variable that we manipulate after each generation
    myseed=rseed
    # LOOP for multiple image generation
    for i in range(0, numimgs):
        imgnumb = i+1
        # Decide how to handle the seed.
        # two checkboxes, 'incrementseed' and 'usesameseed'
        # if the 'incrementseed' is checked, no randomization
        # and seed is incremented by 'x' amount 'after' first image
        # therefore uses sent seed as starting seed.
        # if the 'incrementseed' is UNchecked, USES randomization
        # if the 'usesameseed' is also checked, uses sent seed 
        # as starting seed. elsewise it starts on a random seed
        # and sent seed is not used
        usesameseed=False
        if incrementseed:
            if imgnumb > 1:
                myseed = myseed + incseedamount
        else:
            if not usesameseed:
                myseed=gen_random_seed()    # change to  random start seed rnd_start_seed check
            else:
                if imgnumb > 1:
                    myseed=gen_random_seed()
    
        # set the seed for inference  
        # we use 'diffusers.training_utils.set_seed' instead of 'torch generator'
        # may switch to 'torch generator' later -or- provide 'setting' to switch
        set_seed(myseed)
        
        if len(str(STUDIO["output_image_datetime"]["value"])) > 0:
            # Get the current date and time
            now = datetime.now()
            # Get the current local time as a struct_time object
            timestamp_str = now.strftime(str(STUDIO["output_image_datetime"]["value"]))
            # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
            formatted_time = timestamp_str
        else:
            formatted_time = ""
             
        # go ahead and set the image and txt filename now, so we can display it to user while running inference
        imagebasename = STUDIO["output_image_prefix"]["value"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + STUDIO["output_image_suffix"]["value"] 
        imagefilename = imagebasename + ".png"
        textfilename = imagebasename + ".txt"
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

        # mark start time
        pstart = time.time()
        

        # we init the progress bar, rknote needs to be below check model loaded...
        progress(0, desc=f"Starting Inference. Step 1 of {num_inference_steps} - Image# {imgnumb} of {numimgs}")

        # check if using FreeU or not
        if freeu: 
            pipeline.enable_freeu(s1=float(freeu_s1), s2=float(freeu_s2), b1=float(freeu_b1), b2=float(freeu_b2))
        else:
            pipeline.disable_freeu()



        with torch.no_grad():
            # run inference
            image2 = pipeline(**inference_args).images[0]



        # save the image generated
        image2.save(os.path.join(LLSTUDIO["output_image_dir"], imagefilename), "png")
        
        
        image2 = None
        del image2
        gc.collect()

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["output_image_dir"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["output_image_dir"], imagefilename)
        
        # create text for image generation parameters image'.txt' file
        text_output = prompt + "\n\n"
        if negative_prompt:
            text_output = text_output + negative_prompt + "\n\n"
        text_output = text_output + "Steps: " + str(num_inference_steps) + ", "
        text_output = text_output + "CFG scale: " + str(guidance_scale) + ", "
        text_output = text_output + "Seed: " + str(myseed) + ", "
        text_output = text_output + "Size: " + str(width) + "x"  + str(height)+ "\n"
        text_output = text_output + "Pipeline: " + str(SDPIPELINE['pipeline_class']) + "\n"
        text_output = text_output + "Model Loaded From: " + str(SDPIPELINE['pipeline_source']) + "\n"
        text_output = text_output + "Model Type: " + str(SDPIPELINE['pipeline_model_type']) + "\n"
        text_output = text_output + "Model: " + str(SDPIPELINE['pipeline_model_name']) + "\n"
        if SDPIPELINE["pipeline_text_encoder"] > 0:
            text_output = text_output + "Used Text Encoder from: " + SDPIPELINE["pipeline_text_encoder_name"] + "\n"
            text_output = text_output + "ClipSkip Value: " + str(clip_skip) + "\n"
        text_output = text_output + get_loaded_lora_models_text()
        text_output = text_output + "Image Filename: " + imagefilename + "\n"
        text_output = text_output + "Inference Time: " + format_seconds_strftime(pelapsed) + "\n"
        text_output = text_output + "Generation Method: " + SDPIPELINE["pipeline_gen_mode"] + "\n"
        text_output = text_output + "Prompt Type: " + prompt_type + "\n"
        if freeu: 
            text_output = text_output + "FreeU Enabled:\n"
            text_output = text_output + "FreeU Values: s1=" + freeu_s1 + ", s2=" + freeu_s2 + ", b1=" + freeu_b1 + ", b2=" + freeu_b2 + "\n"


        # write image generation parameters image'.txt' file
        file1 = open(LLSTUDIO['last_prompt_filename'], 'w')
        file1.write(text_output)
        file1.close()
        
        # write image generation parameters to 'last_prompt.txt' file
        file1 = open(os.path.join(".", "last_prompt.txt"), 'w')
        file1.write(text_output)
        file1.close()
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Generating Image# " + str(imgnumb) + " of " + str(numimgs))
        
        
        # # let's try and reduce the size of the font on the output 'label '
        a1 = "Finished Saving: " + str(imagefilename) + "<br>"
        a1 = a1 + "Image " + str(imgnumb) + " of " + str(numimgs)

        # # yield the data to both gradio outputs [progress/text,img]
        yield gr.update(value=a1), gr.update(value=LLSTUDIO['last_image_filename'])
        
        # update header/title - memory stats usage - plus apply rkmemopt
        update_grapptitle_mem()
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# InstructPix2Pix - Image Generation
def ip2pgen_LCM_images(
    prompt, 
    negative_prompt, 
    guidance_scale, 
    num_inference_steps, 
    rseed, 
    numimgs, 
    incrementseed, 
    incseedamount, 
    image, 
    resizeimage, 
    image_guidance_scale, 
    freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2, 
    clip_skip,
    progress=gr.Progress()
    ):
    
    
    global pipeline             # where the model is loaded to
    
    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)

    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] == "StableDiffusionLatentUpscalePipeline":
        gr.Info("UpScaler2X Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
    
        
    # reset Halt generation flag
    LLSTUDIO["halt_gen"] = 0
    # clear last image and last prompt, need to add last_negative_prompt
    LLSTUDIO['last_image_filename'] = ""
    LLSTUDIO['last_prompt_filename'] = ""
    # enables/disables hidden image to visible image 
    # onchange copy from oimage to oimage2
    # 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 1
    
    # # start setting up the inference arguments/parameters

    # setup our inference arguments dictionary
    inference_args = {}

    # PROMPTS - start -----------------------------------------------------------------
    # STUDIO["use_prompt_embeds"]["value"]
    # Use Normal Prompts, Prompt Embeddings or Prompt Weighting (using Compel).
    # ---------------------------------------------------------------------------------
    # 0=Normal Prompts (76 Max Prompt Tokens)                           BOTH SD/SDXL
    # 1=Prompt Embeddings and Padding                                   BOTH SD/SDXL    
    # 2=Prompt Weighting (Compel) and Prompt Embeddings                 BOTH SD/SDXL    
    # 3=Prompt Weighting (Compel) and Prompt Embeddings and Padding     BOTH SD/SDXL   
    #
    # use_prompt_embeds (always pad)
    # if int(STUDIO["use_prompt_embeds"]["value"]) == 1:
    #
    #
    # ---------------------------------------------------------------------------------
    # # # SDXL-PROMPTS
    # # # PADDING + POOLED + EMBEDS
    # prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
    # negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
    # ---------------------------------------------------------------------------------
    
    # we update the 'progress bar' to 0% at the beginning 
    # of each (use_prompt_embeds/pipeline_model_type) section
    # then we update to 100% when finished
    progress(0.0, desc=f"Creating Prompt Embeds...")

    if int(STUDIO["use_prompt_embeds"]["value"]) == 0:
        # DONE
        # No prompt embeds, No prompt weighting, just the plain prompts
        # check the prompt length for SD15, which can not be longer than 76 tokens
        # this is needed for plain prompts for SD15
        # Diffusers library seems to indicate SDXL is different anyway
        # so we do not check SDXL prompt length
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompts...")
            plen = get_prompt_length(prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Prompt Too Long." + "</br>Prompt Length = " + str(plen), duration=5.0, title="Prompt Length > 76")    
                return
            progress(.40, desc=f"Creating Prompts...")
            plen = get_prompt_length(negative_prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Negative Prompt Too Long." + "</br>Negative Prompt Length = " + str(plen), duration=5.0, title="Negative Prompt Length > 76")    
                return
            progress(.95, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SD15)"
            progress(1.0, desc=f"Finished Creating Prompts.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SDXL)"
            progress(1.0, desc=f"Finished Creating Prompts.")

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 1:
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SD15 embedded prompts - diffusers
            prompt_embeds, negative_prompt_embeds = do_prompt_embeds(LLSTUDIO["device"], pipeline, prompt, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # embedded prompts
            inference_args["prompt_embeds"] = prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SDXL embedded prompts - diffusers
            # # SDXL-PROMPTS
            # # PADDING + POOLED + EMBEDS
            prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
            progress(.45, desc=f"Creating Prompt Embeds...")
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # set inference arguments
            inference_args["prompt_embeds"] = prompt_embeds
            inference_args["pooled_prompt_embeds"] = pooled_prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            inference_args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
            
    elif int(STUDIO["use_prompt_embeds"]["value"]) == 2:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel embedded prompts
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # embedded prompts
                inference_args["prompt_embeds"] = compel_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = compel_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel pooled + embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True]
                )
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings from the first text encoder
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 3:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel padded embeds
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                prompt_embeds = compel_proc.build_conditioning_tensor(prompt)
                progress(.5, desc=f"Creating Weighted Prompt Embeds...")
                negative_prompt_embeds = compel_proc.build_conditioning_tensor(negative_prompt)
                progress(.75, desc=f"Creating Weighted Prompt Embeds...")
                [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
                # embedded prompts
                inference_args["prompt_embeds"] = prompt_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = negative_prompt_embeds
                prompt_type = "Compel Embedded Prompts Pad Same Length"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel padded + pooled embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel with padding enabled
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False   # don't truncate, pad instead
                )
                progress(.1, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # Pad to same length using Compel helper
                [pos_prompt_embeds, neg_prompt_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_prompt_embeds, neg_prompt_embeds])
                # progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # # NO REAL EXAMPLE FOUND, but doing it anyway, then check for errors, and image output
                # [pos_pooled_embeds, neg_pooled_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_pooled_embeds, neg_pooled_embeds])
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts Pad Length"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    # PROMPTS - end -------------------------------------------------------------------
       

    # Define the callback function to update the progress bar
    # however we notice difference between StableDiffusionInstructPix2PixPipeline and
    # StableDiffusionXLInstructPix2PixPipeline. 
    # the 'StableDiffusionInstructPix2PixPipeline' uses: callback_on_step_end
    # the 'StableDiffusionXLInstructPix2PixPipeline' uses: callback
    # so we have to use the 'DIffusionPipeline' style callback
    # which also mean we got to check it here before running the inference as to 
    # if the pipeline is an : SD or SDXL type, and use the correct callback type for pipeline.
    # gonna try defining both and use if/else to decide what to add to our inference_args{}
    # SD=callback_on_step_end, SDXL=callback

# ------------------------------------------------------

    def callback_on_each_step_end(step_index: int, timestep: int, latents: torch.Tensor):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")

# ------------------------------------------------------

    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs

# ------------------------------------------------------

    # we do the image resizing outside the loop to save time for each inference

    # resize input image to 512x512
    if resizeimage:
        new_width = 512
        new_height = 512
        progress(0, desc=f"Resizing Input Image...")
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        new_width = 512
        new_height = 512
        resized_img = image

    # get final image size for txt description file
    try:
        width, height = resized_img.size
    except Exception as e:
        if int(STUDIO["app_debug"]["value"]) > 0: print(f"An error occurred: {e}")
        width = new_width
        height = new_height


    # we resize input image to 512x512
    inference_args["image"] = resized_img

    # Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
    # `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
    # linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
    # value of at least `1`.
    inference_args["image_guidance_scale"] = image_guidance_scale

    inference_args["guidance_scale"] = guidance_scale
    inference_args["num_inference_steps"] = num_inference_steps

    # get model class name
    model_class_name = SDPIPELINE["pipeline_class"]
    # may be moving to regular 'callback_on_step_end'
    # callback for StableDiffusionXLInstructPix2PixPipeline
    if model_class_name == "StableDiffusionXLInstructPix2PixPipeline":
        inference_args["callback"] = callback_on_each_step_end
    # callback_on_step_end for StableDiffusionInstructPix2PixPipeline
    if model_class_name == "StableDiffusionInstructPix2PixPipeline":
        inference_args["callback_on_step_end"] = callback_on_step_end

    # clip_skip can only be use on SD15, not SDXL. 
    if SDPIPELINE["pipeline_model_type"]=="SD15":
        # Number of layers to be skipped from CLIP while computing the prompt embeddings. 
        # A value of 1 means that
        # the output of the pre-final layer will be used for computing the prompt embeddings.
        if clip_skip > 0:
            inference_args["clip_skip"] = clip_skip



    # input seed to local seed variable that we manipulate after each generation
    myseed=rseed
    # LOOP for multiple image generation
    for i in range(0, numimgs):
        imgnumb = i+1
        # Decide how to handle the seed.
        # two checkboxes, 'incrementseed' and 'usesameseed'
        # if the 'incrementseed' is checked, no randomization
        # and seed is incremented by 'x' amount 'after' first image
        # therefore uses sent seed as starting seed.
        # if the 'incrementseed' is UNchecked, USES randomization
        # if the 'usesameseed' is also checked, uses sent seed 
        # as starting seed. elsewise it starts on a random seed
        # and sent seed is not used
        usesameseed=False
        if incrementseed:
            if imgnumb > 1:
                myseed = myseed + incseedamount
        else:
            if not usesameseed:
                myseed=gen_random_seed()    # change to  random start seed rnd_start_seed check
            else:
                if imgnumb > 1:
                    myseed=gen_random_seed()
    
        # set the seed for inference  
        # we use 'diffusers.training_utils.set_seed' instead of 'torch generator'
        # may switch to 'torch generator' later -or- provide 'setting' to switch
        set_seed(myseed)
        
        if len(str(STUDIO["output_image_datetime"]["value"])) > 0:
            # Get the current date and time
            now = datetime.now()
            # Get the current local time as a struct_time object
            timestamp_str = now.strftime(str(STUDIO["output_image_datetime"]["value"]))
            # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
            formatted_time = timestamp_str
        else:
            formatted_time = ""
             
        # go ahead and set the image and txt filename now, so we can display it to user while running inference
        imagebasename = STUDIO["output_image_prefix"]["value"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + STUDIO["output_image_suffix"]["value"] 
        imagefilename = imagebasename + ".png"
        textfilename = imagebasename + ".txt"
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

        # we init the progress bar, rknote needs to be below check model loaded...
        progress(0, desc=f"Starting Inference. Step 1 of {num_inference_steps} - Image# {imgnumb} of {numimgs}")

        # mark start time
        pstart = time.time()

        # check if using FreeU or not
        if freeu: 
            pipeline.enable_freeu(s1=float(freeu_s1), s2=float(freeu_s2), b1=float(freeu_b1), b2=float(freeu_b2))
        else:
            pipeline.disable_freeu()



        with torch.no_grad():
            # run inference
            image2 = pipeline(**inference_args).images[0]


        # save the image generated
        image2.save(os.path.join(LLSTUDIO["output_image_dir"], imagefilename), "png")
        
        
        image2 = None
        del image2
        gc.collect()

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["output_image_dir"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["output_image_dir"], imagefilename)
        
        # create text for image generation parameters image'.txt' file
        text_output = prompt + "\n\n"
        if negative_prompt:
            text_output = text_output + negative_prompt + "\n\n"
        text_output = text_output + "Steps: " + str(num_inference_steps) + ", "
        text_output = text_output + "CFG scale: " + str(guidance_scale) + ", "
        text_output = text_output + "Seed: " + str(myseed) + ", "
        text_output = text_output + "Size: " + str(width) + "x"  + str(height)+ "\n"
        text_output = text_output + "Pipeline: " + str(SDPIPELINE['pipeline_class']) + "\n"
        text_output = text_output + "Model Loaded From: " + str(SDPIPELINE['pipeline_source']) + "\n"
        text_output = text_output + "Model Type: " + str(SDPIPELINE['pipeline_model_type']) + "\n"
        text_output = text_output + "Model: " + str(SDPIPELINE['pipeline_model_name']) + "\n"
        if SDPIPELINE["pipeline_text_encoder"] > 0:
            text_output = text_output + "Used Text Encoder from: " + SDPIPELINE["pipeline_text_encoder_name"] + "\n"
            text_output = text_output + "ClipSkip Value: " + str(clip_skip) + "\n"
        text_output = text_output + get_loaded_lora_models_text()
        text_output = text_output + "Image Filename: " + imagefilename + "\n"
        text_output = text_output + "Inference Time: " + format_seconds_strftime(pelapsed) + "\n"
        text_output = text_output + "Generation Method: " + SDPIPELINE["pipeline_gen_mode"] + "\n"
        text_output = text_output + "Prompt Type: " + prompt_type + "\n"
        if freeu: 
            text_output = text_output + "FreeU Enabled:\n"
            text_output = text_output + "FreeU Values: s1=" + freeu_s1 + ", s2=" + freeu_s2 + ", b1=" + freeu_b1 + ", b2=" + freeu_b2 + "\n"


        # write image generation parameters image'.txt' file
        file1 = open(LLSTUDIO['last_prompt_filename'], 'w')
        file1.write(text_output)
        file1.close()
        
        # write image generation parameters to 'last_prompt.txt' file
        file1 = open(os.path.join(".", "last_prompt.txt"), 'w')
        file1.write(text_output)
        file1.close()
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Generating Image# " + str(imgnumb) + " of " + str(numimgs))
        
        
        # # let's try and reduce the size of the font on the output 'label '
        a1 = "Finished Saving: " + str(imagefilename) + "<br>"
        a1 = a1 + "Image " + str(imgnumb) + " of " + str(numimgs)

        # # yield the data to both gradio outputs [progress/text,img]
        yield gr.update(value=a1), gr.update(value=LLSTUDIO['last_image_filename'])
        
        # update header/title - memory stats usage - plus apply rkmemopt
        update_grapptitle_mem()
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# 2X Upscale - Image Generation
def upscale_image(
    prompt, 
    negative_prompt, 
    guidance_scale, 
    num_inference_steps, 
    rseed, 
    inputimage, resizeimage, 
    freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2, 
    progress=gr.Progress()
    ):
    
    
    global pipeline             

    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)

    pipeline_args = {}

    # use safety checker or not
    # don't need 'feature_extractor' if no safety checker, saves more memory
    if not STUDIO["use_safety_checker"]["value"]: 
        pipeline_args["safety_checker"] = None
        pipeline_args["requires_safety_checker"] = False
        pipeline_args["feature_extractor"] = None
    
    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True
    pipeline_args["device"] = LLSTUDIO["device"]

    gr.Info("Loading SD Upscale 2X Model...", duration=3.0, title="Upscale Model")
    if (int(SDPIPELINE['pipeline_loaded']) > 0 and SDPIPELINE['pipeline_class'] == "StableDiffusionLatentUpscalePipeline" and SDPIPELINE['pipeline_model_name'] == STUDIO["sdupscale2x_model_name"]["value"] and SDPIPELINE['pipeline_gen_mode'] == "2x UpScaler"):
        SDPIPELINE['pipeline_class'] = "StableDiffusionLatentUpscalePipeline"
        SDPIPELINE['pipeline_loaded'] = 1
        SDPIPELINE['pipeline_model_name'] = STUDIO["sdupscale2x_model_name"]["value"]
        SDPIPELINE['pipeline_source'] = "HUB"
        SDPIPELINE['pipeline_gen_mode'] = "2x UpScaler"
        gr.Info("2x Upscale Model Already Loaded.", duration=3.0, title="2x Upscale Model")
    else:
        try:
            pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(STUDIO["sdupscale2x_model_name"]["value"], **pipeline_args)
            SDPIPELINE['pipeline_class'] = "StableDiffusionLatentUpscalePipeline"
            SDPIPELINE['pipeline_loaded'] = 1
            SDPIPELINE['pipeline_model_name'] = STUDIO["sdupscale2x_model_name"]["value"]
            SDPIPELINE['pipeline_source'] = "HUB"
            SDPIPELINE['pipeline_gen_mode'] = "2x UpScaler"
            gr.Info("Finished Loading SD Upscale 2X Model.", duration=3.0, title="Upscale Model")
        except Exception as e:
            tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_gen_mode'] + " Model." + f"<br>{e}" + "</h3>"
            yield gr.update(value=tempout)
            gr.Info("<h3>Error Loading: " + SDPIPELINE['pipeline_gen_mode'] + " Model."  + f"<br>{e}" + "</h3>", duration=3.0, title="2x Upscale Model")
            return tempout

    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # redundant... but... ok...
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] != "StableDiffusionLatentUpscalePipeline":
        gr.Info("Incorrect Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
        

    # reset Halt generation flag
    LLSTUDIO["halt_gen"] = 0
    # clear last image and last prompt, need to add last_negative_prompt
    LLSTUDIO['last_image_filename'] = ""
    LLSTUDIO['last_prompt_filename'] = ""
    # enables/disables hidden image to visible image 
    # onchange copy from oimage to oimage2
    # 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 1
    
    # # start setting up the inference arguments/parameters

    # setup our inference arguments dictionary
    inference_args = {}

    # PROMPTS - start -----------------------------------------------------------------
    # STUDIO["use_prompt_embeds"]["value"]
    # Use Normal Prompts, Prompt Embeddings or Prompt Weighting (using Compel).
    # ---------------------------------------------------------------------------------
    # 0=Normal Prompts (76 Max Prompt Tokens)                           SD
    # CAN NOT HAVE EMBEDDED PROMPTS IN StableDiffusionLatentUpscalePipeline
    #
    
    # we update the 'progress bar' to 0% at the beginning 
    # of each (use_prompt_embeds/pipeline_model_type) section
    # then we update to 100% when finished
    progress(0.0, desc=f"Creating Prompts...")

    # DONE
    # No prompt embeds, No prompt weighting, just the plain prompts
    # check the prompt length for SD15, which can not be longer than 76 tokens
    # this is needed for plain prompts for SD15
    # Diffusers library seems to indicate SDXL is different anyway
    # so we do not check SDXL prompt length
    progress(0, desc=f"Creating Prompts...")
    plen = get_prompt_length(prompt)
    if plen > 76:
        gr.Info("Canceled Operation.</br>Prompt Too Long." + "</br>Prompt Length = " + str(plen), duration=5.0, title="Prompt Length > 76")    
        return
    progress(.40, desc=f"Creating Prompts...")
    plen = get_prompt_length(negative_prompt)
    if plen > 76:
        gr.Info("Canceled Operation.</br>Negative Prompt Too Long." + "</br>Negative Prompt Length = " + str(plen), duration=5.0, title="Negative Prompt Length > 76")    
        return
    progress(.95, desc=f"Creating Prompts...")
    # norm prompts
    inference_args["prompt"] = prompt
    # Ignored when not using guidance (`guidance_scale < 1`)
    inference_args["negative_prompt"] = negative_prompt
    prompt_type = "Normal Prompts (SD15)"
    progress(1.0, desc=f"Finished Creating Prompts.")



    # PROMPTS - end -------------------------------------------------------------------

        
    #
    #
    # CALLBACK GOES HERE...
    # Define the callback function to update the progress bar
    def callback_on_each_step_end(step_num: int, timestep: int, latents: torch.Tensor):
        progress_value = (step_num + 1) / num_inference_steps
        if step_num + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_num + 1}/{num_inference_steps}")
    #
    #
    
    # input seed to local seed variable that we manipulate after each generation
    myseed=rseed
    # LOOP for multiple image generation
    numimgs=1
    imgnumb=1

    # set the seed for inference  
    # we use 'diffusers.training_utils.set_seed' instead of 'torch generator'
    # may switch to 'torch generator' later -or- provide 'setting' to switch
    set_seed(myseed)
    
    if len(str(STUDIO["output_image_datetime"]["value"])) > 0:
        # Get the current date and time
        now = datetime.now()
        # Get the current local time as a struct_time object
        timestamp_str = now.strftime(str(STUDIO["output_image_datetime"]["value"]))
        # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
        formatted_time = timestamp_str
    else:
        formatted_time = ""
         
    # go ahead and set the image and txt filename now, so we can display it to user while running inference
    imagebasename = STUDIO["output_image_prefix"]["value"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + STUDIO["output_image_suffix"]["value"] 
    imagefilename = imagebasename + ".png"
    textfilename = imagebasename + ".txt"
    
    if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

    # we init the progress bar, rknote needs to be below check model loaded...
    progress(0, desc=f"Starting Inference. Step 1 of {num_inference_steps} - Image# {imgnumb} of {numimgs}")

    # mark start time
    pstart = time.time()
    
    # check if using FreeU or not
    if freeu: 
        pipeline.enable_freeu(s1=float(freeu_s1), s2=float(freeu_s2), b1=float(freeu_b1), b2=float(freeu_b2))
    else:
        pipeline.disable_freeu()
    
    # resize input image to 512x512
    if resizeimage:
        new_width = 512
        new_height = 512
        resized_img = inputimage.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_img = inputimage

    # we resized? input image to 512x512
    inference_args["image"] = resized_img

    inference_args["guidance_scale"] = guidance_scale
    inference_args["num_inference_steps"] = num_inference_steps
    inference_args["callback"] = callback_on_each_step_end


    with torch.no_grad():
        # run inference
        image2 = pipeline(**inference_args).images[0]


    # save the image generated
    image2.save(os.path.join(LLSTUDIO["output_image_dir"], imagefilename), "png")
    
    
    image2 = None
    del image2
    gc.collect()

    # mark end time
    pend = time.time()
    pelapsed = pend - pstart

    if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
    
    # ONCE an image HAS BEEN generated, we set image and text output filenames
    # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
    # 'UNTIL' replaced with next generated image when more than a single image 
    # is being generated in a batch.
    LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["output_image_dir"], textfilename)
    LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["output_image_dir"], imagefilename)
    
    # get final upscaled image size for txt description file
    try:
        with Image.open(LLSTUDIO['last_image_filename']) as img:
            width, height = img.size
    except FileNotFoundError:
        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Image file not found at {image_path}")
        width = new_width
        height = new_height
    except Exception as e:
        if int(STUDIO["app_debug"]["value"]) > 0: print(f"An error occurred: {e}")
        width = new_width
        height = new_height
    del img
    
    # create text for image generation parameters image'.txt' file
    text_output = prompt + "\n\n"
    if negative_prompt:
        text_output = text_output + negative_prompt + "\n\n"
    text_output = text_output + "Steps: " + str(num_inference_steps) + ", "
    text_output = text_output + "CFG scale: " + str(guidance_scale) + ", "
    text_output = text_output + "Seed: " + str(myseed) + ", "
    text_output = text_output + "Size: " + str(width) + "x"  + str(height)+ "\n"
    text_output = text_output + "Pipeline: " + str(SDPIPELINE['pipeline_class']) + "\n"
    text_output = text_output + "Model Loaded From: " + str(SDPIPELINE['pipeline_source']) + "\n"
    text_output = text_output + "Model Type: " + str(SDPIPELINE['pipeline_model_type']) + "\n"
    text_output = text_output + "Model: " + str(SDPIPELINE['pipeline_model_name']) + "\n"
    if SDPIPELINE["pipeline_text_encoder"] > 0:
        text_output = text_output + "Used Text Encoder from: " + SDPIPELINE["pipeline_text_encoder_name"] + "\n"
    text_output = text_output + get_loaded_lora_models_text()
    text_output = text_output + "Image Filename: " + imagefilename + "\n"
    text_output = text_output + "Inference Time: " + format_seconds_strftime(pelapsed) + "\n"
    text_output = text_output + "Generation Method: " + SDPIPELINE["pipeline_gen_mode"] + "\n"
    text_output = text_output + "Prompt Type: " + prompt_type + "\n"
    if freeu: 
        text_output = text_output + "FreeU Enabled:\n"
        text_output = text_output + "FreeU Values: s1=" + freeu_s1 + ", s2=" + freeu_s2 + ", b1=" + freeu_b1 + ", b2=" + freeu_b2 + "\n"


    # write image generation parameters image'.txt' file
    file1 = open(LLSTUDIO['last_prompt_filename'], 'w')
    file1.write(text_output)
    file1.close()
    
    # write image generation parameters to 'last_prompt.txt' file
    file1 = open(os.path.join(".", "last_prompt.txt"), 'w')
    file1.write(text_output)
    file1.close()
    
    if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Generating Image# " + str(imgnumb) + " of " + str(numimgs))
    
    
    # # let's try and reduce the size of the font on the output 'label '
    a1 = "Finished Saving: " + str(imagefilename) + "<br>"
    a1 = a1 + "Image " + str(imgnumb) + " of " + str(numimgs)

    # # yield the data to both gradio outputs [progress/text,img]
    yield gr.update(value=a1), gr.update(value=LLSTUDIO['last_image_filename'])
    
    # update header/title - memory stats usage - plus apply rkmemopt
    update_grapptitle_mem()
        
    # check if user has halted after image generation current inference finished
    if LLSTUDIO["halt_gen"] == 1:
        gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
        # # return the data to both gradio outputs [progress/text,img], because we halted
        return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# ControlNet - Image Generation
def cnetgen_LCM_images(
    prompt, 
    negative_prompt, 
    width, height, 
    guidance_scale, cnetgen_guidance_start, cnetgen_guidance_end, cnetgen_conditioningguidance, cnetgen_conditioningguidance2, 
    num_inference_steps, 
    rseed, 
    numimgs, 
    incrementseed, 
    incseedamount, 
    cnetimage, 
    cnetresizeimage, 
    cnetimage2, 
    cnetresizeimage2, 
    freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2, 
    clip_skip,
    use_guess_mode,
    progress=gr.Progress()
    ):
    
    
    
    global pipeline             # where the model is loaded to
    
    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)


    
    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] == "StableDiffusionLatentUpscalePipeline":
        gr.Info("UpScaler2X Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
    
       

    # SD ControlNet can be used on SD15, not SDXL. 
    if SDPIPELINE["pipeline_model_type"]=="SDXL":
        gr.Info("SD Controlnet Pipeline can not use SDXL models.</br>Uses SD Only. Please load an SD model.", duration=5.0, title="SD Controlnet Pipeline")    
        return

    # check if any Controlnet Models are loaded
    if int(SDPIPELINE['pipeline_controlnet_loaded']) < 1:
        gr.Info("No SD Controlnet Model Loaded.</br>Please load an SD Controlnet Model.", duration=5.0, title="SD Controlnet Pipeline")    
        return

    # check if two Controlnet Models are loaded, then check that second image is loaded too
    if int(SDPIPELINE['pipeline_controlnet_loaded']) > 1:  
        if cnetimage2 is None:
            gr.Info("Two ControlNets are loaded. You need two images loaded to run inference.</br>Please load an image into each image input.", duration=5.0, title="SD Controlnet Pipeline")    
            return
        
    # check if first image is loaded
    if cnetimage is None:   
        gr.Info("Need at least one image loaded to run inference.</br>Please load an input image.", duration=5.0, title="SD Controlnet Pipeline")    
        return
    
 
        
    # reset Halt generation flag
    LLSTUDIO["halt_gen"] = 0
    # clear last image and last prompt, need to add last_negative_prompt
    LLSTUDIO['last_image_filename'] = ""
    LLSTUDIO['last_prompt_filename'] = ""
    # enables/disables hidden image to visible image 
    # onchange copy from oimage to oimage2
    # 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 1
    
    # # start setting up the inference arguments/parameters

    # setup our inference arguments dictionary
    inference_args = {}


    # PROMPTS - start -----------------------------------------------------------------
    # STUDIO["use_prompt_embeds"]["value"]
    # Use Normal Prompts, Prompt Embeddings or Prompt Weighting (using Compel).
    # ---------------------------------------------------------------------------------
    # 0=Normal Prompts (76 Max Prompt Tokens)                           BOTH SD/SDXL
    # 1=Prompt Embeddings and Padding                                   BOTH SD/SDXL    
    # 2=Prompt Weighting (Compel) and Prompt Embeddings                 BOTH SD/SDXL    
    # 3=Prompt Weighting (Compel) and Prompt Embeddings and Padding     BOTH SD/SDXL   
    #
    # use_prompt_embeds (always pad)
    # if int(STUDIO["use_prompt_embeds"]["value"]) == 1:
    #
    #
    # ---------------------------------------------------------------------------------
    # # # SDXL-PROMPTS
    # # # PADDING + POOLED + EMBEDS
    # prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
    # negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
    # ---------------------------------------------------------------------------------
    
    # we update the 'progress bar' to 0% at the beginning 
    # of each (use_prompt_embeds/pipeline_model_type) section
    # then we update to 100% when finished
    progress(0.0, desc=f"Creating Prompt Embeds...")

    if int(STUDIO["use_prompt_embeds"]["value"]) == 0:
        # DONE
        # No prompt embeds, No prompt weighting, just the plain prompts
        # check the prompt length for SD15, which can not be longer than 76 tokens
        # this is needed for plain prompts for SD15
        # Diffusers library seems to indicate SDXL is different anyway
        # so we do not check SDXL prompt length
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompts...")
            plen = get_prompt_length(prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Prompt Too Long." + "</br>Prompt Length = " + str(plen), duration=5.0, title="Prompt Length > 76")    
                return
            progress(.40, desc=f"Creating Prompts...")
            plen = get_prompt_length(negative_prompt)
            if plen > 76:
                gr.Info("Canceled Operation.</br>Negative Prompt Too Long." + "</br>Negative Prompt Length = " + str(plen), duration=5.0, title="Negative Prompt Length > 76")    
                return
            progress(.95, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SD15)"
            progress(1.0, desc=f"Finished Creating Prompts.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompts...")
            # norm prompts
            inference_args["prompt"] = prompt
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt"] = negative_prompt
            prompt_type = "Normal Prompts (SDXL)"
            progress(1.0, desc=f"Finished Creating Prompts.")

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 1:
        if SDPIPELINE["pipeline_model_type"]=="SD15":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SD15 embedded prompts - diffusers
            prompt_embeds, negative_prompt_embeds = do_prompt_embeds(LLSTUDIO["device"], pipeline, prompt, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # embedded prompts
            inference_args["prompt_embeds"] = prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
        if SDPIPELINE["pipeline_model_type"]=="SDXL":
            progress(0, desc=f"Creating Prompt Embeds...")
            # generates SDXL embedded prompts - diffusers
            # # SDXL-PROMPTS
            # # PADDING + POOLED + EMBEDS
            prompt_embeds, pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, prompt)
            progress(.45, desc=f"Creating Prompt Embeds...")
            negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_pooled_embeddings(LLSTUDIO["device"], pipeline, negative_prompt)
            progress(.95, desc=f"Creating Prompt Embeds...")
            # set inference arguments
            inference_args["prompt_embeds"] = prompt_embeds
            inference_args["pooled_prompt_embeds"] = pooled_prompt_embeds
            # Ignored when not using guidance (`guidance_scale < 1`)
            inference_args["negative_prompt_embeds"] = negative_prompt_embeds
            inference_args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            prompt_type = "Embedded Prompts"
            # we update the 'Creating Prompt Embeds' progress bar
            progress(1.0, desc=f"Finished Creating Prompt Embeds.")
            
    elif int(STUDIO["use_prompt_embeds"]["value"]) == 2:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel embedded prompts
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # embedded prompts
                inference_args["prompt_embeds"] = compel_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = compel_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel pooled + embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True]
                )
                progress(.05, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings from the first text encoder
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.45, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    elif int(STUDIO["use_prompt_embeds"]["value"]) == 3:
        if LLSTUDIO["compel_installed"] == 1:
            if SDPIPELINE["pipeline_model_type"]=="SD15":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SD15 Compel padded embeds
                compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                prompt_embeds = compel_proc.build_conditioning_tensor(prompt)
                progress(.5, desc=f"Creating Weighted Prompt Embeds...")
                negative_prompt_embeds = compel_proc.build_conditioning_tensor(negative_prompt)
                progress(.75, desc=f"Creating Weighted Prompt Embeds...")
                [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
                # embedded prompts
                inference_args["prompt_embeds"] = prompt_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = negative_prompt_embeds
                prompt_type = "Compel Embedded Prompts Pad Same Length"
                progress(1, desc=f"Finished Creating Weighted Prompt Embeds.")
            if SDPIPELINE["pipeline_model_type"]=="SDXL":
                progress(0, desc=f"Creating Weighted Prompt Embeds...")
                # generates SDXL Compel padded + pooled embeds
                # Crucial for SDXL: return pooled for the second encoder
                # Initialize Compel with padding enabled
                compel_sdxl_proc = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False   # don't truncate, pad instead
                )
                progress(.1, desc=f"Creating Weighted Prompt Embeds...")
                # The compel object now returns *two* sets of embeddings, one for each text encoder.
                # conditioning = regular embeddings
                # pooled = pooled embeddings from the second text encoder
                # Get embeddings for both positive and negative prompts
                pos_prompt_embeds, pos_pooled_embeds = compel_sdxl_proc(prompt)
                progress(.25, desc=f"Creating Weighted Prompt Embeds...")
                neg_prompt_embeds, neg_pooled_embeds = compel_sdxl_proc(negative_prompt)
                progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # Pad to same length using Compel helper
                [pos_prompt_embeds, neg_prompt_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_prompt_embeds, neg_prompt_embeds])
                # progress(.65, desc=f"Creating Weighted Prompt Embeds...")
                # # # # NO REAL EXAMPLE FOUND, but doing it anyway, then check for errors, and image output
                # [pos_pooled_embeds, neg_pooled_embeds] = compel_sdxl_proc.pad_conditioning_tensors_to_same_length([pos_pooled_embeds, neg_pooled_embeds])
                progress(.95, desc=f"Creating Weighted Prompt Embeds...")
                # set inference arguments
                inference_args["prompt_embeds"] = pos_prompt_embeds
                inference_args["pooled_prompt_embeds"] = pos_pooled_embeds
                # Ignored when not using guidance (`guidance_scale < 1`)
                inference_args["negative_prompt_embeds"] = neg_prompt_embeds
                inference_args["negative_pooled_prompt_embeds"] = neg_pooled_embeds
                prompt_type = "Compel Embedded Prompts Pad Length"
                progress(1.0, desc=f"Finished Creating Weighted Prompt Embeds.")
        else:
            gr.Info("Please Install 'Compel'.</br>Needed for 'Prompt Weighting' to function", duration=5.0, title="Compel Not Installed")    
            return

    # PROMPTS - end -------------------------------------------------------------------
        
   
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs
    

    # resize input cnetimage2 to 512x512
    if cnetresizeimage2:
        new_width = 512
        new_height = 512
        resized_img2 = cnetimage2.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_img2 = cnetimage2
    
    # resize input cnetimage to 512x512
    if cnetresizeimage:
        new_width = 512
        new_height = 512
        resized_img = cnetimage.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_img = cnetimage


    # Common inference parameters for Image to Image, minus the 'strength' parameter
    inference_args["width"] = width
    inference_args["height"] = height

    # Guidance scale is enabled when `guidance_scale > 1
    inference_args["guidance_scale"] = guidance_scale
    inference_args["num_inference_steps"] = num_inference_steps
    inference_args["callback_on_step_end"] = callback_on_step_end

    # clip_skip can only be use on SD15, not SDXL. 
    if SDPIPELINE["pipeline_model_type"]=="SD15":
        # Number of layers to be skipped from CLIP while computing the prompt embeddings. 
        # A value of 1 means that
        # the output of the pre-final layer will be used for computing the prompt embeddings.
        if clip_skip > 0:
            inference_args["clip_skip"] = clip_skip

    # ControlNet inference parameters
    # 0.0, float, The percentage of total steps at which the ControlNet starts applying. (0-100%)
    inference_args["control_guidance_start"] = cnetgen_guidance_start         
    # 1.0, float, The percentage of total steps at which the ControlNet stops applying. (0-100%)
    inference_args["control_guidance_end"] = cnetgen_guidance_end            

    # The ControlNet encoder tries to recognize the content 
    # of the input image even if you remove all prompts. 
    # A guidance_scale value between 3.0 and 5.0 is recommended.
    if use_guess_mode:
        inference_args["guess_mode"] = True 
    else:
        inference_args["guess_mode"] = False
        
    # apply conditioning guidance for each controlnet
    # for idk, MUST create list by assingment to variable
    # then use that to assign the inference_args 'controlnet_conditioning_scale'
    if int(SDPIPELINE['pipeline_controlnet_loaded']) > 1:
        controlnet_conditioning_scale_list = [cnetgen_conditioningguidance, cnetgen_conditioningguidance2]
        inference_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale_list
    else:
        controlnet_conditioning_scale_list = [cnetgen_conditioningguidance]
        inference_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale_list

    # single image or image list?
    if int(SDPIPELINE['pipeline_controlnet_loaded']) > 1:
        image_list = [resized_img, resized_img2]
        inference_args["image"] = image_list
    else:
        image_list = [resized_img]
        inference_args["image"] = image_list

    
    
    # input seed to local seed variable that we manipulate after each generation
    myseed=rseed
    # LOOP for multiple image generation
    for i in range(0, numimgs):
        imgnumb = i+1
        # Decide how to handle the seed.
        # two checkboxes, 'incrementseed' and 'usesameseed'
        # if the 'incrementseed' is checked, no randomization
        # and seed is incremented by 'x' amount 'after' first image
        # therefore uses sent seed as starting seed.
        # if the 'incrementseed' is UNchecked, USES randomization
        # if the 'usesameseed' is also checked, uses sent seed 
        # as starting seed. elsewise it starts on a random seed
        # and sent seed is not used
        usesameseed=False
        
        if incrementseed:
            if imgnumb > 1:
                myseed = myseed + incseedamount
        else:
            if not usesameseed:
                myseed=gen_random_seed()    # change to  random start seed rnd_start_seed check
            else:
                if imgnumb > 1:
                    myseed=gen_random_seed()
    
        # set the seed for inference  
        # we use 'diffusers.training_utils.set_seed' instead of 'torch generator'
        # may switch to 'torch generator' later -or- provide 'setting' to switch
        set_seed(myseed)
        
        if len(str(STUDIO["output_image_datetime"]["value"])) > 0:
            # Get the current date and time
            now = datetime.now()
            # Get the current local time as a struct_time object
            timestamp_str = now.strftime(str(STUDIO["output_image_datetime"]["value"]))
            # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
            formatted_time = timestamp_str
        else:
            formatted_time = ""
             
        # go ahead and set the image and txt filename now, so we can display it to user while running inference
        imagebasename = STUDIO["output_image_prefix"]["value"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + STUDIO["output_image_suffix"]["value"] 
        imagefilename = imagebasename + ".png"
        textfilename = imagebasename + ".txt"
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

        # we init the progress bar, rknote needs to be below check model loaded...
        progress(0, desc=f"Starting Inference. Step 1 of {num_inference_steps} - Image# {imgnumb} of {numimgs}")

        # mark start time
        pstart = time.time()
        

        # Run inference
        pstart = time.time()

        if int(STUDIO["app_debug"]["value"]) > 0: print("Generating Image Filename: " + imagefilename)

        # check if using FreeU or not
        if freeu: 
            pipeline.enable_freeu(s1=float(freeu_s1), s2=float(freeu_s2), b1=float(freeu_b1), b2=float(freeu_b2))
        else:
            pipeline.disable_freeu()
            
            
        with torch.no_grad():
            # run inference
            image2 = pipeline(**inference_args).images[0]


        # save the image generated
        image2.save(os.path.join(LLSTUDIO["output_image_dir"], imagefilename), "png")
        
        
        image2 = None
        del image2
        gc.collect()
        
        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["output_image_dir"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["output_image_dir"], imagefilename)
        
        # create text for image generation parameters image'.txt' file
        text_output = prompt + "\n\n"
        if negative_prompt:
            text_output = text_output + negative_prompt + "\n\n"
        text_output = text_output + "Steps: " + str(num_inference_steps) + ", "
        text_output = text_output + "CFG scale: " + str(guidance_scale) + ", "
        text_output = text_output + "Seed: " + str(myseed) + ", "
        text_output = text_output + "Size: " + str(width) + "x"  + str(height)+ "\n"
        text_output = text_output + "Pipeline: " + str(SDPIPELINE['pipeline_class']) + "\n"
        text_output = text_output + "Model Loaded From: " + str(SDPIPELINE['pipeline_source']) + "\n"
        text_output = text_output + "Model Type: " + str(SDPIPELINE['pipeline_model_type']) + "\n"
        text_output = text_output + "Model: " + str(SDPIPELINE['pipeline_model_name']) + "\n"
        if SDPIPELINE["pipeline_text_encoder"] > 0:
            text_output = text_output + "Used Text Encoder from: " + SDPIPELINE["pipeline_text_encoder_name"] + "\n"
            text_output = text_output + "ClipSkip Value: " + str(clip_skip) + "\n"
        text_output = text_output + get_loaded_lora_models_text()
        text_output = text_output + "Image Filename: " + imagefilename + "\n"
        text_output = text_output + "Inference Time: " + format_seconds_strftime(pelapsed) + "\n"
        text_output = text_output + "Generation Method: " + SDPIPELINE["pipeline_gen_mode"] + "\n"
        text_output = text_output + "Prompt Type: " + prompt_type + "\n"
        if freeu: 
            text_output = text_output + "FreeU Enabled:\n"
            text_output = text_output + "FreeU Values: s1=" + freeu_s1 + ", s2=" + freeu_s2 + ", b1=" + freeu_b1 + ", b2=" + freeu_b2 + "\n"


        # write image generation parameters image'.txt' file
        file1 = open(LLSTUDIO['last_prompt_filename'], 'w')
        file1.write(text_output)
        file1.close()
        
        # write image generation parameters to 'last_prompt.txt' file
        file1 = open(os.path.join(".", "last_prompt.txt"), 'w')
        file1.write(text_output)
        file1.close()
        
        if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Generating Image# " + str(imgnumb) + " of " + str(numimgs))
        
        
        # # let's try and reduce the size of the font on the output 'label '
        a1 = "Finished Saving: " + str(imagefilename) + "<br>"
        a1 = a1 + "Image " + str(imgnumb) + " of " + str(numimgs)

        # # yield the data to both gradio outputs [progress/text,img]
        yield gr.update(value=a1), gr.update(value=LLSTUDIO['last_image_filename'])
        
        # update header/title - memory stats usage - plus apply rkmemopt
        update_grapptitle_mem()
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# sends last image generated to it's own model specific image gallery
def send_to_gallery():

    if (LLSTUDIO['last_image_filename'] == "" or LLSTUDIO['last_prompt_filename'] == ""):
        stdoutput = "Error: No Image/Prompt Found For Gallery"
        return stdoutput
        
    stdoutput = ""
     
    if SDPIPELINE["pipeline_class"]=="StableDiffusionLatentUpscalePipeline":
        directory_path = os.path.dirname(LLSTUDIO['last_image_filename'])
    
        full_path_image = LLSTUDIO['last_image_filename']
        image_directory_path = os.path.dirname(full_path_image)
        filename_with_extension = os.path.basename(full_path_image)
        filename_only = os.path.splitext(filename_with_extension)[0]
        extension_only = os.path.splitext(filename_with_extension)[1]
        image_name_out = filename_only + "_upx2." + extension_only
        full_image_name_out = os.path.join(directory_path,image_name_out)

        full_path_text = LLSTUDIO['last_prompt_filename']
        text_directory_path = os.path.dirname(full_path_text)
        filename_with_extension = os.path.basename(full_path_text)
        filename_only = os.path.splitext(filename_with_extension)[0]
        extension_only = os.path.splitext(filename_with_extension)[1]
        text_name_out = filename_only + "_upx2." + extension_only
        full_text_name_out = os.path.join(directory_path,text_name_out)
    
        # do last image
        try:
            shutil.copy2(LLSTUDIO['last_image_filename'], model_image_path_file)
            stdoutput = stdoutput + f"Copied image file '{LLSTUDIO['last_image_filename']}' to '{model_image_path_file}'</br>"
        except FileNotFoundError:
            stdoutput = stdoutput + f"Error: Source file '{LLSTUDIO['last_image_filename']}' not found, or a directory in the path for '{model_image_path_file}' does not exist."
        except PermissionError:
            stdoutput = stdoutput + f"Error: Permission denied to access '{LLSTUDIO['last_image_filename']}' or write to '{model_image_path_file}'."
        except shutil.SameFileError: # Use shutil.SameFileError for Python 3.4+
            stdoutput = stdoutput + "Error: Source and destination files are the same."
        except OSError as e: # Catch other potential OS errors
            stdoutput = stdoutput + f"An OS error occurred: {e}"
        except Exception as e:
            stdoutput = stdoutput + f"An unexpected error occurred: {e}"

        # do last prompt
        try:
            shutil.copy2(LLSTUDIO['last_prompt_filename'], model_image_path_file)
            stdoutput = stdoutput + f"Copied prompt file '{LLSTUDIO['last_prompt_filename']}' to '{model_image_path_file}'</br>"
        except FileNotFoundError:
            stdoutput = stdoutput + f"Error: Source file '{LLSTUDIO['last_prompt_filename']}' not found, or a directory in the path for '{model_image_path_file}' does not exist."
        except PermissionError:
            stdoutput = stdoutput + f"Error: Permission denied to access '{LLSTUDIO['last_prompt_filename']}' or write to '{model_image_path_file}'."
        except shutil.SameFileError: # Use shutil.SameFileError for Python 3.4+
            stdoutput = stdoutput + "Error: Source and destination files are the same."
        except OSError as e: # Catch other potential OS errors
            stdoutput = stdoutput + f"An OS error occurred: {e}"
        except Exception as e:
            stdoutput = stdoutput + f"An unexpected error occurred: {e}"
    
    else:
    
    
    
        # rkadded slash on trailing end because shutil.copy2 not like no slash ???
        model_image_path_file = get_lcm_model_image_path_file(SDPIPELINE['pipeline_model_name']) + os.sep
        stdoutput = ""

        if not os.path.exists(LLSTUDIO['last_image_filename']):
            stdoutput = stdoutput + f"Error: Last Image Not Found '{LLSTUDIO['last_image_filename']}'</br>"
            return stdoutput
     
        if not os.path.exists(LLSTUDIO['last_prompt_filename']):
            stdoutput = stdoutput + f"Error: Last Prompt Not Found '{LLSTUDIO['last_prompt_filename']}'</br>"
            return stdoutput
        
        if not os.path.exists(model_image_path_file):
            os.makedirs(model_image_path_file)
            stdoutput = stdoutput + f"Created model images directory '{model_image_path_file}'</br>"

        # do last image
        try:
            shutil.copy2(LLSTUDIO['last_image_filename'], model_image_path_file)
            stdoutput = stdoutput + f"Copied image file '{LLSTUDIO['last_image_filename']}' to '{model_image_path_file}'</br>"
        except FileNotFoundError:
            stdoutput = stdoutput + f"Error: Source file '{LLSTUDIO['last_image_filename']}' not found, or a directory in the path for '{model_image_path_file}' does not exist."
        except PermissionError:
            stdoutput = stdoutput + f"Error: Permission denied to access '{LLSTUDIO['last_image_filename']}' or write to '{model_image_path_file}'."
        except shutil.SameFileError: # Use shutil.SameFileError for Python 3.4+
            stdoutput = stdoutput + "Error: Source and destination files are the same."
        except OSError as e: # Catch other potential OS errors
            stdoutput = stdoutput + f"An OS error occurred: {e}"
        except Exception as e:
            stdoutput = stdoutput + f"An unexpected error occurred: {e}"

        # do last prompt
        try:
            shutil.copy2(LLSTUDIO['last_prompt_filename'], model_image_path_file)
            stdoutput = stdoutput + f"Copied prompt file '{LLSTUDIO['last_prompt_filename']}' to '{model_image_path_file}'</br>"
        except FileNotFoundError:
            stdoutput = stdoutput + f"Error: Source file '{LLSTUDIO['last_prompt_filename']}' not found, or a directory in the path for '{model_image_path_file}' does not exist."
        except PermissionError:
            stdoutput = stdoutput + f"Error: Permission denied to access '{LLSTUDIO['last_prompt_filename']}' or write to '{model_image_path_file}'."
        except shutil.SameFileError: # Use shutil.SameFileError for Python 3.4+
            stdoutput = stdoutput + "Error: Source and destination files are the same."
        except OSError as e: # Catch other potential OS errors
            stdoutput = stdoutput + f"An OS error occurred: {e}"
        except Exception as e:
            stdoutput = stdoutput + f"An unexpected error occurred: {e}"

    return stdoutput

    
# --------------------------------------------
def bytes_to_human_readable(num_bytes):
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


# ------------------------------------------------------
def get_all_memory_info():
    mem = psutil.virtual_memory()
    memory_info = {
        "Total": mem.total,
        "Available": mem.available,
        "Percent Used": mem.percent,
        "Used": mem.used,
        "Free": mem.free,
    }
    # Add platform-specific metrics if available
    if hasattr(mem, 'active'):
        memory_info["Active"] = mem.active
    if hasattr(mem, 'inactive'):
        memory_info["Inactive"] = mem.inactive
    if hasattr(mem, 'buffers'):
        memory_info["Buffers"] = mem.buffers
    if hasattr(mem, 'cached'):
        memory_info["Cached"] = mem.cached
    if hasattr(mem, 'shared'):
        memory_info["Shared"] = mem.shared
    if hasattr(mem, 'slab'):
        memory_info["Slab"] = mem.slab
    if hasattr(mem, 'wired'):
        memory_info["Wired"] = mem.wired
    return memory_info


# ------------------------------------------------------
def get_sysinfo_memory():
    myout = "<h3>Memory Report</h3>\n"
    all_memory_data = get_all_memory_info()
    for key, value in all_memory_data.items():
        if (key == "Percent Used"):
            myout = myout + f"{key}: {value} %<br>\n"
        else:
            myout = myout + f"{key}: {bytes_to_human_readable(value)} <br>\n"
    return myout

    
# ------------------------------------------------------
def get_sysinfo_hfcache():
    myout = "<h3>HuggingFace Hub Local Cache Location</h3>\n"
    if not os.path.isdir(LLSTUDIO["hub_model_dir"]):
        return myout + "Huggingface Hub Cache Directoy was NOT Found.<br>You will need to Check the enviroment variable 'HF_HUB_CACHE' -OR- set the location in the LCM-LoRA Studio 'settings' in order to load model via the dropdown box."
    hfcache = LLSTUDIO["hub_model_dir"]    
    myout = myout + f"{hfcache}<br>\n"
    myout = myout + "<h3>HuggingFace Hub Local Cache Model List</h3>\n"
    myout = myout + "<i>(Only StableDiffusionPipelines with SD/SDXL Model Classes.)</i><br>\n"
    entries = [d for d in os.listdir(LLSTUDIO["hub_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["hub_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        if tmp_text != ".locks":
            try:
                hex40str = get_file_content(os.path.join(get_hub_model_path_file(tmp_text), "refs", "main"))
                with open(os.path.join(get_hub_model_path_file(tmp_text), "snapshots", hex40str, "model_index.json"), "r") as f:
                    model_config_data = json.load(f)

                model_class_name = model_config_data["_class_name"]

                if model_class_name == "StableDiffusionPipeline":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionXLPipeline":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionImage2Image":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionXLImage2Image":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionInpaintPipeline":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionXLInpaintPipeline":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionInstructPix2PixPipeline":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
                elif model_class_name == "StableDiffusionXLInstructPix2PixPipeline":
                    myout = myout + f"{model_class_name} - {tmp_text}<br>\n"
               
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(tmp_text + ": MODEL CONFIG NOT FOUND")

    return myout


# ------------------------------------------------------
def get_sysinfo_env():
    myout = "<h3>All Environment Variables</h3>\n"
    for name, value in os.environ.items():
        myout = myout + f"{name}: {value} <br>\n"
    return myout


# ------------------------------------------------------
def get_sysinfo_sysinfo():
    myout = "<h3>System Information</h3>\n"
    try:
        myout = myout + str(f"Running on System: {platform.system()}<br>\n")
        myout = myout + str(f"Release: {platform.release()}<br>\n")
        myout = myout + str(f"Operating System: {platform.platform()}<br>\n")
        myout = myout + str(f"Version: {platform.version()}<br>\n")
        myout = myout + str(f"Processor: {platform.processor()}<br>\n")
        myout = myout + str(f"Machine: {platform.machine()} (aarch64 = ARM64)<br>\n")
        myout = myout + str(f"Hostname: {platform.node()}<br>\n")
        myout = myout + str(f"UName: {platform.uname()}<br>\n")
        myout = myout + str(f"Architecture: {platform.architecture()}<br>\n")
        myout = myout + str(f"Python Version: {platform.python_version()}<br>\n")
        myout = myout + str(f"Python Build: {platform.python_build()}<br>\n")
        myout = myout + str(f"Python Compiler: {platform.python_compiler()}<br>\n")
        myout = myout + str(f"Python Implementation: {platform.python_implementation()}<br>\n")
    except Exception as ex:
        myout = myout + str(f"Error: Getting System Info. {ex}<br>\n")

    return myout



# --------------------------------------------------------
def get_prompt_length_tokens(prompt):
    global pipeline             

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return gr.Textbox(label="Prompt - Tokens[0]")

    tokenizer = pipeline.tokenizer
    tokenized_output = tokenizer.encode(prompt)
    num_tokens = "Prompt - Tokens[" + str(len(tokenized_output)) + "]"
    
    tokenizer = None
    tokenized_output = None
    del tokenizer
    del tokenized_output
    gc.collect()

    return gr.Textbox(label=num_tokens)
    
    
# ------------------------------------------------------
def get_negprompt_length_tokens(prompt):
    global pipeline             

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return gr.Textbox(label="Negative Prompt - Tokens[0]")

    tokenizer = pipeline.tokenizer
    tokenized_output = tokenizer.encode(prompt)
    num_tokens = "Negative Prompt - Tokens[" + str(len(tokenized_output)) + "]"
    
    tokenizer = None
    tokenized_output = None
    del tokenizer
    del tokenized_output
    gc.collect()

    return gr.Textbox(label=num_tokens)


# ------------------------------------------------------
def get_prompt_length(prompt):
    global pipeline             
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return 0
    tokenizer = pipeline.tokenizer
    tokenized_output = tokenizer.encode(prompt)
    num_tokens = len(tokenized_output)
    
    tokenizer = None
    tokenized_output = None
    del tokenizer
    del tokenized_output
    gc.collect()

    return num_tokens



# ------------------------------------------------------
def rkmalloc_trim():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


# ------------------------------------------------------
def delete_pipeline():
    global pipeline             

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        pipeline = None
        if hasattr(pipeline, 'to') and callable(getattr(pipeline, 'to')):
            pipeline.to(LLSTUDIO["device"])
        del pipeline
        pipeline = None
        gc.collect()
        rkmalloc_trim()
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
        
    # is a model loaded?, if there is, kill any loRAs, then kill pipeline and do gc, 
    if hasattr(pipeline, 'to') and callable(getattr(pipeline, 'to')):
        pipeline.to(LLSTUDIO["device"])
        if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
            if int(STUDIO["app_debug"]["value"]) > 0: print ("Unloading LoRA Adapters...")
            tempout = "<h3>Unloading LoRA Weights...</h3>"
            yield gr.update(value=tempout)
            pipeline.unload_lora_weights() 
            adapter_names = pipeline.get_active_adapters()
            tempout = "<h3>Deleting LoRA Adapters...</h3>"
            yield gr.update(value=tempout)
            pipeline.delete_adapters(adapter_names)
            LLSTUDIO["loaded_lora_model_value"]=[]
            LLSTUDIO["loaded_lora_model_name"]=[]
            LLSTUDIO["loaded_lora_model_adapter"]=[]
            LLSTUDIO["lora_adapter_numb"] = 0
            tempout = "<h3>Unloaded Weights and Deleted LoRA Adapters.</h3>"
            yield gr.update(value=tempout)
            if int(STUDIO["app_debug"]["value"]) > 0: print ("Finished Unloading and Deleting LoRA Adapters.")
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print ("No LoRA Models Loaded to Unload.")
         
        del pipeline
        pipeline = None
        gc.collect()
        rkmalloc_trim()
        reset_pipeline_info()
        tempout = "<h3>Unloaded Pipeline, Ready to Load a Model.</h3>"
        yield gr.update(value=tempout)
        gr.Info("<h3>Unloaded Pipeline, Ready to Load a Model.</h3>", duration=3.0, title="Unloaded Model")
        
    return tempout


# ---------------------------------------------------------------
# called after saving a LCM-LoRA model and before loading a model
# should be called BEFORE setting model pipeline info
def slient_delete_pipeline():
    global pipeline             

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        pipeline = None
        if hasattr(pipeline, 'to') and callable(getattr(pipeline, 'to')):
            pipeline.to(LLSTUDIO["device"])
        del pipeline
        pipeline = None
        gc.collect()

        rkmalloc_trim()

        return 

    # is a model loaded?, if there is, kill any loRAs, then kill pipeline and do gc, 
    if hasattr(pipeline, 'to') and callable(getattr(pipeline, 'to')):
        pipeline.to(LLSTUDIO["device"])
        # unload rk style loras
        if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
            pipeline.unload_lora_weights() 
            adapter_names = pipeline.get_active_adapters()
            pipeline.delete_adapters(adapter_names)
            LLSTUDIO["loaded_lora_model_value"]=[]
            LLSTUDIO["loaded_lora_model_name"]=[]
            LLSTUDIO["loaded_lora_model_adapter"]=[]
            LLSTUDIO["lora_adapter_numb"] = 0
         
    del pipeline
    pipeline = None
    gc.collect()
    rkmalloc_trim()
    reset_pipeline_info()
    return 


# ---------------------------------------------------------------
def reset_pipeline_info():
    SDPIPELINE['pipeline_loaded'] = 0                           # model loaded ? 0=no/1=yes, trigger error/alert on No model loaded
    SDPIPELINE['pipeline_class'] = "StableDiffusionPipeline"    # StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImage2Image... default startup value=StableDiffusionPipeline
    SDPIPELINE['pipeline_source'] = ""                          # 'LCMLORA', 'HUB Cached', 'Huggingface', 'Safetensors' basically where model was loaded from, if LCMLORA, already has LCM LoRA added/fused
    SDPIPELINE['pipeline_model_name'] = ""                      # name of model as in dropdowns
    SDPIPELINE['pipeline_gen_mode'] = "Text to Image"           # Text 2 Image, Image 2 Image, Inpainting, Instruct Pix2Pix, UpScaler default startup value=Text 2 Image
    SDPIPELINE['pipeline_model_type'] = "SD15"                  # SD15 or SDXL default=SD15
    SDPIPELINE['pipeline_text_encoder'] = 0                     # use separate text encoder ? 0=no/1=yes
    SDPIPELINE['pipeline_text_encoder_name'] = ""               # name of model of separate text encoder as in dropdowns
    SDPIPELINE['pipeline_model_precision'] = "fp16"             # basically, fp16 or fp32 (default LCM to fp16 so it'll run it's 4 step lcm-lora)
    SDPIPELINE['pipeline_controlnet_loaded'] = 0               # load a controlnet ? 0=no/1=yes
    SDPIPELINE['pipeline_controlnet_name'] = ""                # name of control net
    SDPIPELINE['pipeline_controlnet_name2'] = ""               # name of control net2
    
    return "Pipeline Info Reset."



# ------------------------------------------------------------
def display_pipeline_info(last_ret_value):
    
    
    tempout = ""
    errout = ""
    
    # if int(SDPIPELINE['pipeline_loaded']) < 1:
        # tempout = tempout + str_no_model_loaded()
        # grinfo_no_model_loaded()
        # return tempout, "", "", "", ""

    if len(last_ret_value) > 0:
        main_string = last_ret_value
        substring = "Error"
        index = main_string.find(substring)
        if index != -1:
            errout = errout + "<h3>" + str(last_ret_value) + "</h3>"
            tempout = tempout + str_no_model_loaded()
            if SDPIPELINE['pipeline_source'] == "LCMLORA":
                return tempout, errout, "", "", ""
            if SDPIPELINE['pipeline_source'] == "HUB Cached":
                return tempout, "", errout, "", ""
            if SDPIPELINE['pipeline_source'] == "Huggingface":
                return tempout, "", "", errout, ""
            if SDPIPELINE['pipeline_source'] == "Safetensors":
                return tempout, "", "", "", errout
        else:
            return last_ret_value, "", "", "", ""




# ------------------------------------------------------------
def clear_lcm_model():
    
    return STUDIO["lcm_model_prefix"]["value"] + "MyNewModel" + STUDIO["lcm_model_suffix"]["value"], 1.0, "<h3>LCM-LoRA Model name has been defaulted and LoRA value reset to '1.0'.</h3></br>Replace the 'MyNewModel' part with your model name or completely rename it whatever you want. Keep in mind the user should use some sort of naming convention to keep track as to which models have had the 'LCM-LoRA' added. Like add 'LCM' to the model name on one end or the other. When loading other models you have the option of 'not' adding the LCM-LoRA weights to the loaded model, which would then be fused to the model and saved. And therefore models without the LCM-LoRA weights added to the saved model, it will not run an inference in just the normal 4 steps for an average model and produce a good image. You would need to run that model at it's normal higher number of step to reproduce a good image. Very slow for a CPU to do.<br>Suggestion: Use 'LCM' as the prefix for your 'LCM-LoRA' baked models, and add a suffix of either 'fp16' or 'fp32' to indicate the model's weight precision."



# ------------------------------------------------------------
def save_lcm_model(model_name,lora_value,use_safetensors,fp16):
    global pipeline             

    if not model_name:
        tempout = "<h3>No Model Name specified.<br.You must enter a name for your model.</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    if not bool(re.fullmatch(r'[a-zA-Z0-9_-]+', model_name)):
        tempout = "<h3>Model Name contains invalid characters.<br>Model Name can only contain letters, numbers, hyphens, and the underscore character.</h3>"
        yield gr.update(value=tempout)
        return tempout
        
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
        
    old_model_name = SDPIPELINE['pipeline_model_name']
    new_lcm_model_filename = model_name
    model_image_path_file = get_lcm_model_image_path_file(new_lcm_model_filename)
    
    new_lcm_model_filepathname = os.path.join(LLSTUDIO["lcm_model_dir"], new_lcm_model_filename)
    
    loadedloras = get_loaded_lora_models_text()
    
    try:
        if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
            tempout = "<h3>Fusing LoRA to Pipeline...</h3>"
            yield gr.update(value=tempout)
            pipeline.fuse_lora(lora_scale=lora_value)

            tempout = "<h3>Unloading LoRA Adapters...</h3>"
            yield gr.update(value=tempout)
            pipeline.unload_lora_weights()

            tempout = "<h3>Deleting LoRA Adapters...</h3>"
            adapter_names = pipeline.get_active_adapters()
            pipeline.delete_adapters(adapter_names)

            LLSTUDIO["loaded_lora_model_value"]=[]
            LLSTUDIO["loaded_lora_model_name"]=[]
            LLSTUDIO["loaded_lora_model_adapter"]=[]
            LLSTUDIO["lora_adapter_numb"] = 0
        
            tempout = "<h3>Finished Fusing LoRAs to Pipeline and LoRA Unloading Adapters.</h3>"
            yield gr.update(value=tempout)
    except Exception as e:
        tempout = "<h3>Error Fusing LoRAs to Pipeline Model. " + f"{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    if fp16:
        fp16_tempout = "fp16"
    else:
        fp16_tempout = "fp32"
    
    tempout = "<h3>Converting Pipeline.to " + fp16_tempout + "...</h3>"
    yield gr.update(value=tempout)
    
    try:
        if fp16:
            pipeline = pipeline.to(dtype=torch.float16)
        else:
            pipeline = pipeline.to(dtype=torch.float32)
    except Exception as e:
        tempout = "<h3>Error Converting Pipeline.to " + fp16_tempout + ". " + f"{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    tempout = "<h3>Saving Pipeline as LCM-LoRA Model: " + new_lcm_model_filename + "...</h3>"
    yield gr.update(value=tempout)
    
    pipeline_args = { }
    
    if fp16:
        pipeline_args["variant"] = "fp16"
        
    if use_safetensors:
        pipeline_args["safe_serialization"] = True
    
    try:
        pipeline.save_pretrained(f"{new_lcm_model_filepathname}", **pipeline_args)
    except Exception as e:
        tempout = "<h3>Error Saving Pipeline to Model. " + f"{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout
        
    SDPIPELINE['pipeline_model_name'] = new_lcm_model_filename
    SDPIPELINE['pipeline_source'] = "LCMLORA"
    if fp16:
        SDPIPELINE["pipeline_model_precision"] = "fp16"
    else:
        SDPIPELINE["pipeline_model_precision"] = "fp32"

    # create a model card (*.md) for this model 
    # and put it in the image gallery for this specific LCM-LoRA Model
    if not os.path.exists(model_image_path_file):
        os.makedirs(model_image_path_file)
    file1 = open(os.path.join(model_image_path_file, new_lcm_model_filename) + ".md", 'w')
    content = "## LCM-LoRA Model: " + new_lcm_model_filename + "\n\n\n"
    content = content + "## Original Model: " + old_model_name + "\n\n\n"
    content = content + "Loaded LoRAs: The 'LCM-LoRA', will not be shown if the model IS, an LCM-LoRA model. Because the LCM-LoRA has already been fused to the model, and given a prefixed model name to indicate model type\n\n\n"
    content = content + loadedloras + "\n\n\n"
    content = content + f"*Converted using {LLSTUDIO['app_title']} - {LLSTUDIO['app_version']}*\n\n\n"
    file1.write(content)
    file1.close()    

    tempout = "<h3>" + "Finished Saving New Model: " + new_lcm_model_filename + "</h3><h3>Re-Initializing Pipeline for Model Loading...</h3>"
    yield gr.update(value=tempout)

    slient_delete_pipeline()

    tempout = "<h3>Initialized Pipeline for Model Loading...</h3>"
    tempout = tempout + "<h3>Finished Saving Pipeline Loaded with " + old_model_name + " to LCM-LoRA model " + new_lcm_model_filename + "</h3>LoRAs: </br>" + get_loaded_lora_models_html()
    tempout = tempout + "<p>To use your New Model: " + new_lcm_model_filename + "</p>"
    tempout = tempout + "<p>Click on the 'Pipeline - Models' tab, then Click on the'LCM-LoRA Model List' tab. Once there, Click on the 'Refresh' button on the right, to update the LCM-LoRA Model list in the dropdown box</p>"
    tempout = tempout + "<p>Then select your New Model from the list, then Click the 'Load' button on the right to load the model.</p>"
    yield gr.update(value=tempout)


    return tempout



# ---------------------------------------------------------------
def load_lcm_model(model_name, use_diff_text_enc, text_enc_model_name, text_enc_clipskip, use_controlnet, controlnet_name, use_controlnet2, controlnet_name2, fp16, fp16e, add_lora, lora_value, use_lcm):
    
    global pipeline             
    
    SDPIPELINE['pipeline_source'] = "LCMLORA"
    
    if not model_name:
        tempout = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    pstart = time.time()

    tempout = "<h3>Loading Model: " + model_name + "<br>Initializing Pipeline...</h3>"
    yield gr.update(value=tempout)
    slient_delete_pipeline()
    tempout = "<h3>Pipeline Initialized.<br>Loading Model: " + model_name + "</h3>"
    yield gr.update(value=tempout)

    model_path_file = get_lcm_model_path_file(model_name)
    model_config_filename = os.path.join(model_path_file, "model_index.json")

    with open(model_config_filename, "r") as f:
        model_config_data = json.load(f)

    pipe_class = model_config_data["_class_name"]

    SDPIPELINE['pipeline_model_name'] = model_name
    SDPIPELINE['pipeline_class'] = pipe_class
    SDPIPELINE['pipeline_source'] = "LCMLORA"
    SDPIPELINE['pipeline_model_type'] = PIPECLASSES[pipe_class]['pipeline_model_type']
    SDPIPELINE['pipeline_gen_mode'] = PIPECLASSES[pipe_class]['pipeline_gen_mode']
    SDPIPELINE['pipeline_text_encoder'] = 0
    SDPIPELINE['pipeline_text_encoder_name'] = ""
    SDPIPELINE["pipeline_controlnet_loaded"] = 0
    SDPIPELINE["pipeline_controlnet_name"] = ""
    SDPIPELINE["pipeline_controlnet_name2"] = ""

    pipeline_args = {}

    if fp16:
        pipeline_args["variant"] = "fp16"
    
    if not STUDIO["use_safety_checker"]["value"]: 
        pipeline_args["safety_checker"] = None
        pipeline_args["requires_safety_checker"] = False
        pipeline_args["feature_extractor"] = None
    
    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True
 
 
    if STUDIO["low_memory"]["value"]: 
        
        pipeline_args["low_cpu_mem_usage"] = True


    if SDPIPELINE['pipeline_model_type'] == "SD15":
        text_enc_pipeline_args = {}
        
        # add the parameter for the precision variant we want to load, MUST exist !!
        # we can add a checkbox later fp16/fp32
        if fp16e:
            text_enc_pipeline_args["variant"] = "fp16"
        
        # Conditionally add the 'text_encoder' argument
        if int(text_enc_clipskip) > 1:
            num_hidden_layers = int(12 - (int(text_enc_clipskip) - 1))
            text_enc_pipeline_args["subfolder"] = "text_encoder"
            text_enc_pipeline_args["num_hidden_layers"] =  num_hidden_layers
        else:
            text_enc_pipeline_args["subfolder"] = "text_encoder"

        # do we use a different 'text_encoder' instead of loaded model text_encoder?
        if use_diff_text_enc:
            if text_enc_model_name:
                # Load the CLIP text encoder from a different model
                # and specify the number of layers to use.
                try:
                    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " Separate Text Encoder from " + text_enc_model_name + "</h3>"
                    yield gr.update(value=tempout)
                    text_encoder = transformers.CLIPTextModel.from_pretrained(get_lcm_model_path_file(text_enc_model_name), **text_enc_pipeline_args)
                    pipeline_args["text_encoder"] = text_encoder
                except Exception as e:
                    tempout = "<h3>Error Loading Separate Text Encoder: " + text_enc_model_name + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout

        # if so set it up to load before model 
        # similar to the separate text encoder method, but account for 2 ControlNets
        # 0. Figure out which model repo to use based on 'controlnet_name'
        # using dict 'CNETMODELS' to get actual huggingface model name
        # 1. Load the ControlNet model
        if (use_controlnet or use_controlnet2):
            # max 2 controlnets
            controlnet = []
            if use_controlnet:
                try:
                    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " ControlNet Model: " + controlnet_name + "</h3>"
                    yield gr.update(value=tempout)
                    controlnet.append(ControlNetModel.from_pretrained(CNETMODELS[controlnet_name]))
                except Exception as e: 
                    tempout = "<h3>Error Loading ControlNet Model: " + CNETMODELS[controlnet_name] + "<br>For ControlNet Named: " + controlnet_name + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout
            if use_controlnet2:
                try:
                    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " ControlNet Model: " + controlnet_name2 + "</h3>"
                    yield gr.update(value=tempout)
                    controlnet.append(ControlNetModel.from_pretrained(CNETMODELS[controlnet_name2]))
                except Exception as e: 
                    tempout = "<h3>Error Loading ControlNet Model: " + CNETMODELS[controlnet_name2] + "<br>For ControlNet Named: " + controlnet_name2 + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout
                    
            # if any controlnets got loaded we add the argument for the controlnet(s)    
            if len(controlnet) > 0:
                pipeline_args["controlnet"] = controlnet
                # by changing the pipeline class we can load model with the rest of them...
                # change the pipeline class from SD to CNET ! :)
                SDPIPELINE['pipeline_class'] = "StableDiffusionControlNetPipeline"



    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model: " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + "</h3>"
    yield gr.update(value=tempout)

    try:
        if SDPIPELINE['pipeline_class'] == "StableDiffusionPipeline":
            pipeline = StableDiffusionPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLPipeline":
            pipeline = StableDiffusionXLPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionImage2Image":
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLImage2Image":
            pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInpaintPipeline":
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInpaintPipeline":
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInstructPix2PixPipeline":
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInstructPix2PixPipeline":
            pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionControlNetPipeline":
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)
        else:
            tempout = "<h3>Error - No Pipeline Recognized for model: " + SDPIPELINE['pipeline_model_name'] + "</h3>"
            yield gr.update(value=tempout)
            return tempout
    except Exception as e:
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout
    

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e:
        tempout = "<h3>Error Moving TO device??: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    
    if add_lora:
        tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " LCM-LoRA weights for " + model_name + "</h3>"
        yield gr.update(value=tempout)
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        if SDPIPELINE['pipeline_model_type'] == "SDXL":
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl", weight_name="pytorch_lora_weights.safetensors")
        else:
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
        tempout = "<h3>Fusing " + SDPIPELINE['pipeline_model_type'] + " LCM-LoRA weights to " + model_name + "</h3>"
        yield gr.update(value=tempout)
        pipeline.fuse_lora(lora_scale=lora_value)
        tempout = "<h3>Unloading LoRAs Adapters since they are now 'fused' to the Model...</h3>"
        yield gr.update(value=tempout)
        pipeline.unload_lora_weights()
        adapter_names = pipeline.get_active_adapters()
        pipeline.delete_adapters(adapter_names)
        tempout = "<h3>Finished Deleting LoRA Adapters.</h3>"
        yield gr.update(value=tempout)
        loraout_text = " - with LCM-LoRA weights"
    else:
        if use_lcm:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            loraout_text = " - using LCM Scheduler"
        else:
            loraout_text = ""

    
    if STUDIO["low_memory_inf"]["value"]: 
        pipeline.vae.enable_slicing()
        pipeline.enable_attention_slicing("max")

    textencoder_txtout = ""
    controlnet_txtout = ""
    extra_txtout = ""
    
    if SDPIPELINE['pipeline_model_type'] == "SD15": 
        if use_diff_text_enc and text_enc_model_name:
            # we do this AFTER everything is completely done, with no errors
            SDPIPELINE['pipeline_text_encoder'] = 1
            SDPIPELINE['pipeline_text_encoder_name'] = text_enc_model_name
            textencoder_txtout = "TextEncoder: " + text_enc_model_name
        if (use_controlnet or use_controlnet2):
            # we do this AFTER everything is completely done, with no errors
            SDPIPELINE['pipeline_controlnet_loaded'] = int(len(controlnet))
            if use_controlnet:
                SDPIPELINE['pipeline_controlnet_name'] = controlnet_name
                controlnet_txtout = "ControlNet: " + controlnet_name
            if use_controlnet2:
                SDPIPELINE['pipeline_controlnet_name2'] = controlnet_name2
                controlnet_txtout = "ControlNet: " + controlnet_name2
            if (use_controlnet and use_controlnet2):
                controlnet_txtout = "ControlNets: " + controlnet_name + "/" + controlnet_name2
 
    pend = time.time()
    pelapsed = pend - pstart

    if int(len(textencoder_txtout)) > 0:
        extra_txtout = extra_txtout + textencoder_txtout + "<br>"
    if int(len(controlnet_txtout)) > 0:
        extra_txtout = extra_txtout + controlnet_txtout   

    # line break
    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode']  + loraout_text + "<br>" + extra_txtout + "</h3>"
    # # no break
    # tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Loaded Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + loraout_text + extra_txtout + "</h3>"
    yield gr.update(value=tempout)
    gr.Info(SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model loaded:</br>" + SDPIPELINE['pipeline_model_name'] + "</br>" + format_seconds_strftime(pelapsed), duration=5.0, title=SDPIPELINE['pipeline_source'] + " Model")

    # we do this AFTER everything is completely done, with no errors
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout





# ------------------------------------------------------------------
def load_hub_model(model_name, fp16_check, use_lcmscheduler, lora_value, add_lcmlora):
    global pipeline             

    SDPIPELINE['pipeline_source'] = "HUB Cached"
    
    if not model_name:
        tempout = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    pstart = time.time()

    tempout = "<h3>Loading Model: " + model_name + "<br>Initializing Pipeline...</h3>"
    yield gr.update(value=tempout)
    slient_delete_pipeline()
    tempout = "<h3>Pipeline Initialized.<br>Loading Model: " + model_name + "</h3>"
    yield gr.update(value=tempout)

    try:
        hex40str = get_file_content(os.path.join(get_hub_model_path_file(model_name), "refs", "main"))
        with open(os.path.join(get_hub_model_path_file(model_name), "snapshots", hex40str, "model_index.json"), "r") as f:
            model_config_data = json.load(f)
        model_dir_name = os.path.join(get_hub_model_path_file(model_name), "snapshots", hex40str)
        pipe_class = model_config_data["_class_name"]
    except Exception as e:
        tempout = "<h3>Error: MODEL CONFIG NOT FOUND. " + model_name + "</h3>"
        yield gr.update(value=tempout)
        return tempout   

    SDPIPELINE['pipeline_model_name'] = model_name
    SDPIPELINE['pipeline_class'] = pipe_class
    SDPIPELINE['pipeline_source'] = "HUB Cached"
    SDPIPELINE['pipeline_model_type'] = PIPECLASSES[pipe_class]['pipeline_model_type']
    SDPIPELINE['pipeline_gen_mode'] = PIPECLASSES[pipe_class]['pipeline_gen_mode']
    SDPIPELINE['pipeline_text_encoder'] = 0
    SDPIPELINE['pipeline_text_encoder_name'] = ""

    pipeline_args = {}

    if fp16_check:
        pipeline_args["variant"] = "fp16"


    if not STUDIO["use_safety_checker"]["value"]: 
        pipeline_args["safety_checker"] = None
        pipeline_args["requires_safety_checker"] = False
        pipeline_args["feature_extractor"] = None


    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True

 
 
    if STUDIO["low_memory"]["value"]: 
        pipeline_args["low_cpu_mem_usage"] = True


    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model: " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + "</h3>"
    yield gr.update(value=tempout)


    try:
        if pipe_class == "StableDiffusionPipeline":
            pipeline = StableDiffusionPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLPipeline":
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionImage2Image":
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLImage2Image":
            pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionInpaintPipeline":
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLInpaintPipeline":
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionInstructPix2PixPipeline":
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_dir_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLInstructPix2PixPipeline":
            pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_dir_name, **pipeline_args)
        else:
            tempout = "<h3>Error - No Pipeline Recognized for model: " + SDPIPELINE['pipeline_model_name'] + "</h3>"
            yield gr.update(value=tempout)
            return tempout
    except Exception as e:
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e:
        tempout = "<h3>Error Moving TO device??: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout



    if add_lcmlora:
        tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " LCM-LoRA weights for " + model_name + "</h3>"
        yield gr.update(value=tempout)
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        if SDPIPELINE['pipeline_model_type'] == "SDXL":
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl", weight_name="pytorch_lora_weights.safetensors")
        else:
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
        tempout = "<h3>Fusing " + SDPIPELINE['pipeline_model_type'] + " LCM-LoRA weights to " + model_name + "</h3>"
        yield gr.update(value=tempout)
        pipeline.fuse_lora(lora_scale=lora_value)
        tempout = "<h3>Unloading LoRAs Adapters since they are now 'fused' to the Model...</h3>"
        yield gr.update(value=tempout)
        pipeline.unload_lora_weights()
        adapter_names = pipeline.get_active_adapters()
        pipeline.delete_adapters(adapter_names)
        tempout = "<h3>Finished Deleting LoRA Adapters.</h3>"
        yield gr.update(value=tempout)
        loraout_text = " - with LCM-LoRA weights"
    else:
        if add_lcm_scheduler:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            loraout_text = " - using LCM Scheduler"
        else:
            loraout_text = ""
    

    pend = time.time()
    pelapsed = pend - pstart

    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Loaded Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " - " + loraout_text + "</h3>"
    yield gr.update(value=tempout)
    gr.Info(SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model loaded for " + SDPIPELINE['pipeline_gen_mode'] + ":</br>" + SDPIPELINE['pipeline_model_name'] + " - " + loraout_text + "</br>" + format_seconds_strftime(pelapsed), duration=5.0, title=SDPIPELINE['pipeline_source'] + " Model")
    
   
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout



# ------------------------------------------------------------------
def load_hug_model(model_name, model_class_name, fp16_check):
    global pipeline             

    SDPIPELINE['pipeline_source'] = "Huggingface"

    if len(model_name) < 1:
        tempout = "<h3>Error: MUST Enter valid Huggingface Model Name: " + model_name + "</h3>"
        yield gr.update(value=tempout)
        return tempout   
    
    pstart = time.time()

    tempout = "<h3>Loading Model: " + model_name + "<br>Initializing Pipeline...</h3>"
    yield gr.update(value=tempout)
    slient_delete_pipeline()
    tempout = "<h3>Pipeline Initialized.<br>Loading Model: " + model_name + "</h3>"
    yield gr.update(value=tempout)

    pipe_class = model_class_name
    SDPIPELINE['pipeline_model_name'] = model_name
    SDPIPELINE['pipeline_class'] = pipe_class
    SDPIPELINE['pipeline_source'] = "Huggingface"
    SDPIPELINE['pipeline_model_type'] = PIPECLASSES[pipe_class]['pipeline_model_type']
    SDPIPELINE['pipeline_gen_mode'] = PIPECLASSES[pipe_class]['pipeline_gen_mode']
    SDPIPELINE['pipeline_text_encoder'] = 0
    SDPIPELINE['pipeline_text_encoder_name'] = ""

    pipeline_args = {}

    if fp16_check:
        pipeline_args["variant"] = "fp16"

    if not STUDIO["use_safety_checker"]["value"]: 
        pipeline_args["safety_checker"] = None
        pipeline_args["requires_safety_checker"] = False
        pipeline_args["feature_extractor"] = None


    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True

 
    if STUDIO["low_memory"]["value"]: 
        pipeline_args["low_cpu_mem_usage"] = True


    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model: " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + "</h3>"
    yield gr.update(value=tempout)
    
    try:
        if pipe_class == "StableDiffusionPipeline":
            pipeline = StableDiffusionPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLPipeline":
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionImage2Image":
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLImage2Image":
            pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionInpaintPipeline":
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLInpaintPipeline":
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionInstructPix2PixPipeline":
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, **pipeline_args)
        elif pipe_class == "StableDiffusionXLInstructPix2PixPipeline":
            pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_name, **pipeline_args)
        else:
            tempout = "<h3>Error - No Pipeline Recognized for model: " + SDPIPELINE['pipeline_model_name'] + "</h3>"
            yield gr.update(value=tempout)
            return tempout

    except Exception as e:
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e:
        tempout = "<h3>Error Moving TO device??: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    

    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + "</h3>"
    yield gr.update(value=tempout)
    pend = time.time()
    pelapsed = pend - pstart
    gr.Info(SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model" + SDPIPELINE['pipeline_model_name'] + "loaded for " + SDPIPELINE['pipeline_gen_mode'] + " " + format_seconds_strftime(pelapsed), duration=5.0, title="LCM-LoRA Model")
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout




# ------------------------------------------------------------
def load_safetensors_model(safetensors_model, pipe_class, lora_value, add_lcmlora, use_diff_text_enc, lcm_enc_model_name, lcm_fp16e, add_lcm_scheduler):
    global pipeline             
    
    SDPIPELINE['pipeline_source'] = "Safetensors"

    if not safetensors_model:
        tempout = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    safetensors_model_pathfile = os.path.join(LLSTUDIO["safe_model_dir"], safetensors_model + ".safetensors")
    pstart = time.time()

    tempout = "<h3>Loading Model: " + safetensors_model + "<br>Initializing Pipeline...</h3>"
    yield gr.update(value=tempout)
    slient_delete_pipeline()
    tempout = "<h3>Pipeline Initialized.<br>Loading Model: " + safetensors_model + "</h3>"
    yield gr.update(value=tempout)


    SDPIPELINE['pipeline_model_name'] = safetensors_model
    SDPIPELINE['pipeline_class'] = pipe_class
    SDPIPELINE['pipeline_source'] = "Safetensors"
    SDPIPELINE['pipeline_model_type'] = PIPECLASSES[pipe_class]['pipeline_model_type']
    SDPIPELINE['pipeline_gen_mode'] = PIPECLASSES[pipe_class]['pipeline_gen_mode']
    SDPIPELINE['pipeline_text_encoder'] = 0
    SDPIPELINE['pipeline_text_encoder_name'] = ""
    
    pipeline_args = {}

    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True

 
    if STUDIO["low_memory"]["value"]: 
        pipeline_args["low_cpu_mem_usage"] = True

    if use_diff_text_enc:
        text_enc_pipeline_args = {}
        
        if lcm_fp16e:
            # add the parameter for the precision variant we want to load, MUST exist !!
            # we can add a checkbox later fp16/fp32
            text_enc_pipeline_args["variant"] = "fp16"
        
        # Conditionally add the 'text_encoder' argument
        text_enc_pipeline_args["subfolder"] = "text_encoder"
        # do we use a different 'text_encoder' instead of loaded model text_encoder?
        if use_diff_text_enc:
            if lcm_enc_model_name:
                # Load the CLIP text encoder from a different model
                # and NOT specify the number of layers to use.
                # that can be done AFTER you make your LCM-LoRA model from this one,
                # using the LCM-LoRA model when you load it then.
                try:
                    text_encoder = transformers.CLIPTextModel.from_pretrained(get_lcm_model_path_file(lcm_enc_model_name), **text_enc_pipeline_args)
                    pipeline_args["text_encoder"] = text_encoder
                except Exception as e:
                    tempout = "<h3>Error Loading Separate Text Encoder: " + lcm_enc_model_name + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout

    # use 'original_config_file' when loading the safetensors model
    if STUDIO["safe_use_original_config_file"]["value"]:
        if SDPIPELINE['pipeline_class'] == "StableDiffusionPipeline":
            if len(STUDIO["SD_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SD_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLPipeline":
            if len(STUDIO["SDXL_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXL_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionImage2Image":
            if len(STUDIO["SDImage2Image_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDImage2Image_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLImage2Image":
            if len(STUDIO["SDXLImage2Image_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXLImage2Image_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInpaintPipeline":
            if len(STUDIO["SDInpaint_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDInpaint_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInpaintPipeline":
            if len(STUDIO["SDXLInpaint_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXLInpaint_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInstructPix2PixPipeline":
            if len(STUDIO["SDInstructPix2Pix_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDInstructPix2Pix_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInstructPix2PixPipeline":
            if len(STUDIO["SDXLInstructPix2Pix_original_config"]["value"]) > 0:
                pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXLInstructPix2Pix_original_config"]["value"])   

    # use a reference model when loading the safetensors model
    if STUDIO["safe_use_config"]["value"]:
        if SDPIPELINE['pipeline_class'] == "StableDiffusionPipeline":
            if len(STUDIO["SD_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SD_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLPipeline":
            if len(STUDIO["SDXL_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDXL_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionImage2Image":
            if len(STUDIO["SDImage2Image_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDImage2Image_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLImage2Image":
            if len(STUDIO["SDXLImage2Image_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDXLImage2Image_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInpaintPipeline":
            if len(STUDIO["SDInpaint_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDInpaint_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInpaintPipeline":
            if len(STUDIO["SDXLInpaint_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDXLInpaint_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInstructPix2PixPipeline":
            if len(STUDIO["SDInstructPix2Pix_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDInstructPix2Pix_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInstructPix2PixPipeline":
            if len(STUDIO["SDXLInstructPix2Pix_config"]["value"]) > 0:
                pipeline_args["config"] = STUDIO["SDXLInstructPix2Pix_config"]["value"] 

 
    tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model: " + safetensors_model + " for " + SDPIPELINE['pipeline_gen_mode'] + "</h3>"
    yield gr.update(value=tempout)

    try:
        if pipe_class == "StableDiffusionPipeline":
            pipeline = StableDiffusionPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionXLPipeline":
            pipeline = StableDiffusionXLPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionImage2Image":
            pipeline = StableDiffusionImg2ImgPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionXLImage2Image":
            pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionInpaintPipeline":
            pipeline = StableDiffusionInpaintPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionXLInpaintPipeline":
            pipeline = StableDiffusionXLInpaintPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionInstructPix2PixPipeline":
            pipeline = StableDiffusionInstructPix2PixPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        elif pipe_class == "StableDiffusionXLInstructPix2PixPipeline":
            pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(safetensors_model_pathfile, **pipeline_args)
        else:
            tempout = "<h3>Error - No Pipeline Recognized for model: " + safetensors_model + "</h3>"
            yield gr.update(value=tempout)
            return tempout
    except Exception as e:
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + safetensors_model + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e:
        tempout = "<h3>Error Moving TO device??: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + safetensors_model + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
    
    if add_lcmlora:
        tempout = "<h3>Loading " + SDPIPELINE['pipeline_model_type'] + " LCM-LoRA weights for " + safetensors_model + "</h3>"
        yield gr.update(value=tempout)
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        if SDPIPELINE['pipeline_model_type'] == "SDXL":
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl", weight_name="pytorch_lora_weights.safetensors")
        else:
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
        tempout = "<h3>Fusing " + SDPIPELINE['pipeline_model_type'] + " LCM-LoRA weights to " + safetensors_model + "</h3>"
        yield gr.update(value=tempout)
        pipeline.fuse_lora(lora_scale=lora_value)
        tempout = "<h3>Unloading LoRAs Adapters since they are now 'fused' to the Model...</h3>"
        yield gr.update(value=tempout)
        pipeline.unload_lora_weights()
        adapter_names = pipeline.get_active_adapters()
        pipeline.delete_adapters(adapter_names)
        tempout = "<h3>Finished Deleting LoRA Adapters.</h3>"
        yield gr.update(value=tempout)
        loraout_text = " - with LCM-LoRA Weights"
    else:
        if add_lcm_scheduler:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            loraout_text = " - using LCM Scheduler"
        else:
            loraout_text = ""

    pend = time.time()
    pelapsed = pend - pstart

    if use_diff_text_enc:
        if SDPIPELINE['pipeline_model_type'] == "SD15":
            if lcm_enc_model_name:
                # we do this AFTER everything is completely done, with no errors
                SDPIPELINE['pipeline_text_encoder'] = 1
                SDPIPELINE['pipeline_text_encoder_name'] = lcm_enc_model_name
                if add_lcmlora:
                    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " Text Encoder from : " + lcm_enc_model_name + " - " + loraout_text + "</h3>"
                else:
                    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Loaded Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " - " + loraout_text + "</h3>"
    else:
        tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Loaded Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " - " + loraout_text + "</h3>"

    yield gr.update(value=tempout)
    gr.Info(SDPIPELINE['pipeline_model_type'] + " Safetensors Model loaded for " + SDPIPELINE['pipeline_gen_mode'] + ":</br>" + SDPIPELINE['pipeline_model_name'] + " - with LCM LoRA</br>" + format_seconds_strftime(pelapsed), duration=5.0, title="Safetensors Model")
   
    
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout
    




# --------------------------------------------------------------
def convert_to_safetensors_model(lcm_model_name, use_fp16, safe_model_name, use_half, use_safe, use_all_safe, model_card_info):
    
    if not lcm_model_name:
        tempout = "<h3>No LCM-LoRA Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    if not safe_model_name:
        tempout = "<h3>No Safetensors Model Name entered.<br>Please enter a valid model name. No extension.</h3>"
        yield gr.update(value=tempout)
        return tempout

    binfile_found = ""
    binfile_found_card = ""

    tempout = "<h3>Starting Model Conversion<br>Converting Model: " + lcm_model_name + "</h3>"
    yield gr.update(value=tempout)

    model_path_file = get_lcm_model_path_file(lcm_model_name)
    model_config_filename = os.path.join(model_path_file, "model_index.json")

    try:
        with open(model_config_filename, "r") as f:
            model_config_data = json.load(f)
    except Exception as e:
        tempout = f"<h3>Error: 'model_index.json' found in model folder. {str(e)}</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    if not model_config_data:
        tempout = f"<h3>Error: No data parsed from 'model_index.json' found in model folder.</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    pipe_class = model_config_data["_class_name"]
    pipe_type = PIPECLASSES[pipe_class]['pipeline_model_type']

    safe_model_path_file = get_safe_model_path_file(safe_model_name + ".safetensors")
    safe_model_image_path = get_safe_model_image_path_file(safe_model_name)
    
    try:
        if pipe_type == "SDXL":
            result = convert_sdxl_to_safetensors(model_path_file, use_fp16, safe_model_path_file, use_half, use_safe, use_all_safe)
        else:
            result = convert_sd_to_safetensors(model_path_file, use_fp16, safe_model_path_file, use_half, use_safe, use_all_safe)
    except Exception as e:
        tempout = f"<h3>Error: {str(e)}</h3>"
        yield gr.update(value=tempout)
        return tempout
     
    if result != "OK" and result != "OKB":
        tempout = f"<h3>Conversion Routine Error: {result}</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    
    # fix up some text for model card, for each model, lcm-lora and safetensors
    if use_half:
        safe_precision = "fp16"
    else:
        safe_precision = "fp32"
    
    if use_fp16:
        mod_precision = "fp16"
    else:
        mod_precision = "fp32"

    if result == "OKB":
        binfile_found = "<br>NOTE: At least one component used a *.BIN file as the model for that component.<br>Check BOTH 'Use Safetensors' and 'Use ALL Safetensors ONLY' to use <u>ONLY Safetensors model components</u>"
        binfile_found_card = "## WARNING: At least one component used a *.BIN file as the model for that component."
    
    # create a model card (*.md) for this model 
    # and put it in the image gallery for this specific Model
    if not os.path.exists(safe_model_image_path):
        os.makedirs(safe_model_image_path)
    file1 = open(os.path.join(safe_model_image_path, safe_model_name) + ".md", 'w')
    content = "## Safetensors Model: " + safe_model_name + ".safetensors\n"
    content = content + "### Model Type: " + pipe_type + "\n\n\n"
    content = content + "### Model Precision: " + safe_precision + "\n\n\n"
    content = content + "## Original Model: " + lcm_model_name + "\n"
    content = content + "### Original Model Precision: " + mod_precision + "\n\n\n"
    content = content + f"*Converted using {LLSTUDIO['app_title']} - {LLSTUDIO['app_version']}*\n\n\n"
    content = content + binfile_found_card + "\n\n\n"
    content = content + model_card_info + "\n\n\n"
    file1.write(content)
    file1.close()    

    tempout = f"<h3>Successfully converted: '{lcm_model_name}'<br>to: '{safe_model_name}.safetensors'.</h3><br>Also created model card: '{safe_model_name}.md'{binfile_found}"
    yield gr.update(value=tempout)
    return tempout






# --------------------------------------------------------------
def download_huggingface_model(model_name):
    
    SDPIPELINE['pipeline_source'] = "Huggingface"
    
    if len(model_name) < 1:
        tempout = "<h3>Error: MUST Enter valid Huggingface Model Name.</br>Name Provided: " + model_name + "</br>Ex: stable-diffusion-v1-5/stable-diffusion-v1-5</h3>"
        yield gr.update(value=tempout)
        return tempout   
    tempout = "<h3>" + "Downloading Huggingface Model: " + model_name + " ...</h3></br>"
    yield gr.update(value=tempout)
    try:
        if STUDIO["local_files_only"]["value"]: 
            snapshot_download(repo_id=model_name, force_download=False, local_files_only=True)
        else:
            snapshot_download(repo_id=model_name, force_download=False)
    except Exception as e:    
        tempout = "<h3>Error: Unable to download model from Huggingface: " + model_name + "</h3>"
        yield gr.update(value=tempout)
        return tempout   
    tempout = "<h3>" + "Finished Downloading Huggingface Model: " + model_name + "</h3></br>"
    yield gr.update(value=tempout)
    return tempout



# ----------------------------------------------------------------------------------------------
def sysmodel_start_download(dmc1,dmc2,dmc3,dmc4,dmc5,dmc6,dmc7,dmc8,dmc9,dmc10,dmc11,dmc12,dmc13,dmc14,dmc15,dmc16,dmc17,dmc18,dmc19,dmc20,dmc21,dmc22,dmc23):
    
    global DOWNLOAD_MODELS_FLAG
    
    dmo1=dmc1; dmo2=dmc2; dmo3=dmc3; dmo4=dmc4; dmo5=dmc5; dmo6=dmc6; dmo7=dmc7; dmo8=dmc8; dmo9=dmc9; dmo10=dmc10; 
    dmo11=dmc11; dmo12=dmc12; dmo13=dmc13; dmo14=dmc14; dmo15=dmc15; dmo16=dmc16; dmo17=dmc17; dmo18=dmc18; dmo19=dmc19; 
    dmo20=dmc20; dmo21=dmc21; dmo22=dmc22; dmo23=dmc23 
    
    tempout = "<h3>Ready...</h3>"
    
    # check if already downloading...
    if DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Already downloading models...<br>Please wait until finished, or you may also cancel downloading.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

    if not dmc1 and not dmc2 and not dmc3 and not dmc4 and not dmc5 and not dmc6 and not dmc7 and not dmc8 and not dmc9 and not dmc10 and not dmc11 and not dmc12 and not dmc13 and not dmc14 and not dmc15 and not dmc16 and not dmc17 and not dmc18 and not dmc19 and not dmc20 and not dmc21 and not dmc22 and not dmc23:
        tempout = "<h3>No Models Selected for download.<br>Please select a model and try again.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        


    DOWNLOAD_MODELS_FLAG = True
    pipeline_args = {}


    if dmc1:
        pipeline_args = {}
        model_name = "latent-consistency/lcm-lora-sdv1-5"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            if STUDIO["local_files_only"]["value"]: 
                snapshot_download(repo_id=model_name, force_download=False, local_files_only=True)
            else:
                snapshot_download(repo_id=model_name, force_download=False)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo1=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))   



    if dmc2:
        pipeline_args = {}
        model_name = "latent-consistency/lcm-lora-sdxl"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            if STUDIO["local_files_only"]["value"]: 
                snapshot_download(repo_id=model_name, force_download=False, local_files_only=True)
            else:
                snapshot_download(repo_id=model_name, force_download=False)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo2=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))   


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc3:
        pipeline_args = {}
        pipeline_args["variant"] = "fp32"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "stabilityai/sd-x2-latent-upscaler"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo3=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc4:
        pipeline_args = {}
        pipeline_args["variant"] = "fp16"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo4=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc5:
        pipeline_args = {}
        pipeline_args["variant"] = "fp32"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo5=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()



    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc6:
        pipeline_args = {}
        pipeline_args["variant"] = "fp16"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo6=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc7:
        pipeline_args = {}
        pipeline_args["variant"] = "fp32"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo7=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()



    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc8:
        pipeline_args = {}
        pipeline_args["variant"] = "fp16"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "stable-diffusion-v1-5/stable-diffusion-inpainting"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo8=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc9:
        pipeline_args = {}
        pipeline_args["variant"] = "fp16"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo9=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()



    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))



    if dmc10:
        pipeline_args = {}
        pipeline_args["variant"] = "fp32"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo10=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()




    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))



    if dmc11:
        pipeline_args = {}
        pipeline_args["variant"] = "fp16"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "timbrooks/instruct-pix2pix"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo11=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()



    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc12:
        pipeline_args = {}
        pipeline_args["variant"] = "fp32"
        model_name = "timbrooks/instruct-pix2pix"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo12=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()



    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))



    # snaphot download is better for this one because the repo is broken
    # models are ok. but amgs on fp16, if done separately like the rest.
    # So we just go ahead and get both, we get a few 100 training images, ok.
    if dmc13 or dmc14:
        pipeline_args = {}
        model_name = "diffusers/sdxl-instructpix2pix-768"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            if STUDIO["local_files_only"]["value"]: 
                snapshot_download(repo_id=model_name, force_download=False, local_files_only=True)
            else:
                snapshot_download(repo_id=model_name, force_download=False)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo13=False
            dmo14=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))   




    # snaphot download wa used to get the repo, above...
    # so here we are just checking it. But it will fix replace missing files though...
    if dmc14:
        pipeline_args = {}
        pipeline_args["variant"] = "fp32"
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "diffusers/sdxl-instructpix2pix-768"
        tempout = "Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_name, **pipeline_args)
            tempout = "Finished Downloading " + model_name + " - " +  pipeline_args["variant"] + " ...."
            dmo14=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"Error Downloading {model_name} - {pipeline_args['variant']} ....\n{e}\n\n"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            
        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc15:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-mlsd"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo15=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 
            
        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc16:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-hed"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo16=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc17:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-depth"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo17=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc18:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-scribble"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo18=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))



    if dmc19:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-canny"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo19=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))



    if dmc20:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-normal"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo20=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc21:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-seg"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo21=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc22:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "lllyasviel/sd-controlnet-openpose"
        tempout = "<h3>" + "Downloading Model: " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = ControlNetModel.from_pretrained(model_name, **pipeline_args)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo22=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    if not DOWNLOAD_MODELS_FLAG:
        tempout = "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))


    if dmc23:
        pipeline_args = {}
        if STUDIO["local_files_only"]["value"]:
            pipeline_args["local_files_only"] = True
        model_name = "depth-estimation"
        tempout = "<h3>Downloading " + model_name + " ...</h3>"
        yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))

        try:
            pipeline = transformers.pipeline(model_name)
            tempout = "<h3>Finished Downloading " + model_name + " ....</h3>"
            dmo23=False
            if not DOWNLOAD_MODELS_FLAG:
                tempout = tempout + "<h3>Canceled Downloading of System Models...</br>Downloading has stopped.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
                return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            else:
                yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
        except Exception as e:
            DOWNLOAD_MODELS_FLAG = False
            tempout = f"<h3>Error Downloading {model_name} ....</h3><pre>\n{e}\n\n</pre>"
            yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
            return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23)) 

        pipeline = None
        del pipeline
        gc.collect()


    DOWNLOAD_MODELS_FLAG = False
    
    tempout = "<h3>Finished Downloading System Models.</br>The 'checkbox' will be cleared for models that finished downloading.<br>Any newly downloaded SD or SDXL models that can be used for inference will appear in the dropdown box under the 'Huggingface (Local Cached) Models List' tab.</br>There, you can load them for inference and generate images from there.</br>Note: You will have to refresh the model dropdown box for any of the newer models to show up.</h3>"
    yield (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))
    return (gr.update(value=tempout),gr.update(value=dmo1),gr.update(value=dmo2),gr.update(value=dmo3),gr.update(value=dmo4),gr.update(value=dmo5),gr.update(value=dmo6),gr.update(value=dmo7),gr.update(value=dmo8),gr.update(value=dmo9),gr.update(value=dmo10),gr.update(value=dmo11),gr.update(value=dmo12),gr.update(value=dmo13),gr.update(value=dmo14),gr.update(value=dmo15),gr.update(value=dmo16),gr.update(value=dmo17),gr.update(value=dmo18),gr.update(value=dmo19),gr.update(value=dmo20),gr.update(value=dmo21),gr.update(value=dmo22),gr.update(value=dmo23))






# ----------------------------------------------------------------------------------------------


def sysmodel_cancel_download(dlstatus):
    global DOWNLOAD_MODELS_FLAG
    if not DOWNLOAD_MODELS_FLAG:
        tempout = dlstatus
        yield gr.update(value=tempout)
        return tempout
    
    DOWNLOAD_MODELS_FLAG = False
    tempout = dlstatus + "<br><h3>Canceled Downloading System Models...</br>Downloading will stop, after the current model has finished downloading.</br>The 'checkbox' will be cleared for models that finished downloading.</h3>"
    yield gr.update(value=tempout)
    return tempout




# ----------------------------------------------------------------------------------------------
def sysmodels_uncheckall_checkboxes():

    tempout=False
    return (gr.update(value=tempout),gr.update(value=tempout),
        gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout))


# ----------------------------------------------------------------------------------------------
def sysmodels_checkdefaults_checkboxes():

    tempout=False
    return (gr.update(value=True),gr.update(value=tempout),
        gr.update(value=tempout),gr.update(value=True),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout),gr.update(value=tempout))






# ----------------------------------------------------------------------------------------------
def delete_hub_model(model_name, del_model):

    if not model_name:
        tempout = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    hub_model_full_path = get_hub_model_path_file(model_name)

    hub_model_lock_full_path = os.path.join(LLSTUDIO["hub_model_dir"], ".locks", model_name)
    contents = ""
    
    if del_model:
        
        if os.path.exists(hub_model_full_path):
            try:
                shutil.rmtree(hub_model_full_path)
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Hub Cached Model '{model_name}' directory not found. Can not delete.")
                contents = f"Error: Hub Cached Model '{model_name}' directory not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting Hub Cached Model '{model_name}' directory: {e}")    
                contents = f"Error: Error deleting Hub Cached Model '{model_name}' directory: {e}"
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Hub Cached Model '{model_name}' path or directory does not exist.")
            contents = f"Error: Hub Cached Model '{model_name}' path or directory does not exist."
        
        if os.path.exists(hub_model_lock_full_path):
            try:
                shutil.rmtree(hub_model_lock_full_path)
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Hub Cached Model '.locks' folder for, '{model_name}' directory not found. Can not delete.")
                contents = f"Error: Hub Cached Model '.locks' folder for, '{model_name}' directory not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting Hub Cached Model '.locks' folder for, '{model_name}' directory: {e}")    
                contents = f"Error: Error deleting Hub Cached Model '.locks' folder for, '{model_name}' directory: {e}"
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Hub Cached Model '.locks' folder for, '{model_name}' path or directory does not exist.")
            contents = f"Error: Hub Cached Model '.locks' folder for, '{model_name}' path or directory does not exist."

    else:
        contents = "You Must Select to Delete the Model, Check the box."


    if len(contents) < 1:
        contents = f"Hub Cached Model '{model_name}', its directory and contents deleted successfully."

    tempout = "<h3>" + contents + "</h3>"
    yield gr.update(value=tempout)
    gr.Info("<h3>" + contents + "</h3>", duration=5.0, title="Hub Local Cached Model")
    return tempout



# ----------------------------------------------------------------------------------------------
def add_lora_model(model_name, loravalue, use_lcm):
    global pipeline             
    
    if not model_name:
        tempout = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
        
    pstart = time.time()
    tempout = "<h3>Loading Lora...&nbsp;&nbsp;&nbsp;" + model_name + "</h3>"
    yield gr.update(value=tempout)
    lora_model_full_name = model_name + ".safetensors"
    LLSTUDIO["lora_adapter_numb"] = LLSTUDIO["lora_adapter_numb"] + 1
    lora_adapter_name = "lora" + str(LLSTUDIO["lora_adapter_numb"])

    if use_lcm:
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    pipeline.load_lora_weights(LLSTUDIO["lora_model_dir"], weight_name=lora_model_full_name, adapter_name=lora_adapter_name)
    pipeline.set_adapters([lora_adapter_name], adapter_weights=[loravalue])
    
    LLSTUDIO["loaded_lora_model_value"].append(str(loravalue))
    LLSTUDIO["loaded_lora_model_name"].append(model_name)
    LLSTUDIO["loaded_lora_model_adapter"].append(lora_adapter_name)

    if int(STUDIO["app_debug"]["value"]) > 0: print ("Lora loaded: " + model_name)
    if int(STUDIO["app_debug"]["value"]) > 0: print ("Lora Adapter: " + lora_adapter_name)
    if int(STUDIO["app_debug"]["value"]) > 0: print ("Lora Value: " + str(loravalue))
    tempout = "<h3>Loaded Lora: " + model_name + "</br>Lora Adapter: " + lora_adapter_name + "</br>Lora Value: " + str(loravalue) + "</h3>"
    yield gr.update(value=tempout)
    pend = time.time()
    pelapsed = pend - pstart
    gr.Info("Lora loaded: " + model_name + "</br>" + format_seconds_strftime(pelapsed), duration=3.0, title="Lora Model")
    return tempout

# ----------------------------------------------------------------------------------------------
def change_lora_model(model_name, loravalue):
                 
    if not model_name:
        tempout = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        yield gr.update(value=tempout)
        return tempout

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
    
    if len(LLSTUDIO["loaded_lora_model_adapter"]) < 1:
        tempout = "<h3>No LoRA Model Loaded !!</br>Can not change LoRA Weight !</h3>"
        yield gr.update(value=tempout)
        gr.Info("No LoRA Model Loaded !!</br>Can not change LoRA Weight !", duration=3.0, title="LoRA Change Weight")
        return tempout
    
    pstart = time.time()
    
    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            if model_name == LLSTUDIO["loaded_lora_model_name"][i]:
                loaded_lora_adapter = LLSTUDIO["loaded_lora_model_adapter"][i]
                tempout = "<h3>Changing Lora Weights on model: " + model_name + "</h3>"
                yield gr.update(value=tempout)
                pipeline.set_adapters([loaded_lora_adapter], adapter_weights=[loravalue])
                LLSTUDIO["loaded_lora_model_value"][i] = loravalue
                tempout = "<h3>Changed Lora Weights on model: " + model_name + "</h3>"
                yield gr.update(value=tempout)
    
    pend = time.time()
    pelapsed = pend - pstart
    
    tempout = "<h3>Loaded Lora: " + model_name + "</br>Lora Adapter: " + loaded_lora_adapter + "</br>Lora Value: " + str(loravalue) + "</h3>"
    yield gr.update(value=tempout)
    gr.Info("Lora Weights Changed on model: " + model_name + "</br>", title="Lora Model")
    return tempout




# ------------------------------------------------------
def list_lora_model():
    global pipeline             
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
    
    tempout = "<h3>Loaded LoRA Adapters: " + str(len(LLSTUDIO["loaded_lora_model_adapter"])) + "</h3></br>"

    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        tempout = tempout + "<pre>"
        for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            tempout = tempout + "Adapter Name: " + LLSTUDIO["loaded_lora_model_adapter"][i] + "</br>"
            tempout = tempout + "LoRA Model Name: " + LLSTUDIO["loaded_lora_model_name"][i] + "</br>"
            tempout = tempout + "LoRA Model Value: " + str(LLSTUDIO["loaded_lora_model_value"][i]) + "</br>"
            tempout = tempout + "----------------------------------</br>"
        tempout = tempout + "</pre>"
    
    read_loaded_lora_models()    
    
    yield gr.update(value=tempout)
    return tempout


# ---------------------------------
def get_loaded_lora_models_text():
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        tempout = "Loaded LoRA Adapters: " + str(len(LLSTUDIO["loaded_lora_model_adapter"])) + "\n"
        return tempout
    tempout = "Loaded LoRA Adapters: " + str(len(LLSTUDIO["loaded_lora_model_adapter"])) + "\n"
    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            tempout = tempout + "[" + str(i+1) + "]:LoRA Model Name: " + LLSTUDIO["loaded_lora_model_name"][i] + "\n\n"
            tempout = tempout + "[" + str(i+1) + "]:LoRA Model Value: " + str(LLSTUDIO["loaded_lora_model_value"][i]) + "\n\n"
    read_loaded_lora_models()    
    return tempout



# ------------------------------------------------------
def get_loaded_lora_models_html():
    tempout = "<h3>Loaded LoRA Adapters: " + str(len(LLSTUDIO["loaded_lora_model_adapter"])) + "</h3></br>"
    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        tempout = tempout + "<pre>"
        for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            tempout = tempout + "Adapter Name: " + LLSTUDIO["loaded_lora_model_adapter"][i] + "</br>"
            tempout = tempout + "LoRA Model Name: " + LLSTUDIO["loaded_lora_model_name"][i] + "</br>"
            tempout = tempout + "LoRA Model Value: " + str(LLSTUDIO["loaded_lora_model_value"][i]) + "</br>"
            tempout = tempout + "----------------------------------</br>"
        tempout = tempout + "</pre>"
    read_loaded_lora_models()    
    return tempout


# ---------------------------------
def delete_all_lora_adapters():
    global pipeline             
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
    
    tempout = ""
    
    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        tempout = "<h3>Unloading LoRA Adapters...</h3>"
        yield gr.update(value=tempout)
        if int(STUDIO["app_debug"]["value"]) > 0: print ("Unloading LoRA Adapters...")

        pipeline.unload_lora_weights() 

        adapter_names = pipeline.get_active_adapters()
        pipeline.delete_adapters(adapter_names)
          
        LLSTUDIO["loaded_lora_model_value"]=[]
        LLSTUDIO["loaded_lora_model_name"]=[]
        LLSTUDIO["loaded_lora_model_adapter"]=[]
        LLSTUDIO["lora_adapter_numb"] = 0

        if int(STUDIO["app_debug"]["value"]) > 0: print ("Finished Unloading LoRA Adapters.")
        tempout = "<h3>Finished Unloading LoRA Adapters.</h3>"
        yield gr.update(value=tempout)
    else:
        if int(STUDIO["app_debug"]["value"]) > 0: print ("No LoRA Models Loaded to Unload.")
        tempout = "<h3>No LoRA Models Loaded to Unload.</h3>"
        yield gr.update(value=tempout)

    read_loaded_lora_models()    
    return tempout




# -------------------------------------------------------------------------------
# creates full path from hub_models_dir/model_name
def get_hub_model_path_file(model_name):
    model_path_file = os.path.join(LLSTUDIO["hub_model_dir"], model_name)
    return model_path_file


# -------------------------------------------------------------------------------
# just reloads hub_model_list[] - called when app starts and to refresh hub_model_list[] items
def read_hub_model_dir():
    
    LLSTUDIO["hub_model_list"] = []
    if not os.path.isdir(LLSTUDIO["hub_model_dir"]):
        return "Huggingface Hub Cache Directoy was NOT Found.<br>You will need to Check the enviroment variable 'HF_HUB_CACHE' -OR- set the location in the LCM-LoRA Studio 'settings' in order to load model via the dropdown box."
    entries = [d for d in os.listdir(LLSTUDIO["hub_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["hub_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        if tmp_text != ".locks":
            try:
                hex40str = get_file_content(os.path.join(get_hub_model_path_file(tmp_text), "refs", "main"))
                with open(os.path.join(get_hub_model_path_file(tmp_text), "snapshots", hex40str, "model_index.json"), "r") as f:
                    model_config_data = json.load(f)

                model_class_name = model_config_data["_class_name"]

                if model_class_name == "StableDiffusionPipeline":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionXLPipeline":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionImage2Image":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionXLImage2Image":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionInpaintPipeline":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionXLInpaintPipeline":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionInstructPix2PixPipeline":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
                elif model_class_name == "StableDiffusionXLInstructPix2PixPipeline":
                    LLSTUDIO["hub_model_list"].append(tmp_text)
               
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(tmp_text + ": MODEL CONFIG NOT FOUND")
                
    return ""


# ---------------------------------
# send back an updated grDropdown to update the hub_model_list_dropdown
def update_hub_model_list_dropdown():
    read_hub_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["hub_model_list"], interactive=True)


# -------------------------------------------------------------------------------
def set_title_mode(tab_data: gr.SelectData):
    
    if tab_data.value == "Text to Image":
        my_mode = "Text to Image"
    if tab_data.value == "Image to Image":
        my_mode = "Image to Image"
    if tab_data.value == "Inpaint Image":
        my_mode = "Inpaint Image"
    if tab_data.value == "Instruct Pix2Pix":
        my_mode = "Instruct Pix2Pix"
    if tab_data.value == "SD Upscale 2x":
        my_mode = "SD Upscale 2x"
    if tab_data.value == "ControlNet":
        my_mode = "ControlNet"
    if tab_data.value == "Output Image":
        my_mode = SDPIPELINE["pipeline_gen_mode"]

    title_data = get_system_stats(0)
    
    my_title = f"<table cellspacing='1' cellpadding='1' border='0'><tr><td><b><font size='+1'>Version: {LLSTUDIO['app_version']} - Current Mode: {my_mode}</font></b><br>{title_data}</td></tr></table>"

    return gr.update(value=my_title)




# ---------------------------------------------------------
# changes to the image output tab for generation
def change_tab():
    if int(STUDIO["gen_auto_image_tab"]["value"]) == 1:
        return gr.Tabs(selected="tab_ImageGeneration"), gr.Tabs(selected="tab_iout")
    else:
        return gr.Tabs(selected=""), gr.Tabs(selected="")


# ---------------------------------
# changes to the image output tab for generation
def change_tab_cnet():
    
    # old vers
    # return gr.Tabs(selected="tab_t2i"), gr.Tabs(selected="tab_imageoutput")
    # new vers
    if int(STUDIO["gen_auto_image_tab"]["value"]) == 1:
        return gr.Tabs(selected="tab_ImageGeneration"), gr.Tabs(selected="tab_cnet")
    else:
        return gr.Tabs(selected=""), gr.Tabs(selected="")




# ---------------------------------------------------------
# exit python/gradio back to prompt
def exit_app():
    file1 = open(os.path.join(".", "restart.txt"), 'w')
    file1.write("0")
    file1.close()
    yield
    time.sleep(2)
    os._exit(os.X_OK)    


# ------------------------------------------------------
# exit python/gradio back to script/batch file LOOP for RESTART
def restart_app():
    file1 = open(os.path.join(".", "restart.txt"), 'w')
    file1.write("1")
    file1.close()
    yield
    time.sleep(2)
    os._exit(os.X_OK)    


# ------------------------------------------------------
# Run the 'shutdown' command on Linux only
def sudo_shutdown():
    if LLSTUDIO["current_os"] == "Linux":
        subprocess.run(["sudo", "shutdown", "-h", "now"])


# ------------------------------------------------------
# Run the 'reboot' command on Linux only
def sudo_reboot():
    if LLSTUDIO["current_os"] == "Linux":
        subprocess.run(["sudo", "reboot"])


# ------------------------------------------------------
def huggingface_on_app():
    file1 = open(os.path.join(".", "restart.txt"), 'w')
    file1.write("2")
    file1.close()
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    yield
    time.sleep(2)
    os._exit(os.X_OK)    
    # return "Huggingface Hub is now ON."


# ------------------------------------------------------
def huggingface_off_app():
    file1 = open(os.path.join(".", "restart.txt"), 'w')
    file1.write("3")
    file1.close()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    yield
    time.sleep(2)
    os._exit(os.X_OK)    
    # return "Huggingface Hub is now OFF."


# ------------------------------------------------------
def huggingface_check_status_app():
    hub_online = os.getenv('HF_HUB_OFFLINE', '0')
    if hub_online == '1':
        return "HuggingFace Hub is OFFLINE"
    else:
        return "HuggingFace Hub is ONLINE"


# ------------------------------------------------------
def delete_lcm_model(model_name, del_model, del_images): 

    contents = ""
    contents2 = ""
    
    if not model_name:
        contents = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        return contents, contents2

    full_lcm_model_directory = os.path.join(LLSTUDIO["lcm_model_dir"], model_name)
    full_lcm_model_images_path = os.path.join(LLSTUDIO["lcm_model_image_dir"], model_name)
    if del_model:
        if os.path.exists(full_lcm_model_directory):
            try:
                shutil.rmtree(full_lcm_model_directory)
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"LCM-LoRA Model '{model_name}', its directory and contents deleted successfully.")
                contents = f"LCM-LoRA Model '{model_name}', its directory and contents deleted successfully."
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LCM-LoRA Model '{model_name}' directory not found. Can not delete.")
                contents = f"Error: LCM-LoRA Model '{model_name}' directory not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting LCM-LoRA Model '{model_name}' directory: {e}")    
                contents = f"Error: Error deleting LCM-LoRA Model '{model_name}' directory: {e}"
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LCM-LoRA Model '{model_name}' path or directory does not exist.")
            contents = f"Error: LCM-LoRA Model '{model_name}' path or directory does not exist."
    else:
        if not del_images:
            contents = "You Must Select to Delete the Model, Gallery Images or Both."
        
    if del_images:
        if os.path.exists(full_lcm_model_images_path):
            try:
                shutil.rmtree(full_lcm_model_images_path)
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"LCM-LoRA Model '{model_name}', image gallery deleted successfully.")
                contents2 = f"LCM-LoRA Model '{model_name}', image gallery deleted successfully."
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LCM-LoRA Model '{model_name}' image gallery not found. Can not delete.")
                contents2 = f"Error: LCM-LoRA Model '{model_name}' image gallery not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting LCM-LoRA Model '{model_name}' image gallery : {e}")    
                contents2 = f"Error: Error deleting LCM-LoRA Model '{model_name}' image gallery : {e}"
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LCM-LoRA Model '{model_name}' image gallery does not exist.")
            contents2 = f"Error: LCM-LoRA Model '{model_name}' image gallery does not exist."
    else:
        if not del_model:
            contents = "You Must Select to Delete the Model, Gallery Images or Both."

    return contents, contents2
    

# ==========================================================================================
# model merge ui functions
 

# ---------------------------------
def update_merge_model_list_dropdown():
    read_lcm_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], interactive=True), gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], interactive=True)

    
# ---------------------------------
def update_profile_list_dropdown():
    read_profile_dir()
    return gr.Dropdown(choices=LLSTUDIO["profiles_list"], interactive=True)

   
# ---------------------------------
def read_profile_dir():
    LLSTUDIO["profiles_list"] = []
    entries = [f for f in os.listdir(LLSTUDIO["profiles_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["profiles_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        if tmp_text.endswith('.json'):
            tmp_model = os.path.splitext(os.path.basename(tmp_text))[0]
            LLSTUDIO["profiles_list"].append(tmp_model)

    return


# ==========================================================================================

    


# ----------------------------------------------------------
# create full path from lcm_models_dir/model_name
def get_lcm_model_path_file(model_name):
    model_path_file = os.path.join(LLSTUDIO["lcm_model_dir"], model_name)
    return model_path_file


# ---------------------------------
# just reloads lcm_model_list[] - called when app starts and to refresh lcm_model_list[] items
def read_lcm_model_dir():
    LLSTUDIO["lcm_model_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lcm_model_list"].append(tmp_text)

    return "LCM-LoRA Model List Reloaded."


# ---------------------------------
# send back an updated grDropdown to update the lcm_model_list_dropdown
def update_lcm_model_list_dropdown():
    read_lcm_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], interactive=True)


# ----------------------------------------------------------
# just reloads lcm_model_list[] - called when app starts and to refresh lcm_model_list[] items
def read_lcm_sdonly_model_dir():
    
    LLSTUDIO["lcm_sdonly_model_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        model_name = entries[i]
        try:
            with open(os.path.join(get_lcm_model_path_file(model_name), "model_index.json"), "r") as f:
                model_config_data = json.load(f)
            if model_config_data:
                model_class_name = model_config_data["_class_name"]
                if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SD15":
                    LLSTUDIO["lcm_sdonly_model_list"].append(model_name)

        except Exception as e:
            a=0 # we do nothing, we skip it. or it'll hold up entire rest of the list because of one model
            # return f"Error: 'model_index.json' File Not Found for {model_name}<br>\n"
            # but keeps app from blowing up :)

        
    return "LCM-LoRA (SD Only) Model List Reloaded."


# ---------------------------------
# send back an updated grDropdown to update the lcm_model_list_dropdown for separate text encoder
def update_lcm_sdonly_model_list_dropdown():
    read_lcm_sdonly_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_sdonly_model_list"], interactive=True)


# ----------------------------------------------------------
def get_lcm_pipeclass_model_info(model_name):
    
    myout = ""
    try:
        with open(os.path.join(get_lcm_model_path_file(model_name), "model_index.json"), "r") as f:
            model_config_data = json.load(f)
    except Exception as e:
        return f"Error: 'model_index.json' File Not Found for {model_name}<br>\n"

    if model_config_data:
        model_class_name = model_config_data["_class_name"]

        if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SD15":
            myout = myout + f"SD15 - {model_class_name} - {model_name}<br>\n"
            myout = myout + f"Separate Text Encoder and/or ControlNet Availiable for Use.<br>\n"
            
        if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SDXL":
            myout = myout + f"SDXL - {model_class_name} - {model_name}<br>\n"
            myout = myout + f"NOTE: No Separate Text Encoder or ControlNet for SDXL Models.<br>\n"
            
        if model_class_name == "StableDiffusionPipeline":
            myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
        elif model_class_name == "StableDiffusionXLPipeline":
            myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
        elif model_class_name == "StableDiffusionImage2Image":
            myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
        elif model_class_name == "StableDiffusionXLImage2Image":
            myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
        elif model_class_name == "StableDiffusionInpaintPipeline":
            myout = myout + f"Can be used for Image Inpainting.<br>\n"
        elif model_class_name == "StableDiffusionXLInpaintPipeline":
            myout = myout + f"Can be used for Image Inpainting.<br>\n"
        elif model_class_name == "StableDiffusionInstructPix2PixPipeline":
            myout = myout + f"Can be used for Instruct Pix 2 Pix.<br>\n"
        elif model_class_name == "StableDiffusionXLInstructPix2PixPipeline":
            myout = myout + f"Can be used for Instruct Pix 2 Pix.<br>\n"
        

    return myout


# ----------------------------------------------------------
def get_hub_pipeclass_model_info(model_name):
    hfcache = os.getenv('HF_HUB_CACHE', 'None')
    myout = f"HuggingFace Hub Local Cache Location: {hfcache}<br>"
    try:
        hex40str = get_file_content(os.path.join(get_hub_model_path_file(model_name), "refs", "main"))
        with open(os.path.join(get_hub_model_path_file(model_name), "snapshots", hex40str, "model_index.json"), "r") as f:
            model_config_data = json.load(f)

        if model_config_data:
            model_class_name = model_config_data["_class_name"]

            if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SD15":
                myout = myout + f"SD15 - {model_class_name} - {model_name}<br>\n"
                myout = myout + f"Separate Text Encoder and/or ControlNet Availiable for Use.<br>\n"
                
            if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SDXL":
                myout = myout + f"SDXL - {model_class_name} - {model_name}<br>\n"
                myout = myout + f"NOTE: No Separate Text Encoder or ControlNet for SDXL Models.<br>\n"
                
            if model_class_name == "StableDiffusionPipeline":
                myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
            elif model_class_name == "StableDiffusionXLPipeline":
                myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
            elif model_class_name == "StableDiffusionImage2Image":
                myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
            elif model_class_name == "StableDiffusionXLImage2Image":
                myout = myout + f"Can be used for Text to Image, Image 2 Image.<br>\n"
            elif model_class_name == "StableDiffusionInpaintPipeline":
                myout = myout + f"Can be used for Image Inpainting.<br>\n"
            elif model_class_name == "StableDiffusionXLInpaintPipeline":
                myout = myout + f"Can be used for Image Inpainting.<br>\n"
            elif model_class_name == "StableDiffusionInstructPix2PixPipeline":
                myout = myout + f"Can be used for Instruct Pix 2 Pix.<br>\n"
            elif model_class_name == "StableDiffusionXLInstructPix2PixPipeline":
                myout = myout + f"Can be used for Instruct Pix 2 Pix.<br>\n"
        else:
            myout = myout + f"'model_index.json' for '{model_name}' Contains No Valid Data or the File is Not Found"
            
    except Exception as e:
        myout = myout + f"Error: Model or Configuration Not Found."


    return myout




# ---------------------------------
# just reloads lcm_model_list[] - called when app starts and to refresh lcm_model_list[] items
def read_lcm_pipeclass_model_dir():
    
    LLSTUDIO["lcm_model_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lcm_model_list"].append(tmp_text)

    return "LCM-LoRA Model List Reloaded."


# ---------------------------------
# send back an updated grDropdown to update the lcm_model_list_dropdown
def update_lcm_pipeclass_model_list_dropdown():
    read_lcm_pipeclass_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], interactive=True)



# ---------------------------------
def save_lcm_model_edit(modelname, content):
    if modelname:
        mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
        if mdl_filename:
            try:
                with open(mdl_filename, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error loading file: {e}"
        return ""
    else:
        return "No model selected."


# ---------------------------------
def save_lcm_model_view(modelname, content):
    if modelname:
        mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
        if mdl_filename:
            try:
                with open(mdl_filename, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error loading file: {e}"
        return ""
    else:
        return "No model selected."


# ---------------------------------
def save_lcm_model_save(modelname, content):
    if modelname:
        mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
        if mdl_filename:
            try:
                with open(mdl_filename, 'w') as f:
                    f.write(content)
                return f"File '{mdl_filename}' saved successfully!"
            except Exception as e:
                return f"Error saving file: {e}"
        return "No file selected to save."
    else:
        return "No model selected."


# ---------------------------------
def save_safe_model_save(modelname, content):
    
    if modelname:
        mdl_filename = (os.path.join(LLSTUDIO["safe_model_image_dir"],modelname,modelname + '.md'))
        if mdl_filename:
            try:
                with open(mdl_filename, 'w') as f:
                    f.write(content)
                return f"File '{mdl_filename}' saved successfully!"
            except Exception as e:
                return f"Error saving file: {e}"
        return "No file selected to save."
    else:
        return "No model selected."



# ---------------------------------
def save_lora_model_save(modelname, content):
    
    if modelname:
        mdl_filename = (os.path.join(LLSTUDIO["lora_model_image_dir"],modelname,modelname + '.md'))
        if mdl_filename:
            try:
                with open(mdl_filename, 'w') as f:
                    f.write(content)
                return f"File '{mdl_filename}' saved successfully!"
            except Exception as e:
                return f"Error saving file: {e}"
        return "No file selected to save."
    else:
        return "No model selected."


# ----------------------------------------------------------
# returns full path model images dir, dir/model
def get_lcm_model_image_path_file(model_name):
    
    model_image_path_file = os.path.join(LLSTUDIO["lcm_model_image_dir"], model_name)
    return model_image_path_file



# ---------------------------------
def read_lcm_model_image_dir():
    LLSTUDIO["lcm_model_image_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_image_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_image_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lcm_model_image_list"].append(tmp_text)

    return "LCM-LoRA Model Images Reloaded."
 


# ---------------------------------
 # send back an updated grDropdown to update the lcmmodelview_dropdown    
def update_lcm_model_image_list_dropdown():
    
    read_lcm_model_image_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_image_list"], interactive=True)
   

# ----------------------------------------------------------
def delete_safe_model(model_name, del_model, del_images):   
    
    contents = ""
    contents2 = ""
    
    if not model_name:
        contents = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        return contents, contents2

    full_safemodel_file = os.path.join(LLSTUDIO["safe_model_dir"], model_name + ".safetensors")
    full_safemodel_images_path = os.path.join(LLSTUDIO["safe_model_image_dir"], model_name)
    if del_model:
        if os.path.exists(full_safemodel_file):
            try:
                os.remove(full_safemodel_file)
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Safetensors Model '{model_name}', deleted successfully.")
                contents = f"Safetensors Model '{model_name}', deleted successfully."
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Safetensors Model '{model_name}' not found. Can not delete.")
                contents = f"Error: Safetensors Model '{model_name}' not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting Safetensors Model '{model_name}' directory: {e}")    
                contents = f"Error: Error deleting Safetensors Model '{model_name}' directory: {e}"
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Safetensors Model '{model_name}' path or directory does not exist.")
            contents = f"Error: Safetensors Model '{model_name}' path or directory does not exist."
    else:
        if not del_images:
            contents = "You Must Select to Delete the Model, Gallery Images or Both."

    if del_images:
        if os.path.exists(full_safemodel_images_path):
            try:
                shutil.rmtree(full_safemodel_images_path)
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Safetensors Model '{model_name}', image gallery deleted successfully.")
                contents2 = f"Safetensors Model '{model_name}', image gallery deleted successfully."
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Safetensors Model '{model_name}' image gallery not found. Can not delete.")
                contents2 = f"Error: Safetensors Model '{model_name}' image gallery not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting Safetensors Model '{model_name}' directory: {e} image gallery ")    
                contents2 = f"Error: Error deleting Safetensors Model '{model_name}' directory: {e} image gallery "
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Safetensors Model '{model_name}' image gallery path or directory does not exist.")
            contents2 = f"Error: Safetensors Model '{model_name}' image gallery path or directory does not exist."
    else:
        if not del_model:
            contents = "You Must Select to Delete the Model, Gallery Images or Both."

    return contents, contents2
    
    

# ---------------------------------
# creates full path from lcm_models_dir/model_name
def get_safe_model_path_file(model_name):
    
    safe_model_path_file = os.path.join(LLSTUDIO["safe_model_dir"], model_name)
    return safe_model_path_file


# ---------------------------------
# used ONLY for safetensors viewer
# just reloads list[] - called when app starts and to refresh list items
def read_safe_model_dir():
    LLSTUDIO["safe_model_list"] = []
    entries = [f for f in os.listdir(LLSTUDIO["safe_model_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["safe_model_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        if tmp_text.endswith('.safetensors'):
            tmp_model = os.path.splitext(os.path.basename(tmp_text))[0]
            LLSTUDIO["safe_model_list"].append(tmp_model)
    
    return "Safetensors Model List Reloaded."


# ---------------------------------
def update_safe_model_list_dropdown():
    read_safe_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["safe_model_list"], interactive=True)
   

#rk99
# ------------------------------------------------------
def update_safeload_lmc_text_enc_dropdown():
    read_lcm_sdonly_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_sdonly_model_list"], interactive=True)
   


# ------------------------------------------------------
def update_safe_convert_model_list_dropdown():
    read_safe_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["safe_model_list"], interactive=True)
   

# ---------------------------------
# returns full path model images dir, dir/model
def get_safe_model_image_path_file(model_name):
    return os.path.join(LLSTUDIO["safe_model_image_dir"], model_name)
    

# ---------------------------------
def read_safe_model_image_dir():
    LLSTUDIO["safe_model_image_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["safe_model_image_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["safe_model_image_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["safe_model_image_list"].append(tmp_text)
    
    return "Safetensors Model Images Reloaded."



# ---------------------------------
def update_safe_model_image_list_dropdown():
    read_safe_model_image_dir()
    return gr.Dropdown(choices=LLSTUDIO["safe_model_image_list"], interactive=True)

 

# ------------------------------------------------------------------
def delete_lora_model(model_name, del_model, del_images):    

    contents = ""
    contents2 = ""
    
    if not model_name:
        contents = "<h3>No Model Name selected.<br>Please select a model from the dropdown box.<br>Refresh the dropdown box if needed.</h3>"
        return contents, contents2

    full_lora_model_file = os.path.join(LLSTUDIO["lora_model_dir"], model_name + ".safetensors")
    full_lora_model_images_path = os.path.join(LLSTUDIO["lora_model_image_dir"], model_name)
    if del_model:
        if os.path.exists(full_lora_model_file):
            try:
                os.remove(full_lora_model_file)
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"LoRA Model '{model_name}', deleted successfully.")
                contents = f"LoRA Model '{model_name}', deleted successfully."
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LoRA Model '{model_name}' not found. Can not delete.")
                contents = f"Error: LoRA Model '{model_name}' not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting LoRA Model '{model_name}': {e}")    
                contents = f"Error: Error deleting LoRA Model '{model_name}': {e}"
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LoRA Model '{model_name}' path or directory does not exist.")
            contents = f"Error: LoRA Model '{model_name}' path or directory does not exist."
    else:
        if not del_images:
            contents = "You Must Select to Delete the Model, Gallery Images or Both."

    if del_images:
        if os.path.exists(full_lora_model_images_path):
            try:
                shutil.rmtree(full_lora_model_images_path)
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"LoRA Model '{model_name}', image gallery deleted successfully.")
                contents2 = f"LoRA Model '{model_name}', image gallery deleted successfully."
            except FileNotFoundError:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LoRA Model '{model_name}' image gallery not found. Can not delete.")
                contents2 = f"Error: LoRA Model '{model_name}' image gallery not found. Can not delete."
            except Exception as e:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: Error deleting LoRA Model '{model_name}': {e} image gallery ")    
                contents2 = f"Error: Error deleting LoRA Model '{model_name}': {e} image gallery "
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print(f"Error: LoRA Model '{model_name}' image gallery path or directory does not exist.")
            contents2 = f"Error: LoRA Model '{model_name}' image gallery path or directory does not exist."
    else:
        if not del_model:
            contents = "You Must Select to Delete the Model, Gallery Images or Both."
            
    return contents, contents2
    

   

# ---------------------------------
# creates full path from lcm_models_dir/model_name
def get_lora_model_path_file(model_name):
    return os.path.join(LLSTUDIO["lora_model_dir"], model_name)



# ------------------------------------------------------------------
# just reloads list[] - called when app starts and to refresh list items
def read_lora_model_dir():
    LLSTUDIO["lora_model_list"] = []
    entries = [f for f in os.listdir(LLSTUDIO["lora_model_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["lora_model_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        if tmp_text.endswith('.safetensors'):
            tmp_model = os.path.splitext(os.path.basename(tmp_text))[0]
            LLSTUDIO["lora_model_list"].append(tmp_model)
    
    return "Lora Model List Reloaded."


 
# ---------------------------------
def update_lora_model_list_dropdown():
    read_lora_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lora_model_list"], interactive=True)



# ------------------------------------------------------
def get_lora_model_image_path_file(model_name):
    return os.path.join(LLSTUDIO["lora_model_image_dir"], model_name)



# ---------------------------------
def read_lora_model_image_dir():
    LLSTUDIO["lora_model_image_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lora_model_image_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lora_model_image_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lora_model_image_list"].append(tmp_text)
    
    return "Lora Model Images Reloaded."
 


# ---------------------------------
def update_lora_model_image_list_dropdown():
    read_lora_model_image_dir()
    return gr.Dropdown(choices=LLSTUDIO["lora_model_image_list"], interactive=True)
    
    

# ------------------------------------------------------
def read_loaded_lora_models():
    LLSTUDIO["loaded_lora_model_list"] = []
    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            LLSTUDIO["loaded_lora_model_list"].append(LLSTUDIO["loaded_lora_model_name"][i])
    
    return "Loaded Lora Models Reloaded."


# ------------------------------------------------------
def update_loaded_lora_model_list_dropdown():
    read_loaded_lora_models()
    return gr.Dropdown(choices=LLSTUDIO["loaded_lora_model_list"], interactive=True)
    

# ===================================================================
# does this part...
# (windy:1.1)
# (windy and rain:1.4)
def reformat_weighting_syntax(input_string):
    pattern = r'\((.*?):(-?\d+\.?\d*)\)'
    output_string = re.sub(pattern, r'(\1)\2', input_string)
    return output_string


# ===================================================================
# does this part...
# (windy:1.1)
# (windy and rain:1.4)
# [0|1|2|3|4]
# ("spider man", "robot mech").blend(1, 0.8)
# python function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
def remove_a1111_syntax(hidden_prompt_name, weight_input, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt):

    
    # decide which one input to use
    if (hidden_prompt_name == 't2iprompt_txt'):
        input_string=t2iprompt_txt;
    elif (hidden_prompt_name == 't2inegprompt_txt'):
        input_string=t2inegprompt_txt;
    elif (hidden_prompt_name == 'i2iprompt_txt'):
        input_string=i2iprompt_txt;
    elif (hidden_prompt_name == 'i2inegprompt_txt'):
        input_string=i2inegprompt_txt;
    elif (hidden_prompt_name == 'inpprompt_txt'):
        input_string=inpprompt_txt;
    elif (hidden_prompt_name == 'inpnegprompt_txt'):
        input_string=inpnegprompt_txt
    elif (hidden_prompt_name == 'ip2pprompt_txt'):
        input_string=ip2pprompt_txt
    elif (hidden_prompt_name == 'ip2pnegprompt_txt'):
        input_string=ip2pnegprompt_txt
    elif (hidden_prompt_name == 'up2xprompt_txt'):
        input_string=up2xprompt_txt
    elif (hidden_prompt_name == 'up2xnegprompt_txt'):
        input_string=up2xnegprompt_txt
    elif (hidden_prompt_name == 'cnetprompt_txt'):
        input_string=cnetprompt_txt
    elif (hidden_prompt_name == 'cnetnegprompt_txt'):
        input_string=cnetnegprompt_txt
   


    # does this part...first...
    # [apple|bear|candle]
    # ("apple", "bear", "candle").blend(weight_input, weight_input, weight_input)
    # really couldn't think of anything to with this this a1111 syntax ?!? :)
    def blend_replace(match_obj):
        content = match_obj.group(1)
        items = content.split('|')
        formatted_items = [f'"{item.replace("-", " ")}"' for item in items]
        items_part = f'({", ".join(formatted_items)})'
        blend_values = ", ".join([str(weight_input)] * len(items))
        blend_part = f'.blend({blend_values})'
        return items_part + blend_part

    pattern = r'\[(.*?)\]'
    # calls an internal function to do this last part of the first part...
    my_temp_out = re.sub(pattern, blend_replace, input_string)
    # calls an external function to do this part...second...
    # (windy:1.1) to (windy)1.1
    # (windy and rain:1.4) to (windy and rain)1.4
    my_temp_out2 = reformat_weighting_syntax(my_temp_out)
    
    # decide which one output to use
    if (hidden_prompt_name == 't2iprompt_txt'):
        return my_temp_out2, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 't2inegprompt_txt'):
        return t2iprompt_txt, my_temp_out2, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'i2iprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, my_temp_out2, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'i2inegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, my_temp_out2, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'inpprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, my_temp_out2, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'inpnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, my_temp_out2, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'ip2pprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, my_temp_out2, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'ip2pnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, my_temp_out2, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'up2xprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, my_temp_out2, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'up2xnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, my_temp_out2, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'cnetprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, my_temp_out2, cnetnegprompt_txt
    elif (hidden_prompt_name == 'cnetnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, my_temp_out2
   
 

# ===================================================================
# python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
def clean_compel_prompt(hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt):

    # decide which one input to use
    if (hidden_prompt_name == 't2iprompt_txt'):
        prompt=t2iprompt_txt;
    elif (hidden_prompt_name == 't2inegprompt_txt'):
        prompt=t2inegprompt_txt;
    elif (hidden_prompt_name == 'i2iprompt_txt'):
        prompt=i2iprompt_txt;
    elif (hidden_prompt_name == 'i2inegprompt_txt'):
        prompt=i2inegprompt_txt;
    elif (hidden_prompt_name == 'inpprompt_txt'):
        prompt=inpprompt_txt;
    elif (hidden_prompt_name == 'inpnegprompt_txt'):
        prompt=inpnegprompt_txt
    elif (hidden_prompt_name == 'ip2pprompt_txt'):
        prompt=ip2pprompt_txt
    elif (hidden_prompt_name == 'ip2pnegprompt_txt'):
        prompt=ip2pnegprompt_txt
    elif (hidden_prompt_name == 'up2xprompt_txt'):
        prompt=up2xprompt_txt
    elif (hidden_prompt_name == 'up2xnegprompt_txt'):
        prompt=up2xnegprompt_txt
    elif (hidden_prompt_name == 'cnetprompt_txt'):
        prompt=cnetprompt_txt
    elif (hidden_prompt_name == 'cnetnegprompt_txt'):
        prompt=cnetnegprompt_txt
   

    # Handle .and() with segments: extract inside
    #    e.g. '("part one", "part two").and(1,0.5)' -> 'part one, part two'
    # This regex finds a tuple of quoted parts before .and
    # remove .and(...) syntax first, flatten quoted segments
    def remove_and_syntax(s: str) -> str:
        pattern = re.compile(
            r'^\s*\(\s*("[^"]*"\s*(,\s*"[^"]*"\s*)*)\)\s*\.and\s*(\([^)]*\))?\s*$',
            re.DOTALL
        )
        m = pattern.match(s.strip())
        if m:
            inner = m.group(1)
            parts = re.findall(r'"([^"]*)"', inner)
            return ", ".join(parts)
        else:
            return s


    prompt = remove_and_syntax(prompt)

    # Remove numeric weights, e.g. (phrase)1.2 or phrase1.2
    prompt = re.sub(r'\(\s*([^)]+?)\)\s*\d+(\.\d+)?', r'\1', prompt)
    prompt = re.sub(r'([A-Za-z0-9_\'\"]+)\s*\d+(\.\d+)', r'\1', prompt)

    # Remove plus/minus weights: ++, -- etc
    prompt = re.sub(r'([A-Za-z0-9_\’\"\)\]]+)(\++)', r'\1', prompt)
    prompt = re.sub(r'([A-Za-z0-9_\’\"\)\]]+)(\-+)', r'\1', prompt)
    prompt = re.sub(r'\(\s*([^)]+?)\)\s*(\++)', r'\1', prompt)
    prompt = re.sub(r'\(\s*([^)]+?)\)\s*(\-+)', r'\1', prompt)

    # Remove parentheses that wrap something with no weight or +- after them
    # This will collapse nested parentheses like ((windy)) → windy
    # Using a loop until no such patterns remain.
    pattern_plain_paren = re.compile(r'\(\s*([A-Za-z0-9\,\.\s_\'\"]+?)\s*\)')
    prev = None
    while prev != prompt:
        prev = prompt
        prompt = pattern_plain_paren.sub(r'\1', prompt)

    # Clean up extra spaces, commas, redundant punctuation
    prompt = re.sub(r'\s+,', ',', prompt)
    prompt = re.sub(r',\s+', ', ', prompt)
    prompt = re.sub(r'\s{2,}', ' ', prompt)
    prompt = prompt.strip()

    
    my_temp_out2=prompt
    # decide which one output to use
    if (hidden_prompt_name == 't2iprompt_txt'):
        return my_temp_out2, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 't2inegprompt_txt'):
        return t2iprompt_txt, my_temp_out2, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'i2iprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, my_temp_out2, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'i2inegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, my_temp_out2, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'inpprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, my_temp_out2, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'inpnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, my_temp_out2, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'ip2pprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, my_temp_out2, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'ip2pnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, my_temp_out2, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'up2xprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, my_temp_out2, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'up2xnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, my_temp_out2, cnetprompt_txt, cnetnegprompt_txt
    elif (hidden_prompt_name == 'cnetprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, my_temp_out2, cnetnegprompt_txt
    elif (hidden_prompt_name == 'cnetnegprompt_txt'):
        return t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, my_temp_out2
 


# -------------------------------------------------------------    
def get_sorted_newest_image_list():
    
    output_image_list = []
    entries = [f for f in os.listdir(LLSTUDIO["advanced_gallery_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["advanced_gallery_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get image files only
        if tmp_text.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            output_image_list.append(os.path.join(LLSTUDIO["advanced_gallery_dir"], tmp_text))
    output_image_list.sort(key=os.path.getctime, reverse=True)
    return output_image_list


# ------------------------------------------------------
def get_text_content(evt: gr.SelectData):
    if evt.value is None:
        return "", None
    temp_dict = evt.value
    image_path = os.path.join(".", "output", temp_dict["image"]["orig_name"])
    LLSTUDIO["gallery_selected_image"] = image_path
    base, _ = os.path.splitext(image_path)
    text_path = f"{base}.txt"

    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read(), image_path
    else:
        return f"No Generation Parameter *.txt file found at '{text_path}'", image_path


# ------------------------------------------------------
def delete_items(selected_images):
    if not selected_images:
        return "No image was selected for deletion.", ""
    try:
        # Delete image file
        os.remove(selected_images)
        # Delete corresponding text file
        base, _ = os.path.splitext(selected_images)
        text_path = f"{base}.txt"
        if os.path.exists(text_path):
            os.remove(text_path)
    except OSError as e:
        message = f"Error deleting file {selected_images}: {e}"
        return message, ""
        
    message = f"Successfully deleted image: " + selected_images + "\nSuccessfully deleted text file: " + text_path
    return message, ""




# ===================================================================
# ===================================================================
# for creating a logo that the login screen can use since the gradio_api 
# server is technically not running/allowing connections
# once login via settings is added - future
# ===================================================================
def png_to_base64_string(filepath):
    try:
        # Open the image file in binary read mode ('rb')
        with open(filepath, "rb") as image_file:
            # Read the file's content into memory
            image_data = image_file.read()
            
            # Encode the binary data to a Base64 byte string
            base64_bytes = base64.b64encode(image_data)
            
            # Decode the Base64 byte string to a UTF-8 string for use in HTML
            base64_string = base64_bytes.decode("utf-8")
            
            return base64_string
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




# ------------------------------------------------------
# enables/disables hidden image to visible image on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
def display_generated_image():
    if LLSTUDIO["hidden_image_flag"] == 1:
        return LLSTUDIO['last_image_filename']




# ----------------------------------------
# enables/disables hidden image to visible image on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
def clear_generation_status_and_images():
    LLSTUDIO["hidden_image_flag"] = 0
    yield None, None, None, None



# ----------------------------------------
def grinfo_no_model_loaded():
    gr.Info("<h4>No Model Loaded.<br>Please Load a Model First.</h4>", duration=1.0, title="Load Model")  



# ----------------------------------------
def str_no_model_loaded():
    return "<h4>No Model Loaded. Please Load a Model First. - Select the tab 'Pipeline - Models' to load a model into the pipeline.<br>Then select from where to load your model and pipeline features.</h4>"



# ------------------------------------------------------
# .focus() event for prompt, neg_prompt gr.textboxes
def update_state(name):
    return name




# # ====================================================================================
# # ======END==========FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====
# # ====================================================================================




# ================================================================================
# =======START APP====START APP====START APP====START APP====START APP============
# ================================================================================


tstart = time.time()
pstart = time.time()


# ------------------------------------------------------------
# inference DEVICE selection for CPU ONLY !!
LLSTUDIO["device"] = "cpu"
# set what i think, a 'friendlier' device name for the user to see in the ui, all uppercase
LLSTUDIO["friendly_device_name"] = "CPU"

# ------------------------------------------------------------
# gives UI a different default/starting seed everytime you start the app
default_seed=gen_random_seed()


# ------------------------------------------------------------
# load lists of models and images for model viewer
read_hub_model_dir()
read_lcm_model_dir()
read_lcm_sdonly_model_dir()
read_lcm_model_image_dir()
read_safe_model_dir()
read_safe_model_image_dir()
read_lora_model_dir()
read_lora_model_image_dir()

# load lists of profiles for merge models ui tab
read_profile_dir()


# ------------------------------------------------------------
print("------------------------------------------")
print(LLSTUDIO["app_title"] + " - " + LLSTUDIO["app_version"])
print("------------------------------------------")

# ------------------------------------------------------------
# generate base64 logo for title screen
LLSTUDIO['llstudiologo'] = png_to_base64_string(os.path.join(".", "lcm-lora-studio-logo-main.png"))




# ------------------------------------------------------------
# gradio stuff
#

# ------------------------------------------------------------
# javascript and css code section
   


# ===================================================================
# JavaScript Functions to be called by Gradio Button .click()
# Used by the prompt helper tool (for Compel)
# ===================================================================

# ===================================================================
# input 1.2, returns "1.2" inplace of the highlighted text
# ok - used
# javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
js_modify_param_weight = """
(mystatename, my_param_weight, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt) => {

    // decide which one input to use
    if (mystatename === 't2iprompt_txt') {
      textbox_value=t2iprompt_txt;      // goes with gr.Textbox(elem_id="js_t2iprompt_txt")
    } else if (mystatename === 't2inegprompt_txt') {
      textbox_value=t2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_t2inegprompt_txt")
    } else if (mystatename === 'i2iprompt_txt') {
      textbox_value=i2iprompt_txt;      // goes with gr.Textbox(elem_id="js_i2iprompt_txt")
    } else if (mystatename === 'i2inegprompt_txt') {
      textbox_value=i2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_i2inegprompt_txt")
    } else if (mystatename === 'inpprompt_txt') {
      textbox_value=inpprompt_txt;      // goes with gr.Textbox(elem_id="js_inpprompt_txt")
    } else if (mystatename === 'inpnegprompt_txt') {
      textbox_value=inpnegprompt_txt;      // goes with gr.Textbox(elem_id="js_inpnegprompt_txt")
    } else if (mystatename === 'ip2pprompt_txt') {
      textbox_value=ip2pprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pprompt_txt")
    } else if (mystatename === 'ip2pnegprompt_txt') {
      textbox_value=ip2pnegprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pnegprompt_txt")
    } else if (mystatename === 'up2xprompt_txt') {
      textbox_value=up2xprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xprompt_txt")
    } else if (mystatename === 'up2xnegprompt_txt') {
      textbox_value=up2xnegprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xnegprompt_txt")
    } else if (mystatename === 'cnetprompt_txt') {
      textbox_value=cnetprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetprompt_txt")
    } else if (mystatename === 'cnetnegprompt_txt') {
      textbox_value=cnetnegprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetnegprompt_txt")
    }

    // process input
    const myvname = '#js_' + mystatename + ' textarea';
    const textbox = document.querySelector(myvname);
    const textbox_start = textbox.selectionStart;
    const textbox_end = textbox.selectionEnd;
    const param_weight = my_param_weight;
    const textbox_value_before_selection = textbox_value.substring(0, textbox_start);
    const textbox_value_after_selection = textbox_value.substring(textbox_end);
    const selected_textbox_value = textbox_value.substring(textbox_start, textbox_end);
    const final_text_output = textbox_value_before_selection + param_weight + textbox_value_after_selection;
    
    // decide where the one output goes, and just copy the rest
    if (mystatename === 't2iprompt_txt') {
      return [final_text_output, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 't2inegprompt_txt') {
      return [t2iprompt_txt, final_text_output, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2iprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, final_text_output, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2inegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, final_text_output, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, final_text_output, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, final_text_output, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, final_text_output, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, final_text_output, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, final_text_output, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, final_text_output, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'cnetprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, final_text_output, cnetnegprompt_txt];
    } else if (mystatename === 'cnetnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, final_text_output];
    }
    
    
}
"""





# ===================================================================
# returns "(highlighted text)" inplace of the highlighted text
# ok - used
# javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
js_add_parens = """
(mystatename, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt) => {

    // decide which one input to use
    if (mystatename === 't2iprompt_txt') {
      textbox_value=t2iprompt_txt;      // goes with gr.Textbox(elem_id="js_t2iprompt_txt")
    } else if (mystatename === 't2inegprompt_txt') {
      textbox_value=t2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_t2inegprompt_txt")
    } else if (mystatename === 'i2iprompt_txt') {
      textbox_value=i2iprompt_txt;      // goes with gr.Textbox(elem_id="js_i2iprompt_txt")
    } else if (mystatename === 'i2inegprompt_txt') {
      textbox_value=i2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_i2inegprompt_txt")
    } else if (mystatename === 'inpprompt_txt') {
      textbox_value=inpprompt_txt;      // goes with gr.Textbox(elem_id="js_inpprompt_txt")
    } else if (mystatename === 'inpnegprompt_txt') {
      textbox_value=inpnegprompt_txt;      // goes with gr.Textbox(elem_id="js_inpnegprompt_txt")
    } else if (mystatename === 'ip2pprompt_txt') {
      textbox_value=ip2pprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pprompt_txt")
    } else if (mystatename === 'ip2pnegprompt_txt') {
      textbox_value=ip2pnegprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pnegprompt_txt")
    } else if (mystatename === 'up2xprompt_txt') {
      textbox_value=up2xprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xprompt_txt")
    } else if (mystatename === 'up2xnegprompt_txt') {
      textbox_value=up2xnegprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xnegprompt_txt")
    } else if (mystatename === 'cnetprompt_txt') {
      textbox_value=cnetprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetprompt_txt")
    } else if (mystatename === 'cnetnegprompt_txt') {
      textbox_value=cnetnegprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetnegprompt_txt")
    }

    // process input
    const myvname = '#js_' + mystatename + ' textarea';
    const textbox = document.querySelector(myvname);
    const textbox_start = textbox.selectionStart;
    const textbox_end = textbox.selectionEnd;
    const textbox_value_before_selection = textbox_value.substring(0, textbox_start);
    const textbox_value_after_selection = textbox_value.substring(textbox_end);
    const selected_textbox_value = textbox_value.substring(textbox_start, textbox_end);
    const final_text_output = textbox_value_before_selection + '(' + selected_textbox_value + ')' + textbox_value_after_selection;
    
    // decide where the one output goes, and just copy the rest
    if (mystatename === 't2iprompt_txt') {
      return [final_text_output, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 't2inegprompt_txt') {
      return [t2iprompt_txt, final_text_output, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2iprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, final_text_output, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2inegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, final_text_output, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, final_text_output, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, final_text_output, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, final_text_output, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, final_text_output, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, final_text_output, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, final_text_output, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'cnetprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, final_text_output, cnetnegprompt_txt];
    } else if (mystatename === 'cnetnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, final_text_output];
    }
    
    
}
"""


# ===================================================================
# input 1.2, returns "(highlighted text)1.2" inplace of the highlighted text
# ok - used
# javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
js_add_param_weight = """
(mystatename, my_param_weight, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt) => {

    // decide which one input to use
    if (mystatename === 't2iprompt_txt') {
      textbox_value=t2iprompt_txt;      // goes with gr.Textbox(elem_id="js_t2iprompt_txt")
    } else if (mystatename === 't2inegprompt_txt') {
      textbox_value=t2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_t2inegprompt_txt")
    } else if (mystatename === 'i2iprompt_txt') {
      textbox_value=i2iprompt_txt;      // goes with gr.Textbox(elem_id="js_i2iprompt_txt")
    } else if (mystatename === 'i2inegprompt_txt') {
      textbox_value=i2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_i2inegprompt_txt")
    } else if (mystatename === 'inpprompt_txt') {
      textbox_value=inpprompt_txt;      // goes with gr.Textbox(elem_id="js_inpprompt_txt")
    } else if (mystatename === 'inpnegprompt_txt') {
      textbox_value=inpnegprompt_txt;      // goes with gr.Textbox(elem_id="js_inpnegprompt_txt")
    } else if (mystatename === 'ip2pprompt_txt') {
      textbox_value=ip2pprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pprompt_txt")
    } else if (mystatename === 'ip2pnegprompt_txt') {
      textbox_value=ip2pnegprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pnegprompt_txt")
    } else if (mystatename === 'up2xprompt_txt') {
      textbox_value=up2xprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xprompt_txt")
    } else if (mystatename === 'up2xnegprompt_txt') {
      textbox_value=up2xnegprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xnegprompt_txt")
    } else if (mystatename === 'cnetprompt_txt') {
      textbox_value=cnetprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetprompt_txt")
    } else if (mystatename === 'cnetnegprompt_txt') {
      textbox_value=cnetnegprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetnegprompt_txt")
    }

    // process input
    const myvname = '#js_' + mystatename + ' textarea';
    const textbox = document.querySelector(myvname);
    const textbox_start = textbox.selectionStart;
    const textbox_end = textbox.selectionEnd;
    const param_weight = my_param_weight;
    const textbox_value_before_selection = textbox_value.substring(0, textbox_start);
    const textbox_value_after_selection = textbox_value.substring(textbox_end);
    const selected_textbox_value = textbox_value.substring(textbox_start, textbox_end);
    const final_text_output = textbox_value_before_selection + '(' + selected_textbox_value + ')' + param_weight + textbox_value_after_selection;
    
    // decide where the one output goes, and just copy the rest
    if (mystatename === 't2iprompt_txt') {
      return [final_text_output, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 't2inegprompt_txt') {
      return [t2iprompt_txt, final_text_output, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2iprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, final_text_output, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2inegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, final_text_output, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, final_text_output, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, final_text_output, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, final_text_output, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, final_text_output, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, final_text_output, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, final_text_output, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'cnetprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, final_text_output, cnetnegprompt_txt];
    } else if (mystatename === 'cnetnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, final_text_output];
    }
    

    
}
"""

# ===================================================================
# input 2.1, returns "++" on the END of the highlighted text
# input 0.1, returns "" on the END of the highlighted text
# input -3.1, returns "---" on the END of the highlighted text
# floor and abs allows use of same gr.slider :)
# ok - used
# javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
js_add_param_addweight = """
(mystatename, my_param_addweight, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt) => {

    // decide which one input to use
    if (mystatename === 't2iprompt_txt') {
      textbox_value=t2iprompt_txt;      // goes with gr.Textbox(elem_id="js_t2iprompt_txt")
    } else if (mystatename === 't2inegprompt_txt') {
      textbox_value=t2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_t2inegprompt_txt")
    } else if (mystatename === 'i2iprompt_txt') {
      textbox_value=i2iprompt_txt;      // goes with gr.Textbox(elem_id="js_i2iprompt_txt")
    } else if (mystatename === 'i2inegprompt_txt') {
      textbox_value=i2inegprompt_txt;      // goes with gr.Textbox(elem_id="js_i2inegprompt_txt")
    } else if (mystatename === 'inpprompt_txt') {
      textbox_value=inpprompt_txt;      // goes with gr.Textbox(elem_id="js_inpprompt_txt")
    } else if (mystatename === 'inpnegprompt_txt') {
      textbox_value=inpnegprompt_txt;      // goes with gr.Textbox(elem_id="js_inpnegprompt_txt")
    } else if (mystatename === 'ip2pprompt_txt') {
      textbox_value=ip2pprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pprompt_txt")
    } else if (mystatename === 'ip2pnegprompt_txt') {
      textbox_value=ip2pnegprompt_txt;      // goes with gr.Textbox(elem_id="js_ip2pnegprompt_txt")
    } else if (mystatename === 'up2xprompt_txt') {
      textbox_value=up2xprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xprompt_txt")
    } else if (mystatename === 'up2xnegprompt_txt') {
      textbox_value=up2xnegprompt_txt;      // goes with gr.Textbox(elem_id="js_up2xnegprompt_txt")
    } else if (mystatename === 'cnetprompt_txt') {
      textbox_value=cnetprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetprompt_txt")
    } else if (mystatename === 'cnetnegprompt_txt') {
      textbox_value=cnetnegprompt_txt;      // goes with gr.Textbox(elem_id="js_cnetnegprompt_txt")
    }

    // process input
    const myvname = '#js_' + mystatename + ' textarea';
    const textbox = document.querySelector(myvname);
    const textbox_start = textbox.selectionStart;
    const textbox_end = textbox.selectionEnd;
    const textbox_value_before_selection = textbox_value.substring(0, textbox_start);
    const textbox_value_after_selection = textbox_value.substring(textbox_end);
    const selected_textbox_value = textbox_value.substring(textbox_start, textbox_end);
    let num = my_param_addweight;
    numabs = Math.abs(num);
    numabsflr = Math.floor(numabs);
    let param_addweight = "";
    if (num > 0) {
        for (let i = 0; i < numabsflr; i++) {
          param_addweight += "+";
        }
    } else if (num < 0) {
        for (let i = 0; i < numabsflr; i++) {
          param_addweight += "-";
        }
    }
    const final_text_output = textbox_value_before_selection + selected_textbox_value + param_addweight + textbox_value_after_selection;

    // decide where the one output goes, and just copy the rest
    if (mystatename === 't2iprompt_txt') {
      return [final_text_output, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 't2inegprompt_txt') {
      return [t2iprompt_txt, final_text_output, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2iprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, final_text_output, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'i2inegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, final_text_output, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, final_text_output, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'inpnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, final_text_output, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, final_text_output, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'ip2pnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, final_text_output, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, final_text_output, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'up2xnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, final_text_output, cnetprompt_txt, cnetnegprompt_txt];
    } else if (mystatename === 'cnetprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, final_text_output, cnetnegprompt_txt];
    } else if (mystatename === 'cnetnegprompt_txt') {
      return [t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, final_text_output];
    }
    

    
}
"""

# ===================================================================
# ===================================================================
# ===================================================================
css_code = """
#yellow_button {
  background-color: #D9AB0C;
  color: white;
}
#blue_button {
  background-color: blue;
  color: white;
}
#red_button {
  background-color: red;
  color: white;
}
#green_button {
  background-color: green;
  color: white;
}
#purple_button {
  background-color: purple;
  color: white;
}
#gray_button {
  background-color: gray;
  color: white;
}
#exit_button {
  background-color: darkred;
  color: white;
}
#generates_button {
  background-color: darkblue;
  color: white;
}
#generate_button {
  background-color: darkgreen;
  color: white;
}
#add_button {
  background-color: #333333;
  width: 64px;
}
#view_button {
  background-color: #333333;
  width: 64px;
}
#getsafemodeltype_button {
  background-color: #333333;
  width: 64px;
}
#converttolcmmodel_button {
  background-color: #333333;
  width: 64px;
}
#testprompt_button {
  background-color: #333333;
  width: 64px;
}
#lastprompt_button {
  background-color: #333333;
  width: 64px;
}
#loadmodel_button {
  background-color: #333333;
  width: 64px;
}
#deletemodel_button {
  background-color: #333333;
  width: 64px;
}
#reloadmodellist_button {
  background-color: #333333;
  width: 64px;
}
#icon_button {
  background-color: #333333;
  width: 64px;
  height: 64px;
}
#sendtogallery_button {
  background-color: purple;
  color: white;
}
footer {
/*    visibility: hidden */
}
#my_gallery .gallery {
    height: 500px;
}
/* #my_gallery .gallery-item {
    object-fit: contain;
} */
#no-borders table, #no-borders th, #no-borders tr, #no-borders td {
    border: none !important;
    border-collapse: collapse !important;
    border-style: none !important;
    border-color: #000000 !important;
}
.gradio-container {background-color: #111111}
/* merge model profile load/save status box */
#status-row-bg {
    background-color: #222222 !important;
    padding: 20px;
    border-radius: 8px;
}
/* Targets the label text, input text, and markdown within this specific row */
#status-row .label, 
#status-row input, 
#status-row p {
    color: #cccccc !important; /* Neon teal foreground */
}
"""



# ------------------------------------------------------------
# more gradio stuff - page HEAD tag
head_js_code = """
<script>
</script>
"""


# ===================================================================
# html for the OPENPOSE EDITOR tab, which is just one gr.HTML() component
openpose_html = """
<center>
<h3>
Open Rock's Simple OpenPose Editor in a new window (tab).<br><br>
Offline Mode !!<br><br>
<a href='/gradio_api/file/help/rkopenpose.html' target='_blank'>Rock's Simple OpenPose Editor</a>
</h3>
</center>
"""


# ===================================================================
# html for the HELP tab, which is just one gr.HTML() component

help_html = """
<center>
<h3>
Open LCM-LoRA Studio Help in a new window (tab).<br><br>
<a href='/gradio_api/file/help/index.html' target='_blank'>LCM-LoRA Studio Help</a>
</h3>
<br>
<p>
Note: The Help section is just simple HTML. I did this to enable the user to be able to annotate the help 
section for thier own purpose, make notes, reminders, etc...
</p>
</center>
"""


# ===================================================================
# html for the system model information - Accordian
# which is just one gr.HTML() component
system_model_information = """
<p>
System Models are the models used to perform certain background operations. In total there are 18 'System' models referenced in this program.<br>
Please see 'Help' - 'System Model Information' for more information on the models.<br>
Each model will automatically be downloaded only once needed. This method saves the most storage space. If you use every facet of this program you will eventually have them all.
</p>
<p>
When possible, each model is loaded into a 'pipeline', which helps ensure the complete model was correctly downloaded. In the case of the 2 LCM-LoRA models, it was easier to just do a 'snapshot_download' of the 'repo'. Very minimal overhead.
</p>
<p>
In this dialog, you can 'pre-download' known needed 'system' models. WARNING !! All Models listed here will consume approx <font size='+1'><b>105GB</b></font> total, of storage space.<br>
<b><i>The bare minimum of models needed to just generate an image using SD are already checked by default on each program start up. (Which totals 3 GB of storage space.)</i></b><br>
</p>
"""




# ===================================================================
# --- ui start --- theme
# ===================================================================

# # old theme code was not overiding color in browser, if able
# theme = gr.themes.Default(primary_hue="orange",)

theme = gr.themes.Default(primary_hue="orange").set(
    body_background_fill="*neutral_50",             # entire page
    body_background_fill_dark="*neutral_950",       # dark mode
    background_fill_primary="*neutral_100"          # main content area
)


# ------------------------------------------------------------------------------
def update_grapptitle():
    title_data = get_system_stats(0)
    grapphtml = f"<table cellspacing='1' cellpadding='1' border='0'><tr><td><b><font size='+1'>Version: {LLSTUDIO['app_version']} - Current Mode: {SDPIPELINE['pipeline_gen_mode']}</font></b><br>{title_data}</td></tr></table>"
    return grapphtml



# ------------------------------------------------------------------------------
def update_grapptitle_mem():
    title_data = get_system_stats(1)
    grapphtml = f"<table cellspacing='1' cellpadding='1' border='0'><tr><td><b><font size='+1'>Version: {LLSTUDIO['app_version']} - Current Mode: {SDPIPELINE['pipeline_gen_mode']}</font></b><br>{title_data}</td></tr></table>"
    return grapphtml



# ------------------------------------------------------------------------------------------------------------------
# GRADIO UI
# ------------------------------------------------------------------------------------------------------------------
grapptitle1 = f"<img src='data:image/png;base64,{LLSTUDIO['llstudiologo']}' alt='{LLSTUDIO['app_title']}'>"
grapptitle = update_grapptitle()

# The parameters for Gradio's gr.Blocks() 
blocks_kwargs = {}
blocks_kwargs["fill_height"] = True
blocks_kwargs["delete_cache"] = (3600, 3600)        # (frequency, age) 3600,3600 = clean up once per hour, each run delete any file 3600 secs old
blocks_kwargs["analytics_enabled"] = False          # small effort to make this app 100% offline
blocks_kwargs["title"] = LLSTUDIO["app_title"]      # web browser page/tab/window name
blocks_kwargs["theme"] = theme                      # default gradio theme, orange
blocks_kwargs["head"] = head_js_code                # did have javascript copy/paste code for prompts, but way too browser dependant
blocks_kwargs["css"] = css_code                     # css for gr.Blocks() components

with gr.Blocks(**blocks_kwargs) as lcmlorastudio:
    # hidden controls for dynamic prompts
    hidden_prompt_name = gr.Textbox(value="t2iprompt_txt", visible=False)   # default control name, gets changed, app starts on t2i tab
    hidden_t2iprompt_txt = gr.Textbox(value="t2iprompt_txt", visible=False)         # the rest of these 10 never change...
    hidden_t2inegprompt_txt = gr.Textbox(value="t2inegprompt_txt", visible=False)
    hidden_i2iprompt_txt = gr.Textbox(value="i2iprompt_txt", visible=False)
    hidden_i2inegprompt_txt = gr.Textbox(value="i2inegprompt_txt", visible=False)
    hidden_inpprompt_txt = gr.Textbox(value="inpprompt_txt", visible=False)
    hidden_inpnegprompt_txt = gr.Textbox(value="inpnegprompt_txt", visible=False)
    hidden_ip2pprompt_txt = gr.Textbox(value="ip2pprompt_txt", visible=False)
    hidden_ip2pnegprompt_txt = gr.Textbox(value="ip2pnegprompt_txt", visible=False)
    hidden_up2xprompt_txt = gr.Textbox(value="up2xprompt_txt", visible=False)
    hidden_up2xnegprompt_txt = gr.Textbox(value="up2xnegprompt_txt", visible=False)
    hidden_cnetprompt_txt = gr.Textbox(value="cnetprompt_txt", visible=False)
    hidden_cnetnegprompt_txt = gr.Textbox(value="cnetnegprompt_txt", visible=False)

    # ui start -------------
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=100): 
            app_title_label1 = gr.HTML(elem_id="no-borders", value=grapptitle1)
        with gr.Column(scale=2, min_width=100): 
            app_title_label = gr.HTML(elem_id="no-borders", value=grapptitle)
        with gr.Column(scale=0, min_width=100):
            update_cpumemswap_info = gr.Button("", icon="./icons/refresh64.png", elem_id="deletemodel_button")    
            update_cpumemswap_mem = gr.Button("", icon="./icons/view64.png", elem_id="deletemodel_button")    
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=100): 
            model_list_html = gr.HTML("<h4>No Model Loaded. Please Load a Model First. - Select the tab 'Pipeline - Models' to load a model into the pipeline.<br>Then select from where to load your model and pipeline features.</h4>")
        with gr.Column(scale=0, min_width=100):
            pipeline_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")    

# -------------------------------------------------------------------------------------------------------------

    with gr.Tabs(selected="tab_ImageGeneration") as tabs:

        with gr.Tab("Image Generation", id="tab_ImageGeneration"):

            with gr.Tabs(selected="tab_t2i") as inner_tab_ImageGeneration:

                with gr.Tab("Text to Image", id="tab_t2i"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            t2iprompt_txt = gr.Textbox(value=STUDIO["def_prompt"]["value"], label="Prompt", lines=4, elem_id="js_t2iprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            t2iprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Text to Image - Prompt Controls", open=False):    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                t2iweight_number = gr.Slider(label="Weight", value=1.0, minimum=-5.0, maximum=5.0, step=0.1)
                        with gr.Row(elem_id="icon_row"):     
                            t2iaddweight_button = gr.Button("", icon="./icons/promptpnumb64.png", elem_id="icon_button")
                            t2iaddparens_button = gr.Button("", icon="./icons/promptp64.png", elem_id="icon_button")
                            t2imodifyweight_button = gr.Button("", icon="./icons/promptnumb64.png", elem_id="icon_button")
                            t2iaddpweight_button = gr.Button("", icon="./icons/promptpm64.png", elem_id="icon_button")
                            t2iremove_a1111_syntax_button = gr.Button("", icon="./icons/prompta111164.png", elem_id="icon_button")
                            t2iclean_compel_prompt_button = gr.Button("", icon="./icons/trash64.png", elem_id="icon_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            t2inegprompt_txt = gr.Textbox(value=STUDIO["def_negprompt"]["value"], label="Negative Prompt", lines=4, elem_id="js_t2inegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            t2inegprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Generation Configuration", open=False):
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=2, min_width=100):
                                    t2igen_seedval = gr.Slider(label="Seed", value=default_seed, minimum=1, maximum=4294967294, step=1)
                                with gr.Column(scale=1, min_width=100):
                                    t2igen_sameseed_check = gr.Checkbox(label="Same Seed (Single Image)")
                                    t2igen_randomseed_button = gr.Button("Random#", scale=0)
                                    t2igen_incrementseed_check = gr.Checkbox(label="Increment")
                                    t2igen_incrementseed_amount = gr.Number(label="Amount", value=1)
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row():
                                t2igen_width = gr.Slider(label="Image Width", value=512, minimum=128, maximum=2048, step=64)
                                t2igen_height = gr.Slider(label="Image Height", value=512, minimum=128, maximum=2048, step=64)
                        with gr.Column(scale=1, min_width=100):
                            t2igen_guidance = gr.Slider(label="Guidance Scale", value=1.0, minimum=0.1, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            t2igen_inference_steps = gr.Slider(label="Inference Steps", value=4, minimum=1, maximum=50, step=1)
                        with gr.Column(scale=1, min_width=100):
                            t2igen_num_images = gr.Slider(label="Number of Output Images", value=1, minimum=1, maximum=100000, step=1)
                        with gr.Accordion("FreeU Configuration (Diffusers)", open=False):
                            with gr.Column(scale=1, min_width=100):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        t2igen_freeu_check = gr.Checkbox(label="Enable FreeU")
                                    with gr.Column(scale=1, min_width=100):
                                        t2igen_default_freeu_button = gr.Button("Load Default Values for Loaded Model", scale=1)
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        t2igen_freeu_s1 = gr.Textbox(label="FreeU - 's1' Value", info="Default: SD=0.9, SDXL=0.6", value=LLSTUDIO["freeu_sd_s1"])
                                    with gr.Column(scale=1, min_width=100):
                                        t2igen_freeu_s2 = gr.Textbox(label="FreeU - 's2' Value", info="Default: SD=0.2, SDXL=0.4", value=LLSTUDIO["freeu_sd_s2"])
                                    with gr.Column(scale=1, min_width=100):
                                        t2igen_freeu_b1 = gr.Textbox(label="FreeU - 'b1' Value", info="Default: SD=1.5, SDXL=1.1", value=LLSTUDIO["freeu_sd_b1"])
                                    with gr.Column(scale=1, min_width=100):
                                        t2igen_freeu_b2 = gr.Textbox(label="FreeU - 'b2' Value", info="Default: SD=1.6, SDXL=1.2", value=LLSTUDIO["freeu_sd_b2"])
                    with gr.Row():
                        t2igen_generate_button = gr.Button("Generate", scale=2, elem_id="generate_button")
                        t2igen_halt_gen_button = gr.Button("🛑", scale=1, elem_id="gray_button")


                with gr.Tab("Image to Image", id="tab_i2i"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=100):
                            i2iimage = gr.Image(label="Input Image", type="pil")
                        with gr.Column(scale=0, min_width=100):
                            i2igen_resize_input_image_check = gr.Checkbox(label="Resize")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            i2iprompt_txt = gr.Textbox(value=STUDIO["def_prompt"]["value"], label="Prompt", lines=4, elem_id="js_i2iprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            i2iprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Image to Image - Prompt Controls", open=False):    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                i2iweight_number = gr.Slider(label="Weight", value=1.0, minimum=-5.0, maximum=5.0, step=0.1)
                        with gr.Row(elem_id="icon_row"):     
                            i2iaddweight_button = gr.Button("", icon="./icons/promptpnumb64.png", elem_id="icon_button")
                            i2iaddparens_button = gr.Button("", icon="./icons/promptp64.png", elem_id="icon_button")
                            i2imodifyweight_button = gr.Button("", icon="./icons/promptnumb64.png", elem_id="icon_button")
                            i2iaddpweight_button = gr.Button("", icon="./icons/promptpm64.png", elem_id="icon_button")
                            i2iremove_a1111_syntax_button = gr.Button("", icon="./icons/prompta111164.png", elem_id="icon_button")
                            i2iclean_compel_prompt_button = gr.Button("", icon="./icons/trash64.png", elem_id="icon_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            i2inegprompt_txt = gr.Textbox(value=STUDIO["def_negprompt"]["value"], label="Negative Prompt", lines=4, elem_id="js_i2inegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            i2inegprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Generation Configuration", open=False):
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=2, min_width=100):
                                    i2igen_seedval = gr.Slider(label="Seed", value=default_seed, minimum=1, maximum=4294967294, step=1)
                                with gr.Column(scale=1, min_width=100):
                                    i2igen_randomseed_button = gr.Button("Random#", scale=0)
                                    i2igen_incrementseed_check = gr.Checkbox(label="Increment")
                                    i2igen_incrementseed_amount = gr.Number(label="Amount", value=1)
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row():
                                i2igen_width = gr.Slider(label="Image Width", value=512, minimum=128, maximum=2048, step=64)
                                i2igen_height = gr.Slider(label="Image Height", value=512, minimum=128, maximum=2048, step=64)
                        with gr.Column(scale=1, min_width=100):
                            i2igen_guidance = gr.Slider(label="Guidance Scale", value=1.0, minimum=0.1, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            i2igen_strength = gr.Slider(label="Strength", value=0.80, minimum=0.00, maximum=1.00, step=0.01)
                        with gr.Column(scale=1, min_width=100):
                            i2igen_inference_steps = gr.Slider(label="Inference Steps", value=4, minimum=1, maximum=50, step=1)
                        with gr.Column(scale=1, min_width=100):
                            i2igen_num_images = gr.Slider(label="Number of Output Images", value=1, minimum=1, maximum=100000, step=1)
                        with gr.Accordion("FreeU Configuration (Diffusers)", open=False):
                            with gr.Column(scale=1, min_width=100):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        i2igen_freeu_check = gr.Checkbox(label="Enable FreeU")
                                    with gr.Column(scale=1, min_width=100):
                                        i2igen_default_freeu_button = gr.Button("Load Default Values for Loaded Model", scale=1)
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        i2igen_freeu_s1 = gr.Textbox(label="FreeU - 's1' Value", info="Default: SD=0.9, SDXL=0.6", value=LLSTUDIO["freeu_sd_s1"])
                                    with gr.Column(scale=1, min_width=100):
                                        i2igen_freeu_s2 = gr.Textbox(label="FreeU - 's2' Value", info="Default: SD=0.2, SDXL=0.4", value=LLSTUDIO["freeu_sd_s2"])
                                    with gr.Column(scale=1, min_width=100):
                                        i2igen_freeu_b1 = gr.Textbox(label="FreeU - 'b1' Value", info="Default: SD=1.5, SDXL=1.1", value=LLSTUDIO["freeu_sd_b1"])
                                    with gr.Column(scale=1, min_width=100):
                                        i2igen_freeu_b2 = gr.Textbox(label="FreeU - 'b2' Value", info="Default: SD=1.6, SDXL=1.2", value=LLSTUDIO["freeu_sd_b2"])
                    with gr.Row():
                        i2igen_generate_button = gr.Button("Generate", scale=2, elem_id="generate_button")
                        i2igen_halt_gen_button = gr.Button("🛑", scale=1, elem_id="gray_button")


                with gr.Tab("Inpaint Image", id="tab_inp"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=100):
                            inpimage = gr.Image(label="Input Image", type="pil")
                        with gr.Column(scale=0, min_width=100):
                            inpgen_resize_input_image_check = gr.Checkbox(label="Resize")
                        with gr.Column(scale=2, min_width=100):
                            inpimagemask = gr.Image(label="Mask Image", type="pil")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            inpprompt_txt = gr.Textbox(value=STUDIO["def_prompt"]["value"], label="Prompt", lines=4, elem_id="js_inpprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            inpprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Inpaint Image - Prompt Controls", open=False):    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                inpweight_number = gr.Slider(label="Weight", value=1.0, minimum=-5.0, maximum=5.0, step=0.1)
                        with gr.Row(elem_id="icon_row"):     
                            inpaddweight_button = gr.Button("", icon="./icons/promptpnumb64.png", elem_id="icon_button")
                            inpaddparens_button = gr.Button("", icon="./icons/promptp64.png", elem_id="icon_button")
                            inpmodifyweight_button = gr.Button("", icon="./icons/promptnumb64.png", elem_id="icon_button")
                            inpaddpweight_button = gr.Button("", icon="./icons/promptpm64.png", elem_id="icon_button")
                            inpremove_a1111_syntax_button = gr.Button("", icon="./icons/prompta111164.png", elem_id="icon_button")
                            inpclean_compel_prompt_button = gr.Button("", icon="./icons/trash64.png", elem_id="icon_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            inpnegprompt_txt = gr.Textbox(value=STUDIO["def_negprompt"]["value"], label="Negative Prompt", lines=4, elem_id="js_inpnegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            inpnegprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Generation Configuration", open=False):
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=2, min_width=100):
                                    inpgen_seedval = gr.Slider(label="Seed", value=default_seed, minimum=1, maximum=4294967294, step=1)
                                with gr.Column(scale=1, min_width=100):
                                    inpgen_randomseed_button = gr.Button("Random#", scale=0)
                                    inpgen_incrementseed_check = gr.Checkbox(label="Increment")
                                    inpgen_incrementseed_amount = gr.Number(label="Amount", value=1)
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row():
                                inpgen_width = gr.Slider(label="Image Width", value=512, minimum=128, maximum=2048, step=64)
                                inpgen_height = gr.Slider(label="Image Height", value=512, minimum=128, maximum=2048, step=64)
                        with gr.Column(scale=1, min_width=100):
                            inpgen_guidance = gr.Slider(label="Guidance Scale", value=1.0, minimum=0.1, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            inpgen_strength = gr.Slider(label="Strength", value=0.80, minimum=0.00, maximum=1.00, step=0.01)
                        with gr.Column(scale=1, min_width=100):
                            inpgen_inference_steps = gr.Slider(label="Inference Steps", value=4, minimum=1, maximum=50, step=1)
                        with gr.Column(scale=1, min_width=100):
                            inpgen_num_images = gr.Slider(label="Number of Output Images", value=1, minimum=1, maximum=100000, step=1)
                        with gr.Accordion("FreeU Configuration (Diffusers)", open=False):
                            with gr.Column(scale=1, min_width=100):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        inpgen_freeu_check = gr.Checkbox(label="Enable FreeU")
                                    with gr.Column(scale=1, min_width=100):
                                        inpgen_default_freeu_button = gr.Button("Load Default Values for Loaded Model", scale=1)
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        inpgen_freeu_s1 = gr.Textbox(label="FreeU - 's1' Value", info="Default: SD=0.9, SDXL=0.6", value=LLSTUDIO["freeu_sd_s1"])
                                    with gr.Column(scale=1, min_width=100):
                                        inpgen_freeu_s2 = gr.Textbox(label="FreeU - 's2' Value", info="Default: SD=0.2, SDXL=0.4", value=LLSTUDIO["freeu_sd_s2"])
                                    with gr.Column(scale=1, min_width=100):
                                        inpgen_freeu_b1 = gr.Textbox(label="FreeU - 'b1' Value", info="Default: SD=1.5, SDXL=1.1", value=LLSTUDIO["freeu_sd_b1"])
                                    with gr.Column(scale=1, min_width=100):
                                        inpgen_freeu_b2 = gr.Textbox(label="FreeU - 'b2' Value", info="Default: SD=1.6, SDXL=1.2", value=LLSTUDIO["freeu_sd_b2"])
                    with gr.Row():
                        inpgen_generate_button = gr.Button("Generate", scale=2, elem_id="generate_button")
                        inpgen_halt_gen_button = gr.Button("🛑", scale=1, elem_id="gray_button")

                with gr.Tab("Instruct Pix2Pix", id="tab_ip2p"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=100):
                            ip2pimage = gr.Image(label="Input Image", type="pil")
                        with gr.Column(scale=0, min_width=100):
                            ip2pgen_resize_input_image_check = gr.Checkbox(label="Resize")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            ip2pprompt_txt = gr.Textbox(value=STUDIO["def_prompt"]["value"], label="Prompt", lines=4, elem_id="js_ip2pprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            ip2pprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Instruct Pix2Pix - Prompt Controls", open=False):    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                ip2pweight_number = gr.Slider(label="Weight", value=1.0, minimum=-5.0, maximum=5.0, step=0.1)
                        with gr.Row(elem_id="icon_row"):     
                            ip2paddweight_button = gr.Button("", icon="./icons/promptpnumb64.png", elem_id="icon_button")
                            ip2paddparens_button = gr.Button("", icon="./icons/promptp64.png", elem_id="icon_button")
                            ip2pmodifyweight_button = gr.Button("", icon="./icons/promptnumb64.png", elem_id="icon_button")
                            ip2paddpweight_button = gr.Button("", icon="./icons/promptpm64.png", elem_id="icon_button")
                            ip2premove_a1111_syntax_button = gr.Button("", icon="./icons/prompta111164.png", elem_id="icon_button")
                            ip2pclean_compel_prompt_button = gr.Button("", icon="./icons/trash64.png", elem_id="icon_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            ip2pnegprompt_txt = gr.Textbox(value=STUDIO["def_negprompt"]["value"], label="Negative Prompt", lines=4, elem_id="js_ip2pnegprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            ip2pnegprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Generation Configuration", open=False):
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=2, min_width=100):
                                    ip2pgen_seedval = gr.Slider(label="Seed", value=default_seed, minimum=1, maximum=4294967294, step=1)
                                with gr.Column(scale=1, min_width=100):
                                    ip2pgen_randomseed_button = gr.Button("Random#", scale=0)
                                    ip2pgen_incrementseed_check = gr.Checkbox(label="Increment")
                                    ip2pgen_incrementseed_amount = gr.Number(label="Amount", value=1)
                        with gr.Column(scale=1, min_width=100):
                            ip2pgen_guidance = gr.Slider(label="Guidance Scale", value=2.0, minimum=0.1, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            ip2pgen_imgguidance = gr.Slider(label="Image Guidance Scale", value=1.0, minimum=0.1, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            ip2pgen_inference_steps = gr.Slider(label="Inference Steps", value=4, minimum=1, maximum=50, step=1)
                        with gr.Column(scale=1, min_width=100):
                            ip2pgen_num_images = gr.Slider(label="Number of Output Images", value=1, minimum=1, maximum=100000, step=1)
                        with gr.Accordion("FreeU Configuration (Diffusers)", open=False):
                            with gr.Column(scale=1, min_width=100):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        ip2pgen_freeu_check = gr.Checkbox(label="Enable FreeU")
                                    with gr.Column(scale=1, min_width=100):
                                        ip2pgen_default_freeu_button = gr.Button("Load Default Values for Loaded Model", scale=1)
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        ip2pgen_freeu_s1 = gr.Textbox(label="FreeU - 's1' Value", info="Default: SD=0.9, SDXL=0.6", value=LLSTUDIO["freeu_sd_s1"])
                                    with gr.Column(scale=1, min_width=100):
                                        ip2pgen_freeu_s2 = gr.Textbox(label="FreeU - 's2' Value", info="Default: SD=0.2, SDXL=0.4", value=LLSTUDIO["freeu_sd_s2"])
                                    with gr.Column(scale=1, min_width=100):
                                        ip2pgen_freeu_b1 = gr.Textbox(label="FreeU - 'b1' Value", info="Default: SD=1.5, SDXL=1.1", value=LLSTUDIO["freeu_sd_b1"])
                                    with gr.Column(scale=1, min_width=100):
                                        ip2pgen_freeu_b2 = gr.Textbox(label="FreeU - 'b2' Value", info="Default: SD=1.6, SDXL=1.2", value=LLSTUDIO["freeu_sd_b2"])
                    with gr.Row():
                        ip2pgen_generate_button = gr.Button("Generate", scale=2, elem_id="generate_button")
                        ip2pgen_halt_gen_button = gr.Button("🛑", scale=1, elem_id="gray_button")


                with gr.Tab("SD Upscale 2x", id="tab_up2"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=100):
                            up2ximage = gr.Image(label="Input Image", type="pil")
                        with gr.Column(scale=0, min_width=100):
                            up2xgen_resize_input_image_check = gr.Checkbox(label="Resize")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            up2xprompt_txt = gr.Textbox(value=STUDIO["def_prompt"]["value"], label="Prompt", lines=4, elem_id="js_up2xprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            up2xprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            up2xnegprompt_txt = gr.Textbox(value=STUDIO["def_negprompt"]["value"], label="Negative Prompt", lines=4, elem_id="js_up2xnegprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            up2xnegprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Generation Configuration", open=False):
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=2, min_width=100):
                                    up2xgen_seedval = gr.Slider(label="Seed", value=default_seed, minimum=1, maximum=4294967294, step=1)
                                with gr.Column(scale=1, min_width=100):
                                    up2xgen_randomseed_button = gr.Button("Random#", scale=0)
                                    up2xgen_incrementseed_check = gr.Checkbox(label="Increment")
                                    up2xgen_incrementseed_amount = gr.Number(label="Amount", value=1)
                        with gr.Column(scale=1, min_width=100):
                            up2xgen_guidance = gr.Slider(label="Guidance Scale", value=0.0, minimum=0.0, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            up2xgen_inference_steps = gr.Slider(label="Inference Steps", value=20, minimum=1, maximum=50, step=1)
                        with gr.Accordion("FreeU Configuration (Diffusers)", open=False):
                            with gr.Column(scale=1, min_width=100):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        up2xgen_freeu_check = gr.Checkbox(label="Enable FreeU")
                                    with gr.Column(scale=1, min_width=100):
                                        up2xgen_default_freeu_button = gr.Button("Load Default Values for Loaded Model", scale=1)
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        up2xgen_freeu_s1 = gr.Textbox(label="FreeU - 's1' Value", info="Default: SD=0.9, SDXL=0.6", value=LLSTUDIO["freeu_sd_s1"])
                                    with gr.Column(scale=1, min_width=100):
                                        up2xgen_freeu_s2 = gr.Textbox(label="FreeU - 's2' Value", info="Default: SD=0.2, SDXL=0.4", value=LLSTUDIO["freeu_sd_s2"])
                                    with gr.Column(scale=1, min_width=100):
                                        up2xgen_freeu_b1 = gr.Textbox(label="FreeU - 'b1' Value", info="Default: SD=1.5, SDXL=1.1", value=LLSTUDIO["freeu_sd_b1"])
                                    with gr.Column(scale=1, min_width=100):
                                        up2xgen_freeu_b2 = gr.Textbox(label="FreeU - 'b2' Value", info="Default: SD=1.6, SDXL=1.2", value=LLSTUDIO["freeu_sd_b2"])
                    with gr.Row():
                        up2xgen_generate_button = gr.Button("Upscale 2X", scale=2, elem_id="generate_button")



                with gr.Tab("ControlNet", id="tab_cnet"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=100):
                            cnetimage = gr.Image(label="ControlNet 1 Input Image", type="pil", interactive=True, show_fullscreen_button=True)
                        with gr.Column(scale=0, min_width=100):
                            cnetgen_resize_input_image = gr.Checkbox(label="Resize Image 1")
                            cnetgen_resize_input_image2 = gr.Checkbox(label="Resize Image 2")
                        with gr.Column(scale=2, min_width=100):
                            cnetimage2 = gr.Image(label="ControlNet 2 Input Image", type="pil", interactive=True, show_fullscreen_button=True)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            cnetprompt_txt = gr.Textbox(value=STUDIO["def_prompt"]["value"], label="Prompt", lines=4, elem_id="js_cnetprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            cnetprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("ControlNet - Prompt Controls", open=False):    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                cnetweight_number = gr.Slider(label="Weight", value=1.0, minimum=-5.0, maximum=5.0, step=0.1)
                        with gr.Row(elem_id="icon_row"):     
                            cnetaddweight_button = gr.Button("", icon="./icons/promptpnumb64.png", elem_id="icon_button")
                            cnetaddparens_button = gr.Button("", icon="./icons/promptp64.png", elem_id="icon_button")
                            cnetmodifyweight_button = gr.Button("", icon="./icons/promptnumb64.png", elem_id="icon_button")
                            cnetaddpweight_button = gr.Button("", icon="./icons/promptpm64.png", elem_id="icon_button")
                            cnetremove_a1111_syntax_button = gr.Button("", icon="./icons/prompta111164.png", elem_id="icon_button")
                            cnetclean_compel_prompt_button = gr.Button("", icon="./icons/trash64.png", elem_id="icon_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            cnetnegprompt_txt = gr.Textbox(value=STUDIO["def_negprompt"]["value"], label="Negative Prompt", lines=4, elem_id="js_cnetnegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            cnetnegprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Accordion("Generation Configuration", open=False):
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=2, min_width=100):
                                    cnetgen_seedval = gr.Slider(label="Seed", value=default_seed, minimum=1, maximum=4294967294, step=1)
                                with gr.Column(scale=1, min_width=100):
                                    cnetgen_randomseed_button = gr.Button("Random#", scale=0)
                                    cnetgen_incrementseed_check = gr.Checkbox(label="Increment")
                                    cnetgen_incrementseed_amount = gr.Number(label="Amount", value=1)
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row():
                                cnetgen_width = gr.Slider(label="Image Width", value=512, minimum=128, maximum=2048, step=64)
                                cnetgen_height = gr.Slider(label="Image Height", value=512, minimum=128, maximum=2048, step=64)
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_guidance = gr.Slider(label="Guidance Scale", value=1.0, minimum=0.1, maximum=30, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_guidance_start = gr.Slider(label="ControlNet Guidance Start", value=0.00, minimum=0.00, maximum=1.00, step=0.01, info="The percentage of total steps at which the ControlNet starts applying. (0-100%, ie... 0.00 to 1.00)<br>(Default: 0.00)")
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_guidance_end = gr.Slider(label="ControlNet Guidance End", value=1.00, minimum=0.00, maximum=1.00, step=0.01, info="The percentage of total steps at which the ControlNet stops applying. (0-100%, ie... 0.00 to 1.00)<br>(Default: 1.00)")
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_conditioningguidance = gr.Slider(label="ControlNet 1 Conditioning Scale", value=1.00, minimum=0.00, maximum=1.00, step=0.01, info="The outputs of the ControlNet 1 are multiplied by this value before they are added to the residual in the original unet.<br>(Default: 1.00)")
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_conditioningguidance2 = gr.Slider(label="ControlNet 2 Conditioning Scale", value=1.00, minimum=0.00, maximum=1.00, step=0.01, info="The outputs of the ControlNet 2 are multiplied by this value before they are added to the residual in the original unet.<br>(Default: 1.00)")
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_use_guess_mode = gr.Checkbox(label="Use Guess Mode", info="The ControlNet encoder tries to recognize the content of the input image even if you remove all prompts.<br>(A guidance_scale value between 3.0 and 5.0 is recommended.)")
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_inference_steps = gr.Slider(label="Inference Steps", value=4, minimum=1, maximum=50, step=1)
                        with gr.Column(scale=1, min_width=100):
                            cnetgen_num_images = gr.Slider(label="Number of Output Images", value=1, minimum=1, maximum=100000, step=1)
                        with gr.Accordion("FreeU Configuration (Diffusers)", open=False):
                            with gr.Column(scale=1, min_width=100):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        cnetgen_freeu_check = gr.Checkbox(label="Enable FreeU")
                                    with gr.Column(scale=1, min_width=100):
                                        cnetgen_default_freeu_button = gr.Button("Load Default Values for Loaded Model", scale=1)
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1, min_width=100):
                                        cnetgen_freeu_s1 = gr.Textbox(label="FreeU - 's1' Value", info="Default: SD=0.9, SDXL=0.6", value=LLSTUDIO["freeu_sd_s1"])
                                    with gr.Column(scale=1, min_width=100):
                                        cnetgen_freeu_s2 = gr.Textbox(label="FreeU - 's2' Value", info="Default: SD=0.2, SDXL=0.4", value=LLSTUDIO["freeu_sd_s2"])
                                    with gr.Column(scale=1, min_width=100):
                                        cnetgen_freeu_b1 = gr.Textbox(label="FreeU - 'b1' Value", info="Default: SD=1.5, SDXL=1.1", value=LLSTUDIO["freeu_sd_b1"])
                                    with gr.Column(scale=1, min_width=100):
                                        cnetgen_freeu_b2 = gr.Textbox(label="FreeU - 'b2' Value", info="Default: SD=1.6, SDXL=1.2", value=LLSTUDIO["freeu_sd_b2"])
                    with gr.Row():
                        cnetgen_generate_button = gr.Button("Generate", scale=2, elem_id="generate_button")
                        cnetgen_halt_gen_button = gr.Button("🛑", scale=1, elem_id="gray_button")



                with gr.Tab("Output Image", id="tab_iout"):
                    with gr.Row():
                        with gr.Column(scale=0, min_width=75):
                            outputimage_halt_gen_button = gr.Button("🛑", scale=0, elem_id="gray_button")
                        with gr.Column(scale=0, min_width=150):
                            send_to_gallery_button = gr.Button("Send to Gallery", scale=0, elem_id="sendtogallery_button")
                        with gr.Column(scale=2, min_width=100):
                            inference_status_markdown = gr.Markdown("# Ready", min_height=50)
                    with gr.Row():
                        gallery_html = gr.HTML("")
                    with gr.Row():
                        oimage2 = gr.Image(type="pil", show_label=False)
                        oimage = gr.Image(type="pil", visible=False)
                    
                    
                    
        with gr.Tab("Pipeline - Models", id="tab_PipelineModels"):
            with gr.Tabs(selected="tab_lml") as inner_tab_PipelineModels:
                with gr.Tab("LCM-LoRA Model List", id="tab_lml"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_list_html = gr.HTML("Load saved LCM-LoRA type models here. Auto loads correct pipeline based on model config file. Can select a different Text Encoder from another LCM-LoRA model (SD Only). (NOTE: LCM-LoRA Models that show up in the dropdown have been saved with the 'Save Model' operation, and 'normally with' the LCM-LoRA weights fused.)")
                            gr.Markdown("<br>")
                            lcm_model_list_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], label="Availiable LCM-LoRA Models (Local Saved LCM-LoRA Models - 'Diffusers' Directory Format)")
                            with gr.Row(equal_height=True):
                                load_lcm_model_fp16_check = gr.Checkbox(value=1, label="variant fp16")
                        with gr.Column(scale=0, min_width=100):
                            lcm_model_reload_list_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            lcm_model_info_button = gr.Button("", icon="./icons/about64.png", elem_id="reloadmodellist_button")
                            lcm_model_load_model_button = gr.Button("", icon="./icons/load64.png", elem_id="loadmodel_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            load_lcm_model_lora_value = gr.Slider(label="LCM-LoRA Scale", value=1.0, minimum=0.1, maximum=10, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            load_lcm_model_add_lcmlora = gr.Checkbox(label="Auto Add LCM-LoRA Weights")
                            load_lcm_model_use_lcmscheduler = gr.Checkbox(value=0, label="Use LCMScheduler Only - No Weights (For LCM-LoRA 'baked-in' Models. Very Rare Usage Cases.)")
                    with gr.Row(equal_height=True):
                        gr.Markdown("Use a separate text encoder for image variations from same model (SD Only)")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_use_diff_text_encoder_check = gr.Checkbox(label="Use Separate Text Encoder")
                            lcm_model_liste_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_sdonly_model_list"], label="Availiable LCM-LoRA Models to load Separate Text Encoder (SD Only)")
                            load_lcm_modele_fp16_check = gr.Checkbox(value=1,label="variant fp16")
                        with gr.Column(scale=0, min_width=100):
                            lcm_model_reload_liste_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                    with gr.Row(equal_height=True):
                        gr.Markdown("You can use just one ControlNet *or use two at the same time* :)")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_use_controlnet = gr.Checkbox(label="Use ControlNet 1")
                            lcm_model_cnet_dropdown = gr.Dropdown(choices=LLSTUDIO["cnet_model_name_list"], label="Availiable ControlNet Models to Load. (SD Only)")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_use_controlnet2 = gr.Checkbox(label="Use ControlNet 2")
                            lcm_model_cnet_dropdown2 = gr.Dropdown(choices=LLSTUDIO["cnet_model_name_list"], label="Availiable ControlNet Models to Load. (SD Only)")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_clipskip = gr.Number(label="ClipSkip", value=int(STUDIO["default_clip_skip"]["value"]), minimum=0, maximum=12, step=1, info="When using a sperate text encoder, you can use ClipSkip. Used to control the number of layers to be skipped from CLIP while computing the prompt embeddings.<br>A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.<br>A value of 0 means that clip_skip is NOT used.<br>Alot of models suggest a ClipSkip value of '2', however consult your model card, or model info.<br>Note: Does not work on SDXL models, SD Only.")



                with gr.Tab("Huggingface (Local Cached) Model List", id="tab_hcm"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            hub_model_list_html = gr.HTML("Load Huggingface (Local Cached) Models here. Only SD/SDXL Image Generation Pipelines, all others (LLMs, etc...) filtered out. Auto loads correct pipeline based on model config file.")
                            hub_model_list_dropdown = gr.Dropdown(choices=LLSTUDIO["hub_model_list"], label="Availiable Hub Cached Models (Local)")
                            hub_model_fp16_check = gr.Checkbox(value=1,label="variant fp16")
                        with gr.Column(scale=0, min_width=100):
                            hub_model_reload_list_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            hub_model_info_button = gr.Button("", icon="./icons/about64.png", elem_id="reloadmodellist_button")
                            hub_model_load_model_button = gr.Button("", icon="./icons/load64.png", elem_id="loadmodel_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            hub_model_lora = gr.Slider(label="LCM-LoRA Lora Scale", value=1.0, minimum=0.1, maximum=10, step=0.1)
                        with gr.Column(scale=1, min_width=100):
                            hub_model_add_lcmlora = gr.Checkbox(label="Auto Add LCM-LoRA Weights")
                            hub_model_model_use_lcmscheduler = gr.Checkbox(value=0, label="Use LCMScheduler Only - No Weights")

                with gr.Tab("Huggingface Model", id="tab_hm"):
                    with gr.Row(equal_height=True):
                        hub_mark2 = gr.Markdown("### Download Model to Pipeline")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            hug_model_list_html = gr.HTML("Load Huggingface Models here. Select correct Pipeline from dropdown before loading model.")
                            hug_model_txt = gr.Textbox(info="Huggingface Model Name. Ex: stable-diffusion-v1-5/stable-diffusion-v1-5 \n(Note: Model will be loaded from your cache, or dowloaded if you do not have it. \nIf you have it, you can just select it using the Model List, under the 'Huggingface (Local Cached) Model List' tab.)", label="Load Model Name", lines=1, show_label=True, show_copy_button=True)
                            hug_model_fp16_check = gr.Checkbox(value=1,label="variant fp16")
                        with gr.Column(scale=0, min_width=100):
                            hug_model_download_model_button = gr.Button("", icon="./icons/load64.png", elem_id="loadmodel_button")
                    with gr.Row(equal_height=True):
                        hug_pipeline_classes = gr.Dropdown(choices=PIPELINE_CLASSES, label="You MUST select the correct Pipeline Class to load model", value=PIPELINE_CLASSES[0])
                    with gr.Row(equal_height=True):
                        hub_mark2 = gr.Markdown("### Download Model to Huggingface Models Cache")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            hug_download_model_txt = gr.Textbox(info="Huggingface Model Name. Ex: stable-diffusion-v1-5/stable-diffusion-v1-5 (Note: The entire model repository will be downloaded to your cache. This is a waste of space if you do not need the whole model repository. Use above option for just the parts, precision, etc... that your pipeline needs. But if you do use this method, the Model will NOT be loaded into the pipeline.)<br>However, After the download completes you can navigate to the 'Huggingface (Local Cached) Models List' tab, Refresh the model list and select it from the list of models in the Huggingface cache. Then load it. ", label="Download Model Name", lines=1, show_label=True, show_copy_button=True)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=0, min_width=100):
                            hug_downloadmodel_button = gr.Button("", icon="./icons/download64.png", elem_id="reloadmodellist_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            hug_downloadmodel_html2 = gr.HTML("")
                            hug_downloadmodel_html = gr.HTML("")


                with gr.Tab("Safetensors Model List", id="tab_sml"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            safeload_model_list_html = gr.HTML("Load Safetensors Models here. Select correct Pipeline from dropdown before loading model.")
                            safeload_model_dropdown = gr.Dropdown(choices=LLSTUDIO["safe_model_list"], label="Availiable Safetensors Models (Local)")
                        with gr.Column(scale=0, min_width=100):
                            safeload_model_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            safeload_model_load_button = gr.Button("", icon="./icons/load64.png", elem_id="converttolcmmodel_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            safeload_pipeline_classes = gr.Dropdown(choices=PIPELINE_CLASSES, label="You MUST select the correct Pipeline Class to load model", value=PIPELINE_CLASSES[0])
                        with gr.Column(scale=1, min_width=100):
                            safeload_model_add_lcmlora = gr.Checkbox(label="Auto Add LCM-LoRA Weights")
                            safeload_model_use_lcmscheduler = gr.Checkbox(label="Use LCMScheduler Only - No Weights (For LCM-LoRA 'baked-in' Models)")
                    with gr.Row(equal_height=True):
                        safeload_model_lora = gr.Slider(label="LCM-LoRA Scale", value=1.0, minimum=0.1, maximum=10, step=0.1)
                    with gr.Row(equal_height=True):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2, min_width=100):
                                safeload_use_text_enc = gr.Checkbox(label="Use Separate Text Encoder (Use for Safetensors Models that do not have one, SD Only)")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            safeload_lmc_text_enc_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_sdonly_model_list"], label="Availiable LCM-LoRA Models to load Separate Text Encoder (SD Only)")
                        with gr.Column(scale=1, min_width=100):
                            safeload_lmc_text_enc_refresh = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            safeload_use_text_fp16 = gr.Checkbox(value=1, label="variant fp16")
     
                

                with gr.Tab("Convert LCM-LoRA Model to Safetensors", id="tab_cml"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            convert_lcm_model_list_html = gr.HTML("Convert 'saved' LCM-LoRA type models to Safetensors (Single File) Models.<br>(<font size='-1'><i>NOTE:*Only* converts the UNet, VAE, and Text Encoder</i></font>.)")
                            gr.Markdown("<br>")
                            convert_lcm_model_list_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], label="Availiable LCM-LoRA Models for Conversion. (Local Saved LCM-LoRA Models)")
                            convert_load_lcm_model_fp16_check = gr.Checkbox(value=1,label="variant fp16")
                        with gr.Column(scale=0, min_width=100):
                            convert_lcm_model_reload_list_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            convert_lcm_model_info_button = gr.Button("", icon="./icons/about64.png", elem_id="reloadmodellist_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            convert_safe_model_name = gr.Textbox(label="New Safetensors Base Model Name (No extension)", placeholder="MySafetensorsModelName")
                            convert_lcm_model_load_model_button = gr.Button("Convert LCM-LoRA Model to Safetensors Model")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            gr.Markdown("Safetensors Model Conversion Settings")
                            convert_safe_model_half = gr.Checkbox(label="Use Half Precision (fp16)", value=True)
                            convert_safe_model_use = gr.Checkbox(label="Use Safetensors (Fallback on *.BIN)", value=True)
                            convert_safe_model_only = gr.Checkbox(label="Use ALL Safetensors ONLY (NEVER USE *.BIN, Use Safetensors must be checked.)", value=True)
                            convert_safe_model_card_info = gr.Textbox(label="Add Information to Model Card (using Markdown)", lines=6)
                            




        with gr.Tab("Add Lora Models", id="tab_AddLoraModels") as inner_tab_AddLoraModels:
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    loradropdown = gr.Dropdown(choices=LLSTUDIO["lora_model_list"], label="Availiable Lora Models to Add to Loaded Model")
                with gr.Column(scale=0, min_width=100):
                    reload_lora_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
            with gr.Row():
                with gr.Column(scale=2, min_width=100):
                    loraload_model_use_lcmscheduler = gr.Checkbox(label="Use LCMScheduler - (Must check if adding the LCM-LoRA Weights manually.)")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    lora_scale_value = gr.Slider(label="Lora Scale", value=1.0, minimum=-10.0, maximum=10, step=0.1)
                with gr.Column(scale=0, min_width=100):
                    lora_list_button = gr.Button("", icon="./icons/view64.png", elem_id="add_button")
                    lora_add_button = gr.Button("", icon="./icons/add64.png", elem_id="add_button")
                    lora_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    loaded_loradropdown = gr.Dropdown(choices=LLSTUDIO["loaded_lora_model_list"], label="Loaded LoRA Models")
                with gr.Column(scale=0, min_width=100):
                    loaded_lora_list_refresh = gr.Button("", icon="./icons/refresh64.png", elem_id="add_button")
                    lora_change_weight_button = gr.Button("", icon="./icons/hierarchy64.png", elem_id="add_button")
            with gr.Row():
                with gr.Column(scale=2, min_width=100):
                    lorahtml = gr.HTML("<p>If adding the LCM-LoRAs weights here, rather than loaded automatically when the model is loaded:<br><ol><li>Make sure to check the weight scale for the LoRA before loading. (Should be set to '1.0' for the LCM-LoRAs.) Feel free to experiment. :)</li><li>You must also check 'Use LCMScheduler' to switch the pipeline to use the LCMScheduler rather than the default for the pipeline.</li></ol><br>NOTE: The LoRA Scale 'slider' will go from '-10' to '+10' to account for a few LoRA models I ran across which use both a postive, 0 or a negative value.<br>Consult the model card for your LoRA model for more information on adjusting the LoRA weight.</p>")
                


        with gr.Tab("Save as LCM-LoRA Model", id="tab_SaveLCMModel") as inner_tab_AddLoraModels:
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_htmlt = gr.HTML("<h3>Save Loaded Pipeline (Model) (with fused LoRAs) to New LCM-LoRA Model</h3>")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_txt = gr.Textbox(value=STUDIO["lcm_model_prefix"]["value"]+"MyNewModel"+STUDIO["lcm_model_suffix"]["value"], info="Enter name for new LCM-LoRA model. Will be saved in LCM-LoRA Models Directory. (a-Z,0-9,_ only)", label="New LCM-LoRA Model Name", lines=1, show_label=True, show_copy_button=True)
                with gr.Column(scale=0, min_width=100):
                    save_lcm_model_clear_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                    save_lcm_model_save_button = gr.Button("", icon="./icons/export64.png", elem_id="reloadmodellist_button")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_lora_scale = gr.Slider(label="Lora Scale", value=1.0, minimum=0.0, maximum=10, step=0.1)
            with gr.Row():
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_as_safetensors_check = gr.Checkbox(value=1, label="Save Model as Safetensors Files")
            with gr.Row():
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_fp16_check = gr.Checkbox(value=1,label="Save as fp16")
            with gr.Row():
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_html = gr.HTML("")




        with gr.Tab("Model Gallery Viewers", id="tab_ModelGalleryViewers"):
            with gr.Tabs(selected="tab_ov") as inner_tab_ModelGalleryViewers:
                with gr.Tab("Outputs Viewer", id="tab_ov"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            outputgallery_htmlt = gr.HTML("<h3>Availiable Outputs Gallery</h3>")
                    with gr.Row(equal_height=True):
                        outputgallery_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button")
                        outputgallery_first_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        outputgallery_prev_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        outputgallery_next_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        outputgallery_last_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            hidden_numb0 = gr.Number(label="0", visible=False, value=0)
                            hidden_numb1 = gr.Number(label="1", visible=False, value=1)
                            hidden_numb2 = gr.Number(label="2", visible=False, value=2)
                            hidden_numb3 = gr.Number(label="3", visible=False, value=3)
                            hidden_numb4 = gr.Number(label="4", visible=False, value=4)
                            hidden_numb5 = gr.Number(label="5", visible=False, value=5)
                            outputgallery_html2 = gr.HTML("")
                            outputgallery_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        outputgallery_firstb_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        outputgallery_prevb_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        outputgallery_nextb_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        outputgallery_lastb_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")

                            
                with gr.Tab("LCM-LoRA Model Viewer", id="tab_lmv"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcmmodelview_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_model_image_list"], label="Availiable LCM-LoRA Models Gallery")
                        with gr.Column(scale=0, min_width=100):
                            lcmmodelview_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            lcmmodelview_button = gr.Button("", icon="./icons/gallery64.png", elem_id="view_button")
                    with gr.Row():
                        with gr.Accordion("Model Information", open=False) as lcm_modelcard:
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=2, min_width=100):
                                    lcmmodelview_hiddenhtml = gr.HTML("")
                                with gr.Column(scale=0, min_width=100):
                                    lcmmodelview_save_button = gr.Button("", icon="./icons/save64.png", elem_id="view_button", visible=False)
                                with gr.Column(scale=0, min_width=100):
                                    lcmmodelview_view_button = gr.Button("", icon="./icons/view64.png", elem_id="view_button", visible=False)
                                with gr.Column(scale=0, min_width=100):
                                    lcmmodelview_edit_button = gr.Button("", icon="./icons/settings64.png", elem_id="view_button")
                            with gr.Row(equal_height=False):
                                lcmmodeledit_html2 = gr.Code("", language="markdown", visible=False)
                                lcmmodelview_html2 = gr.Markdown("", visible=True)
                    with gr.Row(equal_height=True):
                        lcmgallery_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button")
                        lcmgallery_first_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        lcmgallery_prev_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        lcmgallery_next_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        lcmgallery_last_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")
                    with gr.Row():
                        lcmmodelview_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        lcmgallery_firstb_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        lcmgallery_prevb_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        lcmgallery_nextb_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        lcmgallery_lastb_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")
     

                with gr.Tab("Safetensors Model Viewer", id="tab_smv"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            safeimageview_dropdown = gr.Dropdown(choices=LLSTUDIO["safe_model_image_list"], label="Availiable Safetensors Models Gallery")
                        with gr.Column(scale=0, min_width=100):
                            safeimageview_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            safeimageview_button = gr.Button("", icon="./icons/gallery64.png", elem_id="view_button")
                    with gr.Row():
                        with gr.Accordion("Model Information", open=False) as safe_modelcard:
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=2, min_width=100):
                                    safeimageview_hiddenhtml = gr.HTML("")
                                with gr.Column(scale=0, min_width=100):
                                    safeimageview_save_button = gr.Button("", icon="./icons/save64.png", elem_id="view_button", visible=False)
                                with gr.Column(scale=0, min_width=100):
                                    safeimageview_view_button = gr.Button("", icon="./icons/view64.png", elem_id="view_button", visible=False)
                                with gr.Column(scale=0, min_width=100):
                                    safeimageview_edit_button = gr.Button("", icon="./icons/settings64.png", elem_id="view_button")
                            with gr.Row(equal_height=False):
                                 safeimageedit_html2 = gr.Code("", language="markdown", visible=False)
                                 safeimageview_html2 = gr.Markdown("", visible=True)
                    with gr.Row(equal_height=True):
                        safegallery_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button")
                        safegallery_first_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        safegallery_prev_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        safegallery_next_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        safegallery_last_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")
                    with gr.Row():
                        safeimageview_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        safegallery_firstb_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        safegallery_prevb_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        safegallery_nextb_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        safegallery_lastb_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")

                            
                with gr.Tab("LoRA Model Viewer", id="tab_lrv"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            loraimageview_dropdown = gr.Dropdown(choices=LLSTUDIO["lora_model_image_list"], label="Availiable LoRA Models Gallery")
                        with gr.Column(scale=0, min_width=100):
                            loraimageview_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            loraimageview_button = gr.Button("", icon="./icons/gallery64.png", elem_id="view_button")
                    with gr.Row():
                        with gr.Accordion("Model Information", open=False) as lora_modelcard:
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=2, min_width=100):
                                    loraimageview_hiddenhtml = gr.HTML("")
                                with gr.Column(scale=0, min_width=100):
                                    loraimageview_save_button = gr.Button("", icon="./icons/save64.png", elem_id="view_button", visible=False)
                                with gr.Column(scale=0, min_width=100):
                                    loraimageview_view_button = gr.Button("", icon="./icons/view64.png", elem_id="view_button", visible=False)
                                with gr.Column(scale=0, min_width=100):
                                    loraimageview_edit_button = gr.Button("", icon="./icons/settings64.png", elem_id="view_button")
                            with gr.Row(equal_height=False):
                                loraimageedit_html2 = gr.Code("", language="markdown", visible=False)
                                loraimageview_html2 = gr.Markdown("", visible=True)
                    with gr.Row(equal_height=True):
                        loragallery_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button")
                        loragallery_first_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        loragallery_prev_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        loragallery_next_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        loragallery_last_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")
                    with gr.Row():
                        loraimageview_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        loragallery_firstb_button = gr.Button("", icon="./icons/first64.png", elem_id="icon_button")
                        loragallery_prevb_button = gr.Button("", icon="./icons/previous64.png", elem_id="icon_button")
                        loragallery_nextb_button = gr.Button("", icon="./icons/next64.png", elem_id="icon_button")
                        loragallery_lastb_button = gr.Button("", icon="./icons/last64.png", elem_id="icon_button")

        with gr.TabItem("Tools", id="tab_Tools"):

            with gr.TabItem("Image Processing", id="tab_Image_Processing"):

                with gr.Row():
                    title_info = gr.Markdown("### Image Processing")
                    save_result = gr.Markdown("", visible=True)
                
                with gr.Accordion("Image Processing Load/Save", open=False):        
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            imgp_status = gr.Markdown("Select PNG File from the dropdown and load. Or enter new PNG filename, then click 'Save' to create a new PNG file.")
                            imgp_selector = gr.Dropdown(choices=imgp_get_file_list(), label="Select PNG filename to Load/Save", interactive=True)
                        with gr.Column(scale=0, min_width=100):
                            imgp_refresh_list_btn = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button")
                            imgp_load_btn = gr.Button("", icon="./icons/load64.png", elem_id="icon_button")
                            imgp_save_btn = gr.Button("", icon="./icons/save64.png", elem_id="icon_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            imgp_new_filename = gr.Textbox(label="Or create a 'New' file by entering a 'new' PNG filename (.png)", placeholder="image.png", interactive=True)
                        with gr.Column(scale=0, min_width=200):
                            imgp_img_location = gr.Dropdown(choices=["Input", "Adjusted", "Grayscale", "Output"], label="Load/Save Target")
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=150):

                        with gr.Row():
                            with gr.Column(scale=0, min_width=300):
                                adjusted_output = gr.Image(label="Stage 1 - Adjusted Image", elem_classes=["custom-image"], scale=0, height=300, show_download_button=False, interactive=False)
                            with gr.Column(scale=0, min_width=300):
                                grayscale_output = gr.Image(label="Stage 2 - Grayscale Image", elem_classes=["custom-image"], scale=0, height=300, show_download_button=False, interactive=False)
                                depth_map_button = gr.Button("Depth Map Only")
                                send_gray_to_cnet_button = gr.Button("Send to ControlNet")
                                
                    with gr.Column(scale=1, min_width=150):
                        with gr.Accordion("Input Image (Hide for better view of UI", open=True):
                            with gr.Row():
                                input_image = gr.Image(label="Input Image", type="numpy", height=300, elem_classes=["custom-image"], show_download_button=False)

                        with gr.Row():
                            run_button = gr.Button("Process Image")
                        with gr.Row():
                            mono_output = gr.Image(label="Stage 3 - Final Output", elem_classes=["custom-image"], scale=1, height=300, show_download_button=False, interactive=False)
                        with gr.Row():
                            post_process_button = gr.Button("Post Process")
                        with gr.Row():
                            send_mono_to_cnet1_button = gr.Button("Send to ControlNet 1")
                        with gr.Row():
                            send_mono_to_cnet2_button = gr.Button("Send to ControlNet 2")
                            
                    with gr.Column(scale=1, min_width=150):
                        with gr.Row():
                            reset_button = gr.Button("Reset Configuration")
                        with gr.Accordion("Stage 1 - Brightness/Contrast/Color/RGB", open=False) as acc_stage1:  
                            with gr.Row():
                                brightness = gr.Slider(0.0, 3.0, value=1.0, label="Brightness")
                                contrast = gr.Slider(0.0, 3.0, value=1.0, label="Contrast")
                                color = gr.Slider(0.0, 3.0, value=1.0, label="Color")

                            with gr.Row():
                                r_weight = gr.Slider(0.0, 2.0, value=1.0, label="Red Weight")
                                g_weight = gr.Slider(0.0, 2.0, value=1.0, label="Green Weight")
                                b_weight = gr.Slider(0.0, 2.0, value=1.0, label="Blue Weight")

                        with gr.Accordion("Stage 2 - Grayscale Thresholds/RGB Color", open=False) as acc_stage2:  
                            with gr.Row():
                                with gr.Column(scale=2, min_width=100):
                                    lower_thresh = gr.Slider(0, 255, value=100, label="Lower Threshold")
                                with gr.Column(scale=2, min_width=100):        
                                    upper_thresh = gr.Slider(0, 255, value=200, label="Upper Threshold")
                                with gr.Column(scale=1, min_width=100):        
                                    invert_grayscale = gr.Checkbox(label="Invert Grayscale Output", value=False)

                            with gr.Row():
                                r_gray_weight = gr.Slider(0.0, 1.0, value=0.2989, label="Red Grayscale Weight")
                                g_gray_weight = gr.Slider(0.0, 2.0, value=0.5870, label="Green Grayscale Weight")
                                b_gray_weight = gr.Slider(0.0, 1.0, value=0.1140, label="Blue Grayscale Weight")

                        with gr.Accordion("Stage 3 - Post-Processing Filters", open=False) as acc_stage3:  
                            with gr.Row():
                                sharpen = gr.Checkbox(label="Apply Sharpening", value=False)
                            with gr.Row():
                                apply_edges = gr.Checkbox(label="Apply Edge Detection", value=False)
                                edge_filters = gr.Dropdown(choices=EDGEFILTERS, label="Edge Detection Filters")
                                with gr.Column(scale=2, min_width=100):
                                    lower_canny_thresh = gr.Slider(0, 255, value=100, label="Canny Lower")
                                with gr.Column(scale=2, min_width=100):        
                                    upper_canny_thresh = gr.Slider(0, 255, value=200, label="Canny Upper")
                            with gr.Row():
                                invert_final = gr.Checkbox(label="Invert Final Output (Last Step)", value=False)

                        with gr.Accordion("Blur Controls (Each Stage)", open=False) as acc_post:  
                            with gr.Row():
                                def blur_controls(stage):
                                    with gr.Row():
                                        g = gr.Checkbox(label=f"{stage}: Gaussian Motion")
                                        g_amt = gr.Slider(0, 50, value=0, label="Amount", interactive=True)
                                        h = gr.Checkbox(label=f"{stage}: Horizontal Motion")
                                        h_amt = gr.Slider(0, 50, value=0, label="Amount", interactive=True)
                                        v = gr.Checkbox(label=f"{stage}: Vertical Motion")
                                        v_amt = gr.Slider(0, 50, value=0, label="Amount", interactive=True)
                                    return g, g_amt, h, h_amt, v, v_amt

                                s1_g, s1_g_amt, s1_h, s1_h_amt, s1_v, s1_v_amt = blur_controls("Stage 1")
                                s2_g, s2_g_amt, s2_h, s2_h_amt, s2_v, s2_v_amt = blur_controls("Stage 2")
                                s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt = blur_controls("Stage 3")



            with gr.TabItem("OpenPose Editor", id="tab_OpenPose_Editor"):
                with gr.Row(equal_height=True):
                    openpose_edit_title = gr.Markdown("## OpenPose Editor")
                with gr.Row(equal_height=True):
                    x_openpose_html = gr.HTML(openpose_html)



            with gr.TabItem("Manage Images", id="tab_ManageImages"):
                # State variable to hold selected files for deletion
                man_image_selected_images_state = LLSTUDIO["gallery_selected_image"]
                man_images_selected_images_state = gr.HTML(visible=False)
                with gr.Row(equal_height=True):
                    gr.Markdown("## Image and Generation Text Parameter Deletion Tool")
                with gr.Row(equal_height=True):
                    gr.Markdown("Click an image in the gallery to view its text.")
                with gr.Row(equal_height=True):
                    man_images_reload_button = gr.Button("Refresh Images", elem_id="green_button")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        man_images_gallery = gr.Gallery(
                            label="Image preview (Click to view text)",
                            value=get_sorted_newest_image_list(),
                            columns=4,
                            rows=3,
                            object_fit="scale-down", 
                            height="auto",
                            type="pil",
                            allow_preview=True,
                            show_download_button=False,
                            selected_index=0,
                            elem_id="my_gallery"
                        )
                    with gr.Column(scale=2):
                        man_images_text_viewer = gr.Textbox(
                            label="Generation Parameters Text File",
                            interactive=False,
                            lines=20,
                            elem_id="text_content"
                        )

                with gr.Row():
                    man_images_output_message = gr.Textbox(label="Status")

                with gr.Accordion("Image/Text Parameter Delete (Hidden for Safety)", open=False):
                    with gr.Row():
                        man_images_delete_btn = gr.Button("Delete Selected", elem_id="red_button")



            with gr.TabItem("Manage Models", id="tab_ManageModels"):

                with gr.TabItem("Merge Models", id="tab_MergeModels"):
                    gr.Markdown("# SD / SDXL Weighted Block Model Merger")
                    gr.Markdown("""Merges 2 of your 'saved' LCM-LoRA Models, and creates a new model from the merge.   
                        *Merges only the U-Net, Text Encoders and VAE components of the models.*   
                        "The 'sliders' control 'how much' of Model B is 'injected' into each structural region (block) of Model A.""")
                    gr.Markdown("")
                    gr.Markdown("## Models")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            model_a = gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], label="Model A", info="Base model. Lower slider values preserve this model.")
                        with gr.Column(scale=0, min_width=64):
                            refresh_model_btn = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button", scale=0)
                        with gr.Column(scale=2, min_width=100):
                            model_b = gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], label="Model B", info="Injected model. Higher slider values favor this model.")
                    with gr.Row():
                        fp16 = gr.Checkbox(value=True,label="variant fp16", scale=2)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            model_type = gr.Radio(choices=["SD1.5", "SDXL"], value="SD1.5", label="Model Type", info="Choose the model family.<br>Note: SD1.5 and SDXL cannot be merged together.")
                        with gr.Column(scale=2, min_width=100):
                            merged_model_name = gr.Textbox(value="My_Merged_Model", label="Merged New Model Name", info="New model name for the merged model. The new 'merged' model will be saved in your 'LCM-LoRA' Models folder.<br>To use the newly created model for generation, just go to the 'Pipeline Models - LCM-LoRA Models List' tab, and refresh the dropdown list. Then select it in the list to load your new 'merged' model.")
                    gr.HTML("<hr>")
                    gr.Markdown("")
                    with gr.Accordion("U-Net Presets", open=False):
                        preset_dropdown = gr.Dropdown(choices=list(PRESETS.keys()), value="Balanced", label="Preset", info="Apply predefined merge weighting strategies. Good for use as a starting point to experiment.<br>Note: Select model type 'SD' or 'SDXL' before applying a 'preset'.")
                        apply_preset_btn = gr.Button("Apply Preset")
                    gr.HTML("<hr>")
                    gr.Markdown("")
                    with gr.Accordion("Profiles", open=False):
                        gr.Markdown("Loads and Saves all the weights and model type settings. Use to apply the same merge, to the same model type in the future.")
                        with gr.Row():
                            gr.Markdown("### Load Profiles")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2, min_width=100):
                                load_profile_dropdown = gr.Dropdown(choices=LLSTUDIO["profiles_list"], label="Load Profile")
                            with gr.Column(scale=0, min_width=100):
                                refresh_profile_btn = gr.Button("", icon="./icons/refresh64.png", elem_id="icon_button", scale=0)
                                load_profile_btn = gr.Button("", icon="./icons/load64.png", elem_id="icon_button", scale=0)
                        with gr.Row(equal_height=True, elem_id="status-row-bg"):
                            with gr.Column(scale=0):
                                gr.HTML("<h2>Load/Save Profile Status:</h2>")
                            with gr.Column(scale=2, min_width=100):
                                loaded_profile_html = gr.HTML("Status: Ready.")
                        with gr.Row():
                            gr.Markdown("### Save Profiles")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2, min_width=100):
                                profile_name = gr.Textbox(label="New Profile Name", value="My_Profile_SDXL", info="Try and be descriptive when naming your profiles.<br>Possibly include an 'SD' or 'SDXL' in the name to indicate which type of model the profile is used for.")
                            with gr.Column(scale=0, min_width=100):
                                save_profile_btn = gr.Button("", icon="./icons/save64.png", elem_id="icon_button", scale=0)
                        with gr.Row():
                            profile_description = gr.Textbox(label="New Profile Description", lines=2, value="My description for a profile.", info="Just a simple description for the profile.")
                    gr.HTML("<hr>")
                    gr.Markdown("")
                    with gr.Accordion("Text Encoder Weight and VAE Weights", open=False):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=100):
                                text_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Text Encoder Alpha", info="Controls prompt understanding/style language blending.\nHigher values favor Model B prompt interpretation.")
                            with gr.Column(scale=1, min_width=100):
                                vae_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="VAE Alpha", info="Controls color processing and latent decoding.\nHigher values favor Model B color science and contrast.")
                    gr.HTML("<hr>")
                    gr.Markdown("")
                    with gr.Accordion("U-Net Block Weight Controls", open=False):
                        sliders = build_block_slider_ui()
                    gr.HTML("<hr>")
                    gr.Markdown("")
                    gr.Markdown("## Saved Merged Output Model Parameters")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            out_fp16 = gr.Checkbox(label="Save using Half Precision (fp16)", value=True)
                        with gr.Column(scale=1, min_width=100):
                            out_safe = gr.Checkbox(label="Save as Safetensors", value=True)
                    gr.Markdown("")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            merge_goto_top_btn = gr.Button("Top of Page", elem_id="gray_button")
                        with gr.Column(scale=2, min_width=200):
                            merge_btn = gr.Button("Merge Models", elem_id="green_button")
                    merge_status_html = gr.HTML("Status - Ready...")
                    gr.HTML("<br><br>")


                with gr.TabItem("LCM-LoRA Models", id="tab_LCMModels"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcmtool_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], label="Availiable LCM-LoRA Models to Delete")
                        with gr.Column(scale=0, min_width=100):
                            lcmtool_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            lcmtool_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            lcmtool_delete_model_check = gr.Checkbox(label="Delete Model")
                            lcmtool_delete_images_check = gr.Checkbox(label="Delete Model Image Gallery")
                            lcmtool_html2 = gr.HTML("")
                            lcmtool_html = gr.HTML("")

                with gr.TabItem("Huggingface (Local Cached) Models", id="tab_HuggingfaceLocalCachedModels"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            hub_tool_dropdown = gr.Dropdown(choices=LLSTUDIO["hub_model_list"], label="Availiable Huggingface (Local Cached) Models Models to Delete")
                        with gr.Column(scale=0, min_width=100):
                            hub_tool_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            hub_tool_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            hub_tool_delete_model_check = gr.Checkbox(label="Delete Model")
                            hub_tool_html2 = gr.HTML("")
                            hub_tool_html = gr.HTML("")

                        
                with gr.TabItem("LoRA Models", id="tab_LoRAModels"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            loratool_dropdown = gr.Dropdown(choices=LLSTUDIO["lora_model_list"], label="Availiable Lora Models to Delete")
                        with gr.Column(scale=0, min_width=100):
                            loratool_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            loratool_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            loratool_delete_model_check = gr.Checkbox(label="Delete Model")
                            loratool_delete_images_check = gr.Checkbox(label="Delete Model Image Gallery")
                            loratool_html2 = gr.HTML("")
                            loratool_html = gr.HTML("")

                with gr.TabItem("Safetensors Models", id="tab_ManageModels_safe"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            safetool_dropdown = gr.Dropdown(choices=LLSTUDIO["safe_model_list"], label="Availiable Safetensors Models to Delete")
                        with gr.Column(scale=0, min_width=100):
                            safetool_reload_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            safetool_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            safetool_delete_model_check = gr.Checkbox(label="Delete Model")
                            safetool_delete_images_check = gr.Checkbox(label="Delete Model Image Gallery")
                            safetool_html2 = gr.HTML("")
                            safetool_html = gr.HTML("")


 
            with gr.TabItem("System", id="tab_System"):
                with gr.Accordion("Control System/Application", open=True):
                    with gr.Row(equal_height=True):
                        sysinfo_haltgen_button = gr.Button("Halt Image Generation", scale=0, elem_id="gray_button")
                    with gr.Row(equal_height=True):
                        sysinfo_logout_button = gr.Button("Logout", scale=0, elem_id="yellow_button")
                        sysinfo_reload_button = gr.Button("Reload Browser", scale=0, elem_id="blue_button")
                        sysinfo_restart_button = gr.Button("RESTART", scale=0, elem_id="exit_button")
                        sysinfo_exit_button = gr.Button("EXIT", scale=0, elem_id="exit_button")
                        if LLSTUDIO["current_os"] == "Linux":
                            sysinfo_sudo_shutdown_button = gr.Button("Linux Shutdown", scale=0, elem_id="exit_button")
                            sysinfo_sudo_reboot_button = gr.Button("Linux Reboot", scale=0, elem_id="exit_button")
                    with gr.Row(equal_height=False):
                        sysinfo_hug_on_button = gr.Button("Huggingface ON", scale=0, elem_id="exit_button")
                        sysinfo_hug_off_button = gr.Button("Huggingface OFF", scale=0, elem_id="exit_button")
                        sysinfo_hug_check_button = gr.Button("Check HF Status", scale=0, elem_id="purple_button")
                        sysinfo_hug_status = gr.Textbox(label="Huggingface On/Off Status", value="Click the 'Check HF Status' to check status", info="You can also check the enviroment variables too.")
                with gr.Accordion("View System Information", open=False):
                    with gr.Row(equal_height=True):
                        sysinfo_cpumemswap_button = gr.Button("CPU/MEM/SWAP", scale=0, elem_id="generates_button")
                        sysinfo_memory_button = gr.Button("Memory", scale=0, elem_id="generates_button")
                        sysinfo_hfcache_button = gr.Button("HF Cache", scale=0, elem_id="generate_button")
                        sysinfo_env_button = gr.Button("Enviroment Variables", scale=1, elem_id="gray_button")
                        sysinfo_sysinfo_button = gr.Button("System Information", scale=1, elem_id="generate_button")
                with gr.Accordion("System Model Information", open=False):
                    with gr.Row(equal_height=True):
                        gr.HTML(system_model_information)
                    with gr.Row(equal_height=True):
                        dlm1 = gr.Checkbox(label="SD", value=True, info="<font size='+1'><b>System Model: latent-consistency/lcm-lora-sdv1-5</b></font><br>LoRA model, needed for the faster 4-step inference SD only")
                    with gr.Row(equal_height=True):
                        dlm2 = gr.Checkbox(label="SDXL", value=False, info="<font size='+1'><b>System Model: latent-consistency/lcm-lora-sdxl</b></font><br>LoRA model, needed for the faster 4-step inference SDXL only")
                    with gr.Row(equal_height=True):
                        dlm3 = gr.Checkbox(label="fp32 only", value=False, info="<font size='+1'><b>System Model: stabilityai/sd-x2-latent-upscaler</b></font><br>SD Base model for X2 Upscaler - X2 Upscaler Only")
                    with gr.Row(equal_height=True):
                        dlm4 = gr.Checkbox(label="fp16", value=True, info="<font size='+1'><b>System/Base Model: stable-diffusion-v1-5/stable-diffusion-v1-5</b></font><br>SD - Base/reference model for SD")
                    with gr.Row(equal_height=True):
                        dlm5 = gr.Checkbox(label="fp32", value=False, info="<font size='+1'><b>System/Base Model: stable-diffusion-v1-5/stable-diffusion-v1-5</b></font><br>SD - Base/reference model for SD")
                    with gr.Row(equal_height=True):
                        dlm6 = gr.Checkbox(label="fp16", value=False, info="<font size='+1'><b>System/Base Model: stabilityai/stable-diffusion-xl-base-1.0</b></font><br>SDXL - Base/reference model for  SDXL")
                    with gr.Row(equal_height=True):
                        dlm7 = gr.Checkbox(label="fp32", value=False, info="<font size='+1'><b>System/Base Model: stabilityai/stable-diffusion-xl-base-1.0</b></font><br>SDXL - Base/reference model for  SDXL")
                    with gr.Row(equal_height=True):
                        dlm8 = gr.Checkbox(label="fp16", value=False, info="<font size='+1'><b>System/Base Model: stable-diffusion-v1-5/stable-diffusion-inpainting</b></font><br>We only use the fp16 Safetensors weights.<br>SD - Base/reference model for SD Inpainting")
                    with gr.Row(equal_height=True):
                        dlm9 = gr.Checkbox(label="fp16", value=False, info="<font size='+1'><b>System/Base Model: diffusers/stable-diffusion-xl-1.0-inpainting-0.1</b></font><br>SDXL - Base/reference model for SDXL Inpainting")
                    with gr.Row(equal_height=True):
                        dlm10 = gr.Checkbox(label="fp32", value=False, info="<font size='+1'><b>System/Base Model: diffusers/stable-diffusion-xl-1.0-inpainting-0.1</b></font><br>SDXL - Base/reference model for SDXL Inpainting")
                    with gr.Row(equal_height=True):
                        dlm11 = gr.Checkbox(label="fp16", value=False, info="<font size='+1'><b>System/Base Model: timbrooks/instruct-pix2pix</b></font><br>SD - Base/reference model for SD instruct-pix2pix")
                    with gr.Row(equal_height=True):
                        dlm12 = gr.Checkbox(label="fp32", value=False, info="<font size='+1'><b>System/Base Model: timbrooks/instruct-pix2pix</b></font><br>SD - Base/reference model for SD instruct-pix2pix")
                    with gr.Row(equal_height=True):
                        dlm13 = gr.Checkbox(label="fp16", value=False, info="<font size='+1'><b>System/Base Model: diffusers/sdxl-instructpix2pix-768</b></font><br>SDXL - Base/reference model for SDXL instruct-pix2pix")
                    with gr.Row(equal_height=True):
                        dlm14 = gr.Checkbox(label="fp32", value=False, info="<font size='+1'><b>System/Base Model: diffusers/sdxl-instructpix2pix-768</b></font><br>SDXL - Base/reference model for SDXL instruct-pix2pix")
                    with gr.Row(equal_height=True):
                        dlm15 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-mlsd</b></font><br>ControlNet Model - Safetensors type<br>M-LSD line detection")
                    with gr.Row(equal_height=True):
                        dlm16 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-hed</b></font><br>ControlNet Model - Safetensors type<br>HED edge detection")
                    with gr.Row(equal_height=True):
                        dlm17 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-depth</b></font><br>ControlNet Model - Safetensors type<br>Midas depth estimation")
                    with gr.Row(equal_height=True):
                        dlm18 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-scribble</b></font><br>ControlNet Model - Safetensors type<br>Hand drawn scribbles")
                    with gr.Row(equal_height=True):
                        dlm19 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-canny</b></font><br>ControlNet Model - Safetensors type<br>Canny edge detection")
                    with gr.Row(equal_height=True):
                        dlm20 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-normal</b></font><br>ControlNet Model - Safetensors type<br>Normal map")
                    with gr.Row(equal_height=True):
                        dlm21 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-seg</b></font><br>ControlNet Model - Safetensors type<br>Semantic segmentation")
                    with gr.Row(equal_height=True):
                        dlm22 = gr.Checkbox(label="SD only", value=False, info="<font size='+1'><b>System Model: lllyasviel/sd-controlnet-openpose</b></font><br>ControlNet Model - Safetensors type<br>OpenPose bone image")
                    with gr.Row(equal_height=True):
                        dlm23 = gr.Checkbox(label="All", value=False, info="<font size='+1'><b>System Model: depth-estimation</b></font><br>Transformers - Safetensors type<br>Creates a 'depth map' image, whichan be used with a ControlNet.")
                    with gr.Row(equal_height=True):
                        sysmodels_goto_top_button = gr.Button("Back to Top of Page", scale=0, elem_id="gray_button")
                        sysmodels_uncheckall_button = gr.Button("UnCheck All", scale=0, elem_id="gray_button")
                        sysmodels_checkdefaults_button = gr.Button("Check Defaults", scale=0, elem_id="gray_button")
                        sysmodels_download_button = gr.Button("Start Download", scale=1, elem_id="green_button")
                        sysmodels_cancel_button = gr.Button("Cancel Download", scale=1, elem_id="red_button")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=100):
                        sysinfo_html = gr.HTML("")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=100):
                        sysinfo_dummyhtml = gr.HTML("<br><br><br>")



            with gr.TabItem("Settings", id="tab_Settings"):
                with gr.Row(equal_height=True):
                    settings_status_text = gr.Textbox(lines=2, value="Some settings ONLY go into effect AFTER you restart the program, unless marked 'LIVE'. The 'LIVE' settings you can tweak, go check results of that tweak, rinse and repeat, then finally save when done.", label="Settings Status")

                with gr.Row(equal_height=True):
                    settings_save_button = gr.Button("Save Settings")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=100):

                        # Call the function to build the settings ui section
                        gr_components = create_settings_ui()
                     
                with gr.Row(equal_height=True):
                    settings_goto_top_button = gr.Button("Back to Top of Page")
                    settings_save_button2 = gr.Button("Save Settings")
                with gr.Row(equal_height=True):
                    settings_status_text2 = gr.Textbox(lines=2, value="Some settings ONLY go into effect AFTER you restart the program, unless marked 'LIVE'. The 'LIVE' settings you can tweak, go check results of that tweak, rinse and repeat, then finally save when done.", label="Settings Status")



            with gr.TabItem("Help", id="tab_Help"):
                with gr.Row(equal_height=True):
                    html_title = gr.Markdown("## LCM-LoRA Studio Help")
                with gr.Row(equal_height=True):
                    x_help_html = gr.HTML(help_html)


# ----------------------------------------------------------------------------------


# ==================================================================================================================
# must have this indentation for the 'with gr.Tab...' line above
# the indentation puts us outside of the 'with gr.Tab...' grouping
# ------------------------------------------------------------------------------------------------------------------
    
    # -----IMAGE PROCESSING-----
    imgp_refresh_list_btn.click(imgp_refresh_file_list_dropdown, inputs=None, outputs=imgp_selector)
    imgp_load_btn.click(fn=imgp_load_file, inputs=[imgp_selector, imgp_img_location, input_image, adjusted_output, grayscale_output, mono_output], outputs=[imgp_status, input_image, adjusted_output, grayscale_output, mono_output])
    imgp_save_btn.click(fn=imgp_save_file, inputs=[input_image, adjusted_output, grayscale_output, mono_output, imgp_img_location, imgp_new_filename], outputs=[imgp_status])
    send_mono_to_cnet1_button.click(fn=send_to_controlnet, inputs=[mono_output], outputs=[cnetimage, imgp_status]).then(change_tab_cnet, None, [tabs, inner_tab_ImageGeneration])
    send_mono_to_cnet2_button.click(fn=send_to_controlnet, inputs=[mono_output], outputs=[cnetimage2, imgp_status]).then(change_tab_cnet, None, [tabs, inner_tab_ImageGeneration])
    send_gray_to_cnet_button.click(fn=send_to_controlnet, inputs=[grayscale_output], outputs=[cnetimage, imgp_status]).then(change_tab_cnet, None, [tabs, inner_tab_ImageGeneration])
    depth_map_button.click(fn=do_depth_map, inputs=[adjusted_output], outputs=[grayscale_output])


    run_button.click(
        fn=image_pipeline,
        inputs=[
            input_image, brightness, contrast, color,
            r_weight, g_weight, b_weight,
            r_gray_weight, g_gray_weight, b_gray_weight,
            lower_thresh, upper_thresh, invert_grayscale, invert_final,
            lower_canny_thresh, upper_canny_thresh,
            s1_g, s1_g_amt, s1_h, s1_h_amt, s1_v, s1_v_amt,
            s2_g, s2_g_amt, s2_h, s2_h_amt, s2_v, s2_v_amt,
            s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt,
            sharpen, apply_edges, edge_filters
            
        ],
        outputs=[
            adjusted_output, grayscale_output, mono_output
        ]
    )
    

# --------------------------------

    
    post_process_button.click(
        fn=post_process_pipeline,
        inputs=[
            grayscale_output,
            lower_thresh, upper_thresh, invert_grayscale, invert_final,
            lower_canny_thresh, upper_canny_thresh,
            s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt,
            sharpen, apply_edges, edge_filters
            
        ],
        outputs=[
            mono_output
        ]
    )
    

# --------------------------------

    
    reset_button.click(
        fn=reset_config,
        inputs=None,
        outputs=[
            brightness, contrast, color,
            r_weight, g_weight, b_weight,
            r_gray_weight, g_gray_weight, b_gray_weight,
            lower_thresh, upper_thresh, invert_grayscale, invert_final,
            lower_canny_thresh, upper_canny_thresh,
            s1_g, s1_g_amt, s1_h, s1_h_amt, s1_v, s1_v_amt,
            s2_g, s2_g_amt, s2_h, s2_h_amt, s2_v, s2_v_amt,
            s3_g, s3_g_amt, s3_h, s3_h_amt, s3_v, s3_v_amt,
            sharpen, apply_edges, edge_filters,
            adjusted_output, grayscale_output, mono_output,
            acc_stage1, acc_stage2, acc_stage3, acc_post
        ]
    )



# ------------------------------------------------------------------------------------------------------------------

    # # # ADVANCED IMAGE GALLERY
    
    # Connect events
    man_images_gallery.select(
        fn=get_text_content,
        outputs=[man_images_text_viewer, man_images_selected_images_state] 
    )


    # Delete the selected image and text
    man_images_delete_btn.click(
        fn=delete_items,
        inputs=man_images_selected_images_state,
        outputs=[man_images_output_message, man_images_text_viewer]
    )
    # Refresh the gallery AFTER deletion
    man_images_delete_btn.click(
        fn=get_sorted_newest_image_list,
        outputs=man_images_gallery
    )

    
    # manual image gallery list refresh
    man_images_reload_button.click(fn=get_sorted_newest_image_list, outputs=man_images_gallery)

    
# ------------------------------------------------------------------------------------------------------------------

    # # # PIPELINE
    
    # Pipeline section
    pipeline_delete_button.click(delete_pipeline, None, outputs=[model_list_html]).then(update_grapptitle, None, app_title_label)


    # # # MODELS - LCMLORA - HUB CACHE - HUGGINGFACE

    # LCM-LoRA Model section
    lcm_model_reload_list_button.click(update_lcm_model_list_dropdown, None, lcm_model_list_dropdown)
    lcm_model_info_button.click(get_lcm_pipeclass_model_info, lcm_model_list_dropdown, lcm_model_list_html)
    # # rknote CONTROLNET must add to input list: [lcm_model_use_controlnet, lcm_model_cnet_dropdown]
    lcm_model_load_model_button.click(load_lcm_model, inputs=[lcm_model_list_dropdown, lcm_model_use_diff_text_encoder_check, lcm_model_liste_dropdown, lcm_model_clipskip, lcm_model_use_controlnet, lcm_model_cnet_dropdown, lcm_model_use_controlnet2, lcm_model_cnet_dropdown2, load_lcm_model_fp16_check, load_lcm_modele_fp16_check, load_lcm_model_add_lcmlora, load_lcm_model_lora_value, load_lcm_model_use_lcmscheduler], outputs=[lcm_model_list_html]).then(display_pipeline_info, inputs=[lcm_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html]).then(update_grapptitle, None, app_title_label)
    # Load separate text encoder LCM-LoRA model list
    lcm_model_reload_liste_button.click(update_lcm_sdonly_model_list_dropdown, None, lcm_model_liste_dropdown)



    # HUB - HUGGGINFACE Local Cache Model section
    hub_model_reload_list_button.click(update_hub_model_list_dropdown, None, hub_model_list_dropdown) 
    hub_model_load_model_button.click(load_hub_model, inputs=[hub_model_list_dropdown, hub_model_fp16_check, hub_model_model_use_lcmscheduler, hub_model_lora, hub_model_add_lcmlora], outputs=[hub_model_list_html]).then(display_pipeline_info, inputs=[hub_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html]).then(update_grapptitle, None, app_title_label)
    hub_model_info_button.click(get_hub_pipeclass_model_info, hub_model_list_dropdown, hub_model_list_html)
   
    
    
    # HUG - HUGGGINFACE Model section
    hug_model_download_model_button.click(load_hug_model, inputs=[hug_model_txt, hug_pipeline_classes, hug_model_fp16_check], outputs=[hug_model_list_html]).then(display_pipeline_info, inputs=[hug_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html]).then(update_grapptitle, None, app_title_label)

    

    # SAFETENSORS Model section
    safeload_model_reload_button.click(update_safe_convert_model_list_dropdown, None, safeload_model_dropdown) 
    safeload_model_load_button.click(load_safetensors_model, inputs=[safeload_model_dropdown, safeload_pipeline_classes, safeload_model_lora, safeload_model_add_lcmlora, safeload_use_text_enc, safeload_lmc_text_enc_dropdown, safeload_use_text_fp16, safeload_model_use_lcmscheduler], outputs=[safeload_model_list_html]).then(display_pipeline_info, inputs=[safeload_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html]).then(update_grapptitle, None, app_title_label)
    safeload_lmc_text_enc_refresh.click(update_safeload_lmc_text_enc_dropdown, None, safeload_lmc_text_enc_dropdown) 


    # CONVERT LCM-LORA MODEL TO SAFETENSORS Model section
    convert_lcm_model_reload_list_button.click(update_lcm_model_list_dropdown, None, convert_lcm_model_list_dropdown)
    convert_lcm_model_info_button.click(get_lcm_pipeclass_model_info, convert_lcm_model_list_dropdown, convert_lcm_model_list_html)
    convert_lcm_model_load_model_button.click(convert_to_safetensors_model, inputs=[convert_lcm_model_list_dropdown, convert_load_lcm_model_fp16_check, convert_safe_model_name, convert_safe_model_half, convert_safe_model_use, convert_safe_model_only, convert_safe_model_card_info], outputs=[convert_lcm_model_list_html]).then(update_grapptitle, None, app_title_label)



# ------------------------------------------------------------------------------------------------------------------
    # # # TAB - Image Generation

    inner_tab_ImageGeneration.select(set_title_mode, None, app_title_label)


    # # # TEXT 2 IMAGE

    t2iprompt_test_button.click(get_prompt_length_tokens, inputs=[t2iprompt_txt], outputs=[t2iprompt_txt])
    t2inegprompt_test_button.click(get_negprompt_length_tokens, inputs=[t2inegprompt_txt], outputs=[t2inegprompt_txt])

    
    # Generation section
    t2igen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[t2igen_seedval])

    t2igen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(t2igen_LCM_images, inputs=[t2iprompt_txt, t2inegprompt_txt, t2igen_width, t2igen_height, t2igen_guidance, t2igen_inference_steps, t2igen_num_images, t2igen_seedval, t2igen_sameseed_check, t2igen_incrementseed_check, t2igen_incrementseed_amount, t2igen_freeu_check, t2igen_freeu_s1, t2igen_freeu_s2, t2igen_freeu_b1, t2igen_freeu_b2, lcm_model_clipskip], outputs=[inference_status_markdown, oimage]).then(update_grapptitle, None, app_title_label)
    
    t2igen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])
    
    t2igen_default_freeu_button.click(set_freeu_values, inputs=[t2igen_freeu_s1, t2igen_freeu_s2, t2igen_freeu_b1, t2igen_freeu_b2], outputs=[t2igen_freeu_s1, t2igen_freeu_s2, t2igen_freeu_b1, t2igen_freeu_b2])

    oimage.change(display_generated_image, None, oimage2)
    
   
    # t2i prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    t2iaddweight_button.click(fn=None, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    t2iaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    t2iaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    t2imodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    t2iremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    t2iclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    
    
    # i2i prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    i2iaddweight_button.click(fn=None, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    i2iaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    i2iaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    i2imodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    i2iremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    i2iclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
    # inp prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    inpaddweight_button.click(fn=None, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    inpaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    inpaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    inpmodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    inpremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    inpclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
    # ip2p prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    ip2paddweight_button.click(fn=None, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    ip2paddpweight_button.click(fn=None, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    ip2paddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    ip2pmodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    ip2premove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    ip2pclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
    # up2x prompt helper
    # -------------------------
    # None, no embedded prompts for SD latent Upscale
 
    # controlnet prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    cnetaddweight_button.click(fn=None, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    cnetaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    cnetaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    cnetmodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, weight_number, ALL PROMPTS]
    cnetremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    cnetclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
 
 
    # t2i, i2i, inp, ip2p, up2x onFocus() (ALL) events for prompt helper
    # --------------------------------------------------------------------
    # onFocus() event sets name of current textbox, invisible to user
    # but user MUST have clicked in the one they want to edit
    t2iprompt_txt.focus(update_state, inputs=[hidden_t2iprompt_txt], outputs=hidden_prompt_name)
    t2inegprompt_txt.focus(update_state, inputs=[hidden_t2inegprompt_txt], outputs=hidden_prompt_name)
    i2iprompt_txt.focus(update_state, inputs=[hidden_i2iprompt_txt], outputs=hidden_prompt_name)
    i2inegprompt_txt.focus(update_state, inputs=[hidden_i2inegprompt_txt], outputs=hidden_prompt_name)
    inpprompt_txt.focus(update_state, inputs=[hidden_inpprompt_txt], outputs=hidden_prompt_name)
    inpnegprompt_txt.focus(update_state, inputs=[hidden_inpnegprompt_txt], outputs=hidden_prompt_name)
    ip2pprompt_txt.focus(update_state, inputs=[hidden_ip2pprompt_txt], outputs=hidden_prompt_name)
    ip2pnegprompt_txt.focus(update_state, inputs=[hidden_ip2pnegprompt_txt], outputs=hidden_prompt_name)
    cnetprompt_txt.focus(update_state, inputs=[hidden_cnetprompt_txt], outputs=hidden_prompt_name)
    cnetnegprompt_txt.focus(update_state, inputs=[hidden_cnetnegprompt_txt], outputs=hidden_prompt_name)
    
    


# ------------------------------------------------------------------------------------------------------------------

    # # # IMAGE 2 IMAGE


    i2iprompt_test_button.click(get_prompt_length_tokens, inputs=[i2iprompt_txt], outputs=[i2iprompt_txt])
    i2inegprompt_test_button.click(get_negprompt_length_tokens, inputs=[i2inegprompt_txt], outputs=[i2inegprompt_txt])

    # Generation section
    i2igen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[i2igen_seedval])

    i2igen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(i2igen_LCM_images, inputs=[i2iprompt_txt, i2inegprompt_txt, i2igen_width, i2igen_height, i2igen_guidance, i2igen_inference_steps, i2igen_seedval, i2igen_num_images, i2igen_incrementseed_check, i2igen_incrementseed_amount, i2iimage, i2igen_resize_input_image_check, i2igen_freeu_check, i2igen_freeu_s1, i2igen_freeu_s2, i2igen_freeu_b1, i2igen_freeu_b2, lcm_model_clipskip, i2igen_strength], outputs=[inference_status_markdown, oimage])
    
    i2igen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    i2igen_default_freeu_button.click(set_freeu_values, inputs=[i2igen_freeu_s1, i2igen_freeu_s2, i2igen_freeu_b1, i2igen_freeu_b2], outputs=[i2igen_freeu_s1, i2igen_freeu_s2, i2igen_freeu_b1, i2igen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # INPAINING


    inpprompt_test_button.click(get_prompt_length_tokens, inputs=[inpprompt_txt], outputs=[inpprompt_txt])
    inpnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[inpnegprompt_txt], outputs=[inpnegprompt_txt])

    # Generation section
    inpgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[inpgen_seedval])

    inpgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(inpgen_LCM_images, inputs=[inpprompt_txt, inpnegprompt_txt, inpgen_width, inpgen_height, inpgen_guidance, inpgen_inference_steps, inpgen_seedval, inpgen_num_images, inpgen_incrementseed_check, inpgen_incrementseed_amount, inpimage, inpgen_resize_input_image_check, inpimagemask, inpgen_freeu_check, inpgen_freeu_s1, inpgen_freeu_s2, inpgen_freeu_b1, inpgen_freeu_b2, lcm_model_clipskip, inpgen_strength], outputs=[inference_status_markdown, oimage])
    
    inpgen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    inpgen_default_freeu_button.click(set_freeu_values, inputs=[inpgen_freeu_s1, inpgen_freeu_s2, inpgen_freeu_b1, inpgen_freeu_b2], outputs=[inpgen_freeu_s1, inpgen_freeu_s2, inpgen_freeu_b1, inpgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # INSTRUCT PIX2PIX


    ip2pprompt_test_button.click(get_prompt_length_tokens, inputs=[ip2pprompt_txt], outputs=[ip2pprompt_txt])
    ip2pnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[ip2pnegprompt_txt], outputs=[ip2pnegprompt_txt])

    # Generation section
    ip2pgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[ip2pgen_seedval])

    ip2pgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(ip2pgen_LCM_images, inputs=[ip2pprompt_txt, ip2pnegprompt_txt, ip2pgen_guidance, ip2pgen_inference_steps, ip2pgen_seedval, ip2pgen_num_images, ip2pgen_incrementseed_check, ip2pgen_incrementseed_amount, ip2pimage, ip2pgen_resize_input_image_check, ip2pgen_imgguidance, ip2pgen_freeu_check, ip2pgen_freeu_s1, ip2pgen_freeu_s2, ip2pgen_freeu_b1, ip2pgen_freeu_b2, lcm_model_clipskip], outputs=[inference_status_markdown, oimage])
    
    ip2pgen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    ip2pgen_default_freeu_button.click(set_freeu_values, inputs=[ip2pgen_freeu_s1, ip2pgen_freeu_s2, ip2pgen_freeu_b1, ip2pgen_freeu_b2], outputs=[ip2pgen_freeu_s1, ip2pgen_freeu_s2, ip2pgen_freeu_b1, ip2pgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # SD UPSCALE 2X
 
    up2xprompt_test_button.click(get_prompt_length_tokens, inputs=[up2xprompt_txt], outputs=[up2xprompt_txt])
    up2xnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[up2xnegprompt_txt], outputs=[up2xnegprompt_txt])

    # Generation section
    up2xgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[up2xgen_seedval])

    up2xgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(upscale_image, inputs=[up2xprompt_txt, up2xnegprompt_txt, up2xgen_guidance, up2xgen_inference_steps, up2xgen_seedval, up2ximage, up2xgen_resize_input_image_check, up2xgen_freeu_check, up2xgen_freeu_s1, up2xgen_freeu_s2, up2xgen_freeu_b1, up2xgen_freeu_b2], outputs=[inference_status_markdown, oimage])

    up2xgen_default_freeu_button.click(set_freeu_values, inputs=[up2xgen_freeu_s1, up2xgen_freeu_s2, up2xgen_freeu_b1, up2xgen_freeu_b2], outputs=[up2xgen_freeu_s1, up2xgen_freeu_s2, up2xgen_freeu_b1, up2xgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # CONTROLNET


    cnetprompt_test_button.click(get_prompt_length_tokens, inputs=[cnetprompt_txt], outputs=[cnetprompt_txt])
    cnetnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[cnetnegprompt_txt], outputs=[cnetnegprompt_txt])

    # Generation section
    cnetgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[cnetgen_seedval])

    cnetgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(cnetgen_LCM_images, inputs=[cnetprompt_txt, cnetnegprompt_txt, cnetgen_width, cnetgen_height, cnetgen_guidance, cnetgen_guidance_start, cnetgen_guidance_end, cnetgen_conditioningguidance, cnetgen_conditioningguidance2, cnetgen_inference_steps, cnetgen_seedval, cnetgen_num_images, cnetgen_incrementseed_check, cnetgen_incrementseed_amount, cnetimage, cnetgen_resize_input_image, cnetimage2, cnetgen_resize_input_image2, cnetgen_freeu_check, cnetgen_freeu_s1, cnetgen_freeu_s2, cnetgen_freeu_b1, cnetgen_freeu_b2, lcm_model_clipskip, cnetgen_use_guess_mode], outputs=[inference_status_markdown, oimage])
    
    cnetgen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    cnetgen_default_freeu_button.click(set_freeu_values, inputs=[cnetgen_freeu_s1, cnetgen_freeu_s2, cnetgen_freeu_b1, cnetgen_freeu_b2], outputs=[cnetgen_freeu_s1, cnetgen_freeu_s2, cnetgen_freeu_b1, cnetgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------


    # Output Image section
    send_to_gallery_button.click(send_to_gallery, inputs=[], outputs=[gallery_html])
    outputimage_halt_gen_button.click(halt_generation, inputs=[], outputs=[]) 
    
# ------------------------------------------------------------------------------------------------------------------

    
    # Output Viewer section
    outputgallery_reload_button.click(show_output_preview, inputs=[hidden_numb2], outputs=[outputgallery_html2, outputgallery_html])
    outputgallery_first_button.click(show_output_preview, inputs=[hidden_numb2], outputs=[outputgallery_html2, outputgallery_html])
    outputgallery_prev_button.click(show_output_preview, inputs=[hidden_numb3], outputs=[outputgallery_html2, outputgallery_html])
    outputgallery_next_button.click(show_output_preview, inputs=[hidden_numb4], outputs=[outputgallery_html2, outputgallery_html])
    outputgallery_last_button.click(show_output_preview, inputs=[hidden_numb5], outputs=[outputgallery_html2, outputgallery_html])
    outputgallery_firstb_button.click(show_output_preview, inputs=[hidden_numb2], outputs=[outputgallery_html2, outputgallery_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    outputgallery_prevb_button.click(show_output_preview, inputs=[hidden_numb3], outputs=[outputgallery_html2, outputgallery_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    outputgallery_nextb_button.click(show_output_preview, inputs=[hidden_numb4], outputs=[outputgallery_html2, outputgallery_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    outputgallery_lastb_button.click(show_output_preview, inputs=[hidden_numb5], outputs=[outputgallery_html2, outputgallery_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    
# ------------------------------------------------------------------------------------------------------------------


    # Save LCM-LoRA Model section
    save_lcm_model_clear_button.click(clear_lcm_model, inputs=[], outputs=[save_lcm_model_txt, save_lcm_model_lora_scale, save_lcm_model_html])
    save_lcm_model_save_button.click(save_lcm_model, inputs=[save_lcm_model_txt, save_lcm_model_lora_scale,save_lcm_model_as_safetensors_check, save_lcm_model_fp16_check], outputs=[save_lcm_model_html]).then(str_no_model_loaded, None, model_list_html).then(update_grapptitle, None, app_title_label)


# ------------------------------------------------------------------------------------------------------------------

    
    # Add Lora Models section
    reload_lora_button.click(update_lora_model_list_dropdown, None, loradropdown)
    loaded_lora_list_refresh.click(update_loaded_lora_model_list_dropdown, None, loaded_loradropdown)
    lora_list_button.click(list_lora_model, None, lorahtml)
    lora_add_button.click(add_lora_model, inputs=[loradropdown, lora_scale_value, loraload_model_use_lcmscheduler], outputs=[lorahtml]).then(update_grapptitle, None, app_title_label)
    lora_change_weight_button.click(change_lora_model, inputs=[loaded_loradropdown, lora_scale_value], outputs=[lorahtml]).then(update_grapptitle, None, app_title_label)
    lora_delete_button.click(delete_all_lora_adapters, None, lorahtml).then(update_grapptitle, None, app_title_label)

# ------------------------------------------------------------------------------------------------------------------


    # MODEL VIEWERS
    # Safetensors Model Viewer section

    safeimageview_reload_button.click(set_modelcard_collapse, None, safe_modelcard).then(update_safe_model_image_list_dropdown, None, safeimageview_dropdown).then(set_modelcard_setcode, safeimageview_html2, outputs=[safeimageview_html2, safeimageedit_html2]).then(set_modelcard_hideedit_buttons, None, outputs=[safeimageview_view_button, safeimageview_save_button]) 
    safeimageview_button.click(set_modelcard_collapse, None, safe_modelcard).then(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb2], outputs=[safeimageview_html2, safeimageview_html]).then(set_modelcard_setcode, safeimageview_html2, outputs=[safeimageview_html2, safeimageedit_html2]).then(set_modelcard_hideedit_buttons, None, outputs=[safeimageview_view_button, safeimageview_save_button])
    safeimageview_edit_button.click(set_modelcard_editmode, inputs=[safeimageview_html2, safeimageedit_html2], outputs=[safeimageview_html2, safeimageedit_html2]).then(set_modelcard_showedit_buttons, None, outputs=[safeimageview_view_button, safeimageview_save_button])
    safeimageview_view_button.click(set_modelcard_viewmode, inputs=[safeimageview_html2, safeimageedit_html2], outputs=[safeimageview_html2, safeimageedit_html2])
    safeimageview_save_button.click(set_modelcard_viewmode, inputs=[safeimageview_html2, safeimageedit_html2], outputs=[safeimageview_html2, safeimageedit_html2]).then(save_safe_model_save, inputs=[safeimageview_dropdown, safeimageedit_html2], outputs=[safeimageview_html]).then(set_modelcard_hideedit_buttons, None, outputs=[safeimageview_view_button, safeimageview_save_button])



    safegallery_reload_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb2], outputs=[safeimageview_html2, safeimageview_html])
    safegallery_first_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb2], outputs=[safeimageview_html2, safeimageview_html])
    safegallery_prev_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb3], outputs=[safeimageview_html2, safeimageview_html])
    safegallery_next_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb4], outputs=[safeimageview_html2, safeimageview_html])
    safegallery_last_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb5], outputs=[safeimageview_html2, safeimageview_html])
    safegallery_firstb_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb2], outputs=[safeimageview_html2, safeimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    safegallery_prevb_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb3], outputs=[safeimageview_html2, safeimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    safegallery_nextb_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb4], outputs=[safeimageview_html2, safeimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    safegallery_lastb_button.click(show_safe_model_preview, inputs=[safeimageview_dropdown, hidden_numb5], outputs=[safeimageview_html2, safeimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")

# ------------------------------------------------------------------------------------------------------------------

    
    # Lora Model Viewer section

    loraimageview_reload_button.click(set_modelcard_collapse, None, lora_modelcard).then(update_lora_model_image_list_dropdown, None, loraimageview_dropdown).then(set_modelcard_setcode, loraimageview_html2, outputs=[loraimageview_html2, loraimageedit_html2]).then(set_modelcard_hideedit_buttons, None, outputs=[loraimageview_view_button, loraimageview_save_button]) 
    loraimageview_button.click(set_modelcard_collapse, None, lora_modelcard).then(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb2], outputs=[loraimageview_html2, loraimageview_html]).then(set_modelcard_setcode, loraimageview_html2, outputs=[loraimageview_html2, loraimageedit_html2]).then(set_modelcard_hideedit_buttons, None, outputs=[loraimageview_view_button, loraimageview_save_button])
    loraimageview_edit_button.click(set_modelcard_editmode, inputs=[loraimageview_html2, loraimageedit_html2], outputs=[loraimageview_html2, loraimageedit_html2]).then(set_modelcard_showedit_buttons, None, outputs=[loraimageview_view_button, loraimageview_save_button])
    loraimageview_view_button.click(set_modelcard_viewmode, inputs=[loraimageview_html2, loraimageedit_html2], outputs=[loraimageview_html2, loraimageedit_html2])
    loraimageview_save_button.click(set_modelcard_viewmode, inputs=[loraimageview_html2, loraimageedit_html2], outputs=[loraimageview_html2, loraimageedit_html2]).then(save_lora_model_save, inputs=[loraimageview_dropdown, loraimageedit_html2], outputs=[loraimageview_html]).then(set_modelcard_hideedit_buttons, None, outputs=[loraimageview_view_button, loraimageview_save_button])



    loragallery_reload_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb2], outputs=[loraimageview_html2, loraimageview_html])
    loragallery_first_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb2], outputs=[loraimageview_html2, loraimageview_html])
    loragallery_prev_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb3], outputs=[loraimageview_html2, loraimageview_html])
    loragallery_next_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb4], outputs=[loraimageview_html2, loraimageview_html])
    loragallery_last_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb5], outputs=[loraimageview_html2, loraimageview_html])
    loragallery_firstb_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb2], outputs=[loraimageview_html2, loraimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    loragallery_prevb_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb3], outputs=[loraimageview_html2, loraimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    loragallery_nextb_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb4], outputs=[loraimageview_html2, loraimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    loragallery_lastb_button.click(show_lora_model_preview, inputs=[loraimageview_dropdown, hidden_numb5], outputs=[loraimageview_html2, loraimageview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")


# ------------------------------------------------------------------------------------------------------------------

    
    # LCM-LoRA Model Viewer section
    lcmmodelview_reload_button.click(set_modelcard_collapse, None, lcm_modelcard).then(update_lcm_model_image_list_dropdown, None, lcmmodelview_dropdown).then(set_modelcard_setcode, lcmmodelview_html2, outputs=[lcmmodelview_html2, lcmmodeledit_html2]).then(set_modelcard_hideedit_buttons, None, outputs=[lcmmodelview_view_button, lcmmodelview_save_button]) 
    lcmmodelview_button.click(set_modelcard_collapse, None, lcm_modelcard).then(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb2], outputs=[lcmmodelview_html2, lcmmodelview_html]).then(set_modelcard_setcode, lcmmodelview_html2, outputs=[lcmmodelview_html2, lcmmodeledit_html2]).then(set_modelcard_hideedit_buttons, None, outputs=[lcmmodelview_view_button, lcmmodelview_save_button])
    lcmmodelview_edit_button.click(set_modelcard_editmode, inputs=[lcmmodelview_html2, lcmmodeledit_html2], outputs=[lcmmodelview_html2, lcmmodeledit_html2]).then(set_modelcard_showedit_buttons, None, outputs=[lcmmodelview_view_button, lcmmodelview_save_button])
    lcmmodelview_view_button.click(set_modelcard_viewmode, inputs=[lcmmodelview_html2, lcmmodeledit_html2], outputs=[lcmmodelview_html2, lcmmodeledit_html2])
    lcmmodelview_save_button.click(set_modelcard_viewmode, inputs=[lcmmodelview_html2, lcmmodeledit_html2], outputs=[lcmmodelview_html2, lcmmodeledit_html2]).then(save_lcm_model_save, inputs=[lcmmodelview_dropdown, lcmmodeledit_html2], outputs=[lcmmodelview_html]).then(set_modelcard_hideedit_buttons, None, outputs=[lcmmodelview_view_button, lcmmodelview_save_button])


    lcmgallery_reload_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb2], outputs=[lcmmodelview_html2, lcmmodelview_html])
    lcmgallery_first_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb2], outputs=[lcmmodelview_html2, lcmmodelview_html])
    lcmgallery_prev_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb3], outputs=[lcmmodelview_html2, lcmmodelview_html])
    lcmgallery_next_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb4], outputs=[lcmmodelview_html2, lcmmodelview_html])
    lcmgallery_last_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb5], outputs=[lcmmodelview_html2, lcmmodelview_html])
    lcmgallery_firstb_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb2], outputs=[lcmmodelview_html2, lcmmodelview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    lcmgallery_prevb_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb3], outputs=[lcmmodelview_html2, lcmmodelview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    lcmgallery_nextb_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb4], outputs=[lcmmodelview_html2, lcmmodelview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    lcmgallery_lastb_button.click(show_lcm_model_preview, inputs=[lcmmodelview_dropdown, hidden_numb5], outputs=[lcmmodelview_html2, lcmmodelview_html]).then(None, None, None, js="() => { window.scrollTo({top: 0}); }")

# ------------------------------------------------------------------------------------------------------------------


    # TOOLS
    
# ------------------------------------------------------------------------------------------------------------------

    # Model Merge section
    
    model_type.change(fn=update_slider_visibility,inputs=model_type,outputs=sliders)

    merge_btn.click(fn=block_merge,inputs=[model_a,model_b,fp16,out_fp16,out_safe,model_type,merged_model_name,text_alpha,vae_alpha,*sliders],outputs=merge_status_html)
    merge_goto_top_btn.click(None, None, None, js="() => { window.scrollTo({top: 0}); }")

    apply_preset_btn.click(fn=apply_preset,inputs=[preset_dropdown],outputs=sliders)

    save_profile_btn.click(fn=save_profile,inputs=[profile_name,profile_description,text_alpha,vae_alpha,model_type,*sliders],outputs=loaded_profile_html)
    load_profile_btn.click(fn=load_profile,inputs=[load_profile_dropdown],outputs=[model_type, loaded_profile_html, profile_name, profile_description, text_alpha, vae_alpha] + sliders)    
    refresh_model_btn.click(update_merge_model_list_dropdown,None,outputs=[model_a, model_b])    
    refresh_profile_btn.click(update_profile_list_dropdown,None,load_profile_dropdown)    

# ------------------------------------------------------------------------------------------------------------------


    # Safetensors Viewer section
    safetool_reload_button.click(update_safe_model_list_dropdown, None, safetool_dropdown)
    safetool_delete_button.click(delete_safe_model, inputs=[safetool_dropdown,safetool_delete_model_check,safetool_delete_images_check], outputs=[safetool_html2, safetool_html])

# ------------------------------------------------------------------------------------------------------------------

    
    # Lora Models section
    loratool_reload_button.click(update_lora_model_list_dropdown, None, loratool_dropdown) 
    loratool_delete_button.click(delete_lora_model, inputs=[loratool_dropdown,loratool_delete_model_check,loratool_delete_images_check], outputs=[loratool_html2, loratool_html])

# ------------------------------------------------------------------------------------------------------------------

    
    # LCM-LoRA Models section
    lcmtool_reload_button.click(update_lcm_model_list_dropdown, None, lcmtool_dropdown) 
    lcmtool_delete_button.click(delete_lcm_model, inputs=[lcmtool_dropdown,lcmtool_delete_model_check,lcmtool_delete_images_check], outputs=[lcmtool_html2, lcmtool_html])



# ------------------------------------------------------------------------------------------------------------------

    
    # HUB Models section
    # HUB - HUGGGINFACE Local Cache Model section
    hub_tool_reload_button.click(update_hub_model_list_dropdown, None, hub_tool_dropdown) 
    hub_tool_delete_button.click(delete_hub_model, inputs=[hub_tool_dropdown, hub_tool_delete_model_check], outputs=[hub_tool_html2])




# ------------------------------------------------------------------------------------------------------------------

    
    # Download Huggingface Models section
    hug_downloadmodel_button.click(download_huggingface_model, hug_download_model_txt, hug_downloadmodel_html2).then(update_grapptitle, None, app_title_label) 

   
   
# ------------------------------------------------------------------------------------------------------------------

    
    # System Info section
    
    update_cpumemswap_mem.click(update_grapptitle_mem, None, app_title_label) 
    update_cpumemswap_info.click(update_grapptitle, None, app_title_label) 
    sysinfo_cpumemswap_button.click(get_system_stats, None, sysinfo_html) 
    sysinfo_memory_button.click(get_sysinfo_memory, None, sysinfo_html) 
    sysinfo_hfcache_button.click(get_sysinfo_hfcache, None, sysinfo_html) 
    sysinfo_env_button.click(get_sysinfo_env, None, sysinfo_html) 
    sysinfo_sysinfo_button.click(get_sysinfo_sysinfo, None, sysinfo_html) 
   
    sysmodels_goto_top_button.click(None, None, None, js="() => { window.scrollTo({top: 0}); }")
    sysmodels_download_button.click(sysmodel_start_download, inputs=[dlm1,dlm2,dlm3,dlm4,dlm5,dlm6,dlm7,dlm8,dlm9,dlm10,dlm11,dlm12,dlm13,dlm14,dlm15,dlm16,dlm17,dlm18,dlm19,dlm20,dlm21,dlm22,dlm23], outputs=[sysinfo_html,dlm1,dlm2,dlm3,dlm4,dlm5,dlm6,dlm7,dlm8,dlm9,dlm10,dlm11,dlm12,dlm13,dlm14,dlm15,dlm16,dlm17,dlm18,dlm19,dlm20,dlm21,dlm22,dlm23]) 
    sysmodels_cancel_button.click(sysmodel_cancel_download, sysinfo_html, sysinfo_html) 
    sysmodels_uncheckall_button.click(sysmodels_uncheckall_checkboxes, None, outputs=[dlm1,dlm2,dlm3,dlm4,dlm5,dlm6,dlm7,dlm8,dlm9,dlm10,dlm11,dlm12,dlm13,dlm14,dlm15,dlm16,dlm17,dlm18,dlm19,dlm20,dlm21,dlm22,dlm23]) 
    sysmodels_checkdefaults_button.click(sysmodels_checkdefaults_checkboxes, None, outputs=[dlm1,dlm2,dlm3,dlm4,dlm5,dlm6,dlm7,dlm8,dlm9,dlm10,dlm11,dlm12,dlm13,dlm14,dlm15,dlm16,dlm17,dlm18,dlm19,dlm20,dlm21,dlm22,dlm23]) 
    
    
    sysinfo_hug_on_button.click(huggingface_on_app, None, sysinfo_hug_status) 
    sysinfo_hug_off_button.click(huggingface_off_app, None, sysinfo_hug_status) 
    sysinfo_hug_check_button.click(huggingface_check_status_app, None, sysinfo_hug_status) 
    
    
    sysinfo_haltgen_button.click(halt_generation, inputs=[], outputs=[])
    sysinfo_exit_button.click(exit_app)
    sysinfo_restart_button.click(restart_app)
    sysinfo_reload_button.click(None, None, None, js="() => { window.location.reload(true); }")
    sysinfo_logout_button.click(None, None, None, js="() => { window.location.href = '/logout'; }")

    if LLSTUDIO["current_os"] == "Linux":
        sysinfo_sudo_shutdown_button.click(sudo_shutdown)
        sysinfo_sudo_reboot_button.click(sudo_reboot)

# ------------------------------------------------------------------------------------------------------------------
 
 
    # # Settings section
    # # # saved parameters go in, output is simple status report to a box...
    settings_save_button.click(update_settings, inputs=gr_components, outputs=[settings_status_text, settings_status_text2])
    settings_save_button2.click(update_settings, inputs=gr_components, outputs=[settings_status_text, settings_status_text2])
    settings_goto_top_button.click(None, None, None, js="() => { window.scrollTo({top: 0}); }")

 
# ------------------------------------------------------------------------------------------------------------------

    
    # Help/About section
    # N/A
    # Help is a link to an HTML page in the 'help' directory.
    # Uses server already running as a web server. 
    # Although slow during inference...




# ------------------------------------------------------------------------------------------------------------------
# Define launch keyword arguments in a dictionary
launch_kwargs = {}
launch_kwargs["share"] = False
launch_kwargs["server_name"] = STUDIO["server_name"]["value"]
launch_kwargs["server_port"] = int(STUDIO["server_port"]["value"])


if STUDIO["app_autolaunch"]["value"]:
    launch_kwargs["inbrowser"] = True
else:
    launch_kwargs["inbrowser"] = False

launch_kwargs["allowed_paths"] = all_allowed_file_paths

launch_kwargs["favicon_path"] = "favicon.ico"

lcmlorastudio.launch(**launch_kwargs)


# --- ui end ---



# ================================================================================
# =======END APP====END APP====END APP====END APP====END APP====END APP===========
# ================================================================================



# -EOF-




