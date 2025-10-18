# lcmlorastudio_v1.py

# ---------------------------------------------------
# ---------------LCM-LoRA Studio---------------------
# -----------------Version 1.3-----------------------
# ---------------------------------------------------


# base libraries, imports for ui
from diffusers.training_utils import set_seed
# ---------------------------------
# added 'ImageEnhance' and 'ImageOps' imports
# which was originally : from PIL import Image
# this was added for the Image Processing section of this app
from PIL import Image, ImageEnhance, ImageOps

# ---------------------------------
import gradio as gr
import time 
import os 
import sys
import random
from gradio_client import Client

# ---------------------------------
# base libraries, imports for Diffusion
import argparse
from diffusers import DiffusionPipeline
import torch

# -------------------------------
# base libraries, imports for SD Pipelines and Scheduler
import string
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline, LCMScheduler)

# --------------------------------
# for ControlNet only
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ------------------------------------------------------------
# gallery
import re           # used by Compel prompt helper I made
import pathlib
from pathlib import Path
import glob
import html

# ------------------------------------------------------------
# clipboard across platforms/os's
# access OS clipboard
import pyperclip

# ------------------------------------------------------------
# copy last prompt and last image to gallery
# file copy
import shutil

# -------------------------------------------------
# t2i, i2i, inp NEW imports
# from diffusers import AutoPipelineForText2Image
# from diffusers import AutoPipelineForImage2Image
# from diffusers import AutoPipelineForInpainting

# -------------------------------------------------
# ip2p NEW imports
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionXLInstructPix2PixPipeline
# using this will return the correct format to send the image to inference RGB/PIL, etc...
from diffusers.utils import load_image


# -------------------------------------------------

# -------------------------------------------------
# imports, read model config settings
import json

# LLSTUDIO import a few starting variables
# although most are built from multiple STUDIO variables later
from config import LLSTUDIO
# ----------------------------------------
# ----------------------------------------
# imports, faster math
import pandas as pd
# ------------------------------------------
# imports, image processing
import numpy as np
# ------------------------------------------

# -------------------------------------------------
# diffusers verbose output control inports
from diffusers import logging

# logging.set_verbosity_error() 
# diffusers.logging.disable_progress_bar()
# diffusers.logging.enable_progress_bar()

# context manager to ignore the UserWarning during model loading
import warnings
# ---------------------------------
# get processor info and OS info, memory garbage collection etc...
import platform
import gc

# ---------------------------------------------------------
# download huggingface repository/model
from huggingface_hub import snapshot_download

# ---------------------------------
# get memory info 
import psutil

# ---------------------------------
# image sd upscale
from diffusers import StableDiffusionLatentUpscalePipeline

# ---------------------------------
# load seperate text encoder
import transformers
from transformers import CLIPTextModel

# ----------------------------------------------
# # for depth_estimator
# # we do not use "transformers import pipeline()" for the 'depth_estimator'
# # to not confuse it with our global 'pipeline' we load our models into
# # To load the 'depth_estimator', we use the transformers.pipeline()
# # Since we already have the 'import transformers' line
# # to not get it confused with the global 'pipeline' for our main model pipeline
# from transformers import pipeline

# # ---------------------------------
# # rknote do we still use this?
# # for download single file from URL
# # was to be used by download safetensors file from url...
# # but download straight from some sites, need an API key so, will not work
# # may use in future for future mod
# import requests

# ----------------------------------------
# for date/time in image filename
from datetime import datetime

# ----------------------------------------
# OpenCV for image processing section, to do edge detection.
# although already an apparent 'requirement' for 
# some other library already used/installed
# but we still need to import it ! :)
# import cv2
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not found. So, Canny edge detection will be forced to use a slower processing method")


# ---------------------------------
# LLSTUDIO imports
#----------------------------------
# ===========DICT-BASED-APP-SETTINGS-LOAD-SAVE-FUNCTIONS=======================
# App settings are a dict named 'GRSETTINGS' in 'configuration.py'
# Only populated with default setting filename
# The rest of the settings are loaded from the default settings file into the dict
# default setting filename 'value' (stored in the dict): 'configurationdictsjson.json'
# ie... GRSETTINGS["settings_file"]["value"]
# main app settings, linked to the 'settings' tab in ui
from config import STUDIO, load_settings, save_settings, print_settings, update_settings, create_settings_ui
# simple sd pipeline class to model type, generation/pipeline mode - lookup table
from config import PIPECLASSES
# status of our single pipeline, model loaded?, which one?, text encoder?, type?, etc...
from config import SDPIPELINE
# class list for pipeline class dropdown boxes
from config import PIPELINE_CLASSES
# ---------------------------------

# ---------------------------------------
# imports for COMPEL prompt parsing
# this import is conditional
# the program can run without the prompt 'weighting' this library provides
# yet your image output will gain from using it.
# I hate using the phrase experimental in software, it either works or not.
# so it is NOT, yet breaking changes breaks things, and given examples, not all worked.
LLSTUDIO["compel_installed"] = 0      # default - not installed
try:
    from compel import Compel, ReturnedEmbeddingsType
    LLSTUDIO["compel_installed"] = 1
except ImportError:
    LLSTUDIO["compel_installed"] = 0



# --------------------------------------------------------
# - this was to overcome no server till logged in, 
# - so i loaded the logo to a base64...
import base64


# ---------------------------------
# for custom 3rd party safety checker    
#from transformers import pipeline
# # but we have already imported it for diffusers inference use

# ----------end of imports-------------------------

# ----------start of settings-------------------------
# we do settings first, then set up all the variables

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# load the app settings from the settings file and update 'STUDIO'
STUDIO.update(load_settings())


# ----------end of settings-------------------------



# ----------start of setup variables from settings-------------------------
# these variables are 'built' from the 'STUDIO' settings.
# some 'STUDIO' setting are used directly, some are used combied with 
# other's to build some of the variables. ie... fullpath = root_dir/dir
# flag to halt multi image generation, after current inference is finished
# rknote, MIGHT-POSSIBLY... be replaced by callback function during inference
# to interupt the generation.
LLSTUDIO["halt_gen"] = 0

# ====================================================

# # rknote need MORE debug levels more like 
# # 0=nothing 1=app 2=important info 3=superflurious info 4=ALL info (TMI)
# debug level - turn on/off printing info to stdout
# 0 = Nothing out from app controled print output
# 1 = app print outputs important info
# 2 = app print outputs important info + superflurious model loading output... TMI
# # Example debug usage:
if int(STUDIO["app_debug"]["value"]) > 0: print("Hello") ### RKREMOVED in Version: vsb ###


# used for help, should be '.'
LLSTUDIO["root_dir"] = STUDIO["root_dir"]["value"]


# ---------------------------------------
# PIPELINES
# ---------------------------------------
# GLOBAL SD/SDXL PIPELINE
pipeline = ""   # the pipeline where all models are loaded to, the ONLY 'model' pipeline

# ---------------------------------------

# if on startup, we know we are using the safety checker, go ahead and load the model for it.
# the 'safety_checker_pipeline' is global only to the 6 inference functions t2i,i2i,inp,ip2p,up2x,cnet
if STUDIO["use_safety_checker"]["value"]: 
    if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
    try:
        safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
        SDPIPELINE["pipeline_safety_checker_loaded"] = 1
        if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
    except Exception as e:
        SDPIPELINE["pipeline_safety_checker_loaded"] = 0
        if int(STUDIO["app_debug"]["value"]) > 0: print("Error Loading Image Classifier Model... '" + STUDIO['safety_checker_model_name']['value'] + "'")

# ---------------------------------------

LLSTUDIO["lcm_model_list"]=['NO MODEL', 'NO MODEL']
LLSTUDIO["lcm_sdonly_model_list"] = ['NO MODEL', 'NO MODEL']

LLSTUDIO["lcm_model_dir"] = os.path.join(STUDIO["lcm_model_rootdir"]["value"], STUDIO["lcm_model_dir"]["value"])


# ----------------------------------------------------------------------

# # Huggingface Cache problems, chicken before the egg?, or just oversight?, idk...
# # On Windows I've have had no problems with HF_HOME or HF_HUB_CACHE env vars.
# # but I control the cache location on Windows, to somewhere other than C:\Users...
# # On every install on Pi5, no enviroment variables are set
# # the huggingface libs apparently do not need them. So hacks performed to control 
# # other hug/hub diffusers cached torch etc... I found it is basically dynamically coded
# # below the current user's directories, IF, HF_HOME AND HF_HUB_CACHE is NOT found.
# # Library uses the dafault location of HF_HUB_CACHE:
# # Linux/macOS: '~/.cache/huggingface/hub'
# # Windows: 'C:\Users\username\.cache\huggingface\hub'
# # Because...
# # In LCM-LoRA Studio you can just load a model from your Hub Cache, from a simple dropdown.
# # You enter it by name once to get it, then it's cached. So why enter it again? Each time?
# #
# # NOTE: IF 'hub_model_dir' in settings is EMPTY, it will use the ENVIROMENT VARS to find it.
# # IF, it can not find it, you have to set it in settings......Settings OVERRIDES Huggingface !!
#

LLSTUDIO["hub_model_list"]=['NO MODEL', 'NO MODEL']

if STUDIO["hub_model_dir"]["value"] != "":
    LLSTUDIO["hub_model_dir"] = os.path.join(STUDIO["hub_model_rootdir"]["value"], STUDIO["hub_model_dir"]["value"])
else:
    hfhubcache = os.getenv('HF_HUB_CACHE', 'None')
    if (hfhubcache != "None" and os.path.isdir(hfhubcache)):
        if int(STUDIO["app_debug"]["value"]) > 0: print("'HF_HUB_CACHE' enviroment variable FOUND, using 'HF_HUB_CACHE' location:")
        if int(STUDIO["app_debug"]["value"]) > 0: print(hfhubcache)
        LLSTUDIO["hub_model_dir"] = hfhubcache
    else:
        if int(STUDIO["app_debug"]["value"]) > 0: print("'HF_HUB_CACHE' NOT FOUND, trying 'HF_HOME'")
        hfhomecache = os.getenv('HF_HOME', 'None')
        if (hfhomecache != "None" and os.path.isdir(hfhomecache)):
            if int(STUDIO["app_debug"]["value"]) > 0: print("'HF_HOME' enviroment variable FOUND, using 'HF_HUB_CACHE' location:")
            if int(STUDIO["app_debug"]["value"]) > 0: print(hfhomecache)
            LLSTUDIO["hub_model_dir"] = hfhomecache
        else:
            if int(STUDIO["app_debug"]["value"]) > 0: print("'HF_HOME' NOT FOUND, trying OS default")
            if (platform.system() != "Windows" and os.path.isdir("~/.cache/huggingface/hub")):
                if int(STUDIO["app_debug"]["value"]) > 0: print("'Using OS default for Hub Cache location:")
                if int(STUDIO["app_debug"]["value"]) > 0: print("~/.cache/huggingface/hub")
                LLSTUDIO["hub_model_dir"] = "~/.cache/huggingface/hub"
            else:
                LLSTUDIO["hub_model_dir"] = "None"

# So the final result will be (in this order):
# LLSTUDIO["hub_model_dir"]=HF_HUB_CACHE    #found
# LLSTUDIO["hub_model_dir"]=HF_HOME    #found
# LLSTUDIO["hub_model_dir"]=OS default    #found
# LLSTUDIO["hub_model_dir"]=None    #not found - set the location in settings


# ----------------------------------------------------------------------


LLSTUDIO["lcm_model_prefix"] = STUDIO["lcm_model_prefix"]["value"]
LLSTUDIO["lcm_model_suffix"] = STUDIO["lcm_model_suffix"]["value"]

LLSTUDIO["lcm_model_image_list"]=['NO MODEL', 'NO MODEL']

LLSTUDIO["lcm_model_image_dir"] = os.path.join(STUDIO["lcm_model_image_rootdir"]["value"], STUDIO["lcm_model_image_dir"]["value"])
data_lcmdir_path =  Path(LLSTUDIO["lcm_model_image_dir"])

LLSTUDIO["safe_model_list"] = ['NO MODEL', 'NO MODEL']

LLSTUDIO["safe_model_dir"] = os.path.join(STUDIO["safe_model_rootdir"]["value"], STUDIO["safe_model_dir"]["value"])

LLSTUDIO["safe_model_image_list"] = ['NO MODEL', 'NO MODEL']

LLSTUDIO["safe_model_image_dir"] = os.path.join(STUDIO["safe_model_image_rootdir"]["value"], STUDIO["safe_model_image_dir"]["value"])
data_safedir_path =  Path(LLSTUDIO["safe_model_image_dir"])

LLSTUDIO["lora_model_list"]=['NO MODEL', 'NO MODEL']

LLSTUDIO["lora_model_dir"] = os.path.join(STUDIO["lora_model_rootdir"]["value"], STUDIO["lora_model_dir"]["value"])

LLSTUDIO["lora_model_image_list"]=['NO MODEL', 'NO MODEL']

LLSTUDIO["lora_model_image_dir"] = os.path.join(STUDIO["lora_model_image_rootdir"]["value"], STUDIO["lora_model_image_dir"]["value"])
data_loradir_path =  Path(LLSTUDIO["lora_model_image_dir"])

LLSTUDIO["output_image_dir"] = os.path.join(STUDIO["output_image_rootdir"]["value"], STUDIO["output_image_dir"]["value"])
data_outputdir_path =  Path(LLSTUDIO["output_image_dir"])


# HELP
LLSTUDIO["help_path"] = os.path.join(LLSTUDIO["root_dir"], "help")
help_path =  Path(LLSTUDIO["help_path"])
# EX URL: http://127.0.0.1:7860/gradio_api/file/help/seed.html


# all_allowed_file_paths = [str(data_safe_dir_path.absolute()),str(data_lcmdir_path.absolute())]
# all_allowed_file_paths = [str(data_safe_dir_path.absolute()),str(data_lcmdir_path.absolute()),str(data_loradir_path.absolute())]
# all_allowed_file_paths = [str(data_safedir_path.absolute()),str(data_lcmdir_path.absolute()),str(data_loradir_path.absolute())]
# all_allowed_file_paths = [str(help_path.absolute()),str(data_safedir_path.absolute()),str(data_lcmdir_path.absolute()),str(data_loradir_path.absolute()),str(data_outputdir_path.absolute())]
all_allowed_file_paths = [str(help_path.absolute()),str(data_safedir_path.absolute()),str(data_lcmdir_path.absolute()),str(data_loradir_path.absolute()),str(data_outputdir_path.absolute())]

LLSTUDIO["def_prompt"] = STUDIO["def_prompt"]["value"]

LLSTUDIO["def_negprompt"] = STUDIO["def_negprompt"]["value"]

LLSTUDIO['last_prompt_filename'] = ""

LLSTUDIO['last_negative_prompt_filename'] = ""

LLSTUDIO['last_image_filename'] = ""

LLSTUDIO["outputfolder"] = os.path.join(STUDIO["outputfolder_rootdir"]["value"], STUDIO["outputfolder"]["value"])

LLSTUDIO["output_image_prefix"] = STUDIO["output_image_prefix"]["value"]

LLSTUDIO["output_image_suffix"] = STUDIO["output_image_suffix"]["value"]

# only lora add model below here...
# ----------------------------------------
LLSTUDIO["lora_adapter_numb"] = 0
LLSTUDIO["loaded_lora_model_value"] = []
LLSTUDIO["loaded_lora_model_name"] = []
LLSTUDIO["loaded_lora_model_adapter"] = []
LLSTUDIO["loaded_lora_model_list"] = []

# # ====================================================================================



# -----------generic stuff--------------
# logging.set_verbosity_error() 
# diffusers.logging.disable_progress_bar()
# diffusers.logging.enable_progress_bar()

# if int(STUDIO["app_debug"]["value"]) < 2: logging.set_verbosity_critical() 
if int(STUDIO["app_debug"]["value"]) < 2: logging.set_verbosity_error() 
# # Suppress all warnings from the 'transformers' library
# if int(STUDIO["app_debug"]["value"]) < 2: logging.getLogger('transformers').setLevel(logging.ERROR) 

# if int(STUDIO["app_debug"]["value"]) < 2: warnings.filterwarnings("ignore", category=UserWarning)

# # Disable the progress bar
# pipeline.disable_progress_bar()
# # Enable the progress bar
# pipeline.enable_progress_bar()
# # Enviroment Variable
# HF_DATASETS_DISABLE_PROGRESS_BARS=1


# holds a global page number to remember for the image gallery, one for, each gallery.
LLSTUDIO["output_page_num"]=1
LLSTUDIO["lcm_page_num"]=1
LLSTUDIO["safe_page_num"]=1
LLSTUDIO["lora_page_num"]=1


# must be lower case?
LLSTUDIO["device"] = "cpu"
# our friendly name is upper case once it is been decided
LLSTUDIO["friendly_device_name"] = "CPU"


# enables/disables hidden image to visible image on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
# it's a FLAG, don't touch it !
LLSTUDIO["hidden_image_flag"] = 0





# ======================================================================================
# advanced image gallery

LLSTUDIO["advanced_gallery_dir"] = os.path.join(STUDIO["advanced_gallery_root"]["value"], STUDIO["advanced_gallery_dir"]["value"])
LLSTUDIO["gallery_selected_image"] = ""

# ===================================================================================
# ===================================================================================

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

# EDGEFILTERS = ["Canny", "Laplacian", "Prewitt", "Scharr", "Sobel", "Simple Gradient", "Canny (Numpy)", "Laplacian (Numpy)", "Prewitt (Numpy)", "Roberts Cross (Numpy)", "Sobel (Numpy)"]
EDGEFILTERS = ["Canny", "Laplacian", "Scharr", "Sobel", "Simple Gradient", "Canny (Numpy)", "Laplacian (Numpy)", "Prewitt (Numpy)", "Roberts Cross (Numpy)", "Sobel (Numpy)"]

# ------------------------------------

# Folders for files
LLSTUDIO["imgp_file_dir"] = os.path.join(STUDIO["imgp_files_root"]["value"], STUDIO["imgp_files_dir"]["value"])

# ------------------------------------------------------
# Lists of files
LLSTUDIO["imgp_file_list"] = []

#----------------------------------------
# needed for the NO IMAGE image, and list of filters for the dropdown
LLSTUDIO["no_image"] = "no_image.png"

# ----------end of defines--------------------

# ----------end of setup variables from settings---------------------------



# # ====================================================================================
# # ======START========FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====
# # ====================================================================================



# # ==============================================================
# # ==============================================================
# # ==============================================================
# # ==============================================================
# # START Image Processing Functions
# # ==============================================================
# # ==============================================================
# # ==============================================================
# # ==============================================================

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# -------------------------
# transformers depthmap
# -------------------------

def do_depth_map(img):
    # Check for input image
    if img is None:
        return None

    # convert to PIL, that what transformers wants to see
    depth_image = numpy_to_pil(img)

    # load depth_estimator, we use the transformers.pipeline()
    # to not get it confused with the global 'pipeline' for our main model pipeline
    depth_estimator = transformers.pipeline('depth-estimation')
 
    # run depth_estimator
    depth_output = depth_estimator(depth_image)['depth']

    # return the image as it is.
    return depth_output



# ====================================================================
# ====================================================================
# ====================================================================
# START - NUMPY ONLY IMAGE PROCESSING UTILS
# ====================================================================
# ====================================================================
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
# ====================================================================
# ====================================================================
# END - NUMPY ONLY IMAGE PROCESSING UTILS
# ====================================================================
# ====================================================================
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

    # Determine mode from shape
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



# ==================================================
# NO NUMPY below here....
# OPENCV is below here...
# ==================================================


# --------------------------------------------------------
# this function does edge detecion and sharpening.
# input:  grayscale image (numpy.ImageArrary)
# output: monochrome image (numpy.ImageArrary)
def process_image(image: np.ndarray, 
                  method: str = 'canny', 
                  sharpen: bool = False,
                  **kwargs) -> np.ndarray:

    #Validate input
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy ndarray.")
    if image.ndim != 2:
        raise ValueError("Image must be a 2D monochrome (grayscale) array.")

    #Clone the image to work on
    result = image.copy()

    #Apply edge detection
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
        # For visual display, it's often beneficial to take the absolute values
        # and then convert to an 8-bit unsigned integer type.
        prewitt_edges = np.sqrt(gradient_x**2 + gradient_y**2)
        result = cv2.normalize(prewitt_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)



    elif method == 'none':
        pass  # No edge detection

    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'canny', 'sobel', 'laplacian', 'scharr', 'prewitt' or 'none'.")

    #Apply sharpening if requested
    if sharpen:
        # Sharpening kernel
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
        result = cv2.filter2D(result, -1, kernel)

    return result

# --------------------------------------------------------
# conversion from color to adjusted color image
# Adjust image (pre-process)
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
    # Check for input image
    if img is None:
        return None
    img = np.array(img).astype(np.float32)
    grayscale = (img[:, :, 0] * r_weight + img[:, :, 1] * g_weight + img[:, :, 2] * b_weight).astype(np.uint8)
    return grayscale

# -------------------------------------------------------------------------------------------------

# Convert to monochrome
def convert_to_monochrome(img, lower_thresh=100, upper_thresh=200, invert=False):
    # Check for input image
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

# rkadded to invert final output image
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


# rkadded to open an image an return to rest of app as a numpy array
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

# image processing pipeline
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
# =======================================================================
# Functions called by the gradio ui in Image Processing
# =======================================================================
# =======================================================================

def send_to_controlnet(img):

    # Check for input image
    if img is None:
        gr.Info("No Valid Image to Send to ControlNet!!<br>Please Process an Image.", duration=5.0, title="Send to ControlNet")
        no_image = load_image_as_numpy(LLSTUDIO["no_image"])
        return no_image, "No Valid Image to Send to ControlNet!! Please Process an Image."

    return numpy_to_pil(img), "Sucessfully Sent Image to ControlNet."


# -------------------------------------------------------------------------------------------------

# saving the numpy type image to a png file
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

# saving the numpy type image to a png file
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
        # pil_image.save(filepathname, format="PNG")
        # pil_image.save(filepathname, format="JPEG")
        return f"Saved {image_input_loc} Image: {filename}"
    except Exception as e:
        return f"Error saving {image_input_loc} Image '{filename}': {e}"
    


# ------------------------------------------------------


# just reloads imgp_file_list[] - called to refresh imgp_file_list[] items
def imgp_get_file_list():
    LLSTUDIO["imgp_file_list"] = []
    entries = sorted([f for f in os.listdir(LLSTUDIO["imgp_file_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["imgp_file_dir"], f))])
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get file with extension .PNG only
        if (tmp_text.lower().endswith('.png') or tmp_text.lower().endswith('.jpg') or tmp_text.lower().endswith('.jpeg')):
            LLSTUDIO["imgp_file_list"].append(tmp_text)

    return LLSTUDIO["imgp_file_list"]


# ------------------------------------------------------

def imgp_refresh_file_list_dropdown():
    imgp_get_file_list()
    return gr.Dropdown(choices=LLSTUDIO["imgp_file_list"], interactive=True)
    
# =======================================================================  

# # ==============================================================
# # ==============================================================
# # ==============================================================
# # ==============================================================
# # END Image Processing Functions
# # ==============================================================
# # ==============================================================
# # ==============================================================
# # ==============================================================


# # ==============================================================
# # Start LLSTUDIO Functions
# # ==============================================================


# get CPU or GPU(CUDA) device. Can also (preference) select which GPU card by index#, base=0
def get_device(gpu_index):
    if gpu_index != "cpu":
        # check for cuda, if not found use the CPU instead
        if torch.cuda.is_available():
            # If you have more than one GPU...
            # To select the second GPU (index 1)
            # specify a particular GPU by its index
            # (0 for the first, 1 for the second, and so on)
            # ie... gpu_index = 1 on the input to this function
            if gpu_index < torch.cuda.device_count():
                device = f"cuda:{gpu_index}"
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"Using specific CUDA GPU: {device}")
            else:
                if int(STUDIO["app_debug"]["value"]) > 0: print(f"GPU index {gpu_index} not available. Using default CUDA GPU or CPU.")
                if torch.cuda.is_available():
                    device = "cuda"
                    if int(STUDIO["app_debug"]["value"]) > 0: print("Using default CUDA GPU.")
                else:
                    device = "cpu"
                    if int(STUDIO["app_debug"]["value"]) > 0: print("Using CPU as default (CUDA not available).")
        else:
            device = "cpu"
            if int(STUDIO["app_debug"]["value"]) > 0: print("Using CPU (CUDA not available).")
    else:
        device = "cpu"
        if int(STUDIO["app_debug"]["value"]) > 0: print("Bypassing CUDA. Using CPU as default.")

    return device
    
    
# ============================================================
# ============================================================

def device_select():
    
    # # inference DEVICE selection
    LLSTUDIO["device"] = get_device(STUDIO["device_name"]["value"])
    
    if int(STUDIO["app_debug"]["value"]) > 0: print("Final Selected Device: " + LLSTUDIO['device'])

    # set what i think, a 'friendlier' device name for the user to see in the ui, all uppercase
    tmp_str = LLSTUDIO["device"]
    LLSTUDIO["friendly_device_name"] = tmp_str.upper()


# ============================================================
# ============================================================



def set_freeu_values(ins1, ins2, inb1, inb2):
    #rkconvert - NOT DONE
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        gr.Info("<h4>No Model Loaded. Please Load a Model First.</h4><h4>Select the tab 'Pipeline - Models' to load a model into the pipeline.</h4>", duration=3.0, title="FreeU Settings")
        return ins1, ins2, inb1, inb2
 
    if (SDPIPELINE['pipeline_model_type']=="SDXL"):
        return LLSTUDIO["freeu_sdxl_s1"], LLSTUDIO["freeu_sdxl_s2"], LLSTUDIO["freeu_sdxl_b1"], LLSTUDIO["freeu_sdxl_b2"]
    else:
        return LLSTUDIO["freeu_sd_s1"], LLSTUDIO["freeu_sd_s2"], LLSTUDIO["freeu_sd_b1"], LLSTUDIO["freeu_sd_b2"]


# ------------------------------------------------------------
# ----------------------------------------------------
# =============================================================================
# =============================================================================



# rk refixed5 func (modification/merge of 5 diff vers from RKAIv0.5 an AI LLM)
def do_prompt_embeds(device, pipeline, prompt, negative_prompt):

    # rk note still 'feel' it needs modification to handle no negative prompt, Hmmmm
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

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
	

# =============================================================================
# =============================================================================




# # SDXL-PROMPTS
# # PADDING + POOLED + EMBEDS
# # from rk_py_sdxl_embedded_prompts_4_lcmgen_v1.py
# tokenize and encode prompt
# rk refixed1 func (modification for diffusers SDXL with padding + pooled + embeds)
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

    return prompt_embeds, pooled_prompt_embeds


# ------------------------------------------------------


def halt_generation():
    #rkconvert - DONE
    LLSTUDIO["halt_gen"] = 1
    gr.Info("Generation Halted</br>Please wait for current inference to complete...", duration=5.0, title="Halt Generation")


# ---------------------------------

def format_seconds_strftime(seconds):
    #rkconvert - DONE
    time_tuple = time.gmtime(seconds)
    # formatted_time = time.strftime("%H hours, %M minutes, %S seconds", time_tuple)
    formatted_time = time.strftime("%M minutes, %S seconds", time_tuple)
    return formatted_time


# ==================================================================================================================


# ------------------------------------------------------------


# Function to read the model file information from the 'modelfilename.txt'
def preview_get_model_info_file(file):
    #rkconvert - DONE
    file = open(file, "r")
    content = file.read()
    file.close()
    return content


# ------------------------------------------------------------



# original Function to read TEXT generation paramaters from the 'image-filename.txt'
def preview_create_text_code_org(file):
    #rkconvert - NOT DONE
    file = open(file, "r")
    content = ""
    while True:
        line = file.readline()
        if not line:
            break
        line = line.replace("Negative prompt: ","",1)
        content = content + "<br><br>" + "<code>" + line + "</code>"
    
    file.close()
    text_code = f'<code>{content}</code>'
    return text_code


# ------------------------------------------------------------



# newer rk Function to read TEXT generation paramaters from the 'image-filename.txt'
def preview_create_text_code(file):
    #rkconvert - NOT DONE
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



# Function to create HTML code to display the image and a link to open image in new window
def preview_create_html_code(file):
    #rk - DONE
    html_enc_file = file.replace(" ","%20")
    html_img_code = f"""<a href=#top>Go to TOP</a></br><img src="/gradio_api/file/{html_enc_file}" width="{STUDIO["img_view_img_width"]["value"]}%" height="auto" style="cursor:pointer" onclick="window.open(this.src)"></img>"""
    return html_img_code


# ---------------------------------------------------------------------------------------

# NOT USED YET... WIP
# rkwip new image/file/folder lister, may return list or text??? idk yet...
def get_image_list():
    #rkconvert - NOT DONE
    LLSTUDIO["safe_model_list"] = []
    entries = [f for f in os.listdir(LLSTUDIO["safe_model_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["safe_model_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        if tmp_text.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            tmp_model = os.path.splitext(os.path.basename(tmp_text))[0]
            LLSTUDIO["safe_model_list"].append(tmp_model)
    
    return "Safetensors Model List Reloaded."



# ---------------------------------------------------------------------------------------


# used ONLY for output viewer, going to try and hijack the code for the other 3 viewers
def get_output_image_list():
    #rkconvert - NOT DONE
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


# used ONLY for lcm model images viewer, going to try and hijack the code for the other 3 viewers
def get_lcm_image_list(modelname):
    #rkconvert - NOT DONE
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


# used ONLY for safetensors model images viewer, going to try and hijack the code for the other 3 viewers
def get_safe_image_list(modelname):
    #rkconvert - NOT DONE
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


# used ONLY for lora model images viewer, going to try and hijack the code for the other 3 viewers
def get_lora_image_list(modelname):
    #rkconvert - NOT DONE
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
# ------------------------------------------------------------------------------------------------------------------

# rknote - Returns a subset of items for a specific page
def paginate_list(items, page_number, page_size):
    start_index = (page_number - 1) * page_size
    end_index = start_index + page_size
    return items[start_index:end_index]


# ---------------------------------------------------------------


 
def show_output_preview(input_cmd):
    #rkconvert - WIP
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
                html_code_list = html_code_list + "<td align='top' width='5%'>"
                html_code_list = html_code_list + "<font size='+2'><button class='copy-button'>Copy</button></font>"
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





# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================

def set_modelcard_editmode(view_content, edit_content):
# swaps which is visible, markdown or code, in that order, for the modelcard
# ie... yield markdown, code
    yield gr.update(visible=False), gr.update(value=view_content, visible=True)

def set_modelcard_viewmode(view_content, edit_content):
# swaps which is visible, markdown or code, in that order, for the modelcard
# ie... yield markdown, code
    yield gr.update(value=edit_content, visible=True), gr.update(visible=False)


# collapes the Model Information Accordian
def set_modelcard_collapse():
    yield gr.update(open=False)

def set_modelcard_setcode(view_content):
# send view back to gr.Code code window for the modelcard after loading using .then()
# after loading model info
    yield gr.update(visible=True), gr.update(value=view_content, visible=False)

def set_modelcard_hideedit_buttons():
# send view back to gr.Code code window for the modelcard after loading using .then()
# after loading model info
    yield gr.update(visible=False), gr.update(visible=False)

def set_modelcard_showedit_buttons():
# send view back to gr.Code code window for the modelcard after loading using .then()
# after loading model info
    yield gr.update(visible=True), gr.update(visible=True)

# ======================================================================

def show_lcm_model_preview(modelname, input_cmd):
    #rkconvert - WIP
    if input_cmd == 0:
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
                html_code_list = html_code_list + "<td align='top' width='5%'>"
                html_code_list = html_code_list + "<font size='+2'><button class='copy-button'>Copy</button></font>"
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



# ======================================================================
# ======================================================================



def show_safe_model_preview(modelname, input_cmd):
    #rkconvert - WIP
    if input_cmd == 0:
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
                html_code_list = html_code_list + "<td align='top' width='5%'>"
                html_code_list = html_code_list + "<font size='+2'><button class='copy-button'>Copy</button></font>"
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


# ===========================================================


def show_lora_model_preview(modelname, input_cmd):
    #rkconvert - WIP
    if input_cmd == 0:
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
                html_code_list = html_code_list + "<td align='top' width='5%'>"
                html_code_list = html_code_list + "<font size='+2'><button class='copy-button'>Copy</button></font>"
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


# ===========================================================
# ===========================================================

# ---------------------------------

# rknote marked for delete?
def get_model_config_info(file):
    #rkconvert - NOT DONE
    file = open(file, "r")
    content = file.read()
    file.close()
    return content



# ------------------------------------------------------------



# get contents of file and return
def get_file_content(file):
    #rkconvert - NOT DONE
    file = open(file, "r")
    content = file.read()
    file.close()
    return content



# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------



def gen_random_seed():
    #rkconvert - DONE
    seed = random.randint(0, 2**32 - 1)
    return seed


# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------
# add prompt embeds and clip_skip


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

#t2isame below--------------------
    
    # rkconvert - NOT DONE
    # rkpipeline - NOT DONE
    #rkconvert - NOT DONE
    # rkpipeline NOT DONE
    global pipeline             # where the model is loaded to
    global safety_checker_pipeline
    
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
    
        
    # if we need the safety checker, load if not loaded    
    if STUDIO["use_safety_checker"]["value"]: 
        if int(SDPIPELINE["pipeline_safety_checker_loaded"]) == 0:
            try:
                if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
                safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
                SDPIPELINE["pipeline_safety_checker_loaded"] = 1
                if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
            except Exception as e:
                SDPIPELINE["pipeline_safety_checker_loaded"] = 0
                gr.Info("Safety Checker Model is NOT Loaded.<br>Either fix the error with model loading<br>Or, turn off the Safety Checker." + e, duration=5.0, title="Safety Checker Model")    
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
        



        
#t2isame above--------------------

    
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

#t2isame below--------------------

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
        imagebasename = LLSTUDIO["output_image_prefix"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + LLSTUDIO["output_image_suffix"] 
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

#t2isame above--------------------

        
        # run inference
        image2 = pipeline(**inference_args).images[0]

        # # run the inference
        # image2 = pipeline(
            # prompt=prompt, 
            # negative_prompt=negative_prompt, 
            # width=width, height=height, 
            # num_inference_steps=num_inference_steps, 
            # guidance_scale=guidance_scale, 
            # callback_on_step_end=callback_on_step_end
            # ).images[0]
 

 
#newsafetychecker below------------------------

        if STUDIO["use_safety_checker"]["value"]: 
            safety_output = safety_checker_pipeline(image2)
            nsfw_percent = 0
            normal_percent = 0
            for x in safety_output:
                if x['label'] == 'nsfw':
                    nsfw_percent = x['score']
                elif x['label'] == 'normal':
                    normal_percent = x['score']
            if normal_percent > nsfw_percent:
                # save the image generated
                image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")
            else:
                # # let's try and reduce the size of the font on the output 'label '
                nsfw_out = "Image Was NOT Saved !!</br>NSFW Content Detected !! " + str(int(nsfw_percent*100)) + "%"
                # # yield the data to both gradio outputs [progress/text,img]
                yield gr.update(value=nsfw_out), gr.update(value=None)
                gr.Info(nsfw_out, duration=10.0, title="NSFW Detected")
                # # return the data to both gradio outputs [progress/text,img], because we halted
                return nsfw_out, None
        else:
            # save the image generated
            image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

#newsafetychecker above------------------------

  
  
#t2isame below--------------------

        # # save the image generated
        # image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filename
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["outputfolder"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["outputfolder"], imagefilename)
        
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
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']




# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------

# add prompt embeds, strength and clip_skip


    
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
    
#i2isame below--------------------
    
    # rkconvert - NOT DONE
    # rkpipeline - NOT DONE
    #rkconvert - NOT DONE
    # rkpipeline NOT DONE
    global pipeline             # where the model is loaded to
    global safety_checker_pipeline
    
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
    
        
    # if we need the safety checker, load if not loaded    
    if STUDIO["use_safety_checker"]["value"]: 
        if int(SDPIPELINE["pipeline_safety_checker_loaded"]) == 0:
            try:
                if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
                safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
                SDPIPELINE["pipeline_safety_checker_loaded"] = 1
                if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
            except Exception as e:
                SDPIPELINE["pipeline_safety_checker_loaded"] = 0
                gr.Info("Safety Checker Model is NOT Loaded.<br>Either fix the error with model loading<br>Or, turn off the Safety Checker." + e, duration=5.0, title="Safety Checker Model")    
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
        
        
        
#i2isame above--------------------
    
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs
    
#i2isame below--------------------

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
        imagebasename = LLSTUDIO["output_image_prefix"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + LLSTUDIO["output_image_suffix"] 
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
            
            
        # run inference
        image2 = pipeline(**inference_args).images[0]

        # # run the inference
        # image2 = pipeline(
            # prompt=prompt, 
            # negative_prompt=negative_prompt, 
            # image=resized_img, 
            # width=width, 
            # height=height, 
            # num_inference_steps=num_inference_steps, 
            # guidance_scale=guidance_scale, 
            # callback_on_step_end=callback_on_step_end
            # ).images[0]


#newsafetychecker below------------------------

        if STUDIO["use_safety_checker"]["value"]: 
            safety_output = safety_checker_pipeline(image2)
            nsfw_percent = 0
            normal_percent = 0
            for x in safety_output:
                if x['label'] == 'nsfw':
                    nsfw_percent = x['score']
                elif x['label'] == 'normal':
                    normal_percent = x['score']
            if normal_percent > nsfw_percent:
                # save the image generated
                image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")
            else:
                # # let's try and reduce the size of the font on the output 'label '
                nsfw_out = "Image Was NOT Saved !!</br>NSFW Content Detected !! " + str(int(nsfw_percent*100)) + "%"
                # # yield the data to both gradio outputs [progress/text,img]
                yield gr.update(value=nsfw_out), gr.update(value=None)
                gr.Info(nsfw_out, duration=10.0, title="NSFW Detected")
                # # return the data to both gradio outputs [progress/text,img], because we halted
                return nsfw_out, None
        else:
            # save the image generated
            image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

#newsafetychecker above------------------------

  
  

#i2isame below--------------------

        # # save the image generated
        # image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["outputfolder"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["outputfolder"], imagefilename)
        
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
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------

# add prompt embeds, strength and clip_skip

    
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
    
    # rkconvert - NOT DONE
    # rkpipeline - NOT DONE
    #rkconvert - NOT DONE
    # rkpipeline NOT DONE
    global pipeline             # where the model is loaded to
    global safety_checker_pipeline
    
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
        
    # if we need the safety checker, load if not loaded    
    if STUDIO["use_safety_checker"]["value"]: 
        if int(SDPIPELINE["pipeline_safety_checker_loaded"]) == 0:
            try:
                if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
                safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
                SDPIPELINE["pipeline_safety_checker_loaded"] = 1
                if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
            except Exception as e:
                SDPIPELINE["pipeline_safety_checker_loaded"] = 0
                gr.Info("Safety Checker Model is NOT Loaded.<br>Either fix the error with model loading<br>Or, turn off the Safety Checker." + e, duration=5.0, title="Safety Checker Model")    
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
        
        
        
#inpsame above--------------------

    
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs

#inpsame below--------------------

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
        imagebasename = LLSTUDIO["output_image_prefix"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + LLSTUDIO["output_image_suffix"] 
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

#inpsame above--------------------
        


        # run inference
        image2 = pipeline(**inference_args).images[0]


        # image2 = pipeline(
            # prompt=prompt, 
            # negative_prompt=negative_prompt, 
            # image=resized_img, 
            # mask_image=resized_maskimg, 
            # width=width, height=height, 
            # num_inference_steps=num_inference_steps, 
            # guidance_scale=guidance_scale, 
            # callback_on_step_end=callback_on_step_end
            # ).images[0]

#newsafetychecker below------------------------

        if STUDIO["use_safety_checker"]["value"]: 
            safety_output = safety_checker_pipeline(image2)
            nsfw_percent = 0
            normal_percent = 0
            for x in safety_output:
                if x['label'] == 'nsfw':
                    nsfw_percent = x['score']
                elif x['label'] == 'normal':
                    normal_percent = x['score']
            if normal_percent > nsfw_percent:
                # save the image generated
                image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")
            else:
                # # let's try and reduce the size of the font on the output 'label '
                nsfw_out = "Image Was NOT Saved !!</br>NSFW Content Detected !! " + str(int(nsfw_percent*100)) + "%"
                # # yield the data to both gradio outputs [progress/text,img]
                yield gr.update(value=nsfw_out), gr.update(value=None)
                gr.Info(nsfw_out, duration=10.0, title="NSFW Detected")
                # # return the data to both gradio outputs [progress/text,img], because we halted
                return nsfw_out, None
        else:
            # save the image generated
            image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

#newsafetychecker above------------------------


#inpsame below--------------------

        # # save the image generated
        # image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["outputfolder"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["outputfolder"], imagefilename)
        
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
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------
    
# add prompt embeds and clip_skip
    
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
    
    # rkconvert - NOT DONE
    # rkpipeline - NOT DONE
    #rkconvert - NOT DONE
    # rkpipeline NOT DONE
    global pipeline             # where the model is loaded to
    global safety_checker_pipeline
    
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
    
        
    # if we need the safety checker, load if not loaded    
    if STUDIO["use_safety_checker"]["value"]: 
        if int(SDPIPELINE["pipeline_safety_checker_loaded"]) == 0:
            try:
                if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
                safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
                SDPIPELINE["pipeline_safety_checker_loaded"] = 1
                if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
            except Exception as e:
                SDPIPELINE["pipeline_safety_checker_loaded"] = 0
                gr.Info("Safety Checker Model is NOT Loaded.<br>Either fix the error with model loading<br>Or, turn off the Safety Checker." + e, duration=5.0, title="Safety Checker Model")    
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
       
        
        
#ip2psame above--------------------

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
    
#ip2psame below--------------------

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
        imagebasename = LLSTUDIO["output_image_prefix"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + LLSTUDIO["output_image_suffix"] 
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

#ip2psame above--------------------



        # run inference
        image2 = pipeline(**inference_args).images[0]



        # image2 = pipeline(
            # prompt=prompt, 
            # negative_prompt=negative_prompt, 
            # image=resized_img, 
            # num_inference_steps=num_inference_steps, 
            # guidance_scale=guidance_scale, 
            # image_guidance_scale=image_guidance_scale, 
            # callback=callback_on_each_step_end).images[0]

#newsafetychecker below------------------------

        if STUDIO["use_safety_checker"]["value"]: 
            safety_output = safety_checker_pipeline(image2)
            nsfw_percent = 0
            normal_percent = 0
            for x in safety_output:
                if x['label'] == 'nsfw':
                    nsfw_percent = x['score']
                elif x['label'] == 'normal':
                    normal_percent = x['score']
            if normal_percent > nsfw_percent:
                # save the image generated
                image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")
            else:
                # # let's try and reduce the size of the font on the output 'label '
                nsfw_out = "Image Was NOT Saved !!</br>NSFW Content Detected !! " + str(int(nsfw_percent*100)) + "%"
                # # yield the data to both gradio outputs [progress/text,img]
                yield gr.update(value=nsfw_out), gr.update(value=None)
                gr.Info(nsfw_out, duration=10.0, title="NSFW Detected")
                # # return the data to both gradio outputs [progress/text,img], because we halted
                return nsfw_out, None
        else:
            # save the image generated
            image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

#newsafetychecker above------------------------


#ip2psame below--------------------

        # # save the image generated
        # image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["outputfolder"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["outputfolder"], imagefilename)
        
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
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']


# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------

# add prompt embeds
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
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
    global safety_checker_pipeline

    # clear both gradio outputs [progress/text,img]
    yield gr.update(value=None), gr.update(value=None)

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
            pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(STUDIO["sdupscale2x_model_name"]["value"], local_files_only=True, safety_checker = None, requires_safety_checker = False, device=LLSTUDIO["device"])
            SDPIPELINE['pipeline_class'] = "StableDiffusionLatentUpscalePipeline"
            SDPIPELINE['pipeline_loaded'] = 1
            SDPIPELINE['pipeline_model_name'] = STUDIO["sdupscale2x_model_name"]["value"]
            SDPIPELINE['pipeline_source'] = "HUB"
            SDPIPELINE['pipeline_gen_mode'] = "2x UpScaler"
            gr.Info("Finished Loading SD Upscale 2X Model.", duration=3.0, title="Upscale Model")
        except Exception as e: # Catch any other unexpected exceptions
            tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_gen_mode'] + " Model." + f"<br>{e}" + "</h3>"
            yield gr.update(value=tempout)
            gr.Info("<h3>Error Loading: " + SDPIPELINE['pipeline_gen_mode'] + " Model."  + f"<br>{e}" + "</h3>", duration=3.0, title="2x Upscale Model")
            return tempout

    # check if model is loaded
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return
    
    # redundant...
    # check if valid model type for image generation
    if SDPIPELINE['pipeline_class'] != "StableDiffusionLatentUpscalePipeline":
        gr.Info("Incorrect Model is Loaded in the Pipeline.<br>Please Load a valid Model Type for Image Generation.", duration=5.0, title="Incorrect Model Type")    
        return
        
    # if we need the safety checker, load if not loaded    
    if STUDIO["use_safety_checker"]["value"]: 
        if int(SDPIPELINE["pipeline_safety_checker_loaded"]) == 0:
            try:
                if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
                safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
                SDPIPELINE["pipeline_safety_checker_loaded"] = 1
                if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
            except Exception as e:
                SDPIPELINE["pipeline_safety_checker_loaded"] = 0
                gr.Info("Safety Checker Model is NOT Loaded.<br>Either fix the error with model loading<br>Or, turn off the Safety Checker." + e, duration=5.0, title="Safety Checker Model")    
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
    imagebasename = LLSTUDIO["output_image_prefix"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + LLSTUDIO["output_image_suffix"] 
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


    # run the inference
    image2 = pipeline(**inference_args).images[0]


    # # run the inference
    # image2 = pipeline(
        # prompt=prompt, 
        # negative_prompt=negative_prompt, 
        # guidance_scale=guidance_scale, 
        # num_inference_steps=num_inference_steps, 
        # image=resized_img, 
        # callback=callback_on_each_step_end
        # ).images[0]

#newsafetychecker below------------------------

    if STUDIO["use_safety_checker"]["value"]: 
        safety_output = safety_checker_pipeline(image2)
        nsfw_percent = 0
        normal_percent = 0
        for x in safety_output:
            if x['label'] == 'nsfw':
                nsfw_percent = x['score']
            elif x['label'] == 'normal':
                normal_percent = x['score']
        if normal_percent > nsfw_percent:
            # save the image generated
            image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")
        else:
            # # let's try and reduce the size of the font on the output 'label '
            nsfw_out = "Image Was NOT Saved !!</br>NSFW Content Detected !! " + str(int(nsfw_percent*100)) + "%"
            # # yield the data to both gradio outputs [progress/text,img]
            yield gr.update(value=nsfw_out), gr.update(value=None)
            gr.Info(nsfw_out, duration=10.0, title="NSFW Detected")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return nsfw_out, None
    else:
        # save the image generated
        image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

#newsafetychecker above------------------------

    # # save the image generated
    # image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

    # mark end time
    pend = time.time()
    pelapsed = pend - pstart

    if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
    
    # ONCE an image HAS BEEN generated, we set image and text output filenames
    # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
    # 'UNTIL' replaced with next generated image when more than a single image 
    # is being generated in a batch.
    LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["outputfolder"], textfilename)
    LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["outputfolder"], imagefilename)
    
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
    
    # check if user has halted after image generation current inference finished
    if LLSTUDIO["halt_gen"] == 1:
        gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
        # # return the data to both gradio outputs [progress/text,img], because we halted
        return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------


# add prompt embeds, strength and clip_skip


    
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
    
#i2isame below--------------------
    
    # rkconvert - NOT DONE
    # rkpipeline - NOT DONE
    #rkconvert - NOT DONE
    # rkpipeline NOT DONE
    global pipeline             # where the model is loaded to
    global safety_checker_pipeline
    
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
    
    # if we need the safety checker, load if not loaded    
    if STUDIO["use_safety_checker"]["value"]: 
        if int(SDPIPELINE["pipeline_safety_checker_loaded"]) == 0:
            try:
                if int(STUDIO["app_debug"]["value"]) > 0: print("Loading Image Classifier... '" + STUDIO["safety_checker_model_name"]["value"] + "' for Safety Checker")
                safety_checker_pipeline = transformers.pipeline("image-classification",model=STUDIO["safety_checker_model_name"]["value"])
                SDPIPELINE["pipeline_safety_checker_loaded"] = 1
                if int(STUDIO["app_debug"]["value"]) > 0: print("Finished Loading Image Classifier.")
            except Exception as e:
                SDPIPELINE["pipeline_safety_checker_loaded"] = 0
                gr.Info("Safety Checker Model is NOT Loaded.<br>Either fix the error with model loading<br>Or, turn off the Safety Checker." + e, duration=5.0, title="Safety Checker Model")    
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
        
        
        
#i2isame cnet above--------------------
    
    # Define the callback function to update the progress bar
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        progress_value = (step_index + 1) / num_inference_steps
        if step_index + 1 == num_inference_steps:
            progress(progress_value, desc=f"Finished Inference. Decoding Image...")
        else:
            progress(progress_value, desc=f"Inference Step {step_index + 1}/{num_inference_steps}")
        return callback_kwargs
    
#i2isame cnet below--------------------
# minus the 'strength' parameter

   # resize input images to 512x512
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
        imagebasename = LLSTUDIO["output_image_prefix"] + str(myseed) + "_" + str(imgnumb) + "_" + str(formatted_time) + LLSTUDIO["output_image_suffix"] 
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
            
            
        # run inference
        image2 = pipeline(**inference_args).images[0]

        # # run the inference
        # image2 = pipeline(
            # prompt=prompt, 
            # negative_prompt=negative_prompt, 
            # image=resized_img, 
            # width=width, 
            # height=height, 
            # num_inference_steps=num_inference_steps, 
            # guidance_scale=guidance_scale, 
            # callback_on_step_end=callback_on_step_end
            # ).images[0]

#newsafetychecker below------------------------

        if STUDIO["use_safety_checker"]["value"]: 
            safety_output = safety_checker_pipeline(image2)
            nsfw_percent = 0
            normal_percent = 0
            for x in safety_output:
                if x['label'] == 'nsfw':
                    nsfw_percent = x['score']
                elif x['label'] == 'normal':
                    normal_percent = x['score']
            if normal_percent > nsfw_percent:
                # save the image generated
                image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")
            else:
                # # let's try and reduce the size of the font on the output 'label '
                nsfw_out = "Image Was NOT Saved !!</br>NSFW Content Detected !! " + str(int(nsfw_percent*100)) + "%"
                # # yield the data to both gradio outputs [progress/text,img]
                yield gr.update(value=nsfw_out), gr.update(value=None)
                gr.Info(nsfw_out, duration=10.0, title="NSFW Detected")
                # # return the data to both gradio outputs [progress/text,img], because we halted
                return nsfw_out, None
        else:
            # save the image generated
            image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

#newsafetychecker above------------------------


#i2isame cnet below--------------------

        # # save the image generated
        # image2.save(os.path.join(LLSTUDIO["outputfolder"], imagefilename), "png")

        # mark end time
        pend = time.time()
        pelapsed = pend - pstart

        if int(STUDIO["app_debug"]["value"]) > 0: print(f"Total Time taken to run inference: {format_seconds_strftime(pelapsed)}")
        
        # ONCE an image HAS BEEN generated, we set image and text output filenames
        # But, NOT until... this way the 'send to gallery' function works with the VISIBLE image
        # 'UNTIL' replaced with next generated image when more than a single image 
        # is being generated in a batch.
        LLSTUDIO['last_prompt_filename'] = os.path.join(LLSTUDIO["outputfolder"], textfilename)
        LLSTUDIO['last_image_filename'] = os.path.join(LLSTUDIO["outputfolder"], imagefilename)
        
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
        
        # check if user has halted after image generation current inference finished
        if LLSTUDIO["halt_gen"] == 1:
            gr.Info("Generation was previously halted</br>Final inference completed.", duration=5.0, title="Generation")
            # # return the data to both gradio outputs [progress/text,img], because we halted
            return imagefilename, LLSTUDIO['last_image_filename']

# # return the data to both gradio outputs [progress/text,img], because we're done
    return imagefilename, LLSTUDIO['last_image_filename']



# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------


# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------

def make_default_dirs(path_file):
    #rkconvert - NOT DONE
    if not os.path.exists(path_file):
        os.makedirs(path_file)

# ------------------------------------------------------

def make_all_default_dirs():
    #rkconvert - NOT DONE

    make_default_dirs(LLSTUDIO["lcm_model_dir"])
    make_default_dirs(LLSTUDIO["lcm_model_image_dir"])
    make_default_dirs(LLSTUDIO["safe_model_dir"])
    make_default_dirs(LLSTUDIO["safe_model_image_dir"])
    make_default_dirs(LLSTUDIO["lora_model_dir"])
    make_default_dirs(LLSTUDIO["lora_model_image_dir"])
    make_default_dirs(LLSTUDIO["output_image_dir"])
    make_default_dirs(LLSTUDIO["outputfolder"])

# ------------------------------------------------------
# ------------------------------------------------------



def send_to_gallery():

    if (LLSTUDIO['last_image_filename'] == "" or LLSTUDIO['last_prompt_filename'] == ""):
        stdoutput = "Error: No Image/Prompt Found For Gallery"
        return stdoutput
        
    stdoutput = ""
    #rkconvert - NOT DONE 
    # NEEDS REWORKED FOR MODEL SOURCE, LCMLORA, HUB, HUG, SAFE, LoRA, etc...
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
        except Exception as e: # Catch any other unexpected exceptions
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
        except Exception as e: # Catch any other unexpected exceptions
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
        except Exception as e: # Catch any other unexpected exceptions
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
        except Exception as e: # Catch any other unexpected exceptions
            stdoutput = stdoutput + f"An unexpected error occurred: {e}"

    return stdoutput

    
# --------------------------------------------

def bytes_to_human_readable(num_bytes):
    #rkconvert - NOT DONE
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

# ------------------------------------------------------

def get_all_memory_info():
    #rkconvert - NOT DONE
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
    #rkconvert - NOT DONE
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
    #rkconvert - NOT DONE
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
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return gr.Textbox(label="Prompt - Tokens[0]")

    tokenizer = pipeline.tokenizer
    tokenized_output = tokenizer.encode(prompt)
    num_tokens = "Prompt - Tokens[" + str(len(tokenized_output)) + "]"
    return gr.Textbox(label=num_tokens)
    
    
# ------------------------------------------------------


def get_negprompt_length_tokens(prompt):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return gr.Textbox(label="Negative Prompt - Tokens[0]")

    tokenizer = pipeline.tokenizer
    tokenized_output = tokenizer.encode(prompt)
    num_tokens = "Negative Prompt - Tokens[" + str(len(tokenized_output)) + "]"
    return gr.Textbox(label=num_tokens)

# ------------------------------------------------------


def get_prompt_length(prompt):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
    if int(SDPIPELINE['pipeline_loaded']) < 1:
        grinfo_no_model_loaded()
        return 0
    tokenizer = pipeline.tokenizer
    tokenized_output = tokenizer.encode(prompt)
    num_tokens = len(tokenized_output)
    return num_tokens

# ------------------------------------------------------


def delete_pipeline():
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl

    if int(SDPIPELINE['pipeline_loaded']) < 1:
        pipeline = ""
        if hasattr(pipeline, 'to') and callable(getattr(pipeline, 'to')):
            pipeline.to(LLSTUDIO["device"])
        del pipeline
        gc.collect()
        tempout = str_no_model_loaded()
        yield gr.update(value=tempout)
        grinfo_no_model_loaded()
        return tempout
        
    if hasattr(pipeline, 'to') and callable(getattr(pipeline, 'to')):
        pipeline.to(LLSTUDIO["device"])
    del pipeline
    gc.collect()

    reset_pipeline_info()
    tempout = "<h3>Unloaded Pipeline, Ready to Load a Model.</h3>"
    yield gr.update(value=tempout)
    gr.Info("<h3>Unloaded Pipeline, Ready to Load a Model.</h3>", duration=3.0, title="Unload Model")
    return tempout


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ------------------------------------------------------------



def reset_pipeline_info():
    #rkconvert -  DONE
    SDPIPELINE['pipeline_loaded'] = 0                           # model loaded ? 0=no/1=yes, trigger error/alert on No model loaded
    SDPIPELINE['pipeline_class'] = "StableDiffusionPipeline"    # StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImage2Image... default startup value=StableDiffusionPipeline
    SDPIPELINE['pipeline_source'] = ""                          # 'LCMLORA', 'HUB Cached', 'Huggingface', 'Safetensors' basically where model was loaded from, if LCMLORA, already has LCM LoRA added/fused
    SDPIPELINE['pipeline_model_name'] = ""                      # name of model as in dropdowns
    SDPIPELINE['pipeline_gen_mode'] = "Text to Image"           # Text 2 Image, Image 2 Image, Inpainting, Instruct Pix2Pix, UpScaler default startup value=Text 2 Image
    SDPIPELINE['pipeline_model_type'] = "SD15"                  # SD15 or SDXL default=SD15
    SDPIPELINE['pipeline_text_encoder'] = 0                     # use seperate text encoder ? 0=no/1=yes
    SDPIPELINE['pipeline_text_encoder_name'] = ""               # name of model of seperate text encoder as in dropdowns
    SDPIPELINE['pipeline_model_precision'] = "fp16"             # basically, fp16 or fp32 (default LCM to fp16 so it'll run it's 4 step lcm-lora)
    SDPIPELINE['pipeline_controlnet_loaded'] = 0,                      # load a controlnet ? 0=no/1=yes
    SDPIPELINE['pipeline_controlnet_name'] = "",                      # name of control net
    SDPIPELINE['pipeline_controlnet_name2'] = "",                      # name of control net2
    SDPIPELINE['pipeline_safety_checker_loaded'] = 0                      # loaded a safety_checker model ? 0=no/1=yes
    
    return "Pipeline Info Reset."



# ------------------------------------------------------------


    
def display_pipeline_info(last_ret_value):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
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
    #rkconvert - NOT DONE
    return LLSTUDIO["lcm_model_prefix"] + "MyNewModel" + LLSTUDIO["lcm_model_suffix"], 1.0, "<h3>LCM-LoRA Model name has been defaulted and LoRA value reset to '1.0'.</h3></br>Replace the 'MyNewModel' part with your model name or completely rename it whatever you want. Keep in mind the user should use some sort of naming convention to keep track as to which models have had the 'LCM-LoRA' added. Like add 'LCM' to the model name on one end or the other. When loading other models you have the option of 'not' adding the LCM-LoRA weights to the loaded model, which would then be fused to the model and saved. And therefore models without the LCM-LoRA weights added to the saved model, it will not run an inference in just the normal 4 steps for an average model and produce a good image. You would need to run that model at it's normal higher number of step to reproduce a good image. Very slow for a CPU to do."



# ------------------------------------------------------------



def save_lcm_model(model_name,lora_value,use_safetensors,fp16):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl

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
    except Exception as e: # Catch any other unexpected exceptions
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
    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Converting Pipeline.to " + fp16_tempout + ". " + f"{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout
    
    tempout = "<h3>Saving Pipeline as LCM-LoRA Model: " + new_lcm_model_filename + "...</h3>"
    yield gr.update(value=tempout)
    
    # Init a dict with the common arguments  **pipeline_args
    pipeline_args = { }
    
    if fp16:
        pipeline_args["variant"] = "fp16"
        
    if use_safetensors:
        pipeline_args["safe_serialization"] = True
    
    try:
        pipeline.save_pretrained(f"{new_lcm_model_filepathname}", **pipeline_args)
    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Saving Pipeline to Model. " + f"{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout
        
    SDPIPELINE['pipeline_model_name'] = new_lcm_model_filename
    SDPIPELINE['pipeline_source'] = "LCMLORA"
    if fp16:
        SDPIPELINE["pipeline_model_precision"] = "fp16"
    else:
        SDPIPELINE["pipeline_model_precision"] = "fp32"

    # create a model card (per say) for the image gallery for this specific LCM-LoRA Model
    if not os.path.exists(model_image_path_file):
        os.makedirs(model_image_path_file)
        file1 = open(os.path.join(model_image_path_file, new_lcm_model_filename) + ".md", 'w')
        content = "## LCM-LoRA Model: " + new_lcm_model_filename + "\n\n"
        content = content + "## Original Model: " + old_model_name + "\n\n"
        content = content + "Loaded LoRAs: The 'LCM-LoRA', will not be shown if the model IS, an LCM-LoRA model. Because the LCM-LoRA has already been fused to the model, and given a prefixed model name to indicate model type\n\n"
        content = content + loadedloras + "\n\n"
        file1.write(content)
        file1.close()    

    
    tempout = "<h3>" + "Finished Saving Pipeline Loaded with " + old_model_name + " to LCM-LoRA model " + new_lcm_model_filename + "</h3>" + "LoRAs: </br>" + get_loaded_lora_models_html()
    yield gr.update(value=tempout)
    return tempout


# ------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------


def load_lcm_model(model_name, use_diff_text_enc, text_enc_model_name, text_enc_clipskip, use_controlnet, controlnet_name, use_controlnet2, controlnet_name2, fp16, fp16e):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
    
    pstart = time.time()
    
    tempout = "<h3>Loading Model... " + model_name + "</h3>"
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

    # Init a dict for arguments
    pipeline_args = {}

    if fp16:
        pipeline_args["variant"] = "fp16"
    
    # always no safety checker here... we check the image after gereation, before saving.
    # NSFW = no save, SFW = save
    # doing it this way ensures when you turn if off in the settings, it's off. 
    # because it is already, by overriding any defaults which may have it set to True
    # and circumvent the error because no actual 'safety_checker' is defined. :) 
    pipeline_args["safety_checker"] = None
    pipeline_args["requires_safety_checker"] = False

    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True
        

    if SDPIPELINE['pipeline_model_type'] == "SD15":
        text_enc_pipeline_args = {}
        
        # add the parameter for the precision variant we want to load, MUST exist !!
        # we can add a checkbox later fp16/fp32
        if fp16e:
            text_enc_pipeline_args["variant"] = "fp16"
        
        # hmmmm do we actually need this ??
        # if STUDIO["local_files_only"]["value"]: 
            # text_enc_pipeline_args["local_files_only"] = True
        
        # Conditionally add the 'text_encoder' argument
        if int(text_enc_clipskip) > 1:
            num_hidden_layers = int(12 - (int(text_enc_clipskip) - 1))
            text_enc_pipeline_args["subfolder"] = "text_encoder"
            text_enc_pipeline_args["num_hidden_layers"] =  num_hidden_layers
            # Init a dict with the clip_skip layer arguments
            # text_enc_pipeline_args = {
                # "variant": "fp16",
                # "subfolder": "text_encoder",
                # "num_hidden_layers": 12 - text_enc_clipskip - 1,
            # }
        else:
            text_enc_pipeline_args["subfolder"] = "text_encoder"
            # Init a dict with the common arguments
            # text_enc_pipeline_args = {
                # "variant": "fp16",
                # "subfolder": "text_encoder",
            # }

        # do we use a different 'text_encoder' instead of loaded model text_encoder?
        if use_diff_text_enc:
            if text_enc_model_name:
                # Load the CLIP text encoder from a different model
                # and specify the number of layers to use.
                try:
                    text_encoder = transformers.CLIPTextModel.from_pretrained(get_lcm_model_path_file(text_enc_model_name), **text_enc_pipeline_args)
                    pipeline_args["text_encoder"] = text_encoder
                except Exception as e: # Catch any other unexpected exceptions
                    tempout = "<h3>Error Loading Seperate Text Encoder: " + text_enc_model_name + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout

        # if so set it up to load before model 
        # similar to the seperate text encoder method, but account for 2 ControlNets
        # 0. Figure out which model repo to use based on 'controlnet_name'
        # using dict 'CNETMODELS' to get actual huggingface model name
        # 1. Load the ControlNet model
        if (use_controlnet or use_controlnet2):
            # rknew method (max 2 controlnets)
            controlnet = []
            if use_controlnet:
                try:
                    controlnet.append(ControlNetModel.from_pretrained(CNETMODELS[controlnet_name]))
                except Exception as e: 
                    tempout = "<h3>Error Loading ControlNet Model: " + CNETMODELS[controlnet_name] + "<br>For ControlNet Named: " + controlnet_name + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout
            if use_controlnet2:
                try:
                    controlnet.append(ControlNetModel.from_pretrained(CNETMODELS[controlnet_name2]))
                except Exception as e: 
                    tempout = "<h3>Error Loading ControlNet Model: " + CNETMODELS[controlnet_name2] + "<br>For ControlNet Named: " + controlnet_name2 + f"<br>{e}" + "</h3>"
                    yield gr.update(value=tempout)
                    return tempout
                    
            # if any controlnets got loaded we add the argument for the controlnet(s)    
            if len(controlnet) > 0:
                # 2. add to a dictionary with the controlnet arguments
                # this one argument hadles one or more (list[]) of controlnets :)
                pipeline_args["controlnet"] = controlnet
            
                # by changing the pipeline class we can load model with the rest of them...
                # change the pipeline class from SD to CNET ! :)
                SDPIPELINE['pipeline_class'] = "StableDiffusionControlNetPipeline"

                # 3. Load the base Stable Diffusion model and combine with ControlNet
                # rknote: we do this later in this function...
                # pipeline = StableDiffusionControlNetPipeline.from_pretrained(get_lcm_model_path_file(model_name), **pipeline_args)    


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
    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout
    

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Moving TO device??: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    
    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + "</h3>"
    if use_diff_text_enc:
        if SDPIPELINE['pipeline_model_type'] == "SD15":
            if text_enc_model_name:
                # we do this AFTER everything is completely done, with no errors
                SDPIPELINE['pipeline_text_encoder'] = 1
                SDPIPELINE['pipeline_text_encoder_name'] = text_enc_model_name
                tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " Text Encoder from : " + text_enc_model_name + "</h3>"
    if (use_controlnet or use_controlnet2):
        if SDPIPELINE['pipeline_model_type'] == "SD15":
            # we do this AFTER everything is completely done, with no errors
            SDPIPELINE['pipeline_controlnet_loaded'] = int(len(controlnet))
            if use_controlnet:
                SDPIPELINE['pipeline_controlnet_name'] = controlnet_name
                tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " ControlNet : " + controlnet_name + "</h3>"
            if use_controlnet2:
                SDPIPELINE['pipeline_controlnet_name2'] = controlnet_name2
                tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " ControlNet : " + controlnet_name2 + "</h3>"
            if (use_controlnet and use_controlnet2):
                tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " ControlNets : " + controlnet_name + " / " + controlnet_name2 + "</h3>"
            
    yield gr.update(value=tempout)
    pend = time.time()
    pelapsed = pend - pstart
    gr.Info(SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model" + SDPIPELINE['pipeline_model_name'] + "loaded for " + SDPIPELINE['pipeline_gen_mode'] + " " + format_seconds_strftime(pelapsed), duration=5.0, title="LCM-LoRA Model")
    # we do this AFTER everything is completely done, with no errors
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout




# ---------------------------------
# ---------------------------------

 

def load_hub_model(model_name, fp16_check, lora_value, add_lcmlora):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl

    pstart = time.time()
    if len(model_name) < 1:
        tempout = "<h3>Error: MUST Enter valid model name... " + model_name + "</h3>"
        yield gr.update(value=tempout)
        return tempout   

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


    # Init a dict for arguments
    pipeline_args = {}

    if fp16_check:
        pipeline_args["variant"] = "fp16"


    # always no safety checker here... we check the image after gereation, before saving.
    # NSFW = no save, SFW = save
    # doing it this way ensures when you turn if off in the settings, it's off. 
    # because it is already, by overriding any defaults which may have it set to True
    # and circumvent the error because no actual 'safety_checker' is defined. :) 
    pipeline_args["safety_checker"] = None
    pipeline_args["requires_safety_checker"] = False


    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True

        
            

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
    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e: # Catch any other unexpected exceptions
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
        loraout_text = " - with LCM LoRA"
    else:
        loraout_text = " - without LCM LoRA"
    
    pend = time.time()
    pelapsed = pend - pstart

    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Loaded Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " - " + loraout_text + "</h3>"
    yield gr.update(value=tempout)
    gr.Info(SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Model loaded for " + SDPIPELINE['pipeline_gen_mode'] + ":</br>" + SDPIPELINE['pipeline_model_name'] + " - " + loraout_text + "</br>" + format_seconds_strftime(pelapsed), duration=5.0, title=SDPIPELINE['pipeline_source'] + " Model")
    
   
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout



# ---------------------------------
# ---------------------------------



def load_hug_model(model_name, model_class_name, fp16_check):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
    pstart = time.time()
    if len(model_name) < 1:
        tempout = "<h3>Error: MUST Enter valid model name... " + model_name + "</h3>"
        yield gr.update(value=tempout)
        return tempout   
    tempout = "<h3>Loading Model... " + model_name + "</h3>"
    yield gr.update(value=tempout)

    pipe_class = model_class_name
    SDPIPELINE['pipeline_model_name'] = model_name
    SDPIPELINE['pipeline_class'] = pipe_class
    SDPIPELINE['pipeline_source'] = "Huggingface"
    SDPIPELINE['pipeline_model_type'] = PIPECLASSES[pipe_class]['pipeline_model_type']
    SDPIPELINE['pipeline_gen_mode'] = PIPECLASSES[pipe_class]['pipeline_gen_mode']
    SDPIPELINE['pipeline_text_encoder'] = 0
    SDPIPELINE['pipeline_text_encoder_name'] = ""


    # Init a dict for arguments
    pipeline_args = {}

    if fp16_check:
        pipeline_args["variant"] = "fp16"


    # always no safety checker here... we check the image after gereation, before saving.
    # NSFW = no save, SFW = save
    # doing it this way ensures when you turn if off in the settings, it's off. 
    # because it is already, by overriding any defaults which may have it set to True
    # and circumvent the error because no actual 'safety_checker' is defined. :) 
    pipeline_args["safety_checker"] = None
    pipeline_args["requires_safety_checker"] = False


    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True

        

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

    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + SDPIPELINE['pipeline_model_name'] + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e: # Catch any other unexpected exceptions
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



# ---------------------------------
# ---------------------------------

# ------------------------------------------------------------


    

# safeload_model_dropdown,safeload_pipeline_classes, safeload_model_lora,safeload_model_add_lcmlora   
# uses input to function for model type of safetensors file
def load_safetensors_model(safetensors_model, pipe_class, lora_value, add_lcmlora):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
    safetensors_model_pathfile = os.path.join(LLSTUDIO["safe_model_dir"], safetensors_model + ".safetensors")
    pstart = time.time()
    SDPIPELINE['pipeline_model_name'] = safetensors_model
    SDPIPELINE['pipeline_class'] = pipe_class
    SDPIPELINE['pipeline_source'] = "Safetensors"
    SDPIPELINE['pipeline_model_type'] = PIPECLASSES[pipe_class]['pipeline_model_type']
    SDPIPELINE['pipeline_gen_mode'] = PIPECLASSES[pipe_class]['pipeline_gen_mode']
    SDPIPELINE['pipeline_text_encoder'] = 0
    SDPIPELINE['pipeline_text_encoder_name'] = ""

 
# ------------------------------------------------------------

    
    # Init a dict with the common arguments for safetensors
    pipeline_args = {}

    if STUDIO["local_files_only"]["value"]: 
        pipeline_args["local_files_only"] = True

    
    # use 'original_config_file' when loading the safetensors model
    if STUDIO["safe_use_original_config_file"]["value"]:
        if SDPIPELINE['pipeline_class'] == "StableDiffusionPipeline":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SD_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLPipeline":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXL_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionImage2Image":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDImage2Image_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLImage2Image":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXLImage2Image_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInpaintPipeline":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDInpaint_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInpaintPipeline":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXLInpaint_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInstructPix2PixPipeline":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDInstructPix2Pix_original_config"]["value"])   
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInstructPix2PixPipeline":
            pipeline_args["original_config"] = os.path.join(".", "configs", STUDIO["SDXLInstructPix2Pix_original_config"]["value"])   

    # use a reference model when loading the safetensors model
    if STUDIO["safe_use_config"]["value"]:
        if SDPIPELINE['pipeline_class'] == "StableDiffusionPipeline":
            pipeline_args["config"] = STUDIO["SD_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLPipeline":
            pipeline_args["config"] = STUDIO["SDXL_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionImage2Image":
            pipeline_args["config"] = STUDIO["SDImage2Image_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLImage2Image":
            pipeline_args["config"] = STUDIO["SDXLImage2Image_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInpaintPipeline":
            pipeline_args["config"] = STUDIO["SDInpaint_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInpaintPipeline":
            pipeline_args["config"] = STUDIO["SDXLInpaint_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionInstructPix2PixPipeline":
            pipeline_args["config"] = STUDIO["SDInstructPix2Pix_config"]["value"] 
        elif SDPIPELINE['pipeline_class'] == "StableDiffusionXLInstructPix2PixPipeline":
            pipeline_args["config"] = STUDIO["SDXLInstructPix2Pix_config"]["value"] 


 
# ------------------------------------------------------------
 
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
    except Exception as e: # Catch any other unexpected exceptions
        tempout = "<h3>Error Loading: " + SDPIPELINE['pipeline_model_type'] + " Model for " + SDPIPELINE['pipeline_gen_mode'] + ": " + safetensors_model + f"<br>{e}" + "</h3>"
        yield gr.update(value=tempout)
        return tempout

    try:
        pipeline.to(LLSTUDIO["device"])
    except Exception as e: # Catch any other unexpected exceptions
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
        loraout_text = " - with LCM LoRA"
    else:
        loraout_text = " - without LCM LoRA"
    
    pend = time.time()
    pelapsed = pend - pstart

    tempout = "<h3>" + SDPIPELINE['pipeline_model_type'] + " " + SDPIPELINE['pipeline_source'] + " Loaded Model " + SDPIPELINE['pipeline_model_name'] + " for " + SDPIPELINE['pipeline_gen_mode'] + " - " + loraout_text + "</h3>"
    yield gr.update(value=tempout)
    gr.Info(SDPIPELINE['pipeline_model_type'] + " Safetensors Model loaded for " + SDPIPELINE['pipeline_gen_mode'] + ":</br>" + SDPIPELINE['pipeline_model_name'] + " - with LCM LoRA</br>" + format_seconds_strftime(pelapsed), duration=5.0, title="Safetensors Model")
   
    
    
    SDPIPELINE['pipeline_loaded'] = 1
    return tempout
    

# --------------------------------------------------------------



def download_huggingface_model(model_name):
    #rkconvert - NOT DONE
    if len(model_name) < 1:
        tempout = "<h3>Error: MUST Enter valid Huggingface Model Name.</br>Name Provided: " + model_name + "</br>Ex: stable-diffusion-v1-5/stable-diffusion-v1-5</h3>"
        yield gr.update(value=tempout)
        return tempout   
    tempout = "<h3>" + "Downloading Huggingface Model: " + model_name + " ...</h3></br>"
    yield gr.update(value=tempout)
    snapshot_download(repo_id=model_name, force_download=True)
    tempout = "<h3>" + "Finished Downloading Huggingface Model: " + model_name + "</h3></br>"
    yield gr.update(value=tempout)
    return tempout



# --------------------------------------------------------------
# --------------------------------------------------------------


def delete_hub_model(model_name, del_model):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED

    if len(model_name) < 1:
        tempout = "<h3>Error: MUST Enter valid Hub Cached Model name... " + model_name + "</h3>"
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


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------



# # Callback function to update/print steps

# # Define a callback function to interrupt the upscaling process
# def upscale_display_interruption_callback(current_upscaler_pipeline, i, t, callback_kwargs): # Renamed 'pipe' to 'current_upscaler_pipeline' for clarity
    # print(f"Upscaling step {i}. Time Scale {t}.\n")
    # return callback_kwargs


# ------------------------------------------------------


# # Define a callback function to interrupt the upscaling process
# def upscale_interruption_callback(current_upscaler_pipeline, i, t, callback_kwargs): # Renamed 'pipe' to 'current_upscaler_pipeline' for clarity
    # if i == 5:  # Interrupt upscaling after 5 steps
        # current_upscaler_pipeline._interrupt = True  # Access the pipeline instance passed as an argument
        # print(f"Upscaling interrupted after {i} steps.\n")
    # return callback_kwargs


# ------------------------------------------------------




# def my_callback_function(pipe, step_index):
    # global pipeline
    # print(f"Callback executed at step {step_index}.\n")



# ------------------------------------------------------



# mine
# Call the callback at every step
# image = pipeline(prompt=prompt, image=low_res_image, callback=print_current_step).images[0]
def print_current_step(step: int, timestep: int, latents: torch.Tensor):
    #rkconvert - NOT DONE
    print(f"Current Step: {step}, Timestep: {timestep}, Latent Shape: {latents.shape}\n")


# ------------------------------------------------------



# gai
# Call the callback every 5 steps
# image = pipeline(prompt=prompt, image=low_res_image, callback=my_callback_function, callback_steps=5).images[0]
def my_callback_function(step: int, timestep: int, latents: torch.Tensor):
    #rkconvert - NOT DONE
    print(f"Current Step: {step}, Timestep: {timestep}, Latent Shape: {latents.shape}\n")


# -------------------------------------

# How to use Diffusers callbacks
# To use a callback, you define a function that takes specific arguments and pass it to the callback_on_step_end parameter of your pipeline's __call__ method.
# MUST USE 'return callback_kwargs'
# image = pipeline(prompt="A landscape", callback_on_step_end=my_diffusion_callback).images[0]
def my_diffusion_callback(pipeline, step_index, timestep, callback_kwargs):
    #rkconvert - NOT DONE
    # Your custom logic here
    print(f"Step {step_index}: Timestep {timestep}\n")
    return callback_kwargs



# ------------------------------------------------------




# Dynamic CFG (Classifier-Free Guidance)
# Example: You can use a callback to reduce the guidance_scale (CFG) after a certain percentage of the denoising steps.
# MUST USE 'return callback_kwargs'
# image = pipeline(prompt="A dog", guidance_scale=7.5, callback_on_step_end=dynamic_cfg_callback, callback_on_step_end_tensor_inputs=["prompt_embeds"]).images[0]
def dynamic_cfg_callback(pipeline, step_index, timestep, callback_kwargs):
    #rkconvert - NOT DONE
    if step_index / pipeline.num_timesteps >= 0.4: # Disable CFG after 40% of steps
        pipeline._guidance_scale = 0.0
        # If the pipeline modifies prompt_embeds based on guidance_scale, 
        # ensure prompt_embeds batch size is also adjusted if guidance_scale is set to 0
        callback_kwargs["prompt_embeds"] = callback_kwargs["prompt_embeds"][0].unsqueeze(0) 
        # callback_kwargs["negative_prompt_embeds"] = callback_kwargs["negative_prompt_embeds"][0].unsqueeze(0) 
    return callback_kwargs



# ------------------------------------------------------



# Prompt scheduling (changing prompts during generation)
# Example: You can create a callback to swap or blend different prompt_embeds at specific steps. 
# MUST USE 'return callback_kwargs'
# image = pipeline(prompt="A dog", callback_on_step_end=prompt_scheduling_callback, callback_on_step_end_tensor_inputs=["prompt_embeds"]).images[0]
def prompt_scheduling_callback(pipeline, step_index, timestep, callback_kwargs):
    #rkconvert - NOT DONE
    # Example: Switch to a different prompt after 20 steps
    if step_index == 20:
        new_prompt = "A cat"
        new_prompt_embeds = pipeline._encode_prompt(new_prompt, ...).prompt_embeds
        callback_kwargs["prompt_embeds"] = new_prompt_embeds
        # callback_kwargs["negative_prompt_embeds"] = new_negative_prompt_embeds
    return callback_kwargs


# ------------------------------------------------------




# MUST USE 'return callback_kwargs'
# image = pipeline(prompt="A sunset", num_inference_steps=20, callback_on_step_end=interruption_callback).images[0]
def interruption_callback(pipeline, i, t, callback_kwargs):
    #rkconvert - NOT DONE
    if i >= 10:  # Stop after 10 steps
        pipeline._interrupt = True
    return callback_kwargs


 
# ----------------------------------



# ---------------------------------

def add_lora_model(model_name, loravalue):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
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

# ---------------------------------

def change_lora_model(model_name, loravalue):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
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
    #rkconvert - NOT DONE
    # rkpipeline FINISHED
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
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
    #rkconvert - NOT DONE
    # rkpipeline FINISHED

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
    global pipeline             # where the model is loaded to, convert/get_model_type uses private pipeline: piepline_xl
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



# ---------------------------------
# ---------------------------------


def tmyfunction(test_data):
    #rkconvert - NOT DONE
    temp_out = "Testing Dummy Function:</br>" + str(test_data)
    gr.Info(temp_out, duration=2.0, title="Dummy Function")
    return temp_out

# ---------------------------------

#used lora add
def lmyfunction(test_data, lval):
    #rkconvert - NOT DONE
    temp_out = "Testing Dummy Function:</br>" + "Lora: " + str(test_data) + "</br>Value: " + str(lval)
    gr.Info(temp_out, duration=2.0, title="Dummy Function")
    return temp_out

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------



# ---------------------------------


# creates full path from hub_models_dir/model_name
def get_hub_model_path_file(model_name):
    #rkconvert - NOT DONE
    model_path_file = os.path.join(LLSTUDIO["hub_model_dir"], model_name)
    return model_path_file


# ---------------------------------

# just reloads hub_model_list[] - called when app starts and to refresh hub_model_list[] items
def read_hub_model_dir():
    #rkconvert - NOT DONE
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
    #rkconvert - NOT DONE
    read_hub_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["hub_model_list"], interactive=True)


# ---------------------------------


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


# uses pyperclip to get the prompt from the OS clipboard.
# rknote WORKS in Win10,11, but NOT on a Raspberry Pi5 
# (latest version of Raspberry Pi OS, with desktop software as of 01/2025) ??
# new rkinfo found: RASPI5 uses 'Wayland' desktop, maybe 'pyperclip' only works with X11 ?? hmmm..
def paste_model_prompt():
    #rkconvert - NOT DONE
    clipboard_content = pyperclip.paste()
    return clipboard_content






# -----------------------------------------------------



def set_title_mode(tab_data: gr.SelectData):
    #rkconvert - NOT DONE
    # rkpipeline FINISHED

    # if int(LLSTUDIO["app_debug"]) > 0: print("set_title_mode() = Selected Mode: " + tab_data.value)
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

    my_title = f"<table cellspacing='1' cellpadding='1' border='0'><tr><td><img src='data:image/png;base64,{LLSTUDIO['llstudiologo']}' alt='{LLSTUDIO['app_title']}'></td><td><b><font size='+1'>Version: {LLSTUDIO['app_version']} - Device: {LLSTUDIO['friendly_device_name']} - Current Mode: {my_mode}</font></b></td</td></table>"
    
    return gr.update(value=my_title)




# ---------------------------------------------------------



# changes to the image output tab for generation
def change_tab():
    #rkconvert - NOT DONE
    # old vers
    # return gr.Tabs(selected="tab_t2i"), gr.Tabs(selected="tab_imageoutput")
    # new vers
    if int(STUDIO["gen_auto_image_tab"]["value"]) == 1:
        return gr.Tabs(selected="tab_ImageGeneration"), gr.Tabs(selected="tab_iout")
    else:
        return gr.Tabs(selected=""), gr.Tabs(selected="")

# ---------------------------------

# changes to the image output tab for generation
def change_tab_cnet():
    #rkconvert - NOT DONE
    # old vers
    # return gr.Tabs(selected="tab_t2i"), gr.Tabs(selected="tab_imageoutput")
    # new vers
    if int(STUDIO["gen_auto_image_tab"]["value"]) == 1:
        return gr.Tabs(selected="tab_ImageGeneration"), gr.Tabs(selected="tab_cnet")
    else:
        return gr.Tabs(selected=""), gr.Tabs(selected="")

# ---------------------------------


# not used, yet?
def change_to_tab(id):
    #rkconvert - NOT DONE
#    gr.Info(f"Changing tab to index {id}", duration=5.0, title="Generation")
    return gr.Tabs(selected=id)


# ---------------------------------------------------------
    
# rk default exit python/gradio back to prompt
def exit_app():
    #rkconvert - NOT DONE
    file1 = open(os.path.join(".", "restart.txt"), 'w')
    file1.write("0")
    file1.close()
    yield
    time.sleep(2)
    os._exit(os.X_OK)    



# ------------------------------------------------------



def restart_app():
    #rkconvert - NOT DONE
    file1 = open(os.path.join(".", "restart.txt"), 'w')
    file1.write("1")
    file1.close()
    yield
    time.sleep(2)
    os._exit(os.X_OK)    



# ------------------------------------------------------



def huggingface_on_app():
    #rkconvert - NOT DONE
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
    #rkconvert - NOT DONE
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


def huggingface_check_status_app():
    # # using - os.getenv - RKPREFFERED WAY
    hub_online = os.getenv('HF_HUB_OFFLINE', '1')
    if hub_online == '1':
        return "HuggingFace Hub is OFFLINE"
    else:
        return "HuggingFace Hub is ONLINE"


# ================================================================================
# ================================================================================
# ================================================================================

# =========================
# **** LCM-LoRA models **** 
# =========================
    
    
def delete_lcm_model(model_name, del_model, del_images): 
    #rkconvert - DONE
    contents = ""
    contents2 = ""
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
    


    

# ---------------------------------

# rknote not used yet, but WILL be used
def check_lcm_lora_model(model_name):
    model_path_file = os.path.join(LLSTUDIO["lcm_model_dir"], model_name)
    model_config_filename = os.path.join(model_path_file, "model_index.json")
    with open(model_config_filename, "r") as f:
        data = json.load(f)
    for key, value in data.items():
        if key == "scheduler":
            valuelist = value              # ['diffusers', 'LCMScheduler']
            if valuelist[0] == "diffusers":
                if valuelist[1] == "LCMScheduler":
                    return "LCM-LoRA Model"
                else:
                    return "NOT an LCM-LoRA Model"
            else:
                return "NOT an LCM-LoRA Model"



# ----------------------------------------------------------
# ----------------------------------------------------------


# creates full path from lcm_models_dir/model_name
def get_lcm_model_path_file(model_name):
    #rkconvert - DONE
    model_path_file = os.path.join(LLSTUDIO["lcm_model_dir"], model_name)
    return model_path_file


# ---------------------------------

# just reloads lcm_model_list[] - called when app starts and to refresh lcm_model_list[] items
def read_lcm_model_dir():
    #rkconvert - DONE
    LLSTUDIO["lcm_model_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lcm_model_list"].append(tmp_text)

    return "LCM-LoRA Model List Reloaded."


# ---------------------------------

# send back an updated grDropdown to update the lcm_model_list_dropdown
def update_lcm_model_list_dropdown():
    #rkconvert - NOT DONE
    read_lcm_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], interactive=True)


# ----------------------------------------------------------
# ----------------------------------------------------------

# just reloads lcm_model_list[] - called when app starts and to refresh lcm_model_list[] items
def read_lcm_sdonly_model_dir():
    #rkconvert - DONE
    LLSTUDIO["lcm_sdonly_model_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        model_name = entries[i]
        try:
            with open(os.path.join(get_lcm_model_path_file(model_name), "model_index.json"), "r") as f:
                model_config_data = json.load(f)
        except Exception as e:
            a=0 # we do nothing, we skip it. or it'll hold up entire rest of the list because of one model
            # return f"Error: 'model_index.json' File Not Found for {model_name}<br>\n"
            # but keeps app from blowing up :)

        if model_config_data:
            model_class_name = model_config_data["_class_name"]
            if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SD15":
                LLSTUDIO["lcm_sdonly_model_list"].append(model_name)

        
    return "LCM-LoRA (SD Only) Model List Reloaded."


# ---------------------------------


# send back an updated grDropdown to update the lcm_model_list_dropdown for seperate text encoder
def update_lcm_sdonly_model_list_dropdown():
    #rkconvert - NOT DONE
    read_lcm_sdonly_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_sdonly_model_list"], interactive=True)


# ----------------------------------------------------------
# ----------------------------------------------------------



def get_lcm_pipeclass_model_info(model_name):
    #rkconvert - NOT DONE
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
            myout = myout + f"Seperate Text Encoder and/or ControlNet Availiable for Use.<br>\n"
            
        if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SDXL":
            myout = myout + f"SDXL - {model_class_name} - {model_name}<br>\n"
            myout = myout + f"NOTE: No Seperate Text Encoder or ControlNet for SDXL Models.<br>\n"
            
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
                myout = myout + f"Seperate Text Encoder and/or ControlNet Availiable for Use.<br>\n"
                
            if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SDXL":
                myout = myout + f"SDXL - {model_class_name} - {model_name}<br>\n"
                myout = myout + f"NOTE: No Seperate Text Encoder or ControlNet for SDXL Models.<br>\n"
                
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


# ------------------------------------------------------

def get_lcm_pipeclass_model_list():
    #rkconvert - NOT DONE
    myout = "<h3>LCM-LoRA Model List</h3>\n"
    myout = myout + "<i>(Only StableDiffusionPipelines with SD/SDXL Model Classes.)</i><br>\n"
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        try:
            with open(os.path.join(get_lcm_model_path_file(tmp_text), "model_index.json"), "r") as f:
                model_config_data = json.load(f)

            model_class_name = model_config_data["_class_name"]

            if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SD15":
                myout = myout + f"SD15 - {model_class_name} - {tmp_text}<br>\n"
                
            if PIPECLASSES[model_class_name]['pipeline_model_type'] == "SDXL":
                myout = myout + f"SDXL - {model_class_name} - {tmp_text}<br>\n"
                
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
           
           
        except Exception as e:
            if int(STUDIO["app_debug"]["value"]) > 0: print(tmp_text + ": MODEL CONFIG NOT FOUND")

        myout = myout + "----------------------------------------<br>\n"
        
    return myout


# ------------------------------------------------------


# creates full path from lcm_models_dir/model_name
def get_lcm_pipeclass_model_path_file(model_name):
    #rkconvert - DONE
    model_path_file = os.path.join(LLSTUDIO["lcm_model_dir"], model_name)
    return model_path_file


# ---------------------------------

# just reloads lcm_model_list[] - called when app starts and to refresh lcm_model_list[] items
def read_lcm_pipeclass_model_dir():
    #rkconvert - DONE
    LLSTUDIO["lcm_model_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lcm_model_list"].append(tmp_text)

    return "LCM-LoRA Model List Reloaded."


# ---------------------------------

# send back an updated grDropdown to update the lcm_model_list_dropdown
def update_lcm_pipeclass_model_list_dropdown():
    #rkconvert - NOT DONE
    read_lcm_pipeclass_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], interactive=True)


# ----------------------------------------------------------
# ----------------------------------------------------------



# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------


def load_file_content(modelname):
    #rkconvert - NOT DONE
    mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
    if mdl_filename:
        try:
            with open(mdl_filename, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error loading file: {e}"
    return ""



# ---------------------------------


# **** LCM-LoRA images model info file **** 

def save_lcm_model_edit(modelname, content):
    #rkconvert - NOT DONE
    #rkconvert - NOT DONE
    mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
    if mdl_filename:
        try:
            with open(mdl_filename, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error loading file: {e}"
    return ""






# ---------------------------------


# **** LCM-LoRA images model info file **** 

def save_lcm_model_view(modelname, content):
    #rkconvert - NOT DONE
    gr.Info("<h3>View Model Card DISABLED !!</h3>", duration=3.0, title="View Model Card")
    return "View Model Card DISABLED !!"
    #rkconvert - NOT DONE
    mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
    if mdl_filename:
        try:
            with open(mdl_filename, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error loading file: {e}"
    return ""




# ---------------------------------


# **** LCM-LoRA images model info file **** 

def save_lcm_model_save(modelname, content):
    #rkconvert - NOT DONE
    mdl_filename = (os.path.join(LLSTUDIO["lcm_model_image_dir"],modelname,modelname + '.md'))
    if mdl_filename:
        try:
            with open(mdl_filename, 'w') as f:
                f.write(content)
            return f"File '{mdl_filename}' saved successfully!"
        except Exception as e:
            return f"Error saving file: {e}"
    return "No file selected to save."



# ---------------------------------


# **** SAFE images model info file **** 

def save_safe_model_save(modelname, content):
    #rkconvert - NOT DONE
    mdl_filename = (os.path.join(LLSTUDIO["safe_model_image_dir"],modelname,modelname + '.md'))
    if mdl_filename:
        try:
            with open(mdl_filename, 'w') as f:
                f.write(content)
            return f"File '{mdl_filename}' saved successfully!"
        except Exception as e:
            return f"Error saving file: {e}"
    return "No file selected to save."



# ---------------------------------


# **** LORA images model info file **** 

def save_lora_model_save(modelname, content):
    #rkconvert - NOT DONE
    mdl_filename = (os.path.join(LLSTUDIO["lora_model_image_dir"],modelname,modelname + '.md'))
    if mdl_filename:
        try:
            with open(mdl_filename, 'w') as f:
                f.write(content)
            return f"File '{mdl_filename}' saved successfully!"
        except Exception as e:
            return f"Error saving file: {e}"
    return "No file selected to save."



# ---------------------------------
 

 
# def select_lcm_model_view(themodel):
    # #rkconvert - NOT DONE
     # return "<h3>Selected LCM-LoRA Model: " + themodel +"</h3>", ""

# ---------------------------------

# def select_lcm_model_view_edit(themodel):
    # #rkconvert - NOT DONE
    # return load_file_content(themodel), "<h3>Selected LCM-LoRA Model: " + themodel +"</h3>"


# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------

# returns full path model images dir, dir/model
def get_lcm_model_image_path_file(model_name):
    #rkconvert - NOT DONE
    model_image_path_file = os.path.join(LLSTUDIO["lcm_model_image_dir"], model_name)
    return model_image_path_file

# ---------------------------------

def read_lcm_model_image_dir():
    #rkconvert - NOT DONE
    LLSTUDIO["lcm_model_image_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lcm_model_image_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lcm_model_image_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lcm_model_image_list"].append(tmp_text)

    return "LCM-LoRA Model Images Reloaded."
 

# ---------------------------------
 

# send back an updated grDropdown to update the lcmmodelview_dropdown    
def update_lcm_model_image_list_dropdown():
    #rkconvert - NOT DONE
    read_lcm_model_image_dir()
    return gr.Dropdown(choices=LLSTUDIO["lcm_model_image_list"], interactive=True)
   

# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------

# ====================
# **** SAFE models **** 
# ====================

def delete_safe_model(model_name, del_model, del_images):   
    #rkconvert - DONE
    contents = ""
    contents2 = ""
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
    #rkconvert - DONE
    safe_model_path_file = os.path.join(LLSTUDIO["safe_model_dir"], model_name)
    return safe_model_path_file

# ---------------------------------

# used ONLY for safetensors viewer, going to try and hijack the code for the other 3 viewers
# just reloads list[] - called when app starts and to refresh list items
def read_safe_model_dir():
    #rkconvert - DONE
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
    #rkconvert - NOT DONE
    read_safe_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["safe_model_list"], interactive=True)
   


# ------------------------------------------------------



def update_safe_convert_model_list_dropdown():
    #rkconvert - NOT DONE
    read_safe_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["safe_model_list"], interactive=True)
   

# ---------------------------------
# **** SAFE images **** 
   
# def select_safe_model_view(themodel):
    # #rkconvert - NOT DONE
    # return "<h3>Selected Safetensors Model: " + themodel +"</h3>", ""

# ---------------------------------
# returns full path model images dir, dir/model
def get_safe_model_image_path_file(model_name):
    #rkconvert - DONE
    return os.path.join(LLSTUDIO["safe_model_image_dir"], model_name)
    

# ---------------------------------
    

def read_safe_model_image_dir():
    #rkconvert - NOT DONE
    LLSTUDIO["safe_model_image_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["safe_model_image_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["safe_model_image_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["safe_model_image_list"].append(tmp_text)
    
    return "Safetensors Model Images Reloaded."

# ---------------------------------
 
    
def update_safe_model_image_list_dropdown():
    #rkconvert - NOT DONE
    read_safe_model_image_dir()
    return gr.Dropdown(choices=LLSTUDIO["safe_model_image_list"], interactive=True)

 

# ---------------------------------

# ====================
# **** LORA models **** 
# ====================
    
def delete_lora_model(model_name, del_model, del_images):    
    #rkconvert - DONE
    contents = ""
    contents2 = ""
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
    #rkconvert - DONE
    return os.path.join(LLSTUDIO["lora_model_dir"], model_name)



# ---------------------------------

# just reloads list[] - called when app starts and to refresh list items
def read_lora_model_dir():
    #rkconvert - NOT DONE
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
    #rkconvert - NOT DONE
    read_lora_model_dir()
    return gr.Dropdown(choices=LLSTUDIO["lora_model_list"], interactive=True)



# ------------------------------------------------------



# **** LORA images **** 
  
# def select_lora_model_view(themodel):
    # #rkconvert - NOT DONE
    # return "<h3>Selected Lora Model: " + themodel +"</h3>", ""


# ------------------------------------------------------




def get_lora_model_image_path_file(model_name):
    #rkconvert - DONE
    return os.path.join(LLSTUDIO["lora_model_image_dir"], model_name)

# ---------------------------------


def read_lora_model_image_dir():
    #rkconvert - NOT DONE
    LLSTUDIO["lora_model_image_list"] = []
    entries = [d for d in os.listdir(LLSTUDIO["lora_model_image_dir"]) if os.path.isdir(os.path.join(LLSTUDIO["lora_model_image_dir"], d))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        LLSTUDIO["lora_model_image_list"].append(tmp_text)
    
    return "Lora Model Images Reloaded."
 


# ---------------------------------
    
def update_lora_model_image_list_dropdown():
    #rkconvert - NOT DONE
    read_lora_model_image_dir()
    return gr.Dropdown(choices=LLSTUDIO["lora_model_image_list"], interactive=True)
    
    

    
# ---------------------------------
 
# ===LOADED LoRAs=============================================================================

  
# def select_loaded_lora_model(themodel):
    # #rkconvert - NOT DONE
   # return "<h3>Selected: " + themodel +"</h3>", ""



# ------------------------------------------------------



# def get_loaded_lora_models():
    # #rkconvert - NOT DONE
    # global pipeline
    # tempout = "Loaded LoRA Adapters: " + str(len(LLSTUDIO["loaded_lora_model_adapter"])) + "\n"

    # if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        # for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            # # tempout = tempout + "[" + str(i+1) + "]:Adapter Name: " + LLSTUDIO["loaded_lora_model_adapter"][i] + "\n"
            # tempout = tempout + "[" + str(i+1) + "]:LoRA Model Name: " + LLSTUDIO["loaded_lora_model_name"][i] + "\n"
            # tempout = tempout + "[" + str(i+1) + "]:LoRA Model Value: " + str(LLSTUDIO["loaded_lora_model_value"][i]) + "\n"
    
    # return tempout


# ------------------------------------------------------




def read_loaded_lora_models():
    #rkconvert - NOT DONE
    LLSTUDIO["loaded_lora_model_list"] = []
    if len(LLSTUDIO["loaded_lora_model_adapter"]) > 0:
        for i in range(len(LLSTUDIO["loaded_lora_model_adapter"])):
            LLSTUDIO["loaded_lora_model_list"].append(LLSTUDIO["loaded_lora_model_name"][i])
    
    return "Loaded Lora Models Reloaded."


# ------------------------------------------------------


 

def update_loaded_lora_model_list_dropdown():
    #rkconvert - NOT DONE
    read_loaded_lora_models()
    return gr.Dropdown(choices=LLSTUDIO["loaded_lora_model_list"], interactive=True)
    

# ---------------------------------



# ===================================================================
# Python Functions to be called for Prompt Weighting
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
# python function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
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
    # rknote: really couldn't think of anything to with this this a1111 syntax ?!? :)
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
# ===================================================================
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
   



    # 1. Handle .and() with segments: extract inside
    #    e.g. '("part one", "part two").and(1,0.5)' -> 'part one, part two'
    # This regex finds a tuple of quoted parts before .and
    # Helper: remove .and(...) syntax first, flatten quoted segments
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

    # **New part**: Remove parentheses that wrap something with no weight or +- after them
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
 



 

# ===================================================================
# ===================================================================
# ===================================================================

# # Your custom weights dictionary

def compute_weight_from_custom_orig(token: str, custom_dict: dict) -> float:
    key = token.lower()
    if key in custom_dict:
        return custom_dict[key]
    return 1.0



def compute_weight_from_custom(token: str, custom_dict: dict) -> float:
    key = token.lower()
    return custom_dict.get(key, 1.0)


# Builds a prompt string where tokens with weight differing enough from 1.0
# are replaced by the numeric weight syntax (word)weight.
# E.g. happy → (happy)1.5 if custom_weights["happy"]==1.5
def make_weighted_prompt_numeric_orig(prompt: str, custom_dict, threshold: float = 0.1) -> str:
    tokens = prompt.split()
    weighted_tokens = []
    for t in tokens:
        # You might want to strip punctuation for lookup, but preserve in output
        stripped = re.sub(r'[^A-Za-z0-9\s]', '', t).lower()
        weight = compute_weight_from_custom(stripped, custom_dict)
        if abs(weight - 1.0) < threshold:
            weighted_tokens.append(t)
        else:
            # Format to one decimal (or more) as needed
            # Put the stripped token (or original lowercase/original case) inside parentheses
            # followed by the numeric weight
            # If original token has punctuation, you may want to reattach
            weighted_tokens.append(f"({stripped}){weight:.1f}")
    return " ".join(weighted_tokens)


# called by python function - make_weighted_prompts()
# Replace words or phrases from the prompt with (token)weight if in custom_dict
# and their weight is different from 1.0 by threshold.
# If any two phrases overlap (like "epic" and "epic adventure"), the longer one takes precedence.
# and... Punctuation is preserved in the output.
def make_weighted_prompt_numeric(prompt: str, custom_dict: dict, threshold: float = 0.1) -> str:
    prompt_lower = prompt.lower()
    output = prompt
    replacements = []

    # Sort phrases by length descending to match longer phrases first
    sorted_keys = sorted(custom_dict.keys(), key=lambda x: -len(x))

    for key in sorted_keys:
        pattern = re.escape(key.lower())
        matches = list(re.finditer(r'\b' + pattern + r'\b', prompt_lower))
        
        for match in matches:
            start, end = match.span()
            original_text = prompt[start:end]
            weight = compute_weight_from_custom(key, custom_dict)

            if abs(weight - 1.0) >= threshold:
                weighted = f"({original_text}){weight:.1f}"
                replacements.append((start, end, weighted))

    # Avoid overlapping replacements
    replacements.sort()
    result = []
    last_index = 0
    for start, end, weighted in replacements:
        if start >= last_index:
            result.append(output[last_index:start])
            result.append(weighted)
            last_index = end

    result.append(output[last_index:])
    return ''.join(result)
    
# ===================================================================================    
# ===================================================================================    


# Gradio Gallery expects a list of paths
# rk note used2
# 1. to populate gallery on load/launch 
# 2. to refresh list after delete
# 3. will be used to change image directory
# from 
def get_sorted_newest_image_list():
    #rkconvert - NOT DONE
    output_image_list = []
    entries = [f for f in os.listdir(LLSTUDIO["advanced_gallery_dir"]) if os.path.isfile(os.path.join(LLSTUDIO["advanced_gallery_dir"], f))]
    for i in range(len(entries)):
        tmp_text = entries[i]
        # get image files only
        if tmp_text.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            output_image_list.append(os.path.join(LLSTUDIO["advanced_gallery_dir"], tmp_text))
    # rknote - since i used full path, we can sort by date, newest first.
    # modified time 'm'
    # output_image_list.sort(key=os.path.getmtime, reverse=True)
    # created time 'c'
    output_image_list.sort(key=os.path.getctime, reverse=True)
    return output_image_list

# ------------------------------------------------------

# rk note used 1
def get_text_content(evt: gr.SelectData):
    if evt.value is None:
        return "", None
    temp_dict = evt.value
    # print(temp_dict)
    image_path = os.path.join(".", "output", temp_dict["image"]["orig_name"])
    LLSTUDIO["gallery_selected_image"] = image_path
    # print("\n")
    # print("myimagepath=")
    # print(image_path)
    # print("\n")
    base, _ = os.path.splitext(image_path)
    text_path = f"{base}.txt"

    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read(), image_path
    else:
        return f"No Generation Parameter *.txt file found at '{text_path}'", image_path

# ------------------------------------------------------

# rk note used1
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

# ------------------------------------------------------

def man_images_outputs_fun():
    LLSTUDIO["advanced_gallery_dir"] = os.path.join(".","output") + os.sep

def man_images_lcm_fun():
    LLSTUDIO["advanced_gallery_dir"] = os.path.join(".","lcm_models_images", "LCM_frankjoshua_toonyou_beta6") + os.sep



# ===================================================================
# ===================================================================
# for creating a logo that the login screen can use since the gradio_api 
# server is technically not running/allowing connections
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



# ------------------------------------------------------------------------------

# rknote: rknotused code, but keeping it in here, maybe useful in the future
# A single, reusable function to clear a variable number of outputs
def clear_outputs(num_outputs):
    yield tuple([None] * int(num_outputs))



# ------------------------------------------------------



# RKMAGIC to see hidden image 
def display_generated_image():
# enables/disables hidden image to visible image on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
    if LLSTUDIO["hidden_image_flag"] == 1:
        return LLSTUDIO['last_image_filename']

#oimage,oimage2 



# ----------------------------------------

# rknotused no more for now, good for something maybe...
# was replaced by: clear_generation_status_and_images()
# RKMAGIC to clear both images upon start of generation
def clear_generated_images():
# enables/disables hidden image to visible image on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 0
    yield None, None


#oimage,oimage2 


# ----------------------------------------

# RKMAGIC to clear both images upon start of generation
def clear_generation_status_and_images():
# enables/disables hidden image to visible image on change copy from oimage to oimage2
# 0 = disabled, 1 = enabled
    LLSTUDIO["hidden_image_flag"] = 0
    yield None, None, None, None


#oimage, oimage2, inference_status_markdown, gallery_html


# ----------------------------------------


def grinfo_no_model_loaded():
    gr.Info("<h4>No Model Loaded. Please Load a Model First.</h4><h4>Select the tab 'Pipeline - Models' to load a model into the pipeline.</h4>", duration=3.0, title="Load Model")  

# could combine these two into one string...
# ----------------------------------------
# ...but this one needs the H4 tag

def str_no_model_loaded():
    return "<h4>No Model Loaded. Please Load a Model First. - Select the tab 'Pipeline - Models' to load a model into the pipeline.</h4>"



# ------------------------------------------------------
# ------------------------------------------------------


def update_state(name):
    return name



# ------------------------------------------------------
# ------------------------------------------------------


# # ====================================================================================
# # ======END==========FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====FUNCTIONS====
# # ====================================================================================




# ================================================================================
# =======START APP====START APP====START APP====START APP====START APP============
# ================================================================================


tstart = time.time()
pstart = time.time()

# # creates default model, image directories if they do not exist.
# # uses the information in the settings to create them
# make_all_default_dirs()

# select the inference DEVICE
device_select()

# gives UI a different default/starting seed everytime you start the app
default_seed=gen_random_seed()


# load lists of models and images for model viewer
read_hub_model_dir()
read_lcm_model_dir()
read_lcm_sdonly_model_dir()
read_lcm_model_image_dir()
read_safe_model_dir()
read_safe_model_image_dir()
read_lora_model_dir()
read_lora_model_image_dir()


print("------------------------------------------")
print(LLSTUDIO["app_title"] + " - " + LLSTUDIO["app_version"])
if STUDIO["auth_use"]["value"]:
    print("Authentication will be required !")
print("------------------------------------------")

# generate base64 logo for title and login screen
LLSTUDIO['llstudiologo'] = png_to_base64_string(os.path.join(".", "lcm-lora-studio-logo.png"))
LLSTUDIO['llstudiologo_login'] = png_to_base64_string(os.path.join(".", "lcm-lora-studio-logo2.png"))



# ===================================================================
# ===================================================================
# ===================================================================

# ===================================================
# gradio stuff
#

# javascript and css code section
   
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
#pasteprompt_button {
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
"""


# ===================================================================
# ===================================================================
# ===================================================================




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


# ===================================================================
# ===================================================================
# ===================================================================


head_js_code = """
<script>
document.addEventListener('click', function (event) {
  if (event.target && event.target.matches('.copy-button')) {
    try {
      var button = event.target;
      var row = button.closest('tr');
      var codeTag = row ? row.querySelector('code') : null;
      if (!codeTag) return;

      var textToCopy = codeTag.textContent || '';

      function showCopiedIndicator() {
        // Ensure button container is positioned before measuring
        var container = button.parentElement;
        var containerWasRelative = container.style.position === 'relative';
        if (!containerWasRelative) container.style.position = 'relative';

        // Get accurate positioning
        var buttonRect = button.getBoundingClientRect();
        var containerRect = container.getBoundingClientRect();

        var indicator = document.createElement('span');
        indicator.textContent = 'Copied!';
        indicator.style.position = 'absolute';
        indicator.style.background = '#333';
        indicator.style.color = '#fff';
        indicator.style.padding = '3px 8px';
        indicator.style.borderRadius = '4px';
        indicator.style.fontSize = '12px';
        indicator.style.whiteSpace = 'nowrap';
        indicator.style.top = (button.offsetTop - 25) + 'px';
        indicator.style.left = (button.offsetLeft + button.offsetWidth / 2 - 25) + 'px';
        indicator.style.transition = 'opacity 0.3s ease';
        indicator.style.opacity = '1';
        indicator.style.pointerEvents = 'none';

        container.appendChild(indicator);

        setTimeout(function () {
          indicator.style.opacity = '0';
          setTimeout(function () {
            if (indicator.parentElement) indicator.parentElement.removeChild(indicator);
            // restore original positioning style
            if (!containerWasRelative) container.style.position = '';
          }, 300);
        }, 1200);
      }

      function fallbackCopyText(text) {
        try {
          var textarea = document.createElement('textarea');
          textarea.value = text;
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          document.body.removeChild(textarea);
          showCopiedIndicator();
        } catch (e) {
          // silently fail
        }
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(textToCopy)
          .then(showCopiedIndicator)
          .catch(function () {
            fallbackCopyText(textToCopy);
          });
      } else {
        fallbackCopyText(textToCopy);
      }

    } catch (err) {
      // silently catch errors
    }
  }
});

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


# ------------------------------------------------------------------------------



# --- ui start ---

theme = gr.themes.Default(primary_hue="orange",)





# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# GRADIO UI
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
grapptitle = f"<table cellspacing='1' cellpadding='1' border='0'><tr><td><img src='data:image/png;base64,{LLSTUDIO['llstudiologo']}' alt='{LLSTUDIO['app_title']}'></td><td><b><font size='+1'>Version: {LLSTUDIO['app_version']} - Device: {LLSTUDIO['friendly_device_name']} - Current Mode: Text to Image</font></b></td</td></table>"

# The parameters for Gradio's gr.Blocks() 
# most of them are static
# not seeing a reason to turn any of these into settings.
blocks_kwargs = {}
blocks_kwargs["fill_height"] = True
blocks_kwargs["delete_cache"] = (3600, 3600)
blocks_kwargs["analytics_enabled"] = False          # small effort to make this app 100% offline
blocks_kwargs["title"] = LLSTUDIO["app_title"]      # web browser page/tab/window name
blocks_kwargs["theme"] = theme
blocks_kwargs["head"] = head_js_code
blocks_kwargs["css"] = css_code

with gr.Blocks(**blocks_kwargs) as lcmlorastudio:
    # hidden controls for dynamic prompts
    hidden_prompt_name = gr.Textbox(value="t2iprompt_txt", visible=False)   # default control name, gets changed
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
        app_title_label = gr.HTML(elem_id="no-borders", value=grapptitle)
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=100): 
            model_list_html = gr.HTML("<h4>No Model Loaded. Please Load a Model First. - Select the tab 'Pipeline - Models' to load a model into the pipeline.</h4>")
        with gr.Column(scale=0, min_width=100):
            pipeline_delete_button = gr.Button("", icon="./icons/trash64.png", elem_id="deletemodel_button")    

# ==================================================================================================================

    with gr.Tabs(selected="tab_ImageGeneration") as tabs:

        with gr.Tab("Image Generation", id="tab_ImageGeneration"):

            with gr.Tabs(selected="tab_t2i") as inner_tab_ImageGeneration:

                with gr.Tab("Text to Image", id="tab_t2i"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            t2iprompt_txt = gr.Textbox(value=LLSTUDIO["def_prompt"], label="Prompt", lines=4, elem_id="js_t2iprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            t2iprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            t2inegprompt_txt = gr.Textbox(value=LLSTUDIO["def_negprompt"], label="Negative Prompt", lines=4, elem_id="js_t2inegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            t2inegprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            i2iprompt_txt = gr.Textbox(value=LLSTUDIO["def_prompt"], label="Prompt", lines=4, elem_id="js_i2iprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            i2iprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            i2inegprompt_txt = gr.Textbox(value=LLSTUDIO["def_negprompt"], label="Negative Prompt", lines=4, elem_id="js_i2inegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            i2inegprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                                i2igen_width = gr.Slider(label="Image Width", value=512, minimum=256, maximum=1024, step=256)
                                i2igen_height = gr.Slider(label="Image Height", value=512, minimum=256, maximum=1024, step=256)
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
                            inpprompt_txt = gr.Textbox(value=LLSTUDIO["def_prompt"], label="Prompt", lines=4, elem_id="js_inpprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            inpprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            inpnegprompt_txt = gr.Textbox(value=LLSTUDIO["def_negprompt"], label="Negative Prompt", lines=4, elem_id="js_inpnegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            inpnegprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                                inpgen_width = gr.Slider(label="Image Width", value=512, minimum=256, maximum=1024, step=256)
                                inpgen_height = gr.Slider(label="Image Height", value=512, minimum=256, maximum=1024, step=256)
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
                            ip2pprompt_txt = gr.Textbox(value=LLSTUDIO["def_prompt"], label="Prompt", lines=4, elem_id="js_ip2pprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            ip2pprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            ip2pnegprompt_txt = gr.Textbox(value=LLSTUDIO["def_negprompt"], label="Negative Prompt", lines=4, elem_id="js_ip2pnegprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            ip2pnegprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            up2xprompt_txt = gr.Textbox(value=LLSTUDIO["def_prompt"], label="Prompt", lines=4, elem_id="js_up2xprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            up2xprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
                            up2xprompt_test_button = gr.Button("", icon="./icons/test64.png", elem_id="testprompt_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            up2xnegprompt_txt = gr.Textbox(value=LLSTUDIO["def_negprompt"], label="Negative Prompt", lines=4, elem_id="js_up2xnegprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            up2xnegprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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


# ---------controlnet-----------------------------------------------------------------controlnet---------



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
                            cnetprompt_txt = gr.Textbox(value=LLSTUDIO["def_prompt"], label="Prompt", lines=4, elem_id="js_cnetprompt_txt", show_label=True, show_copy_button=True)
                        with gr.Column(scale=0, min_width=100):
                            cnetprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                            cnetnegprompt_txt = gr.Textbox(value=LLSTUDIO["def_negprompt"], label="Negative Prompt", lines=4, elem_id="js_cnetnegprompt_txt", show_label=True, show_copy_button=True, info="Ignored when not using guidance (`guidance_scale < 1`)")
                        with gr.Column(scale=0, min_width=100):
                            cnetnegprompt_paste_button = gr.Button("", icon="./icons/pastetext64.png", elem_id="pasteprompt_button")
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
                                cnetgen_width = gr.Slider(label="Image Width", value=512, minimum=256, maximum=1024, step=256)
                                cnetgen_height = gr.Slider(label="Image Height", value=512, minimum=256, maximum=1024, step=256)
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





# -----------------------------------------------------------------------------------


                with gr.Tab("Output Image", id="tab_iout"):
                    with gr.Row():
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
            # -----------------------------------------------------------------------------------------------------
            with gr.Tabs(selected="tab_lml") as inner_tab_PipelineModels:
                with gr.Tab("LCM-LoRA Model List", id="tab_lml"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_list_html = gr.HTML("Load saved LCM-LoRA type models here. Auto loads correct pipeline based on model config file. Can select a different Text Encoder from another LCM-LoRA model (SD Only). (NOTE: LCM-LoRA Models that show up in the dropdown have been saved with the 'Save Model' operation, and 'normally with' the LCM-LoRA weights fused.)")
                            gr.Markdown("<br>")
                            lcm_model_list_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_model_list"], label="Availiable LCM-LoRA Models (Local Saved LCM-LoRA Models)")
                            load_lcm_model_fp16_check = gr.Checkbox(value=1,label="variant fp16")
                        with gr.Column(scale=0, min_width=100):
                            lcm_model_reload_list_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
                            lcm_model_info_button = gr.Button("", icon="./icons/about64.png", elem_id="reloadmodellist_button")
                            lcm_model_load_model_button = gr.Button("", icon="./icons/load64.png", elem_id="loadmodel_button")
                    with gr.Row(equal_height=True):
                        gr.Markdown("Use a seperate text encoder for image variations from same model (SD Only)")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            lcm_model_use_diff_text_encoder_check = gr.Checkbox(label="Use Seperate Text Encoder")
                            lcm_model_liste_dropdown = gr.Dropdown(choices=LLSTUDIO["lcm_sdonly_model_list"], label="Availiable LCM-LoRA Models to load Seperate Text Encoder (SD Only)")
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
                            hub_model_add_lcmlora = gr.Checkbox(label="Auto Add LCM-LoRA")


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
                        hug_pipeline_classes = gr.Dropdown(choices=PIPELINE_CLASSES, label="Select a Pipeline Class to load model", value=PIPELINE_CLASSES[0])
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
                            safeload_model_get_type_button = gr.Button("", icon="./icons/question64.png", elem_id="getsafemodeltype_button")
                            safeload_model_load_button = gr.Button("", icon="./icons/load64.png", elem_id="converttolcmmodel_button")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=100):
                            safeload_pipeline_classes = gr.Dropdown(choices=PIPELINE_CLASSES, label="Select a Pipeline Class to load model", value=PIPELINE_CLASSES[0])
                        with gr.Column(scale=1, min_width=100):
                            safeload_model_add_lcmlora = gr.Checkbox(label="Auto Add LCM-LoRA")
                    with gr.Row(equal_height=True):
                        safeload_model_lora = gr.Slider(label="LCM-LoRA Scale", value=1.0, minimum=0.1, maximum=10, step=0.1)
     
                
        with gr.Tab("Add Lora Models", id="tab_AddLoraModels") as inner_tab_AddLoraModels:
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    loradropdown = gr.Dropdown(choices=LLSTUDIO["lora_model_list"], label="Availiable Lora Models to Add to Loaded Model")
                with gr.Column(scale=0, min_width=100):
                    reload_lora_button = gr.Button("", icon="./icons/refresh64.png", elem_id="reloadmodellist_button")
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
                    lorahtml = gr.HTML("<p>If adding the LCM-LoRAs weights here, rather than loaded automatically when the model is loaded<br>Make sure to check the weight scale for the LoRA before loading.<br>Should be set to '1.0' for the LCM-LoRAs. Feel free to experiment. :)<br>NOTE: The LoRA Scale 'slider' will go from '-10' to '+10' to account for a few LoRA models I ran across which use both a postive, 0 or a negative value. Consult the model card for your LoRA model for more information on adjusting the LoRA weight.</p>")
                


        with gr.Tab("Save as LCM-LoRA Model", id="tab_SaveLCMModel") as inner_tab_AddLoraModels:
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_htmlt = gr.HTML("<h3>Save Loaded Pipeline (Model) (with fused LoRAs) to New LCM-LoRA Model</h3>")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=100):
                    save_lcm_model_txt = gr.Textbox(value=LLSTUDIO["lcm_model_prefix"]+"MyNewModel"+LLSTUDIO["lcm_model_suffix"], info="Enter name for new LCM-LoRA model. Will be saved in LCM-LoRA Models Directory. (a-Z,0-9,_ only)", label="New LCM-LoRA Model Name", lines=1, show_label=True, show_copy_button=True)
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
            # -----------------------------------------------------------------------------------------------------
            with gr.Tabs(selected="tab_ov") as inner_tab_ModelGalleryViewers:
                # -------------------------------------------------------------------------------------------------
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
                                # lcmmodelview_html2 = gr.HTML("")
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

# ------tools tab --------------------------

        with gr.TabItem("Tools", id="tab_Tools"):



# -----------------begin image processing tab-------------------------------

            with gr.TabItem("Image Processing", id="tab_Image_Processing"):
# -----------------begin image processing ui--------------------------------

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
                    # main left column
                    with gr.Column(scale=1, min_width=150):

                        with gr.Row():
                            with gr.Column(scale=0, min_width=300):
                                adjusted_output = gr.Image(label="Stage 1 - Adjusted Image", elem_classes=["custom-image"], scale=0, height=300, show_download_button=False, interactive=False)
                            with gr.Column(scale=0, min_width=300):
                                grayscale_output = gr.Image(label="Stage 2 - Grayscale Image", elem_classes=["custom-image"], scale=0, height=300, show_download_button=False, interactive=False)
                                depth_map_button = gr.Button("Depth Map Only")
                                send_gray_to_cnet_button = gr.Button("Send to ControlNet")
                                
                    # main middle column
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
                            
                    # main right column
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

                        # rows() built with embedded function 'blur_controls()'
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

# ----------------------------------------------------------------------------------------------


            with gr.TabItem("OpenPose Editor", id="tab_OpenPose_Editor"):
                with gr.Row(equal_height=True):
                    openpose_edit_title = gr.Markdown("## OpenPose Editor")
                with gr.Row(equal_height=True):
                    x_openpose_html = gr.HTML(openpose_html)

# ----------------------------------------------------------------------------------



            with gr.TabItem("Manage Images", id="tab_ManageImages"):
                # # State variable to hold selected files for deletion
                man_image_selected_images_state = LLSTUDIO["gallery_selected_image"]
                man_images_selected_images_state = gr.HTML(visible=False)
                # STUDIO["advanced_gallery_dir"]["value"]
                with gr.Row(equal_height=True):
                    gr.Markdown("## Image and Generation Text Parameter Deletion Tool")
                with gr.Row(equal_height=True):
                    gr.Markdown("Click an image in the gallery to view its text.")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        # Gallery to preview the images and handle selection for text view
                        man_images_gallery = gr.Gallery(
                            label="Image preview (Click to view text)",
                            value=get_sorted_newest_image_list(),
                            columns=4,
                            rows=3,
                            object_fit="scale-down",       # Can be "contain", "cover", "fill", "none", or "scale-down"
                            height="auto",
                            type="pil",             #Can be "pil", "filepath"
                            allow_preview=True,
                            show_download_button=False,
                            selected_index=0,
                            elem_id="my_gallery"
                        )
                    with gr.Column(scale=2):
                        # Textbox to view content of image generation text file
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

                with gr.Row(equal_height=True):
                    gr.Markdown("## Manage models by deleting unused/unwanted models.")

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


# ----------------------------------------------------------------------------------

            with gr.TabItem("System", id="tab_System"):
                with gr.Accordion("Control System/Application", open=True):
                    with gr.Row(equal_height=True):
                        sysinfo_haltgen_button = gr.Button("Halt Image Generation", scale=0, elem_id="gray_button")
                    with gr.Row(equal_height=True):
                        sysinfo_logout_button = gr.Button("Logout", scale=0, elem_id="yellow_button")
                        sysinfo_reload_button = gr.Button("Reload Browser", scale=0, elem_id="blue_button")
                        sysinfo_restart_button = gr.Button("RESTART", scale=0, elem_id="exit_button")
                        sysinfo_exit_button = gr.Button("EXIT", scale=0, elem_id="exit_button")
                    with gr.Row(equal_height=False):
                        sysinfo_hug_on_button = gr.Button("Huggingface ON", scale=0, elem_id="exit_button")
                        sysinfo_hug_off_button = gr.Button("Huggingface OFF", scale=0, elem_id="exit_button")
                        sysinfo_hug_check_button = gr.Button("Check HF Status", scale=0, elem_id="purple_button")
                        sysinfo_hug_status = gr.Textbox(label="Huggingface On/Off Status", value="Click the 'Check HF Status' to check status", info="You can also check the enviroment variables too.")
                with gr.Accordion("View System Information", open=False):
                    with gr.Row(equal_height=True):
                        sysinfo_memory_button = gr.Button("Memory", scale=1, elem_id="generates_button")
                        sysinfo_hfcache_button = gr.Button("HF Cache", scale=1, elem_id="generate_button")
                        sysinfo_env_button = gr.Button("Enviroment Variables", scale=1, elem_id="gray_button")
                        sysinfo_sysinfo_button = gr.Button("System Information", scale=1, elem_id="generate_button")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=100):
                        sysinfo_html = gr.HTML("")

# ----------------------------------------------------------------------------------


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



# ----------------------------------------------------------------------------------




            with gr.TabItem("Help", id="tab_Help"):
                with gr.Row(equal_height=True):
                    html_title = gr.Markdown("## LCM-LoRA Studio Help")
                with gr.Row(equal_height=True):
                    x_help_html = gr.HTML(help_html)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------



# --------------------------------

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


# -----------------end of image processing ui event code--------------------------------

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
    # Refresh the gallery AFTER deletion, rk note - why not use a .then() ???
    man_images_delete_btn.click(
        fn=get_sorted_newest_image_list,
        outputs=man_images_gallery
    )

    
    # # either the .select() or .change() event listener.
    # man_images_gallery.change(
        # fn=get_sorted_newest_image_list,
        # outputs=man_images_gallery
    # )


    # man_images_reload_button.click(None, None, None, js="() => { window.location.reload(true); }")
    

# ------------------------------------------------------------------------------------------------------------------

    # # # PIPELINE
    
    # Pipeline section
    pipeline_delete_button.click(delete_pipeline, None, outputs=[model_list_html])


    # # # MODELS - LCMLORA - HUB CACHE - HUGGINGFACE

    # LCM-LoRA Model section
    lcm_model_reload_list_button.click(update_lcm_model_list_dropdown, None, lcm_model_list_dropdown)
    lcm_model_info_button.click(get_lcm_pipeclass_model_info, lcm_model_list_dropdown, lcm_model_list_html)
    # # rknote CONTROLNET must add to input list: [lcm_model_use_controlnet, lcm_model_cnet_dropdown]
    lcm_model_load_model_button.click(load_lcm_model, inputs=[lcm_model_list_dropdown, lcm_model_use_diff_text_encoder_check, lcm_model_liste_dropdown, lcm_model_clipskip, lcm_model_use_controlnet, lcm_model_cnet_dropdown, lcm_model_use_controlnet2, lcm_model_cnet_dropdown2, load_lcm_model_fp16_check, load_lcm_modele_fp16_check], outputs=[lcm_model_list_html]).then(display_pipeline_info, inputs=[lcm_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html])
    # Load seperate text encoder LCM-LoRA model list
    lcm_model_reload_liste_button.click(update_lcm_sdonly_model_list_dropdown, None, lcm_model_liste_dropdown)





    # HUB - HUGGGINFACE Local Cache Model section
    hub_model_reload_list_button.click(update_hub_model_list_dropdown, None, hub_model_list_dropdown) 
    hub_model_load_model_button.click(load_hub_model, inputs=[hub_model_list_dropdown, hub_model_fp16_check, hub_model_lora, hub_model_add_lcmlora], outputs=[hub_model_list_html]).then(display_pipeline_info, inputs=[hub_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html])
    hub_model_info_button.click(get_hub_pipeclass_model_info, hub_model_list_dropdown, hub_model_list_html)
   
    
    
    # HUG - HUGGGINFACE Model section
    hug_model_download_model_button.click(load_hug_model, inputs=[hug_model_txt, hug_pipeline_classes, hug_model_fp16_check], outputs=[hug_model_list_html]).then(display_pipeline_info, inputs=[hug_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html])

    # SAFETENSORS Model section
    safeload_model_reload_button.click(update_safe_convert_model_list_dropdown, None, safeload_model_dropdown) 
    safeload_model_load_button.click(load_safetensors_model, inputs=[safeload_model_dropdown,safeload_pipeline_classes, safeload_model_lora,safeload_model_add_lcmlora], outputs=[safeload_model_list_html]).then(display_pipeline_info, inputs=[safeload_model_list_html], outputs=[model_list_html, lcm_model_list_html, hub_model_list_html, hug_model_list_html, safeload_model_list_html])



# ------------------------------------------------------------------------------------------------------------------
    # # # TAB - Image Generation

    inner_tab_ImageGeneration.select(set_title_mode, None, app_title_label)


    # # # TEXT 2 IMAGE

    t2iprompt_paste_button.click(paste_model_prompt, None, t2iprompt_txt)
    t2iprompt_test_button.click(get_prompt_length_tokens, inputs=[t2iprompt_txt], outputs=[t2iprompt_txt])

    t2inegprompt_paste_button.click(paste_model_prompt, None, t2inegprompt_txt)
    t2inegprompt_test_button.click(get_negprompt_length_tokens, inputs=[t2inegprompt_txt], outputs=[t2inegprompt_txt])

    
    # Generation section
    t2igen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[t2igen_seedval])

    t2igen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(t2igen_LCM_images, inputs=[t2iprompt_txt, t2inegprompt_txt, t2igen_width, t2igen_height, t2igen_guidance, t2igen_inference_steps, t2igen_num_images, t2igen_seedval, t2igen_sameseed_check, t2igen_incrementseed_check, t2igen_incrementseed_amount, t2igen_freeu_check, t2igen_freeu_s1, t2igen_freeu_s2, t2igen_freeu_b1, t2igen_freeu_b2, lcm_model_clipskip], outputs=[inference_status_markdown, oimage])
    
    t2igen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])
    
    t2igen_default_freeu_button.click(set_freeu_values, inputs=[t2igen_freeu_s1, t2igen_freeu_s2, t2igen_freeu_b1, t2igen_freeu_b2], outputs=[t2igen_freeu_s1, t2igen_freeu_s2, t2igen_freeu_b1, t2igen_freeu_b2])

    # RKMAGIC
    oimage.change(display_generated_image, None, oimage2)
    
    
    # t2i prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    t2iaddweight_button.click(fn=None, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    t2iaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    t2iaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    t2imodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    t2iremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, t2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    t2iclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    
    
    # i2i prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    i2iaddweight_button.click(fn=None, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    i2iaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    i2iaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    i2imodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    i2iremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, i2iweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    i2iclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
    # inp prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    inpaddweight_button.click(fn=None, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    inpaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    inpaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    inpmodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    inpremove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, inpweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    inpclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
    # ip2p prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    ip2paddweight_button.click(fn=None, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    ip2paddpweight_button.click(fn=None, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    ip2paddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    ip2pmodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    ip2premove_a1111_syntax_button.click(fn=remove_a1111_syntax, inputs=[hidden_prompt_name, ip2pweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
    # python function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    ip2pclean_compel_prompt_button.click(fn=clean_compel_prompt, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt])
 
    # up2x prompt helper
    # -------------------------
    # None, no embedded prompts for SD latent Upscale
 
    # controlnet prompt helper
    # -------------------------
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    cnetaddweight_button.click(fn=None, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_weight)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    cnetaddpweight_button.click(fn=None, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_param_addweight)
    # javascript function call - inputs=[hidden_prompt_name, ALL PROMPTS]
    cnetaddparens_button.click(fn=None, inputs=[hidden_prompt_name, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_add_parens)
    # javascript function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
    cnetmodifyweight_button.click(fn=None, inputs=[hidden_prompt_name, cnetweight_number, t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], outputs=[t2iprompt_txt, t2inegprompt_txt, i2iprompt_txt, i2inegprompt_txt, inpprompt_txt, inpnegprompt_txt, ip2pprompt_txt, ip2pnegprompt_txt, up2xprompt_txt, up2xnegprompt_txt, cnetprompt_txt, cnetnegprompt_txt], js=js_modify_param_weight)
    # python function call - inputs=[hidden_prompt_name, ZZZZweight_number, ALL PROMPTS]
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


    i2iprompt_paste_button.click(paste_model_prompt, None, i2iprompt_txt)
    i2iprompt_test_button.click(get_prompt_length_tokens, inputs=[i2iprompt_txt], outputs=[i2iprompt_txt])

    i2inegprompt_paste_button.click(paste_model_prompt, None, i2inegprompt_txt)
    i2inegprompt_test_button.click(get_negprompt_length_tokens, inputs=[i2inegprompt_txt], outputs=[i2inegprompt_txt])

    # Generation section
    i2igen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[i2igen_seedval])

    i2igen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(i2igen_LCM_images, inputs=[i2iprompt_txt, i2inegprompt_txt, i2igen_width, i2igen_height, i2igen_guidance, i2igen_inference_steps, i2igen_seedval, i2igen_num_images, i2igen_incrementseed_check, i2igen_incrementseed_amount, i2iimage, i2igen_resize_input_image_check, i2igen_freeu_check, i2igen_freeu_s1, i2igen_freeu_s2, i2igen_freeu_b1, i2igen_freeu_b2, lcm_model_clipskip, i2igen_strength], outputs=[inference_status_markdown, oimage])
    
    i2igen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    i2igen_default_freeu_button.click(set_freeu_values, inputs=[i2igen_freeu_s1, i2igen_freeu_s2, i2igen_freeu_b1, i2igen_freeu_b2], outputs=[i2igen_freeu_s1, i2igen_freeu_s2, i2igen_freeu_b1, i2igen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # INPAINING


    inpprompt_paste_button.click(paste_model_prompt, None, inpprompt_txt)
    inpprompt_test_button.click(get_prompt_length_tokens, inputs=[inpprompt_txt], outputs=[inpprompt_txt])

    inpnegprompt_paste_button.click(paste_model_prompt, None, inpnegprompt_txt)
    inpnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[inpnegprompt_txt], outputs=[inpnegprompt_txt])

    # Generation section
    inpgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[inpgen_seedval])

    inpgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(inpgen_LCM_images, inputs=[inpprompt_txt, inpnegprompt_txt, inpgen_width, inpgen_height, inpgen_guidance, inpgen_inference_steps, inpgen_seedval, inpgen_num_images, inpgen_incrementseed_check, inpgen_incrementseed_amount, inpimage, inpgen_resize_input_image_check, inpimagemask, inpgen_freeu_check, inpgen_freeu_s1, inpgen_freeu_s2, inpgen_freeu_b1, inpgen_freeu_b2, lcm_model_clipskip, inpgen_strength], outputs=[inference_status_markdown, oimage])
    
    inpgen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    inpgen_default_freeu_button.click(set_freeu_values, inputs=[inpgen_freeu_s1, inpgen_freeu_s2, inpgen_freeu_b1, inpgen_freeu_b2], outputs=[inpgen_freeu_s1, inpgen_freeu_s2, inpgen_freeu_b1, inpgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # INSTRUCT PIX2PIX


    ip2pprompt_paste_button.click(paste_model_prompt, None, ip2pprompt_txt)
    ip2pprompt_test_button.click(get_prompt_length_tokens, inputs=[ip2pprompt_txt], outputs=[ip2pprompt_txt])

    ip2pnegprompt_paste_button.click(paste_model_prompt, None, ip2pnegprompt_txt)
    ip2pnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[ip2pnegprompt_txt], outputs=[ip2pnegprompt_txt])

    # Generation section
    ip2pgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[ip2pgen_seedval])

    ip2pgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(ip2pgen_LCM_images, inputs=[ip2pprompt_txt, ip2pnegprompt_txt, ip2pgen_guidance, ip2pgen_inference_steps, ip2pgen_seedval, ip2pgen_num_images, ip2pgen_incrementseed_check, ip2pgen_incrementseed_amount, ip2pimage, ip2pgen_resize_input_image_check, ip2pgen_imgguidance, ip2pgen_freeu_check, ip2pgen_freeu_s1, ip2pgen_freeu_s2, ip2pgen_freeu_b1, ip2pgen_freeu_b2, lcm_model_clipskip], outputs=[inference_status_markdown, oimage])
    
    ip2pgen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    ip2pgen_default_freeu_button.click(set_freeu_values, inputs=[ip2pgen_freeu_s1, ip2pgen_freeu_s2, ip2pgen_freeu_b1, ip2pgen_freeu_b2], outputs=[ip2pgen_freeu_s1, ip2pgen_freeu_s2, ip2pgen_freeu_b1, ip2pgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # SD UPSCALE 2X
 
    up2xprompt_paste_button.click(paste_model_prompt, None, up2xprompt_txt)
    up2xprompt_test_button.click(get_prompt_length_tokens, inputs=[up2xprompt_txt], outputs=[up2xprompt_txt])

    up2xnegprompt_paste_button.click(paste_model_prompt, None, up2xnegprompt_txt)
    up2xnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[up2xnegprompt_txt], outputs=[up2xnegprompt_txt])

    # Generation section
    up2xgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[up2xgen_seedval])

    up2xgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(upscale_image, inputs=[up2xprompt_txt, up2xnegprompt_txt, up2xgen_guidance, up2xgen_inference_steps, up2xgen_seedval, up2ximage, up2xgen_resize_input_image_check, up2xgen_freeu_check, up2xgen_freeu_s1, up2xgen_freeu_s2, up2xgen_freeu_b1, up2xgen_freeu_b2], outputs=[inference_status_markdown, oimage])

    up2xgen_default_freeu_button.click(set_freeu_values, inputs=[up2xgen_freeu_s1, up2xgen_freeu_s2, up2xgen_freeu_b1, up2xgen_freeu_b2], outputs=[up2xgen_freeu_s1, up2xgen_freeu_s2, up2xgen_freeu_b1, up2xgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------

    # # # CONTROLNET


    cnetprompt_paste_button.click(paste_model_prompt, None, cnetprompt_txt)
    cnetprompt_test_button.click(get_prompt_length_tokens, inputs=[cnetprompt_txt], outputs=[cnetprompt_txt])

    cnetnegprompt_paste_button.click(paste_model_prompt, None, cnetnegprompt_txt)
    cnetnegprompt_test_button.click(get_negprompt_length_tokens, inputs=[cnetnegprompt_txt], outputs=[cnetnegprompt_txt])

    # Generation section
    cnetgen_randomseed_button.click(gen_random_seed, inputs=[], outputs=[cnetgen_seedval])

    cnetgen_generate_button.click(clear_generation_status_and_images, None, outputs=[oimage, oimage2, inference_status_markdown, gallery_html]).then(change_tab, None, [tabs, inner_tab_ImageGeneration]).then(cnetgen_LCM_images, inputs=[cnetprompt_txt, cnetnegprompt_txt, cnetgen_width, cnetgen_height, cnetgen_guidance, cnetgen_guidance_start, cnetgen_guidance_end, cnetgen_conditioningguidance, cnetgen_conditioningguidance2, cnetgen_inference_steps, cnetgen_seedval, cnetgen_num_images, cnetgen_incrementseed_check, cnetgen_incrementseed_amount, cnetimage, cnetgen_resize_input_image, cnetimage2, cnetgen_resize_input_image2, cnetgen_freeu_check, cnetgen_freeu_s1, cnetgen_freeu_s2, cnetgen_freeu_b1, cnetgen_freeu_b2, lcm_model_clipskip, cnetgen_use_guess_mode], outputs=[inference_status_markdown, oimage])
    
    cnetgen_halt_gen_button.click(halt_generation, inputs=[], outputs=[])

    cnetgen_default_freeu_button.click(set_freeu_values, inputs=[cnetgen_freeu_s1, cnetgen_freeu_s2, cnetgen_freeu_b1, cnetgen_freeu_b2], outputs=[cnetgen_freeu_s1, cnetgen_freeu_s2, cnetgen_freeu_b1, cnetgen_freeu_b2])


# ------------------------------------------------------------------------------------------------------------------


    # Output Image section
    send_to_gallery_button.click(send_to_gallery, inputs=[], outputs=[gallery_html])

    
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
    save_lcm_model_save_button.click(save_lcm_model, inputs=[save_lcm_model_txt, save_lcm_model_lora_scale,save_lcm_model_as_safetensors_check, save_lcm_model_fp16_check], outputs=[save_lcm_model_html])


# ------------------------------------------------------------------------------------------------------------------

    
    # Add Lora Models section
    reload_lora_button.click(update_lora_model_list_dropdown, None, loradropdown)
    loaded_lora_list_refresh.click(update_loaded_lora_model_list_dropdown, None, loaded_loradropdown)
    lora_list_button.click(list_lora_model, None, lorahtml)
    lora_add_button.click(add_lora_model, inputs=[loradropdown, lora_scale_value], outputs=[lorahtml])
    lora_change_weight_button.click(change_lora_model, inputs=[loaded_loradropdown, lora_scale_value], outputs=[lorahtml])
    lora_delete_button.click(delete_all_lora_adapters, None, lorahtml)
    # loaded_loradropdown.change(select_loaded_lora_model, loaded_loradropdown, lorahtml)

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
    hug_downloadmodel_button.click(download_huggingface_model, hug_download_model_txt, hug_downloadmodel_html2) 

   
   
# ------------------------------------------------------------------------------------------------------------------

    
    # System Info section
    sysinfo_memory_button.click(get_sysinfo_memory, None, sysinfo_html) 
    sysinfo_hfcache_button.click(get_sysinfo_hfcache, None, sysinfo_html) 
    sysinfo_env_button.click(get_sysinfo_env, None, sysinfo_html) 
    sysinfo_sysinfo_button.click(get_sysinfo_sysinfo, None, sysinfo_html) 
   
    sysinfo_hug_on_button.click(huggingface_on_app, None, sysinfo_html) 
    sysinfo_hug_off_button.click(huggingface_off_app, None, sysinfo_html) 
    sysinfo_hug_check_button.click(huggingface_check_status_app, None, sysinfo_hug_status) 
    
    
    sysinfo_haltgen_button.click(halt_generation, inputs=[], outputs=[])
    sysinfo_exit_button.click(exit_app)
    sysinfo_restart_button.click(restart_app)
    
    sysinfo_reload_button.click(None, None, None, js="() => { window.location.reload(true); }")
    sysinfo_logout_button.click(None, None, None, js="() => { window.location.href = '/logout'; }")


# ------------------------------------------------------------------------------------------------------------------
 
 
    # # Settings section
    # # # saved parameters go in, output is simple status report to a box...
    settings_save_button.click(update_settings, inputs=gr_components, outputs=[settings_status_text, settings_status_text2])
    settings_save_button2.click(update_settings, inputs=gr_components, outputs=[settings_status_text, settings_status_text2])
    settings_goto_top_button.click(None, None, None, js="() => { window.scrollTo({top: 0}); }")

 
# ------------------------------------------------------------------------------------------------------------------

    
    # Help/About section
    # None - yet...


# ------------------------------------------------------------------------------------------------------------------

# Define launch keyword arguments in a dictionary
launch_kwargs = {}
launch_kwargs["share"] = False
launch_kwargs["server_name"] = STUDIO["server_name"]["value"]
launch_kwargs["server_port"] = int(STUDIO["server_port"]["value"])

if STUDIO["auth_use"]["value"]:
    launch_kwargs["auth"] = (STUDIO["auth_user"]["value"], STUDIO["auth_pass"]["value"])
    if len(STUDIO["auth_message"]["value"]) > 0:
        launch_kwargs["auth_message"] = f"<center><b><font size='+2'>Welcome to</font></b><br><br><img src='data:image/png;base64,{LLSTUDIO['llstudiologo_login']}' alt='{LLSTUDIO['app_title']}'></center><br><br>{STUDIO['auth_message']['value']}"


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




