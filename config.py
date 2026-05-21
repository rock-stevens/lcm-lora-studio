# ---------------------------------------------------
# ---------------LCM-LoRA Studio---------------------
# -----------------Version 1.5b----------------------
# ---------------------------------------------------
# libraries, imports for configuration settings ui
# Filename: config.py
# ---------------------------------------------------
# Copyright (C) 2025 Rock Stevens
# ---------------------------------------------------


import os 
import gradio as gr
import json

# ---------------------------------------------------

pipeline = None



# main app settings
# Some app settings are seperate from STUDIO because they are
# created from multiple STUDIO values. And use a LLSTUDIO["setting"] variable.
# This will aid in both future expansion and modularization of the application.
# Also because the 'settings' was/is seperate from the application anyway.
# See possible?? future Github repo 'auto-gradio-app-settings'. :)
# EX:
# lora_model_rootdir and lora_model_dir from STUDIO are combined
# using an os.join() to make an LLSTUDIO["lora_model_dir"] value
# which is the full (drive/root) path to folder where the LoRA models are.
# Also is OS independant 'path seperator' doing it this way :)
LLSTUDIO = {
    "app_title": "LCM-LoRA Studio",
    "app_version": "v1.5b",
    "app_author": "Rock Stevens",
    "app_url": "rockstevens.com/lcm-lora-studio",
    "app_github": "github.com/rock-stevens/lcm-lora-studio",
    "freeu_sd_s1": 0.9,     # defaults for SD15
    "freeu_sd_s2": 0.2,
    "freeu_sd_b1": 1.5,
    "freeu_sd_b2":1.6,
    "freeu_sdxl_s1": 0.6,   # defaults for SDXL
    "freeu_sdxl_s2": 0.4,
    "freeu_sdxl_b1": 1.1,
    "freeu_sdxl_b2": 1.2
    
}


# =============================================================================



# ###############################################
# #    GLOBAL DICTS for LCM-LoRA Studio  v1.5b  #
# ###############################################


# NOTE: This IS the JSON configuration file 'lcm-lora-studio.json'... basically.
# Therefore, if you edit this dict, you will need to delete the JSON file. Or you'll not see your edits from this dict.

STUDIO = {

    "settings_file": {
        "value": "lcm-lora-studio.json", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Default = 'lcm-lora-studio.json'<br>NOTE: The application user should leave this value alone. For the programmer type, it's here because of the way I wrote the settings section. Yep, Sorry one long page. BUT, You can <b>EASILY</b> add more 'settings' to LCM-LoRA Studio, look in the file 'config.py'. :)", 
        "label": "Configuration Filename",
        "visible": False
    },
    "setting_HTML0": {
        "value": "<b>Server Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "server_name": {
        "value": "0.0.0.0", 
        "type": "Textbox",
        "lines": 1, 
        "info": "IP Address Format (x.x.x.x)<br>Ex: '127.0.0.1' Local Host Only, '0.0.0.0' Local Network Only (Default)<br>The default is to run on local network only.<br>If you are ONLY going to be running it locally, you should change it to: '127.0.0.1'", 
        "label": "Server Name",
        "visible": True
    },
    "server_port": {
        "value": "7860", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Port to run Server on.<br>Ex: 7860", 
        "label": "Server Port",
        "visible": True
        },
    "setting_HTML1": {
        "value": "<b>Application Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "app_debug": {
        "value": 0, 
        "type": "Number", 
        "info": "0 = 'Least Amount' of command line output. 1 = Running 'App info' only. 2 = Running 'App info' + 'Diffusers' output", 
        "label": "Debug Level - (LIVE)",
        "minimum": 0, 
        "maximum": 5, 
        "step": 1
    },
    "app_autolaunch": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Auto Launch</font></u><br>If checked, LCM-LoRA Studio will automatically launch in your Default Web Browser on StartUp.<br>Do NOT use if running LCM-LoRA Studio on any personal computer (PC or PI5) remotely. It will launch a browser ON THE REMOTE machine and consume precious RAM memory you will need to run LCM-LoRA-Studio.", 
        "label": "Auto Launch"
    },
    "setting_HTML2": {
        "value": "<b>File and Directory Locations for Application, Models and Images</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "root_dir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Application Root Directory. Either '.' or 'full path' to the LCM-LoRA Studio Main Folder. Just leave it set to '.'", 
        "label": "Application Root Directory",
        "visible": True
    },
    "hub_online": {
        "value": True, 
        "type": "Checkbox", 
        "info": "<b><u>Huggingface Hub Online/Offline - (On Startup)</u></b><br>If checked, Huggingface Hub will be ONLINE on program start.<br>If NOT checked, Huggingface Hub will be OFFLINE on program start.", 
        "label": "Huggingface Hub Online/Offline - (On Startup)"
    },
    "hub_model_dir": {
        "value": "", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Full path to the locarion Hugginface Hub Cached models are located in.<br>NOTE: If this field is left blank, the default location for the OS, is used as the location. If you do use this field, please use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.<br>NOTE: As stated above, this setting has nothing to do with 'where' Huggingface lirbaries decides to put the 'Hub Cache' folder.<br>This setting is to allow LCM-LoRA-Studio to find it, that's all. And only, so you can load models from the 'Hub Cache' via a simple dropdown, rather than typing them in each time.<br>The normal full path is:<br>Windows: '<b>C:\\Users\\USERNAME\\.cache\\huggingface\\hub</b>' (the USERNAME is your username)<br>Linux (Pi5) for user 'pi' is: '<b>/home/pi/.cache/huggingface/hub</b>'",
        "label": "Hugginface Hub Cache - Directory",
        "visible": True
    },
    "lcm_model_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'LCM-LoRA Models - Directory'.", 
        "label": "LCM-LoRA Models - ROOT Directory",
        "visible": True
    },
    "lcm_model_dir": {
        "value": "lcmlora_models", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the models are located in, under the 'LCM-LoRA Models - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "LCM-LoRA Models - Directory",
        "visible": True
    },
    "lcm_model_prefix": {
        "value": "LCM_", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Prefix to apply in front of the name you choose for your converted LCM-LoRA Model. (Default: LCM_)", 
        "label": "LCM-LoRA Model Prefix",
        "visible": True
    },
    "lcm_model_suffix": {
        "value": "", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Suffix to apply to the end of the name you choose for your converted LCM-LoRA Model. (Default: NONE)", 
        "label": "LCM-LoRA Model Suffix",
        "visible": True
    },
    "lcm_model_image_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'LCM-LoRA Model Images - Directory'.", 
        "label": "LCM-LoRA Model Images - ROOT Directory",
        "visible": True
    },
    "lcm_model_image_dir": {
        "value": "lcmlora_models_images", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the model images are located in, under the 'LCM-LoRA Model Images - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.",  
        "label": "LCM-LoRA Model Images - Directory",
        "visible": True
    },
    "safe_model_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Safetensors Models - Directory'.", 
        "label": "Safetensors Models - ROOT Directory",
        "visible": True
    },
    "safe_model_dir": {
        "value": "safetensors_models", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the models are located in, under the 'Safetensors Models - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "Safetensors Models - Directory",
        "visible": True
    },
    "safe_model_image_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Safetensors Model Images - Directory'.", 
        "label": "Safetensors Model Images - ROOT Directory",
        "visible": True
    },
    "safe_model_image_dir": {
        "value": "safetensors_models", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the model images are located in, under the 'Safetensors Model Images - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "Safetensors Model Images - Directory",
        "visible": True
    },
    "lora_model_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'LoRA Models - Directory'.", 
        "label": "LoRA Models - ROOT Directory",
        "visible": True
    },
    "lora_model_dir": {
        "value": "lora_models", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the models are located in, under the 'LoRA Models - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "LoRA Models - Directory",
        "visible": True
    },
    "lora_model_image_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'LoRA Model Images - Directory'.", 
        "label": "LoRA Model Images - ROOT Directory",
        "visible": True
    },
    "lora_model_image_dir": {
        "value": "lora_models_images", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the model images are located in, under the 'LoRA Model Images - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "LoRA Model Images - Directory",
        "visible": True
    },
    "output_image_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Output Images - Directory'.", 
        "label": "Outputs Images - ROOT Directory",
        "visible": True
    },
    "output_image_dir": {
        "value": "output", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the output images are located in, under the 'Output Images - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "Output Images - Directory",
        "visible": True
    },
    "advanced_gallery_root": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Advanced Image Gallery - Directory'.", 
        "label": "Advanced Image Gallery - ROOT Directory",
        "visible": True
    },
    "advanced_gallery_dir": {
        "value": "output", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the Advanced Image Gallery is located, under the 'Advanced Image Gallery - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "Advanced Image Gallery - Directory",
        "visible": True
    },
    "imgp_files_root": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Image Processing Files - Directory'.", 
        "label": "Image Processing Files - ROOT Directory",
        "visible": True
    },
    "imgp_files_dir": {
        "value": "imgp_files", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the Image Processing Files are located, under the 'Image Processing Files - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "Image Processing Files - Directory",
        "visible": True
    },
    "setting_HTML3": {
        "value": "<b>Prompt Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "def_prompt": {
        "value": "a creek running down a hill and through a field, surrounded by lush plants and tall trees, with the trees slightly blocking a sunny day, photo realistic, hyperdetailed", 
        "type": "Textbox",
        "lines": 3, 
        "info": "Default prompt that will to be displayed in the prompt textbox.", 
        "label": "Default Prompt",
        "visible": True
    },
    "def_negprompt": {
        "value": "blurry, out of focus, poorly drawn, anime, cartoon, low resolution, bad anatomy, bad fingers, bad eyes, bad arms", 
        "type": "Textbox",
        "lines": 3, 
        "info": "Default prompt that will to be displayed in the negative prompt textbox.", 
        "label": "Default Negative Prompt",
        "visible": True
    },
    "use_prompt_embeds": {
        "value": 0, 
        "type": "Number", 
        "info": "Use Prompt Embeddings or Normal Prompts.<br>NOTE: Does not work with StableDiffusionLatentUpscalePipeline. It will automatically fall back to Normal Prompts for StableDiffusionLatentUpscalePipeline. So, watch your prompt size for that pipeline type.<br>0=Normal Prompts (76 Max Prompt Tokens)<br>1=Prompt Embeddings and Padding<br>2=Prompt Weighting (Compel) and Prompt Embeddings<br>3=Prompt Weighting (Compel) and Prompt Embeddings and Padding", 
        "label": "Use Prompt Embeddings - (LIVE)",
        "minimum": 0, 
        "maximum": 3, 
        "step": 1
    },
    "setting_HTML4": {
        "value": "<b>Output Image Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "output_image_prefix": {
        "value": "LCMLORA_", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Prefix applied to output image filename. (Optional)", 
        "label": "Output Image Prefix - (LIVE)",
        "visible": True
    },
    "output_image_suffix": {
        "value": "", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Suffix applied to output image filename. (Optional)", 
        "label": "Output Image Suffix - (LIVE)",
        "visible": True
    },
    "output_image_datetime": {
        "value": "%y%m%d%H%M%S_%f", 
        "type": "Textbox",
        "lines": 1, 
        "info": "The date and time added to the end of the output image filename, before the 'Output Image Suffix' is added from the above setting. (Optional)<br>Date and Time Format default is: %y%m%d%H%M%S_%f<br>Example: (Jan 27, 2012 02:17:53PM and 036373uS) = 120127141753_036373", 
        "label": "Output Image Date and Time Suffix - (LIVE)",
        "visible": True
    },
    "setting_HTML5": {
        "value": "<b>Image and Image Gallery Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "gen_auto_image_tab": {
        "value": True, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Auto Select Output Image Tab</font></u><br>If checked, LCM-LoRA Studio automatically switches to the Output Image Tab as soon image generation begins.", 
        "label": "Auto Select Output Image Tab - (LIVE)"
    },
    "img_view_img_per_page": {
        "value": 5, 
        "type": "Slider", 
        "info": "Number of Images to show per page in Image viewers.", 
        "label": "Gallery Images per Page - (LIVE)",
        "minimum": 5, 
        "maximum": 50, 
        "step": 1
    },
    "img_view_img_width": {
        "value": 75, 
        "type": "Slider", 
        "info": "Controls Image size in the Image viewers.<br>NOTE: You can control the width (percentage) of the images, so the gallery can be laid out to where you can see the image generation paramters good. Adjust if needed.", 
        "label": "Gallery Images Size (Width) - (LIVE)",
        "minimum": 10, 
        "maximum": 100, 
        "step": 1
    },
    "setting_HTML7": {
        "value": "<b>Model Loading Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "default_clip_skip": {
        "value": 0, 
        "type": "Number", 
        "info": "Default Clip Skip Value.<br>Default=0 No Clip Skip, 1=Clip Skip1, 2=Clip Skip2, etc...<br>This is just a default for the user interface, and can be changed when an LCM model is loaded so ClipSkip will work.<br>NOTE: Does not work with SDXL type models, nor StableDiffusionLatentUpscalePipeline.", 
        "label": "Default Clip Skip Value",
        "minimum": 0, 
        "maximum": 11, 
        "step": 1
    },
    "use_safety_checker": {
        "value": True, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use Safety Checker</font></u><br>If checked, it estimates whether generated images could be considered offensive or harmful is enabled and will NOT generate images if offensive or harmful content is found in the generated images.<br>Not using the Safety Checker will save memory, because it will not load the model used for the Safety Chcker. NOTE: On by default.<br>IMPORTANT: Does NOT WORK on models where the safety checker has been disabled or removed.<br>Please refer to the model's 'model card' for more details about a model’s potential harms.<br>NOTE: Although this application was designed to expose many parameters to adjust by the user, (including this one), the best advise is to leave it checked.<br>VERY IMPORTANT: Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public.", 
        "label": "Use Safety Checker"
    },
    "local_files_only": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use Local FIles ONLY</font></u><br>If checked, load ONLY, LOCAL model weights and configuration files. If checked, the model won’t be downloaded from the Hub. Also if Huggingface Hub is ONLINE per enviroment variable 'HF_HUB_OFFLINE=0', and your internet is off, you may need this checked. Tells the Diffusers library to nevermind your internet connection. Which is what you want for full offline operation. NOTE: Needs to be UNCHECKED to get anythng from Huggingface Hub.", 
        "label": "Use Local FIles ONLY - LIVE (RELOAD MODEL)"
    },
    "setting_HTML8": {
        "value": "<b>SD Upscaler 2X Model Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "sdupscale2x_model_name": {
        "value": "stabilityai/sd-x2-latent-upscaler", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Model Name used to load into the StableDiffusionLatentUpscalePipeline<br>Default model name: 'stabilityai/sd-x2-latent-upscaler'<br>NOTE: If not already in your Hugginface Hub Cache, it will be downloaed upon first use of the Upscaler.<br>NOTE2: Can be replaced by another diffusers 'compatible' Huggingface model, if availiable. So, if you find one at Huggingface that will work, just change the name to the new model.", 
        "label": "StableDiffusionLatentUpscalePipeline Model Name - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "setting_HTML9": {
        "value": "<b>Safetensors Model Loading/Converting 'Original Config File' Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
     "safe_use_original_config_file": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use original config file</font></u><br>If checked, when loading Safetensors models, this will use the original config file (a YAML file describing the model's architecture) that was used to train the model. There is a section right below to name one for each pipeline type. (pre-filled out with the default for that type.) Some models will not need this checked, some will. So if you have problems with certain Safetensors model loading, check this. As well as check the (Safetesnsors models - Use 'reference' 'base-model'), may need that too. Although there is none for SDXL, using the 'Config Reference model' takes care of it, if any loading problems. This is mainly for loading, Sadetensors type of SD models. And is also model dependant. Some need it, some do not.", 
        "label": "Use original config file when loading (see list below) - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SD_original_config": {
        "value": "v1-inference.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXL_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDImage2Image_original_config": {
        "value": "v1-inference.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD Image2Image Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXLImage2Image_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL Image2Image Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDInpaint_original_config": {
        "value": "v1-inpainting-inference.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD Inpaint Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXLInpaint_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL Inpaint Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDInstructPix2Pix_original_config": {
        "value": "instruct-pix2pix.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD InstructPix2Pix Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXLInstructPix2Pix_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL InstructPix2Pix Original Config File - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "setting_HTML10": {
        "value": "<b>Safetensors Model Loading/Converting 'Reference' Base-Model Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "safe_use_config": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use a 'reference' 'base-model'</font></u><br>If checked, when loading Safetensors models, this will use a 'reference' 'base-model' as a reference for loading the Safetensors model. There is a section right below to name one for each pipeline type. (pre-filled out with the default reference model name for that type.) Some models will not need this checked, some will. So if you have problems with certain Safetensors model loading, check this. As well as check the ('Safetesnsors models - Use original config file'), may need that too. A string, the repo id (for example CompVis/ldm-text2im-large-256) of a pretrained pipeline hosted on the Hub. -or- A path to a directory (for example ./my_pipeline_directory/) containing the pipeline component configs in Diffusers format.", 
        "label": "Use a 'reference' 'base-model' when loading (see list below) - LIVE (RELOAD MODEL)"
    },
    "SD_config": {
        "value": "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXL_config": {
        "value": "stabilityai/stable-diffusion-xl-base-1.0", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDImage2Image_config": {
        "value": "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD Image2Image model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD Image2Image Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXLImage2Image_config": {
        "value": "stabilityai/stable-diffusion-xl-base-1.0", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL Image2Image model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL Image2Image Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDInpaint_config": {
        "value": "stable-diffusion-v1-5/stable-diffusion-inpainting", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD Inpaint model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD Inpaint Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXLInpaint_config": {
        "value": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL Inpaint model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL Inpaint Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDInstructPix2Pix_config": {
        "value": "timbrooks/instruct-pix2pix", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD InstructPix2Pix model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD InstructPix2Pix Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "SDXLInstructPix2Pix_config": {
        "value": "diffusers/sdxl-instructpix2pix-768", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL InstructPix2Pix model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL InstructPix2Pix Config Reference model - LIVE (RELOAD MODEL)",
        "visible": True
    },
    "setting_HTML11": {
        "value": "<b>Memory Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "low_memory": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use Low Memory Settings for Model Loading</font></u><br>If checked, configures models during loading to use less memory.<br>NOTE: Using less memory images will generate a little slower. (Only use if needed.)", 
        "label": "Use Low Memory Settings for Model Loading - LIVE (RELOAD MODEL)"
    },
    "low_memory_inf": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use Low Memory Settings for Running an Inference</font></u> (LCM-LoRA Models Only)<br>If checked, configures models during running an inference to use less memory.<br>NOTE: Using less memory images will generate a little slower. (Only use if needed.)", 
        "label": "Use Low Memory Settings for Running Inference - LIVE (RELOAD MODEL)"
    }
}





# =============================================================================


def load_settings():
    if os.path.exists(os.path.join(".", STUDIO["settings_file"]["value"])):
        with open(os.path.join(".", STUDIO["settings_file"]["value"]), "r") as f:
            return json.load(f)
    else:
        save_settings()
        return STUDIO 


# =============================================================================


def save_settings():
    with open(os.path.join(".", STUDIO["settings_file"]["value"]), "w") as f:
        json.dump(STUDIO, f, indent=4)



# =============================================================================


def update_settings(*args):
    # Get the keys in the correct order (retains insertion order in Python 3.7+)
    keys = list(STUDIO.keys())
    for key, arg in zip(keys, args):
        # Update the 'value' for the key in the dictionary.
        # The only key we use in the inner dict is 'value'. 
        # 'if' statement blocks updating 'Label' and 'HTML' compnoents which are display info only.
        # ie... read from dict, but write back into 'value', they are saved in the JSON file
        if (STUDIO[key]["type"] != "HTML" and STUDIO[key]["type"] != "Label"):
            STUDIO[key]["value"] = arg  # store value
            
    # save the settings
    save_settings()
    
    # check for safety checker, if not, give warning on return
    # we return two values, one for top of the settings ui, one for the bottom of the settings ui
    if STUDIO["use_safety_checker"]["value"] == True:
        return f"Settings saved successfully! Safety Checker is ON.", f"Settings saved successfully! Safety Checker is ON."
    else:
        return safety_checker_warning, safety_checker_warning
    

# =============================================================================


# automatically create the Gradio UI for the settings tab, from the STUDIO dict :)
def create_settings_ui():

    auto_components = []

    for setting_name, item in STUDIO.items():
        if STUDIO[setting_name]['type'] == "Textbox":
            settings_components = gr.Textbox(
            label=STUDIO[setting_name]['label'], 
            value=STUDIO[setting_name]['value'], 
            lines=int(STUDIO[setting_name]['lines']),
            info=STUDIO[setting_name]['info'],
            visible=STUDIO[setting_name]['visible']
            )
            auto_components.append(settings_components)
            
        if STUDIO[setting_name]['type'] == "Number":
            settings_components = gr.Number(
            label=STUDIO[setting_name]['label'], 
            value=STUDIO[setting_name]['value'], 
            info=STUDIO[setting_name]['info'],
            minimum=STUDIO[setting_name]['minimum'],
            maximum=STUDIO[setting_name]['maximum'],
            step=STUDIO[setting_name]['step']
            )
            auto_components.append(settings_components)
            
        if STUDIO[setting_name]['type'] == "Checkbox":
            settings_components = gr.Checkbox(
            label=STUDIO[setting_name]['label'], 
            value=STUDIO[setting_name]['value'], 
            info=STUDIO[setting_name]['info']
            )
            auto_components.append(settings_components)
            
        if STUDIO[setting_name]['type'] == "Slider":
            settings_components = gr.Slider(
            label=STUDIO[setting_name]['label'], 
            value=STUDIO[setting_name]['value'],
            info=STUDIO[setting_name]['info'],
            minimum=STUDIO[setting_name]['minimum'],
            maximum=STUDIO[setting_name]['maximum'],
            step=STUDIO[setting_name]['step']
            )
            auto_components.append(settings_components)
            
        if STUDIO[setting_name]['type'] == "Label":
            settings_components = gr.Label(
            label=STUDIO[setting_name]['label'], 
            show_label=STUDIO[setting_name]['show_label'], 
            value=STUDIO[setting_name]['value']
            )
            auto_components.append(settings_components)
            
        if STUDIO[setting_name]['type'] == "HTML":
            settings_components = gr.HTML(
            label=STUDIO[setting_name]['label'], 
            value=STUDIO[setting_name]['value']
            )
            auto_components.append(settings_components)

    return auto_components




# =============================================================================


# obvious safety checker is OFF warning.
safety_checker_warning = f"""Settings saved successfully!
WARNING:
You have disabled the safety checker. 
Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public.
Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results.
For more information, please have a look at https://github.com/huggingface/diffusers/pull/254"""


# =============================================================================


# simple StableDiffusion pipeline class to model type, generation/pipeline mode - lookup table
PIPECLASSES = {
    'StableDiffusionPipeline': {'pipeline_model_type': 'SD15', 'pipeline_gen_mode': 'Text to Image', 'pipeline_model_mode': 't2i'},
    'StableDiffusionXLPipeline': {'pipeline_model_type': 'SDXL', 'pipeline_gen_mode': 'Text to Image', 'pipeline_model_mode': 't2i'},
    'StableDiffusionImage2Image': {'pipeline_model_type': 'SD15', 'pipeline_gen_mode': 'Image to Image', 'pipeline_model_mode': 'i2i'},
    'StableDiffusionXLImage2Image': {'pipeline_model_type': 'SDXL', 'pipeline_gen_mode': 'Image to Image', 'pipeline_model_mode': 'i2i'},
    'StableDiffusionInpaintPipeline': {'pipeline_model_type': 'SD15', 'pipeline_gen_mode': 'Inpainting', 'pipeline_model_mode': 'inp'},
    'StableDiffusionXLInpaintPipeline': {'pipeline_model_type': 'SDXL', 'pipeline_gen_mode': 'Inpainting', 'pipeline_model_mode': 'inp'},
    'StableDiffusionInstructPix2PixPipeline': {'pipeline_model_type': 'SD15', 'pipeline_gen_mode': 'Instruct Pix2Pix', 'pipeline_model_mode': 'ip2p'},
    'StableDiffusionXLInstructPix2PixPipeline': {'pipeline_model_type': 'SDXL', 'pipeline_gen_mode': 'Instruct Pix2Pix', 'pipeline_model_mode': 'ip2p'},
    'StableDiffusionLatentUpscalePipeline': {'pipeline_model_type': 'SD15', 'pipeline_gen_mode': '2x UpScaler', 'pipeline_model_mode': 'up2x'},
    'StableDiffusionControlNetPipeline': {'pipeline_model_type': 'SD15', 'pipeline_gen_mode': 'ControlNet', 'pipeline_model_mode': 'cnet'}
}



# =============================================================================


# class list for pipeline class dropdown boxes
PIPELINE_CLASSES = ["StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusionImage2Image", "StableDiffusionXLImage2Image", "StableDiffusionInpaintPipeline", "StableDiffusionXLInpaintPipeline", "StableDiffusionInstructPix2PixPipeline", "StableDiffusionXLInstructPix2PixPipeline"]



# =============================================================================

# LIVE pipeline info lookup table - can be read while running inference, but not changed
# data populated/updated after loading a model, or deleting model from pipline.
# used for mainly information for user interaction, but one thing it does
# is keep a note of which type of pipeline is loaded and used, which controls parts of the app
SDPIPELINE = {
    "pipeline_loaded": 0,                           # model loaded ? 0=no/1=yes, used to trigger an error/alert on No model loaded
    "pipeline_class": "StableDiffusionPipeline",    # StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImage2Image... default startup value=StableDiffusionPipeline
    "pipeline_source": "",                          # 'LCMLORA', 'HUB Cached', 'Huggingface', 'Safetensors' basically where model was loaded from, if LCMLORA, already has LCM LoRA added/fused
    "pipeline_model_name": "",                      # name of model as in dropdowns
    "pipeline_gen_mode": "Text to Image",           # Text 2 Image, Image 2 Image, Inpainting, Instruct Pix2Pix, UpScaler default startup value=Text 2 Image
    "pipeline_model_type": "SD15",                  # SD15 or SDXL default=SD15
    "pipeline_text_encoder": 0,                     # use seperate text encoder ? 0=no/1=yes
    "pipeline_text_encoder_name": "",               # name of model of seperate text encoder as in dropdowns
    "pipeline_model_precision": "fp16",             # basically, fp16 or fp32 (default for LCM is fp16 so it'll run it's 4 step lcm-lora)
    "pipeline_controlnet_loaded": 0,                # load a controlnet ? 0=no/1=yes
    "pipeline_controlnet_name": "",                 # name of control net1
    "pipeline_controlnet_name2": ""                 # name of control net2
    
}




# =============================================================================
# =============================================================================


# -EOF-



