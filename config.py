# config.py
# base libraries, imports for settings ui
# ---------------------------------------------------
import os 
import gradio as gr
import json
# ---------------------------------------------------

# ###############################################
# #    GLOBAL DICTS for LCM-LoRA Studio  v1.3a  #
# ###############################################

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
    "setting_HTML0b": {
        "value": "<b>Authentication</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "auth_use": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use authentication.</font></u><br>If checked, a Login Screen will appear for the user to Login.", 
        "label": "Use Authentication."
    },
    "auth_message": {
        "value": "Please use the credentials previously provided.", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Custom message to display on login screen to inform user.<br>Default: 'Please use the credentials previously provided.'<br>Leave 'blank' for No Custom Login Message.", 
        "label": "Login Screen Message",
        "visible": True
    },
    "auth_user": {
        "value": "username", 
        "type": "Textbox",
        "lines": 1, 
        "info": "User's Username<br>Default: username", 
        "label": "Username",
        "visible": True
    },
    "auth_pass": {
        "value": "password", 
        "type": "Textbox",
        "lines": 1, 
        "info": "User's Password<br>Default: password", 
        "label": "Password",
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
        "info": "<u><font size='+1'>Auto Launch</font></u><br>If checked, LCM-LoRA Studio will automatically launch in your Default Web Browser on StartUp.<br>Do NOT use if running LCM-LoRA Studio on any personal computer (PC or PI5) remotely. It will launch a browser and consume precious RAM memory you will need to run LCM-LoRA-Studio.", 
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
    "hub_model_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Hugginface Hub Cache - Directory'.<br>NOTE: This setting has nothing to do with 'where' Huggingface lirbaries decides to put the 'Hub Cache' folder.<br>.This setting is to allow LCM-LoRA-Studio to find it, that's all. And only, so you can load models from the 'Hub Cache'.<br>If it can NOT be found the app will let you know. I do it same as Huggingface library, check 'HF_HUB_CACHE' enviroment variable, then 'HF_HOME', and then the default directory for the current user. However, the normal full path on Windows is: 'C:\\Users\\%USERNAME%\\.cache\\huggingface\\hub' (the %USERNAME% is your username), on Linux (Pi5) for user 'pi' is: '/home/pi/.cache/huggingface/hub'", 
        "label": "Hugginface Hub Cache - ROOT Directory",
        "visible": True
    },
    "hub_model_dir": {
        "value": "", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the Hugginface Hub Cached models are located in, under the 'Hugginface Hub Cache - ROOT Directory'<br>NOTE: If this field is left blank, the enviroment variable 'HF_HUB_CACHE', 'HF_HOME' or the default location for the OS, is used as the location. If you do use this field, please use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.<br>Also NOTE: This setting has nothing to do with 'where' Huggingface lirbaries decides to put the 'Hub Cache' folder.<br>.This setting is to allow LCM-LoRA-Studio to find it, that's all. And only, so you can load models from the 'Hub Cache'.<br>The normal full path is:<br>Windows: 'C:\\Users\\%USERNAME%\\.cache\\huggingface\\hub' (the %USERNAME% is your username)<br>Linux (Pi5) for user 'pi' is: '/home/pi/.cache/huggingface/hub'",
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
        "value": "LCMLORA_", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Prefix to apply in front of the name you choose for your converted LCM-LoRA Model. (Default: LCMLORA_)", 
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
    "outputfolder_rootdir": {
        "value": ".", 
        "type": "Textbox",
        "lines": 1, 
        "info": "(.) = Root folder, On another drive ex: (D:&#92;) On Linux, leave blank, and use 'full path' with leading slash in the box for the 'Generation Output Folder - Directory'.", 
        "label": "Generation Output Folder - ROOT Directory",
        "visible": True
    },
    "outputfolder": {
        "value": "output", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Directory the generation images are to be stored in, under the 'Generation Output Folder - ROOT Directory. Use the appropriate path seperator, for Windows, use '&#92;' and for Linux use a '/'.", 
        "label": "Generation Output Folder - Directory",
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
        "info": "Controls Image size in the Image viewers.<br>NOTE: You can control the widths of the images so the gallery can be laid out to where you can see the image generation paramters good. And adjust if needed.", 
        "label": "Gallery Images Size (Width) - (LIVE)",
        "minimum": 10, 
        "maximum": 100, 
        "step": 1
    },
    "setting_HTML6": {
        "value": "<b>Model Device Settings</b> - <a href='#' onclick='window.scrollTo(0, 0);'>Go to top</a>",
        "type": "HTML",
        "label": ""
    },
    "device_name": {
        "value": "cpu", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Enter a number 0-x, or 'cpu'. If you looked ahead... and installed the CUDA driver rather than the CPU driver (LCM-LoRA Studio Default is CPU), and a CUDA capable device is found, you can select which device if there is more than one in the system.<br>This index starts with '0'.<br>So if you want the second GPU card to be used, enter a '1' in this box.<br>NOTE: This is ONLY for GPU, NOT CPU.<br>However, If you enter 'cpu', this value bypasses the check for CUDA and defaults to CPU. The code is there to do it, if you want to try GPU, replace the CPU drivers in the 'lcm-lora-studio-requirement.txt' file with the GPU drivers and re-install LCM-LoRA Studio. You WILL have to find out how on your own, search the internet. I have nothing to test it with anyway, if I did, I probably would have never written this app...", 
        "label": "Device Name",
        "visible": True
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
        "info": "<u><font size='+1'>Use Safety Checker</font></u><br>If checked, a 'classification module' that estimates whether generated images could be considered offensive or harmful is enabled and will NOT generate images if offensive or harmful content is found in the generated images.<br>Not using the Safety Checker will save memory, because it will not load the model used for the Safety Chcker on application start up. You WILL need to restart the application after 'unchecking' the box to regain the memory used by the Safety Chcker model.<br>IMPORTANT: Please refer to the model's 'model card' for more details about a model’s potential harms.<br>NOTE: Although this application was designed to expose many parameters to adjust by the user, (including this one), the best advise is to leave it checked.<br>VERY IMPORTANT: Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public.", 
        "label": "Use Safety Checker"
    },
    "safety_checker_model_name": {
        "value": "Falconsai/nsfw_image_detection", 
        "type": "Textbox",
        "lines": 1, 
        "info": "Image Classifier Model Name used as the Safety Checker<br>Default model name: 'Falconsai/nsfw_image_detection'<br>NOTE: If not already in your Hugginface Hub Cache, it will be downloaed upon first use of the Safety Checker.<br>NOTE2: Can be replaced by another diffusers 'compatible' Huggingface model, if availiable. If you find one at Huggingface just change the name to the new model.", 
        "label": "Safety Checker Model",
        "visible": True
    },
    "local_files_only": {
        "value": False, 
        "type": "Checkbox", 
        "info": "<u><font size='+1'>Use Local FIles ONLY</font></u><br>If checked, load ONLY, LOCAL model weights and configuration files or not. If checked, the model won’t be downloaded from the Hub. Also if Huggingface Hub is ONLINE per enviroment variable 'HF_HUB_OFFLINE=0', and your internet is off, you may need this checked. Tells the Diffusers library to nevermind your internet connection. Which is what you want for full offline operation. NOTE: Needs to be UNCHECKED to get anythng from Huggingface Hub.", 
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
        "info": "Model Name used to load into the StableDiffusionLatentUpscalePipeline<br>Default model name: 'stabilityai/sd-x2-latent-upscaler'<br>NOTE: If not already in your Hugginface Hub Cache, it will be downloaed upon first use of the Upscaler.<br>NOTE2: Can be replaced by another diffusers 'compatible' Huggingface model, if availiable. If you find one at Huggingface just change the name to the new model.", 
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
        "info": "<u><font size='+1'>Use original config file</font></u><br>If checked, when loading Safetensors models, this will use the original config file (a YAML file describing the model's architecture) that was used to train the model. There is a section right below to name one for each pipeline type. (pre-filled out with the default for that type.) Some models will not need this checked, some will. So if you have problems with certain Safetensors model loading, check this. As well as check the (Safetesnsors models - Use 'reference' 'base-model'), may need that too. Although there is none for SDXL, using the 'Config Reference model' takes care of it, if any loading problems. This is mainly for SD models.", 
        "label": "Use original config file when loading (see list below) - LIVE",
        "visible": True
    },
    "SD_original_config": {
        "value": "v1-inference.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD Original Config File - LIVE",
        "visible": True
    },
    "SDXL_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL Original Config File - LIVE",
        "visible": True
    },
    "SDImage2Image_original_config": {
        "value": "v1-inference.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD Image2Image Original Config File - LIVE",
        "visible": True
    },
    "SDXLImage2Image_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL Image2Image Original Config File - LIVE",
        "visible": True
    },
    "SDInpaint_original_config": {
        "value": "v1-inpainting-inference.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD Inpaint Original Config File - LIVE",
        "visible": True
    },
    "SDXLInpaint_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL Inpaint Original Config File - LIVE",
        "visible": True
    },
    "SDInstructPix2Pix_original_config": {
        "value": "instruct-pix2pix.yaml", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "The name of the original config file that was used to train the model. If not provided, the config file will be inferred from the checkpoint file.", 
        "label": "SD InstructPix2Pix Original Config File - LIVE",
        "visible": True
    },
    "SDXLInstructPix2Pix_original_config": {
        "value": "", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "no original_config file for SDXL, supposedly... Leave Blank, but it's here if needed in future. Using the 'Config Reference model' takes care of it, if any loading problems.", 
        "label": "SDXL InstructPix2Pix Original Config File - LIVE",
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
        "label": "Use a 'reference' 'base-model' when loading (see list below) - LIVE"
    },
    "SD_config": {
        "value": "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD Config Reference model - LIVE",
        "visible": True
    },
    "SDXL_config": {
        "value": "stabilityai/stable-diffusion-xl-base-1.0", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL Config Reference model - LIVE",
        "visible": True
    },
    "SDImage2Image_config": {
        "value": "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD Image2Image model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD Image2Image Config Reference model - LIVE",
        "visible": True
    },
    "SDXLImage2Image_config": {
        "value": "stabilityai/stable-diffusion-xl-base-1.0", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL Image2Image model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL Image2Image Config Reference model - LIVE",
        "visible": True
    },
    "SDInpaint_config": {
        "value": "stable-diffusion-v1-5/stable-diffusion-inpainting", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD Inpaint model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD Inpaint Config Reference model - LIVE",
        "visible": True
    },
    "SDXLInpaint_config": {
        "value": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL Inpaint model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL Inpaint Config Reference model - LIVE",
        "visible": True
    },
    "SDInstructPix2Pix_config": {
        "value": "timbrooks/instruct-pix2pix", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SD InstructPix2Pix model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SD InstructPix2Pix Config Reference model - LIVE",
        "visible": True
    },
    "SDXLInstructPix2Pix_config": {
        "value": "diffusers/sdxl-instructpix2pix-768", 
        "type": "Textbox", 
        "lines": 1, 
        "info": "'Reference model' to use (if needed) when loading a 'Safetensors' SDXL InstructPix2Pix model.<br>NOTE: If the 'reference' model is not already in your Hugginface Hub Cache, it will be downloaed upon first use. Only if enabled above, to be used.", 
        "label": "SDXL InstructPix2Pix Config Reference model - LIVE",
        "visible": True
    },
}





# =============================================================================


def load_settings():
    #rkconvert - NOT DONE
    if os.path.exists(os.path.join(".", STUDIO["settings_file"]["value"])):
        with open(os.path.join(".", STUDIO["settings_file"]["value"]), "r") as f:
            return json.load(f)
    else:
        save_settings()
        return STUDIO 


# =============================================================================


def save_settings():
    #rkconvert - NOT DONE
    with open(os.path.join(".", STUDIO["settings_file"]["value"]), "w") as f:
        json.dump(STUDIO, f, indent=4)


# =============================================================================



def print_settings():
    #rkconvert - NOT DONE
    print("-" * 30)
    for setting_name, details in STUDIO.items():
        print(f"Setting: {setting_name}")
        print(f"  value: {STUDIO[setting_name]['value']}")
        print(f"  label: {STUDIO[setting_name]['label']}")
        print("-" * 30)


# =============================================================================


 

def update_settings(*args):
    #rkconvert - NOT DONE
    # Get the keys in the correct order (retains insertion order in Python 3.7+)
    keys = list(STUDIO.keys())
    # Use zip() to loop through both the keys and the args simultaneously
    for key, arg in zip(keys, args):
        # Update the 'value' for the key in the dictionary.
        # The only key we use in the inner dict is 'value'. 
        # 'if' statement blocks updating 'Label' and 'HTML' compnoents which are display info only.
        # ie... read from dict, but write back into 'value', but are saved in the JSON file
        if (STUDIO[key]["type"] != "HTML" and STUDIO[key]["type"] != "Label"):
            STUDIO[key]["value"] = arg  # store value
            
    # save the settings
    save_settings()
    
    # check for safety checker, if not, give warning on return
    if STUDIO["use_safety_checker"]["value"] == True:
        return f"Settings saved successfully! Safety Checker is ON.", f"Settings saved successfully! Safety Checker is ON."
    else:
        return safety_checker_warning, safety_checker_warning
    

# =============================================================================



def create_settings_ui():

    auto_components = []
    x_components = []
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



safety_checker_warning = f"""Settings saved successfully!
WARNING:
You have disabled the safety checker. 
Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public.
Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results.
For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."""


# =============================================================================



# main app settings
# these settings are seperate from STUDIO because they are
# created from multiple STUDIO values. 
# EX:
# lora_model_rootdir and lora_model_dir from STUDIO are combined
# using an os.join() to make an LLSTUDIO["lora_model_dir"] value
# which is the full (drive/root) path to folder where the LoRA models are.
LLSTUDIO = {
    "app_title": "LCM-LoRA Studio",
    "app_version": "v1.3a",
    "app_author": "Rock Stevens",
    "app_url": "rockstevens.com/lcm-lora-studio",
    "app_github": "github.com/rock-stevens/lcm-lora-studio",
    "freeu_sd_s1": 0.9,     # defaults fo SD15
    "freeu_sd_s2": 0.2,
    "freeu_sd_b1": 1.5,
    "freeu_sd_b2":1.6,
    "freeu_sdxl_s1": 0.6,   # defaults for SDXL
    "freeu_sdxl_s2": 0.4,
    "freeu_sdxl_b1": 1.1,
    "freeu_sdxl_b2": 1.2
    
}


# =============================================================================



# simple sd pipeline class to model type, generation/pipeline mode - lookup table
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
# used for mainly informational for user interaction, but one thing it does
# is keep a note of which type of pipeline is loaded and used, which controls parts of the app
SDPIPELINE = {
    "pipeline_loaded": 0,                           # model loaded ? 0=no/1=yes, trigger error/alert on No model loaded
    "pipeline_class": "StableDiffusionPipeline",    # StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImage2Image... default startup value=StableDiffusionPipeline
    "pipeline_source": "",                          # 'LCMLORA', 'HUB Cached', 'Huggingface', 'Safetensors' basically where model was loaded from, if LCMLORA, already has LCM LoRA added/fused
    "pipeline_model_name": "",                      # name of model as in dropdowns
    "pipeline_gen_mode": "Text to Image",            # Text 2 Image, Image 2 Image, Inpainting, Instruct Pix2Pix, UpScaler default startup value=Text 2 Image
    "pipeline_model_type": "SD15",                  # SD15 or SDXL default=SD15
    "pipeline_text_encoder": 0,                     # use seperate text encoder ? 0=no/1=yes
    "pipeline_text_encoder_name": "",                # name of model of seperate text encoder as in dropdowns
    "pipeline_model_precision": "fp16",              # basically, fp16 or fp32 (default LCM to fp16 so it'll run it's 4 step lcm-lora)
    "pipeline_controlnet_loaded": 0,                      # load a controlnet ? 0=no/1=yes
    "pipeline_controlnet_name": "",                      # name of control net
    "pipeline_controlnet_name2": "",                      # name of control net
    "pipeline_safety_checker_loaded": 0                      # loaded a safety_checker model ? 0=no/1=yes
    
}




# =============================================================================
# =============================================================================


# -EOF-



