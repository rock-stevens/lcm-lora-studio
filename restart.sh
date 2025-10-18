#!/bin/bash

# Version 1.3 - Adds Huggingface Hub On/Off Control
# RESTART_FILE Legend:
# 0=exit
# 1=just restart python and app
# 2=turn on huggingface Hub
# 3=turn off huggingface Hub


echo Starting LCM-LoRA Studio LOOP...

# exit immediately if any command fails
# set -e

# we set our default python command
PYCMD="python3"

# check if ANY python can be found on the pi, comes installed, but who knows...
# first we try our 'default' python3..., then we try python..., if none, error...
if ! command -v python3 &>/dev/null; then
    if ! command -v python &>/dev/null; then
        echo "Error: NO Python found installed on your system, or in your path."
        echo "Install Python 3.10.8 or higher, in order to install LCM-LoRA Studio."
        exit 1
    fi
fi

# create the command line command for python, if 'python' found ok
# it will override the default 'python3' we set above
if command -v python &>/dev/null; then
   PYCMD="python"
fi

# inform user which python 'name' we found
echo "Using $PYCMD as the PYTHON command to run python."

# just display the python version, that's all
python_version=$($PYCMD --version 2>&1 | awk '{print $2}')  
echo "Python Version : $python_version"

# get absolute path of the current working directory to the 
# environment variable 'LCMLORASTUDIOHOME' for the duration of the current shell session.
LCMLORASTUDIOHOME=$(pwd)


echo "Activating LCM-LoRA Studio Python Virtual Enviroment..."
source "$LCMLORASTUDIOHOME/env/bin/activate"


# -----START the loop code--------------------
RESTART_FILE="restart.txt"
# Check if the RESTART_FILE exists
if [[ ! -f $RESTART_FILE ]]; then
	echo "Error: Missing File '$RESTART_FILE'. It is needed to run LCM-LoRA Studio in LOOP mode."
	echo "0" > $RESTART_FILE
	echo "Created new '$RESTART_FILE' file. With the default 'EXIT' command."
	echo "Please re-run the 'restart.sh' script again !"
	# Exit script
	echo "Deactivating LCM-LoRA Studio Python Virtual Enviroment..."
	deactivate
	exit 1
fi

# ----- Set a 'Default' Startup State for Huggingface Hub -----
# ----- Uncomment or Change if needed -----
# export HF_HUB_OFFLINE=0
# export TRANSFORMERS_OFFLINE=0
# export HF_DATASETS_OFFLINE=0

# -----START the actual loop--------------------
while true; do
    echo "Launching LCM-LoRA Studio (Loop)..."
    python lcm-lora-studio.py

    RESTART_STATUS="0"
    if [[ -f $RESTART_FILE ]]; then
        RESTART_STATUS=$(cat "$RESTART_FILE")
    fi
    
    if [[ $RESTART_STATUS = "1" ]]; then
        echo "Restart requested by LCM-LoRA Studio. Rerunning LCM-LoRA Studio..."
        # The loop continues
    elif [[ $RESTART_STATUS = "2" ]]; then
        echo "Turning ON Huggingface Hub. Rerunning LCM-LoRA Studio..."
		export HF_HUB_OFFLINE=0
		export TRANSFORMERS_OFFLINE=0
		export HF_DATASETS_OFFLINE=0
        # The loop continues
    elif [[ $RESTART_STATUS = "3" ]]; then
        echo "Turning OFF Huggingface Hub. Rerunning LCM-LoRA Studio..."
		export HF_HUB_OFFLINE=1
		export TRANSFORMERS_OFFLINE=1
		export HF_DATASETS_OFFLINE=1
        # The loop continues
    elif [[ $RESTART_STATUS = "0" ]]; then
        echo "LCM-LoRA Studio requested exit."
        # Exit the loop
        break
    else
        echo "Unknown restart status: $RESTART_STATUS. LCM-LoRA Studio Exiting."
        # Exit the loop
        break
    fi
done
# -----END the loop and code--------------------


echo "Deactivating LCM-LoRA Studio Python Virtual Enviroment..."
deactivate

echo "Thanks for trying LCM-LoRA Studio."

# Exit the script with a success code after the loop has been broken
exit 0



