#!/bin/bash

# Version 1.3 - Adds Huggingface Hub On/Off Control

echo Starting LCM-LoRA Studio Installation...

# exit immediately if any command fails
set -e

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

echo "Installing LCM-LoRA Studio in: $LCMLORASTUDIOHOME"

echo "Creating LCM-LoRA Studio Python Virtual Enviroment..."
python -m venv "$LCMLORASTUDIOHOME/env"

echo "Activating LCM-LoRA Studio Python Virtual Enviroment..."
source "$LCMLORASTUDIOHOME/env/bin/activate"

echo "Installing LCM-LoRA Studio Python Packages and Requirements..."
pip install -r "$LCMLORASTUDIOHOME/lcm-lora-studio-requirement.txt"

echo "Deactivating LCM-LoRA Studio Python Virtual Enviroment..."
deactivate

echo "Finished Installation of Packages and Requirements for LCM-LoRA Studio."
echo "Thanks for installing LCM-LoRA Studio."
echo "Type: './run.sh' to run LCM-LoRA Studio."






