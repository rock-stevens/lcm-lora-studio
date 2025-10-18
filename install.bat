@echo off

REM create a localized environment for changes to our environment variables
setlocal

REM get absolute path of the current working directory to the 
REM environment variable 'LCMLORASTUDIOHOME' for the duration of the current shell session.
REM get the drive letter
set DRIVEBASE=%~d0
REM get current dir
set LCMLORASTUDIOHOME=%CD%
REM change to drive 'drive letter'
%DRIVEBASE%
REM change to LCMLORASTUDIOHOME dir
cd %LCMLORASTUDIOHOME%


REM ===================================================================
REM ===================================================================
REM ===================================================================
rem Rock's portable Python/AI junk was here...
REM ===========================================================
REM ===========================================================
REM ===========================================================

echo Starting LCM-LoRA Studio Installation...

REM check if python can be found...
REM old school, i don't use powershell.
REM i didn't use 'command' && ( ok ) || ( err exit ) way of error checking
REM up to the user anyway to ensure Python 3.10.6 or higher is installed
call python --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python :OK
) else (
    echo Error: NO Python found installed on your system, or it is not in your path. It is needed in order to install LCM-LoRA Studio.
    echo Ensure Python 3.10.8 or higher is installed on your system, AND also in your PATH.
    echo If NOT installed, Install Python 3.10.8 or higher, make sure you check the PATH box at end of the installation process. 
	echo Python 3.10.8 or higher is needed in order to install LCM-LoRA Studio.
    pause
    exit /b 1
)



REM just display the python version, that's all
for /f "tokens=2" %%I in ('python --version 2^>^&1') do (
    echo Python Version : %%I
)


echo Creating LCM-LoRA Studio Python Virtual Enviroment...
python -m venv .\env

echo Activating Virtual Enviroment for LCM-LoRA Studio...
call .\env\Scripts\activate.bat
echo LCM-LoRA Studio Virtual Enviroment Activated.

echo Installing LCM-LoRA Studio Python Packages and Requirements...
pip install -r "lcm-lora-studio-requirement.txt"

echo Deactivating Virtual Enviroment for LCM-LoRA Studio...
call .\env\Scripts\deactivate.bat
echo LCM-LoRA Studio Virtual Enviroment Deactivated.

echo Finished Installation of Packages and Requirements for LCM-LoRA Studio.
echo Thanks for installing.
echo Type: 'run.bat' to Run LCM-LoRA Studio.

REM wait for user to see what happened, error or not
pause

