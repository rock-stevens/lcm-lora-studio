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

echo Starting LCM-LoRA Studio LOOP...

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



echo Activating Virtual Enviroment for LCM-LoRA Studio...
call .\env\Scripts\activate.bat
echo LCM-LoRA Studio Virtual Enviroment Activated.

REM ----- Set a 'Default' Startup State for Huggingface Hub -----
REM ----- Uncomment or Change if needed -----
REM set HF_HUB_OFFLINE=0
REM set HF_DATASETS_OFFLINE=0
REM set TRANSFORMERS_OFFLINE=0

:start_app
    echo Launching LCM-LoRA Studio (Loop)...
	python lcm-lora-studio.py

    set RESTART_STATUS=0
    if exist restart.txt (
        for /f "delims=" %%a in (restart.txt) do (
            set RESTART_STATUS=%%a
        )
    )

    if "%RESTART_STATUS%"=="1" (
        echo Restart requested by LCM-LoRA Studio. Rerunning LCM-LoRA Studio...
        goto start_app
    )

    if "%RESTART_STATUS%"=="2" (
        echo Turning ON Huggingface Hub. Rerunning LCM-LoRA Studio...
		set HF_HUB_OFFLINE=0
		set HF_DATASETS_OFFLINE=0
		set TRANSFORMERS_OFFLINE=0
        goto start_app
    )

    if "%RESTART_STATUS%"=="3" (
        echo Turning OFF Huggingface Hub. Rerunning LCM-LoRA Studio...
		set HF_HUB_OFFLINE=1
		set HF_DATASETS_OFFLINE=1
		set TRANSFORMERS_OFFLINE=1
        goto start_app
    )

    if "%RESTART_STATUS%"=="0" (
        echo LCM-LoRA Studio requested exit.
        goto :studiodone
    )


:studiodone
echo Deactivating Virtual Enviroment for LCM-LoRA Studio...
call .\env\Scripts\deactivate.bat
echo LCM-LoRA Studio Virtual Enviroment Deactivated.

echo Thanks for using LCM-LoRA Studio.

REM wait for user to see what happened, error or not
pause

