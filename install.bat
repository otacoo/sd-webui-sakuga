@echo off
title SD WebUI Sakuga - Install ^& Setup

setlocal enabledelayedexpansion

set "BAT_FILE=%~dp0webui-user.bat"
set "NEW_BAT=%~dp0start-webui.bat"
set "ARGS=--no-download-sd-model --theme dark"

:: Rename webui-user.bat to start-webui.bat if not already done
if not exist "%NEW_BAT%" (
    if exist "%BAT_FILE%" (
        echo.
        echo Rename webui-user.bat to start-webui.bat?
        choice /c YN /n /m "[Y/N]: "
        if !errorlevel! == 1 (
            move "%BAT_FILE%" "%NEW_BAT%" >nul
            echo Renamed to start-webui.bat
        ) else (
            set "NEW_BAT=%BAT_FILE%"
        )
    ) else (
        echo Creating start-webui.bat...
        copy "%~dp0webui.bat" "%NEW_BAT%" >nul 2>nul
    )
)

echo.
echo =============================================
echo  SD WebUI Sakuga - Optional Install Features
echo =============================================
echo.

:: Flash Attention
echo Install Flash Attention? (needed for some models like SD3, FLUX)
choice /c YN /n /m "[Y/N]: "
if !errorlevel! == 1 (
    set "ARGS=!ARGS! --flash"
    echo   Added --flash
)

:: SageAttention
echo.
echo Install SageAttention? (optimized attention for speed)
choice /c YN /n /m "[Y/N]: "
if !errorlevel! == 1 (
    set "ARGS=!ARGS! --sage"
    echo   Added --sage
)

:: uv (fast package manager)
echo.
echo Use uv package manager? (faster dependency installs)
choice /c YN /n /m "[Y/N]: "
if !errorlevel! == 1 (
    set "ARGS=!ARGS! --uv"
    echo   Added --uv
)

echo.
echo =============================================
echo  Memory Optimizations
echo =============================================
echo.

:: cuda-malloc
echo Enable --cuda-malloc? (faster memory allocation)
choice /c YN /n /m "[Y/N]: "
if !errorlevel! == 1 (
    set "ARGS=!ARGS! --cuda-malloc"
    echo   Added --cuda-malloc
)

:: cuda-stream
echo.
echo Enable --cuda-stream? (overlapping compute/data transfer)
choice /c YN /n /m "[Y/N]: "
if !errorlevel! == 1 (
    set "ARGS=!ARGS! --cuda-stream"
    echo   Added --cuda-stream
)

:: pin-shared-memory
echo.
echo Enable --pin-shared-memory? (pinned memory for faster transfers)
choice /c YN /n /m "[Y/N]: "
if !errorlevel! == 1 (
    set "ARGS=!ARGS! --pin-shared-memory"
    echo   Added --pin-shared-memory
)

:: Write the updated start-webui.bat
echo.
echo Writing configuration to !NEW_BAT!...
(
    echo @echo off
    echo.
    echo :: set PYTHON=
    echo :: set GIT=
    echo :: set VENV_DIR=
    echo.
    echo set COMMANDLINE_ARGS=!ARGS!
    echo.
    echo :: --xformers --sage --uv
    echo :: --pin-shared-memory --cuda-malloc --cuda-stream
    echo :: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install
    echo.
    echo call %%~dp0webui.bat
) > "!NEW_BAT!"

echo.
echo =============================================
echo  Setup complete!
echo.
echo  ARGS: !ARGS!
echo.
echo  To start:         start-webui.bat
echo  To update:        update.bat
echo  To reconfigure:   run install.bat again
echo =============================================
pause
