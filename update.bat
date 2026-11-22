@echo off
setlocal
cd /d "%~dp0"
title SD WebUI Sakuga - Update

echo Pulling latest changes...
echo.

git fetch
if errorlevel 1 goto failed

for /f "delims=" %%f in ('git diff --name-only "HEAD..@{u}"') do (
    git ls-files --error-unmatch "%%f" >nul 2>&1
    if errorlevel 1 (
        if exist "%%f" (
            echo Backing up "%%f" to "%%f.local-backup"
            move /Y "%%f" "%%f.local-backup" >nul
            if errorlevel 1 goto failed
        )
    )
)

git reset --hard "@{u}"
if errorlevel 1 goto failed

echo.
echo Update finished.
pause
exit /b 0

:failed
echo.
echo Update failed.
pause
exit /b 1
