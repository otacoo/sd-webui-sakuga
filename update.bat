@echo off
title SD WebUI Sakuga - Update

echo Pulling latest changes...
echo.

git reset --hard
git pull

echo.
echo Update finished.
pause
