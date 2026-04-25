@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
python -m streamlit run eye\ui\app.py
pause
