@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt
py src\student_performance_analysis.py
pause
