@ECHO OFF
setlocal
set PYTHONPATH=.

:: 시뮬레이터 실행
echo Starting simulator...
python PlanningSim_scenario_DRL.py

:: 시뮬레이터 프로세스가 완전히 종료될 때까지 대기
:wait_loop
tasklist | find "python.exe" > nul
if %errorlevel% == 0 (
    timeout /t 1 /nobreak > nul
    goto wait_loop
)

:: 가시화 프로그램 실행
echo Starting visualization...
python visualize_env.py

endlocal
