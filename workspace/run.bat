@echo off
REM Windows 入口。用法: run.bat [prob|embed] [--llm-mode replay] [--max-rounds 5]
cd /d "%~dp0"
if "%TAO_PY%"=="" set TAO_PY=python
set BASELINE=
if "%1"=="prob" ( set BASELINE=--baseline prob & shift )
if "%1"=="embed" ( set BASELINE=--baseline embed & shift )
set ARGS=
:loop
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto loop
:run
"%TAO_PY%" -m harness.loop %BASELINE% %ARGS%
