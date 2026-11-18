@echo off
REM ============================================================
REM  git_sync.bat -- 本机一键同步到 GitHub（我这边的沙箱没有外网，推送只能由你执行）
REM  用法:  git_sync.bat ["提交说明"]
REM  行为:  检查 .env 未被跟踪 -> add -A -> commit -> pull --rebase -> push
REM ============================================================
cd /d "%~dp0"
set MSG=%~1
if "%MSG%"=="" set MSG=update %date% %time%

git ls-files --error-unmatch workspace\.env >nul 2>&1 && (
  echo [git_sync] !!! workspace\.env is tracked. Aborting to protect the API key.
  exit /b 1
)

git add -A
git diff --cached --quiet && echo [git_sync] nothing to commit || git commit -m "%MSG%"
git pull --rebase origin main
if errorlevel 1 ( echo [git_sync] pull --rebase failed, resolve conflicts then rerun & exit /b 1 )
git push -u origin main
echo [git_sync] done.
