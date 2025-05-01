@echo off
REM 切換到指定目錄
cd /d "D:\MA system\LangGraph"

REM 啟動虛擬環境（若 activate.bat 不存在，可試著使用 activate）
call langgraph-env\Scripts\activate.bat

REM 啟動 LangGraph Studio
langgraph dev

REM 暫停視窗（可選）
pause
