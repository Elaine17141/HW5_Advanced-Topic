@echo off
REM 安裝相依套件
echo 正在安裝相依套件...
pip install -r requirements.txt

REM 啟動 Streamlit 應用
echo 正在啟動 AI 偵測器應用...
streamlit run ai_detector.py
