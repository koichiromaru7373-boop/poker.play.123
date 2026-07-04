@echo off
rem テストプレイ用サーバー起動(初回は環境構築も行う)
cd /d %~dp0

where python >nul 2>nul || (
    echo Python が見つかりません。https://www.python.org/downloads/ からインストールし、
    echo "Add Python to PATH" にチェックを入れてください。
    pause & exit /b 1
)

if not exist venv\Scripts\activate.bat (
    echo 初回セットアップ中(数分かかります)...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo.
echo サーバーを起動します。ブラウザで http://127.0.0.1:8000/play を開いてください。
echo (スマホから遊ぶ場合は同じWi-Fiで http://このPCのIP:8000/play )
echo 終了するにはこのウィンドウで Ctrl+C
echo.
uvicorn api_server:app --host 0.0.0.0 --port 8000
pause
