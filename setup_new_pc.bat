@echo off
rem 別PCでの環境構築: このフォルダ一式をコピーしてから実行する
rem (venvフォルダはコピー不要。このスクリプトが作り直す)
cd /d %~dp0

where python >nul 2>nul || (echo Python が見つかりません。python.org からインストールしてください & pause & exit /b 1)

python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo セットアップ完了。学習の再開例:
echo   .\venv\Scripts\python.exe train_nfsp.py --resume experiments\nfsp\nfsp_last.pth
pause
