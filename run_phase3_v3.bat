@echo off
rem Phase 3 (v3): DQN長時間学習 (200000ハンド)
rem v2のベストから再開したい場合は下の行を書き換える:
rem   .\venv\Scripts\python.exe train_dqn_v3.py --resume experiments\dqn_v2\dqn_model_best.pth --num_episodes 100000
cd /d %~dp0

.\venv\Scripts\python.exe train_dqn_v3.py

pause
