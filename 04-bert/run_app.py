import os
import subprocess
import sys

# 尝试杀掉可能残留的旧 Streamlit 进程
os.system("taskkill /F /IM streamlit.exe 2>nul")
os.system("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq *streamlit*\" 2>nul")

os.system(r"streamlit run D:\heima\4\TMF\04-bert\app.py")