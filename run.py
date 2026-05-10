# run.py
import sys
import os
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # 处理 PyInstaller 打包后的临时文件路径问题
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    # 你的主程序文件名
    app_path = os.path.join(base_path, "face.py")

    # 检查文件是否存在
    if not os.path.exists(app_path):
        print(f"错误：找不到应用入口文件 {app_path}")
        sys.exit(1)

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=true",
        "--server.enableXsrfProtection=false",
        "--global.developmentMode=false",
        "--browser.serverAddress=localhost",
        "--server.port=8501"
    ]
    sys.exit(stcli.main())