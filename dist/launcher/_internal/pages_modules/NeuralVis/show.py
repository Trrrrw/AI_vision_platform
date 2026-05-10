import sys
import json
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.gui.main_window import MainWindow


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 解析命令行参数：--config <json文件路径>
    config_path = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]

    # 如果存在配置文件，读取参数
    initial_params = None
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                initial_params = json.load(f)
        except Exception as e:
            print(f"[Warning] 无法读取配置文件: {e}", file=sys.stderr)

    window = MainWindow(initial_params=initial_params)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
