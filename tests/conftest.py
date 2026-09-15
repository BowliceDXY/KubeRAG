"""pytest 全局配置：确保项目根目录可被 import（app.* 模块）"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
