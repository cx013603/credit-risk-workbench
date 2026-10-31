"""让直接运行的脚本共用 src 布局，不安装包或修改全局环境。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
