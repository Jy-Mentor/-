"""初始化数据库：创建所有表."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许从项目根目录导入 src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iron_aging.db.connection import get_engine, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化铁衰老项目数据库")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="数据库 URL, 默认读取 config.yaml 或环境变量 IRON_DATABASE_URL",
    )
    args = parser.parse_args()

    engine = get_engine(url=args.url)
    init_db(engine)
    logger.info("数据库初始化完成: %s", engine.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
