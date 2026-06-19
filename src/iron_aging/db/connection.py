"""数据库连接管理.

支持 PostgreSQL (生产) 与 SQLite (本地开发/测试) 双后端.
URL 优先级:
1. 环境变量 IRON_DATABASE_URL
2. config.yaml database.url (支持 ${VAR:-default} 占位符做显式标记)
3. 默认 SQLite (项目根目录 iron_aging.db)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from iron_aging import PROJECT_ROOT
from iron_aging.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_URL = f"sqlite:///{PROJECT_ROOT / 'iron_aging.db'}"
_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_placeholders(value: str) -> str:
    """解析字符串中的 ${VAR:-default} 占位符."""

    def _repl(match: re.Match[str]) -> str:
        expr = match.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
        else:
            var, default = expr, ""
        return os.environ.get(var, default)

    return _ENV_PLACEHOLDER_RE.sub(_repl, value)


def get_database_url(config_path: Path | str | None = None) -> str:
    """返回解析后的数据库 URL."""
    if os.environ.get("IRON_DATABASE_URL"):
        return os.environ["IRON_DATABASE_URL"]

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        config = {}

    db_config = config.get("database", {})
    url = db_config.get("url", DEFAULT_SQLITE_URL)
    if isinstance(url, str):
        url = _resolve_env_placeholders(url)
    else:
        url = DEFAULT_SQLITE_URL
    return url


def get_engine(url: str | None = None, **kwargs: Any) -> Engine:
    """创建 SQLAlchemy Engine.

    Args:
        url: 数据库 URL; 为 None 时自动解析.
        **kwargs: 透传给 create_engine 的额外参数.

    Returns:
        SQLAlchemy Engine 实例.
    """
    url = url or get_database_url()
    is_sqlite = url.startswith("sqlite")

    engine_kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
        "echo": kwargs.pop("echo", False),
    }
    if not is_sqlite:
        engine_kwargs.update(
            {
                "pool_size": kwargs.pop("pool_size", 10),
                "max_overflow": kwargs.pop("max_overflow", 20),
            }
        )
    else:
        # SQLite 单连接即可, 禁用连接池避免多线程问题
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    engine_kwargs.update(kwargs)
    logger.info("创建数据库引擎: %s", url.split("//")[0] + "//***")
    return create_engine(url, **engine_kwargs)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Any]:
    """返回 Session 工厂."""
    engine = engine or get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine | None = None) -> Engine:
    """创建所有表."""
    from iron_aging.db.models import Base

    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    logger.info("数据库表初始化完成: %s", engine.url)
    return engine
