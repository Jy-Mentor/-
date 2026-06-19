#!/usr/bin/env python3
"""铁衰老项目专用 MCP 服务器。

通过标准 MCP 协议为 Trae 提供项目级工具：
- 代码质量门禁（ruff / tests）
- 输入文件验证
- 配置读取
- 项目状态概览
- 缺失数据日志（零造假铁律）

依赖：
    pip install mcp pyyaml

启动方式（stdio，供 Trae MCP 客户端调用）：
    python mcp_project_server.py

注意：本服务器不生成/模拟任何数据，所有返回均来自真实文件或命令输出。
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import concurrency_utils as cu

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    print(
        "ERROR: 缺少依赖 `mcp`。请运行：pip install mcp pyyaml",
        file=sys.stderr,
    )
    traceback.print_exc()
    raise

# ---------------------------------------------------------------------------
# 项目根目录（本文件位于项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# MCP 服务器实例
# ---------------------------------------------------------------------------
mcp = FastMCP("iron-aging-project-server")


# ---------------------------------------------------------------------------
# 工具：运行 ruff 静态检查
# ---------------------------------------------------------------------------
@mcp.tool()
def run_ruff(target: str = ".") -> dict[str, Any]:
    """在项目根目录运行 ruff check。

    Args:
        target: 检查目标，默认为当前目录 "."。

    Returns:
        {"success": bool, "returncode": int, "stdout": str, "stderr": str}
    """
    cmd = [sys.executable, "-m", "ruff", "check", target]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# 工具：运行项目测试
# ---------------------------------------------------------------------------
@mcp.tool()
def run_tests(test_file: str = "test_module3.py") -> dict[str, Any]:
    """运行指定测试文件。

    Args:
        test_file: 测试文件路径，相对于项目根目录。
                   常用值：test_module3.py, test_config_loading.py

    Returns:
        {"success": bool, "returncode": int, "stdout": str, "stderr": str}
    """
    allowed = {
        "test_module3.py",
        "test_config_loading.py",
        "test_module3",
        "test_config_loading",
    }
    if test_file not in allowed:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"只允许运行以下测试之一：{allowed}",
        }

    cmd = [sys.executable, test_file]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# 工具：运行 validate_inputs.py
# ---------------------------------------------------------------------------
@mcp.tool()
def validate_inputs() -> dict[str, Any]:
    """运行 python validate_inputs.py 验证项目输入文件。

    Returns:
        {"success": bool, "returncode": int, "stdout": str, "stderr": str}
    """
    cmd = [sys.executable, "validate_inputs.py"]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# 工具：读取 config.yaml
# ---------------------------------------------------------------------------
@mcp.tool()
def read_config() -> dict[str, Any]:
    """读取并返回 config.yaml 内容。

    Returns:
        {"success": bool, "config": dict | None, "error": str | None}
    """
    config_path = PROJECT_ROOT / "config.yaml"
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return {"success": True, "config": config, "error": None}
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {"success": False, "config": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# 工具：检查文件是否存在
# ---------------------------------------------------------------------------
@mcp.tool()
def check_file_exists(file_path: str) -> dict[str, Any]:
    """检查项目根目录下的文件是否存在，并返回大小与修改时间。

    Args:
        file_path: 相对于项目根目录的文件路径。

    Returns:
        {"exists": bool, "size_bytes": int | None, "mtime": str | None, "error": str | None}
    """
    target = PROJECT_ROOT / file_path
    try:
        # 禁止访问项目根目录之外的文件
        target.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return {
            "exists": False,
            "size_bytes": None,
            "mtime": None,
            "error": "只能检查项目根目录内的文件",
        }

    if not target.exists():
        return {
            "exists": False,
            "size_bytes": None,
            "mtime": None,
            "error": None,
        }

    stat = target.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "error": None,
    }


# ---------------------------------------------------------------------------
# 工具：记录缺失数据警告（零造假铁律）
# ---------------------------------------------------------------------------
@mcp.tool()
def log_missing_data_warning(
    file_path: str,
    reason: str,
    fallback_action: str = "未执行回退",
) -> dict[str, Any]:
    """将缺失数据警告写入项目日志。

    根据零造假铁律，任何缺失数据必须显式记录，禁止静默补零/均值填充。

    Args:
        file_path: 缺失或异常的文件路径。
        reason: 缺失原因说明。
        fallback_action: 已采取的回退动作（如使用零向量）。

    Returns:
        {"success": bool, "log_file": str, "error": str | None}
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "missing_data_warnings.log"

    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "file_path": file_path,
        "reason": reason,
        "fallback_action": fallback_action,
    }

    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"success": True, "log_file": str(log_file), "error": None}
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {"success": False, "log_file": str(log_file), "error": str(exc)}


# ---------------------------------------------------------------------------
# 工具：获取项目状态概览
# ---------------------------------------------------------------------------
@mcp.tool()
def get_project_status() -> dict[str, Any]:
    """返回项目关键目录与文件的存在性概览。

    Returns:
        {"success": bool, "status": dict, "error": str | None}
    """
    try:
        status = {
            "project_root": str(PROJECT_ROOT),
            "timestamp": datetime.now().isoformat(),
            "key_files": {
                "config.yaml": (PROJECT_ROOT / "config.yaml").exists(),
                "ruff.toml": (PROJECT_ROOT / "ruff.toml").exists(),
                "validate_inputs.py": (PROJECT_ROOT / "validate_inputs.py").exists(),
                "module3_hgt.py": (PROJECT_ROOT / "module3_hgt.py").exists(),
                "test_module3.py": (PROJECT_ROOT / "test_module3.py").exists(),
            },
            "key_directories": {
                "L1": (PROJECT_ROOT / "L1").exists(),
                "L2_WGCNA_input": (PROJECT_ROOT / "L2_WGCNA_input").exists(),
                "L3": (PROJECT_ROOT / "L3").exists(),
                "network_files": (PROJECT_ROOT / "network_files").exists(),
                "L3_results": (PROJECT_ROOT / "L3_results").exists(),
            },
        }
        return {"success": True, "status": status, "error": None}
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {"success": False, "status": {}, "error": str(exc)}


# ---------------------------------------------------------------------------
# 工具：运行白名单命令
# ---------------------------------------------------------------------------
ALLOWED_COMMANDS: dict[str, list[str]] = {
    "git_status": ["git", "status", "--short"],
    "git_diff": ["git", "diff"],
    "git_log": ["git", "log", "--oneline", "-5"],
}


@mcp.tool()
def run_whitelisted_command(command_name: str) -> dict[str, Any]:
    """运行预定义的白名单命令。

    Args:
        command_name: 命令名称，可选值：git_status, git_diff, git_log

    Returns:
        {"success": bool, "returncode": int, "stdout": str, "stderr": str}
    """
    if command_name not in ALLOWED_COMMANDS:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"只允许以下命令：{list(ALLOWED_COMMANDS.keys())}",
        }

    cmd = ALLOWED_COMMANDS[command_name]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# 工具：获取系统资源与并发推荐
# ---------------------------------------------------------------------------
@mcp.tool()
def get_system_resources(
    memory_reserve_mb: float = 2048.0,
    memory_per_worker_mb: float = 512.0,
) -> dict[str, Any]:
    """获取 CPU、内存资源并返回推荐并发 worker 数。

    Args:
        memory_reserve_mb: 保留内存（MB）。
        memory_per_worker_mb: 每个 worker 预估内存（MB）。

    Returns:
        {"success": bool, "resources": dict, "error": str | None}
    """
    try:
        resources = cu.get_system_resources(memory_reserve_mb, memory_per_worker_mb)
        return {
            "success": True,
            "resources": resources.to_dict(),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {"success": False, "resources": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# 工具：并发检查多个文件
# ---------------------------------------------------------------------------
@mcp.tool()
def run_parallel_file_checks(
    file_paths: list[str],
    max_workers: int | None = None,
) -> dict[str, Any]:
    """并发检查多个项目文件是否存在。

    Args:
        file_paths: 相对于项目根目录的文件路径列表。
        max_workers: 最大并发数，None 则自动推荐。

    Returns:
        {"success": bool, "completed": int, "failed": int, "results": list, "errors": list}
    """
    try:
        result = cu.parallel_file_checks(
            file_paths=file_paths,
            base_dir=PROJECT_ROOT,
            max_workers=max_workers,
        )
        return result
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {
            "success": False,
            "completed": 0,
            "failed": 1,
            "results": [],
            "errors": [str(exc)],
        }


# ---------------------------------------------------------------------------
# 工具：并发执行质量门禁
# ---------------------------------------------------------------------------
@mcp.tool()
def run_parallel_quality_gate(max_workers: int | None = None) -> dict[str, Any]:
    """并发执行 ruff、validate_inputs、test_config_loading、test_module3。

    任务彼此独立，并发执行可快速暴露问题。若任一任务失败，返回 success=false。

    Args:
        max_workers: 最大并发数，None 则自动推荐。

    Returns:
        {"success": bool, "completed": int, "failed": int, "results": list, "errors": list}
    """
    try:
        return cu.parallel_quality_gate(
            project_root=PROJECT_ROOT,
            max_workers=max_workers,
        )
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return {
            "success": False,
            "completed": 0,
            "failed": 1,
            "results": [],
            "errors": [str(exc)],
        }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 使用 stdio 传输，供 Trae MCP 客户端调用
    mcp.run(transport="stdio")
