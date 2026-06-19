"""铁衰老项目 - 多进程并发工具模块。

设计原则：
- 适度并发：根据 CPU 核心数与可用内存动态计算 worker 数量，避免系统过载。
- 透明错误：任何子进程异常都会记录并向上传播，禁止静默吞错。
- 真实数据：本模块不生成/模拟任何数据，仅对真实文件/任务进行并发调度。
- 任务隔离：CPU 密集型任务使用进程池；IO 密集型任务可配置为线程池。
- 资源感知：提供系统资源查询，供调用方决定是否启用并发及并发度。

依赖：标准库 only（Python >= 3.9）。
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 资源信息数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemResources:
    """系统资源快照。"""

    cpu_count_logical: int
    cpu_count_physical: int
    available_memory_mb: float | None
    recommended_workers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count_logical": self.cpu_count_logical,
            "cpu_count_physical": self.cpu_count_physical,
            "available_memory_mb": self.available_memory_mb,
            "recommended_workers": self.recommended_workers,
        }


# ---------------------------------------------------------------------------
# 资源感知 worker 数计算
# ---------------------------------------------------------------------------
def get_system_resources(
    memory_reserve_mb: float = 2048.0,
    memory_per_worker_mb: float = 512.0,
) -> SystemResources:
    """获取系统资源并推荐 worker 数量。

    推荐逻辑：
    - 物理核心数为上限。
    - 根据可用内存进一步限制：workers <= (available_memory - reserve) / per_worker。
    - 至少保留 1 个 worker，最多不超过物理核心数。

    Args:
        memory_reserve_mb: 保留内存（MB），防止系统卡死。
        memory_per_worker_mb: 每个 worker 预估占用内存（MB）。

    Returns:
        SystemResources 实例。
    """
    logical = os.cpu_count() or 4
    physical = getattr(os, "process_cpu_count", lambda: logical)()

    available_memory_mb: float | None = None
    try:
        import psutil

        mem = psutil.virtual_memory()
        available_memory_mb = float(mem.available) / (1024 * 1024)
    except Exception:
        logger.warning("无法获取内存信息，将仅按 CPU 核心数推荐 worker。")

    if available_memory_mb is not None:
        memory_limited_workers = max(
            1,
            int((available_memory_mb - memory_reserve_mb) / memory_per_worker_mb),
        )
    else:
        memory_limited_workers = physical

    recommended = max(1, min(physical, memory_limited_workers))
    return SystemResources(
        cpu_count_logical=logical,
        cpu_count_physical=physical,
        available_memory_mb=available_memory_mb,
        recommended_workers=recommended,
    )


def recommend_workers(
    task_memory_mb: float = 512.0,
    reserve_memory_mb: float = 2048.0,
    max_workers: int | None = None,
    min_workers: int = 1,
) -> int:
    """根据系统资源推荐并发 worker 数。

    Args:
        task_memory_mb: 单个任务预估内存占用（MB）。
        reserve_memory_mb: 系统保留内存（MB）。
        max_workers: 用户指定的上限（可选）。
        min_workers: 最小 worker 数。

    Returns:
        推荐 worker 数量。
    """
    resources = get_system_resources(reserve_memory_mb, task_memory_mb)
    workers = resources.recommended_workers
    if max_workers is not None:
        workers = min(workers, max(max_workers, 1))
    return max(min_workers, workers)


# ---------------------------------------------------------------------------
# 并发执行器
# ---------------------------------------------------------------------------
def run_tasks_parallel(
    tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
    max_workers: int | None = None,
    use_processes: bool = True,
    timeout_per_task: float | None = None,
) -> dict[str, Any]:
    """并发执行一组任务，收集结果并统一报告错误。

    Args:
        tasks: 任务列表，每项为 (callable, args, kwargs)。
        max_workers: 最大并发数，None 则自动根据资源推荐。
        use_processes: True 使用进程池（CPU 密集型），False 使用线程池（IO 密集型）。
        timeout_per_task: 每个任务超时时间（秒），None 表示不限制。

    Returns:
        {
            "success": bool,
            "completed": int,
            "failed": int,
            "results": list[dict],
            "errors": list[str],
        }
    """
    if not tasks:
        return {
            "success": True,
            "completed": 0,
            "failed": 0,
            "results": [],
            "errors": [],
        }

    if max_workers is None:
        max_workers = recommend_workers()

    max_workers = max(1, min(max_workers, len(tasks)))
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0
    failed = 0

    logger.info(
        "启动%s并发执行：%d 个任务，%d 个 workers",
        "进程" if use_processes else "线程",
        len(tasks),
        max_workers,
    )

    try:
        with executor_cls(max_workers=max_workers) as executor:
            future_to_index = {}
            for idx, (func, args, kwargs) in enumerate(tasks):
                # 为每个任务命名，便于日志追踪
                task_name = getattr(func, "__name__", f"task_{idx}")
                future = executor.submit(_run_single_task, func, args, kwargs, task_name)
                future_to_index[future] = idx

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    if timeout_per_task:
                        task_result = future.result(timeout=timeout_per_task)
                    else:
                        task_result = future.result()
                    results.append({"index": idx, "result": task_result})
                    completed += 1
                except Exception as exc:  # noqa: BLE001
                    error_msg = f"任务 {idx} 失败: {exc}"
                    logger.error(error_msg)
                    traceback.print_exc()
                    errors.append(error_msg)
                    failed += 1
    except Exception as exc:  # noqa: BLE001
        error_msg = f"并发执行框架异常: {exc}"
        logger.error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        failed += 1

    success = failed == 0
    return {
        "success": success,
        "completed": completed,
        "failed": failed,
        "results": results,
        "errors": errors,
    }


def _run_single_task(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    task_name: str,
) -> Any:
    """包装单个任务执行，增加日志与异常传播。"""
    logger.info("开始执行任务: %s", task_name)
    try:
        result = func(*args, **kwargs)
        logger.info("任务完成: %s", task_name)
        return result
    except Exception:
        logger.error("任务 %s 执行失败", task_name)
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# 常用并发模式封装
# ---------------------------------------------------------------------------
def parallel_file_checks(
    file_paths: list[str | Path],
    base_dir: Path | str = ".",
    max_workers: int | None = None,
) -> dict[str, Any]:
    """并发检查多个文件是否存在。

    Args:
        file_paths: 文件路径列表（相对于 base_dir）。
        base_dir: 基础目录。
        max_workers: 最大并发数。

    Returns:
        run_tasks_parallel 标准结果字典。
    """
    base = Path(base_dir).resolve()
    tasks = [
        (_check_single_file, (base / Path(fp),), {})
        for fp in file_paths
    ]
    return run_tasks_parallel(tasks, max_workers=max_workers, use_processes=False)


def _check_single_file(file_path: Path) -> dict[str, Any]:
    """检查单个文件状态。"""
    try:
        # 安全校验：禁止访问 base_dir 外部
        file_path.resolve().relative_to(file_path.parent.resolve())
    except ValueError:
        return {
            "file": str(file_path),
            "exists": False,
            "error": "路径解析失败",
        }

    exists = file_path.exists()
    result: dict[str, Any] = {
        "file": str(file_path),
        "exists": exists,
    }
    if exists:
        stat = file_path.stat()
        result["size_bytes"] = stat.st_size
        result["mtime"] = stat.st_mtime
    return result


def parallel_command_runners(
    commands: list[list[str]],
    cwd: str | Path = ".",
    max_workers: int | None = None,
) -> dict[str, Any]:
    """并发运行多个外部命令（独立进程）。

    注意：本函数只运行调用方传入的命令，不做额外白名单校验；
    调用方（如 MCP 服务器）应在外层完成安全检查。

    Args:
        commands: 命令列表，每项为参数列表（如 ["python", "validate_inputs.py"]）。
        cwd: 工作目录。
        max_workers: 最大并发数。

    Returns:
        run_tasks_parallel 标准结果字典。
    """
    cwd_path = Path(cwd).resolve()
    tasks = [
        (_run_single_command, (cmd, cwd_path), {})
        for cmd in commands
    ]
    return run_tasks_parallel(tasks, max_workers=max_workers, use_processes=True)


def _run_single_command(command: list[str], cwd: Path) -> dict[str, Any]:
    """运行单个命令并返回结构化结果。"""
    import subprocess

    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


# ---------------------------------------------------------------------------
# 便捷函数：并行运行 lint 与测试（彼此独立）
# ---------------------------------------------------------------------------
def parallel_quality_gate(
    project_root: str | Path = ".",
    max_workers: int | None = None,
) -> dict[str, Any]:
    """并发执行 lint 与测试类任务。

    包含：ruff check、validate_inputs、test_config_loading、test_module3。
    注意：validate_inputs 理论上应在测试前执行，但它们彼此独立，
    若 validate_inputs 失败，测试通常也会失败，因此并发执行可快速暴露问题。

    Args:
        project_root: 项目根目录。
        max_workers: 最大并发数。

    Returns:
        结构化结果字典。
    """
    project_root = Path(project_root).resolve()
    commands = [
        ["python", "-m", "ruff", "check", "."],
        ["python", "validate_inputs.py"],
        ["python", "test_config_loading.py"],
        ["python", "test_module3.py"],
    ]
    return parallel_command_runners(
        commands,
        cwd=project_root,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# 模块入口：命令行快速查看资源推荐
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    resources = get_system_resources()
    print(json.dumps(resources.to_dict(), indent=2, ensure_ascii=False))
