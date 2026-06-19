"""测试异构图构建与缓存."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iron_aging.data.graph_builder import (  # noqa: E402
    build_heterogeneous_graph,
    build_pyg_data,
    clear_graph_cache,
)


@pytest.fixture(scope="module", autouse=True)
def _clear_cache_once():
    """整个模块只清除一次缓存, 避免重复构建."""
    clear_graph_cache()
    yield
    clear_graph_cache()


def test_build_heterogeneous_graph_structure():
    """验证图数据结构完整."""
    graph_data = build_heterogeneous_graph(use_cache=True)

    for nt in ["gene", "pathway", "compound", "disease", "pocket"]:
        assert nt in graph_data
        assert "x" in graph_data[nt]
        assert "names" in graph_data[nt]
        assert "n" in graph_data[nt]

    assert "edges" in graph_data
    assert len(graph_data["edges"]) > 0


def test_graph_build_cache_speedup():
    """验证缓存能显著加速第二次图构建."""
    start1 = time.time()
    graph_data1 = build_heterogeneous_graph(use_cache=True)
    elapsed1 = time.time() - start1

    start2 = time.time()
    graph_data2 = build_heterogeneous_graph(use_cache=True)
    elapsed2 = time.time() - start2

    assert elapsed2 < elapsed1
    assert graph_data1["gene"]["n"] == graph_data2["gene"]["n"]


def test_build_pyg_data():
    """验证能转换为 PyG HeteroData."""
    graph_data = build_heterogeneous_graph(use_cache=True)
    data = build_pyg_data(graph_data)

    assert hasattr(data, "node_types")
    assert "gene" in data.node_types
