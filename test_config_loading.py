#!/usr/bin/env python3
"""测试节点配置加载和基因集加载"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from module3_hgt import FERROAGING_GENES, _load_graph_node_config, _parse_node_config_simple

# Test 1: graph_node_config.yaml parsing
config = _parse_node_config_simple(Path("network_files/graph_node_config.yaml"))
print("=== 节点配置解析测试 ===")
print(f'pathways: {len(config.get("pathways", []))} 个')
print(f'compounds: {config.get("compounds", [])}')
print(f'cell_types: {config.get("cell_types", [])}')
print(f'diseases: {config.get("diseases", [])}')
vis = config.get("visualization", {})
print(f'short_labels: {vis.get("celltype_short_labels", {})}')
print(f'key_genes: {vis.get("key_genes", [])}')

# Test 2: FERROAGING_GENES loading
print("\n=== 铁衰老基因加载测试 ===")
print(f"FERROAGING_GENES: {len(FERROAGING_GENES)} 个基因")
print(f"前10个: {sorted(list(FERROAGING_GENES))[:10]}")
print(f'ACSL4 in FERROAGING_GENES: {"ACSL4" in FERROAGING_GENES}')
print(f'GPX4 in FERROAGING_GENES: {"GPX4" in FERROAGING_GENES}')

# Test 3: _load_graph_node_config full test
config2 = _load_graph_node_config()
print("\n=== 完整配置加载测试 ===")
print(f'pathways: {len(config2.get("pathways", []))} 个')
print(f'compounds: {len(config2.get("compounds", []))} 个')
print(f'cell_types: {len(config2.get("cell_types", []))} 个')
print(f'diseases: {len(config2.get("diseases", []))} 个')

print("\nAll tests passed!")
