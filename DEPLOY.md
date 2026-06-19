# 铁衰老 × CIRI GAT-HGT 药物预测系统 - 部署与使用指南

版本: 3.0.0

---

## 1. 项目简介

本项目通过融合 **GAT-HGT 异构图神经网络**, 从多源生物网络中识别可通过 **铁衰老 (Ferro-aging)** 通路靶向缓解 **CIRI (脑缺血再灌注损伤)** 的中药单体. 系统覆盖 L1 差异表达元分析、L2 多组学/WGCNA、L3 异质图网络建模、L4 药物指纹与 TCM 单体筛选.

---

## 2. 环境要求

- Python >= 3.10
- PyTorch >= 2.0 (CPU 或 CUDA)
- PyTorch Geometric >= 2.3
- R >= 4.2 (仅 L2 GSEA/通路分析需要)
- Windows 10/11 或 Linux

---

## 3. 依赖安装

```bash
# 创建并激活 conda 环境
conda create -n iron-aging python=3.10 -y
conda activate iron-aging

# 安装 Python 依赖
pip install -r requirements.txt
```

`requirements.txt` 包含核心包: `torch`, `torch-geometric`, `torch-scatter`, `pandas`, `numpy`, `scikit-learn`, `rdkit`, `pyyaml`, `openpyxl`, `pytest`, `ruff`.

---

## 4. 快速开始

### 4.1 输入验证

每次运行核心模块前必须执行:

```bash
python validate_inputs.py
```

输出应为 `ERROR: 0, WARNING: 0, OK: 54`.

### 4.2 配置加载测试

```bash
python test_config_loading.py
```

### 4.3 静态检查

```bash
ruff check .
```

---

## 5. 训练与推理

### 5.1 训练 GAT-HGT 模型

```bash
# 使用默认配置 config.yaml
python run_hgt_pipeline.py

# 指定配置文件并清除图构建缓存
python run_hgt_pipeline.py --config config.yaml --clear-cache
```

主要输出:

- `checkpoints/best_model.pt`: 最佳验证 AUC 模型权重.
- `L3_results/module3_metrics.json`: 训练/验证/测试指标.
- `L3_results/acsl4_hub_ranking.csv`: ACSL4 铁衰老 Hub 排名.
- `L3_results/cell_communication_flow.csv`: 细胞通讯流分析结果.

### 5.2 TCM 单体筛选 (Phase 4)

```bash
python run_phase4_tcm_screening.py
```

主要输出:

- `L3_results/phase4_compound_target_ranking.csv`: 化合物-ACSL4 靶点预测排名.
- `L3_results/phase4_acsl4_explainability.csv`: ACSL4 可解释性分析.
- `TCM_monomer_recommendation.xlsx`: 综合推荐表, 含 BBB 透过率、机制类别、对接状态等.

---

## 6. 测试套件

```bash
# 运行全部单元测试
python -m pytest tests/ -q

# 模块 3 回归测试
python test_module3.py
```

---

## 7. 关键结果文件说明

| 文件路径 | 说明 |
| --- | --- |
| `L1/l1_results/L1_genome_wide_de.csv` | L1 全基因组差异表达结果 |
| `L2_results/L2_ACSL4_expression_data.csv` | L2 ACSL4 表达数据 |
| `network_files/string_ppi_edges.csv` | STRING PPI 边 (score >= 400) |
| `network_files/compound_smiles.csv` | 63 个化合物 SMILES |
| `network_files/acsl4_pocket_features.csv` | ACSL4 17 维口袋结构特征 |
| `checkpoints/pretrained_attentivefp_moleculenet.pt` | AttentiveFP 预训练权重 |
| `L3_results/module3_metrics.json` | 模型训练指标 |
| `TCM_monomer_recommendation.xlsx` | TCM 单体综合推荐表 |

---

## 8. 配置说明

核心配置位于 `config.yaml`:

- `data.compound_features.n_components`: 化合物指纹 PCA 维数 (默认 10).
- `model.hidden_dim`: GAT/HGT 隐藏层维数.
- `training.epochs`: 训练轮数.
- `training.early_stopping.patience`: 早停耐心值.

---

## 9. 注意事项

- 图构建第一次运行可能耗时数分钟, 二次运行将自动读取缓存 (<0.01s).
- 若修改了网络输入文件, 需使用 `--clear-cache` 重新构建图.
- GPU 加速在小规模图 (668 节点 / 5,914 边) 上可能不如 CPU, 属正常现象.
