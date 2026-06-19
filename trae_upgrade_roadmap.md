# 铁衰老 × CIRI × GAT-HGT 系统升级改造技术路线

> 版本: 3.0.0-dev  
> 目标: 通过融合 GAT-HGT 异构图网络, 发现能够靶向铁衰老通路、缓解脑缺血再灌注损伤(CIRI)的中药单体, 并完成可验证、可部署、可维护的系统升级.  
> 制定日期: 2026-06-19  
> 适用项目: c:\Users\Jy-Mentor-7\Desktop\铁衰老

---

## 一、项目现状与需求分析

### 1.1 现有成果 (Baseline)

| 维度 | 现状 | 说明 |
|------|------|------|
| 数据层 | L1 跨物种铁衰老验证完成 | 5 个 bulk 数据集、IDSP 评分、Meta 分析、敏感性分析 |
| 单细胞 | module2_sc.py 完成 | GSE174574 小鼠 scRNA-seq 铁衰老评分、细胞通讯 |
| 异质图 | module3_hgt.py 运行通过 | 668 节点 / 5,914 边, best val_AUC=0.9384, ACSL4 Hub 排名 188/286 |
| 化合物 | 63 个 TCM 单体扩展完成 | AttentiveFP 预训练、化合物特征、BBB 评估 |
| 报告 | dry_lab_report 已生成 | 干实验分析报告及图表 |
| 代码 | 脚本式为主, 已开始模块化 | src/iron_aging 包、config.py、run_hgt_pipeline.py 入口 |

### 1.2 核心问题识别

| 优先级 | 问题 | 风险 | 影响 |
|--------|------|------|------|
| P0 | module3_hgt.py 图构建阶段耗时过长(>120s) | 回归测试阻塞、并发门禁挂起 | 开发效率与 CI 稳定性 |
| P0 | 部分扩展化合物 SMILES fallback 不完整 | 零向量特征、模型偏差 | 单体筛选可靠性 |
| P1 | 脚本式代码耦合度高、可维护性差 | 难以复现、难以扩展 | 长期维护 |
| P1 | 缺乏图构建缓存机制 | 重复计算、训练迭代慢 | 模型迭代效率 |
| P1 | GPU 加速比未达预期 (CPU 更快) | 资源浪费 | 训练成本 |
| P2 | 单元测试覆盖率不足 | 回归风险 | 代码质量 |
| P2 | 部署与文档不完善 | 难以交付 | 项目可移植性 |

### 1.3 需求总览

1. **科学目标**: 通过 GAT-HGT 异构图, 识别铁衰老 hub 基因(尤其 ACSL4), 并筛选能靶向这些 hub 的中药单体(尤其 BCP).
2. **工程目标**: 将现有脚本式系统重构为模块化、可配置、可测试、可部署的 v3.0 系统.
3. **数据目标**: 优先复用现有数据; 不足时从 GitHub 等开源平台补充高质量、合规数据.
4. **质量目标**: 所有修改通过 `ruff check .`, `python validate_inputs.py`, `python test_config_loading.py`, `python test_module3.py`.

---

## 二、总体架构设计

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│ 应用层 (Apps)                                                │
│   run_hgt_pipeline.py  ──  训练/推理/解释一体化入口           │
│   dry_lab_ciri_ferroaging_pipeline.py  ──  干实验报告生成      │
├─────────────────────────────────────────────────────────────┤
│ 评估层 (Evaluation)                                          │
│   链路预测评估 (AUC/AP/AUC-PR)                               │
│   Hub 排名评估、SHAP/Integrated Gradients 可解释性            │
│   化合物-靶点结合概率评估、BBB 预测评估                       │
├─────────────────────────────────────────────────────────────┤
│ 训练层 (Training)                                            │
│   HGT/GAT 训练器、多任务损失、负采样、课程学习、EMA、DropEdge  │
├─────────────────────────────────────────────────────────────┤
│ 模型层 (Models)                                              │
│   GATEncoder、HGTEncoder、AttentiveFPCompoundEncoder         │
│   链路预测头、VIB 正则化、自蒸馏                             │
├─────────────────────────────────────────────────────────────┤
│ 数据层 (Data)                                                │
│   图构建器 (缓存)、特征加载器、边加载器、数据验证器           │
├─────────────────────────────────────────────────────────────┤
│ 工具层 (Utils)                                               │
│   配置中心、日志、并发资源评估、输入验证、版本管理            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构 (Target)

```
铁衰老/
├── src/iron_aging/              # 新模块化包
│   ├── __init__.py
│   ├── config.py                # 配置中心
│   ├── data/                    # 数据层
│   │   ├── graph_builder.py     # 异构图构建 + 缓存
│   │   ├── feature_loaders.py   # 节点/边特征加载
│   │   └── validators.py        # 数据验证
│   ├── models/                  # 模型层
│   │   ├── gat_encoder.py
│   │   ├── hgt_encoder.py
│   │   ├── compound_encoder.py
│   │   └── link_predictor.py
│   ├── training/                # 训练层
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── negative_sampling.py
│   ├── evaluation/              # 评估层
│   │   ├── metrics.py
│   │   └── explainability.py
│   ├── apps/                    # 应用入口
│   │   └── hgt_pipeline.py
│   └── utils/                   # 工具层
│       ├── logging_utils.py
│       └── concurrency_utils.py
├── config.yaml                  # 全局配置
├── run_hgt_pipeline.py          # 主入口
├── tests/                       # 测试目录
│   ├── test_config_loading.py
│   ├── test_graph_builder.py
│   ├── test_module3.py
│   └── test_models.py
├── network_files/               # 网络数据
├── L3/                          # L1/L3 输入
├── L3_results/                  # 输出结果
├── external_data/               # 开源外部数据 + readme
└── trae_upgrade_roadmap.md      # 本文件
```

### 2.3 技术选型

| 层面 | 选型 | 理由 |
|------|------|------|
| 图神经网络 | PyTorch Geometric + HGTConv/GATv2Conv | 现有代码已验证, 社区活跃 |
| 分子表示 | DGL-LifeSci / DeepChem AttentiveFP | 预训练权重可用, 捕捉分子图结构 |
| 配置管理 | YAML + 环境变量覆盖 | 简单、可版本化、易覆盖 |
| 代码质量 | ruff + pytest | 快速静态检查 + 单元测试 |
| 并发控制 | concurrency_utils.get_system_resources() | 项目硬约束, 避免无限制进程 |
| 数据缓存 | pickle / PyG Data 序列化 | 减少图构建耗时 |
| 部署 | 命令行入口 + 结果目录 | 轻量, 无需复杂服务 |
| 版本管理 | Semantic Versioning | 已存在 VERSION/CHANGELOG |

### 2.4 外部数据源 (GitHub 优先)

| 数据 | 来源仓库 | 用途 |
|------|----------|------|
| STRING PPI | STRING 官网 / 本地缓存 | 基因-基因 PPi 边 |
| DisGeNET | dhimmel/disgenet (GitHub) | 疾病-基因关联 |
| TRRUST TF-target | molbio-trrust/trrust | TF-target 调控边 |
| B3DB BBB 数据 | [B3DB 项目](https://github.com) | BBB 透过率训练 |
| KEGG/Reactome 通路 | 本地 + KEGGREST | 通路成员基因 |
| AttentiveFP 预训练 | Open Molecular Graph Benchmark | 分子嵌入初始化 |

---

## 三、阶段规划与执行

### 阶段 1: 架构重构与代码质量升级 (Week 1-2)

#### 3.1.1 目标

建立模块化包 `src/iron_aging`, 统一配置中心, 实现图构建缓存, 将现有脚本式代码逐步迁移为可测试模块.

#### 3.1.2 任务清单

| 任务 ID | 任务 | 输出 | 验收标准 |
|---------|------|------|----------|
| 1.1 | 完成 src/iron_aging 目录与 __init__.py | 目录结构、版本号 | `import iron_aging` 成功 |
| 1.2 | 完善 config.py, 支持模型/训练/路径配置 | config.py | `test_config_loading.py` 通过 |
| 1.3 | 封装 graph_builder.py (过渡复用 module3_hgt) | graph_builder.py | 返回与 module3_hgt 一致的图字典 |
| 1.4 | 实现图构建缓存 (pickle 序列化 HeteroData) | cache/hetero_data.pkl | 二次构建时间 < 10s |
| 1.5 | 拆分 models 模块 (GAT/HGT/Compound/LinkPredictor) | models/*.py | dummy forward 测试通过 |
| 1.6 | 拆分 training 模块 (Trainer/Losses/NegativeSampling) | training/*.py | 无 ruff 错误 |
| 1.7 | 拆分 evaluation 模块 (Metrics/Explainability) | evaluation/*.py | 无 ruff 错误 |
| 1.8 | 完善 run_hgt_pipeline.py 入口 | run_hgt_pipeline.py | 可读取配置并运行训练 |
| 1.9 | 全项目 ruff 检查 | ruff report | 关键文件 0 错误 |
| 1.10 | 运行 validate_inputs.py + test_config_loading.py + test_module3.py | 测试报告 | 全部通过 |

#### 3.1.3 资源分配

- 主导: AI 助手执行代码重构
- 依赖: 现有 module3_hgt.py 作为 legacy 参考
- 时间: 2 周内迭代完成

#### 3.1.4 风险与应对

| 风险 | 应对 |
|------|------|
| 模块拆分后 behavior drift | 保留 legacy 路径对比, test_module3.py 作为回归测试 |
| 缓存文件过大 | 使用 pickle protocol=5, 必要时压缩 |
| 测试时间过长 | 缓存 + 减少 test_module3.py 中图构建次数 |

---

### 阶段 2: 数据层增强与开源数据整合 (Week 2-3)

#### 3.2.1 目标

补充和修正节点特征、边数据, 解决化合物 SMILES fallback 不完整、PPI 分数异常、疾病-基因关联缺失等问题.

#### 3.2.2 任务清单

| 任务 ID | 任务 | 数据来源 | 输出 |
|---------|------|----------|------|
| 2.1 | 完善化合物 SMILES 映射 | PubChem / ChEMBL / GitHub | 全 63 个化合物 SMILES 覆盖 |
| 2.2 | 重新生成 AttentiveFP 预训练嵌入 | DGL-LifeSci 预训练权重 | compound_attentivefp_embeddings.csv |
| 2.3 | 修正 ACSL4 口袋特征标准化问题 | PDB 5W8I / AlphaFold DB | acsl4_pocket_features.csv (非全零) |
| 2.4 | 补充/修正 CIRI 疾病-基因关联 | L3/L1_genome_wide_de.csv + DisGeNET | disease_gene_associations.csv |
| 2.5 | 验证 STRING PPI