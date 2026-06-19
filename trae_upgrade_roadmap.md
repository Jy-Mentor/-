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
| 2.5 | 验证 STRING PPI 边分数 | STRING 官网下载 / 本地缓存 | string_ppi_edges.csv (分数 ≥400) |
| 2.6 | 扩展 TCM 单体集至 50+ | 现有 20 核心 + 43 扩展 | compound_smiles.csv |
| 2.7 | 建立外部数据 README | 手动整理 | external_data/README.md |
| 2.8 | 重新运行 generate_all_network_files.py | 本地数据 | 更新 network_files/ |
| 2.9 | 运行 validate_inputs.py | - | 全部检查通过 |

#### 3.2.3 资源分配

- 主导: AI 助手执行数据脚本
- 依赖: PubChem API、STRING 数据、DisGeNET GitHub 数据
- 时间: 1-2 周

#### 3.2.4 风险与应对

| 风险 | 应对 |
|------|------|
| PubChem API 限流 | 本地缓存 + 批量请求 + 失败重试 |
| 外部数据下载失败 | 记录失败日志, 使用本地备份 |
| 数据版权合规 | 仅使用公开学术数据库, 记录来源与版本 |

---

### 阶段 3: 模型升级与训练策略优化 (Week 3-5)

#### 3.3.1 目标

将 HGT/GAT 模型从 module3_hgt.py 迁移到 src/iron_aging/models, 引入预训练、课程学习、负采样优化, 并解决 GPU 加速比问题.

#### 3.3.2 任务清单

| 任务 ID | 任务 | 输出 | 验收标准 |
|---------|------|------|----------|
| 3.1 | 实现 src/iron_aging/models/gat_encoder.py | GATv2 编码器 | dummy 输入 forward 通过 |
| 3.2 | 实现 src/iron_aging/models/hgt_encoder.py | HGT 编码器 | dummy 输入 forward 通过 |
| 3.3 | 实现 compound_encoder.py (AttentiveFP) | 化合物编码器 | 真实 SMILES 嵌入非零 |
| 3.4 | 实现 link_predictor.py (异质边解码器) | 链路预测头 | 输出形状正确 |
| 3.5 | 迁移多任务训练 (GAT+HGT+课程学习+负采样+EMA) | training/trainer.py | val_AUC ≥ 0.90 |
| 3.6 | 实现图构建缓存集成训练流程 | trainer.py | 训练启动时间 < 30s |
| 3.7 | 优化 GPU 使用策略 | trainer.py | 大图用 GPU, 小图用 CPU |
| 3.8 | 集成 TensorBoard / CSVLogger | training/logger.py | 训练日志可记录 |
| 3.9 | 实现 Early Stopping + 学习率调度 | trainer.py | 训练稳定收敛 |
| 3.10 | 跑通完整训练并记录指标 | L3_results/ | test AUC 达到 baseline |

#### 3.3.3 资源分配

- 主导: AI 助手
- 依赖: PyTorch Geometric, DGL-LifeSci
- 时间: 2-3 周

#### 3.3.4 风险与应对

| 风险 | 应对 |
|------|------|
| 模型迁移后性能下降 | 与 module3_hgt.py 结果对比, 调参对齐 |
| GPU 内存不足 | 梯度累积 / 子图采样 |
| 训练不稳定 | EMA、DropEdge、VIB 预热 |

---

### 阶段 4: TCM 单体筛选与机制阐释 (Week 5-6)

#### 3.4.1 目标

基于升级后的 HGT-GAT 模型, 系统筛选中药单体, 解释 BCP 靶向 ACSL4 的机制.

#### 3.4.2 任务清单

| 任务 ID | 任务 | 输出 |
|---------|------|------|
| 4.1 | 化合物-靶点链路预测打分 | compound_target_ranking.csv |
| 4.2 | 铁衰老 Hub 基因排名 (ACSL4 等) | hub_genes_ranking.csv |
| 4.3 | 跨细胞通讯注意力流分析 | cell_communication_flow.csv |
| 4.4 | SHAP / Integrated Gradients 解释 ACSL4 Hub 边贡献 | acsl4_shap_importance.csv |
| 4.5 | BBB 透过率预测对比 (BCP vs VC vs Fer-1) | bbb_prediction.csv |
| 4.6 | 分子对接 (AutoDock Vina) | docking_results.csv |
| 4.7 | 生成 TCM 单体 top-K 推荐表 | tcm_monomer_recommendation.xlsx |

#### 3.4.3 资源分配

- 主导: AI 助手
- 依赖: RDKit, AutoDock Vina, LightGBM
- 时间: 1-2 周

#### 3.4.4 风险与应对

| 风险 | 应对 |
|------|------|
| 对接软件未安装 | 先检查命令可用性, 缺失则记录并跳过 |
| 可解释性结果不稳定 | 多次采样取平均 |
| BCP 排名不突出 | 检查特征质量, 不人为调整权重 |

---

### 阶段 5: 验证、文档与部署上线 (Week 6-7)

#### 3.5.1 目标

完成单元测试、集成测试、文档更新, 并将最终代码提交到 GitHub.

#### 3.5.2 任务清单

| 任务 ID | 任务 | 输出 |
|---------|------|------|
| 5.1 | 编写/补全单元测试 (pytest) | tests/*.py, 覆盖率 ≥80% |
| 5.2 | 运行 ruff + validate_inputs + test_config_loading + test_module3 | 质量报告 |
| 5.3 | 更新 CHANGELOG.md 与 VERSION | v3.0.0 |
| 5.4 | 更新 dry_lab_report.md / 生成 Excel 报告 | 技术路线执行报告 |
| 5.5 | 清理大文件、确认 .gitignore | 仓库体积合规 |
| 5.6 | 提交代码到 GitHub | GitHub commit/PR |
| 5.7 | 编写部署说明 (环境安装、运行命令) | DEPLOY.md |
| 5.8 | 建立后续维护计划 | MAINTENANCE.md |

#### 3.5.3 资源分配

- 主导: AI 助手
- 依赖: GitHub MCP 工具
- 时间: 1 周

#### 3.5.4 风险与应对

| 风险 | 应对 |
|------|------|
| 测试覆盖不足 | 重点覆盖 data/models/training 核心路径 |
| GitHub 推送失败 | 检查文件大小, 使用 .gitattributes LFS |
| 文档过期 | 文档与代码同步更新 |

---

## 四、时间节点总览

```
Week 1-2: 阶段 1  架构重构与代码质量升级
Week 2-3: 阶段 2  数据层增强与开源数据整合
Week 3-5: 阶段 3  模型升级与训练策略优化
Week 5-6: 阶段 4  TCM 单体筛选与机制阐释
Week 6-7: 阶段 5  验证、文档与部署上线
```

---

## 五、资源分配

| 资源 | 说明 |
|------|------|
| 计算资源 | CPU: 多核(并发评估); GPU: NVIDIA CUDA (训练大图) |
| 存储资源 | 项目目录 + 缓存目录 + 外部数据目录 |
| 数据资源 | 现有 network_files/, L3/, L1/ + GitHub 开源补充 |
| 人力资源 | AI 助手主导, 用户关键决策 |
| 工具资源 | ruff, pytest, PyTorch Geometric, DGL-LifeSci, RDKit, MCP 工具 |

---

## 六、风险评估与应对策略

| 风险类别 | 具体风险 | 可能性 | 影响 | 应对策略 |
|----------|----------|--------|------|----------|
| 数据风险 | 外部数据下载失败/限流 | 中 | 中 | 本地缓存 + 多源备份 + 记录日志 |
| 技术风险 | 模型迁移后性能下降 | 中 | 高 | 保留 legacy 对比 + 回归测试 |
| 性能风险 | 图构建耗时过长 | 高 | 高 | 图构建缓存 + 增量更新 |
| 质量风险 | 测试覆盖不足 | 中 | 中 | 核心模块 pytest 覆盖 ≥80% |
| 合规风险 | 数据版权/学术不端 | 低 | 高 | 记录来源、不调整 bio 先验权重 |
| 部署风险 | GitHub 大文件限制 | 中 | 中 | .gitignore + LFS + 清理 |

---

## 七、验收标准

### 7.1 功能验收

- [ ] 运行 `python run_hgt_pipeline.py --config config.yaml` 完成训练
- [ ] ACSL4 在 hub 排名中进入 top-K (K ≤ 30)
- [ ] BCP-ACSL4 结合概率排名进入 top-N (N ≤ 30)
- [ ] 生成 TCM 单体推荐表

### 7.2 质量验收

- [ ] `ruff check .` 关键文件 0 错误
- [ ] `python validate_inputs.py` 全部通过
- [ ] `python test_config_loading.py` 通过
- [ ] `python test_module3.py` 通过
- [ ] 新增核心模块单元测试 ≥80% 覆盖

### 7.3 文档验收

- [ ] trae_upgrade_roadmap.md 已更新并执行
- [ ] CHANGELOG.md 记录 v3.0.0 变更
- [ ] DEPLOY.md 包含环境安装与运行命令
- [ ] dry_lab_report.md 更新为升级后结果

### 7.4 部署验收

- [ ] 代码提交到 GitHub
- [ ] 仓库无大文件 (>100MB)
- [ ] .gitignore 正确配置

---

## 八、执行纪律

1. **零造假铁律**: 不生成/模拟数据, 所有数据从真实文件读取.
2. **异常传播**: 不使用 `try-except: pass`, 必须 `traceback.print_exc()` 并传播.
3. **缺失数据日志**: 缺失数据必须写日志警告, 不准静默补零.
4. **先验证后运行**: 每次运行核心模块前必须运行 `python validate_inputs.py`.
5. **修改后必 lint**: 每次修改 Python 文件后必须运行 `ruff check .`.
6. **持续监控**: 长时间任务使用 `CheckCommandStatus` 轮询.
7. **真实进度报告**: 每轮回复报告已完成、正在运行、卡住的环节.

---

## 九、附录: 关键命令速查

```bash
# 静态检查
ruff check .

# 输入验证
python validate_inputs.py

# 配置加载测试
python test_config_loading.py

# 模块 3 回归测试
python test_module3.py

# 训练入口
python run_hgt_pipeline.py --config config.yaml --device cuda

# 并发资源评估
python concurrency_utils.py

# 生成网络文件
python generate_all_network_files.py
```

---

*本文件为指导项目升级改造的活的文档, 各阶段执行过程中可根据实际进展迭代更新.*
