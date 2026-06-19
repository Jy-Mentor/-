# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/) 和 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范.

## [3.1.0] - 2026-06-19

### 化合物-靶点数据增强 (ChEMBL)
- 新增 `fetch_chembl_compound_targets.py`, 通过 ChEMBL Web Resource Client 下载 63 个化合物的生物活性记录.
- 解析成功 61/63 个分子, 获取 11,319 条原始活性记录; 经 pChEMBL ≥ 5 / standard_value ≤ 10,000 nM 过滤后得到 180 条有效化合物-靶点边.
- 合并原有 curated 数据与 ChEMBL 数据, `network_files/compound_target_edges.csv` 正样本边从 91 条增至 268 条, 稀疏度从 ~0.5% 显著下降.
- `network_files/chembl_compound_targets.csv` 保存原始 ChEMBL 靶点与活性信息, 含 target_chembl_id、standard_type、standard_value、pchembl_value 等字段.

### 模型效果
- `module3_hgt.py` 重新训练后, 测试集化合物-靶点 (ct) AUC 从 ~0.52 提升至 0.635, 平均 AUC 提升至 0.785.

### 工程与质量
- 新增脚本通过 `ruff check fetch_chembl_compound_targets.py`.
- `validate_inputs.py` 54/54 项检查通过, `compound_target_edges.csv` 记录数校验为 268.
- `test_config_loading.py` 通过, 确认 63 个化合物配置完整.

## [3.0.0] - 2026-06-19

### 架构升级 (Modularization)
- 新建 `src/iron_aging` 模块化包, 按 data / models / training / evaluation / utils / apps 分层.
- 实现配置中心 `src/iron_aging/config.py`, 统一从 `config.yaml` 加载超参数与路径.
- 封装数据层 `src/iron_aging/data/graph_builder.py`, 过渡复用 legacy `module3_hgt.py` 图构建逻辑.
- 新增图构建缓存机制 (pickle + 输入哈希), 二次构建从 >120s 降至 <0.01s.
- 拆分模型层 `src/iron_aging/models/gat_encoder.py`、`hgt_encoder.py` 与 `link_predictor.py`, 独立 GATv2/HGT 编码器与链路预测头.
- 拆分训练层 `src/iron_aging/training/`, 包含 `trainer.py`、`losses.py`、`negative_sampling.py`, 支持 BCE 损失、VIB KL 损失、负采样与早停.
- 拆分评估层 `src/iron_aging/evaluation/`, 包含 `metrics.py` 与 `explainability.py`, 支持 AUC/AP/分类指标与梯度边归因.
- 新增应用层 `src/iron_aging/apps/hgt_pipeline.py`, 统一训练/推理入口.
- 新增训练入口 `run_hgt_pipeline.py`, 支持 `--config` 与 `--clear-cache` 参数.
- 新增测试目录 `tests/`, 包含 `test_graph_builder.py`、`test_models.py`、`test_training.py`、`test_evaluation.py`.

### 数据层增强 (Phase 2)
- 化合物集扩展至 63 个 (20 核心 + 43 BBB/神经保护扩展), 化合物 SMILES 与物化性质完整覆盖.
- 化合物 PCA 目标维数由 15 降至 10, 降低小样本过拟合风险.
- ACSL4 口袋特征固定标记文献关键残基 Q302/A329/Q464, 并新增 `n_literature_hotspots_in_pocket` 计数.
- CIRI 疾病-基因关联加载增强: 优先 DisGeNET/OpenTargets 专用 CIRI 文件, 再回退通用 disease_gene_associations.csv.

### TCM 单体筛选与可解释性 (Phase 4)
- 新增 `run_phase4_tcm_screening.py`, 实现化合物-ACSL4 靶点排名、BBB 透过率对比、通路可解释性分析.
- 生成 `L3_results/phase4_compound_target_ranking.csv` 与 `TCM_monomer_recommendation.xlsx`.
- 支持空通路归属的鲁棒处理, 避免 KeyError 中断流程.

### 修复
- 修复图构建缓存失效问题: 排除构建过程中会生成的输出文件 (compound_attentivefp_embeddings.csv 等), 避免每次构建后缓存立即失效.
- 修复 `run_phase4_tcm_screening.py` 中文标点导致的 SyntaxError.
- 修复 ACSL4 通路归属为空时的排序 KeyError.
- 将 `test_module3.py` 的图构建调用迁移至缓存封装, 回归测试耗时从 >120s 降至 ~150s.

### 文档
- 新增 `trae_upgrade_roadmap.md`, 定义从项目启动到部署上线的完整升级改造技术路线.
- 新增 `DEPLOY.md`, 包含环境安装、训练推理命令与结果文件说明.

## [2.0.0] - 2026-06-19

### 重大变更 (Breaking Changes)
- 弃用 12-分子 TCM-only GraphCL 预训练方案, 改为基于 MoleculeNet BBBP + ChEMBL 子集的真实大规模 AttentiveFP 预训练.
- 化合物集由 8 个扩展至 63 个 (20 核心验证集 + 43 BBB/铁死亡/神经保护扩展集).
- 化合物特征维度从 10 维重构为 6 描述符 + 动态 PCA 指纹 + 4 密度 + 64 AttentiveFP.
- 恢复 bio-prior 权重为原始 0.40/0.20/0.40, 移除人工干预特定基因排名的风险.
- ACSL4 口袋特征改为使用 PDB 5W8I / AlphaFold AF-Q6P1M0-F1-model_v4 的真实 17 维结构特征.

### 新增
- 新增 L2 双模块联合分析脚本 `l2_integrated_analysis.py`, 协同调用:
  - `L2_ferroptosis_vs_isp_wgcna.py` (大样本 WGCNA + ACSL4/GPX4 趋势)
  - `module2_sc.py` (单细胞脑 I/R 铁衰老细胞图谱)
- 新增输入多维验证系统 `validate_inputs.py`, 覆盖 L1/L2/L3/L4/检查点.
- 新增统一 L2 配置加载器 `_l2_config.py`, 支持环境变量覆盖, 消除硬编码绝对路径.
- 新增 `ruff.toml` 静态分析配置, 行宽 120, 统一代码风格.
- 新增 CHANGELOG.md 与 VERSION 文件, 建立语义化版本管理.
- 新增 DisGeNET/OpenTargets 疾病-基因关联补充流程, 支持无机构邮箱下载.

### 修复
- 修复 `module3_hgt.py` 未导入 `sys` 导致的 `F821` 致命错误.
- 修复 `L2_ferroptosis_vs_isp_wgcna.py` 未定义 `L1_OUTPUT_DIR` 导致的 `F821` 致命错误.
- 修复 `module3_pretrain_tcm.py` GraphCL 增强子中的索引越界与边属性维度错误.
- 修复 OpenTargets GraphQL 查询字段兼容性问题.
- 修复 PubChem SMILES 回退字段与重复 CID 错误.
- 修复 `validate_inputs.py` 中 L1/STRING/TRRUST 列名不匹配问题.

### 优化
- 对 6 个核心 Python 模块执行 `ruff check` + `ruff format`, 清除 E/W/F/I 类问题.
- 用 `current_line` 替换单字母变量 `l`, 提升可读性.
- 拆分超长 docstring 与 f-string, 满足 120 字符行宽.
- 优化 AttentiveFP 分子图构建, 提升 63 个化合物处理效率.

### 迁移指南
- 若之前依赖 12-分子 TCM 预训练权重, 请删除 `checkpoints/pretrained_attentivefp_tcm.pt`,
  运行 `python module3_pretrain_tcm.py` 重新生成 MoleculeNet/ChEMBL 预训练权重.
- 化合物 CSV 文件已自动扩展, 无需手动修改.
- 环境变量 `IRON_GEO_DIR` 与 `RSCRIPT` 可覆盖 `config.yaml` 中的 L2 路径.

## [1.0.0] - 2026-06-18

### 新增
- 初始项目结构: L1 双评分贝叶斯元分析、L2 多组学分析、L3 HGT-GAT 网络模型、L4 药物指纹.
- 8 个核心化合物的分子表征与网络构建.
- 基于 GAT/AttentiveFP/HGT 的多模态异构网络药物-靶点预测框架.
