# 铁衰老 × CIRI 系统全面技术评估报告

> 评估范围：c:\Users\Jy-Mentor-7\Desktop\铁衰老
> 评估日期：2026-06-19
> 评估版本：v3.1.0 → 计划重构至 v4.0.0

---

## 一、评估结论（Executive Summary）

**判定：存在重大技术债务与架构缺陷，建议启动系统性重构（非一次性重写）。**

当前系统已完成科学研究目标验证（GAT-HGT 异构图、化合物扩展、ChEMBL 数据增强），但在工程化层面仍停留在“脚本集合”阶段。核心模块 `module3_hgt.py` 为 **6,213 行单体文件**，承担图构建、特征工程、模型定义、训练、评估、可视化等全部职责；`src/iron_aging/` 模块化包虽已建立，但数据层严重缺失或不完整，无法独立支撑 `module3_hgt.py` 的运行。L1 模块代码质量差（52 个 lint 问题集中在一文件），无依赖清单文件，pickle 缓存存在安全隐患。这些问题已影响可维护性、可测试性、可扩展性与 CI 稳定性。

**重构策略**：采用“**增量替换 + 数据库化 + 接口先行**”，逐步将 `module3_hgt.py` 拆解到新的分层架构中，同时引入 PostgreSQL/MySQL 作为结构化主存储、MongoDB 作为文档/原始响应缓存，确保科学结果不变、工程底座升级。

---

## 二、功能板块现状评估

| 板块 | 当前状态 | 主要问题 | 风险等级 |
|---|---|---|---|
| **L1 差异表达与元分析** | 可运行，输出真实 DE/Meta 结果 | `l1_dual_analysis.py` 52 个 lint 问题；函数复杂度过高；大量延迟导入 | 高 |
| **L2 WGCNA/共表达** | R + Python 混合，可运行 | R 脚本与 Python 耦合松散；配置分散 | 中 |
| **Module2 单细胞** | `module2_sc.py` 可运行 | 依赖 Scanpy/ComBat，缺少 scVI/Harmony 等现代批次校正 | 中 |
| **L3 异质图建模** | 训练可收敛，ct AUC 0.635 | `module3_hgt.py` 单体 6,213 行；src/iron_aging/data/ 缺失；图构建首次 150s | **极高** |
| **L4 化合物特征** | 63 个化合物特征完整 | 指纹生成脚本在 L1，职责边界不清 | 中 |
| **Phase4 TCM 筛选** | 可生成排名 CSV | 排名结果对 BCP-ACSL4 支持弱（prob≈0.5） | 科学风险 |
| **配置中心** | `config.yaml` + `src/iron_aging/config.py` 已建立 | 部分模块仍直接读取 YAML 或硬编码常量 | 中 |
| **测试** | 14 个单元测试通过 | 未覆盖训练主循环、AttentiveFP、可解释性完整流程 | 高 |
| **部署/依赖** | DEPLOY.md 已存在 | `requirements.txt` 缺失；无 Docker/CI | 高 |
| **MCP/Tooling** | `mcp_project_server.py` 已存在 | 路径参数校验不足；缺少测试 | 中 |

---

## 三、架构设计评估

### 3.1 当前架构（实际运行态）

```
外部数据库 (ChEMBL/STRING/DisGeNET/PubChem/GEO...)
         │
         ▼
   network_files/*.csv   ← 手工/脚本生成的中间文件
         │
         ▼
   module3_hgt.py (6,213 行)
   ├─ 图构建
   ├─ 特征工程 (AttentiveFP, 指纹, 口袋)
   ├─ 模型定义 (GAT/HGT/LinkPredictor)
   ├─ 训练循环 (Multi-task + GradNorm + VIB)
   ├─ 评估与解释
   └─ 可视化
         │
         ▼
   L3_results/*.csv
```

### 3.2 目标架构（重构后）

```
┌─────────────────────────────────────────────────────────────┐
│ 应用层 (Apps/API)                                            │
│  CLI (run_hgt_pipeline.py) / FastAPI / MCP Server           │
├─────────────────────────────────────────────────────────────┤
│ 编排层 (Orchestration)                                       │
│  Pipeline DAG、实验追踪、缓存管理                            │
├─────────────────────────────────────────────────────────────┤
│ 评估层 (Evaluation)                                          │
│  Metrics / Explainability / Ranking / Reporting             │
├─────────────────────────────────────────────────────────────┤
│ 训练层 (Training)                                            │
│  Trainer / Losses / Negative Sampling / Early Stop           │
├─────────────────────────────────────────────────────────────┤
│ 模型层 (Models)                                              │
│  GATEncoder / HGTEncoder / LinkPredictor / VIB              │
├─────────────────────────────────────────────────────────────┤
│ 特征层 (Features)                                            │
│  CompoundEncoder / PocketEncoder / GeneFeatureBuilder       │
├─────────────────────────────────────────────────────────────┤
│ 图构建层 (Graph)                                             │
│  HeteroGraphBuilder (from DB) + Cache                       │
├─────────────────────────────────────────────────────────────┤
│ 数据层 (Data / ETL)                                          │
│  PostgreSQL (主存储) + MongoDB (原始文档) + 对象存储        │
│  ETL Sources: ChEMBL, STRING, DisGeNET, GEO, PubChem...     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 重大架构缺陷

1. **单体核心模块**：`module3_hgt.py` 6,213 行，违反单一职责原则，任何小改动都可能影响全局。
2. **数据层缺失**：`src/iron_aging/data/` 目录不存在或不完整，图构建逻辑仍在 `module3_hgt.py` 中，导致 `src/iron_aging/` 包无法独立使用。
3. **文件系统作为数据库**：所有中间状态以 CSV 文件形式存在，缺乏事务、版本、索引和并发控制。
4. **配置与代码混合**：`module3_hgt.py` 中仍有大量模块级常量（`ATTENTIVEFP_DIM`、`PRETRAIN_CKPT` 等）。
5. **R/Python 混合边界不清**：L2 的 WGCNA、单细胞部分依赖 R 脚本，但调用方式不一致。

---

## 四、代码质量评估

### 4.1 Ruff 检查结果

- 总错误数：**52**
- 问题集中文件：`L1/l1_dual_analysis.py`（~45 个，占 87%）
- 主要问题类型：
  - E501 行宽超限（19）
  - W293 空行空白（10）
  - E741 变量名 `l`（6）
  - F841 未使用局部变量（6）
  - E702 分号多语句（3）
  - C901 `main()` 复杂度 84 > 55（1）

### 4.2 关键代码坏味道

| 坏味道 | 位置 | 影响 |
|---|---|---|
| 巨型函数 | `module3_hgt.py:main()` 复杂度 84 | 难以测试、难以维护 |
| 模块级副作用 | `module3_hgt.py:523-526` 导入即读文件 | 测试困难、循环导入风险 |
| 全局可变状态 | `FERROAGING_GENES = None` 后赋值 | 并发不安全、行为不可预测 |
| 函数内延迟导入 | `module3_hgt.py` 9 处 | 反映模块拆分不清 |
| 硬编码绝对路径 | `generate_all_network_files.py:58` | 跨平台失败 |
| lambda 赋值 | `L1/drug_fingerprint_generator.py` | 可读性差 |
| 循环内 O(n·m) 扫描 | `module3_hgt.py:2114-2127` | 性能差 |
| 宽泛异常捕获 | `module3_hgt.py` 多处 `except Exception:` | 掩盖真实错误 |

---

## 五、性能评估

| 指标 | 现状 | 评价 |
|---|---|---|
| 图构建首次耗时 | ~150s | 阻塞 CI/开发 |
| 图构建缓存后 | <0.01s | 良好 |
| 训练 200 epoch | GPU 约 23s / CPU 约 17s（100 epoch 基准） | 小图 GPU 无优势 |
| 测试耗时 | pytest 14 个用例约 154s | 图构建是瓶颈 |
| 主要性能瓶颈 | `module3_hgt.py` 中 O(n·m) 注释计数、重复 CSV 读取、`nx.degree` 逐节点调用 | 可优化 |

---

## 六、安全评估

| 风险 | 位置 | 等级 | 说明 |
|---|---|---|---|
| pickle 反序列化 | `src/iron_aging/data/graph_builder.py` 缓存 | 中 | 缓存文件被篡改可导致 RCE |
| `torch.load` 无 `weights_only` | `module3_hgt.py:1564, 1598` | 中 | PyTorch 权重文件可能被篡改 |
| YAML 加载 | `_expand_compounds.py:492` | 低-中 | 默认 safe，但写法易误改 |
| MCP 路径未校验 | `mcp_project_server.py:57, 67` | 低 | 可指向任意路径执行 ruff |
| 外部 URL 请求 | `module3_hgt.py:1307` | 低 | 无超时/重试/校验 |

**结论**：无严重漏洞，但缓存与模型加载存在反序列化风险，需在重构中消除。

---

## 七、可扩展性评估

| 扩展场景 | 当前难度 | 原因 |
|---|---|---|
| 新增化合物 | 中 | 需修改多个 CSV 并重新生成指纹/嵌入 |
| 新增疾病 | 高 | 疾病-基因关联需重新抓取和合并 |
| 新增边类型 | 高 | `module3_hgt.py` 中边构建逻辑硬编码 |
| 新增数据源 | 中 | 每个新源需新增独立脚本 |
| 横向扩展（多用户/API） | 不可行 | 无服务化层，文件系统无法并发 |

---

## 八、技术债务清单（优先级）

### P0 — 必须立即处理
1. `module3_hgt.py` 单体化拆分
2. `src/iron_aging/data/` 图构建层补齐
3. 引入结构化数据库（PostgreSQL/MySQL）替代文件系统作为中间存储
4. 修复 `L1/l1_dual_analysis.py` lint 问题
5. 创建 `requirements.txt` / `pyproject.toml`

### P1 — 近期处理
6. 消除模块级副作用与全局可变状态
7. 统一配置读取入口（`iron_aging.config`）
8. 治理 `except Exception:`，改为具体异常
9. 消除硬编码路径
10. 图构建性能优化（O(n·m) → 向量化）

### P2 — 中期处理
11. 增加训练主循环、AttentiveFP、解释性流程测试覆盖
12. 替换 pickle 缓存为安全格式（torch/Parquet/MSGPack）
13. 引入 MongoDB 缓存原始 API 响应
14. 增加 MPS / 多设备支持

### P3 — 长期
15. 服务化：FastAPI + Celery
16. CI/CD：GitHub Actions
17. Docker 化部署
18. 实验追踪：MLflow / Weights & Biases

---

## 九、重构原则

1. **科学结果不变**：重构不改变现有 network_files 与 L3_results 的核心计算逻辑，先搬运再优化。
2. **数据库优先**：所有实体（gene/compound/disease/pathway/celltype/experiment/edge）先入 PostgreSQL，CSV 作为导出/备份。
3. **接口先行**：定义清晰的 Repository / Service / Pipeline 接口，再填充实现。
4. **增量替换**：`module3_hgt.py` 不一次性删除，而是逐步将其功能迁移到新模块，保留旧入口直至新入口验证通过。
5. **零造假**：所有数据必须从真实文件/数据库读取，禁止模拟或硬编码。
6. **测试驱动**：每个新模块必须附带单元测试，核心路径覆盖率目标 ≥80%。

---

## 十、总体建议

当前系统已达到“科研可用”状态，但远未达到“工程可维护、生产可部署”状态。建议：

1. 启动 **v4.0 重构项目**，目标是在 4–6 周内完成数据库化与核心模块拆分；
2. 优先解决 **P0 债务**，使 `src/iron_aging/` 包能够独立运行完整 pipeline；
3. 引入 **PostgreSQL + MongoDB** 作为数据底座，CSV 仅作为与外部工具（R、Cytoscape）交换的导出格式；
4. 建立 **FastAPI 服务层**，为后续 Web UI、多用户协作、模型版本管理打基础；
5. 保留现有研究成果不变，以 **数据库迁移脚本** 方式将历史 CSV 数据导入新系统。
