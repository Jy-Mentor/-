# 铁衰老 × CIRI 系统 v4.0 重构技术路线图

> 版本：v4.0.0-dev  
> 目标：将现有脚本式系统重构为数据库驱动、分层清晰、可测试、可部署的 GAT-HGT 药物预测平台  
> 制定日期：2026-06-19  
> 承接文档：[technical_assessment_report_v4.md](./technical_assessment_report_v4.md)

---

## 一、重构目标与验收标准

### 1.1 总体目标

在保持现有科学结果（network_files、L3_results）可复现的前提下，完成以下升级：

1. **数据库化**：将核心实体与边关系从 CSV 文件迁移到 PostgreSQL/MySQL，原始 API 响应缓存到 MongoDB。
2. **架构分层**：建立 Data → Features → Graph → Models → Training → Evaluation → Apps 的清晰分层。
3. **拆分单体**：将 `module3_hgt.py`（6,213 行）按职责拆分到 `src/iron_aging/` 各层。
4. **接口规范**：定义 Repository、Service、Pipeline 接口，支持未来 FastAPI / CLI / MCP 多入口。
5. **质量门禁**：所有 Python 文件通过 `ruff check .`，核心模块单元测试覆盖率 ≥80%，`validate_inputs` 54/54 通过。
6. **依赖管理**：提供 `pyproject.toml` + `requirements.lock`，支持 Docker 部署。

### 1.2 验收标准

| 验收项 | 标准 |
|---|---|
| 数据库迁移 | 现有 network_files 全部 CSV 可通过脚本导入 PostgreSQL，且能导出回 CSV |
| 新图构建 | `src/iron_aging/data/graph_builder.py` 能从数据库构建 PyG HeteroData，结果与旧 `module3_hgt.py` 一致 |
| 训练入口 | `python -m iron_aging run train` 可完成 200 epoch 训练，ct AUC ≥ 0.635 |
| 静态检查 | `ruff check .` 0 error |
| 单元测试 | `pytest tests/` 全部通过，核心模块覆盖率 ≥80% |
| 输入验证 | `python validate_inputs.py` 54/54 通过 |

---

## 二、技术选型标准

### 2.1 选型原则

1. **真实可用**：优先选择已验证、社区活跃、文档完善的开源组件。
2. **生物信息学友好**：优先支持 Python/R 双生态、与 Bioconductor/PyG 生态兼容。
3. **渐进替换**：不因为引入新组件而破坏现有工作流。
4. **自托管优先**：核心数据存储使用可本地部署的数据库，降低外部依赖。
5. **License 合规**：避免 GPL 传染风险，优先 MIT/Apache/BSD。

### 2.2 技术栈

| 层级 | 选型 | 备选 | 说明 |
|---|---|---|---|
| 编程语言 | Python 3.11+ | Python 3.10 | 必需 |
| R 生态 | R 4.2+ | 无 | L2 WGCNA/GSEA 保留 |
| 结构化数据库 | **PostgreSQL 15+** | MySQL 8.0 | 主存储，支持 JSONB、数组类型 |
| 文档数据库 | **MongoDB 6+** | SQLite（轻量） | 缓存原始 API 响应、GEO 元数据 |
| ORM/迁移 | **SQLAlchemy 2.0 + Alembic** | Peewee + 手写迁移 | 类型安全、迁移可控 |
| 缓存 | **Redis 7+** | 本地 dict | 可选，用于图缓存、API 限流 |
| 对象存储 | 本地文件系统 | MinIO | 大文件（checkpoint、原始 GEO） |
| Web 框架 | **FastAPI** | Flask | 后续服务化 |
| 任务队列 | **Celery + Redis** | RQ | 异步 ETL、训练任务 |
| GNN 框架 | **PyTorch Geometric 2.3+** | DGL | 保留现有 HGT/GAT 实现 |
| 分子特征 | **RDKit + AttentiveFP** | DeepChem | 保留现有指纹与预训练 |
| 测试 | **pytest + coverage** | unittest | 必需 |
| 静态检查 | **ruff + mypy** | flake8 | 必需 |
| 容器化 | **Docker + docker-compose** | conda env | 部署 |
| 工作流 | **GitHub Actions** | 本地脚本 | CI/CD |

---

## 三、架构分层设计

### 3.1 目标目录结构

```
铁衰老/
├── src/iron_aging/
│   ├── __init__.py
│   ├── config.py                  # 配置中心
│   ├── cli.py                     # 统一命令行入口
│   ├── db/                        # 数据库层
│   │   ├── __init__.py
│   │   ├── connection.py          # 数据库连接管理
│   │   ├── models.py              # SQLAlchemy ORM 模型
│   │   ├── repositories.py        # Repository 模式封装
│   │   └── migrations/            # Alembic 迁移脚本
│   ├── etl/                       # 数据抽取层
│   │   ├── __init__.py
│   │   ├── base.py                # ETL Source 抽象基类
│   │   ├── chembl.py              # ChEMBL Web Resource Client
│   │   ├── string.py              # STRING PPI
│   │   ├── disgenet.py            # DisGeNET
│   │   ├── opentargets.py         # OpenTargets
│   │   ├── pubchem.py             # PubChem PUG-REST
│   │   ├── geo.py                 # GEO 数据下载
│   │   └── mongo_cache.py         # MongoDB 原始响应缓存
│   ├── data/                      # 数据层（原缺失）
│   │   ├── __init__.py
│   │   ├── graph_builder.py       # 从 DB 构建 PyG HeteroData
│   │   ├── loaders.py             # 特征/边加载器
│   │   ├── validators.py          # 数据验证
│   │   └── cache.py               # 安全图缓存（torch/Parquet）
│   ├── features/                  # 特征工程层（新增）
│   │   ├── __init__.py
│   │   ├── compounds.py           # 化合物指纹/AttentiveFP
│   │   ├── genes.py               # 基因特征（表达、中心性、通路）
│   │   ├── pockets.py             # ACSL4 口袋特征
│   │   └── cell_types.py          # 细胞类型特征
│   ├── models/                    # 模型层（已存在）
│   │   ├── gat_encoder.py
│   │   ├── hgt_encoder.py
│   │   ├── link_predictor.py
│   │   └── compound_encoder.py    # AttentiveFP 封装
│   ├── training/                  # 训练层（已存在）
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── negative_sampling.py
│   │   └── callbacks.py           # EMA/早停/课程学习
│   ├── evaluation/                # 评估层（已存在）
│   │   ├── metrics.py
│   │   ├── explainability.py
│   │   └── ranking.py             # 化合物/基因排名
│   ├── pipelines/                 # 编排层（新增）
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── train_pipeline.py
│   │   ├── infer_pipeline.py
│   │   └── tcm_screening_pipeline.py
│   ├── api/                       # 服务层（新增）
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app
│   │   ├── schemas.py             # Pydantic 模型
│   │   └── routers/
│   └── apps/                      # 应用入口（已存在）
│       └── hgt_pipeline.py        # 兼容旧入口
├── tests/                         # 测试（已存在，需扩展）
├── scripts/                       # 迁移与运维脚本
│   ├── migrate_csv_to_db.py
│   ├── init_db.py
│   └── backup_db.py
├── alembic/                       # 数据库迁移
├── docker/                        # Docker 配置
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                          # 文档
├── pyproject.toml
├── requirements.txt               # 新增
├── CHANGELOG.md
└── VERSION
```

### 3.2 分层职责

| 层级 | 职责 | 不允许做的事 |
|---|---|---|
| ETL | 从外部数据库/API 下载原始数据，清洗后写入 PostgreSQL/MongoDB | 直接修改 CSV 文件作为持久化 |
| DB | 定义实体关系、约束、索引；提供 Repository 接口 | 包含业务逻辑或模型训练代码 |
| Data | 从 DB 读取数据，构建 PyG HeteroData，管理缓存 | 直接调用外部 API |
| Features | 生成化合物/基因/口袋/细胞类型特征向量 | 直接操作图结构 |
| Models | 定义 GNN 编码器、链路预测头 | 包含数据加载或训练循环 |
| Training | 训练循环、损失计算、负采样、优化器调度 | 直接读取 CSV 文件 |
| Evaluation | 指标计算、可解释性、排名生成 | 修改模型参数 |
| Pipelines | 编排 ETL → Features → Graph → Train → Eval 全流程 | 实现具体算法 |
| API/CLI | 接收用户请求，调用 Pipeline/Service | 包含算法实现 |

---

## 四、数据流转机制

### 4.1 总体数据流

```
┌─────────────┐     ETL      ┌─────────────┐     ORM      ┌─────────────┐
│ 外部数据源  │ ────────────> │ PostgreSQL  │ ───────────> │ Repository  │
│ (API/文件)  │               │ 结构化存储   │              │ 接口        │
└─────────────┘               └─────────────┘              └──────┬──────┘
       │                                                          │
       │ 原始响应                                                  │ 查询
       ▼                                                          ▼
┌─────────────┐                                          ┌─────────────┐
│ MongoDB     │                                          │ Features /  │
│ 文档缓存    │                                          │ GraphBuilder│
└─────────────┘                                          └──────┬──────┘
                                                                │
                                                                ▼
                                                       ┌─────────────┐
                                                       │ PyG HeteroData│
                                                       └──────┬──────┘
                                                              │
                                ┌─────────────┐              │
                                │ L3_results  │ <────────────┘
                                │ (导出 CSV)  │        训练/推理/评估
                                └─────────────┘
```

### 4.2 核心实体关系（ER）

```
Gene (id, symbol, name, organism, synonyms, ...)
  │
  ├─< GeneDisease >─ Disease (id, name, doid, mesh, mondo, ...)
  ├─< PPIEdge >───── Gene (score, source)
  ├─< TFTargetEdge >─ Gene (tf_symbol, target_symbol, source)
  ├─< GenePathway >─ Pathway (id, name, source)
  ├─< CellTypeMarker >─ CellType (id, name, tissue)
  └─< GeneCoexpEdge >─ Gene (correlation, source)

Compound (id, name, smiles, pubchem_cid, chembl_id, ...)
  │
  ├─< CompoundTargetEdge >─ Gene (activity_type, value, unit, pchembl, source)
  ├─< CompoundFingerprint >─ FingerprintVector
  ├─< CompoundEmbedding >─ EmbeddingVector
  └─< CompoundProperty >─ Property (name, value)

Experiment / Dataset (id, geo_accession, organism, condition, platform, ...)
  │
  └─< DifferentialExpression >─ Gene (log2fc, padj, dataset)

Pocket (id, gene_symbol, pdb_id, residue_features_json, ...)
```

### 4.3 数据流关键节点

1. **原始数据入库**：ETL 脚本将外部数据写入 PostgreSQL；API 原始响应写入 MongoDB。
2. **特征生成**：Features 层从 DB 读取化合物 SMILES、基因表达、口袋残基，生成向量后写回 DB。
3. **图构建**：GraphBuilder 从 Repository 查询所有节点与边，构建 PyG HeteroData；结果缓存到安全格式（torch/Parquet）。
4. **训练/推理**：Training/Evaluation 层消费 HeteroData，输出模型权重与结果 CSV。
5. **结果导出**：关键结果（ranking、embeddings、metrics）写回 DB，同时导出 CSV 供 R/可视化使用。

---

## 五、接口规范定义

### 5.1 Repository 接口

```python
from abc import ABC, abstractmethod
from typing import Iterable

class BaseRepository(ABC):
    @abstractmethod
    def get_all(self) -> Iterable[dict]: ...
    @abstractmethod
    def get_by_id(self, id: str) -> dict | None: ...
    @abstractmethod
    def bulk_upsert(self, records: list[dict]) -> int: ...

class GeneRepository(BaseRepository): ...
class CompoundRepository(BaseRepository): ...
class EdgeRepository(BaseRepository):
    @abstractmethod
    def get_edges(self, edge_type: str) -> Iterable[tuple[str, str, dict]]: ...
```

### 5.2 ETL Source 接口

```python
class DataSource(ABC):
    name: str
    @abstractmethod
    def fetch(self, **params) -> list[dict]: ...
    @abstractmethod
    def normalize(self, raw: list[dict]) -> list[dict]: ...
    @abstractmethod
    def validate(self, records: list[dict]) -> bool: ...
```

### 5.3 Pipeline 接口

```python
class Pipeline(ABC):
    @abstractmethod
    def run(self, config: PipelineConfig) -> PipelineResult: ...

@dataclass
class PipelineConfig:
    experiment_name: str
    model_config: dict
    training_config: dict
    use_cache: bool = True

@dataclass
class PipelineResult:
    model_path: Path
    metrics: dict
    ranking_path: Path
```

### 5.4 API 接口（FastAPI）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/etl/run` | 触发指定数据源 ETL |
| GET | `/api/v1/compounds` | 列出化合物 |
| GET | `/api/v1/compounds/{id}/targets` | 查询化合物靶点 |
| POST | `/api/v1/pipelines/train` | 启动训练 pipeline |
| GET | `/api/v1/pipelines/{id}/status` | 查询任务状态 |
| POST | `/api/v1/screening/acsl4` | ACSL4 单体筛选 |
| GET | `/api/v1/health` | 健康检查 |

---

## 六、数据库 Schema 设计（PostgreSQL）

### 6.1 表清单

| 表名 | 说明 | 核心字段 |
|---|---|---|
| `genes` | 基因实体 | `id`, `symbol`, `name`, `organism`, `synonyms[]`, `created_at` |
| `compounds` | 化合物实体 | `id`, `name`, `canonical_smiles`, `pubchem_cid`, `chembl_id`, `inchi_key` |
| `diseases` | 疾病实体 | `id`, `name`, `doid`, `mesh_id`, `mondo_id` |
| `pathways` | 通路实体 | `id`, `name`, `source`, `external_id` |
| `cell_types` | 细胞类型 | `id`, `name`, `tissue`, `species` |
| `datasets` | GEO/实验数据集 | `id`, `accession`, `title`, `organism`, `platform`, `condition` |
| `ppi_edges` | PPI 边 | `protein_a_id`, `protein_b_id`, `score`, `source` |
| `tf_target_edges` | TF-靶点边 | `tf_id`, `target_id`, `source` |
| `compound_target_edges` | 化合物-靶点 | `compound_id`, `gene_id`, `activity_type`, `standard_value`, `standard_unit`, `pchembl_value`, `source` |
| `disease_gene_edges` | 疾病-基因 | `disease_id`, `gene_id`, `score`, `source` |
| `gene_pathway_edges` | 基因-通路 | `gene_id`, `pathway_id`, `source` |
| `cell_type_marker_edges` | 细胞类型标记 | `cell_type_id`, `gene_id`, `marker_type` |
| `ligand_receptor_edges` | 配体-受体 | `ligand_id`, `receptor_id`, `source` |
| `gene_coexp_edges` | 基因共表达 | `gene_a_id`, `gene_b_id`, `correlation`, `source` |
| `differential_expressions` | 差异表达 | `dataset_id`, `gene_id`, `log2fc`, `padj`, `stat` |
| `compound_properties` | 化合物物化性质 | `compound_id`, `property_name`, `property_value` |
| `compound_fingerprints` | 化合物指纹 | `compound_id`, `fingerprint_type`, `fingerprint_vector` |
| `compound_embeddings` | 化合物嵌入 | `compound_id`, `model_name`, `embedding_vector` |
| `pockets` | 蛋白口袋 | `id`, `gene_id`, `pdb_id`, `residue_features_json` |
| `experiments` | 训练实验 | `id`, `name`, `config_json`, `status`, `started_at`, `finished_at` |
| `experiment_metrics` | 实验指标 | `experiment_id`, `epoch`, `metric_name`, `metric_value` |
| `model_checkpoints` | 模型检查点 | `experiment_id`, `epoch`, `path`, `is_best` |

### 6.2 关键约束

- 所有实体表使用 `id` 主键，采用 UUID v4 或业务稳定 ID（如 gene symbol）。
- 边表使用复合主键 `(source_id, target_id, source)`，避免重复。
- `genes.symbol`、`compounds.name`、`diseases.name` 建立唯一索引（区分大小写）。
- `compound_target_edges` 建立 `(compound_id, gene_id)` 与 `(gene_id, compound_id)` 索引。
- `differential_expressions` 建立 `(dataset_id, gene_id)` 唯一索引。

### 6.3 MongoDB Collections

| Collection | 说明 |
|---|---|
| `chembl_responses` | ChEMBL API 原始响应 |
| `pubchem_responses` | PubChem PUG-REST 响应 |
| `geo_metadata` | GEO 平台/样本元数据 |
| `external_api_logs` | API 调用日志（URL、状态码、时间戳） |

---

## 七、性能优化策略

### 7.1 数据库性能

1. **索引**：所有外键、查询字段建立 B-tree 索引；`canonical_smiles`、`inchi_key` 建立唯一索引。
2. **批量写入**：ETL 使用 `COPY` 或 SQLAlchemy `bulk_save_objects`，单次批量 ≥1,000 条。
3. **连接池**：使用 SQLAlchemy `QueuePool`，pool_size=10，max_overflow=20。
4. **查询优化**：图构建时使用 `selectinload` 或原生 JOIN 一次性加载所需边。
5. **分区**：`differential_expressions` 按 `dataset_id` 分区，加速按数据集查询。

### 7.2 图构建性能

1. **缓存**：图缓存使用输入文件/数据库内容哈希，安全格式（torch/Parquet）。
2. **向量化**：GO/KEGG/InterPro 计数改用 `groupby().size()` 或 sparse matrix。
3. **避免重复读取**：GraphBuilder 一次性从 DB 加载所有边，按类型分组后构造 PyG。
4. **NetworkX 优化**：`nx.degree(G)` 一次性计算；betweenness centrality 使用近似或跳过。
5. **Lazy Loading**：特征只在需要时生成，结果写入 DB 复用。

### 7.3 训练性能

1. **设备选择**：自动检测 CUDA/MPS/CPU；小图默认 CPU。
2. **DataLoader**：如果图较大，使用 PyG NeighborLoader。
3. **混合精度**：可选 `torch.cuda.amp` 加速大模型训练。
4. **早停**：基于验证集 AUC 的早停，避免无效 epoch。
5. **EMA**：保留指数移动平均权重。

### 7.4 可扩展性

1. **水平扩展**：FastAPI + Celery 支持多 worker；PostgreSQL 主从分离。
2. **异步 ETL**：Celery task 处理耗时外部 API 调用。
3. **模型版本管理**：检查点与实验元数据存入 DB，支持 A/B 比较。
4. **插件化数据源**：新数据源只需实现 `DataSource` 接口并注册。

---

## 八、分阶段实施计划

### Phase 1：基础设施（Week 1）

- [ ] 创建 `pyproject.toml` 与 `requirements.txt`
- [ ] 搭建 `src/iron_aging/db/`：SQLAlchemy 模型、连接池、Alembic 初始化
- [ ] 搭建 `src/iron_aging/etl/`：抽象基类 + 一个示例实现（ChEMBL）
- [ ] 创建 `docker-compose.yml`（PostgreSQL + MongoDB + Redis）
- [ ] 编写 `scripts/init_db.py` 初始化数据库

### Phase 2：数据库迁移（Week 1–2）

- [ ] 编写 `scripts/migrate_csv_to_db.py`，将 network_files 全部 CSV 导入 PostgreSQL
- [ ] 验证导入后数据完整性（记录数、唯一约束、外键关系）
- [ ] 将 MongoDB 缓存接入 ETL 流程
- [ ] 更新 `validate_inputs.py` 支持数据库连接验证

### Phase 3：数据层重构（Week 2–3）

- [ ] 实现 `src/iron_aging/data/graph_builder.py`（从 DB 构建 HeteroData）
- [ ] 实现安全图缓存（torch/Parquet）替代 pickle
- [ ] 编写 `tests/test_graph_builder_db.py` 验证与旧逻辑一致性
- [ ] 实现 `src/iron_aging/data/loaders.py`

### Phase 4：特征层与模型层（Week 3–4）

- [ ] 迁移化合物特征生成到 `src/iron_aging/features/compounds.py`
- [ ] 迁移基因/口袋/细胞类型特征生成
- [ ] 将 `src/iron_aging/models/` 与 Features 层对接
- [ ] 增加 `compound_encoder.py` 封装 AttentiveFP

### Phase 5：训练与评估层（Week 4–5）

- [ ] 重构训练循环到 `src/iron_aging/training/trainer.py`
- [ ] 增加 callbacks（EMA、早停、课程学习、GradNorm）
- [ ] 扩展 `src/iron_aging/evaluation/` 的排名与报告功能
- [ ] 实现 `src/iron_aging/pipelines/train_pipeline.py`

### Phase 6：应用入口与服务化（Week 5–6）

- [ ] 统一 CLI：`python -m iron_aging`
- [ ] 搭建 FastAPI：`src/iron_aging/api/`
- [ ] 迁移 Phase4 TCM 筛选到 `src/iron_aging/pipelines/tcm_screening_pipeline.py`
- [ ] 编写 API 测试

### Phase 7：质量门禁与部署（Week 6）

- [ ] 全项目 `ruff check .` 清零
- [ ] 单元测试覆盖率 ≥80%
- [ ] GitHub Actions CI
- [ ] Docker 镜像构建与文档更新
- [ ] 版本升级至 v4.0.0，更新 CHANGELOG

---

## 九、风险与应对

| 风险 | 影响 | 应对策略 |
|---|---|---|
| 重构破坏现有结果 | 高 | 旧入口保留；新旧结果 diff 对比；科学指标不下降才替换 |
| 数据库迁移数据丢失 | 高 | 迁移前备份 CSV；迁移后校验记录数；支持回滚 |
| 外部 API 限流/变更 | 中 | MongoDB 缓存原始响应；ETL 支持增量更新与重试 |
| 性能不如预期 | 中 | 分阶段基准测试；保留 CSV 缓存路径作为 fallback |
| 学习曲线/时间不足 | 中 | 优先完成 P0；分阶段交付，每阶段可独立验证 |

---

## 十、与现有工作的衔接

1. **保留 `module3_hgt.py`**：在 v4.0 完成前作为 fallback 入口继续可用。
2. **保留 `network_files/`**：作为数据库初始化数据源，同时继续生成。
3. **保留 `validate_inputs.py`**：扩展为同时校验 CSV 和数据库状态。
4. **保留 `tests/`**：新测试加入 `tests/v4/`，旧测试逐步迁移而非删除。
5. **版本管理**：v4.0 开发在独立分支 `dev/v4.0` 进行，完成后合并至 master。

---

## 十一、GitHub 开源资源集成清单

| 领域 | 推荐项目 | 集成方式 |
|---|---|---|
| HGT/GNN | [mims-harvard/TxGNN](https://github.com/mims-harvard/TxGNN) | 借鉴零样本推理层 |
| HGT/GNN | [bcb-sut/HGTDR](https://github.com/bcb-sut/HGTDR) | 借鉴图构建与关系评分 |
| DTI | [samsledje/ConPLex](https://github.com/samsledje/ConPLex) | 蛋白语言模型嵌入作为靶点特征 |
| DTI | [kexinhuang12345/MolTrans](https://github.com/kexinhuang12345/MolTrans) | 子结构 Transformer 备选化合物编码 |
| 单细胞 | [scverse/scvi-tools](https://github.com/scverse/scvi-tools) | 批次校正升级 |
| 单细胞 | [sqjin/CellChat](https://github.com/sqjin/CellChat) | LR 数据库补充 |
| WGCNA | [smorabit/hdWGCNA](https://github.com/smorabit/hdWGCNA) | 单细胞 WGCNA 升级 |
| 分子特征 | [deepchem/deepchem](https://github.com/deepchem/deepchem) | 深度学习描述符扩展 |
| 分子对接 | [gcorso/DiffDock](https://github.com/gcorso/DiffDock) | 可选对接验证 |
| RNA-seq 流程 | [nf-core/rnaseq](https://github.com/nf-core/rnaseq) | 上游 FASTQ→count 参考 |

---

## 十二、总结

v4.0 重构的核心是“**用工程化底座支撑科学假设**”。通过引入 PostgreSQL + MongoDB 作为数据底座、拆分 `module3_hgt.py` 为分层架构、建立清晰的接口规范，系统将从“可运行的科研脚本集合”升级为“可维护、可扩展、可部署的药物预测平台”。重构过程必须遵循“科学结果不变、数据库优先、接口先行、增量替换、零造假、测试驱动”六大原则，确保每一步都可验证、可回滚。
