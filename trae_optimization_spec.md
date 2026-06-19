> 该方案由项目规则（.trae/rules/project_rules.md）与项目记忆（project_memory.md）动态引用，
> 请与 `mcp_project_server.py`、`project_commands.ps1`、`.trae/skills/iron-aging-workflow/SKILL.md` 一并阅读。

# Trae 优化方案总规：铁衰老项目

## 1. 设计目标

针对当前 Trae 在铁衰老项目中的使用痛点，建立一套**五位一体**的优化体系：

1. **MCP（主控制面板）**：把项目常用操作（lint、验证、测试、状态查询）封装为标准 MCP 工具，供 AI 直接调用。
2. **规则（Rules）**：把零造假铁律、硬约束、强制行为、工具路由写入 `.trae/rules/project_rules.md`，让 Trae 每次交互前自动加载。
3. **技能（Skills）**：创建 `iron-aging-workflow` skill，作为项目级入口协调器，负责任务分发、合规检查与进度报告。
4. **命令（Commands）**：创建 `project_commands.ps1`，把常用命令封装为可复用函数，供用户与 AI 在终端中一键执行。
5. **并发机制（Concurrency）**：通过 `concurrency_utils.py` 提供资源感知的进程/线程池，在质量门禁、批量文件检查等独立任务上实现安全并行，遵循“适度并发”原则。

五者协同关系如下图所示：

```
用户请求
   │
   ▼
.trae/rules/project_rules.md  （行为约束、零造假铁律）
   │
   ▼
iron-aging-workflow skill    （任务路由、SOP、决策树）
   │
   ├─── 生物机制/作图/通路/差异/单细胞/WGCNA/MR
   │        └── 调用对应 bio-* / r-plotting / ciri-* skill
   │
   ├─── Excel / GitHub / 序列分析
   │        └── 调用 mcp_Excel / mcp_GitHub / mcp_biotools
   │
   ├─── 项目静态检查 / 输入验证 / 回归测试
   │        └── 调用 mcp_project_server (MCP)
   │
   └─── 代码/安全审查
            └── 调用 TRAE-code-review / TRAE-security-review
   │
   ▼
project_commands.ps1         （人类与 AI 共用的终端命令）
   │
   ├─ 串行质量门禁
   └─ 并发质量门禁（调用 concurrency_utils.py）
   │
   ▼
真实代码运行 / 结果 / 日志
```

---

## 2. MCP（主控制面板）设计规范

### 2.1 服务器定位

- **名称**：`iron-aging-project-server`
- **实现**：`mcp_project_server.py`（基于 FastMCP + stdio）
- **配置**：`mcp_project_server_config.json`
- **原则**：只读取真实文件、只运行真实命令、不生成/模拟数据。

### 2.2 工具清单与接口

| 工具名 | 输入参数 | 输出 | 用途 |
|---|---|---|---|
| `run_ruff` | `target`（默认 `"."`） | success, returncode, stdout, stderr | 静态代码检查 |
| `run_tests` | `test_file`（白名单内） | success, returncode, stdout, stderr | 运行回归测试 |
| `validate_inputs` | 无 | success, returncode, stdout, stderr | 输入文件验证 |
| `read_config` | 无 | success, config, error | 读取 config.yaml |
| `check_file_exists` | `file_path` | exists, size_bytes, mtime, error | 文件存在性检查 |
| `log_missing_data_warning` | `file_path`, `reason`, `fallback_action` | success, log_file, error | 缺失数据日志 |
| `get_project_status` | 无 | success, status, error | 项目状态概览 |
| `run_whitelisted_command` | `command_name` | success, returncode, stdout, stderr | 运行白名单命令 |
| `get_system_resources` | `memory_reserve_mb`, `memory_per_worker_mb` | success, resources, error | 查询 CPU/内存与推荐 worker 数 |
| `run_parallel_file_checks` | `file_paths`, `max_workers` | success, completed, failed, results, errors | 并发检查多个文件是否存在 |
| `run_parallel_quality_gate` | `max_workers` | success, completed, failed, results, errors | 并发执行 lint + 验证 + tests |

### 2.3 安全与约束

- 所有文件路径限制在项目根目录内（通过 `Path.resolve().relative_to(PROJECT_ROOT)` 校验）。
- 命令执行使用白名单，禁止任意 shell 执行。
- `run_tests` 只能运行 `test_module3.py` 和 `test_config_loading.py`。
- `run_whitelisted_command` 仅开放 `git status --short`、`git diff`、`git log --oneline -5`。

### 2.4 注册方式

在 Trae 的 MCP 配置中引入 `mcp_project_server_config.json`：

```json
{
  "mcpServers": {
    "iron-aging-project-server": {
      "command": "python",
      "args": ["mcp_project_server.py"],
      "cwd": "c:\\Users\\Jy-Mentor-7\\Desktop\\铁衰老",
      "env": {"PYTHONIOENCODING": "utf-8"},
      "disabled": false
    }
  }
}
```

安装依赖：

```powershell
python -m pip install mcp pyyaml
```

---

## 3. 规则（Rules）设计规范

### 3.1 规则文件

- **路径**：`.trae/rules/project_rules.md`
- **加载时机**：Trae 在每次会话开始时自动读取。
- **与 project_memory.md 的关系**：规则文件约束即时行为；项目记忆保存跨会话经验教训。

### 3.2 规则章节

| 章节 | 内容 |
|---|---|
| 零造假铁律 | 禁止生成数据、捏造 API/论文/路径、静默吞错、跳过 QC、掩盖不足 |
| 项目硬约束 | 特征维度、数据来源、Bio 权重、CIRI 提取标准、PPI 过滤阈值 |
| 强制行为规则 | 能用 skill/MCP 不手写、GitHub 搜索用 mcp_GitHub、任务到可验证终态、持续监控、报告进度 |
| 智能路由表 | 任务 → 首选工具 → 完整流程 → 备选 → 禁止事项 |
| 工程约定 | 非监督边保留、缺失 L1 数据需日志、通路特征学习、废弃 TCM-only GraphCL |
| 代码质量门禁 | 修改后必跑 ruff / validate_inputs / test_config_loading / test_module3 |
| 项目命令速查 | 常用命令与运行时机 |
| 自我检查清单 | AI 每次操作前必须确认的 10 个问题 |

### 3.3 规则执行策略

- AI 在每次回复前应隐性执行自我检查清单。
- 若用户要求与规则冲突（如“把不足盖掉”），AI 必须拒绝并说明原因。
- 规则更新后，应同步更新 `project_memory.md` 与本总规。

---

## 4. 技能（Skills）设计规范

### 4.1 技能文件

- **路径**：`.trae/skills/iron-aging-workflow/SKILL.md`
- **触发关键词**：铁衰老、module2、module3、L1-L4、ACSL4、CIRI、铜死亡、铁死亡、BCP、GraphCL、HGT、WGCNA、validate_inputs、ruff
- **角色**：项目级入口协调器（orchestrator），不负责具体生物计算。

### 4.2 技能结构

1. **技能定位**：说明本 skill 是中央协调器。
2. **触发条件**：关键词列表。
3. **标准操作程序（SOP）**：接到任务后的起手式、修改代码后的收尾、长时间任务监控。
4. **工具选择决策树**：任务如何路由到不同 skill/MCP/命令。
5. **项目关键路径**：数据流与关键文件清单。
6. **反造假检查清单**：7 条必确认项。
7. **命令速查**：对应 `project_commands.ps1` 函数。
8. **并发处理规范**：适用/不适用场景、资源评估、错误处理。
9. **输出格式约定**：进度报告、代码修改报告格式。
10. **异常处理模板**：`traceback.print_exc()` + `raise`。
11. **与其他 skill 的关系**：明确本 skill 只做分发，不替代领域 skill。

### 4.3 技能开发标准

未来新增 skill 时应遵循：

- **单一职责**：一个 skill 解决一类问题，避免大杂烩。
- **触发明确**：必须有 `triggers` 或清晰的触发条件。
- **输入/输出可验证**：skill 内部调用的工具必须能返回可验证结果。
- **不违背零造假铁律**：skill 不能引导 AI 生成/模拟数据。
- **优先调用现有 skill/MCP**：避免重复实现。

---

## 5. 命令系统架构

### 5.1 命令文件

- **路径**：`project_commands.ps1`
- **加载方式**：`. ./project_commands.ps1`
- **设计原则**：幂等、可组合、返回布尔状态、输出彩色提示。

### 5.2 命令清单

| 命令 | 功能 | 对应 MCP 工具 |
|---|---|---|
| `Show-IronAgingHelp` | 帮助 | - |
| `Start-IronAgingLint` | ruff check | `run_ruff` |
| `Start-IronAgingValidation` | validate_inputs | `validate_inputs` |
| `Start-IronAgingConfigTest` | test_config_loading | `run_tests` |
| `Start-IronAgingModule3Test` | test_module3 | `run_tests` |
| `Start-IronAgingQualityGate` | lint + validation + tests | 组合调用 |
| `Start-IronAgingModule3` | 运行 module3_hgt.py | - |
| `Update-IronAgingNetworkFiles` | generate_all_network_files.py | - |
| `Get-IronAgingProjectStatus` | 项目状态 | `get_project_status` |
| `Get-IronAgingGitStatus` | git 状态 | `run_whitelisted_command` |
| `Install-IronAgingMcpDeps` | 安装 mcp/pyyaml | - |
| `Start-IronAgingParallelQualityGate` | 并发 lint + validation + tests | `run_parallel_quality_gate` |
| `Get-IronAgingSystemResources` | 显示 CPU/内存与推荐 worker 数 | `get_system_resources` |

### 5.3 命令与 MCP 的协同

- 人类用户可直接在 PowerShell 中使用 `project_commands.ps1`。
- AI 助手优先通过 MCP 工具调用，减少直接 shell 执行。
- 两者底层调用的是同一批脚本（如 `validate_inputs.py`），结果一致。

### 5.4 并发机制设计

#### 5.4.1 设计原则

- **适度并发**：根据 CPU 物理核心数与可用内存动态计算 `max_workers`，禁止无限制创建进程。
- **透明错误**：任何子任务异常都会记录并向上传播，禁止静默吞错。
- **任务隔离**：CPU 密集型任务使用进程池；IO 密集型任务使用线程池。
- **资源感知**：启用并发前必须调用 `get_system_resources()` 评估资源。

#### 5.4.2 核心实现

- **`concurrency_utils.py`**：
  - `get_system_resources()`：返回 CPU、内存与推荐 worker 数。
  - `recommend_workers()`：根据任务内存与系统资源计算 worker 数。
  - `run_tasks_parallel()`：通用并发执行器，支持进程/线程池与超时。
  - `parallel_file_checks()`：并发文件存在性检查。
  - `parallel_command_runners()`：并发运行外部命令。
  - `parallel_quality_gate()`：并发执行 lint + 验证 + tests。

#### 5.4.3 适用与不适用的场景

| 适用场景 | 工具 | 说明 |
|---|---|---|
| 批量文件存在性检查 | `run_parallel_file_checks` | IO 轻量，可并行 |
| lint + validate_inputs + test_config_loading + test_module3 | `run_parallel_quality_gate` | 彼此独立，快速暴露问题 |

| 不适用场景 | 原因 |
|---|---|
| 顺序依赖强的 pipeline | 如必须先 validate_inputs 再 module3 |
| GPU 训练任务 | 显存受限，不应与 CPU 密集型任务并行 |
| 单个大模型训练 | 本身已占满 GPU/CPU |

#### 5.4.4 协作流程

```
用户：并发运行质量门禁
 │
 ▼
iron-aging-workflow skill 触发
 │
 ├─ 调用 mcp_project_server.get_system_resources()
 ├─ 推荐 worker 数 = min(物理核心数, (可用内存 - 保留内存)/任务内存)
 ├─ 调用 mcp_project_server.run_parallel_quality_gate(max_workers=推荐值)
 │     ├─ 并发执行：ruff check .
 │     ├─ 并发执行：python validate_inputs.py
 │     ├─ 并发执行：python test_config_loading.py
 │     └─ 并发执行：python test_module3.py
 │
 ▼
报告结果：成功/失败任务数、错误日志
```

---

## 6. 组件接口与协作流程

### 6.1 典型任务：修改 module3_hgt.py

```
用户：优化 module3_hgt.py 的注意力机制
 │
 ▼
iron-aging-workflow skill 触发
 │
 ├─ 读取 .trae/rules/project_rules.md
 ├─ 读取 config.yaml
 ├─ 读取 module3_hgt.py
 │
 ▼
实施修改
 │
 ▼
调用 mcp_project_server.run_ruff         → 检查代码风格
调用 mcp_project_server.validate_inputs  → 验证输入文件
调用 mcp_project_server.run_tests        → 运行 test_module3.py
 │
 ▼
报告结果：lint/validation/test 状态
```

### 6.2 典型任务：搜索 GitHub 开源数据

```
用户：搜 GitHub 上最新的铁死亡相关数据集
 │
 ▼
iron-aging-workflow skill 触发
 │
 ▼
调用 mcp_GitHub.search_repositories(query="ferroptosis RNA-seq", perPage=10)
 │
 ▼
展示结果，等待用户选择
 │
 ▼
调用 mcp_GitHub.get_file_contents 下载选定文件
```

### 6.3 典型任务：缺失数据警告

```
代码运行时发现 L3/L1_genome_wide_de.csv 不存在
 │
 ▼
调用 mcp_project_server.log_missing_data_warning(
    file_path="L3/L1_genome_wide_de.csv",
    reason="文件不存在，无法提取 CIRI 疾病-基因关联",
    fallback_action="使用零向量并停止相关边构建"
)
 │
 ▼
抛出异常，向上传播，不静默补零
```

---

## 7. 验证与验收标准

### 7.1 单元验证

- [ ] `ruff check mcp_project_server.py concurrency_utils.py` 通过。
- [ ] `python -c "from mcp_project_server import mcp; print('OK')"` 可加载。
- [ ] `python concurrency_utils.py` 可正确输出系统资源与推荐 worker 数。
- [ ] `. ./project_commands.ps1` 在 PowerShell 中加载无报错。
- [ ] `Get-IronAgingProjectStatus` 显示关键文件状态。

### 7.2 集成验证

- [ ] 在 Trae 中加载 MCP 配置后，11 个工具可被识别。
- [ ] 修改任意 Python 文件后，AI 自动调用 `run_ruff` 与 `validate_inputs`。
- [ ] 触发 `iron-aging-workflow` skill 关键词时，AI 优先按 SOP 执行。

### 7.3 合规验证

- [ ] 所有组件不生成/模拟数据。
- [ ] 异常处理使用 `traceback.print_exc()` 并向上传播。
- [ ] 缺失数据通过 `log_missing_data_warning` 记录。

---

## 8. 维护与演进

- **版本控制**：所有规则/skill/MCP/命令文件纳入 Git 管理。
- **变更流程**：修改规则 → 同步更新 `project_memory.md` 与 `trae_optimization_spec.md` → 跑通 quality gate。
- **扩展指南**：新增 skill 时，需在 `iron-aging-workflow` skill 的决策树中注册路由；新增 MCP 工具时，需更新本总规与 `mcp_project_server_config.json` 的 `autoApprove` 列表。

---

## 9. 文件清单

| 文件 | 作用 |
|---|---|
| `.trae/rules/project_rules.md` | Trae 即时行为规则 |
| `.trae/skills/iron-aging-workflow/SKILL.md` | 项目入口协调 skill |
| `mcp_project_server.py` | 项目 MCP 服务器实现 |
| `mcp_project_server_config.json` | MCP 注册配置 |
| `project_commands.ps1` | PowerShell 命令集 |
| `concurrency_utils.py` | 资源感知并发执行器 |
| `trae_optimization_spec.md` | 本总规文档 |
| `project_memory.md` | 跨会话项目记忆 |
