---
name: "claude-code"
description: "Claude Code agentic coding toolkit: code review, Git workflow automation, debugging, and security scanning. Invoke when user asks for code review, PR review, commit, branch management, debugging, security audit, or mentions Claude Code."
---

# Claude Code - Agentic Coding Toolkit

Claude Code 的智能编程工具包，集成代码审查、Git 工作流、调试助手和安全扫描四大核心能力。

## 1. 代码审查 (Code Review)

### 触发条件
- 用户要求 "review 代码"、"代码审查"、"code review"
- 用户要求审查 PR / Merge Request
- 用户提交代码前需要检查

### 审查流程

1. **分析代码差异**：使用 `git diff` 或 `git log` 获取变更内容
2. **结构化审查**，按以下维度逐一检查：
   - **正确性**：逻辑错误、边界条件、空值处理
   - **性能**：不必要的循环、重复计算、内存泄漏风险
   - **安全性**：注入风险、敏感信息泄露、权限问题
   - **可维护性**：命名规范、函数长度、代码重复、注释质量
   - **最佳实践**：语言/框架惯例、设计模式使用
3. **输出格式**：
   ```
   ## Code Review 报告
   
   ### 概要
   - 变更文件数：X
   - 严重问题：X | 建议：X | 优化：X
   
   ### 严重问题 (必须修复)
   - [文件路径:行号] 问题描述 + 修复建议
   
   ### 改进建议
   - [文件路径:行号] 建议描述
   
   ### 优化建议
   - [文件路径:行号] 优化方案
   ```

## 2. Git 工作流 (Git Workflow)

### 触发条件
- 用户要求 commit、提交代码、推送
- 用户要求创建分支、合并、PR
- 用户要求管理 Git 仓库

### Git 安全协议
- **永远不要** 修改 git config
- **永远不要** 运行 `push --force`、`reset --hard`、`checkout .`、`restore .`、`clean -f` 等破坏性命令（除非用户明确要求）
- **永远不要** force push 到 main/master
- 提交前使用 `git add` 指定具体文件，不要用 `git add -A` 或 `git add .`
- 不要在用户未明确要求的情况下提交代码

### 智能 Commit 流程
1. 运行 `git status` 查看变更
2. 运行 `git diff` 查看具体变更
3. 查看最近 5 条 `git log --oneline -5` 了解提交风格
4. 生成简洁的 commit message（1-2 句话，侧重 "为什么" 而非 "做了什么"）
5. 使用 HEREDOC 格式提交：
   ```bash
   git commit -m "$(cat <<'EOF'
   简短描述变更原因
   
   EOF
   )"
   ```

### 分支操作
- 创建分支：`git checkout -b feature/描述`
- 切换分支：`git checkout 分支名`
- 查看分支：`git branch -a`

## 3. 调试助手 (Debugger)

### 触发条件
- 用户报告 Bug、错误、异常
- 用户说 "调试"、"debug"、"排查问题"
- 代码运行异常需要定位原因

### 科学调试流程

1. **假设形成**：
   - 阅读错误信息和相关代码
   - 分析可能的根因（数据问题、逻辑问题、环境问题、依赖问题）
   - 列出假设列表（按可能性排序）

2. **插桩验证**：
   - 在关键路径添加日志输出
   - 使用断点或条件打印缩小范围
   - 每次只验证一个假设

3. **复现 → 分析 → 修复 → 验证**：
   - 确认可以稳定复现
   - 分析根因
   - 实施修复（最小改动原则）
   - 验证修复有效且不引入新问题

4. **注意事项**：
   - 先读代码，不要盲目猜测
   - 检查 linter 错误和类型错误
   - 检查依赖版本兼容性
   - 修改后运行相关测试确保无回归

## 4. 安全扫描 (Security Review)

### 触发条件
- 用户要求安全扫描、安全审查
- 用户部署前需要安全检查
- 用户提到 "security"、"漏洞"、"安全检查"

### 扫描维度

1. **代码安全**：
   - SQL 注入（拼接查询字符串）
   - XSS 跨站脚本（未转义的用户输入）
   - 命令注入（shell 拼接）
   - 路径遍历（未验证的文件路径）
   - 反序列化漏洞

2. **密钥和敏感信息**：
   - 硬编码的密码、API Key、Token
   - 日志中泄露的敏感数据
   - `.env` 文件是否在 `.gitignore` 中

3. **依赖安全**：
   - 检查 `package.json`/`requirements.txt` 中已知漏洞的依赖
   - 过时的依赖版本

4. **配置安全**：
   - CORS 配置是否过于宽松
   - 调试模式是否在生产环境关闭
   - 权限控制是否完善

### 输出格式
```
## 安全扫描报告

### 严重漏洞 (立即修复)
- [文件:行号] 漏洞描述 | 风险等级 | 修复方案

### 中危风险
- [文件:行号] 风险描述 | 修复建议

### 低危建议
- [文件:行号] 改进建议
```

## 综合使用

当用户说 "Claude Code" 或 "全面检查" 时，依次执行全部四个模块：
1. 代码审查
2. Git 状态检查
3. 调试/错误排查（如有错误）
4. 安全扫描
