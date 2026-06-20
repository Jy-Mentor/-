﻿﻿﻿# 铁衰老项目 - PowerShell 命令集
# 用法：在项目根目录的 PowerShell 中执行
#   . ./project_commands.ps1
#   Show-IronAgingHelp

$script:IronAgingRoot = $PSScriptRoot
if (-not $script:IronAgingRoot) {
    $script:IronAgingRoot = Get-Location | Select-Object -ExpandProperty Path
}

# ---------------------------------------------------------------------------
# 帮助信息
# ---------------------------------------------------------------------------
function Show-IronAgingHelp {
    <#
    .SYNOPSIS
        显示铁衰老项目可用命令列表。
    #>
    Write-Host "铁衰老项目命令集" -ForegroundColor Cyan
    Write-Host "================" -ForegroundColor Cyan
    Write-Host "Show-IronAgingHelp                 显示本帮助"
    Write-Host "Start-IronAgingLint                运行 ruff 静态检查"
    Write-Host "Start-IronAgingValidation          运行 validate_inputs.py"
    Write-Host "Start-IronAgingConfigTest          运行 test_config_loading.py"
    Write-Host "Start-IronAgingModule3Test         运行 test_module3.py"
    Write-Host "Start-IronAgingQualityGate         依次运行 lint + validation + tests"
    Write-Host "Start-IronAgingParallelQualityGate 并发执行 lint + validation + tests"
    Write-Host "Start-IronAgingModule3             运行 module3_hgt.py"
    Write-Host "Update-IronAgingNetworkFiles       运行 generate_all_network_files.py"
    Write-Host "Get-IronAgingProjectStatus         显示项目关键文件/目录状态"
    Write-Host "Get-IronAgingSystemResources       显示系统资源与并发推荐"
    Write-Host "Get-IronAgingGitStatus             显示 git 状态"
    Write-Host "Install-IronAgingMcpDeps           安装 MCP 服务器依赖 (mcp, pyyaml)"
    Write-Host ""
    Write-Host "加载方式：. ./project_commands.ps1" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 内部辅助：运行 Python 脚本并输出结果
# ---------------------------------------------------------------------------
function Invoke-IronAgingPythonScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptName,

        [string]$Description = "执行脚本"
    )

    $fullPath = Join-Path $script:IronAgingRoot $ScriptName
    if (-not (Test-Path $fullPath)) {
        Write-Host "[ERROR] 找不到脚本：$fullPath" -ForegroundColor Red
        return $false
    }

    Write-Host "[$Description] python $ScriptName" -ForegroundColor Cyan
    Push-Location $script:IronAgingRoot
    try {
        python $ScriptName
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -eq 0) {
        Write-Host "[$Description] 成功" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "[$Description] 失败 (exit code: $exitCode)" -ForegroundColor Red
        return $false
    }
}

# ---------------------------------------------------------------------------
# 1. 静态检查
# ---------------------------------------------------------------------------
function Start-IronAgingLint {
    <#
    .SYNOPSIS
        在项目根目录运行 ruff check .
    #>
    Write-Host "[Lint] ruff check ." -ForegroundColor Cyan
    Push-Location $script:IronAgingRoot
    try {
        python -m ruff check .
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -eq 0) {
        Write-Host "[Lint] 通过" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "[Lint] 发现 $exitCode 个问题" -ForegroundColor Red
        return $false
    }
}

# ---------------------------------------------------------------------------
# 2. 输入验证
# ---------------------------------------------------------------------------
function Start-IronAgingValidation {
    <#
    .SYNOPSIS
        运行 validate_inputs.py 验证项目输入文件。
    #>
    return Invoke-IronAgingPythonScript -ScriptName "validate_inputs.py" -Description "输入验证"
}

# ---------------------------------------------------------------------------
# 3. 配置加载测试
# ---------------------------------------------------------------------------
function Start-IronAgingConfigTest {
    <#
    .SYNOPSIS
        运行 test_config_loading.py。
    #>
    return Invoke-IronAgingPythonScript -ScriptName "test_config_loading.py" -Description "配置测试"
}

# ---------------------------------------------------------------------------
# 4. module3 回归测试
# ---------------------------------------------------------------------------
function Start-IronAgingModule3Test {
    <#
    .SYNOPSIS
        运行 test_module3.py。
    #>
    return Invoke-IronAgingPythonScript -ScriptName "test_module3.py" -Description "module3 回归测试"
}

# ---------------------------------------------------------------------------
# 5. 完整质量门禁
# ---------------------------------------------------------------------------
function Start-IronAgingQualityGate {
    <#
    .SYNOPSIS
        依次执行 lint、输入验证、配置测试、module3 测试。
        任一环节失败即停止并返回 $false。
    #>
    $results = @(
        (Start-IronAgingLint),
        (Start-IronAgingValidation),
        (Start-IronAgingConfigTest),
        (Start-IronAgingModule3Test)
    )

    if ($results -contains $false) {
        Write-Host "`n[Quality Gate] 未通过" -ForegroundColor Red
        return $false
    }
    else {
        Write-Host "`n[Quality Gate] 全部通过" -ForegroundColor Green
        return $true
    }
}

# ---------------------------------------------------------------------------
# 6. 运行 module3 HGT 训练
# ---------------------------------------------------------------------------
function Start-IronAgingModule3 {
    <#
    .SYNOPSIS
        运行 module3_hgt.py。
        注意：运行前请确保已通过 validate_inputs.py。
    #>
    return Invoke-IronAgingPythonScript -ScriptName "module3_hgt.py" -Description "module3 HGT 训练"
}

# ---------------------------------------------------------------------------
# 7. 重新生成网络文件
# ---------------------------------------------------------------------------
function Update-IronAgingNetworkFiles {
    <#
    .SYNOPSIS
        运行 generate_all_network_files.py。
    #>
    return Invoke-IronAgingPythonScript -ScriptName "generate_all_network_files.py" -Description "生成网络文件"
}

# ---------------------------------------------------------------------------
# 8. 项目状态概览
# ---------------------------------------------------------------------------
function Get-IronAgingProjectStatus {
    <#
    .SYNOPSIS
        显示项目关键文件与目录的存在状态。
    #>
    Write-Host "铁衰老项目状态概览" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    Write-Host "项目根目录：$script:IronAgingRoot"

    $keyFiles = @(
        "config.yaml",
        "ruff.toml",
        "validate_inputs.py",
        "module3_hgt.py",
        "test_module3.py",
        "test_config_loading.py",
        "mcp_project_server.py",
        "project_commands.ps1"
    )

    $keyDirs = @(
        "L1",
        "L2_WGCNA_input",
        "L3",
        "network_files",
        "L3_results",
        "external_data",
        "checkpoints"
    )

    Write-Host "`n关键文件：" -ForegroundColor Yellow
    foreach ($file in $keyFiles) {
        $fullPath = Join-Path $script:IronAgingRoot $file
        $exists = Test-Path $fullPath
        $color = if ($exists) { "Green" } else { "Red" }
        $status = if ($exists) { "存在" } else { "缺失" }
        Write-Host "  $file : $status" -ForegroundColor $color
    }

    Write-Host "`n关键目录：" -ForegroundColor Yellow
    foreach ($dir in $keyDirs) {
        $fullPath = Join-Path $script:IronAgingRoot $dir
        $exists = Test-Path $fullPath -PathType Container
        $color = if ($exists) { "Green" } else { "Red" }
        $status = if ($exists) { "存在" } else { "缺失" }
        Write-Host "  $dir/ : $status" -ForegroundColor $color
    }
}

# ---------------------------------------------------------------------------
# 9. Git 状态
# ---------------------------------------------------------------------------
function Get-IronAgingGitStatus {
    <#
    .SYNOPSIS
        显示项目 git 状态与最近 5 条提交。
    #>
    Push-Location $script:IronAgingRoot
    try {
        Write-Host "Git 状态：" -ForegroundColor Cyan
        git status --short
        Write-Host "`n最近 5 条提交：" -ForegroundColor Cyan
        git log --oneline -5
    }
    finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# 10. 安装 MCP 服务器依赖
# ---------------------------------------------------------------------------
function Install-IronAgingMcpDeps {
    <#
    .SYNOPSIS
        安装 mcp_project_server.py 所需的 Python 依赖。
    #>
    Write-Host "[MCP Deps] pip install mcp pyyaml" -ForegroundColor Cyan
    python -m pip install --upgrade mcp pyyaml
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[MCP Deps] 安装成功" -ForegroundColor Green
    }
    else {
        Write-Host "[MCP Deps] 安装失败" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# 11. 并发质量门禁
# ---------------------------------------------------------------------------
function Start-IronAgingParallelQualityGate {
    <#
    .SYNOPSIS
        并发执行 ruff、validate_inputs、test_config_loading、test_module3。
        自动根据 CPU/内存推荐 worker 数。
    #>
    Write-Host "[Parallel Quality Gate] 启动并发 lint + validation + tests" -ForegroundColor Cyan
    Push-Location $script:IronAgingRoot
    try {
        python -c "
import json
from concurrency_utils import parallel_quality_gate
result = parallel_quality_gate()
print(json.dumps(result, indent=2, ensure_ascii=False))
exit(0 if result['success'] else 1)
"
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -eq 0) {
        Write-Host "[Parallel Quality Gate] 全部通过" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "[Parallel Quality Gate] 存在失败任务，请查看上方输出" -ForegroundColor Red
        return $false
    }
}

# ---------------------------------------------------------------------------
# 12. 系统资源与并发推荐
# ---------------------------------------------------------------------------
function Get-IronAgingSystemResources {
    <#
    .SYNOPSIS
        显示 CPU、内存资源与推荐并发 worker 数。
    #>
    return Invoke-IronAgingPythonScript -ScriptName "concurrency_utils.py" -Description "系统资源评估"
}

# ---------------------------------------------------------------------------
# 加载完成提示
# ---------------------------------------------------------------------------
Write-Host "铁衰老项目命令集已加载。输入 Show-IronAgingHelp 查看可用命令。" -ForegroundColor Green
