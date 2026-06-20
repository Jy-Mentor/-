# 项目交付物归档目录

本目录用于保存可复现、可追踪的关键结果与可视化，避免所有输出被 `.gitignore` 忽略后无法版本化。

## 归档机制

- `L3_results/` 目录仍用于存放运行期生成的大量中间/临时结果（被 `.gitignore` 忽略）。
- `reports/` 目录用于存放经人工确认、需要长期保留的最终交付物（报告、图表、关键 CSV/JSON）。
- 每次生成新结果后，使用项目脚本或以下命令将最终交付物同步到本目录：

```powershell
python -c "
import shutil, glob, os
src = 'L3_results/tcm_ferroptosis_ciri_gnn'
dst = 'reports/tcm_ferroptosis_ciri_gnn'
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f'{src}/*'):
    if f.endswith(('.csv', '.json', '.png', '.md')):
        shutil.copy(f, dst)
print('synced:', sorted(os.listdir(dst)))
"
```

## 当前归档内容

| 目录 | 说明 |
|---|---|
| `tcm_ferroptosis_ciri_gnn/` | 中药单体×铁衰老×CIRI GNN 预测模块最终交付物 |

## 注意事项

- 纳入 Git 的文件应控制在合理大小（单张 PNG 建议 < 5MB）；
- 大型模型权重（`.pt`）、原始中间矩阵（`.npz`、`.mtx`）仍应排除在版本控制外；
- 更新交付物后请及时提交 Git。
