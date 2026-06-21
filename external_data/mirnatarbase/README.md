# miRTarBase 数据来源说明

## 数据库
- **名称**: miRTarBase 10.0
- **类型**: 实验验证的 miRNA-靶基因相互作用数据库
- **下载日期**: 2026-06-21
- **下载 URL**: https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/cache/download/10.0/miRTarBase_SE_WR.csv

## 使用的子集
- **文件名**: `miRTarBase_SE_WR.csv`
- **说明**: Strong experimental evidences (Reporter assay or Western blot)
- **原始记录数**: 27,595
- **人类 (hsa) 记录数**: 23,049
- **项目基因过滤后**: 492 条唯一 miRNA-靶基因边

## 过滤条件
1. 仅保留 `Species (miRNA)` 为 `hsa` 的人类 miRNA
2. 仅保留靶基因符号在项目 STRING PPI 基因集合中的记录
3. 按 (miRNA, target_gene) 去重

## 引用
- Cui S, Yu S, Huang HY, et al. miRTarBase 2025: updates to the collection of experimentally validated microRNA-target interactions. Nucleic Acids Res. 2025;53(D1):D147-D156. doi:10.1093/nar/gkae1072
