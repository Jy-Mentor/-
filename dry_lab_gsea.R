# 铁衰老 × CIRI 基因集富集分析 (GSEA)
# 输入: module2_results/cell_type_DEG_MCAO_vs_Sham.csv
# 方法: 按 log2FC 对所有基因排序，检测铁衰老相关 GO/KEGG/Reactome 通路的协调变化

library(clusterProfiler)
library(org.Hs.eg.db)
library(ReactomePA)
library(enrichplot)
library(ggplot2)
library(dplyr)

setwd("C:/Users/Jy-Mentor-7/Desktop/铁衰老")
out_dir <- "dry_lab_report"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 读取完整 DEG 表作为背景 + 排序
deg <- read.csv("module2_results/cell_type_DEG_MCAO_vs_Sham.csv", encoding = "UTF-8")
deg$names <- toupper(as.character(deg$names))

# 取每个基因在所有细胞类型中的最大 |log2FC| 作为排序依据
gene_rank <- deg %>%
  group_by(names) %>%
  summarise(max_log2FC = max(abs(logfoldchanges), na.rm = TRUE),
            mean_log2FC = mean(logfoldchanges, na.rm = TRUE)) %>%
  ungroup()

# SYMBOL -> ENTREZID
id_df <- bitr(gene_rank$names, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
id_df <- id_df[!duplicated(id_df$SYMBOL), ]
gene_rank <- merge(gene_rank, id_df, by.x = "names", by.y = "SYMBOL", all.x = FALSE)
gene_rank <- gene_rank[!duplicated(gene_rank$ENTREZID), ]

# 构建 ranked gene list (以 mean log2FC 为排序统计量)
ranked_list <- setNames(gene_rank$mean_log2FC, gene_rank$ENTREZID)
ranked_list <- ranked_list[is.finite(ranked_list)]
ranked_list <- sort(ranked_list, decreasing = TRUE)
message("排序基因列表长度: ", length(ranked_list))

# GO GSEA
tryCatch({
  gse_go <- gseGO(
    geneList = ranked_list,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    minGSSize = 10,
    maxGSSize = 500,
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    verbose = FALSE
  )
  if (!is.null(gse_go) && nrow(as.data.frame(gse_go)) > 0) {
    write.csv(as.data.frame(gse_go), file.path(out_dir, "gsea_go_bp.csv"),
              row.names = FALSE, fileEncoding = "UTF-8")
    p <- dotplot(gse_go, showCategory = 15, title = "GSEA GO BP") +
      theme(axis.text.y = element_text(size = 8))
    ggsave(file.path(out_dir, "fig_gsea_go_bp_dotplot.png"), p, width = 10, height = 8, dpi = 300)
    message("GO GSEA 完成")
  } else {
    message("GO GSEA 无显著结果")
  }
}, error = function(e) {
  message("GO GSEA 错误: ", conditionMessage(e))
})

# Reactome GSEA (本地数据库，不依赖在线 KEGG)
tryCatch({
  gse_rea <- gsePathway(
    geneList = ranked_list,
    organism = "human",
    minGSSize = 10,
    maxGSSize = 500,
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    verbose = FALSE
  )
  if (!is.null(gse_rea) && nrow(as.data.frame(gse_rea)) > 0) {
    write.csv(as.data.frame(gse_rea), file.path(out_dir, "gsea_reactome.csv"),
              row.names = FALSE, fileEncoding = "UTF-8")
    p <- dotplot(gse_rea, showCategory = 15, title = "GSEA Reactome")
    ggsave(file.path(out_dir, "fig_gsea_reactome_dotplot.png"), p, width = 10, height = 8, dpi = 300)
    message("Reactome GSEA 完成")
  } else {
    message("Reactome GSEA 无显著结果")
  }
}, error = function(e) {
  message("Reactome GSEA 错误: ", conditionMessage(e))
})

# KEGG GSEA (增加超时容忍)
options(timeout = 300)
tryCatch({
  gse_kegg <- gseKEGG(
    geneList = ranked_list,
    organism = "hsa",
    minGSSize = 10,
    maxGSSize = 500,
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    verbose = FALSE
  )
  if (!is.null(gse_kegg) && nrow(as.data.frame(gse_kegg)) > 0) {
    write.csv(as.data.frame(gse_kegg), file.path(out_dir, "gsea_kegg.csv"),
              row.names = FALSE, fileEncoding = "UTF-8")
    p <- dotplot(gse_kegg, showCategory = 15, title = "GSEA KEGG")
    ggsave(file.path(out_dir, "fig_gsea_kegg_dotplot.png"), p, width = 10, height = 8, dpi = 300)
    message("KEGG GSEA 完成")
  } else {
    message("KEGG GSEA 无显著结果")
  }
}, error = function(e) {
  message("KEGG GSEA 错误: ", conditionMessage(e))
})

message("GSEA 分析完成")
