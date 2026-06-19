# 铁衰老 × CIRI 功能富集分析
# 输入: dry_lab_report/ferroaging_deg_gene_list.csv
# 输出: dry_lab_report/go_kegg_enrichment_*.csv + 可视化图

library(clusterProfiler)
library(org.Hs.eg.db)
library(ReactomePA)
library(enrichplot)
library(ggplot2)
library(dplyr)

setwd("C:/Users/Jy-Mentor-7/Desktop/铁衰老")
out_dir <- "dry_lab_report"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 读取基因列表
df <- read.csv(file.path(out_dir, "ferroaging_deg_gene_list.csv"), encoding = "UTF-8")
gene_symbols <- unique(df$names)
message("输入基因数: ", length(gene_symbols))

# SYMBOL -> ENTREZID
id_df <- bitr(gene_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
gene_entrez <- id_df$ENTREZID
message("成功转换 ENTREZID: ", length(gene_entrez))

# 背景基因集：所有检测到的基因
all_symbols <- unique(df$names)
universe_df <- bitr(all_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
universe_entrez <- unique(universe_df$ENTREZID)
message("背景基因数: ", length(universe_entrez))

# GO 富集 (BP, MF, CC)
run_go <- function(ont) {
  ego <- enrichGO(
    gene = gene_entrez,
    universe = universe_entrez,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = ont,
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2,
    readable = TRUE
  )
  if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {
    ego_simplified <- simplify(ego, cutoff = 0.7, by = "p.adjust", select_fun = min)
    write.csv(as.data.frame(ego_simplified),
              file.path(out_dir, paste0("go_enrichment_", ont, ".csv")),
              row.names = FALSE, fileEncoding = "UTF-8")

    # 条形图
    p <- barplot(ego_simplified, showCategory = 15, title = paste("GO", ont, "Enrichment")) +
      theme(axis.text.y = element_text(size = 8))
    ggsave(file.path(out_dir, paste0("fig_go_", tolower(ont), "_barplot.png")),
           p, width = 10, height = 8, dpi = 300)

    # 点图
    p2 <- dotplot(ego_simplified, showCategory = 15, title = paste("GO", ont, "Enrichment"))
    ggsave(file.path(out_dir, paste0("fig_go_", tolower(ont), "_dotplot.png")),
           p2, width = 10, height = 8, dpi = 300)

    return(ego_simplified)
  } else {
    message("GO ", ont, " 无显著富集结果")
    return(NULL)
  }
}

ego_bp <- run_go("BP")
ego_mf <- run_go("MF")
ego_cc <- run_go("CC")

# KEGG 富集
kk <- enrichKEGG(
  gene = gene_entrez,
  organism = "hsa",
  universe = universe_entrez,
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2
)

if (!is.null(kk) && nrow(as.data.frame(kk)) > 0) {
  kk_readable <- setReadable(kk, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
  write.csv(as.data.frame(kk_readable),
            file.path(out_dir, "kegg_enrichment.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  p <- dotplot(kk_readable, showCategory = 15, title = "KEGG Pathway Enrichment")
  ggsave(file.path(out_dir, "fig_kegg_dotplot.png"),
         p, width = 10, height = 8, dpi = 300)
} else {
  message("KEGG 无显著富集结果")
}

# Reactome 富集
rea <- enrichPathway(
  gene = gene_entrez,
  organism = "human",
  universe = universe_entrez,
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2,
  readable = TRUE
)

if (!is.null(rea) && nrow(as.data.frame(rea)) > 0) {
  write.csv(as.data.frame(rea),
            file.path(out_dir, "reactome_enrichment.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")

  p <- dotplot(rea, showCategory = 15, title = "Reactome Pathway Enrichment")
  ggsave(file.path(out_dir, "fig_reactome_dotplot.png"),
         p, width = 10, height = 8, dpi = 300)
} else {
  message("Reactome 无显著富集结果")
}

message("富集分析完成，结果保存在 ", out_dir)
