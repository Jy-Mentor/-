library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)

# 延长 KEGG 在线注释下载超时
options(timeout = 300)

args <- commandArgs(trailingOnly = TRUE)
exp_name <- ifelse(length(args) >= 1, args[1], "hgt_compare_seed42")
exp_dir <- file.path("L3_results", exp_name)

if (!dir.exists(exp_dir)) {
  stop(paste("实验目录不存在:", exp_dir))
}

pred_path <- file.path(exp_dir, "top100_novel_predictions.csv")
all_path <- file.path(exp_dir, "all_compound_gene_scores.csv")

if (!file.exists(pred_path)) {
  stop(paste("预测文件不存在:", pred_path))
}
if (!file.exists(all_path)) {
  stop(paste("候选分数文件不存在:", all_path))
}

pred <- read.csv(pred_path, stringsAsFactors = FALSE)
all_genes <- read.csv(all_path, stringsAsFactors = FALSE)

if (!("gene_name" %in% colnames(pred))) {
  stop("top100_novel_predictions.csv 中缺少 gene_name 列")
}
if (!("gene_name" %in% colnames(all_genes))) {
  stop("all_compound_gene_scores.csv 中缺少 gene_name 列")
}

gene_list <- unique(trimws(pred$gene_name))
background <- unique(trimws(all_genes$gene_name))

cat("实验:", exp_name, "\n")
cat("Top predicted genes:", length(gene_list), "\n")
cat("Background genes:", length(background), "\n")

if (length(gene_list) < 5 || length(background) < 50) {
  stop("基因数量不足，无法执行富集分析")
}

id_top <- bitr(gene_list, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
id_bg <- bitr(background, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)

cat("Mapped top genes:", nrow(id_top), "\n")
cat("Mapped background genes:", nrow(id_bg), "\n")

run_go <- function() {
  if (nrow(id_top) < 5) {
    cat("Too few mapped genes for GO enrichment.\n")
    return(NULL)
  }
  ego <- enrichGO(
    gene = id_top$ENTREZID,
    universe = id_bg$ENTREZID,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    readable = TRUE
  )
  return(ego)
}

run_kegg <- function() {
  if (nrow(id_top) < 5) {
    cat("Too few mapped genes for KEGG enrichment.\n")
    return(NULL)
  }
  kegg <- enrichKEGG(
    gene = id_top$ENTREZID,
    organism = "hsa",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    universe = id_bg$ENTREZID
  )
  return(kegg)
}

# GO BP
tryCatch(
  {
    ego <- run_go()
    if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {
      ego_df <- as.data.frame(ego)
      write.csv(ego_df, file.path(exp_dir, "go_bp_enrichment_top100.csv"), row.names = FALSE)
      cat("GO BP enriched terms:", nrow(ego_df), "\n")
      print(head(ego_df[, c("Description", "p.adjust", "Count")], 10))
    } else {
      cat("No significant GO BP enrichment.\n")
    }
  },
  error = function(e) {
    cat("GO BP 富集分析失败:", conditionMessage(e), "\n")
  }
)

# KEGG
tryCatch(
  {
    kegg <- run_kegg()
    if (!is.null(kegg) && nrow(as.data.frame(kegg)) > 0) {
      kegg_readable <- setReadable(kegg, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
      kegg_df <- as.data.frame(kegg_readable)
      write.csv(kegg_df, file.path(exp_dir, "kegg_enrichment_top100.csv"), row.names = FALSE)
      cat("KEGG enriched pathways:", nrow(kegg_df), "\n")
      print(head(kegg_df[, c("Description", "p.adjust", "Count")], 10))
    } else {
      cat("No significant KEGG enrichment.\n")
    }
  },
  error = function(e) {
    cat("KEGG 富集分析失败:", conditionMessage(e), "\n")
  }
)

cat("富集分析完成.\n")
