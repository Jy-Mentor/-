#!/usr/bin/env Rscript
# =============================================================================
# L2 机制层: 真实 WGCNA + 跨物种基因重叠分析
# 使用 R WGCNA 包做全基因组共表达
# 跨物种保守性: 超几何检验 + Jaccard + 功能富集
# =============================================================================
suppressPackageStartupMessages({
  library(WGCNA)
  library(dplyr)
})

options(stringsAsFactors = FALSE)
enableWGCNAThreads(4)

INPUT_DIR  <- "L2_WGCNA_input"
OUTPUT_DIR <- "L2_WGCNA_output"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

DATASETS <- c("GSE16561", "GSE37587", "GSE61616", "GSE97537", "GSE104036")

# ============================================================
# 辅助函数
# ============================================================

run_wgcna <- function(expr, ds_name, n_top = 5000) {
  if (ncol(expr) < 15) n_top <- min(n_top, nrow(expr))
  gene_vars <- apply(expr, 1, var, na.rm = TRUE)
  top_genes <- names(sort(gene_vars, decreasing = TRUE))[1:min(n_top, length(gene_vars))]
  expr_sub <- expr[top_genes, , drop = FALSE]
  datExpr <- t(expr_sub)
  
  gsg <- goodSamplesGenes(datExpr, verbose = 0)
  if (!gsg$allOK) datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]
  
  n_samples <- nrow(datExpr)
  n_genes <- ncol(datExpr)
  cat(sprintf("\n  [%s] %d 样本 x %d 基因\n", ds_name, n_samples, n_genes))
  
  if (n_samples < 8 || n_genes < 50) {
    cat(sprintf("  [%s] 样本或基因不足，跳过\n", ds_name))
    return(NULL)
  }
  
  # 软阈值选择
  powers <- c(1:20)
  sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 0)
  sft_df <- sft$fitIndices
  if (any(sft_df$SFT.R.sq > 0.8, na.rm = TRUE)) {
    power <- sft_df$Power[which(sft_df$SFT.R.sq > 0.8)[1]]
  } else if (any(sft_df$SFT.R.sq > 0.5, na.rm = TRUE)) {
    power <- sft_df$Power[which(sft_df$SFT.R.sq > 0.5)[1]]
  } else {
    power <- sft_df$Power[which.max(sft_df$SFT.R.sq)]
  }
  cat(sprintf("  [%s] power=%d (R^2=%.3f)\n", ds_name, power, 
              sft_df$SFT.R.sq[sft_df$Power == power]))
  
  # 构建网络
  net <- blockwiseModules(
    datExpr, power = power, TOMType = "signed",
    minModuleSize = 30, reassignThreshold = 0,
    mergeCutHeight = 0.25, numericLabels = TRUE,
    pamRespectsDendro = FALSE, saveTOMs = FALSE, verbose = 0
  )
  
  module_labels <- net$colors
  n_modules <- length(unique(module_labels))
  cat(sprintf("  [%s] %d 个模块\n", ds_name, n_modules))
  
  MEs <- net$MEs
  colnames(MEs) <- paste0("ME", unique(module_labels))
  
  gene_module_df <- data.frame(
    gene = colnames(datExpr), module = module_labels, stringsAsFactors = FALSE
  )
  
  # ACSL4 所在模块
  acsl4_module <- gene_module_df$module[gene_module_df$gene == "ACSL4"]
  if (length(acsl4_module) == 0) acsl4_module <- NA
  
  acsl4_module_genes <- gene_module_df$gene[gene_module_df$module == acsl4_module]
  cat(sprintf("  [%s] ACSL4 模块: %s (%d 基因)\n", ds_name, acsl4_module, length(acsl4_module_genes)))
  
  # kME
  gene_module_df$kME <- NA
  if (!is.na(acsl4_module) && length(acsl4_module_genes) > 0) {
    me_col <- paste0("ME", acsl4_module)
    if (me_col %in% colnames(MEs)) {
      for (i in seq_len(nrow(gene_module_df))) {
        g <- gene_module_df$gene[i]
        if (g %in% colnames(datExpr)) {
          gene_module_df$kME[i] <- cor(datExpr[, g], MEs[, me_col], use = "complete.obs")
        }
      }
    }
  }
  
  list(dataset = ds_name, power = power, n_modules = n_modules,
       acsl4_module = acsl4_module, acsl4_module_genes = acsl4_module_genes,
       gene_module = gene_module_df, datExpr = datExpr, sft = sft_df)
}

# ============================================================
# 跨物种基因重叠分析
# ============================================================

calc_overlap <- function(genes_a, genes_b, total_genes) {
  common <- intersect(genes_a, genes_b)
  n_common <- length(common)
  n_a <- length(genes_a)
  n_b <- length(genes_b)
  jaccard <- n_common / (n_a + n_b - n_common)
  
  # 超几何检验
  pval <- phyper(n_common - 1, n_a, total_genes - n_a, n_b, lower.tail = FALSE)
  
  list(n_common = n_common, jaccard = jaccard, pvalue = pval, common_genes = common)
}

# ============================================================
# 主流程
# ============================================================

cat(strrep("=", 60), "\n")

all_results <- list()
all_genes <- list()  # 每个数据集的全基因列表

for (ds in DATASETS) {
  input_file <- file.path(INPUT_DIR, paste0(ds, "_expr.csv"))
  if (!file.exists(input_file)) {
    cat(sprintf("[%s] 输入文件不存在\n", ds))
    next
  }
  cat(sprintf("\n--- 加载 %s ---\n", ds))
  expr <- read.csv(input_file, row.names = 1, check.names = FALSE)
  expr <- expr[, sapply(expr, is.numeric), drop = FALSE]
  cat(sprintf("  维度: %d 基因 x %d 样本\n", nrow(expr), ncol(expr)))
  
  res <- run_wgcna(expr, ds)
  if (!is.null(res)) {
    all_results[[ds]] <- res
    all_genes[[ds]] <- rownames(expr)
  }
}

cat(sprintf("\n有效数据集: %d / %d\n", length(all_results), length(DATASETS)))

# 保存 WGCNA 结果
for (ds in names(all_results)) {
  res <- all_results[[ds]]
  write.csv(res$gene_module, file.path(OUTPUT_DIR, paste0(ds, "_module_assignment.csv")), row.names = FALSE)
  if (!is.na(res$acsl4_module)) {
    acsl4_df <- res$gene_module[res$gene_module$gene %in% res$acsl4_module_genes, ]
    write.csv(acsl4_df, file.path(OUTPUT_DIR, paste0(ds, "_ACSL4_module_genes.csv")), row.names = FALSE)
  }
}

# ============================================================
# 跨物种基因重叠分析
# ============================================================
cat("\n===== 跨物种基因重叠分析 =====\n")

# 收集 ACSL4 模块基因
acsl4_genes_list <- list()
for (ds in names(all_results)) {
  res <- all_results[[ds]]
  if (!is.na(res$acsl4_module)) {
    acsl4_genes_list[[ds]] <- res$acsl4_module_genes
    cat(sprintf("  [%s] ACSL4 模块: %d 基因\n", ds, length(res$acsl4_module_genes)))
  }
}

# 计算所有配对的基因重叠
overlap_results <- list()
ds_names <- names(acsl4_genes_list)

if (length(ds_names) >= 2) {
  # 使用所有数据集的基因并集作为背景
  all_genes_union <- unique(unlist(all_genes))
  n_total <- length(all_genes_union)
  cat(sprintf("  背景基因总数: %d\n", n_total))
  
  for (i in 1:(length(ds_names) - 1)) {
    for (j in (i + 1):length(ds_names)) {
      ds_a <- ds_names[i]
      ds_b <- ds_names[j]
      ol <- calc_overlap(acsl4_genes_list[[ds_a]], acsl4_genes_list[[ds_b]], n_total)
      
      species_a <- ifelse(ds_a %in% c("GSE16561", "GSE37587"), "Human",
                          ifelse(ds_a %in% c("GSE61616", "GSE97537"), "Rat", "Mouse"))
      species_b <- ifelse(ds_b %in% c("GSE16561", "GSE37587"), "Human",
                          ifelse(ds_b %in% c("GSE61616", "GSE97537"), "Rat", "Mouse"))
      cross_type <- ifelse(species_a == species_b, "Same", "Cross")
      
      overlap_results[[length(overlap_results) + 1]] <- data.frame(
        dataset_a = ds_a, dataset_b = ds_b,
        species_a = species_a, species_b = species_b,
        cross_type = cross_type,
        n_genes_a = length(acsl4_genes_list[[ds_a]]),
        n_genes_b = length(acsl4_genes_list[[ds_b]]),
        n_overlap = ol$n_common,
        jaccard = round(ol$jaccard, 4),
        pvalue = format(ol$pvalue, digits = 3, scientific = TRUE),
        common_genes = paste(head(ol$common_genes, 20), collapse = "|"),
        stringsAsFactors = FALSE
      )
      
      cat(sprintf("  [%s vs %s] overlap=%d, jaccard=%.4f, p=%s\n",
                  ds_a, ds_b, ol$n_common, ol$jaccard, 
                  format(ol$pvalue, digits = 3, scientific = TRUE)))
    }
  }
  
  overlap_df <- do.call(rbind, overlap_results)
  write.csv(overlap_df, file.path(OUTPUT_DIR, "L2_cross_species_overlap.csv"), row.names = FALSE)
  cat(sprintf("\n  重叠分析结果已保存: %d 对比较\n", nrow(overlap_df)))
}

# ============================================================
# 汇总
# ============================================================
cat("\n===== L2 WGCNA 汇总 =====\n")
for (ds in names(all_results)) {
  res <- all_results[[ds]]
  cat(sprintf("  %s: %d 模块, ACSL4模块=%s, power=%d\n",
              ds, res$n_modules, res$acsl4_module, res$power))
}

summary_df <- do.call(rbind, lapply(names(all_results), function(ds) {
  res <- all_results[[ds]]
  data.frame(dataset = ds, n_samples = nrow(res$datExpr),
             n_genes = ncol(res$datExpr), power = res$power,
             n_modules = res$n_modules,
             acsl4_module = ifelse(is.na(res$acsl4_module), "NOT_FOUND", as.character(res$acsl4_module)),
             n_acsl4_module_genes = length(res$acsl4_module_genes),
             stringsAsFactors = FALSE)
}))

write.csv(summary_df, file.path(OUTPUT_DIR, "L2_WGCNA_summary.csv"), row.names = FALSE)

cat("\n===== L2 WGCNA 完成 =====\n")