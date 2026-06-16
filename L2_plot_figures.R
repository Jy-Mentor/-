#!/usr/bin/env Rscript
# =============================================================================
# L2 图表生成: 图2A-D
# 基于 D:\R语言绘图模板 风格
# =============================================================================
suppressPackageStartupMessages({
  library(ggplot2)
  library(ggpubr)
  library(pheatmap)
  library(dplyr)
  library(tidyr)
  library(RColorBrewer)
})

# 工作目录
setwd("c:/Users/Jy-Mentor-7/Desktop/铁衰老")

# ============================================================
# 图2A: ACSL4 在 ISP 高/低组中的表达 (小提琴图 + 箱线图)
# 参考: 11小提琴图/bioR11.vioplot.R + 12多基因小提琴图
# ============================================================

plot_fig2A_acsl4_violin <- function() {
  cat("\n===== Fig 2A: ACSL4 小提琴图 =====\n")
  
  data_file <- "L2_results/L2_ACSL4_expression_data.csv"
  if (!file.exists(data_file)) {
    cat(sprintf("  [SKIP] 数据文件不存在: %s\n", data_file))
    return(NULL)
  }
  
  rt <- read.csv(data_file, header = TRUE, stringsAsFactors = FALSE)
  colnames(rt) <- c("Dataset", "Species", "Group", "Expression")
  
  # 美化标签
  rt$Dataset <- factor(rt$Dataset, levels = c("GSE16561", "GSE37587", "GSE61616", "GSE97537", "GSE104036"))
  rt$Group <- factor(rt$Group, levels = c("High_ISP", "Low_ISP"))
  
  # 自定义颜色: ISP 高=红色系, ISP 低=蓝色系
  colors <- c("High_ISP" = "#E74C3C", "Low_ISP" = "#3498DB")
  
  p <- ggviolin(rt, x = "Dataset", y = "Expression", 
                fill = "Group",
                palette = colors,
                add = "boxplot", 
                add.params = list(fill = "white", width = 0.15),
                xlab = "", 
                ylab = "ACSL4 Expression",
                title = "ACSL4 Expression in High vs Low ISP Groups") +
    theme_classic(base_size = 14) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
      axis.title.y = element_text(size = 13),
      legend.position = "top",
      plot.title = element_text(hjust = 0.5, face = "bold", size = 15)
    ) +
    stat_compare_means(aes(group = Group), 
                       label = "p.signif", 
                       method = "t.test",
                       hide.ns = FALSE,
                       label.y = max(rt$Expression, na.rm = TRUE) * 1.05)
  
  pdf("L2_results/Fig2A_ACSL4_violin_plot.pdf", width = 10, height = 6)
  print(p)
  dev.off()
  
  cat("  Fig 2A 完成: L2_results/Fig2A_ACSL4_violin_plot.pdf\n")
}

# ============================================================
# 图2B: WGCNA 模块热图 (ACSL4 模块基因表达模式)
# 参考: 17热图/bioR17.pheatmap.R
# ============================================================

plot_fig2B_wgcna_module_heatmap <- function() {
  cat("\n===== Fig 2B: WGCNA 模块热图 =====\n")
  
  # 先尝试加载 ACSL4 模块基因
  ds_name <- "GSE16561"  # 人类主数据集
  module_file <- sprintf("L2_WGCNA_output/%s_ACSL4_module_genes.csv", ds_name)
  expr_file <- sprintf("L2_WGCNA_input/%s_expr.csv", ds_name)
  
  if (!file.exists(module_file)) {
    # 尝试其他数据集
    for (ds in c("GSE37587", "GSE61616", "GSE97537", "GSE104036")) {
      module_file <- sprintf("L2_WGCNA_output/%s_ACSL4_module_genes.csv", ds)
      expr_file <- sprintf("L2_WGCNA_input/%s_expr.csv", ds)
      if (file.exists(module_file)) { ds_name <- ds; break }
    }
  }
  
  if (!file.exists(module_file) || !file.exists(expr_file)) {
    cat(sprintf("  [SKIP] 模块文件不存在: %s\n", module_file))
    return(NULL)
  }
  
  # 加载数据
  module_genes <- read.csv(module_file, stringsAsFactors = FALSE)
  expr <- read.csv(expr_file, row.names = 1, check.names = FALSE)
  
  # 选择 ACSL4 模块基因（取 top 50 个）
  acsl4_mod <- module_genes$module[1]
  mod_genes <- module_genes$gene
  
  # 取前 50 个（按 kME 排序）
  if ("kME" %in% colnames(module_genes)) {
    mod_genes <- module_genes[order(abs(module_genes$kME), decreasing = TRUE), "gene"]
  }
  mod_genes <- head(mod_genes, 50)
  
  # 取交集
  common_genes <- intersect(mod_genes, rownames(expr))
  if (length(common_genes) < 5) {
    cat(sprintf("  [SKIP] 共同基因不足: %d\n", length(common_genes)))
    return(NULL)
  }
  
  expr_sub <- expr[common_genes, , drop = FALSE]
  expr_sub <- expr_sub[rowSums(is.na(expr_sub)) == 0, , drop = FALSE]
  
  if (nrow(expr_sub) < 3) {
    cat(sprintf("  [SKIP] 有效基因不足: %d\n", nrow(expr_sub)))
    return(NULL)
  }
  
  cat(sprintf("  [%s] ACSL4 模块: %s, %d 基因\n", ds_name, acsl4_mod, nrow(expr_sub)))
  
  # 绘制热图
  ann_colors <- list(
    module = setNames(c("#E74C3C"), acsl4_mod)
  )
  
  annotation_row <- data.frame(
    module = rep(acsl4_mod, nrow(expr_sub)),
    row.names = rownames(expr_sub)
  )
  
  pdf("L2_results/Fig2B_WGCNA_module_heatmap.pdf", width = 12, height = 10)
  pheatmap(expr_sub,
           annotation_row = annotation_row,
           annotation_colors = ann_colors,
           cluster_rows = TRUE,
           cluster_cols = TRUE,
           color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
           scale = "row",
           show_rownames = (nrow(expr_sub) <= 30),
           show_colnames = TRUE,
           fontsize = 8,
           fontsize_row = 7,
           fontsize_col = 6,
           main = sprintf("ACSL4 Module Genes (%s) - %s", ds_name, acsl4_mod),
           border_color = NA)
  dev.off()
  
  cat(sprintf("  Fig 2B 完成: L2_results/Fig2B_WGCNA_module_heatmap.pdf\n"))
}

# ============================================================
# 图2C: 跨物种 Zsummary 热图 (模块保留性)
# 参考: 17热图/bioR17.pheatmap.R + 23相关性热图
# ============================================================

plot_fig2C_jaccard_heatmap <- function() {
  cat("\n===== Fig 2C: 跨物种 ACSL4 模块基因重叠热图 =====\n")
  
  data_file <- "L2_results/L2_jaccard_heatmap_data.csv"
  if (!file.exists(data_file)) {
    cat(sprintf("  [SKIP] 数据文件不存在: %s\n", data_file))
    return(NULL)
  }
  
  jmat <- read.csv(data_file, row.names = 1, check.names = FALSE)
  
  if (nrow(jmat) < 2 || ncol(jmat) < 2) {
    cat(sprintf("  [SKIP] 矩阵太小: %d x %d\n", nrow(jmat), ncol(jmat)))
    return(NULL)
  }
  
  cat(sprintf("  Jaccard 矩阵: %d x %d\n", nrow(jmat), ncol(jmat)))
  
  # 填充 NA 为 0
  jmat[is.na(jmat)] <- 0
  
  pdf("L2_results/Fig2C_Jaccard_overlap_heatmap.pdf", width = 9, height = 7)
  pheatmap(as.matrix(jmat),
           cluster_rows = TRUE,
           cluster_cols = TRUE,
           color = colorRampPalette(c("white", "lightyellow", "orange", "red", "darkred"))(100),
           display_numbers = TRUE,
           number_format = "%.3f",
           fontsize_number = 10,
           fontsize = 11,
           fontsize_row = 10,
           fontsize_col = 10,
           main = "Cross-Species ACSL4 Module Gene Overlap (Jaccard Index)",
           angle_col = 0,
           border_color = "grey80",
           na_col = "grey90",
           legend = TRUE)
  dev.off()
  
  cat("  Fig 2C 完成: L2_results/Fig2C_Jaccard_overlap_heatmap.pdf\n")
}

# ============================================================
# 图2D: GPX4 趋势图 (ISP 高/中/低组的 GPX4 表达)
# 参考: 折线+散点+误差棒+显著性 模板
# ============================================================

plot_fig2D_gpx4_trend <- function() {
  cat("\n===== Fig 2D: GPX4 趋势图 =====\n")
  
  data_file <- "L2_results/L2_GPX4_trend_data.csv"
  if (!file.exists(data_file)) {
    cat(sprintf("  [SKIP] 数据文件不存在: %s\n", data_file))
    return(NULL)
  }
  
  gpx4 <- read.csv(data_file, stringsAsFactors = FALSE)
  
  # 转换为长格式
  gpx4_long <- gpx4 %>%
    select(dataset, species, gpx4_high_isp, gpx4_mid_isp, gpx4_low_isp,
           gpx4_high_std, gpx4_mid_std, gpx4_low_std) %>%
    pivot_longer(
      cols = c(gpx4_high_isp, gpx4_mid_isp, gpx4_low_isp),
      names_to = "ISP_Group",
      values_to = "GPX4_Expression",
      names_pattern = "gpx4_(.*)_isp"
    ) %>%
    pivot_longer(
      cols = c(gpx4_high_std, gpx4_mid_std, gpx4_low_std),
      names_to = "ISP_Group_sd",
      values_to = "SD",
      names_pattern = "gpx4_(.*)_std"
    ) %>%
    filter(ISP_Group == ISP_Group_sd) %>%
    select(-ISP_Group_sd)
  
  gpx4_long$ISP_Group <- factor(gpx4_long$ISP_Group, levels = c("high", "mid", "low"),
                                 labels = c("High ISP", "Mid ISP", "Low ISP"))
  gpx4_long$dataset <- factor(gpx4_long$dataset)
  
  # 自定义颜色
  group_colors <- c("High ISP" = "#E74C3C", "Mid ISP" = "#F39C12", "Low ISP" = "#3498DB")
  
  p <- ggplot(gpx4_long, aes(x = ISP_Group, y = GPX4_Expression, 
                              color = ISP_Group, group = dataset)) +
    geom_point(size = 3) +
    geom_line(aes(group = dataset), color = "grey50", linetype = "dashed", alpha = 0.5) +
    geom_errorbar(aes(ymin = GPX4_Expression - SD, ymax = GPX4_Expression + SD),
                  width = 0.2) +
    facet_wrap(~ dataset, scales = "free_y", nrow = 1) +
    scale_color_manual(values = group_colors) +
    labs(x = "", y = "GPX4 Expression",
         title = "GPX4 Expression Across ISP Groups (ISP != Ferroptosis)") +
    theme_classic(base_size = 13) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1, size = 10),
      legend.position = "top",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      strip.text = element_text(face = "bold")
    )
  
  pdf("L2_results/Fig2D_GPX4_trend.pdf", width = 12, height = 5)
  print(p)
  dev.off()
  
  cat("  Fig 2D 完成: L2_results/Fig2D_GPX4_trend.pdf\n")
}

# ============================================================
# 主函数
# ============================================================

cat(strrep("=", 60), "\n")
cat("L2 图表生成: 图2A-D\n")
cat(strrep("=", 60), "\n")

plot_fig2A_acsl4_violin()
plot_fig2B_wgcna_module_heatmap()
plot_fig2C_jaccard_heatmap()
plot_fig2D_gpx4_trend()

cat("\n===== L2 图表生成完成 =====\n")