# 中药单体-铁衰老-CIRI 候选化合物可视化
# 数据来源: L3_results/tcm_monomer_screening/iron_aging_ciri_candidates.csv
# 仅使用真实数据, 不模拟.

library(ggplot2)
library(ggrepel)
library(patchwork)

input_csv <- "L3_results/tcm_monomer_screening/iron_aging_ciri_candidates.csv"
out_pdf <- "L3_results/tcm_monomer_screening/candidate_visualization.pdf"
out_png <- "L3_results/tcm_monomer_screening/candidate_visualization.png"

df <- read.csv(input_csv, stringsAsFactors = FALSE)
stopifnot(nrow(df) > 0)

# 取 Top 20
df_top <- head(df, 20)
df_top$compound <- factor(df_top$compound, levels = rev(df_top$compound))

# 图1: 综合评分条形图
p1 <- ggplot(df_top, aes(x = candidate_score, y = compound, fill = candidate_score)) +
  geom_bar(stat = "identity", width = 0.7) +
  scale_fill_gradient(low = "#4575B4", high = "#D73027", name = "Score") +
  labs(
    x = "Candidate score",
    y = NULL,
    title = "Top 20 TCM monomers for iron-aging-related CIRI therapy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.y = element_text(size = 10),
    legend.position = "right"
  )

# 图2: 气泡图: 铁衰老靶点 vs CIRI靶点, 大小=平均置信度, 颜色=综合评分
p2 <- ggplot(df_top, aes(x = iron_aging_target_count, y = ciri_target_count)) +
  geom_point(aes(size = mean_confidence, color = candidate_score), alpha = 0.8) +
  geom_text_repel(aes(label = compound), size = 3, max.overlaps = 20) +
  scale_color_gradient(low = "#4575B4", high = "#D73027", name = "Score") +
  scale_size_continuous(range = c(2, 8), name = "Mean confidence") +
  labs(
    x = "Iron-aging target count",
    y = "CIRI target count",
    title = "Iron-aging vs CIRI target coverage"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(size = 14, face = "bold"))

combined <- p1 + p2 + plot_layout(ncol = 1, heights = c(1.2, 1))

ggsave(out_pdf, combined, width = 10, height = 14, dpi = 300)
ggsave(out_png, combined, width = 10, height = 14, dpi = 300, bg = "white")

cat("Saved:", out_pdf, "\n")
cat("Saved:", out_png, "\n")
