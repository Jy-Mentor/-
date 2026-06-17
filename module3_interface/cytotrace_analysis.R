suppressPackageStartupMessages({
  library(CytoTRACE)
  library(Matrix)
  library(readr)
})
cat("Loading sparse matrix...\n")
counts <- readMM("raw_counts_for_cytotrace.mtx")
genes <- read_csv("raw_genes.csv", col_names = FALSE)[[1]]
cells <- read_csv("raw_cells.csv", col_names = FALSE)[[1]]
rownames(counts) <- genes
colnames(counts) <- cells
cat(sprintf("Matrix: %d genes x %d cells\n", nrow(counts), ncol(counts)))
results <- CytoTRACE(counts, ncores = 4, subsamplesize = 1000)
write.csv(results$CytoTRACE, "cytotrace_scores.csv")
cat("CytoTRACE done.\n")
