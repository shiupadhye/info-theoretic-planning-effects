library(caret)
library(tidyverse)
library(lme4)
library(ggpubr)
library(reshape2)
library(lmtest)
library(viridis)
library(patchwork)
library(ggeffects)
library(lmerTest)


scores <- read_csv('SWBD_DurAnalysisData.csv')
scores <- scores %>%
  filter(CLASS != 'OTHER')

pred.scores <- scores[,c('unigram_logProb','AutoregFw_logProb','AutoregBw_logProb','infillFw_logProb','infillBw_logProb','delta_infillBw_logProb','infill_pmiFP')]



plot_correlation_from_file <- function(file) {
  # Load required packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
  if (!requireNamespace("reshape2", quietly = TRUE)) install.packages("reshape2")
  if (!requireNamespace("viridis", quietly = TRUE)) install.packages("viridis")
  
  library(ggplot2)
  library(reshape2)
  library(viridis)
  
  # Read CSV and convert to matrix
  df_raw <- read.csv(file, row.names = 1, check.names = FALSE)
  mat <- as.matrix(df_raw)
  
  # Variable names
  vars <- colnames(mat)
  n <- length(vars)
  
  # Parse correlation and stars
  extract_r <- function(label) as.numeric(sub(" \\(.*", "", label))
  extract_star <- function(label) {
    m <- regmatches(label, regexpr("\\(\\*+\\)", label))
    if (length(m) == 0) return("")
    gsub("[()]", "", m)
  }
  
  # Initialize matrices
  cor_mat <- matrix(NA, n, n)
  star_mat <- matrix("", n, n)
  rownames(cor_mat) <- colnames(cor_mat) <- vars
  rownames(star_mat) <- colnames(star_mat) <- vars
  
  # Fill upper triangle only
  for (i in 1:n) {
    for (j in 1:n) {
      if (i <= j) {
        cor_mat[i, j] <- extract_r(mat[i, j])
        star_mat[i, j] <- extract_star(mat[i, j])
      }
    }
  }
  
  # Melt for plotting
  df_long <- melt(cor_mat, varnames = c("Var1", "Var2"), value.name = "Correlation")
  df_long$Stars <- melt(star_mat)$value
  
  # Create simple label: "0.85 (***)"
  df_long$plot_label <- mapply(function(r, s) {
    if (is.na(r)) return(NA)
    if (s == "") sprintf("%.2f", r)
    else sprintf("%.2f (%s)", r, s)
  }, df_long$Correlation, df_long$Stars)
  
  # Drop lower triangle (NAs)
  df_long <- na.omit(df_long)
  
  # Plot
  p <- ggplot(df_long, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile(color = "white") +
    scale_fill_viridis(option = "B", limits = c(-1, 1), name = "Correlation") +
    geom_text(aes(label = plot_label), color = "black", fontface = "bold", size = 3) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, color = "black", face = "bold"),
      axis.text.y = element_text(color = "black", face = "bold"),
      axis.title = element_blank()
    ) +
    coord_fixed()
  return(p)
}

p <- plot_correlation_from_file("correlation_matrix.csv")
ggsave("cor_plot.png", p, width = 10, height = 10, dpi = 600, bg = 'white')