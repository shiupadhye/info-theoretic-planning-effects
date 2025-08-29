library(broom)
library(tidyverse)
library(ggpubr)
library(MuMIn)
library(ggplot2)
library(reshape2)
library(viridis)
library(lmtest)
library(doParallel)
library(patchwork)
library(relaimpo)
library(dominanceanalysis)

plt.modelCoeffs <- readRDS("plots/rds/Study2_ModelCoefficients_allCategories.rds")
plt.LRT <- readRDS("plots/rds/Study2_LRT.rds")

plt.AllSubsModel <- plt.modelCoeffs + plt.LRT + plot_layout(widths = c(2, 1))
ggsave("plots/png/Study2_ModelCoeffs&LRT.png",plt.AllSubsModel , width = 12, height = 5, dpi = 300)

### create a patchwork for fine-grained analysis
plt.modelCoeffs.sem <- readRDS("plots/rds/Study2_ModelCoefficients_sem.rds")
plt.modelCoeffs.sem <- plt.modelCoeffs.sem + theme(axis.title.x = element_blank(),
                            axis.ticks.x = element_blank(),
                            axis.text.x  = element_blank(),
                            axis.title.y = element_blank())

plt.modelCoeffs.phon <- readRDS("plots/rds/Study2_ModelCoefficients_phon.rds")
plt.modelCoeffs.phon <- plt.modelCoeffs.phon + theme(axis.title.x = element_blank(),
                            axis.ticks.x = element_blank(),
                            axis.text.x  = element_blank(),
                            axis.title.y = element_blank())

plt.modelCoeffs.mixed <- readRDS("plots/rds/Study2_ModelCoefficients_mixed.rds")
plt.modelCoeffs.mixed <- plt.modelCoeffs.mixed + theme(axis.title.y = element_blank())

plt.modelCoeffs.ms <- readRDS("plots/rds/Study2_ModelCoefficients_ms.rds")
plt.modelCoeffs.ms <- plt.modelCoeffs.ms + theme(axis.title.y = element_blank())


plt.combined <- (plt.modelCoeffs.sem + plt.modelCoeffs.phon) / (plt.modelCoeffs.mixed + plt.modelCoeffs.ms) 
plt.combined
ggsave("plots/png/Study2_errorAnalysis.png", plt.combined, width = 12, height = 8)  
