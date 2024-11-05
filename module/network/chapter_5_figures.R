#`Author : Cansu Dincer
#`Date : 10 February 2023
#`Last Update : 28 June 2024
#`Input : Omics data
#`Output : Chapter 5 Figures
#` ------------------------------------------------------------------------#


path_base <- "/Volumes/team215_pipeline/Cansu/CombDrug/"
input_path <- paste0(path_base, "output/data/")
omics_output_path <- paste0(input_path, "figures/")
library("dplyr")
library("tidyverse")
library("readr")
library("tibble")
library("tidyr")
library("gtools")
library("ggplot2")
library("ggrepel")
library("ggthemes")
library("reshape2")
library("ComplexHeatmap")
library("circlize")
library("RColorBrewer")
library("gridBase")


##########################################################################################
### Figure 5-XA


module_pathway_df <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/network/figure_files/module_pathway.csv", sep=",")

g<- ggplot(module_pathway_df, aes(drug, pathway, fill= ratio)) +
  geom_tile() +
  theme_bw() +
  theme(axis.text.x = element_text(size=rel(1.15)),
        axis.title = element_text(size=rel(1.15))) 

ggsave(
  filename=paste0(omics_output_path, "pathway_dot_heatmap.pdf"), plot = g, width = 24, height = 20, units = "cm", dpi = 300)
dev.off()


# Order same
module_pathway_df2 <- module_pathway_df[order(module_pathway_df$X),]
module_pathway_df2 <- module_pathway_df2[,order(colnames(module_pathway_df2))]

# Make colnames
rownames(module_pathway_df2) <- module_pathway_df2$X
module_pathway_df2 <- module_pathway_df2[, !(colnames(module_pathway_df2) %in% c("X"))]

# Make hm_df3 as matrix
p_matrix <- data.matrix(module_pathway_df2)

pdf(file= paste0(omics_output_path, "perturbation_heatmap.pdf"))

hm <- Heatmap(p_matrix, name = "Enrichment", na_col ="grey95", cluster_columns=F, cluster_rows = F, 
              show_column_names = T, show_row_names = T, 
              rect_gp = gpar(col = "white", lwd = 0.001),
              column_names_gp = grid::gpar(fontsize = 4),
              row_names_gp = grid::gpar(fontsize = 4))
