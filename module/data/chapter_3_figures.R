#`Author : Cansu Dincer
#`Date : 10 February 2023
#`Last Update : 28 June 2024
#`Input : Omics data
#`Output : Chapter 3 Figures
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
### Figure 3-2A

pan_colors = c("pansolid" = "#8c2f39", "panliquid"="#001f54")

tissue_colors = c(
  "Adrenal Gland"="#C1A72F",
  "Biliary Tract"="#9BABBA", 
  "Bladder"="#EBDEF0", 
  "Bone"= "#D98880",
  "Breast"="#F5CBA7",
  "Central Nervous System"="#D6DBDF",
  "Cervix"="#FBEEE9", 
  "Endometrium"= "#FAD7A0",
  "Esophagus"="#ABEBC6", 
  "Head and Neck"="#A084BD", 
  "Kidney"="#B1D0C1",
  "Large Intestine"="#F5B7B1",
  "Liver"="#ABDBDA", 
  "Lung"="#CE661E", 
  "Ovary"="#E5E8B3", 
  "Pancreas"="#F2D7D5",
  "Peripheral Nervous System"="#C2DEB4", 
  "Prostate"="#2874A6", 
  "Skin"="#FFB3BC", 
  "Small Intestine"="#D6DEB4", 
  "Soft Tissue"="#F89420", 
  "Stomach"="#1ABC9C", 
  "Testes"="#E5B6F3", 
  "Thyroid"="#C28B90", 
  "Uterus"="#FBE3C7", 
  "Vulva"="#B2509E",
  
  "Leukemia"="#2f6690", "Lymphoma"="#3a7ca5","Myeloma"="#81c3d7",
  "Other liquid tumours"="#90e0ef") 

df <- read.csv(paste0(input_path, "omics/omics_availability.csv"), row.names =NULL, sep = ',')

df2 <-t(df)
colnames(df2) <- df2[1, ]
df3 <- df2[-c(1), ]
df3 <- as.data.frame(df3)

df4 <- df3[order(df3$Pan, df3$Tissue),]
split = factor(df4$Pan)
df4[, c("Mutation", "GEx", "PEx", "CNV", "Methylation", "CRISPR", "Drug Combinations")] <- sapply(df4[, c("Mutation", "GEx", "PEx", "CNV", "Methylation", "CRISPR", "Drug Combinations")], as.numeric)

col_freq = colorRamp2(c(0, 1, median(df4[, "Drug Combinations"]), max(df4[, "Drug Combinations"])), 
                      c("grey", "pink", "red", "darkred"))
col_availability = colorRamp2(c(0, 1), c("grey95", "grey40"))


pdf(file=paste0(omics_output_path, "omics_availability_new_legend.pdf"))

circos.par("track.height" = 0.1, start.degree = 90, gap.after=c(3, 30))

circos.heatmap(df4[, "Pan"], split=split, col = pan_colors, track.height = 0.03, cell.lwd=0.02, show.sector.labels=TRUE)

circos.heatmap(df4[, "Tissue"], split=split, col = tissue_colors, track.height = 0.03, cell.lwd=0.015, show.sector.labels=TRUE)

circos.heatmap(df4[, "Drug Combinations"], split=split, col=col_freq, track.height = 0.07, cell.lwd=0.02,show.sector.labels=TRUE)

circos.heatmap(df4[, c("Mutation", "GEx", "PEx", "CNV", "Methylation")], 
               col = col_availability, track.height = 0.2, cell.lwd=0.015, show.sector.labels=TRUE)

lgd = Legend(title = "#DC", col_fun = col_freq)
grid.draw(lgd)
#png(file=paste0(omics_output_path, "omics_availability_new.png"), res=600)
dev.off()
circos.clear()



##########################################################################################
### FIGURE 3-1B

hm_df <- read.csv(paste0(input_path, "figure_files/availability_heatmap_df.csv"), sep = ',')

hm_df2 <- read.csv(paste0(input_path, "figure_files/availability_heatmap_main_df.csv"), sep=",")
hm_df2 <- hm_df2[, !(colnames(hm_df2) %in% c("names"))]

hm_df_row = read.csv(paste0(input_path, "figure_files/availability_heatmap_main_row_annotation_df.csv"), sep=",")
hm_df_col = read.csv(paste0(input_path, "figure_files/availability_heatmap_main_col_annotation_df.csv"), sep=",")

hm_df3 <- hm_df2
hm_df3[hm_df3 == 0] <- NA


# Order same
hm_df3 <- hm_df3[order(hm_df3$DrugComb),]
hm_df3 <- hm_df3[,order(colnames(hm_df3))]
hm_df_row <- hm_df_row[order(hm_df_row$DrugComb),]
hm_df_col <- hm_df_col[order(hm_df_col$SIDM),]

# Make colnames
rownames(hm_df3) <- hm_df3$DrugComb
hm_df3 <- hm_df3[, !(colnames(hm_df3) %in% c("DrugComb"))]

# Make hm_df3 as matrix
hm_matrix <- data.matrix(hm_df3)

# Select annotations
rownames(hm_df_row) <- hm_df_row$DrugComb
hm_row_factors <- factor(hm_df_row$Drug.Type.Pairs)

rownames(hm_df_col) <- hm_df_col$SIDM
hm_col_factors <- factor(hm_df_col$Tissue)

hm_pal = colorRampPalette(c("#427aa1", "#7ca982", "#ed6a5a"))(3)
colors = structure(hm_pal, names = c(1, 2, 3))

pdf(file= paste0(omics_output_path, "perturbation_heatmap.pdf"))

hm <- Heatmap(hm_matrix, name = "Screens", na_col ="grey95", cluster_columns=FALSE, cluster_rows = FALSE, 
              show_column_names = FALSE, show_row_names = FALSE, 
              col=colors,  rect_gp = gpar(col = "white", lwd = 0.001), 
              row_split = hm_row_factors, column_split = hm_col_factors, border = TRUE,
              column_title_rot = 90, row_title_rot = 0, row_title_gp = gpar(fontsize = 5), column_title_gp = gpar(fontsize = 5), 
              heatmap_legend_param = list(at = c(1, 2, 3), labels = c("Anchor", "Matrix", "Both")),
              use_raster = TRUE)

draw(hm)
dev.off()


##########################################################################################
### FIGURE 3-2

pathway_df <- read.csv(paste0(input_path, "figure_files/pathway_drug_combination_freq.csv"), sep=",")


g<- ggplot(pathway_df) +
  geom_point(aes(x = Drug1_pathway, y = Drug2_pathway, size = Number.of.Drug.Combinations, color=combination_category)) +
  scale_color_manual(name = "Combination Categories",
                     values = c("CHEM-CHEM" = "#277da1", "CD-CHEM" ="#577590", "CHEM-DDR"="#4d908e", "CHEM-CS"="#43aa8b",
                                "CD-CD"= "#90be6d", "CD-CS"="#f9c74f", "CD-DDR"="#f9844a", "CS-CS"="#f8961e",
                                "CS-DDR"="#f3722c","DDR-DDR"="#f94144"))+
  theme_bw() +
  theme(axis.text.x = element_text(size=rel(1.15)),
        axis.title = element_text(size=rel(1.15))) +
  xlab("Targeted Pathway") +
  ylab("Targeted Pathway") +
  theme(text = element_text(size = 10), axis.text.x = element_text(angle = 90, hjust = 1))

ggsave(
  filename=paste0(omics_output_path, "pathway_dot_heatmap.pdf"), plot = g, width = 24, height = 20, units = "cm", dpi = 300)
dev.off()


##########################################################################################
### FIGURE 3-3C pansolid

solid_df <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/pansolid_oncoprint.csv", sep = ',')
rownames(solid_df) <- solid_df$X
solid_df <- solid_df[, !(colnames(solid_df) %in% c("X"))]
solid_df[is.na(solid_df)] <- "    "


solid_df <- solid_df[,order(colnames(solid_df))]

col = c("DEL" = "#427aa1", "AMP" = "#7ca982", "MUT" = "#ed6a5a")
alter_fun = list(
  background = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h-unit(2, "pt"), 
              gp = gpar(fill = "grey95", col = NA))
  },
  MUT = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h*0.33, 
              gp = gpar(fill = col["MUT"], col = NA))
  },
  AMP = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h-unit(2, "pt"), 
              gp = gpar(fill = col["AMP"], col = NA))
  },
  DEL = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h-unit(2, "pt"), 
              gp = gpar(fill = col["DEL"], col = NA))
  }
)

alter_fun = list(
  background = alter_graphic("rect", fill = "grey95"),   
  DEL = alter_graphic("rect", fill = col["DEL"]),
  AMP = alter_graphic("rect", fill = col["AMP"]),
  MUT = alter_graphic("rect", height = 0.33, fill = col["MUT"])
)

solid_df_annot <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/pansolid_oncoprint_annotation.csv", sep = ',')
solid_df_annot <- solid_df_annot[order(solid_df_annot$X),]
rownames(solid_df_annot) <- solid_df_annot$X

tissue = solid_df_annot[, "Tissue"]
tissue_factors <- factor(solid_df_annot$Tissue)

tissue_colors = c(
  "Adrenal Gland"="#C1A72F",
  "Biliary Tract"="#9BABBA", 
  "Bladder"="#EBDEF0", 
  "Bone"= "#D98880",
  "Breast"="#F5CBA7",
  "Central Nervous System"="#D6DBDF",
  "Cervix"="#FBEEE9", 
  "Endometrium"= "#FAD7A0",
  "Esophagus"="#ABEBC6", 
  "Head and Neck"="#A084BD", 
  "Kidney"="#B1D0C1",
  "Large Intestine"="#F5B7B1",
  "Liver"="#ABDBDA", 
  "Lung"="#CE661E", 
  "Ovary"="#E5E8B3", 
  "Pancreas"="#F2D7D5",
  "Peripheral Nervous System"="#C2DEB4", 
  "Prostate"="#2874A6", 
  "Skin"="#FFB3BC", 
  "Small Intestine"="#D6DEB4", 
  "Soft Tissue"="#F89420", 
  "Stomach"="#1ABC9C", 
  "Testis"="#E5B6F3", 
  "Thyroid"="#C28B90", 
  "Uterus"="#FBE3C7", 
  "Vulva"="#B2509E") 

ha = HeatmapAnnotation(Tissue = tissue, 
                       col = list(Tissue = tissue_colors,show_legend=c(Tissue=FALSE),
                       annotation_height = unit(c(5, 5, 15), "mm"),
                       annotation_legend_param = list(Tissue = list(title = "Tissue"))))

column_title = "OncoPrint for Pan-solid cell lines"
heatmap_legend_param = list(title = "Alternations", at = c("DEL", "AMP", "MUT"), 
                            labels = c("Deletion", "Amplification", "Mutation"))

pdf(file= paste0(omics_output_path, "oncoprint_pansolid.pdf"), width=16, height=7)
op <- oncoPrint(solid_df, remove_empty_rows = TRUE, alter_fun = alter_fun, 
                col = col, bottom_annotation = ha, column_title = column_title, 
                heatmap_legend_param = heatmap_legend_param)


draw(op)
dev.off()


##########################################################################################
### FIGURE 3-3D Panliquid oncoprint

liquid_df <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/panliquid_oncoprint.csv", sep = ',')
rownames(liquid_df) <- liquid_df$X
liquid_df <- liquid_df[, !(colnames(liquid_df) %in% c("X"))]
liquid_df[is.na(liquid_df)] <- "    "

liquid_df <- liquid_df[,order(colnames(liquid_df))]

col = c("DEL" = "#427aa1", "AMP" = "#7ca982", "MUT" = "#ed6a5a")
alter_fun = list(
  background = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h-unit(2, "pt"), 
              gp = gpar(fill = "grey95", col = NA))
  },
  MUT = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h*0.33, 
              gp = gpar(fill = col["MUT"], col = NA))
  },
  AMP = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h-unit(2, "pt"), 
              gp = gpar(fill = col["AMP"], col = NA))
  },
  DEL = function(x, y, w, h) {
    grid.rect(x, y, w-unit(2, "pt"), h-unit(2, "pt"), 
              gp = gpar(fill = col["DEL"], col = NA))
  }
)

alter_fun = list(
  background = alter_graphic("rect", fill = "grey95"),   
  DEL = alter_graphic("rect", fill = col["DEL"]),
  AMP = alter_graphic("rect", fill = col["AMP"]),
  MUT = alter_graphic("rect", height = 0.33, fill = col["MUT"])
)

liquid_df_annot <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/panliquid_oncoprint_annotation.csv", sep = ',')
liquid_df_annot <- liquid_df_annot[order(liquid_df_annot$X),]
rownames(liquid_df_annot) <- liquid_df_annot$X

tissue = liquid_df_annot[, "Tissue"]
tissue_factors <- factor(liquid_df_annot$Tissue)

tissue_colors = c("Leukemia"="#2f6690", "Lymphoma"="#3a7ca5","Myeloma"="#81c3d7",
                  "Other liquid tumours"="#90e0ef")

ha = HeatmapAnnotation(Tissue = tissue, 
                       col = list(Tissue = tissue_colors, show_legend=c(Tissue=FALSE),
                                  annotation_height = unit(c(5, 5, 15), "mm"),
                                  annotation_legend_param = list(Tissue = list(title = "Tissue"))))

column_title = "OncoPrint for Pan-liquid cell lines"
heatmap_legend_param = list(title = "Alternations", at = c("DEL", "AMP", "MUT"), 
                            labels = c("Deletion", "Amplification", "Mutation"))

pdf(file= paste0(omics_output_path, "oncoprint_panliquid.pdf"), width=12, height=4)
op <- oncoPrint(liquid_df, remove_empty_rows = TRUE, alter_fun = alter_fun, 
                col = col, bottom_annotation = ha, column_title = column_title, 
                heatmap_legend_param = heatmap_legend_param)


draw(op)
dev.off()






##########################################################################################
### FIGURE 3-4A Colo Clinical

colo_clinical_df <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/clinical_colo_mutation.csv", sep = ',')
colo_clinical_col <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/clinical_colo_col_annotation.csv", sep = ',')
colo_clinical_col[colo_clinical_col == ""] <- NA

rownames(colo_clinical_df) <- colo_clinical_df$X
colo_clinical_df <- colo_clinical_df[, !(colnames(colo_clinical_df) %in% c("X"))]

colo_clinical_col <- colo_clinical_col[order(colo_clinical_col$X),]
rownames(colo_clinical_col) <- colo_clinical_col$X
colo_clinical_col <- colo_clinical_col[, !(colnames(colo_clinical_col) %in% c("X"))]


# Order same
colo_clinical_df <- colo_clinical_df[,order(colnames(colo_clinical_df))]

# Make hm_df3 as matrix
colo_clinical_m <- data.matrix(colo_clinical_df)

# Select annotations
msi_factors <- factor(colo_clinical_col$MSI)
cris_factors <- factor(colo_clinical_col$CRIS)
cms_factors <- factor(colo_clinical_col$CMS)


pal = colorRampPalette(c("grey95", "#ff002b"))(2)
colors = structure(pal, names = c(0, 1))


cms_pal = colorRampPalette(c("#eaac8b", "#e56b6f", "#b56576", "#6d597a", "white"))(5)
cms_colors = structure(cms_pal, names = c("CMS1", "CMS2", "CMS3", "CMS4", NA))

cris_pal = colorRampPalette(c("#bce784", "#5dd39e", "#348aa7", "#525174", "#513b56", "white"))(6)
cris_colors = structure(cris_pal, names = c("CRISA", "CRISB", "CRISC", "CRISD", "CRISE", NA))

pdf(file= paste0(omics_output_path, "heatmap_clinical_colo.pdf"), width=12, height=4)

col_annot = HeatmapAnnotation(CMS = cms_factors,CRIS = cris_factors, #MSI=msi_factors,
  col = list(CMS = c("CMS1"="#eaac8b", "CMS2" = "#e56b6f", "CMS3"= "#b56576", "CMS4"= "#6d597a"),
             CRIS = c("CRISA"="#bce784", "CRISB"= "#5dd39e", "CRISC"= "#348aa7", "CRISD"= "#525174", "CRISE"= "#513b56")),
             #MSI= c("MSS" = "grey80", "MSI" = "grey30")),
             na_col = "white",
  gp = gpar(col = "grey20"))

hm <- Heatmap(colo_clinical_m, name = "Mutation", na_col ="white", 
              cluster_columns=T, cluster_rows = FALSE, 
              show_column_names = TRUE, show_row_names = TRUE, 
              col=colors, 
              column_split = msi_factors, border = TRUE,
              rect_gp = gpar(col = "white", lwd = 1.5), 
              top_annotation = col_annot,
              column_title_rot = 0, row_title_rot = 0, 
              row_title_gp = gpar(fontsize = 10), column_title_gp = gpar(fontsize = 10), 
              row_names_gp = grid::gpar(fontsize = 8), column_names_gp = grid::gpar(fontsize = 8),
              width = ncol(colo_clinical_m)*unit(5, "mm"), 
              height = nrow(colo_clinical_m)*unit(5, "mm"),
              use_raster = TRUE)

draw(hm)
dev.off()


##########################################################################################
### FIGURE 3-4B Breast Clinical

breast_clinical_df <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/clinical_breast_omics.csv", 
                               sep = ',')

rownames(breast_clinical_df) <- breast_clinical_df$X
breast_clinical_df <- breast_clinical_df[, !(colnames(breast_clinical_df) %in% c("X"))]

breast_clinical_df <- as.data.frame(x= t(breast_clinical_df), stringsAsFactors = FALSE)

breast_clinical_col <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/figure_files/clinical_breast_col_annotation.csv", sep = ',')
breast_clinical_col[breast_clinical_col == ""] <- NA

breast_clinical_col <- breast_clinical_col[order(breast_clinical_col$X),]
rownames(breast_clinical_col) <- breast_clinical_col$X
breast_clinical_col <- breast_clinical_col[, !(colnames(breast_clinical_col) %in% c("X"))]


# Order same
breast_clinical_df <- breast_clinical_df[,order(colnames(breast_clinical_df))]

# Make hm_df3 as matrix
breast_clinical_m <- data.matrix(breast_clinical_df)

# Select annotations
pam50_factors <- factor(breast_clinical_col)
omics_factors <- factor(c("BRCA1" ="Mutation", "BRCA2"= "Mutation", "PIK3CA"= "Mutation", "PTEN"= "Mutation", "TP53"= "Mutation", 
                          "ESR1" ="GEx", "PGR"= "GEx", "ERBB2"= "GEx"))

pal = colorRamp2(c(0.0, 1.0), c("grey95", "#ff002b"))

pdf(file= paste0(omics_output_path, "heatmap_clinical_breast.pdf"), width=15, height=4)

col_annot = HeatmapAnnotation(PAM50 = pam50_factors,
                              col = list(PAM50 = c("Basal"="#f07167", "Her2" = "#b6ccfe", "LumA"= "#ffdab9", "LumB"= "#f7e1d7")),
                              na_col = "white",
                              gp = gpar(col = "grey20"))

hm <- Heatmap(breast_clinical_m, name = "Omics", na_col ="white", 
              cluster_columns=T, cluster_rows = FALSE, 
              show_column_names = TRUE, show_row_names = TRUE, 
              rect_gp = gpar(col = "white", lwd = 1.5), 
              top_annotation = col_annot, col=pal,
              row_split = omics_factors, border=T, column_split = pam50_factors,
              column_title_rot = 0, row_title_rot = 0, 
              row_title_gp = gpar(fontsize = 10), column_title_gp = gpar(fontsize = 10), 
              row_names_gp = grid::gpar(fontsize = 8), column_names_gp = grid::gpar(fontsize = 8),
              width = ncol(breast_clinical_m)*unit(5, "mm"), 
              height = nrow(breast_clinical_m)*unit(5, "mm"),
              use_raster = TRUE)

draw(hm)
dev.off()








drug_path_df = read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/data/drug_info/manually_grouped_drugs.csv", sep=",")
drug_paths = drug_path_df[, (colnames(drug_path_df) %in% c("drug_name", "general_targeted_pathway"))]
drug_cats = drug_path_df[, (colnames(drug_path_df) %in% c("drug_name", "drug_category"))]
rownames(drug_paths) <- drug_paths$drug_name
rownames(drug_cats) <- drug_cats$drug_name

pathway_ratio_df <- read.csv("/Volumes/team215_pipeline/Cansu/CombDrug/output/network/figure_files/module_pathway_ratios.csv", sep=",", header=T)
#pathway_ratio_df <- as.data.frame(x= t(pathway_ratio_df), stringsAsFactors = FALSE)

# Order same
rownames(pathway_ratio_df) <- pathway_ratio_df$X
pathway_ratio_df <- pathway_ratio_df[, !(colnames(pathway_ratio_df) %in% c("X"))]

pathway_ratio_df_mat <- data.matrix(pathway_ratio_df)

drug_paths <- drug_paths[rownames(drug_paths) %in% rownames(pathway_ratio_df),]
drug_cats <- drug_cats[rownames(drug_cats) %in% rownames(pathway_ratio_df),]

idx <- match(rownames(pathway_ratio_df), drug_paths$drug_name)
drug_paths <- drug_paths[idx,]
drug_cats <- drug_cats[idx,]
drug_paths <- drug_paths[, !(colnames(drug_paths) %in% c("drug_name"))]
drug_cats <- drug_cats[, !(colnames(drug_cats) %in% c("drug_name"))]
pathway_factors <- factor(drug_paths)
category_factors <- factor(drug_cats)

pal = colorRamp2(c(0.0, 1.0), c("grey95", "#ff002b"))

col_annot = HeatmapAnnotation(
  Pathway = pathway_factors, na_col = "white", gp = gpar(col = "grey20"), 
  Category=category_factors,
  col = list(Category = c("Cell Death"="#b56576", "Cell Signalling" = "#b6ccfe", 
                          "Chemotherapeutic"="#E5E8B3", "DNA Damage Response"= "#5dd39e"),
             Pathway = c("Mitosis" = "#bce784", "Apoptosis regulation" = "#F5B7B1",
                         "Genome integrity"="#ABEBC6", "Other signalling" = "#90e0ef",
                         "Chromatin histone acetylation" ="#EBDEF0", "Cell cycle"="#ABDBDA",
                         "DNA replication"="#D6DBDF", "RTK signalling"="#277da1",
                         "p53 pathway"="#f07167", "Chromatin other"="#FFD9C1",
                         "Protein stability and degradation"="#b56556")))

pdf(file= paste0("/Volumes/team215_pipeline/Cansu/CombDrug/output/network/figures/pathway_KEGG_ratios.pdf"),
    width=12, height=9)
hm <- Heatmap(pathway_ratio_df_mat, name = "Pathway Ratio", col = pal,
        cluster_columns=T, cluster_rows = T, 
        top_annotation = col_annot,
        show_column_names = T, show_row_names = T, 
        rect_gp = gpar(col = "white", lwd = 0.1),
        column_names_gp = grid::gpar(fontsize = 6),row_dend_reorder = TRUE,
        row_names_gp = grid::gpar(fontsize = 6), use_raster = TRUE)

draw(hm)
dev.off()






