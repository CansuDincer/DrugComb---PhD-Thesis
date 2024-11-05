#Author : Cansu Dincer
#Date : 02 September 2022
#Last Update : 18 January 2024
#Input : Gene Lists
#Output : ORA (Over-Representation Analysis) or GSEA analysis 
# ------------------------------------------------------------------------#

# Load the packages and paths

path_base <- "/nfs/team215_pipeline/Cansu/CombDrug/"
input_path <- paste0(path_base, "output/network/")

#install.packages("WebGestaltR")
library("WebGestaltR")
library("dplyr")
library("Xmisc")


# ------------------------------------------------------------------------#

# Load the input
require(Xmisc)
parser <- ArgumentParser$new()
parser$add_description('Pathway Analysis - WebgGestaltR')
parser$add_argument('--interactome',type='character', help='The name of the interactome')
parser$add_argument('--drug',type='character', help='The Sanger ID of the drug')
parser$add_argument('--enrichment_method',type='character', help='ORA or GSEA')
parser$add_argument('--a',type='character', help='dampling factor')
parser$add_argument('--random',type='character', help='if the modelling on random interactome or not')
parser$add_argument('--s',type='character', help='seed number of the random interactome')

args <- parser$get_args()
interactome <- args$interactome
drug <- args$drug
enrichment_method <- args$enrichment_method
alpha <- args$a
random_tag <- args$random
seed <- args$s


if(random_tag == "yes") {
	random_text<- paste0("_random_", str(seed))
	random_col<- "/random/"
	folder <- paste0("/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/network/modelling/PPR/drugs/random/modules/", drug, "/")
} else {
	random_text<- "_empiric"
	random_col<- "/empiric/"
	folder <- paste0(input_path, "modelling/PPR/drugs/", interactome_name, "/drugs/empiric/", drug)
}

alpha_text <- as.character(as.integer(as.double(alpha) * 100))


pathway_path<- paste0(input_path, "modelling/PPR/", interactome, "/drugs/pathway_analysis/webgestaltr/analysis/", drug, "/")
dir.create(file.path(pathway_path), showWarnings = FALSE)

node_path<- paste0(input_path, "modelling/PPR/", interactome, "/drugs/pathway_analysis/webgestaltr/gene_lists/")
dir.create(file.path(node_path), showWarnings = FALSE)

# ------------------------------------------------------------------------#

# Retrieving the gene list

f <- NaN
if(random_tag == "yes") {
	if(file.exists(paste0(folder, "ppr_network_", drug, "_", alpha_text, random_text, ".csv"))) {

		f<- read.table(file=paste0(folder, "ppr_network_", drug, "_", alpha_text, random_text, ".csv"), sep =",", header=TRUE)}
	}
if(random_tag == "no"){
	if(file.exists(paste0(folder, "ppr_network_", drug, "_", alpha_text, random_text, ".csv"))) {

		f<- read.table(file=paste0(folder, "ppr_network_", drug, "_", alpha_text, random_text, ".csv"), sep =",", header=TRUE)}
}

if(is.data.frame(f) == TRUE) {
	genes1 <- f %>% select(From) %>% unique()
	colnames(genes1) <- "genes"

	genes2 <- f %>% select(To) %>% unique()
	colnames(genes2) <- "genes"

	genes <- genes1 %>% full_join(genes2)

	write.table(x=genes, file=paste0(node_path, drug, "_", alpha_text, "_module_gene_set_list.txt"),
				col.names = FALSE, row.names = FALSE, sep="\t")

	file_name <- paste0(node_path, drug, "_", alpha_text, "_module_gene_set_list.txt")
} else {
		file_name <- NaN}


# ------------------------------------------------------------------------#

# Reference gene list

ref_gene_file <- paste0(input_path, "modelling/PPR/", interactome, "/drugs/pathway_analysis/webgestaltr/reference_nodes.txt")


#` ------------------------------------------------------------------------#

# Pathway analysis

if(is.na(file_name) == FALSE) {
	for(db in c("pathway_Reactome", "pathway_Wikipathway")) {
		#, "pathway_Panther", "pathway_Wikipathway")) {

		result <- WebGestaltR(enrichMethod = enrichment_method, organism = "hsapiens",
							  enrichDatabase = db, interestGeneType="genesymbol",
							  referenceGeneType = "genesymbol",
							  referenceGeneFile = ref_gene_file,
							  interestGeneFile = file_name, nThreads = 20, minNum = 2,
							  maxNum = 500, sigMethod = "fdr", fdrMethod = "BH",
							  fdrThr = 0.05, isOutput = FALSE)

		outputpath <- paste0(pathway_path, db, "_", drug, "_", alpha_text, "module_pathways.txt")

		write.table(result, outputpath, sep = "\t")
	}
}


