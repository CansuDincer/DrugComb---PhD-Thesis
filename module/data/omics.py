"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 07 February 2023
Last Update : 1 May 2024
Output : Prepared Omics files
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#

import os, sys, pandas, pickle, numpy, scipy, sklearn, sys, matplotlib, networkx, re, warnings
from sklearn.preprocessing import MinMaxScaler, quantile_transform
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)

from CombDrug.module.path import *
from CombDrug.module.data.drug import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.responses import *


# ---------------------------------------------------------------------------#
#                                   Omics                              	 	 #
# ---------------------------------------------------------------------------#

def get_drug_targets():
	if "drug_targets.p" not in os.listdir(output_path + "data/drug_info/"):
		drug_targets = list()
		for d in get_drug_names():
			x = Drug(d).targets
			if x is not None:
				for t in x:
					if t not in drug_targets:
						drug_targets.append(t)
		pickle.dump(drug_targets, open(output_path + "data/drug_info/drug_targets.p", "wb"))
	else:
		drug_targets = pickle.load(open(output_path + "data/drug_info/drug_targets.p", "rb"))

	return drug_targets


def get_cancer_genes():
	"""
	Retrieving all the cancer related genes from Cell Model Passport
	:return: list of cancer genes
	"""
	# Cancer Driver Genes
	cancer_genes = x_driver_genes()
	return list(cancer_genes["symbol"].unique())


def write_table_cancer_drug_genes():

	if "cancer_drug_target_genes.csv" not in os.listdir(output_path + "data/omics"):
		targets, cancers = get_drug_targets(), get_cancer_genes()
		df = pandas.DataFrame(index=set(cancers).union(set(targets)), columns = ["Type"])
		df["Type"] = df.apply(lambda x: "Drug target gene" if x.name in targets and x.name not in cancers else (
			"Cancer gene" if x.name in cancers and x.name not in targets else (
				"Both" if x.name in cancers and x.name in targets else None)), axis=1)

		df.to_csv(output_path + "data/omics/cancer_drug_target_genes.csv", index=True)
	else:
		df = pandas.read_csv(output_path + "data/omics/cancer_drug_target_genes.csv", index_col=0)

	return df


def x_mutation():
	"""
	Get mutation information
	:return: Mutation data from CMP for the cell lines perturbed
	"""
	if "omics_mutation.csv" not in os.listdir(output_path + "data/omics/"):
		mutations = pandas.read_csv(input_path + "cell_model_passports/mutations_all_20230202.csv")

		# Remove silent mutations or mutations which are not hit the protein coding sites
		mutations = mutations[~mutations.protein_mutation.isin(["p.?", "-"])]
		mutations = mutations[mutations.effect != "silent"]

		# Cancer Models
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		mutations = mutations[mutations.model_id.isin(sidms)]
		sanger = mutations[mutations.source == "Sanger"]
		sanger_data = list(sanger.groupby(["model_id", "gene_symbol", "protein_mutation"]).groups.keys())
		broad = mutations[mutations.source == "Broad"]
		broad_data = list(broad.groupby(["model_id", "gene_symbol", "protein_mutation"]).groups.keys())
		broad_specific = set(broad_data).difference(set(sanger_data))
		broad2 = pandas.concat([broad.groupby(["model_id", "gene_symbol", "protein_mutation"]).get_group(g) for g in broad_specific])
		mutations = pandas.concat([sanger, broad2])
		mutations = mutations[[col for col in mutations.columns if col not in ["coding", "gene_id", "rna_mutation", "cdna_mutation", "vaf", "source"]]]
		mutations.to_csv(output_path + "data/omics/omics_mutation.csv", index=False)
	else:
		mutations = pandas.read_csv(output_path + "data/omics/omics_mutation.csv")
	return mutations


def x_driver_genes():
	"""
	Get driver genes from CMP
	:return: Driver gene data from CMP
	"""
	mutation_moa = pandas.read_csv(input_path + "cell_model_passports/driver_genes_20221018.csv")
	return mutation_moa


def x_mutation_moa():
	"""
	Get mutation mode of action information
	:return: Driver gene data from CMP
	"""
	if "omics_mutation_moa.csv" not in os.listdir(output_path + "data/omics/"):
		mutation_moa = x_driver_genes()
		mutation_moa = mutation_moa[["symbol", "method_of_action"]]
		mutation_moa.to_csv(output_path + "data/omics/omics_mutation_moa.csv", index=False)
	else:
		mutation_moa = pandas.read_csv(output_path + "data/omics/omics_mutation_moa.csv")
	return mutation_moa


def x_GEx():
	if "omics_transcription_log2tpm.csv" not in os.listdir(output_path + "data/omics/"):
		# Cancer Gene Expressions (TPMs)
		transcription = pandas.read_csv(input_path + "cell_model_passports/rnaseq_tpm_20220624.csv")
		transcription = transcription.T
		transcription.columns = transcription.loc["Unnamed: 1"]
		transcription = transcription[2:]
		transcription = transcription[[col for col in transcription.columns if col not in [numpy.nan, "symbol"]]]
		transcription = transcription.reset_index()

		# Cancer Models
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		transcription = transcription[transcription["index"].isin(sidms)]

		# Prepare gene names
		transcription = transcription.set_index(["index"])
		transcription = transcription.rename_axis(None, axis=1)
		transcription = transcription.rename_axis(None, axis=0)

		# Transpose
		transcription = transcription.T

		# TPM values
		transcription = transcription.astype(float)

		# Log2
		log_transcription = numpy.log2(transcription + 1)

		log_transcription.to_csv(output_path + "data/omics/omics_transcription_log2tpm.csv", index=True)
	else:
		log_transcription = pandas.read_csv(output_path + "data/omics/omics_transcription_log2tpm.csv", index_col=0)
	return log_transcription


def x_GEx_count():
	if "omics_transcription_qn_counts.csv" not in os.listdir(output_path + "data/omics/"):
		# Cancer Gene Expressions (TPMs)
		transcription = pandas.read_csv(input_path + "cell_model_passports/rnaseq_read_count_20220624.csv")
		transcription = transcription.T
		transcription.columns = transcription.loc["Unnamed: 1"]
		transcription = transcription[2:]
		transcription = transcription[[col for col in transcription.columns if col not in [numpy.nan, "symbol"]]]
		transcription = transcription.reset_index()

		# Cancer Models
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		transcription = transcription[transcription["index"].isin(sidms)]

		# Prepare gene names
		transcription = transcription.set_index(["index"])
		transcription = transcription.rename_axis(None, axis=1)
		transcription = transcription.rename_axis(None, axis=0)

		# Transpose
		transcription = transcription.T

		# Counts
		transcription = transcription.astype(float)

		# Log2
		qn_transcription = pandas.DataFrame(quantile_transform(transcription),
											columns = transcription.columns,
											index= transcription.index)

		qn_transcription.to_csv(output_path + "data/omics/omics_transcription_qn_counts.csv", index=True)
	else:
		qn_transcription = pandas.read_csv(output_path + "data/omics/omics_transcription_qn_counts.csv", index_col=0)
	return qn_transcription


def x_PEx():
	if "omics_proteome_log2.csv" not in os.listdir(output_path + "data/omics/"):
		# Averaged data
		proteomics = pandas.read_csv(input_path + "cell_model_passports/Protein_matrix_averaged_20221214.tsv",sep="\t", index_col=0).T
		proteomics = proteomics.set_index(["symbol"])
		proteomics.columns = proteomics.loc[numpy.nan]

		proteomics = proteomics.rename_axis(None, axis=1)
		proteomics =proteomics.reset_index()
		proteomics = proteomics.loc[1:]
		proteomics = proteomics.set_index(["symbol"])
		proteomics = proteomics.rename_axis(None, axis=0)
		proteomics = proteomics[[col for col in proteomics.columns if col not in ["model_id"]]]

		# Cancer Models
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		proteomics = proteomics[[i for i in sidms if i in proteomics.columns]]

		# Averaged values
		proteomics = proteomics.astype(float)

		proteomics.to_csv(output_path + "data/omics/omics_proteome_log2.csv", index=True)
	else:
		proteomics = pandas.read_csv(output_path + "data/omics/omics_proteome_log2.csv", index_col=0)
	return proteomics


def x_CNV_binary():
	if "omics_cnv_binary.csv" not in os.listdir(output_path + "data/omics/"):
		# Cancer Copy Number Variations (pureCN) - Gene
		cnv_category = pandas.read_csv(
			input_path + "cell_model_passports/WES_pureCN_CNV_genes_cn_category_20221213.csv")
		cnv_category = cnv_category.T
		cnv_category.columns = cnv_category.loc["model_name"]
		cnv_category = cnv_category.reset_index()

		sanger = cnv_category[cnv_category.source == "Sanger"]
		broad = cnv_category[cnv_category.source == "Broad"]
		broad = broad[broad.model_id.isin(list(set(broad.model_id.unique()).difference(set(sanger.model_id.unique()))))]
		cnv_category = pandas.concat([sanger, broad])
		cnv_category = cnv_category[[col for col in cnv_category.columns if col not in ["index", "symbol", "source"]]]

		# Cancer Models
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		cnv_category = cnv_category[cnv_category["model_id"].isin(sidms)]
		cnv_category = cnv_category.set_index(["model_id"])

		cnv_category = cnv_category.rename_axis(None, axis=0)
		cnv_category = cnv_category.rename_axis(None, axis=1)

		cnv_category = cnv_category.replace("Neutral", 0)
		cnv_category = cnv_category.replace("Loss", -0.5)
		cnv_category = cnv_category.replace("Deletion", -1)
		cnv_category = cnv_category.replace("Gain", 0.5)
		cnv_category = cnv_category.replace("Amplification", 1)
		cnv_category = cnv_category.T

		cnv_category.to_csv(output_path + "data/omics/omics_cnv_binary.csv", index=True)
	else:
		cnv_category = pandas.read_csv(output_path + "data/omics/omics_cnv_binary.csv",index_col=0)
	return cnv_category


def x_mehylation():
	if "omics_methylation.csv" not in os.listdir(output_path + "data/omics/"):
		# Methylation files Table S2J and S2H
		# Ac.num GSE68379 (Mapping between cell line cosmic identifiers and methylation data sample identifiers)
		# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68379
		table_s2j = pandas.read_csv(input_path + "cell_model_passports/methylation_s2j.csv")[["Sample", "HyperMethylated iCpG"]]
		table_s2h = pandas.read_csv(input_path + "cell_model_passports/methylation_s2h.csv")

		table_s2j.columns = ["model_name", "hypermethy_icpg"]
		table_s2j["affected_genes"] = table_s2j.apply(
			lambda x: table_s2h[table_s2h["Genomic Coordinates"] == x.hypermethy_icpg]["GN"].values[
				0] if x.hypermethy_icpg in table_s2h["Genomic Coordinates"].values else None, axis=1)

		table_s2j["genes"] = table_s2j.apply(lambda x: x.affected_genes, axis=1)

		table_s2j = table_s2j[table_s2j.genes != ""][["model_name", "genes"]]
		gene_list = [gene for genes in table_s2j.genes for gene in genes.split("; ")]
		table_s2j["sanger_id"] = table_s2j.apply(lambda x: CellLine(x.model_name).id, axis=1)

		# Cancer Models
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		table_s2j = table_s2j[table_s2j.sanger_id.isin(sidms)]

		methylation = pandas.DataFrame(index=table_s2j.sanger_id.unique(), columns=gene_list)
		for ind, row in methylation.iterrows():
			for gene in methylation.columns:
				if len(table_s2j[(table_s2j.sanger_id == ind)]) > 0:
					x = table_s2j[(table_s2j.sanger_id == ind)]
					x_genes = [gene for genes in x.genes for gene in genes.split("; ")]
					if gene in x_genes:
						methylation.loc[ind, gene] = 1
					else:
						methylation.loc[ind, gene] = 0

		methylation = methylation.T
		methylation = methylation.drop_duplicates()

		methylation.to_csv(output_path + "data/omics/omics_methylation.csv", index=True)
	else:
		methylation = pandas.read_csv(output_path + "data/omics/omics_methylation.csv", index_col=0)
	return methylation


def x_crispr():
	if "omics_crispr.csv" not in os.listdir(output_path + "data/omics/"):
		# Read the essentiality file (Sanger + Broad)
		crispr = pandas.read_csv(
			input_path + "cell_model_passports/Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_scaled_bayesian_factors_20240111.tsv",
			sep="\t")
		crispr = crispr.T
		crispr.columns = crispr.loc["model_name"]
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		crispr = crispr[crispr.model_id.isin(sidms)]
		crispr = crispr[crispr.qc_pass == "TRUE"]

		# There are two values per cell line and gene both from Sanger and Broad
		# We will continue with the Sanger ones  when there are both of them
		sanger = crispr[crispr.source == "Sanger"]
		broad = crispr[crispr.source == "Broad"]
		broad = broad[broad.model_id.isin(list(set(broad.model_id.unique()).difference(set(sanger.model_id.unique()))))]
		crispr = pandas.concat([sanger, broad])

		crispr = crispr.reset_index()
		crispr = crispr.rename_axis(None, axis=1)
		crispr = crispr.set_index("model_id")
		crispr = crispr.rename_axis(None, axis=0)
		crispr = crispr[[col for col in crispr.columns if
						 col not in ["source", "qc_pass", "symbol", "index", "model"]]]
		crispr = crispr.T
		crispr.to_csv(output_path + "data/omics/omics_crispr.csv", index=True)
	else:
		crispr = pandas.read_csv(output_path + "data/omics/omics_crispr.csv", index_col=0, low_memory=False)
	return crispr


def x_crispr_logfc():
	if "omics_crispr_logfc.csv" not in os.listdir(output_path + "data/omics/"):
		# Read the essentiality file (Sanger + Broad)
		crispr = pandas.read_csv(
			input_path + "cell_model_passports/Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_fold_change_values_20240111.tsv",
			sep="\t")
		crispr = crispr.T
		crispr.columns = crispr.loc["model_name"]
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		crispr = crispr[crispr.model_id.isin(sidms)]
		crispr = crispr[crispr.qc_pass == "TRUE"]

		# There are two values per cell line and gene both from Sanger and Broad
		# We will continue with the Sanger ones  when there are both of them
		sanger = crispr[crispr.source == "Sanger"]
		broad = crispr[crispr.source == "Broad"]
		broad = broad[broad.model_id.isin(list(set(broad.model_id.unique()).difference(set(sanger.model_id.unique()))))]
		crispr = pandas.concat([sanger, broad])

		crispr = crispr.reset_index()
		crispr = crispr.rename_axis(None, axis=1)
		crispr = crispr.set_index("model_id")
		crispr = crispr.rename_axis(None, axis=0)
		crispr = crispr[[col for col in crispr.columns if
						 col not in ["source", "qc_pass", "symbol", "index", "model"]]]
		crispr = crispr.T
		crispr.to_csv(output_path + "data/omics/omics_crispr_logfc.csv", index=True)
	else:
		crispr = pandas.read_csv(output_path + "data/omics/omics_crispr_logfc.csv", index_col=0, low_memory=False)
	return crispr


def x_essential():
	if "omics_essentials.csv" not in os.listdir(output_path + "data/omics/"):
		essentiality = pandas.read_csv(
			input_path + "cell_model_passports/Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_binary_matrix_20240111.tsv",
			sep="\t")

		essentiality = essentiality.T
		essentiality.columns = essentiality.loc["model_name"]
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		essentiality = essentiality[essentiality.model_id.isin(sidms)]
		essentiality = essentiality[essentiality.qc_pass == "TRUE"]

		sanger = essentiality[essentiality.source == "Sanger"]
		broad = essentiality[essentiality.source == "Broad"]
		broad = broad[broad.model_id.isin(list(set(broad.model_id.unique()).difference(set(sanger.model_id.unique()))))]
		essentiality = pandas.concat([sanger, broad])

		essentiality = essentiality.reset_index()
		essentiality = essentiality.rename_axis(None, axis=1)
		essentiality = essentiality.set_index("model_id")
		essentiality = essentiality.rename_axis(None, axis=0)
		essentiality = essentiality[[col for col in essentiality.columns if
											   col not in ["source", "qc_pass", "symbol", "index"]]]
		essentiality = essentiality.T

		essentiality.to_csv(output_path + "data/omics/omics_essentials.csv", index=True)

	else:
		essentiality = pandas.read_csv(output_path + "data/omics/omics_essentials.csv", index_col=0, low_memory=False)
	return essentiality


def x_msi():

	if "msi_binary.csv" not in os.listdir(output_path + "data/omics/"):
		msi_df = cancer_models()[["sanger_id", "msi_status"]]
		msi_df = msi_df.dropna()
		msi_df = msi_df.replace("MSS", 0)
		msi_df = msi_df.replace("MSI", 1)

		# Cancer Models
		colo_sidms = get_sidm_tissue(tissue_type="Large Intestine")
		endometrium_sidms = get_sidm_tissue(tissue_type="Endometrium")
		ovary_sidms = get_sidm_tissue(tissue_type="Ovary")
		stomach_sidms = get_sidm_tissue(tissue_type="Stomach")
		sidms = colo_sidms + endometrium_sidms + ovary_sidms + stomach_sidms

		msi_df = msi_df[msi_df.sanger_id.isin(sidms)]
		msi_df = msi_df.drop_duplicates()

		msi_df = msi_df.set_index(["sanger_id"])
		msi_df.columns = ["MSI"]
		msi_df.to_csv(output_path + "data/omics/msi_binary.csv", index=True)
	else:
		msi_df = pandas.read_csv(output_path + "data/omics/msi_binary.csv", index_col =0)
	return msi_df


def x_variance_gex():
	if "omics_transcription_log2tpm_Z_filtered.csv" not in os.listdir(output_path + "data/omics/"):

		gex = x_GEx()

		# Cancer Genes
		cancer_genes = get_cancer_genes()
		drug_targets = get_drug_targets()
		genes = list(set(cancer_genes).union(set(drug_targets)))

		g = list(set(gex.index).intersection(set(genes)))
		filt_transcription = gex.loc[g]

		# Filtered genes according to the variance across cell lines
		t = filt_transcription.T
		transcription_variance = pandas.DataFrame(t.std())
		transcription_variance.columns = ["sd"]
		sd_outlier_thr = test_Z(values=list(transcription_variance["sd"]), pvalue=0.05, selected_side="right")

		if sd_outlier_thr is not None:
			transcription_variance["sd_selection"] = transcription_variance.apply(
				lambda x: True if x.sd >= sd_outlier_thr else False, axis=1)
		else:
			transcription_variance["sd_selection"] = False

		transcription_variance = transcription_variance.sort_values(by=["sd"])
		selected_genes = transcription_variance[transcription_variance.sd_selection].index

		transcription2 = filt_transcription.loc[selected_genes]

		transcription2.to_csv(output_path + "data/omics/omics_transcription_log2tpm_Z_filtered.csv", index=True)

	else:
		transcription2 = pandas.read_csv(output_path + "data/omics/omics_transcription_log2tpm_Z_filtered.csv", index_col=0)

	return transcription2


def x_loss():
	"""
	LOSS
	--> If gene is Tumour suppressor
		1. Check CNV as lost | gene expression log2(TPM + 1) <1
		2. Check if the gene has any cancer driver mutations
	--> If gene is Ambiguous
		1. Check CNV as lost | gene expression log2(TPM + 1) <1
		2. Check if protein has driver mutations as frameshift, nonsense, start lost
	"""

	if "loss_binary.csv" not in os.listdir(output_path + "data/omics/"):

		# First collect Tumour Suppressor
		moa = x_mutation_moa()
		tsg = list(moa[moa.method_of_action == "LoF"].symbol.unique()) # 257 TSGs
		ambiguous = list(moa[moa.method_of_action == "ambiguous"].symbol.unique()) # 201 ambiguous

		models = all_screened_SIDMs(project_list=None, integrated=True)
		loss_df = pandas.DataFrame(0, index=models, columns = tsg + ambiguous)

		# Get Omics
		cnv, gex, mut = x_CNV_binary(), x_variance_gex(), x_mutation()

		for model_id in loss_df.index:
			if model_id in cnv.columns:
				cnv_depleted_genes = list(cnv[cnv[model_id] == -1].index)
			else: cnv_depleted_genes = list()
			if model_id in gex.columns:
				gex_not_expressed_genes = list(gex[gex[model_id] < 1].index)
			else: gex_not_expressed_genes=list()
			if model_id in mut.model_id.unique():
				mut_driver_lof = list(mut[(mut.model_id == model_id) & mut.cancer_driver].gene_symbol.unique())
				mut_driver_amb = list(mut[(mut.model_id == model_id) & (mut.effect.isin(
					["frameshift", "nonsense", "start_lost", "ess_splice", "stop_lost"]))].gene_symbol.unique())
			else:
				mut_driver_lof, mut_driver_amb = list(), list()

			mut_loss_genes = list(set(mut_driver_lof).intersection(set(tsg)).union(set(mut_driver_amb).intersection(set(ambiguous))))
			cnv_loss_genes = list(set(cnv_depleted_genes).intersection(set(tsg + ambiguous)))
			gex_loss_genes = list(set(gex_not_expressed_genes).intersection(set(tsg + ambiguous)))
			lost_genes = list(set(cnv_loss_genes).union(set(gex_loss_genes).union(set(mut_loss_genes))))

			loss_df.loc[model_id, lost_genes] = 1
		loss_df = loss_df.T

		loss_df.to_csv(output_path + "data/omics/loss_binary.csv", index=True)
	else:
		loss_df = pandas.read_csv(output_path + "data/omics/loss_binary.csv", index_col=0)

	return loss_df


def x_gain():
	"""
	GAIN
	--> If gene is Oncogene
		1. Check CNV as gain | gene expression log2(TPM + 1) > 5
		2. Check if the gene has any cancer driver mutations
	--> If gene is Ambiguous
		1. Check CNV as gain | gene expression log2(TPM + 1) > 5
	"""

	if "gain_binary.csv" not in os.listdir(output_path + "data/omics/"):

		# First collect Oncogenes
		moa = x_mutation_moa()
		onc = list(moa[moa.method_of_action == "Act"].symbol.unique())  # Oncogenes
		ambiguous = list(moa[moa.method_of_action == "ambiguous"].symbol.unique())  # 201 ambiguous

		models = all_screened_SIDMs(project_list=None, integrated=True)
		gain_df = pandas.DataFrame(0, index=models, columns=onc + ambiguous)

		# Get Omics
		cnv, gex, mut, loss = x_CNV_binary(), x_GEx(), x_mutation(), x_loss()

		for model_id in gain_df.index:
			if model_id in cnv.columns:
				cnv_gained_genes = list(cnv[cnv[model_id] == 1].index)
			else: cnv_gained_genes = list()
			if model_id in gex.columns:
				gex_expressed_genes = list(gex[gex[model_id] >= 5].index)
			else: gex_expressed_genes = list()
			if model_id in mut.model_id.unique():
				mut_driver_act = list(mut[(mut.model_id == model_id) & mut.cancer_driver].gene_symbol.unique())
				mut_driver_amb = list(mut[(mut.model_id == model_id) & (~mut.effect.isin(
					["frameshift", "nonsense", "start_lost", "ess_splice", "stop_lost"]))].gene_symbol.unique())
			else: mut_driver_act = list()

			mut_gain_genes = list(set(mut_driver_act).intersection(set(onc)).union(set(mut_driver_amb).intersection(set(ambiguous))))
			cnv_gain_genes = list(set(cnv_gained_genes).intersection(set(onc)))
			gex_gain_genes = list(set(gex_expressed_genes).intersection(set(onc)))

			# Loss genes for corresponding cell line
			loss_genes = list(loss[loss[model_id] == 1].index)

			# Gained genes
			initial_gained_genes = list(set(cnv_gain_genes).union(set(gex_gain_genes).union(set(mut_gain_genes))))
			gained_genes = set(initial_gained_genes).difference(set(loss_genes))
			gain_df.loc[model_id, gained_genes] = 1
		gain_df = gain_df.T

		gain_df.to_csv(output_path + "data/omics/gain_binary.csv", index=True)
	else:
		gain_df = pandas.read_csv(output_path + "data/omics/gain_binary.csv", index_col=0)

	return gain_df


def x_subgroups():

	if "omics_subgroup.csv" not in os.listdir(output_path + "data/omics/"):

		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		subgroup_df = pandas.DataFrame(0, index=sidms, columns=["CRISA", "CRISB", "CRISC", "CRISD", "CRISE",
																"CMS1", "CMS2", "CMS3", "CMS4",
																"Basal", "Her2", "LumA", "LumB"])

		for model_id in sidms:
			obj = CellLine(sanger2model(model_id))
			if obj.tissue == "Breast":
				if pandas.isna(obj.get_breast_subgroup("sanger")) is False:
					subgroup_df.loc[model_id, obj.breast_subgroup] = 1
			elif obj.tissue == "Large Intestine":
				if pandas.isna(obj.get_colon_subgroup()[1]) is False:
					subgroup_df.loc[model_id, obj.get_colon_subgroup()[1]] = 1
				if pandas.isna(obj.get_colon_subgroup()[0]) is False:
					subgroup_df.loc[model_id, obj.get_colon_subgroup()[0]] = 1

		subgroup_df.to_csv(output_path + "data/omics/omics_subgroup.csv", index=True)
	else:
		subgroup_df = pandas.read_csv(output_path + "data/omics/omics_subgroup.csv", index_col=0)

	return subgroup_df


def x_availability():

	if "omics_availability.csv" not in os.listdir(output_path + "data/omics/"):
		sidms = all_screened_SIDMs(project_list=None, integrated=True)
		availability = pandas.DataFrame(0, columns = sidms,
										index=["Pan", "Tissue", "Mutation", "GEx", "PEx", "CNV", "Methylation",
											   "CRISPR", "Drug Combinations"])
		mut, exp, pxp, cnv, met, cris = x_mutation(), x_GEx(), x_PEx(), x_CNV_binary(), x_mehylation(), x_crispr()

		combo = combine_combi(estimate_data="XMID", treatment="combination")

		availability.loc["Pan"] = [CellLine(sanger2model(m)).get_pan_type() for m in sidms]
		availability.loc["Tissue"] = [CellLine(sanger2model(m)).get_tissue_type() for m in sidms]
		availability.loc["Mutation", list(mut.model_id.unique())] = 1
		availability.loc["GEx", list(exp.columns)] = 1
		availability.loc["PEx", list(pxp.columns)] = 1
		availability.loc["CNV", list(cnv.columns)] = 1
		availability.loc["Methylation", list(met.columns)] = 1
		availability.loc["CRISPR", list(cris.columns)] = 1
		for col in availability.columns:
			availability.loc["Drug Combinations", col] = len(combo.loc[combo.groupby(["SIDM"]).groups[col]].DrugComb.unique())
		availability.to_csv(output_path + "data/omics/omics_availability.csv", index=True)
	else:
		availability = pandas.read_csv(output_path + "data/omics/omics_availability.csv", index_col=0)
	return availability



# ---------------------------------------------------------------------------#
#                                 LR Feature                                 #
# ---------------------------------------------------------------------------#

def test_Z(values, pvalue, selected_side):
	x = numpy.array(values)
	x = x[~numpy.isnan(x)]
	if len(x) != 0:
		min_outlier = max(x)
		m, sd = scipy.stats.norm.fit(x)
		for v in x:
			z_score = (v - m) / sd
			p_value = scipy.stats.norm.sf(abs(z_score)) * 2
			if p_value < pvalue:
				if v < min_outlier:
					if selected_side == "right":
						if v > m:
							min_outlier = v
					else:
						min_outlier = v
		return min_outlier
	else:
		return None


def check_mutations(model_id, feature_element, level, mutation_df):
	"""
	Filling the feature matrix
	:param model_id: Sanger cell model id
	:param feature_element: The selected feature element
	:param level: Level of feature selection (genes / mutations / 3d_mutations)
	:param mutation_df: Data Frame of mutations for the selected models and genes (get_mutations())
	:return:
	"""

	if level in ["genes_cancer", "genes_driver_mut"]:
		element_column = "gene_symbol"
	elif level == "mutations_driver":
		element_column = "Variant"

	df = mutation_df[mutation_df.model_id == model_id]
	if feature_element in df[element_column].unique(): return 1
	else: return 0


def x_omics(omics_type, level):
	"""
	Get all omics data without filtering # CL
	:param omics_type: The type of omics
	:param level: feature level
	:return:
	"""

	level_text = "_".join(level.split(" "))

	if "%s_%s_feature_matrix.csv" % (omics_type, level_text) not in os.listdir(output_path + "biomarker/features/%s/" % omics_type):
		if omics_type == "mutation":

			# Get whole mutation data
			mutations = x_mutation()

			if level in ["genes_cancer"]:
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				mutations = mutations[mutations.gene_symbol.isin(genes)]

			elif level in ["genes_driver_mut", "mutations_driver"]:
				mutations = mutations[mutations.cancer_driver]

			if level in ["genes_cancer", "genes_driver_mut"]:
				feature_element_matrix = pandas.DataFrame(columns=mutations["gene_symbol"].unique(),
														  index=list(mutations.model_id.unique()))

			elif level == "mutations_driver":
				mutations["Variant"] = mutations.apply(lambda x: x.gene_symbol + " " + x.protein_mutation, axis=1)
				feature_element_matrix = pandas.DataFrame(columns=mutations["Variant"].unique(),
														  index=list(mutations.model_id.unique()))

			for ind, row in feature_element_matrix.iterrows():
				for e in feature_element_matrix.columns:
					feature_element_matrix.loc[ind, e] = check_mutations(ind, e, level, mutations)

		elif omics_type == "transcription":

			# Get whole expression data
			count_transcription = x_GEx_count()

			if level == "genes_cancer":
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				g = list(set(count_transcription.index).intersection(set(genes)))
				count_transcription = count_transcription.loc[g]

			feature_element_matrix = count_transcription.T

		elif omics_type == "proteomics":

			# Get whole proteomics data
			proteomics = x_PEx()

			if level in ["genes_cancer"]:
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				g = list(set(proteomics.index).intersection(set(genes)))
				proteomics = proteomics.loc[g]

			feature_element_matrix = proteomics.T

		elif omics_type == "CNV":

			# Get whole CNV data
			cnv = x_CNV_binary()

			if level == "genes_cancer":
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				g = list(set(cnv.index).intersection(set(genes)))
				cnv = cnv.loc[g]

			cnv = cnv.T
			cnv = cnv.replace(0.5, 0)
			feature_element_matrix = cnv.replace(-0.5, 0)

		elif omics_type == "hypermethylation":

			# Get whole methylation data
			methylation = x_mehylation()

			if level in ["genes_cancer"]:
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				g = list(set(methylation.index).intersection(set(genes)))
				methylation = methylation.loc[g]

			feature_element_matrix = methylation.T

		elif omics_type == "msi":

			# Get whole MSI data
			msi = x_msi()
			feature_element_matrix = msi.copy()

		elif omics_type == "gain":

			# Get whole gain data
			gain = x_gain()

			if level in ["genes_cancer"]:
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				g = list(set(gain.index).intersection(set(genes)))
				gain = gain.loc[g]

			feature_element_matrix = gain.T

		elif omics_type == "loss":

			# Get whole loss data
			loss = x_gain()

			if level in ["genes_cancer"]:
				# Cancer Genes
				cancer_genes = get_cancer_genes()
				drug_targets = get_drug_targets()
				genes = list(set(cancer_genes).union(set(drug_targets)))
				g = list(set(loss.index).intersection(set(genes)))
				loss = loss.loc[g]

			feature_element_matrix = loss.T

		elif omics_type == "clinical_subtypes":

			# Get whole subgroup data
			sub = x_subgroups()
			sidms = get_sidm_tissue(tissue_type="Breast")
			sub = pandas.DataFrame(sub.loc[sidms])

			# Subgroups are the levels
			feature_element_matrix = sub[[level]]

		feature_element_matrix.to_csv(output_path + "biomarker/features/%s/%s_%s_feature_matrix.csv"
									  % (omics_type, omics_type, level_text), index=True)

	else:
		feature_element_matrix = pandas.read_csv(output_path + "biomarker/features/%s/%s_%s_feature_matrix.csv"
												 % (omics_type, omics_type, level_text), index_col=0)
	return feature_element_matrix


def get_mutations(level, tissue, selection_criteria, min_cl_criteria):
	"""
	Retrieving cell line specific mutation data
	:param level: Level of feature selection (genes / mutations)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 5
	:param min_cl_criteria: The minimum number of cell lines having the feature value (default = 20)
	:return: Data Frame of mutations for the selected models and genes
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if "mutation_%s%s_%s_feature_element_matrix.csv" % (level, t_title, str(selection_criteria)) not in os.listdir(output_path + "biomarker/features/mutation/"):
		# Get whole mutation data
		mutations = x_mutation()

		# Cancer Models
		models = get_sidm_tissue(tissue_type=tissue)
		mutations = mutations[mutations.model_id.isin(models)]

		if level in ["genes_cancer"]:
			# Cancer Genes
			cancer_genes = get_cancer_genes()
			drug_targets = get_drug_targets()
			genes = list(set(cancer_genes).union(set(drug_targets)))
			mutations = mutations[mutations.gene_symbol.isin(genes)]

		elif level in ["genes_driver_mut", "mutations_driver"]:
			mutations = mutations[mutations.cancer_driver]

		if level in ["genes_cancer", "genes_driver_mut"]:
			feature_element_matrix = pandas.DataFrame(columns=mutations["gene_symbol"].unique(),
													  index=models)

		elif level == "mutations_driver":
			mutations["Variant"] = mutations.apply(lambda x: x.gene_symbol + " " + x.protein_mutation, axis=1)
			feature_element_matrix = pandas.DataFrame(columns=mutations["Variant"].unique(),
													  index=models)

		print("Creating Feature matrix")
		i, t = 0, len(feature_element_matrix.index)
		for ind, row in feature_element_matrix.iterrows():
			for e in feature_element_matrix.columns:
				feature_element_matrix.loc[ind, e] = check_mutations(ind, e, level, mutations)
			i += 1
			if i % 10 == 0:
				print(i * 100.0 / t)

		feature_element_matrix = feature_element_matrix[[
			col for col in feature_element_matrix if len(feature_element_matrix[col].unique()) > 1 and
													 feature_element_matrix[col].value_counts()[1] >= int(
				selection_criteria)]]
		feature_element_matrix = feature_element_matrix[[
			col for col in feature_element_matrix if len(feature_element_matrix[col].unique()) > 1 and
													 feature_element_matrix[col].value_counts()[0] >= int(
				selection_criteria)]]
		feature_element_matrix2 = feature_element_matrix.reset_index()
		feature_element_matrix2 = feature_element_matrix2.drop_duplicates()
		feature_element_matrix = feature_element_matrix2.set_index("index")
		feature_element_matrix = feature_element_matrix.rename_axis(None, axis=0)

		satisfactory_genes = list()
		for col in feature_element_matrix:
			if len(feature_element_matrix[~pandas.isna(feature_element_matrix[col])].index) >= min_cl_criteria:
				satisfactory_genes.append(col)

		feature_element_matrix = feature_element_matrix[satisfactory_genes]

		feature_element_matrix.to_csv(output_path + "biomarker/features/mutation/mutation_%s%s_%s_feature_element_matrix.csv"
									  % (level, t_title, str(selection_criteria)), index=True)
	else:
		feature_element_matrix = pandas.read_csv(
			output_path + "biomarker/features/mutation/mutation_%s%s_%s_feature_element_matrix.csv"
			% (level, t_title, str(selection_criteria)), index_col=0)

	return feature_element_matrix


def get_transcriptomics(level, tissue, plotting, selection_criteria, min_cl_criteria):
	"""
	Retrieving cell line specific transcription data
	:param level: Level of feature selection (genes_cancer / all)
	:param tissue: Which tissue the data will be separated into
	:param plotting: Distribution and bo boxplot T/F
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:return:
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	title_addition = "quantile_normalised"
	transcription_feature = None
	if "transcription_%s%s_%s_%s_feature_element_matrix.csv" % (level, t_title, title_addition, selection_criteria) not in os.listdir(
			output_path + "biomarker/features/transcriptomics/"):

		count_transcription = x_GEx_count()

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)

		filt_transcription = count_transcription[[model for model in all_models if model in count_transcription.columns]]

		title_addition = "quantile_normalised"
		ytitle_box = "Quantile Normalised Counts"
		id_vars_index = "index"

		# Cancer Genes
		cancer_genes = get_cancer_genes()
		drug_targets = get_drug_targets()
		genes = list(set(cancer_genes).union(set(drug_targets)))

		if level == "genes_cancer":
			g = list(set(filt_transcription.index).intersection(set(genes)))
			filt_transcription = filt_transcription.loc[g]

		if selection_criteria == "variance":

			if "transcription_variance%s_%s.csv" % (t_title, title_addition) not in os.listdir(output_path + "biomarker/features/transcriptomics/gene_selection/"):
				# Filtered genes according to the variance across cell lines
				t = filt_transcription.T
				transcription_variance = pandas.DataFrame(t.std())
				transcription_variance.columns = ["sd"]
				transcription_variance["cv"] = transcription_variance.apply(
					lambda x: 0 if x.sd == 0 or numpy.nanmean(list(t[x.name])) == 0 else x.sd / numpy.nanmean(list(t[x.name])),axis=1)

				cv_outlier_thr = test_Z(values= list(transcription_variance["cv"]), pvalue=0.05, selected_side="right")
				sd_outlier_thr = test_Z(values=list(transcription_variance["sd"]), pvalue=0.05, selected_side="right")

				if sd_outlier_thr is not None:
					transcription_variance["cv_selection"] = transcription_variance.apply(
						lambda x: True if x.cv >= cv_outlier_thr else False, axis=1)

					transcription_variance["sd_selection"] = transcription_variance.apply(
						lambda x: True if x.sd >= sd_outlier_thr else False, axis=1)
				else:
					transcription_variance["cv_selection"] = False
					transcription_variance["sd_selection"] = False

				transcription_variance.to_csv(output_path + "biomarker/features/transcriptomics/gene_selection/transcription_variance%s_%s.csv"
											  % (t_title, title_addition), index=True)

			else:
				transcription_variance = pandas.read_csv(
					output_path + "biomarker/features/transcriptomics/gene_selection/transcription_variance%s_%s.csv"
					% (t_title, title_addition), index_col=0)

			if plotting:
				gene_order = transcription_variance.sort_values(by=["sd"]).index

				melt_transcription = transcription_variance.copy()
				melt_transcription = melt_transcription.reset_index()
				cv_melt_transcription = melt_transcription.melt(id_vars = id_vars_index, value_vars = "cv")
				sd_melt_transcription = melt_transcription.melt(id_vars=id_vars_index, value_vars="sd")
				cv_outlier_thr = test_Z(values= list(transcription_variance["cv"]), pvalue=0.05, selected_side="right")
				sd_outlier_thr = test_Z(values=list(transcription_variance["sd"]), pvalue=0.05, selected_side="right")

				sns.distplot(cv_melt_transcription["value"], hist=True, color="red", kde=False)

				if tissue is None:
					plt.title("Coefficient of variance of cancer genes\nAcross all cell lines")
				else:
					plt.title("Coefficient of variance of cancer genes\nAcross %s cell lines" % " ".join(t_title.split("_")))
				plt.xlabel("Coefficient of variance")
				plt.ylabel("Frequency")
				if cv_outlier_thr is not None:
					plt.axvline(x=cv_outlier_thr, label="Upper limit for p-value 0.05", color="gray", ls="--")
				plt.tight_layout()
				plt.savefig(output_path + "biomarker/figures/gene_selection/transcription_cv_frequency%s_%s_%s.pdf"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/transcription_cv_frequency%s_%s_%s.jpg"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.close()

				sns.distplot(sd_melt_transcription["value"], hist=True, color="red", kde=False)
				if tissue is None:
					plt.title(
						"Standard deviation of cancer genes\nAcross all cell lines")
				else:
					plt.title("Standard deviation of cancer genes\nAcross %s cell lines" % " ".join(t_title.split("_")))
				plt.xlabel("Standard Deviation")
				plt.ylabel("Frequency")
				if sd_outlier_thr is not None:
					plt.axvline(x=sd_outlier_thr, label="Upper limit for p-value 0.05", color="gray", ls="--")
				plt.tight_layout()
				plt.savefig(output_path + "biomarker/figures/gene_selection/transcription_sd_frequency%s_%s_%s.pdf"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/transcription_sd_frequency%s_%s_%s.jpg"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.close()

				t = filt_transcription.T
				transcription2 = t[gene_order]

				flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
								  markeredgecolor="none")
				medianprops = dict(linestyle="-", linewidth=1.5, color="red")
				boxprops = dict(color="darkgrey")
				whiskerprops = dict(color="darkgrey")

				plt.figure(facecolor="white", figsize=(50, 15))
				if tissue is None:
					plt.title("Coefficient of variance of the genes across all cell lines", fontsize=18)
				else:

					plt.title("Coefficient of variance of the genes across %s cell lines"
							  % " ".join(t_title.split("_")),fontsize=18)

				transcription2.boxplot(medianprops=medianprops, flierprops=flierprops,
									   labels=list(t.columns), grid=False, boxprops=boxprops,
									   whiskerprops=whiskerprops)

				plt.xticks(rotation=90, fontsize=5)
				plt.xlabel("Cancer Genes", fontsize=16)
				plt.ylabel("%s" % ytitle_box , fontsize=16)
				plt.tight_layout()
				plt.savefig(output_path + "biomarker/figures/gene_selection/transcription_boxplots%s_%s_%s.pdf"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/transcription_boxplots%s_%s_%s.jpg"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.close()

			transcription_variance= transcription_variance.sort_values(by=["sd"])
			selected_genes = transcription_variance[transcription_variance.sd_selection].index

			transcription2 = filt_transcription.loc[selected_genes]
			transcription_feature = transcription2.T

			satisfactory_genes = list()
			for col in transcription_feature:
				if len(transcription_feature[~pandas.isna(transcription_feature[col])].index) >= min_cl_criteria:
					satisfactory_genes.append(col)

			transcription_feature = transcription_feature[satisfactory_genes]

			transcription_feature.to_csv(output_path + "biomarker/features/transcriptomics/transcription_%s%s_%s_%s_feature_element_matrix.csv"
										 % (level, t_title, title_addition, selection_criteria),
										 index=True)
	else:
		transcription_feature = pandas.read_csv(output_path + "biomarker/features/transcriptomics/transcription_%s%s_%s_%s_feature_element_matrix.csv"
												% (level, t_title, title_addition, selection_criteria),
												index_col=0)
	return transcription_feature


def get_cnv(feature, level, tissue, selection_criteria, min_cl_criteria):
	"""

	:param feature: The feature that will be test for association with drug data (amplification / deletion)
	:param level: Level of feature selection (genes_cancer)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:return:
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if "%s_%s%s_%s_feature_element_matrix.csv" % (feature, level, t_title, str(selection_criteria)) not in os.listdir(
			output_path + "biomarker/features/%s/" % feature):

		cnv_category = x_CNV_binary().T

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)
		cnv_category = cnv_category[cnv_category.index.isin(all_models)]

		# Cancer Genes
		cancer_genes = get_cancer_genes()
		drug_targets = get_drug_targets()
		genes = list(set(cancer_genes).union(set(drug_targets)))

		if feature == "amplification":
			cnv_category = cnv_category.replace(0.5, 0)
			cnv_category = cnv_category.replace(-0.5, 0)
			cnv_category = cnv_category.replace(-1, 0)
			if level == "genes_cancer":
				g = list(set(cnv_category.columns).intersection(set(genes)))
				cnv_category = cnv_category[g]
			cnv_category = cnv_category[[col for col in cnv_category if cnv_category[col].sum(axis=0) > 0]]
			cnv_feature = cnv_category[[col for col in cnv_category if cnv_category[col].value_counts(dropna=True)[1] >= int(selection_criteria)]]
			cnv_feature = cnv_feature[[col for col in cnv_feature if cnv_feature[col].value_counts(dropna=True)[0] >= int(selection_criteria)]]

		elif feature == "deletion":
			cnv_category = cnv_category.replace(0.5, 0)
			cnv_category = cnv_category.replace(-0.5, 0)
			cnv_category = cnv_category.replace(1, 0)
			cnv_category = cnv_category.replace(-1, 1)
			if level == "genes_cancer":
				g = list(set(cnv_category.columns).intersection(set(genes)))
				cnv_category = cnv_category[g]
			cnv_category = cnv_category[[col for col in cnv_category if cnv_category[col].sum(axis=0) > 0]]
			cnv_feature = cnv_category[[col for col in cnv_category if cnv_category[col].value_counts(dropna=True)[1] >= int(selection_criteria)]]
			cnv_feature = cnv_feature[[col for col in cnv_feature if cnv_feature[col].value_counts(dropna=True)[0] >= int(selection_criteria)]]

		cnv_feature2 = cnv_feature.reset_index()
		cnv_feature2 = cnv_feature2.drop_duplicates()
		cnv_feature = cnv_feature2.set_index("index")

		satisfactory_genes = list()
		for col in cnv_feature:
			if len(cnv_feature[~pandas.isna(cnv_feature[col])].index) >= int(min_cl_criteria):
				satisfactory_genes.append(col)

		cnv_feature = cnv_feature[satisfactory_genes]

		cnv_feature.to_csv(output_path + "biomarker/features/%s/%s_%s%s_%s_feature_element_matrix.csv"
						   % (feature, feature, level, t_title, str(selection_criteria)), index=True)

	else:
		cnv_feature = pandas.read_csv(output_path + "biomarker/features/%s/%s_%s%s_%s_feature_element_matrix.csv"
									  % (feature, feature, level, t_title, str(selection_criteria)), index_col=0)

	return cnv_feature


def get_msi(tissue, selection_criteria, min_cl_criteria):
	"""
	Retrieving cell line specific msi data
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 3
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:return: Data Frame of mutations for the selected models and genes
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if "msi_level%s_positive_%s_feature_element_matrix.csv" % (t_title, str(selection_criteria)) not in os.listdir(output_path + "biomarker/features/msi/"):

		msi_df = x_msi()

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)
		msi_df = msi_df[msi_df.index.isin(all_models)]

		if len(msi_df.index) >= min_cl_criteria:
			msi_df.to_csv(output_path + "biomarker/features/msi/msi_level%s_positive_%s_feature_element_matrix.csv"
						  % (t_title, str(selection_criteria)), index=True)
			return msi_df
		else:
			return None

	else:
		msi_df = pandas.read_csv(output_path + "biomarker/features/msi/msi_level%s_positive_%s_feature_element_matrix.csv"
								 % (t_title, str(selection_criteria)), index_col=0)
	return msi_df


def get_proteomics(level, tissue, selection_criteria, plotting, min_cl_criteria):
	"""
	Retrieving cell line specific proteomics data
	:param level: Level of feature selection (genes_cancer)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param plotting: Distribution and bo boxplot T/F
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:return: Data Frame of mutations for the selected models and genes
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	title_addition ="averaged"

	if "proteomics_%s%s_%s_%s_feature_element_matrix.csv" % (level, t_title, title_addition, selection_criteria) \
			not in os.listdir(output_path + "biomarker/features/proteomics/"):

		proteomics = x_PEx()

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)

		proteomics = proteomics[[cl for cl in all_models if cl in proteomics.columns]]
		filtering_matrix = pandas.DataFrame(proteomics.count(axis=1))
		filtering_matrix.columns = ["count"]
		proteins_to_keep = list(filtering_matrix[filtering_matrix["count"] >= min_cl_criteria].index)
		#proteins_to_keep = feature_matrix.columns[feature_matrix.isnull().sum() >= min_cl_criteria]
		filt_proteomics = proteomics.T[proteins_to_keep]

		ytitle_box = "averaged"
		id_vars_index = "index"

		if level in ["genes_cancer"]:
			# Cancer Genes
			cancer_genes = get_cancer_genes()
			drug_targets = get_drug_targets()
			genes = list(set(cancer_genes).union(set(drug_targets)))
			filt_proteomics = filt_proteomics[[gene for gene in genes if gene in filt_proteomics.columns]]

		if selection_criteria == "variance":
			if "proteome_variance%s_%s.csv" % (t_title, title_addition) not in os.listdir(output_path + "biomarker/features/proteomics/gene_selection/"):
				# Filtered genes according to the variance across cell lines
				proteome_variance = pandas.DataFrame(filt_proteomics.std())
				proteome_variance.columns = ["sd"]
				proteome_variance["cv"] = proteome_variance.apply(
					lambda x: 0 if x.sd == 0 or list(filt_proteomics[x.name]) == [] or numpy.nanmean(list(filt_proteomics[x.name])) == 0
					else x.sd / numpy.nanmean(list(filt_proteomics[x.name])),axis=1)

				cv_outlier_thr = test_Z(values= list(proteome_variance["cv"]), pvalue=0.05, selected_side="right")
				sd_outlier_thr = test_Z(values=list(proteome_variance["sd"]), pvalue=0.05, selected_side="right")

				if sd_outlier_thr is not None:
					proteome_variance["cv_selection"] = proteome_variance.apply(
						lambda x: True if x.cv >= cv_outlier_thr else False, axis=1)

					proteome_variance["sd_selection"] = proteome_variance.apply(
						lambda x: True if x.sd >= sd_outlier_thr else False, axis=1)
				else:
					proteome_variance["cv_selection"] = False
					proteome_variance["sd_selection"] = False

				proteome_variance.to_csv(output_path + "biomarker/features/proteomics/gene_selection/proteome_variance%s_%s.csv"
										 % (t_title, title_addition))

			else:
				proteome_variance = pandas.read_csv(
					output_path + "biomarker/features/proteomics/gene_selection/proteome_variance%s_%s.csv"
					% (t_title, title_addition), index_col=0)
				id_vars_index = "index"

			gene_order = proteome_variance.sort_values(by=["sd"]).index
			melt_proteome = proteome_variance.copy()
			melt_proteome = melt_proteome.reset_index()
			cv_melt_proteome = melt_proteome.melt(id_vars = id_vars_index, value_vars = "cv")
			sd_melt_proteome = melt_proteome.melt(id_vars=id_vars_index, value_vars="sd")
			cv_outlier_thr = test_Z(values= list(proteome_variance["cv"]), pvalue=0.05, selected_side="right")
			sd_outlier_thr = test_Z(values=list(proteome_variance["sd"]), pvalue=0.05, selected_side="right")

			if plotting:

				sns.distplot(cv_melt_proteome["value"], hist=True, color="red", kde=False)
				if tissue is None:
					plt.title("Coefficient of variance of cancer genes\nAcross all cell lines")
				else:
					plt.title("Coefficient of variance of cancer genes\nAcross %s cell lines" % " ".join(t_title.split("_")))
				plt.xlabel("Coefficient of variance")
				plt.ylabel("Frequency")
				if cv_outlier_thr is not None:
					plt.axvline(x=cv_outlier_thr, label="Upper limit for p-value 0.05", color="gray", ls="--")
				plt.tight_layout()
				plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_cv_frequency%s_%s_%s.pdf"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_cv_frequency%s_%s_%s.jpg"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_cv_frequency%s_%s_%s.png"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.close()

				sns.distplot(sd_melt_proteome["value"], hist=True, color="red", kde=False)
				if tissue is None:
					plt.title("Standard deviation of cancer genes\nAcross all cell lines")
				else:
					plt.title(
						"Standard deviation of cancer genes\nAcross %s cell lines" % " ".join(t_title.split("_")))
				plt.xlabel("Standard Deviation")
				plt.ylabel("Frequency")
				if sd_outlier_thr is not None:
					plt.axvline(x=sd_outlier_thr, label="Upper limit for p-value 0.05", color="gray", ls="--")
				plt.tight_layout()
				plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_sd_frequency%s_%s_%s.pdf"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_sd_frequency%s_%s_%s.jpg"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_sd_frequency%s_%s_%s.png"
							% (t_title, title_addition, selection_criteria), dpi=300)
				plt.close()

				gene_order2 = [g for g in gene_order if g in filt_proteomics.columns]
				filt_proteomics2 = filt_proteomics[gene_order2]

				if len(filt_proteomics2.columns) > 0:
					flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
									  markeredgecolor="none")
					medianprops = dict(linestyle="-", linewidth=1.5, color="red")
					boxprops = dict(color="darkgrey")
					whiskerprops = dict(color="darkgrey")

					plt.figure(facecolor="white", figsize=(50, 15))
					if tissue is None:
						plt.title("Coefficient of variance of the genes across all cell lines", fontsize=18)
					else:
						plt.title(
							"Coefficient of variance of the genes across %s cell lines" % " ".join(t_title.split("_")),
							fontsize=18)

					filt_proteomics2.boxplot(medianprops=medianprops, flierprops=flierprops,
											 labels=list(filt_proteomics2.columns),
											 grid=False, boxprops=boxprops, whiskerprops=whiskerprops)

					plt.xticks(rotation=90, fontsize=5)
					plt.xlabel("Cancer Genes", fontsize=16)
					plt.ylabel("%s" % ytitle_box , fontsize=16)
					plt.tight_layout()
					plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_boxplots%s_%s_%s.pdf"
								% (t_title, title_addition, selection_criteria), dpi=300)
					plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_boxplots%s_%s_%s.jpg"
								% (t_title, title_addition, selection_criteria), dpi=300)
					plt.savefig(output_path + "biomarker/figures/gene_selection/proteome_boxplots%s_%s_%s.png"
								% (t_title, title_addition, selection_criteria), dpi=300)
					plt.close()

			proteome_variance= proteome_variance.sort_values(by=["sd"])
			selected_genes = proteome_variance[proteome_variance.sd_selection].index
			selected_genes2 = [g for g in selected_genes if g in filt_proteomics.columns]

			filt_proteomics2 = filt_proteomics.T
			filt_proteomics3 = filt_proteomics2.loc[selected_genes2]
			proteomics_feature = filt_proteomics3.T

			satisfactory_genes = list()
			for col in proteomics_feature:
				if len(proteomics_feature[~pandas.isna(proteomics_feature[col])].index) >= min_cl_criteria:
					satisfactory_genes.append(col)

			proteomics_feature = proteomics_feature[satisfactory_genes]

			proteomics_feature.to_csv(output_path + "biomarker/features/proteomics/proteomics_%s%s_%s_%s_feature_element_matrix.csv"
									 % (level, t_title, title_addition, selection_criteria), index=True)
	else:
		proteomics_feature = pandas.read_csv(output_path + "biomarker/features/proteomics/proteomics_%s%s_%s_%s_feature_element_matrix.csv"
											 % (level, t_title, title_addition, selection_criteria), index_col=0)

	return proteomics_feature


def get_methylation(level, tissue, selection_criteria, min_cl_criteria):
	"""
	Retrieving cell line specific methylation data
	:param level: Level of feature selection (genes/ all)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:return: Data Frame of methylation for the selected models and genes
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if "hypermethylation_%s%s_%s_feature_element_matrix.csv" % (level, t_title, str(selection_criteria)) not in os.listdir(output_path + "biomarker/features/methylation/"):

		methylation = x_mehylation()

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)
		filt_methylation = methylation[[cl for cl in all_models if cl in methylation.columns]]

		if level in ["genes_cancer"]:
			# Cancer Genes
			cancer_genes = get_cancer_genes()
			drug_targets = get_drug_targets()
			genes = list(set(cancer_genes).union(set(drug_targets)))
			filt_methylation = filt_methylation[filt_methylation.index.isin(genes)]

		#feature_matrix = feature_matrix[feature_matrix.columns[feature_matrix.sum() >= selection_criteria]]
		feature_matrix = filt_methylation.T

		feature_matrix = feature_matrix[[
			col for col in feature_matrix if len(feature_matrix[col].unique()) > 1 and
											 feature_matrix[col].value_counts()[1] >= int(selection_criteria)]]
		feature_matrix = feature_matrix[[
			col for col in feature_matrix if len(feature_matrix[col].unique()) > 1 and
											 feature_matrix[col].value_counts()[0] >= int(selection_criteria)]]

		satisfactory_genes = list()
		for col in feature_matrix:
			if len(feature_matrix[~pandas.isna(feature_matrix[col])].index) >= min_cl_criteria:
				satisfactory_genes.append(col)

		feature_matrix = feature_matrix[satisfactory_genes]
		feature_matrix.to_csv(output_path + "biomarker/features/methylation/hypermethylation_%s%s_%s_feature_element_matrix.csv"
							  % (level, t_title, str(selection_criteria)), index=True)
	else:
		feature_matrix = pandas.read_csv(output_path + "biomarker/features/methylation/hypermethylation_%s%s_%s_feature_element_matrix.csv"
										 % (level, t_title, str(selection_criteria)), index_col=0)

	return feature_matrix


def get_gain(level, tissue, selection_criteria, min_cl_criteria):
	"""
	:param level: Level of feature selection (genes_cancer)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 5
	:param min_cl_criteria: The minimum number of cell lines having the feature value (default=15)
	:return:
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if "gain_%s%s_%s_feature_element_matrix.csv" % (level, t_title, str(selection_criteria)) not in os.listdir(
			output_path + "biomarker/features/gain/"):

		gain_category = x_gain().T

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)
		gain_category = gain_category[gain_category.index.isin(all_models)]

		# Cancer Genes
		cancer_genes = get_cancer_genes()
		drug_targets = get_drug_targets()
		genes = list(set(cancer_genes).union(set(drug_targets)))

		if level == "genes_cancer":
			g = list(set(gain_category.columns).intersection(set(genes)))
			gain_category = gain_category[g]

		gain_category = gain_category[[col for col in gain_category if gain_category[col].sum(axis=0) > 0]]
		gain_feature = gain_category[[col for col in gain_category if gain_category[col].value_counts(dropna=True)[1] >= int(selection_criteria)]]
		gain_feature = gain_feature[[col for col in gain_feature if 0 in gain_feature[col].value_counts(dropna=True).keys() and gain_feature[col].value_counts(dropna=True)[0] >= int(selection_criteria)]]

		gain_feature2 = gain_feature.reset_index()
		gain_feature2 = gain_feature2.drop_duplicates()
		gain_feature = gain_feature2.set_index("index")

		satisfactory_genes = list()
		for col in gain_feature:
			if len(gain_feature[~pandas.isna(gain_feature[col])].index) >= int(min_cl_criteria):
				satisfactory_genes.append(col)

		gain_feature = gain_feature[satisfactory_genes]

		gain_feature.to_csv(output_path + "biomarker/features/gain/gain_%s%s_%s_feature_element_matrix.csv"
							% (level, t_title, str(selection_criteria)), index=True)

	else:
		gain_feature = pandas.read_csv(output_path + "biomarker/features/gain/gain_%s%s_%s_feature_element_matrix.csv"
									   % (level, t_title, str(selection_criteria)), index_col=0)

	return gain_feature


def get_loss(level, tissue, selection_criteria, min_cl_criteria):
	"""
	:param level: Level of feature selection (genes_cancer)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 5
	:param min_cl_criteria: The minimum number of cell lines having the feature value (default=15)
	:return:
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if "loss_%s%s_%s_feature_element_matrix.csv" % (level, t_title, str(selection_criteria)) not in os.listdir(
			output_path + "biomarker/features/loss/"):

		loss_category = x_loss().T

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)
		loss_category = loss_category[loss_category.index.isin(all_models)]

		# Cancer Genes
		cancer_genes = get_cancer_genes()
		drug_targets = get_drug_targets()
		genes = list(set(cancer_genes).union(set(drug_targets)))

		if level == "genes_cancer":
			g = list(set(loss_category.columns).intersection(set(genes)))
			loss_category = loss_category[g]

		loss_category = loss_category[[col for col in loss_category if loss_category[col].sum(axis=0) > 0]]
		loss_feature = loss_category[[col for col in loss_category if loss_category[col].value_counts(dropna=True)[1] >= int(selection_criteria)]]
		loss_feature = loss_feature[[col for col in loss_feature if 0 in loss_feature[col].value_counts(dropna=True).keys() and loss_feature[col].value_counts(dropna=True)[0] >= int(selection_criteria)]]

		loss_feature2 = loss_feature.reset_index()
		loss_feature2 = loss_feature2.drop_duplicates()
		loss_feature = loss_feature2.set_index("index")

		satisfactory_genes = list()
		for col in loss_feature:
			if len(loss_feature[~pandas.isna(loss_feature[col])].index) >= int(min_cl_criteria):
				satisfactory_genes.append(col)

		loss_feature = loss_feature[satisfactory_genes]

		loss_feature.to_csv(output_path + "biomarker/features/loss/loss_%s%s_%s_feature_element_matrix.csv"
							% (level, t_title, str(selection_criteria)), index=True)

	else:
		loss_feature = pandas.read_csv(output_path + "biomarker/features/loss/loss_%s%s_%s_feature_element_matrix.csv"
									   % (level, t_title, str(selection_criteria)), index_col=0)

	return loss_feature


def get_clinical_subtype(level, tissue, selection_criteria, min_cl_criteria):
	"""
	:param level: Level of feature selection (genes_cancer)
	:param tissue: Which tissue the data will be separated into
	:param selection_criteria: Binary feature = 5
	:param min_cl_criteria: The minimum number of cell lines having the feature value (default=15)
	:return:
	"""
	level_text = "_".join(level.split(" "))
	
	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if "clinicalsubtype_%s%s_%s_feature_element_matrix.csv" % (level_text, t_title, str(selection_criteria)) not in os.listdir(
			output_path + "biomarker/features/clinical_subtypes/"):

		subgroup_category = x_subgroups()

		# Cancer Models
		all_models = get_sidm_tissue(tissue_type=tissue)
		subgroup_category = subgroup_category[subgroup_category.index.isin(all_models)]

		subgroup_category = pandas.DataFrame(subgroup_category[level])

		subgroup_feature = subgroup_category[[col for col in subgroup_category if subgroup_category[col].value_counts(dropna=True)[1] >= int(selection_criteria)]]
		if len(subgroup_feature.columns)> 0:
			subgroup_feature = subgroup_feature[[col for col in subgroup_feature if subgroup_feature[col].value_counts(dropna=True)[0] >= int(selection_criteria)]]

		subgroup_feature2 = subgroup_feature.reset_index()
		subgroup_feature2 = subgroup_feature2.drop_duplicates()
		subgroup_feature = subgroup_feature2.set_index("index")

		satisfactory_genes = list()
		for col in subgroup_feature:
			if len(subgroup_feature[~pandas.isna(subgroup_feature[col])].index) >= int(min_cl_criteria):
				satisfactory_genes.append(col)

		subgroup_feature = subgroup_feature[satisfactory_genes]

		subgroup_feature.to_csv(output_path + "biomarker/features/clinical_subtypes/"
											  "clinicalsubtype_%s%s_%s_feature_element_matrix.csv"
							% (level_text, t_title, str(selection_criteria)), index=True)

	else:
		subgroup_feature = pandas.read_csv(output_path + "biomarker/features/clinical_subtypes/"
														 "clinicalsubtype_%s%s_%s_feature_element_matrix.csv"
										   % (level_text, t_title, str(selection_criteria)), index_col=0)

	return subgroup_feature


def create_feature_matrix(feature, level, tissue, selection_criteria, min_cl_criteria=15):
	"""
	Constructing the adjacency feature matrix
	:param feature: The feature that will be test for association with drug data
			(mutation / transcription / cnv / amplification / deletion / methylation / protein expression )
	:param level: Level of feature selection
			(For all genes_cancer / for mutation + genes_driver_mut + driver_mutations)
	:param selection_criteria: Binary feature = 5 / continuous feature = variance
	:param min_cl_criteria: The minimum number of cell lines having the feature value (default =15)
	:param tissue: Which tissue the data will be separated into
	:return: Adjacency matrix of having or nor having the features for binary / values of the features for continuous
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if feature == "transcription": title = "quantile_normalised_" + str(selection_criteria)
	elif feature == "proteomics": title = "averaged_" + str(selection_criteria)
	elif feature == "msi": title, level = "positive_" + str(selection_criteria), "level"
	else: title = str(selection_criteria)

	level_text = "_".join(level.split(" "))

	if "%s_%s%s_%s_feature_element_matrix.csv" % (feature, level_text, t_title, title) not in os.listdir(output_path + "biomarker/features/"):
		print("""
		Feature matrix is preparing.""")
		if feature == "mutation":
			feature_element_matrix = get_mutations(level=level, tissue=tissue, selection_criteria=selection_criteria,
												   min_cl_criteria=min_cl_criteria)

		elif feature == "transcription":
			feature_element_matrix = get_transcriptomics(level=level, tissue=tissue,  plotting=True,
														 selection_criteria=selection_criteria, min_cl_criteria=min_cl_criteria)

		elif feature in ["amplification", "deletion"]:
			feature_element_matrix = get_cnv(feature=feature, level=level, tissue=tissue,
											 selection_criteria=selection_criteria, min_cl_criteria=min_cl_criteria)

		elif feature == "msi":
			feature_element_matrix = get_msi(tissue=tissue, min_cl_criteria=min_cl_criteria,
											 selection_criteria = selection_criteria)

		elif feature == "hypermethylation":
			feature_element_matrix = get_methylation(level=level, tissue=tissue, selection_criteria = selection_criteria,
													 min_cl_criteria=min_cl_criteria)

		elif feature == "proteomics":
			feature_element_matrix = get_proteomics(level=level, tissue=tissue, plotting=True,
													selection_criteria=selection_criteria,  min_cl_criteria=min_cl_criteria)

		elif feature == "gain":
			feature_element_matrix = get_gain(level=level, tissue=tissue, selection_criteria=selection_criteria,
											  min_cl_criteria=min_cl_criteria)

		elif feature == "loss":
			feature_element_matrix = get_loss(level=level, tissue=tissue, selection_criteria=selection_criteria,
											  min_cl_criteria=min_cl_criteria)

		elif feature == "clinicalsubtype":
			feature_element_matrix = get_clinical_subtype(level=level, tissue=tissue, selection_criteria=selection_criteria,
														  min_cl_criteria=min_cl_criteria)

		print("""
		Feature matrix is prepared.\n""")
	else:
		feature_element_matrix = pandas.read_csv(output_path + "biomarker/features/%s_%s%s_%s_feature_element_matrix.csv"
												 % (feature, level_text, t_title, title), index_col=0)
		print("""
		Feature matrix is read.\n""")
	return feature_element_matrix
