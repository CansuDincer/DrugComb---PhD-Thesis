"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 6 November 2023
Last Update : 1 April 2024
Input : Features
Output : Selected features
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas

from CombDrug.module.path import output_path
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.omics import *


# ---------------------------------------------------------------------------#
#                                  Functions                                 #
# ---------------------------------------------------------------------------#

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
			feature_element_matrix = get_transcriptomics(level=level, tissue=tissue,  plotting=False,
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


def feature_naming(feature, feature_level, element):
	"""
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:param element: The selected element for the feature (gene/ mutations etc.)
	"""
	if feature == "mutation":
		if feature_level == "genes_driver_mut":
			return "%s_M" % element
		elif feature_level == "mutations_driver":
			return "%s_dM" % element
	if feature == "transcription":
		return "%s_GEx" % element
	if feature == "proteomics":
		return "%s_PEx" % element
	if feature == "amplification":
		return "%s_AMP" % element
	if feature == "deletion":
		return "%s_DEL" % element
	if feature == "hypermethylation":
		return "%s_MET" % element
	if feature == "clinicalsubtype":
		return "%s_SubT" % element
	if feature == "gain":
		return "%s-(+)" % element
	if feature == "loss":
		return "%s-(-)" % element


def f_tissue(treatment):
	if "feature_tissue_%s.csv" % treatment not in os.listdir(
			output_path + "prediction/features/"):
		# Combi data
		if treatment == "combo":
			combi = combine_combi(estimate_data="XMID", treatment="combination")
		else:
			combi = combine_combi(estimate_data="XMID", treatment="mono")

		# Tissue for pancancer
		tissue_df = combi[["SIDM", "tissue_type"]]
		tissue_df = tissue_df.set_index(["SIDM"])
		tissue_df = pandas.get_dummies(tissue_df)
		tissue_df = tissue_df.reset_index()
		tissue_df = tissue_df.drop_duplicates()
		tissue_df = tissue_df.set_index(["SIDM"])

		tissue_df.to_csv(output_path + "prediction/features/feature_tissue_%s.csv"
						 % treatment, index=True)
	else:
		tissue_df = pandas.read_csv(output_path + "prediction/features/feature_tissue_%s.csv"
									% treatment, index_col=0)

	return tissue_df


def f_msi():
	if "feature_msi.csv" not in os.listdir(output_path + "prediction/features/"):
		# MSI Status
		# MSI for Colorectal, Stomach, Endometrium and Ovary
		model_info = cancer_models()
		msi_df = model_info[["sanger_id", "msi_status"]].drop_duplicates()
		msi_df = msi_df.rename({"sanger_id": "SIDM"}, axis=1)
		msi_df = msi_df.set_index(["SIDM"])
		msi_df = pandas.get_dummies(msi_df)
		msi_df = msi_df.reset_index()
		msi_df = msi_df.drop_duplicates()
		msi_df = msi_df.set_index(["SIDM"])

		msi_df.to_csv(output_path + "prediction/features/feature_msi.csv", index=True)
	else:
		msi_df = pandas.read_csv(output_path + "prediction/features/feature_msi.csv", index_col=0)

	return msi_df


def f_property():
	if "feature_property.csv" not in os.listdir(output_path + "prediction/features/"):
		# Growth Property
		model_info = cancer_models()
		property_df = model_info[["sanger_id", "growth_properties"]].drop_duplicates()
		property_df = property_df.rename({"sanger_id": "SIDM"}, axis=1)
		property_df = property_df.set_index(["SIDM"])
		property_df = pandas.get_dummies(property_df)
		property_df = property_df.reset_index()
		property_df = property_df.drop_duplicates()
		property_df = property_df.set_index(["SIDM"])
		property_df.to_csv(output_path + "prediction/features/feature_property.csv", index=True)
	else:
		property_df = pandas.read_csv(output_path + "prediction/features/feature_property.csv", index_col=0)

	return property_df


def f_growth():
	if "feature_growth.csv" not in os.listdir(output_path + "prediction/features/"):
		# Growth Rate
		# Growth Rate (missing values from CMP)
		growth_df = get_growth()
		growth_df.columns = ["SIDM", "growth_rate"]
		growth_df = growth_df.set_index(["SIDM"])
		growth_df.to_csv(output_path + "prediction/features/feature_growth.csv", index=True)
	else:
		growth_df = pandas.read_csv(output_path + "prediction/features/feature_growth.csv", index_col=0)

	return growth_df


def f_mut(level):
	if "feature_mutation_%s.csv" % level not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="mutation", level=level, selection_criteria=5,
										tissue=None,  min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="mutation", feature_level=level, element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_mutation_%s.csv"
					   % level, index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_mutation_%s.csv"
								  % level, index_col=0)

	return omic_df


def f_gex():
	if "feature_gex.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="transcription", level="genes_cancer", selection_criteria="variance",
										tissue=None,  min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="transcription", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_gex.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_gex.csv",
								  index_col=0)

	return omic_df


def f_pex():
	if "feature_pex.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="proteomics", level="genes_cancer", selection_criteria="variance",
										tissue=None, min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="proteomics", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_pex.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_pex.csv",
								  index_col=0)

	return omic_df


def f_amp():
	if "feature_amp.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="amplification", level="genes_cancer", selection_criteria=5,
										tissue=None, min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="amplification", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_amp.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_amp.csv",
								  index_col=0)

	return omic_df


def f_del():
	if "feature_del.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="deletion", level="genes_cancer", selection_criteria=5,
										tissue=None, min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="deletion", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_del.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_del.csv",
								  index_col=0)

	return omic_df


def f_met():
	if "feature_met.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="hypermethylation", level="genes_cancer", selection_criteria=5,
										tissue=None, min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="hypermethylation", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_met.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_met.csv",
								  index_col=0)

	return omic_df


def f_gain():
	if "feature_gain.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="gain", level="genes_cancer", selection_criteria=5,
										tissue=None, min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="gain", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_gain.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_gain.csv",
								  index_col=0)

	return omic_df


def f_loss():
	if "feature_loss.csv" not in os.listdir(output_path + "prediction/features/"):
		omic_df = create_feature_matrix(feature="loss", level="genes_cancer", selection_criteria=5,
										tissue=None, min_cl_criteria=15)
		new_col_dict = {}
		for col in omic_df.columns:
			new_col_dict[col] = feature_naming(feature="loss", feature_level="genes_cancer", element=col)
		omic_df.rename(columns=new_col_dict, inplace=True)

		omic_df.to_csv(output_path + "prediction/features/feature_loss.csv", index=True)
	else:
		omic_df = pandas.read_csv(output_path + "prediction/features/feature_loss.csv",
								  index_col=0)

	return omic_df


def f_clinical():
	if "feature_clinicalsubtype.csv" not in os.listdir(output_path + "prediction/features/"):
		omics_dfs = list()
		for level in ["CMS1", "CMS2", "CMS3", "CMS4", "CRISA", "CRISB", "CRISC",
					  "CRISD", "CRISE", "Basal", "LumA", "LumB", "Her2"]:
			if level in ["Basal", "LumA", "LumB", "Her2"]:
				omic_df = create_feature_matrix(feature="clinicalsubtype", level=level,
												selection_criteria=5, tissue="Breast",
												min_cl_criteria=15)
			else:
				omic_df = create_feature_matrix(feature="clinicalsubtype", level=level,
												selection_criteria=5, tissue="Large Intestine",
												min_cl_criteria=15)
			omics_dfs.append(omic_df)

		full_omic_df = pandas.concat(omics_dfs)
		new_col_dict = {}
		for col in full_omic_df.columns:
			new_col_dict[col] = feature_naming(feature="clinicalsubtype",
											   feature_level=None, element=col)
		full_omic_df.rename(columns=new_col_dict, inplace=True)

		full_omic_df.to_csv(output_path + "prediction/features/feature_clinicalsubtype.csv", index=True)
	else:
		full_omic_df = pandas.read_csv(output_path + "prediction/features/feature_clinicalsubtype.csv",
								  index_col=0)

	return full_omic_df


def x_omics_features(treatment, tissue, tissue_flag, msi_flag, media_flag, growth_flag, mutation_mut_flag,
					 mutation_gene_flag, gex_flag, pex_flag, met_flag, amp_flag, del_flag, gain_flag, loss_flag,
					 clinical_flag):
	"""
	:param tissue: Which tissue the data will be separated into
	:param treatment: mono/ combination
	:param tissue_flag: Boolean - If tissues will be used
	:param msi_flag: Boolean - If MSI status will be used
	:param media_flag: Boolean - If media properties will be used
	:param growth_flag: Boolean - If growth rate will be used
	:param mutation_mut_flag: Boolean - If driver mutations will be used
	:param mutation_gene_flag: Boolean - If mutations on genes will be used
	:param gex_flag: Boolean - If GEx will be used
	:param pex_flag: Boolean - If PEx will be used
	:param met_flag: Boolean - If methylation will be used
	:param amp_flag: Boolean - If amplification will be used
	:param del_flag: Boolean - If deletion will be used
	:param gain_flag: Boolean - If gain will be used
	:param loss_flag: Boolean - If loss will be used
	:param clinical_flag: Boolean - If clinical subtype will be used
	:return: Features for predictions from LR
	"""

	if tissue is not None:
		models = get_sidm_tissue(tissue_type=tissue)
	else:
		models = all_screened_SIDMs(project_list=None, integrated=True)

	num_omics, cat_omics = list(), list()
	feature_df = pandas.DataFrame(index=models)
	if msi_flag:
		df = f_msi()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if tissue_flag:
		df = f_tissue(treatment=treatment)
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if media_flag:
		df = f_property()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if growth_flag:
		df = f_growth()
		feature_df = pandas.concat([feature_df, df], axis=1)
		num_omics.extend(list(df.columns))

	if mutation_mut_flag:
		df = f_mut(level="mutations_driver")
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if mutation_gene_flag:
		df = f_mut(level="genes_driver_mut")
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if gex_flag:
		df = f_gex()
		feature_df = pandas.concat([feature_df, df], axis=1)
		num_omics.extend(list(df.columns))

	if pex_flag:
		df = f_pex()
		feature_df = pandas.concat([feature_df, df], axis=1)
		num_omics.extend(list(df.columns))

	if met_flag:
		df = f_met()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if amp_flag:
		df = f_amp()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if del_flag:
		df = f_del()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if loss_flag:
		df = f_loss()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if gain_flag:
		df = f_gain()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	if clinical_flag:
		df = f_clinical()
		feature_df = pandas.concat([feature_df, df], axis=1)
		cat_omics.extend(list(df.columns))

	feature_df = feature_df.loc[[model for model in models if model in list(feature_df.index)]]
	feature_df = feature_df.reset_index()
	feature_df = feature_df.drop_duplicates()
	#feature_df = feature_df.dropna()
	feature_df = feature_df.set_index(["index"])

	return feature_df, num_omics, cat_omics



