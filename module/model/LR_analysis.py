"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 11 April 2023
Last Update : 9 April 2024
Input : Drug Combination Analysis
Output : Drug Combination - Biomarkers
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, scipy, sys, matplotlib, networkx, re, argparse
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import r2_score
from statannot import add_stat_annotation
from datetime import datetime
from statannotations.Annotator import Annotator
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from CombDrug.module.path import output_path
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.omics import *
from CombDrug.module.data.responses import get_response_data

flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
				  markeredgecolor="none")
medianprops = dict(linestyle="-", linewidth=1, color="red")
boxprops = dict(facecolor="white", edgecolor="darkgrey")
whiskerprops = dict(color="darkgrey")


def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-run_for", dest="RUN_FOR", required=True)
	parser.add_argument("-tissue", dest="TISSUE", required=False)
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                         Combination Effect Analysis                        #
# ---------------------------------------------------------------------------#

def feature_annotation(feature, feature_level):
	"""
	Annotate biomarkers with the feature and feature levels
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:return: The biomarker annotation
	"""
	if feature == "mutation":
		if feature_level == "genes_cancer":
			return "MG"
		elif feature_level == "genes_driver_mut":
			return "dMG"
		elif feature_level == "mutations_driver":
			return "dM"
	if feature == "transcription":
		return "GEx"
	if feature == "proteomics":
		return "PEx"
	if feature == "cnv":
		return "CNV"
	if feature == "amplification":
		return "AMP"
	if feature == "deletion":
		return "DEL"
	if feature == "msi":
		return "MSI"
	if feature == "hypermethylation":
		return "hMet"
	if feature == "gain":
		return "(+))"
	if feature == "loss":
		return "(-))"
	if feature == "clinicalsubtype":
		return feature_level


def feature_annotation_title(feature, feature_level):
	"""
	Feature annotation title which will be used in the figures
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:return:
	"""
	if feature == "mutation":
		if feature_level == "genes_cancer":
			return "Gene Mutation"
		elif feature_level == "genes_driver_mut":
			return "Driver Gene Mutation"
		elif feature_level == "mutations_driver":
			return "Driver Mutation"
	if feature == "transcription":
		return "Gene Expression"
	if feature == "proteomics":
		return "Protein Expression"
	if feature == "cnv":
		return "CNV"
	if feature == "amplification":
		return "Amplification"
	if feature == "deletion":
		return "Deletion"
	if feature == "msi":
		return "MSI"
	if feature == "hypermethylation":
		return "Hypermethylation"
	if feature == "gain":
		return "GoF"
	if feature == "loss":
		return "LoF"
	if feature == "clinicalsubtype":
		return "Clinical Subtype %s" % feature_level


def feature_plot_annotation(feature, feature_level, element):
	"""
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:param element: The selected element for the feature (gene/ mutations etc.)
	"""
	if feature == "mutation":
		if feature_level == "genes_driver_mut":
			return "$%s^M$" % element, "$%s^{WT}$" % element
		elif feature_level == "mutations_driver":
			gene = element.split(" p.")[0]
			mutation = element.split(" p.")[1]
			return "$%s^{%s}$" % (gene, mutation), "$%s^{WT}$" % element
	if feature == "transcription":
		return "High $%s$ GEx" % element, "Low $%s$ GEx" % element
	if feature == "proteomics":
		return "High $%s$ PEx" % element, "Low $%s$ PEx" % element
	if feature == "amplification":
		return "$%s^{AMP}$" % element, "$%s$" % element
	if feature == "deletion":
		return "$%s^{DEL}$" % element, "$%s$" % element
	if feature == "hypermethylation":
		return "$%s^{MET}$" % element, "$%s$" % element
	if feature == "gain":
		return "$%s^{(+)}$" % element, "$%s$" % element
	if feature == "loss":
		return "$%s^{(-)}$" % element, "$%s$" % element
	if feature == "clinicalsubtype":
		return "$%s$" % element, "Other"


def feature_column_annotation(feature, feature_level, element):
	"""
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:param element: The selected element for the feature (gene/ mutations etc.)
	"""
	if feature == "mutation":
		if feature_level == "genes_driver_mut":
			return "$%s^M$" % element
		elif feature_level == "mutations_driver":
			gene = element.split(" p.")[0]
			mutation = element.split(" p.")[1]
			return "$%s^{%s}$" % (gene, mutation)
	if feature == "transcription":
		return "$%s^{GEx}$" % element
	if feature == "proteomics":
		return "$%s^{PEx}$" % element
	if feature == "amplification":
		return "$%s^{AMP}$" % element
	if feature == "deletion":
		return "$%s^{DEL}$" % element
	if feature == "CNV":
		return "$%s^{CNV}$" % element
	if feature == "hypermethylation":
		return "$%s^{MET}$" % element
	if feature == "gain":
		return "$%s^{(+)}$" % element
	if feature == "loss":
		return "$%s^{(-)}$" % element
	if feature == "clinicalsubtype":
		return "$%s$" % element


def feature_simple_annotation(feature, feature_level, element):
	"""
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:param element: The selected element for the feature (gene/ mutations etc.)
	"""
	if feature == "mutation":
		if feature_level == "genes_driver_mut":
			return "%s_M" % element
		elif feature_level == "mutations_driver":
			gene = element.split(" p.")[0]
			mutation = element.split(" p.")[1]
			return "%s_%s" % (gene, mutation)
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
	if feature == "gain":
		return "%s_(+)" % element
	if feature == "loss":
		return "%s_(-)" % element
	if feature == "clinicalsubtype":
		return "%s" % element


def get_feature_criteria(feature, feature_level):
	"""
	Feature criteria used in Linear Regression
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:return:
	"""
	if feature == "mutation":
		return "5"
	elif feature == "transcription":
		return "variance"
	elif feature == "proteomics":
		return "variance"
	elif feature == "cnv":
		if feature_level == "variance":
			return "variance"
		elif feature_level in ["amplification", "deletion"]:
			return "5"
	elif feature_level == "msi":
		return "5"
	elif feature_level == "hypermethylation":
		return "5"
	elif feature_level == "gain":
		return "5"
	elif feature_level == "loss":
		return "5"
	elif feature_level == "clinicalsubtype":
		return "5"


def get_LR_response(drug_info, response_df, drug_col):
	"""
	Create response matrix for LR models
	:param drug_info: Drug Combination or library drug
	:param response_df: Whole response data frame
	:param drug_col: Name of the drug column
	:return: response matrix
	"""
	response = response_df.copy()
	# response = response.set_index("SIDM")
	response = response[response[drug_col] == drug_info]
	response = response[["response"]]
	response = response.drop_duplicates()
	return response


def get_LR_covariate(whole_combi, tissue, stage, msi_flag):
	"""
	Create covariate matrix for LR models
	Pancancer the selected covariates are growth rate and tissue
	Tissue-based analysis the selected covariates are growth rate and growth properties
	:param whole_combi: Whole combi data frame after max delta integration
	:param tissue: Which tissue the data will be separated into
	:param stage: LIBRARY / COMBO / DELTA
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:return: Linear Regression input covariate matrix
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if stage == "mono":
		drug_col = "library_name"
	else:
		drug_col = "DrugComb"

	if "covariate_%s%s%s.csv" % (stage, t_title, msi_title) not in os.listdir(
			output_path + "biomarker/covariates/files/"):

		# Combi data
		combi = whole_combi.copy()

		# Tissue for pancancer
		if tissue in ["panliquid", "pansolid"]:
			tissue_df = combi[["SIDM", "tissue_type"]]
			tissue_df = tissue_df.set_index(["SIDM"])
			tissue_df = pandas.get_dummies(tissue_df["tissue_type"])
			tissue_cols = list(tissue_df.columns)
			tissue_df = tissue_df.reset_index()
			combi = combi[[i for i in combi.columns if i != "tissue_type"]]
			combi2 = pandas.merge(combi, tissue_df, on=["SIDM"])

		else:
			combi2 = combi.copy()

		# Growth Rate (missing values from CMP)
		growth_rate = get_growth()
		growth_rate.columns = ["SIDM", "growth_rate"]
		growth_rate = growth_rate[growth_rate.SIDM.isin(combi2.SIDM.unique())]
		combi3 = pandas.merge(combi2, growth_rate, on=["SIDM"])

		# Experiment type
		screen = combi[["SIDM", "screen_type", drug_col]]
		screen = screen.set_index(["SIDM", drug_col])
		screen = pandas.get_dummies(screen["screen_type"])
		screen_columns = list(screen.columns) + [drug_col]
		screen = screen.reset_index()
		screen = screen[screen.SIDM.isin(combi3.SIDM.unique())]
		screen = screen.drop_duplicates()
		combi4 = pandas.merge(combi3, screen, on=["SIDM", drug_col])

		if tissue is None:
			if msi_flag:
				columns = ["SIDM", "growth_rate"] + screen_columns + msi_cols + tissue_cols
			else:
				columns = ["SIDM", "growth_rate"] + screen_columns + tissue_cols
		else:
			if msi_flag:
				columns = ["SIDM", "growth_rate"] + screen_columns + msi_cols
			else:
				columns = ["SIDM", "growth_rate"] + screen_columns

		combi_last = combi4[columns]
		combi_last = combi_last.drop_duplicates()

		if tissue is not None:
			models = get_sidm_tissue(tissue_type=tissue)
		else:
			models = all_screened_SIDMs(project_list=None, integrated=True)

		covariate = combi_last[combi_last["SIDM"].isin(models)]
		covariate = covariate.set_index(["SIDM"])
		# covariate["growth_properties"] = covariate["growth_properties"].astype('category')
		covariate.to_csv(output_path + "biomarker/covariates/files/covariate_%s%s%s.csv" % (stage, t_title, msi_title))
	else:
		covariate = pandas.read_csv(
			output_path + "biomarker/covariates/files/covariate_%s%s%s.csv" % (stage, t_title, msi_title), index_col=0)

	return covariate


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
	else:
		t_title = ""

	if feature == "transcription":
		title = "log2_filtered_" + str(selection_criteria)
	elif feature == "proteomics":
		title = "averaged_" + str(selection_criteria)
	elif feature == "msi":
		title, level = "positive_" + str(selection_criteria), "level"
	else:
		title = str(selection_criteria)

	level_text = "_".join(level.split(" "))

	if "%s_%s%s_%s_feature_element_matrix.csv" % (feature, level_text, t_title, title) not in os.listdir(
			output_path + "biomarker/features/"):
		print("""
		Feature matrix is preparing.""")
		if feature == "mutation":
			feature_element_matrix = get_mutations(level=level, tissue=tissue, selection_criteria=selection_criteria,
												   min_cl_criteria=min_cl_criteria)

		elif feature == "transcription":
			feature_element_matrix = get_transcriptomics(level=level, tissue=tissue, plotting=True,
														 selection_criteria=selection_criteria,
														 min_cl_criteria=min_cl_criteria)

		elif feature in ["amplification", "deletion"]:
			feature_element_matrix = get_cnv(feature=feature, level=level, tissue=tissue,
											 selection_criteria=selection_criteria, min_cl_criteria=min_cl_criteria)

		elif feature == "msi":
			feature_element_matrix = get_msi(tissue=tissue, min_cl_criteria=min_cl_criteria,
											 selection_criteria=selection_criteria)

		elif feature == "hypermethylation":
			feature_element_matrix = get_methylation(level=level, tissue=tissue, selection_criteria=selection_criteria,
													 min_cl_criteria=min_cl_criteria)

		elif feature == "proteomics":
			feature_element_matrix = get_proteomics(level=level, tissue=tissue, plotting=True,
													selection_criteria=selection_criteria,
													min_cl_criteria=min_cl_criteria)

		elif feature == "gain":
			feature_element_matrix = get_gain(level=level, tissue=tissue, selection_criteria=selection_criteria,
											  min_cl_criteria=min_cl_criteria)

		elif feature == "loss":
			feature_element_matrix = get_loss(level=level, tissue=tissue, selection_criteria=selection_criteria,
											  min_cl_criteria=min_cl_criteria)

		elif feature == "clinicalsubtype":
			feature_element_matrix = get_clinical_subtype(level=level, tissue=tissue,
														  selection_criteria=selection_criteria,
														  min_cl_criteria=min_cl_criteria)

		print("""
		Feature matrix is prepared.\n""")
	else:
		feature_element_matrix = pandas.read_csv(
			output_path + "biomarker/features/%s_%s%s_%s_feature_element_matrix.csv"
			% (feature, level_text, t_title, title), index_col=0)
		print("""
		Feature matrix is read.\n""")
	return feature_element_matrix


def get_LR_feature(element, matrix, scale):
	"""
	Selecting specific elements (genes/mutations etc) from the adjacency feature matrix
	:param element: The selected element for the feature (gene/ mutations etc.)
	:param matrix: The adjacency feature matrix
	:param scale: Boolean value for continuous features [0,1]
	:return: Linear regression input feature matrix
	"""
	matrix = pandas.DataFrame(matrix[[element]])
	if scale:
		# Min-Max Standardise [0,1]
		matrix = pandas.DataFrame(
			MinMaxScaler(feature_range=(0, 1)).fit_transform(matrix),
			columns=matrix.columns, index=matrix.index)

	return matrix


def get_effect_size(w_biomarker, wout_biomarker, effect_function):
	"""
	Calculate Cohen's D
	:param w_biomarker: Estimate values from the models having biomarker
	:param wout_biomarker: Estimate values from the models not having biomarker
	:param effect_function: Which effect size method will be used (cohens_d or glass_d)
	:return: Cohen's d or Glass Delta
	"""

	if effect_function == "cohens_d":
		# Retrieved from https://machinelearningmastery.com/effect-size-measures-in-python/
		n1, n2 = len(w_biomarker), len(wout_biomarker)
		m1, m2 = numpy.mean(w_biomarker), numpy.mean(wout_biomarker)
		s1, s2 = numpy.var(w_biomarker, ddof=1), numpy.var(wout_biomarker, ddof=1)
		# Pooled standard deviation
		s = numpy.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
		return (m1 - m2) / s

	elif effect_function == "glass_d":
		# Function is retrieved from https://www.uv.es/~friasnav/EffectSizeBecker.pdf
		m1, m2 = numpy.mean(w_biomarker), numpy.mean(wout_biomarker)
		sd = numpy.std(wout_biomarker)
		return (m1 - m2) / sd


def get_cohens_f2(response, feature):
	# Function retrieved from https://www.analysisinn.com/post/cohen-s-f2-definition-criterion-and-example/
	r2 = r2_score(feature, response)
	return r2 / (1.0 - r2)


def plot_response_w_feature(element, feature, level, tissue, stage, drug_info, selection_criteria, msi_flag, random):
	"""
	:param element: The selected element for the feature (gene/ mutations etc.)
	:param feature: The feature that will be test for association with drug data
			(mutation / transcription / cnv / amplification / deletion / methylation / protein expression )
	:param level: Level of feature selection
			(For all genes_cancer / for mutation + genes_driver_mut + driver_mutations)
	:param tissue: Which tissue the data will be separated into
	:param stage: mono / combo / delta
	:param drug_info: Name of library of drug combination name
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param selection_criteria: How to select the features (3 for binary, variance for continuous)
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if feature == "transcription":
		title = "log2_filtered_" + str(selection_criteria)
	elif feature == "proteomics":
		title = "averaged_" + str(selection_criteria)
	else:
		title = str(selection_criteria)

	if len(drug_info.split("/")) > 1:
		d_title = drug_info.split("/")[0] + "_" + drug_info.split("/")[1]
	else:
		d_title = drug_info

	feature_title = feature_annotation_title(feature=feature, feature_level=level)

	dim_df = pandas.read_csv(output_path + "biomarker/LR/input_LR/%s/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
							 % (stage, t_title[1:], d_title, stage, feature, level, element, d_title, t_title, title,
								msi_title, random_text),
							 index_col=0)

	response_df, feature_df = dim_df[["response"]], dim_df[[element]]
	df = pandas.concat([response_df, feature_df], axis=1)

	if feature in ["mutation", "amplification", "deletion", "hypermethylation", "gain", "loss", "clinicalsubtype"]:

		df[feature_title] = df.apply(
			lambda x: feature_plot_annotation(feature=feature, feature_level=level, element=element)[0] if x[
																											   element] == 1
			else feature_plot_annotation(feature=feature, feature_level=level, element=element)[1], axis=1)

		cohen_d = get_effect_size(w_biomarker=df[df[element] == 1]["response"].values,
								  wout_biomarker=df[df[element] == 0]["response"].values,
								  effect_function="cohens_d")

		_, mann_pval = stats.mannwhitneyu(df[df[element] == 0]["response"].values,
										  df[df[element] == 1]["response"].values)

		fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(3.75, 6))

		plt.suptitle("Therapy Effects of %s\n%s on $%s$\n%s"
					 % (drug_info, feature_title, element, tissue if tissue is not None else "Pancancer"), fontsize=12)

		order = [feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
				 feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]]

		sns.stripplot(x=feature_title, y="response", data=df,
					  palette={
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]: "#C5C5C5",
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[0]: (
						  0.0, 0.0, 1.0)},
					  ax=axis[0, 0], alpha=0.5, size=6, order=order)

		sns.boxplot(x=feature_title, y="response", data=df, ax=axis[0, 0],
					flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
					whiskerprops=whiskerprops, order=order, width=0.5)

		axis[0, 0].legend([], [], frameon=False)
		axis[0, 0].set_title(
			"Cohen's d = %.3f\n%s (%d) | %s (%d)"
			% (cohen_d, feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
			   len(df[df[element] == 1]["response"].values),
			   feature_plot_annotation(feature=feature, feature_level=level, element=element)[1],
			   len(df[df[element] == 0]["response"].values)), fontsize=10)

		axis[0, 0].set_xlabel(feature_title, fontsize=12)
		axis[0, 0].tick_params(axis="x", which="major", labelsize=5)
		axis[0, 0].set_ylabel("Scaled IC50 (XMID)", fontsize=12)

		add_stat_annotation(axis[0, 0], data=df, x=feature_title, y="response", order=order,
							box_pairs=[
								(feature_plot_annotation(feature=feature, feature_level=level, element=element)[1],
								 feature_plot_annotation(feature=feature, feature_level=level, element=element)[0])],
							perform_stat_test=False, pvalues=[mann_pval], loc='inside', verbose=2, text_format="full")

	else:
		fig, axis = plt.subplots(1, 2, squeeze=False, figsize=(7.5, 8))

		# Group continuous data with their quantiles
		q4 = feature_df[element].quantile([0.4, 0.6], interpolation="midpoint")
		selected_models = list(feature_df[feature_df[element] > q4[0.6]].index)

		df[feature_title] = df.apply(
			lambda x: feature_plot_annotation(feature=feature, feature_level=level, element=element)[
				0] if x.name in selected_models
			else feature_plot_annotation(feature=feature, feature_level=level, element=element)[1], axis=1)

		cohen_f2 = get_cohens_f2(response=df.response.values, feature=df[element].values)

		cohen_d = get_effect_size(w_biomarker=df[df.index.isin(selected_models)]["response"].values,
								  wout_biomarker=df[~df.index.isin(selected_models)]["response"].values,
								  effect_function="cohens_d")

		_, mann_pval = stats.mannwhitneyu(df[~df.index.isin(selected_models)]["response"].values,
										  df[df.index.isin(selected_models)]["response"].values, alternative="greater")

		slope, intercept, r_value, p_value, _ = scipy.stats.linregress(df[element].values, df["response"].values)
		r2 = r_value ** 2

		plt.suptitle("Therapy Effects of %s\n%s on $%s$\n%s"
					 % (drug_info, feature_title, element, tissue if tissue is not None else "Pancancer"), fontsize=12)

		order = [feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
				 feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]]

		sns.stripplot(x=feature_title, y="response", data=df,
					  palette={
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]: "#C5C5C5",
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[0]: (
						  0.0, 0.0, 1.0)},
					  ax=axis[0, 0], alpha=0.5, size=6, order=order)

		sns.boxplot(x=feature_title, y="response", data=df, ax=axis[0, 0],
					flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
					whiskerprops=whiskerprops, order=order, width=0.5)
		axis[0, 0].legend([], [], frameon=False)
		axis[0, 0].set_title(
			"Cohen's d = %.3f\n%s (%d) | %s (%d)"
			% (cohen_d[0], feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
			   len(df[df.name.isin(selected_models)]["response"].values),
			   feature_plot_annotation(feature=feature, feature_level=level, element=element)[1],
			   len(df[~df.name.isin(selected_models)]["response"].values)), fontsize=10)

		axis[0, 0].set_xlabel('')
		axis[0, 0].tick_params(axis="x", which="major", labelsize=10)
		axis[0, 0].set_ylabel("Scaled IC50 (XMID)")

		add_stat_annotation(axis[0, 0], data=df, x=feature_title, y="response", order=order,
							box_pairs=[(feature_plot_annotation(feature=feature, feature_level=level,
																element=element)[1],
										feature_plot_annotation(feature=feature, feature_level=level,
																element=element)[0])],
							perform_stat_test=False, pvalues=mann_pval, loc='inside', verbose=2, text_format="full")

		axis[0, 1].plot(df[element].values, df["response"].values, 'o', alpha=0.5, markersize=3,
						linewidth=0, color="navy", label="Scaled IC50 (XMID)")
		axis[0, 1].plot(df[element].values, intercept + (slope * df[element].values),
						'navy', label="Fitted Line")

		axis[0, 1].vlines(x=q4[0.4], ymin=0, ymax=numpy.max(df.response.values), colors='darkgrey', alpha=0.5,
						  ls='--', lw=.75)
		axis[0, 1].vlines(x=q4[0.6], ymin=0, ymax=numpy.max(df.response.values), colors='darkgrey', alpha=0.5,
						  ls='--', lw=.75)
		axis[0, 1].legend()
		axis[0, 1].set_xlabel(feature_title)
		axis[0, 1].set_ylabel("Scaled IC50 (XMID)")
		axis[0, 1].set_title(
			"Cohen's f\u00b2 = %.3f | R\u00b2 = %.3f\nResponse (%d)"
			% (cohen_f2, r2, len(df.response.values)), fontsize=10)
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)
	plt.tight_layout()

	if os.path.isdir(output_path + "biomarker/figures/responses/") is False:
		os.system("mkdir %s/biomarker/figures/responses/")

	plt.savefig(
		output_path + "biomarker/figures/responses/response_%s_%s_%s_%s%s_with_%s%s%s_boxplots.pdf"
		% (d_title, element, feature, level, t_title, selection_criteria, msi_title, random_text), dpi=300)
	plt.savefig(
		output_path + "biomarker/figures/responses/response_%s_%s_%s_%s%s_with_%s%s%s_boxplots.jpg"
		% (d_title, element, feature, level, t_title, selection_criteria, msi_title, random_text), dpi=300)
	plt.savefig(
		output_path + "biomarker/figures/responses/response_%s_%s_%s_%s%s_with_%s%s%s_boxplots.png"
		% (d_title, element, feature, level, t_title, selection_criteria, msi_title, random_text), dpi=300)
	plt.close()
	return 1


def plot_all_response_w_feature(element, feature, level, tissue, stage, drug_info, selection_criteria, msi_flag):
	"""
	:param element: The selected element for the feature (gene/ mutations etc.)
	:param feature: The feature that will be test for association with drug data
			(mutation / transcription / cnv / amplification / deletion / methylation / protein expression )
	:param level: Level of feature selection
			(For all genes_cancer / for mutation + genes_driver_mut + driver_mutations)
	:param tissue: Which tissue the data will be separated into
	:param stage: mono / combo / delta
	:param drug_info: Name of library of drug combination name
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param selection_criteria: How to select the features (3 for binary, variance for continuous)
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if len(drug_info.split("/")) > 1:
		d_title = drug_info.split("/")[0] + "_" + drug_info.split("/")[1]
		d_col = "DrugComb"
	else:
		d_title = drug_info
		d_col = "library_name"

	if feature == "transcription":
		title = "log2_filtered_" + str(selection_criteria)
		scale = True
	elif feature == "proteomics":
		title = "averaged_" + str(selection_criteria)
		scale = True
	else:
		title = str(selection_criteria)
		scale = False

	feature_title = feature_annotation_title(feature=feature, feature_level=level)

	whole_response = get_response_data(tissue=tissue, stage=stage, estimate_lr="XMID")
	response_df = get_LR_response(drug_info=drug_info, response_df=whole_response, drug_col=d_col)

	whole_feature = create_feature_matrix(feature=feature, level=level, tissue=tissue,
										  selection_criteria=selection_criteria, stage=stage,
										  estimate_lr="XMID", min_cl_criteria=10)
	feature_df = get_LR_feature(element=element, matrix=whole_feature, scale=scale)

	df = pandas.concat([response_df, feature_df], axis=1)
	df = df.dropna()

	if feature in ["mutation", "amplification", "deletion", "hypermethylation", "gain", "loss", "clinicalsubtype"]:

		df[feature_title] = df.apply(
			lambda x: feature_plot_annotation(feature=feature, feature_level=level, element=element)[0] if x[
																											   element] == 1
			else feature_plot_annotation(feature=feature, feature_level=level, element=element)[1], axis=1)

		cohen_d = get_effect_size(w_biomarker=df[df[element] == 1]["response"].values,
								  wout_biomarker=df[df[element] == 0]["response"].values,
								  effect_function="cohens_d")

		_, mann_pval = stats.mannwhitneyu(df[df[element] == 0]["response"].values,
										  df[df[element] == 1]["response"].values)

		fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(3.75, 6))

		plt.suptitle("Therapy Effects of %s\n%s on $%s$\n%s"
					 % (drug_info, feature_title, element, tissue), fontsize=12)

		order = [feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
				 feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]]

		sns.stripplot(x=feature_title, y="response", data=df,
					  palette={
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]: "#C5C5C5",
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[0]: (
						  0.0, 0.0, 1.0)},
					  ax=axis[0, 0], alpha=0.5, size=6, order=order)

		sns.boxplot(x=feature_title, y="response", data=df, ax=axis[0, 0],
					flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
					whiskerprops=whiskerprops, order=order, width=0.5)

		axis[0, 0].legend([], [], frameon=False)
		axis[0, 0].set_title(
			"Cohen's d = %.3f\n%s (%d) | %s (%d)"
			% (cohen_d, feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
			   len(df[df[element] == 1]["response"].values),
			   feature_plot_annotation(feature=feature, feature_level=level, element=element)[1],
			   len(df[df[element] == 0]["response"].values)), fontsize=10)

		axis[0, 0].set_xlabel(feature_title, fontsize=12)
		axis[0, 0].tick_params(axis="x", which="major", labelsize=5)
		axis[0, 0].set_ylabel("Scaled IC50 (XMID)", fontsize=12)

		add_stat_annotation(axis[0, 0], data=df, x=feature_title, y="response", order=order,
							box_pairs=[
								(feature_plot_annotation(feature=feature, feature_level=level, element=element)[1],
								 feature_plot_annotation(feature=feature, feature_level=level, element=element)[0])],
							perform_stat_test=False, pvalues=[mann_pval], loc='inside', verbose=2, text_format="full")

	else:
		fig, axis = plt.subplots(1, 2, squeeze=False, figsize=(7.5, 8))

		# Group continuous data with their quantiles
		q4 = feature_df[element].quantile([0.4, 0.6], interpolation="midpoint")
		selected_models = list(feature_df[feature_df[element] > q4[0.6]].index)

		df[feature_title] = df.apply(
			lambda x: feature_plot_annotation(feature=feature, feature_level=level, element=element)[
				0] if x.name in selected_models
			else feature_plot_annotation(feature=feature, feature_level=level, element=element)[1], axis=1)

		cohen_f2 = get_cohens_f2(response=df.response.values, feature=df.feature.values)

		cohen_d = get_effect_size(w_biomarker=df[df.name.isin(selected_models)]["response"].values,
								  wout_biomarker=df[~df.name.isin(selected_models)]["response"].values,
								  effect_function="cohens_d")

		_, mann_pval = stats.mannwhitneyu(df[~df.name.isin(selected_models)]["response"].values,
										  df[df.name.isin(selected_models)]["response"].values, alternative="greater")

		slope, intercept, r_value, p_value, _ = scipy.stats.linregress(df[element].values, df["response"].values)
		r2 = r_value ** 2

		plt.suptitle("Therapy Effects of %s\n%s on $%s$\n%s"
					 % (drug_info, feature_title, element, tissue), fontsize=12)

		order = [feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
				 feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]]

		sns.stripplot(x=feature_title, y="response", data=df,
					  palette={
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[1]: "#C5C5C5",
						  feature_plot_annotation(feature=feature, feature_level=level, element=element)[0]: (
						  0.0, 0.0, 1.0)},
					  ax=axis[0, 0], alpha=0.5, size=6, order=order)

		sns.boxplot(x=feature_title, y="response", data=df, ax=axis[0, 0],
					flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
					whiskerprops=whiskerprops, order=order, width=0.5)
		axis[0, 0].legend([], [], frameon=False)
		axis[0, 0].set_title(
			"Cohen's d = %.3f\n%s (%d) | %s (%d)"
			% (cohen_d[0], feature_plot_annotation(feature=feature, feature_level=level, element=element)[0],
			   len(df[df.name.isin(selected_models)]["response"].values),
			   feature_plot_annotation(feature=feature, feature_level=level, element=element)[1],
			   len(df[~df.name.isin(selected_models)]["response"].values)), fontsize=10)

		axis[0, 0].set_xlabel('')
		axis[0, 0].tick_params(axis="x", which="major", labelsize=10)
		axis[0, 0].set_ylabel("Scaled IC50 (XMID)")

		add_stat_annotation(axis[0, 0], data=df, x=feature_title, y="response", order=order,
							box_pairs=[(feature_plot_annotation(feature=feature, feature_level=level,
																element=element)[1],
										feature_plot_annotation(feature=feature, feature_level=level,
																element=element)[0])],
							perform_stat_test=False, pvalues=mann_pval, loc='inside', verbose=2, text_format="full")

		axis[0, 1].plot(df[element].values, df["response"].values, 'o', alpha=0.5, markersize=3,
						linewidth=0, color="navy", label="Scaled IC50 (XMID)")
		axis[0, 1].plot(df[element].values, intercept + (slope * df[element].values),
						'navy', label="Fitted Line")

		axis[0, 1].vlines(x=q4[0.4], ymin=0, ymax=numpy.max(df.response.values), colors='darkgrey', alpha=0.5,
						  ls='--', lw=.75)
		axis[0, 1].vlines(x=q4[0.6], ymin=0, ymax=numpy.max(df.response.values), colors='darkgrey', alpha=0.5,
						  ls='--', lw=.75)
		axis[0, 1].legend()
		axis[0, 1].set_xlabel(feature_title)
		axis[0, 1].set_ylabel("Scaled IC50 (XMID)")
		axis[0, 1].set_title(
			"Cohen's f\u00b2 = %.3f | R\u00b2 = %.3f\nResponse (%d)"
			% (cohen_f2, r2, len(df.response.values)), fontsize=10)
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)
	plt.tight_layout()

	if os.path.isdir(output_path + "biomarker/figures/responses/") is False:
		os.system("mkdir %s/biomarker/figures/responses/")

	plt.savefig(
		output_path + "biomarker/figures/responses/all_response_%s_%s_%s_%s%s_with_%s%s_boxplots.pdf"
		% (d_title, element, feature, level, t_title, selection_criteria, msi_title), dpi=300)
	plt.savefig(
		output_path + "biomarker/figures/responses/all_response_%s_%s_%s_%s%s_with_%s%s_boxplots.jpg"
		% (d_title, element, feature, level, t_title, selection_criteria, msi_title), dpi=300)
	plt.savefig(
		output_path + "biomarker/figures/responses/all_response_%s_%s_%s_%s%s_with_%s%s_boxplots.png"
		% (d_title, element, feature, level, t_title, selection_criteria, msi_title), dpi=300)
	plt.close()
	return 1


def read_biomarker(feature, level, tissue, stage, estimate_lr, selection_criteria, msi_flag, random):
	"""
	Running biomarker analysis
	:param feature: The feature that will be test for association with drug data
	:param level: Level of feature selection
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param level: Level of feature selection (genes / mutations)
	:param tissue: Which tissue the data will be separated into
	:param stage: LIBRARY / COMBO / DELTA
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return: Biomarker Data Frame
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if feature == "transcription":
		title = "log2_filtered_" + str(selection_criteria)
	elif feature == "proteomics":
		title = "averaged_" + str(selection_criteria)
	else:
		title = str(selection_criteria)

	if "LR_%s_%s_%s_%s%s_%s%s%s.csv" % (
	stage, estimate_lr, feature, level, t_title, title, msi_title, random_text) in os.listdir(
			output_path + "biomarker/LR/run_LR/"):
		biomarker_df = pandas.read_csv(output_path + "biomarker/LR/run_LR/LR_%s_%s_%s_%s%s_%s%s%s.csv"
									   % (stage, estimate_lr, feature, level, t_title, title, msi_title, random_text),
									   low_memory=False, index_col=0)
		return biomarker_df
	else:
		return None


def collect_biomarker(tissue, merged, msi_flag, random):
	"""
	:param tissue: Which tissue the data will be separated into
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	if tissue is not None:
		t_title = "_".join(tissue.split(" "))

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if merged:
		if "whole_biomarker%s%s.csv" % (msi_title, random_text) not in os.listdir(
				output_path + "biomarker/LR/combined_LR/"):
			biomarkers = list()
			feature_dictionary = {"mutation": {"levels": {"genes_driver_mut": {"selection_criteria": [5]},
														  "mutations_driver": {"selection_criteria": [5]}}},
								  "transcription": {"levels": {"genes_cancer": {"selection_criteria": ["variance"]}}},
								  "proteomics": {"levels": {"genes_cancer": {"selection_criteria": ["variance"]}}},
								  "amplification": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "deletion": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "hypermethylation": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "gain": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "loss": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "msi": {"levels": {"genes_cancer": {"selection_criteria": [5]}}}}

			estimate_list = ["XMID"]
			stage_list = ["mono", "combo", "delta"]
			tissue_list = [i for i in get_tissue_types() if i is not None or pandas.isna(i) is False] + ["pansolid",
																										 "panliquid"]
			"""
			no_feature_project_list = ["Testis", "Vulva", "Uterus", "Small Intestine", "Prostate", "Biliary Tract",
									   "Adrenal Gland", "Thyroid", "Myeloma", "Endometrium", "Cervix"]
			tissue_list = [t for t in tissue_list if t not in no_feature_project_list]
			"""
			for tissue in tissue_list:
				for feature, levels in feature_dictionary.items():
					for s in stage_list:
						for e in estimate_list:
							for l in levels["levels"].keys():
								for criteria in levels["levels"][l]["selection_criteria"]:
									print(tissue)
									print(feature)
									print(l)
									biomarker_df = read_biomarker(feature=feature, level=l, tissue=tissue,
																  stage=s, estimate_lr=e, selection_criteria=criteria,
																  msi_flag=False, random=random)
									if biomarker_df is not None:
										biomarker_df["tissue_type"] = tissue
										biomarkers.append(biomarker_df)

									if tissue == "Breast":
										for clinical_level in ["LumA", "LumB", "Basal", "Her2"]:
											c_biomarker_df = read_biomarker(feature="clinicalsubtype",
																			level=clinical_level, tissue=tissue,
																			stage=s, estimate_lr=e,
																			selection_criteria=criteria,
																			msi_flag=False, random=random)
											if c_biomarker_df is not None:
												c_biomarker_df["tissue_type"] = tissue
												biomarkers.append(c_biomarker_df)

									if tissue == "Large Intestine":
										for clinical_level in ["CRISA", "CRISB", "CRISC", "CRISD", "CRISE", "CMS1",
															   "CMS2",
															   "CMS3", "CMS4"]:
											c_biomarker_df = read_biomarker(feature="clinicalsubtype",
																			level=clinical_level, tissue=tissue,
																			stage=s, estimate_lr=e,
																			selection_criteria=criteria,
																			msi_flag=False, random=random)
											if c_biomarker_df is not None:
												c_biomarker_df["tissue_type"] = tissue
												biomarkers.append(c_biomarker_df)

			whole_biomarker_df = pandas.concat(biomarkers, ignore_index=True)
			whole_biomarker_df.to_csv(output_path + "biomarker/LR/combined_LR/whole_biomarker%s%s.csv"
									  % (msi_title, random_text))
		else:
			whole_biomarker_df = pandas.read_csv(output_path + "biomarker/LR/combined_LR/whole_biomarker%s%s.csv"
												 % (msi_title, random_text), index_col=0, low_memory=False)
	else:
		if "%s_biomarker%s%s.csv" % (t_title, msi_title, random_text) not in os.listdir(
				output_path + "biomarker/LR/combined_LR/"):
			biomarkers = list()
			feature_dictionary = {"mutation": {"levels": {"genes_driver_mut": {"selection_criteria": [5]},
														  "mutations_driver": {"selection_criteria": [5]}}},
								  "transcription": {"levels": {"genes_cancer": {"selection_criteria": ["variance"]}}},
								  "proteomics": {"levels": {"genes_cancer": {"selection_criteria": ["variance"]}}},
								  "amplification": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "deletion": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "hypermethylation": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "gain": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "loss": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
								  "msi": {"levels": {"genes_cancer": {"selection_criteria": [5]}}}}

			estimate_list = ["XMID"]
			stage_list = ["mono", "combo", "delta"]
			for feature, levels in feature_dictionary.items():
				for s in stage_list:
					for e in estimate_list:
						for l in levels["levels"].keys():
							for criteria in levels["levels"][l]["selection_criteria"]:
								biomarker_df = read_biomarker(feature=feature, level=l, tissue=tissue,
															  stage=s, estimate_lr=e, selection_criteria=criteria,
															  msi_flag=False, random=random)
								if biomarker_df is not None:
									biomarker_df["tissue_type"] = tissue
									biomarkers.append(biomarker_df)

								if tissue == "Breast":
									for clinical_level in ["LumA", "LumB", "Basal", "Her2"]:
										c_biomarker_df = read_biomarker(feature="clinicalsubtype",
																		level=clinical_level, tissue=tissue,
																		stage=s, estimate_lr=e,
																		selection_criteria=criteria,
																		msi_flag=False, random=random)
										if c_biomarker_df is not None:
											c_biomarker_df["tissue_type"] = tissue
											biomarkers.append(c_biomarker_df)

								if tissue == "Large Intestine":
									for clinical_level in ["CRISA", "CRISB", "CRISC", "CRISD", "CRISE", "CMS1", "CMS2",
														   "CMS3", "CMS4"]:
										c_biomarker_df = read_biomarker(feature="clinicalsubtype",
																		level=clinical_level, tissue=tissue,
																		stage=s, estimate_lr=e,
																		selection_criteria=criteria,
																		msi_flag=False, random=random)
										if c_biomarker_df is not None:
											c_biomarker_df["tissue_type"] = tissue
											biomarkers.append(c_biomarker_df)

			whole_biomarker_df = pandas.concat(biomarkers, ignore_index=True)
			whole_biomarker_df.to_csv(output_path + "biomarker/LR/combined_LR/%s_biomarker%s%s.csv"
									  % (t_title, msi_title, random_text))
		else:
			whole_biomarker_df = pandas.read_csv(output_path + "biomarker/LR/combined_LR/%s_biomarker%s%s.csv"
												 % (t_title, msi_title, random_text), index_col=0, low_memory=False)
	return whole_biomarker_df


def get_fda_association(msi_flag, random):
	fda = pandas.read_csv(input_path + "fda/pharmacogenomic_biomarkers_drug_fda.csv")
	fda["drug_name"] = fda.apply(lambda x: x.Drug.split(" (")[0], axis=1)
	fda["biomarker"] = fda.apply(lambda x: x["Biomarker†"].split(" (")[0].strip(), axis=1)
	fda["biomarker"] = fda.apply(lambda x: x["Biomarker†"].split(" (")[0].strip(), axis=1)
	fda["area"] = fda.apply(lambda x: x["Therapeutic Area*"].strip(), axis=1)
	fda = fda[fda.area.isin(["Oncology", "Onoclogy"])]
	fda = fda[["drug_name", "biomarker"]]
	fda["in_screens"] = fda.apply(lambda x: True if x.drug_name in single_agents else False, axis=1)
	fda = fda[fda.in_screens]
	fda = fda.replace("ESR", "ESR1")
	fda_associtions = [row.drug_name + ":" + i for ind, row in fda.iterrows() for i in row.biomarker.split(", ")]
	for i in fda_associtions:
		if i.split(":")[1] == "RAS":
			fda_associtions.append(i.split(":")[0] + ":" + "KRAS")
			fda_associtions.append(i.split(":")[0] + ":" + "NRAS")
			fda_associtions.append(i.split(":")[0] + ":" + "HRAS")
		if i.split(":")[1] == "BRCA":
			fda_associtions.append(i.split(":")[0] + ":" + "BRCA1")
			fda_associtions.append(i.split(":")[0] + ":" + "BRCA2")

	whole_biomarker_df = collect_biomarker(tissue=None, merged=True, msi_flag=msi_flag, random=random)
	mono_biomarkers = whole_biomarker_df[whole_biomarker_df.Stage == "mono"]
	single_agents = mono_biomarkers.library_name.unique()
	mono_associations = mono_biomarkers[mono_biomarkers.Association == "sensitivity"].Association_name.unique()

	common_associations = set(mono_associations).intersection(set(fda_associtions))
	notfound_associations = set(fda_associtions).difference(set(mono_associations))

	return common_associations, notfound_associations


def take_mono_associations(tissue, fdr_limit, merged, msi_flag, random):
	"""
	Collecting mono therapy associated biomarkers
	:param tissue: Which tissue the data will be separated into
	:param fdr_limit: The highest limit for FDR
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))

	if merged: t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if "mono_biomarker%s%s%s.csv" % (t_title, msi_title, random_text) not in os.listdir(
			output_path + "biomarker/LR/combined_LR/"):
		association_df = collect_biomarker(tissue=tissue, merged=merged, msi_flag=msi_flag, random=random)
		if tissue is not None:
			association_df = association_df[association_df.tissue_type == tissue]

		if len(association_df.index) > 0:
			library_association_df = association_df[association_df.Stage == "mono"]

			library_association_df["Association_tag"] = library_association_df.apply(
				lambda x: x.Gene + " : " + feature_annotation(x.Feature, x.Feature_level) + ">" + x.library_name,
				axis=1)

			library_association_df["Significance"] = library_association_df.apply(
				lambda x: "Significant" if x.FDR < fdr_limit else "Not significant", axis=1)

			library_association_df["Group"] = library_association_df.apply(
				lambda x: x.Stage + "-" + x.Estimate_LR, axis=1)

			library_association_df.to_csv(output_path + "biomarker/LR/combined_LR/mono_biomarker%s%s%s.csv"
										  % (t_title, msi_title, random_text), index=True)
		else:
			library_association_df = None
	else:
		library_association_df = pandas.read_csv(output_path + "biomarker/LR/combined_LR/mono_biomarker%s%s%s.csv"
												 % (t_title, msi_title, random_text), index_col=0)

	return library_association_df


def take_combination_associations(tissue, fdr_limit, merged, msi_flag, random):
	"""
	Collecting combinatorial therapy associated biomarkers
	:param tissue: Which tissue the data will be separated into
	:param fdr_limit: The highest limit for FDR
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))

	if merged: t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if "combination_biomarker%s%s%s.csv" % (t_title, msi_title, random_text) not in os.listdir(
			output_path + "biomarker/LR/combined_LR/"):
		association_df = collect_biomarker(tissue=tissue, merged=merged, msi_flag=msi_flag, random=random)

		if tissue is not None:
			association_df = association_df[association_df.Tissue == tissue]

		if len(association_df.index) > 0:

			combination_association_df = association_df[association_df.Stage.isin(["combo", "delta"])]

			combination_association_df["Association_tag"] = combination_association_df.apply(
				lambda x: x.Gene + " : " + feature_annotation(x.Feature, x.Feature_level) + ">" + x.DrugComb, axis=1)

			combination_association_df["Significance"] = combination_association_df.apply(
				lambda x: "Significant" if x.FDR < fdr_limit else "Not significant", axis=1)

			combination_association_df["Group"] = combination_association_df.apply(
				lambda x: x.Stage + "-" + x.Estimate_LR, axis=1)

			combination_association_df.to_csv(output_path + "biomarker/LR/combined_LR/combination_biomarker%s%s%s.csv"
											  % (t_title, msi_title, random_text), index=True)

		else:
			combination_association_df = None
	else:
		combination_association_df = pandas.read_csv(
			output_path + "biomarker/LR/combined_LR/combination_biomarker%s%s%s.csv"
			% (t_title, msi_title, random_text), index_col=0)
	return combination_association_df


def separate_mono_association(tissue, fdr_limit, merged, msi_flag, random):
	"""
	Separation of mono drugs according to their association from LR
	:param tissue: Which tissue the data will be separated into
	:param fdr_limit: The highest limit for FDR
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	library_association_df = take_mono_associations(tissue=tissue, fdr_limit=fdr_limit,
													merged=merged, msi_flag=msi_flag, random=random)

	active_sensitive_mono, active_resistant_mono, nonactive_mono = list(), list(), list()
	for library in library_association_df["library_name"].unique():
		library_df = library_association_df[library_association_df.library_name == library]
		active_df = library_df[library_df.FDR <= fdr_limit]
		if len(active_df.index) == 0:
			nonactive_mono.append(library)
		else:
			sensitive_df = active_df[active_df.Effect == "sensitivity"]
			if len(sensitive_df.index) != 0:
				active_sensitive_mono.append(library)
			resistant_df = active_df[active_df.Effect == "resistance"]
			if len(resistant_df.index) != 0:
				active_resistant_mono.append(library)

	return library_association_df, active_sensitive_mono, active_resistant_mono, nonactive_mono


def separate_combination_association(tissue, fdr_limit, merged, msi_flag, random):
	"""
	Separation of combinatorial drugs according to their association from LR
	:param tissue: Which tissue the data will be separated into
	:param fdr_limit: The highest limit for FDR
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	combination_association = take_combination_associations(tissue=tissue, fdr_limit=fdr_limit,
															merged=merged, msi_flag=msi_flag, random=random)

	active_sensitive_combi, active_resistant_combi, nonactive_combi = list(), list(), list()
	for combi in combination_association["DrugComb"].unique():
		combi_df = combination_association[combination_association.DrugComb == combi]
		active_df = combi_df[combi_df.FDR <= fdr_limit]
		if len(active_df.index) == 0:
			nonactive_combi.append(combi)
		else:
			sensitive_df = active_df[active_df.Effect == "sensitivity"]
			if len(sensitive_df.index) != 0:
				active_sensitive_combi.append(combi)
			resistant_df = active_df[active_df.Effect == "resistance"]
			if len(resistant_df.index) != 0:
				active_resistant_combi.append(combi)
			synergy_df = active_df[active_df.Effect == "synergy"]
			if len(synergy_df.index) != 0:
				active_synergy_combi.append(combi)
			antagonist_df = active_df[active_df.Effect == "antagonism"]
			if len(antagonist_df.index) != 0:
				active_antagonist_combi.append(combi)

	return combination_association, active_sensitive_combi, active_resistant_combi, \
		   active_synergy_combi, active_antagonist_combi, nonactive_combi


def count_association(tissue, plot_name, plotting, treatment, fdr_limit, merged, msi_flag, random):
	"""
	Plot the separation of library drugs according to their association from LR
	:param tissue: Which tissue the data will be separated into
	:param plot_name: The name of the resulting plot
	:param plotting: T/F
	:param treatment: library/combination
	:param fdr_limit: FDR limit
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	if treatment == "mono":
		_, sens, res, non_active = separate_mono_association(tissue=tissue, fdr_limit=fdr_limit,
															 merged=merged, msi_flag=msi_flag, random=random)
	else:
		_, sens, res, syn, ant, non_active = separate_combination_association(tissue=tissue, fdr_limit=fdr_limit,
																			  merged=merged, msi_flag=msi_flag,
																			  random=random)

	print("Statistics:\nNumber of non active %s drugs: %d" % (treatment, len(non_active)))
	print("Number of active %s drugs: %d" % (treatment, len(set(sens).union(set(res)))))
	print("Number of only sensitive %s drugs: %d" % (treatment, len(set(sens).difference(set(res)))))
	# print("Number of only resistant %s drugs: %d" % (treatment, len(set(res).difference(set(sens)))))
	# print("Number of both sensitive and resistant %s drugs: %d" % (treatment, len(set(res).intersection(set(sens)))))
	if treatment == "combination":
		print("Number of only synergistic combination drugs: %d" % (len(set(syn).difference(set(sen)))))
		print("Number of synergistic and sensitive combination drugs: %d" % (len(set(syn).intersection(set(sen)))))

	df = pandas.DataFrame(columns=["Active %s" % treatment, "Non Active %s" % treatment], index=["Number of Drugs"])
	df.loc["Number of Drugs", "Active %s" % treatment] = len(list(set(sens + res)))
	df.loc["Number of Drugs", "Non Active %s" % treatment] = len(non_active)
	df = df.melt()
	df.columns = ["Activity", "Number of Drugs"]
	if plotting:
		plt.figure(figsize=(5, 6.5))
		plt.xlim(-0.5, 1.5)
		plt.bar(list(range(len(df["Activity"].index))), list(df["Number of Drugs"]), color=["red", "grey"],
				align='center', width=0.4, alpha=0.7)
		plt.xticks(list(range(len(df["Activity"].index))), df["Activity"])
		plt.xlabel("Activity")
		plt.ylabel("Number of Drugs")
		plt.title("LR Associated and non-associated %s drugs" % treatment)
		plt.savefig(output_path + "biomarker/figures/drug_activity/" + plot_name + ".pdf", dpi=300)
		plt.savefig(output_path + "biomarker/figures/drug_activity/activity_number_" + plot_name + ".jpg", dpi=300)
		plt.close()

	return df


def test_combo_library_combination(drug_comb, comb_type, estimate_lr, feature,
								   feature_level, element, selection_criteria,
								   plotting, tissue, min_limitation, msi_flag, random):
	"""
	Plot regression of COMBO EMAX/IC50
	:param drug_comb: The drug combination
	:param comb_type: Type of the combination (S-S-S-SYN / N-N-S-SYN etc.)
	:param estimate_lr: The estimate that will be used in the LR response - XMID  / EMAX
	:param feature: The main omics feature
	:param feature_level: The level of the feature
	:param element: The selected element for the feature (gene/ mutations etc.)
	:param selection_criteria: variance
	:param plotting: T/F
	:param tissue: Which tissue the data will be separated into
	:param min_limitation: The minimum number of responses for XXXXX
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if feature == "transcription":
		title = "log2_filtered_" + str(selection_criteria)
	elif feature == "proteomics":
		title = "averaged_" + str(selection_criteria)
	else:
		title = str(selection_criteria)

	if len(drug_comb.split("/")) > 1:
		d_title = drug_comb.split("/")[0] + "_" + drug_comb.split("/")[1]
	else:
		d_title = drug_comb

	# Boxplot information
	flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
					  markeredgecolor="none")
	medianprops = dict(linestyle="-", linewidth=1.5, color="red")
	boxprops = dict(facecolor="white", edgecolor="darkgrey")
	whiskerprops = dict(color="darkgrey")

	# Titles
	if estimate_lr == "XMID":
		estimate_title = "Scaled IC50"
	else:
		estimate_title = "Emax"

	feature_title = feature_annotation_title(feature=feature, feature_level=feature_level)

	if comb_type in ["N-N-S-N", "N-N-S-SYN", "N-R-S-N", "R-N-S-N", "N-R-S-SYN", "R-N-S-SYN"]:
		comb_title = "sensitizer"
	elif comb_type in ["N-S-S-N", "N-S-S-SYN", "S-N-S-SYN", "S-N-S-N", "N-R-S-N", "N-R-S-SYN",
					   "S-R-S-N", "S-R-S-SYN"]:
		comb_title = "improvers"
	else:
		comb_title = "remainings"

	if os.path.isdir(output_path + "biomarker/LR/input_LR/combo/%s/%s/" % (t_title[1:], d_title)) and \
			"input_%s_%s_%s_%s_%s%s_%s%s%s.csv" % (
	"combo", feature, feature_level, element, d_title, t_title, title, msi_title, random_text) \
			in os.listdir(output_path + "biomarker/LR/input_LR/combo/%s/%s/" % (t_title[1:], d_title)):

		dim_combo_df = pandas.read_csv(
			output_path + "biomarker/LR/input_LR/combo/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
			% (t_title[1:], d_title, "combo", feature, feature_level, element, d_title, t_title,
			   title, msi_title, random_text), index_col=0)

		feature_combo_df = dim_combo_df[[element]]

		if feature in ["mutation", "amplification", "deletion", "hypermethylation", "msi", "gain", "loss",
					   "clinicalsubtype"]:
			selected_models = list(feature_combo_df[feature_combo_df[element] == 1].index)
			other_models = list(feature_combo_df[feature_combo_df[element] != 1].index)
		else:
			# Group continuous data with their quantiles
			# mid=feature_combo_df[element].quantile(q=0.5, interpolation="midpoint")
			q4 = feature_combo_df[element].quantile([0.4, 0.6], interpolation="midpoint")
			# mid_selected_models = list(feature_combo_df[feature_combo_df[element] > mid].index)
			selected_models = list(feature_combo_df[feature_combo_df[element] > q4[0.6]].index)
			# mid_other_models = list(feature_combo_df[feature_combo_df[element] <= mid].index)
			other_models = list(feature_combo_df[feature_combo_df[element] <= q4[0.4]].index)

		# D1 with feature
		drug1 = drug_comb.split("/")[0]

		if os.path.isdir(output_path + "biomarker/LR/input_LR/mono/%s/%s/" % (t_title[1:], drug1)) and \
				"input_%s_%s_%s_%s_%s%s_%s%s%s.csv" % (
		"mono", feature, feature_level, element, drug1, t_title, title, msi_title, random_text) \
				in os.listdir(output_path + "biomarker/LR/input_LR/mono/%s/%s/" % (t_title[1:], drug1)):

			dim_mono_df1 = pandas.read_csv(
				output_path + "biomarker/LR/input_LR/mono/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
				% (t_title[1:], drug1, "mono", feature, feature_level, element, drug1, t_title,
				   title, msi_title, random_text), index_col=0)

			if "response" in dim_mono_df1.keys():
				d1_response = dim_mono_df1[["response"]]
				d1_in_response = d1_response[d1_response.index.isin(selected_models)]
				d1_in_responses = d1_in_response[["response"]]
				d1_in_response.columns = ["%s" % drug1]
				# D1 w/out feature
				d1_out_response = d1_response[d1_response.index.isin(other_models)]
				d1_out_responses = d1_out_response[["response"]]
				d1_out_response.columns = ["%s" % drug1]
			else:
				d1_in_responses, d1_iout_responses = None, None
		else:
			d1_in_responses, d1_iout_responses = None, None

		# D2 with feature
		drug2 = drug_comb.split("/")[1]

		if os.path.isdir(output_path + "biomarker/LR/input_LR/mono/%s/%s/" % (t_title[1:], drug2)) and \
				"input_%s_%s_%s_%s_%s%s_%s%s%s.csv" % (
		"mono", feature, feature_level, element, drug2, t_title, title, msi_title, random_text) \
				in os.listdir(output_path + "biomarker/LR/input_LR/mono/%s/%s/" % (t_title[1:], drug2)):

			dim_mono_df2 = pandas.read_csv(
				output_path + "biomarker/LR/input_LR/mono/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
				% (t_title[1:], drug2, "mono", feature, feature_level, element, drug2, t_title,
				   title, msi_title, random_text), index_col=0)

			if "response" in dim_mono_df2.keys():
				# d2_response = get_LR_response(drug_info=drug2, response_df=mono_response_matrix, drug_col="library_name")
				d2_response = dim_mono_df2[["response"]]
				d2_in_response = d2_response[d2_response.index.isin(selected_models)]
				d2_in_responses = d2_in_response[["response"]]
				d2_in_response.columns = ["%s" % drug2]
				# D2 w/out feature
				d2_out_response = d2_response[d2_response.index.isin(other_models)]
				d2_out_responses = d2_out_response[["response"]]
				d2_out_response.columns = ["%s" % drug2]
			else:
				d2_in_responses, d2_iout_responses = None, None
		else:
			d2_in_responses, d2_iout_responses = None, None

		# C with feature
		# comb_response = get_LR_response(drug_info=drug_comb, response_df=comb_response_matrix, drug_col="DrugComb")
		comb_response = dim_combo_df[["response"]]
		comb_in_response = comb_response[comb_response.index.isin(selected_models)]
		comb_in_responses = comb_in_response[["response"]]
		comb_in_response.columns = ["%s" % drug_comb]
		# C w/out feature
		comb_out_response = comb_response[comb_response.index.isin(other_models)]
		comb_out_responses = comb_out_response[["response"]]
		comb_out_response.columns = ["%s" % drug_comb]

		if d1_in_responses is not None and d2_in_responses is not None and comb_in_responses is not None:
			if feature in ["transcription", "proteomics"]:
				comb_d = pandas.concat([comb_response, feature_combo_df], axis=1)
				comb_d = comb_d.dropna()
				d1_d = pandas.concat([d1_response, dim_mono_df1[[element]]], axis=1)
				d1_d = d1_d.dropna()
				d2_d = pandas.concat([d2_response, dim_mono_df2[[element]]], axis=1)
				d2_d = d2_d.dropna()
				cohen_f2 = get_cohens_f2(response=comb_d["response"].values,
										 feature=comb_d[element].values)
				slope, intercept, r_value, p_value, _ = \
					scipy.stats.linregress(comb_d[element].values, comb_d["response"].values)
				slope_d1, intercept_d1, r_value_d1, _, _ = \
					scipy.stats.linregress(d1_d[element].values, d1_d["response"].values)
				slope_d2, intercept_d2, r_value_d2, _, _ = \
					scipy.stats.linregress(d2_d[element].values, d2_d["response"].values)
				r2 = r_value ** 2
				r2_d1 = r_value_d1 ** 2
				r2_d2 = r_value_d2 ** 2
			else:
				cohen_f2, r2 = None, None
		else:
			cohen_f2, r2 = None, None

		if comb_in_responses is not None and d1_in_responses is not None and d2_in_responses is not None \
				and len(comb_in_responses) >= min_limitation and len(d1_in_responses) >= min_limitation \
				and len(d2_in_responses) >= min_limitation:

			_, kruskal_pval = stats.kruskal(d1_in_responses.response.values, d2_in_responses.response.values,
											comb_in_responses.response.values)
			_, mann_pval1 = stats.mannwhitneyu(d1_in_responses.response.values, comb_in_responses.response.values,
											   alternative="greater")
			_, mann_pval2 = stats.mannwhitneyu(d2_in_responses.response.values, comb_in_responses.response.values,
											   alternative="greater")
			_, mann_pval1_out = stats.mannwhitneyu(d1_out_responses.response.values, d1_in_responses.response.values,
												   alternative="greater")
			_, mann_pval2_out = stats.mannwhitneyu(d2_out_responses.response.values, d2_in_responses.response.values,
												   alternative="greater")
			_, mann_comb_out = stats.mannwhitneyu(comb_out_responses.response.values, comb_in_responses.response.values,
												  alternative="greater")

			cohen_d = get_effect_size(w_biomarker=comb_in_responses, wout_biomarker=comb_out_responses,
									  effect_function="cohens_d")

			glass_d = get_effect_size(w_biomarker=comb_in_responses, wout_biomarker=comb_out_responses,
									  effect_function="glass_d")

			df_in = pandas.concat([d1_in_response, d2_in_response, comb_in_response], axis=1)
			melt_df_in = df_in.melt(value_vars=df_in.columns)
			melt_df_in.columns = ["Treatment", estimate_lr]
			melt_df_in["Feature"] = "Yes"

			df_out = pandas.concat([d1_out_response, d2_out_response, comb_out_response], axis=1)
			melt_df_out = df_out.melt(value_vars=df_out.columns)
			melt_df_out.columns = ["Treatment", estimate_lr]
			melt_df_out["Feature"] = "No"

			melt_df = pandas.concat([melt_df_in, melt_df_out], ignore_index=True)
			melt_df.to_csv(
				output_path + "biomarker/LR/association_controls/mono_combo%s_%s_%s_%s_comparison_on_%s_%s_%s_w_%s%s%s.csv"
				% (t_title, drug1, drug2, estimate_lr, element, feature, feature_level, selection_criteria, msi_title,
				   random_text), index=False)
			melt_df = pandas.read_csv(
				output_path + "biomarker/LR/association_controls/mono_combo%s_%s_%s_%s_comparison_on_%s_%s_%s_w_%s%s%s.csv"
				% (t_title, drug1, drug2, estimate_lr, element, feature, feature_level, selection_criteria, msi_title,
				   random_text))

			if plotting:
				if feature in ["mutation", "amplification", "deletion", "hypermethylation", "msi", "gain", "loss",
							   "clinicalsubtype"]:
					fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(6, 8))

					plt.suptitle("Mono - Combinatorial Therapy Effects on %s \n%s on $%s$ | %s"
								 % (estimate_title, feature_title, element, drug_comb), fontsize=12)

					order = [drug1, drug2, drug_comb]

					sns.stripplot(x="Treatment", y=estimate_lr, data=melt_df, hue="Feature",
								  palette={"No": "#C5C5C5",
										   "Yes": (0.0, 0.0, 1.0)},
								  ax=axis[0, 0], alpha=0.4, dodge=True, order=order)

					sns.boxplot(x="Treatment", y=estimate_lr, data=melt_df, ax=axis[0, 0], hue="Feature",
								flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
								whiskerprops=whiskerprops, order=order)
					axis[0, 0].legend([], [], frameon=False)
					axis[0, 0].set_title(
						"%s : Kruskal-Wallis P-value = {:.2e}\nCohen's d = %.3f | Glass Δ = %.3f\n\n# Models with Biomarker\n%s = %d, %s = %d, %s = %d".format(
							kruskal_pval)
						% (comb_type, cohen_d[0], glass_d[0], drug1, len(d1_in_responses), drug2, len(d2_in_responses),
						   drug_comb,
						   len(comb_in_responses)),
						fontsize=10)
					axis[0, 0].set_xlabel('')
					axis[0, 0].tick_params(axis="x", which="major", labelsize=5)
					axis[0, 0].set_ylabel(estimate_title)

					pvalues = [mann_pval1_out, mann_pval2_out, mann_comb_out, mann_pval1, mann_pval2]
					add_stat_annotation(axis[0, 0], data=melt_df, x="Treatment", y=estimate_lr, order=order,
										hue="Feature",
										box_pairs=[((drug1, "Yes"), (drug1, "No")), ((drug2, "Yes"), (drug2, "No")),
												   ((drug_comb, "Yes"), (drug_comb, "No")),
												   ((drug1, "Yes"), (drug_comb, "Yes")),
												   ((drug2, "Yes"), (drug_comb, "Yes"))],
										perform_stat_test=False, pvalues=pvalues, loc='inside', verbose=2,
										text_format="full")

				else:
					fig, axis = plt.subplots(1, 2, squeeze=False, figsize=(12, 8))

					plt.suptitle("Mono - Combinatorial Therapy Effects on %s \n%s $%s$ | %s"
								 % (estimate_title, feature_title, element, drug_comb), fontsize=12)

					order = [drug1, drug2, drug_comb]

					sns.stripplot(x="Treatment", y=estimate_lr, data=melt_df, hue="Feature",
								  palette={"No": "#C5C5C5",
										   "Yes": (0.0, 0.0, 1.0)},
								  ax=axis[0, 0], alpha=0.4, dodge=True, order=order)

					sns.boxplot(x="Treatment", y=estimate_lr, data=melt_df, ax=axis[0, 0], hue="Feature",
								flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
								whiskerprops=whiskerprops, order=order)
					axis[0, 0].legend([], [], frameon=False)
					axis[0, 0].set_title(
						"%s : Kruskal-Wallis P-value = {:.2e}\nCohen's d = %.3f | Glass Δ = %.3f\n\n# Models with Biomarker\n%s = %d, %s = %d, %s = %d".format(
							kruskal_pval)
						% (comb_type, cohen_d[0], glass_d[0], drug1, len(d1_in_responses), drug2, len(d2_in_responses),
						   drug_comb,
						   len(comb_in_responses)),
						fontsize=10)
					axis[0, 0].set_xlabel('')
					axis[0, 0].tick_params(axis="x", which="major", labelsize=10)
					axis[0, 0].set_ylabel(estimate_title)

					pvalues = [mann_pval1_out, mann_pval2_out, mann_comb_out, mann_pval1, mann_pval2]
					add_stat_annotation(axis[0, 0], data=melt_df, x="Treatment", y=estimate_lr, order=order,
										hue="Feature",
										box_pairs=[((drug1, "Yes"), (drug1, "No")), ((drug2, "Yes"), (drug2, "No")),
												   ((drug_comb, "Yes"), (drug_comb, "No")),
												   ((drug1, "Yes"), (drug_comb, "Yes")),
												   ((drug2, "Yes"), (drug_comb, "Yes"))],
										perform_stat_test=False, pvalues=pvalues, loc='inside', verbose=2,
										text_format="full")

					axis[0, 1].plot(comb_d[element].values, comb_d["response"].values, 'o', alpha=0.5, markersize=3,
									linewidth=0, color="navy", label="Combination")
					axis[0, 1].plot(d1_d[element].values, d1_d["response"].values, 'o', alpha=0.35, markersize=3,
									linewidth=0, color="cadetblue", label="Drug 1")
					axis[0, 1].plot(d2_d[element].values, d2_d["response"].values, 'o', alpha=0.35, markersize=3,
									linewidth=0, color="firebrick", label="Drug2")
					axis[0, 1].plot(comb_d[element].values, intercept + (slope * comb_d[element].values),
									'navy', label="Combination Fitted Line")
					axis[0, 1].plot(d1_d[element].values, intercept_d1 + (slope_d1 * d1_d[element].values),
									'cadetblue', label="Drug 1 Fitted Line")
					axis[0, 1].plot(d2_d[element].values, intercept_d2 + (slope_d2 * d2_d[element].values),
									'firebrick', label="Drug 2 Fitted Line")
					max_y = max(
						[max(comb_d["response"].values), max(d1_d["response"].values), max(d2_d["response"].values)])

					axis[0, 1].vlines(x=q4[0.4], ymin=0, ymax=max_y, colors='darkgrey', alpha=0.5,
									  ls='--', lw=.75)
					axis[0, 1].vlines(x=q4[0.6], ymin=0, ymax=max_y, colors='darkgrey', alpha=0.5,
									  ls='--', lw=.75)
					axis[0, 1].legend()
					axis[0, 1].set_xlabel(feature_title)
					axis[0, 1].set_ylabel(estimate_title)
					axis[0, 1].set_title(
						"%s : Cohen's f2 = %.3f\n\nR\u00b2 = %.3f | R\u00b2 Drug 1 = %.3f | R\u00b2 Drug 2 = %.3f\n# D1 response = %d | # D2 response = %d | # Combination response = %d"
						% (comb_type, cohen_f2, r2, r2_d1, r2_d2,
						   len(d1_d[element].values), len(d2_d[element].values),
						   len(comb_d[element].values)), fontsize=10)
				plt.xticks(fontsize=10)
				plt.yticks(fontsize=10)
				plt.tight_layout()
				if os.path.isdir(output_path + "biomarker/figures/association_controls/%s/" % comb_title) is False:
					os.system("mkdir %s/biomarker/figures/association_controls/%s/" % (output_path, comb_title))
				plt.savefig(
					output_path + "biomarker/figures/association_controls/%s/mono_combo%s_%s_%s_%s_%s_%s_%s_with_%s%s%s_boxplots.pdf"
					% (comb_title, t_title, drug1, drug2, estimate_lr, element, feature, feature_level,
					   selection_criteria, msi_title, random_text),
					dpi=300)
				plt.savefig(
					output_path + "biomarker/figures/association_controls/%s/mono_combo%s_%s_%s_%s_%s_%s_%s_with_%s%s%s_boxplots.jpg"
					% (comb_title, t_title, drug1, drug2, estimate_lr, element, feature, feature_level,
					   selection_criteria, msi_title, random_text),
					dpi=300)
				plt.close()

			return {"df": melt_df, "KW": kruskal_pval, "MW_D1": mann_pval1_out, "MW_D2": mann_pval2_out,
					"MW_C": mann_comb_out, "MW_D1C": mann_pval1, "MW_D2C": mann_pval2,
					"cohens_d": cohen_d[0], "glass_d": glass_d[0], "cohens_f2": cohen_f2, "R2": r2}
		else:
			return None
	else:
		return None


def write_response(response):
	if response is None or pandas.isna(response):
		return "None"
	else:
		return str(response)


def collect_robust_biomarkers(tissue, estimate_lr, fdr_limit, merged, msi_flag, random):
	"""
	Annotate the biomarker with their statistics
	:param tissue: Which tissue the data will be separated into
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:param fdr_limit: FDR limit
	:param merged: If everything is together (all tissues including pancancer or not)
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param random: Boolean if random effect will be added or not
	:return:
	"""

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	else:
		t_title = ""

	if msi_flag:
		msi_title = "_msi"
	else:
		msi_title = ""

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if merged:
		if "annotated_merged_LR%s_%s_FDR_%d%s%s.csv" % (
		t_title, estimate_lr, fdr_limit, msi_title, random_text) not in os.listdir(
				output_path + "biomarker/LR/annotated_combination_effect/"):
			effect_dfs = list()
			for f in os.listdir(output_path + "biomarker/LR/annotated_combination_effect/"):
				df = pandas.read_csv(output_path + "biomarker/LR/annotated_combination_effect/" + f)
				tissue_text = "_".join("_".join(f.split("_")[3:]).split("_FDR")[0])
				df["tissue"] = tissue_text
				effect_dfs.append(df)
			effect_df = pandas.concat(effect_dfs)
			effect_df.to_csv(
				output_path + "biomarker/LR/annotated_combination_effect/annotated_merged_LR%s_%s_FDR_%d%s%s.csv"
				% (t_title, estimate_lr, fdr_limit, msi_title, random_text), index=False)
		else:
			effect_df = pandas.read_csv(
				output_path + "biomarker/LR/annotated_combination_effect/annotated_merged_LR%s_%s_FDR_%d%s%s.csv"
				% (t_title, estimate_lr, fdr_limit, msi_title, random_text))

	else:
		if "library_merged_LR%s_%s_FDR_%d%s%s.csv" % (
				t_title, estimate_lr, fdr_limit, msi_title, random_text) not in os.listdir(
			output_path + "biomarker/LR/merged_LR/"):
			library_associations = take_mono_associations(tissue=tissue, fdr_limit=fdr_limit,
														  merged=merged, msi_flag=msi_flag, random=random)
			library_associations = library_associations[library_associations["Estimate_LR"] == estimate_lr]

			combination_association = take_combination_associations(tissue=tissue, fdr_limit=fdr_limit,
																	merged=merged, msi_flag=msi_flag, random=random)
			combination_association = combination_association[combination_association["Estimate_LR"] == estimate_lr]

			s_effect, r_effect = "sensitivity", "resistance"
			combination_association["Drug1"] = combination_association.apply(lambda x: x.DrugComb.split("/")[0], axis=1)
			combination_association["Drug2"] = combination_association.apply(lambda x: x.DrugComb.split("/")[1], axis=1)
			combination_effect_df = combination_association.copy()
			combination_effect_df["Drug1_effect"] = None
			combination_effect_df["Drug2_effect"] = None
			combination_effect_df["Drug1_FDR"] = None
			combination_effect_df["Drug2_FDR"] = None

			indices = pandas.Index(combination_effect_df.index)
			if indices.is_unique is False:
				combination_effect_df.index = pandas.RangeIndex(start=0, stop=len(indices), step=1)

			i, t = 0, len(combination_effect_df.index)
			for feature_group, feature_df in combination_effect_df.groupby(["Feature", "Feature_level"]):
				lib_feature_df = library_associations[(library_associations.Feature == feature_group[0]) &
													  (library_associations.Feature_level == feature_group[1])]
				for ind, row in feature_df.iterrows():
					if row.Gene in lib_feature_df.Gene.unique():
						x = lib_feature_df[lib_feature_df.Gene == row.Gene]
						if row.Drug1 in x.library_name.unique():
							x1 = x[x.library_name == row.Drug1]
							combination_effect_df.loc[ind, "Drug1_FDR"] = x1.FDR.values[0]
							if x1.FDR.values[0] < fdr_limit:
								combination_effect_df.loc[ind, "Drug1_effect"] = "S" if x1.Association.values[
																							0] == s_effect else (
									"R" if x1.Association.values[0] == r_effect else None)
							else:
								combination_effect_df.loc[ind, "Drug1_effect"] = "N"

						if row.Drug2 in x.library_name.unique():
							x2 = x[x.library_name == row.Drug2]
							combination_effect_df.loc[ind, "Drug2_FDR"] = x2.FDR.values[0]
							if x2.FDR.values[0] < fdr_limit:
								combination_effect_df.loc[ind, "Drug2_effect"] = "S" if x2.Association.values[
																							0] == s_effect else (
									"R" if x2.Association.values[0] == r_effect else None)
							else:
								combination_effect_df.loc[ind, "Drug2_effect"] = "N"
					i += 1
					print(i * 100.0 / t)

			combination_effect_df["combo_effect"] = combination_effect_df.apply(
				lambda x: "S" if (x.Stage == "combo") and (x.FDR < fdr_limit) and (x.Association == s_effect) else (
					"R" if (x.Stage == "combo") and (x.FDR < fdr_limit) and (x.Association == r_effect) else (
						"N" if (x.Stage == "combo") and (x.FDR >= fdr_limit) else None)), axis=1)

			combination_effect_df["delta_effect"] = combination_effect_df.apply(
				lambda x: "SYN" if (x.Stage == "delta") and (x.FDR < fdr_limit) and (x.Association == "synergy") else (
					"ANT" if (x.Stage == "delta") and (x.FDR < fdr_limit) and (x.Association == "antagonism") else (
						"N" if (x.Stage == "delta") and (x.FDR >= fdr_limit) else None)), axis=1)

			combination_effect_df.to_csv(output_path + "biomarker/LR/merged_LR/library_merged_LR%s_%s_FDR_%d%s%s.csv"
										 % (t_title, estimate_lr, fdr_limit, msi_title, random_text), index=False)
		else:
			combination_effect_df = pandas.read_csv(
				output_path + "biomarker/LR/merged_LR/library_merged_LR%s_%s_FDR_%d%s%s.csv"
				% (t_title, estimate_lr, fdr_limit, msi_title, random_text))

		if "annotated_merged_LR%s_%s_FDR_%d%s%s.csv" % (
		t_title, estimate_lr, fdr_limit, msi_title, random_text) not in os.listdir(
				output_path + "biomarker/LR/annotated_combination_effect/"):
			# Labelling
			combination_effect_df = combination_effect_df[combination_effect_df.Estimate_LR == estimate_lr]
			delta_df = combination_effect_df[combination_effect_df.Stage == "delta"]
			combo_df = combination_effect_df[combination_effect_df.Stage == "combo"]

			effect_df = pandas.merge(combo_df, delta_df, how="outer", validate='m:m', indicator=True,
									 on=["Gene", "DrugComb", "Feature", "Feature_level"])

			effect_df = effect_df[
				["Gene", "DrugComb", "Drug1_x", "Drug2_x", "Feature", "Feature_level", "Selection_Criteria_x",
				 "Association_x", "Association_name_x", "Association_tag_x", "Coefficient_x", "Pvalue_x", "# Sample_x",
				 "Adj_p_x", "FDR_x", "Effect_x", "gene_target_1_x", "gene_target_2_x", "pathway_target_1_x",
				 "pathway_target_2_x",
				 "drug_type_1_x", "drug_type_2_x", "Stage_x", "Estimate_LR_x", "Significance_x", "Group_x",
				 "Drug1_effect_x", "Drug2_effect_x", "Drug1_FDR_x", "Drug2_FDR_x", "combo_effect_x", "Coefficient_y",
				 "Pvalue_y", "Adj_p_y", "FDR_y", "Effect_y", "Stage_y", "Estimate_LR_y", "Significance_y",
				 "Group_y", "delta_effect_y"]]
			effect_df.columns = ["Gene", "DrugComb", "Drug1", "Drug2", "Feature", "Feature_level", "Selection_Criteria",
								 "Association", "Association_name", "Association_tag", "Coefficient_Combo",
								 "Pvalue_Combo", "# Sample", "Adj_p_Combo", "FDR_Combo", "Effect_Combo",
								 "Gene_target_1", "Gene_target_2", "Pathway_target_1", "Pathway_target_2",
								 "Drug_type_1", "Drug_type_2", "Stage_Combo", "Estimate_Combo", "Significance_Combo",
								 "Group_Combo", "Drug1_effect", "Drug2_effect", "Drug1_FDR", "Drug2_FDR",
								 "Response_Combo", "Coefficient_Delta", "Pvalue_Delta", "Adj_p_Delta", "FDR_Delta",
								 "Effect_Delta", "Stage_Delta", "Estimate_Delta", "Significance_Delta",
								 "Group_Delta", "Response_Delta"]

			effect_df["Biomarker"] = effect_df.apply(
				lambda x: x.Association_tag.split(" : ")[1].split(">")[0] + "-" + x.Association_tag.split(" : ")[0]
				if x.Association_tag is not None and type(x.Association_tag) != float else None, axis=1)

			effect_df["Response_Combined"] = effect_df.apply(
				lambda x: write_response(x.Drug1_effect) + "-" + write_response(x.Drug2_effect) + "-" +
						  write_response(x.Response_Combo) + "-" + write_response(x.Response_Delta), axis=1)

			effect_df["KW_pval"] = None
			effect_df["MW_D1C"] = None
			effect_df["MW_D2C"] = None
			effect_df["cohens_d"] = None
			effect_df["glass_d"] = None
			effect_df["cohens_f2"] = None
			effect_df["R2"] = None
			effect_df["sensitive"] = None
			effect_df["synergistic"] = None
			effect_df["robust"] = None
			i, t = 0, len(effect_df.Association_tag)
			effect_df2 = effect_df.copy()
			effect_df2 = effect_df2[pandas.isna(effect_df2.robust)]
			indices = pandas.Index(effect_df.index)
			if indices.is_unique is False:
				effect_df.index = pandas.RangeIndex(start=0, stop=len(indices), step=1)

			for ind, row in effect_df.iterrows():
				if ind in effect_df2.index:
					if row.Response_Combo == "S":
						print(row.Feature)
						print(row.Feature_level)
						d = test_combo_library_combination(
							drug_comb=row.DrugComb, comb_type=row["Response_Combined"], estimate_lr=estimate_lr,
							feature=row.Feature, feature_level=row.Feature_level,
							element=row.Gene, selection_criteria=row.Selection_Criteria,
							plotting=True, tissue=tissue, min_limitation=3, msi_flag=msi_flag, random=random)

						if d is not None:
							effect_df.loc[ind, "KW_pval"] = d["KW"]
							effect_df.loc[ind, "MW_D1C"] = d["MW_D1C"]
							effect_df.loc[ind, "MW_D2C"] = d["MW_D2C"]
							effect_df.loc[ind, "cohens_d"] = d["cohens_d"]
							effect_df.loc[ind, "glass_d"] = d["glass_d"]
							effect_df.loc[ind, "cohens_f2"] = d["cohens_f2"]
							effect_df.loc[ind, "R2"] = d["R2"]
							effect_df.loc[ind, "sensitive"] = True if row.Response_Combo == "S" else False
							effect_df.loc[ind, "synergistic"] = True if row.Response_Delta == "SYN" else False
							effect_df.loc[ind, "robust"] = True if row.Response_Combo == "S" and \
																   d["MW_D1C"] < 0.05 and d["MW_D2C"] < 0.05 and \
																   d["MW_C"] < 0.05 else False
						else:
							effect_df.loc[ind, "sensitive"] = True if row.Response_Combo == "S" else False
							effect_df.loc[ind, "synergistic"] = True if row.Response_Delta == "SYN" else False
							effect_df.loc[ind, "robust"] = False
					else:
						effect_df.loc[ind, "sensitive"] = False
						effect_df.loc[ind, "synergistic"] = True if row.Response_Delta == "SYN" else False
						effect_df.loc[ind, "robust"] = False

					i += 1
					print(i * 100.0 / t)

			effect_df.to_csv(
				output_path + "biomarker/LR/annotated_combination_effect/annotated_merged_LR%s_%s_FDR_%d%s%s.csv"
				% (t_title, estimate_lr, fdr_limit, msi_title, random_text), index=False)
		else:
			effect_df = pandas.read_csv(
				output_path + "biomarker/LR/annotated_combination_effect/annotated_merged_LR%s_%s_FDR_%d%s%s.csv"
				% (t_title, estimate_lr, fdr_limit, msi_title, random_text))

	return effect_df


def main(args):
	if args["RUN_FOR"] == "collection":
		_ = collect_biomarker(tissue=None, merged=True, msi_flag=False, random=False)

	elif args["RUN_FOR"] == "robust":
		_ = collect_robust_biomarkers(tissue=args["TISSUE"], estimate_lr="XMID", fdr_limit=10, merged=False,
									  msi_flag=False, random=False)
	return True


if __name__ == '__main__':
	args = take_input()
	print(args)

	_ = main(args)

