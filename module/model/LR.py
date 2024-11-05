"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 19 April 2022
Last Update : 22 March 2024
Input : Biomarker Analysis
Output : Biomarkers - Association Object
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, scipy, sklearn, sys, matplotlib, requests, networkx, re, argparse
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler
import limix
from limix.qtl import scan
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import PathPatch
from sklearn.metrics import r2_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from CombDrug.module.path import output_path
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.responses import *
from CombDrug.module.data.omics import *


# ---------------------------------------------------------------------------#
#                               	Inputs                                   #
# ---------------------------------------------------------------------------#

def take_input():
	parser = argparse.ArgumentParser(prog="CombDrug Linear Regression",
									 usage="%(prog)s [inputs]",
									 description="""
                                     **********************************
                                     		   Find biomarkers
                                     **********************************""")

	for group in parser._action_groups:
		if group.title == "optional arguments":
			group.title = "Inputs"
		elif "positional arguments":
			group.title = "Mandatory Inputs"

	# Feature
	parser.add_argument("-feature", dest="FEATURE", required=True)
	parser.add_argument("-feature_level", dest="LEVEL", required=True)
	parser.add_argument("-selection_criteria", dest="CRITERIA", required=True)
	parser.add_argument("-plot", dest="PLOTTING", action="store_true")
	parser.add_argument("-stage", dest="STAGE", required=True)
	parser.add_argument("-estimate_lr", dest="EST_LR", required=True)
	parser.add_argument("-tissue", dest="TISSUE", default=None)
	parser.add_argument("-min_cl", dest="MIN_CL", required=True)
	parser.add_argument("-msi_cov", dest="MSI_COV", action="store_true")
	parser.add_argument("-random", dest="RANDOM", action="store_true")
	parser.add_argument("-fdr", dest="FDR", required=True)
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                               Response Matrix                              #
# ---------------------------------------------------------------------------#


def get_LR_response(drug_info, response_df, drug_col):
	"""
	Create response matrix for LR models
	:param drug_info: Drug Combination or library drug
	:param response_df: Whole response data frame
	:param drug_col: Name of the drug column
	:return: response matrix
	"""
	response = response_df.copy()
	#response = response.set_index("SIDM")
	response = response[response[drug_col] == drug_info]
	response = response[["response"]]
	response = response.drop_duplicates()
	return response


# ---------------------------------------------------------------------------#
#                              Covariate Matrix                              #
# ---------------------------------------------------------------------------#

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

	if stage == "mono": drug_col = "library_name"
	else: drug_col = "DrugComb"

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

# ---------------------------------------------------------------------------#
#                              Feature Matrix                                #
# ---------------------------------------------------------------------------#

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


# ---------------------------------------------------------------------------#
#                           Random Effect Matrix                             #
# ---------------------------------------------------------------------------#

def create_raneff_matrix(combi_df, drug_col, response_df):

	response_df = response_df.reset_index()
	# Exp type
	screen = combi_df[["SIDM", drug_col, "RESEARCH_PROJECT"]]
	screen = screen[screen.SIDM.isin(response_df.SIDM.unique())]
	screen_k = 0
	screens = screen["RESEARCH_PROJECT"].unique()
	for s in screens:
		screen = screen.replace(s, screen_k)
		screen_k += 1
	screen = screen.drop_duplicates()
	screen.columns = ["SIDM", drug_col, "screen"]
	return screen


def get_LR_raneff(drug_info, drug_col, matrix):
	"""
	Get screen names of the perturbation as random effect
	:param matrix: The random effect matrix
	:param drug_info: Drug Combination or library drug
	:param drug_col: Name of the drug column
	:return: Linear regression input random effect
	"""
	matrix_d = matrix.copy()
	print(matrix_d)
	matrix_d = matrix_d[matrix_d[drug_col] == drug_info]
	matrix_d = matrix_d[["SIDM", "screen"]]
	matrix_d = matrix_d.drop_duplicates()
	matrix_d = matrix_d.set_index(["SIDM"])
	return matrix_d


# ---------------------------------------------------------------------------#
#                                    LR		                                 #
# ---------------------------------------------------------------------------#


def get_lmm(response, feature, covariate, random):
	"""
	Model the linear mixed effect model (limix package)
	:param response: Response matrix from get_LR_response()
	:param feature: Feature matrix from get_LR_feature()
	:param covariate: Covariate matrix from get_LR_covariate()
	:param random: If there is random effects - screens as batches
	:return: LMM model (H0 and H1 created --> compared with likelihood ratio test
	"""
	# Linear Mixed-Effect Model from Limix package
	lmm = scan(G=feature, Y=response, K=random,
			   M=covariate, lik="normal", verbose=False)
	return lmm


def get_pvalue(lmm):
	"""
	Retrieving the Pvalue for significance of the difference between  Null and Alternative LMM models
	:param lmm: LMM model
	:return: Pvalue
	"""
	return lmm.stats.loc[0, "pv20"]


def get_coefficient(lmm, element):
	"""
	Retrieving the coefficient of the association between element and input response data
	:param lmm: LMM model
	:param element: The selected element for the feature (gene/ mutations/ segment etc.)
	:return: Coefficient value
	"""
	coef_df = lmm.effsizes["h2"]
	return coef_df[coef_df.effect_name == element]["effsize"].values[0]


def get_effect(stage, coeff):
	"""
	Labelling the effect of the feature on the response
	:param stage: LIBRARY / COMBO / DELTA
	:param coeff: Coefficient value
	:return: Effect labels as resistance or sensitivity
	"""
	if stage in ["mono", "combo"]:
		if coeff > 0: return "resistance"
		elif coeff < 0: return "sensitivity"
		else: return "No effect"
	elif stage == "delta":
		if coeff > 0: return "synergy"
		if coeff < 0: return "antagonism"
		else: return "No effect"
	else: return None


def check_dimension(drug_name, stage, feature, level, element, data_type, covariate_matrix, feature_matrix, response_matrix,
					random_matrix, random, min_cl_criteria, selection_criteria, t_title, title, msi_title, d_title):
	"""
	Checking the dimensions of the input matrices after removing the None values
	:param drug_name: Name of drug or drug comnbination in matrices
	:param stage: mono/combo/delta
	:param feature: The feature that will be test for association with drug data
	:param level: Level of feature selection
	:param element: Gene or mutation name
	:param data_type: Type of the data binary/continuous
	:param covariate_matrix: Covariate matrix
	:param feature_matrix: Feature matrix
	:param response_matrix: Response matrix
	:param random_matrix: Random effect matrix
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param t_title: Information on subset of cell lines as tissue/pancancer
	:param title: Information on feature selection
	:param msi_title: if msi used as covariate or not
	:param d_title: drug name in title
	:param random: Boolean - if random effect will be added or not
	:return: Dictionary having all input matrices
	"""

	if t_title == "":
		t_title = "_pancancer"

	level_text = "_".join(level.split(" "))

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if stage == "mono": drug_col = "library_name"
	else: drug_col = "DrugComb"

	try:
		if not os.path.isdir(output_path + "biomarker/LR/input_LR/%s/" % stage):
			os.mkdir(output_path + "biomarker/LR/input_LR/%s/" % stage)
	except FileExistsError: pass

	try:
		if not os.path.isdir(output_path + "biomarker/LR/input_LR/%s/%s/" % (stage, t_title[1:])):
			os.mkdir(output_path + "biomarker/LR/input_LR/%s/%s/" % (stage, t_title[1:]))
	except FileExistsError: pass

	try:
		if not os.path.isdir(output_path + "biomarker/LR/input_LR/%s/%s/%s/" % (stage, t_title[1:], d_title)):
			os.mkdir(output_path + "biomarker/LR/input_LR/%s/%s/%s/" % (stage, t_title[1:], d_title))
	except FileExistsError: pass

	if "input_%s_%s_%s_%s_%s%s_%s%s%s.csv" % (stage, feature, level_text, element, d_title, t_title, title, msi_title, random_text) not in \
			os.listdir(output_path + "biomarker/LR/input_LR/%s/%s/%s/" %(stage, t_title[1:], d_title)):

		feature_check = False

		feature_matrix = feature_matrix.dropna()
		feature_matrix2 = feature_matrix.reset_index()
		feature_matrix2 = feature_matrix2.drop_duplicates()

		if "SIDM" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("SIDM")
		elif "sanger_id" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("sanger_id")
		elif "index" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("index")
		elif "0" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("0")
		elif 0 in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index(0)
		elif "uniprot_id" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("uniprot_id")
		else:
			feature_matrix = feature_matrix2.set_index("Unnamed: 1")
		fet_models = feature_matrix.index.unique()

		response_matrix = response_matrix.reset_index()
		response_matrix = response_matrix.drop_duplicates()
		response_matrix = response_matrix.dropna()
		response_matrix = response_matrix.set_index(["SIDM"])
		res_models = response_matrix.index.unique()

		covariate_matrix = covariate_matrix[covariate_matrix[drug_col] == drug_name].reset_index()
		covariate_matrix = covariate_matrix.drop([drug_col], axis=1)
		covariate_matrix = covariate_matrix.dropna()
		covariate_matrix = covariate_matrix.drop_duplicates()
		covariate_matrix = covariate_matrix.set_index(["SIDM"])
		cov_models = covariate_matrix.index.unique()

		if random:
			random_matrix = random_matrix.dropna()
			random_matrix = random_matrix.reset_index()
			random_matrix = random_matrix.drop_duplicates()
			random_matrix = random_matrix.set_index(["SIDM"])
			ran_models = random_matrix.index.unique()

			intersected_models = set(cov_models).intersection(set(fet_models).intersection(
				set(res_models).intersection(set(ran_models))))
		else:

			intersected_models = set(cov_models).intersection(set(fet_models).intersection(set(res_models)))

		if len(intersected_models) >= int(min_cl_criteria):

			feature_matrix = feature_matrix.loc[intersected_models]
			# Check again if the feature still satisfies selection criteria
			f_col = feature_matrix.columns[0]
			if data_type == "continuous":
				if len(set(feature_matrix[f_col].values)) > 1:
					feature_check = True
			else:
				if len(feature_matrix[f_col].value_counts(dropna=True).index) == 2:
					if feature_matrix[f_col].value_counts(dropna=True)[1] >= int(selection_criteria) and \
							feature_matrix[f_col].value_counts(dropna=True)[0] >= int(selection_criteria):
						feature_check = True

			if feature_check:
				feature_matrix = feature_matrix.sort_index()

				covariate_matrix = covariate_matrix.loc[intersected_models]
				covariate_matrix = covariate_matrix.reset_index()
				covariate_matrix = covariate_matrix.drop_duplicates()
				covariate_matrix = covariate_matrix.set_index(["SIDM"])
				covariate_matrix = covariate_matrix.sort_index()

				response_matrix = response_matrix.loc[intersected_models]
				response_matrix = response_matrix.reset_index()
				response_matrix = response_matrix.drop_duplicates()
				for g, g_df in response_matrix.groupby(["SIDM"]):
					if len(g_df.index) > 1:
						print(g)
				response_matrix = response_matrix.set_index(["SIDM"])
				response_matrix =response_matrix.sort_index()

				if random:
					random_matrix = random_matrix.loc[intersected_models]
					random_matrix = random_matrix.sort_index()

					dim_df = pandas.concat([response_matrix, covariate_matrix, feature_matrix, random_matrix], axis=1)

				else:
					print("Response shape")
					print(response_matrix.shape)
					print(len(list(response_matrix.index)))
					print(len(set(list(response_matrix.index))))

					print("Covariate shape")
					print(covariate_matrix.shape)

					print("Feature shape")
					print(feature_matrix.shape)
					dim_df = pandas.concat([response_matrix, covariate_matrix, feature_matrix], axis=1)

				if len(dim_df.index) == 0:
					dim_df = pandas.DataFrame()

			else: dim_df = pandas.DataFrame()
		else: dim_df = pandas.DataFrame()

		dim_df.to_csv(output_path + "biomarker/LR/input_LR/%s/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
					  % (stage, t_title[1:], d_title, stage, feature, level_text, element, d_title, t_title,
						 title, msi_title, random_text), index=True)

	else:
		dim_df = pandas.read_csv(output_path + "biomarker/LR/input_LR/%s/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
								 % (stage, t_title[1:], d_title, stage, feature, level_text, element, d_title, t_title,
									title, msi_title, random_text), index_col=0)

	if len(dim_df.index) == 0:
		dim_df = None

	return dim_df


def get_stage_matrices(element, drug_info, stage, feature, level, tissue, estimate_lr):

	if stage != "mono":
		drug_col = "DrugComb"
		combi_df = combine_combi(estimate_data="XMID", treatment="combination")
	else:
		drug_col = "library_name"
		combi_df = combine_combi(estimate_data="XMID", treatment="mono")

	whole_response_df = get_response_data(tissue=tissue, stage=stage, estimate_lr=estimate_lr)
	whole_feature_matrix = create_feature_matrix(feature=feature, level=level, selection_criteria=selection_criteria,
												 tissue=tissue, min_cl_criteria = min_cl_criteria)
	response_matrix = get_LR_response(drug_info=drug_info, response_df=whole_response_df, drug_col=drug_col)
	covariate_matrix = get_LR_covariate(whole_combi=combi_df, stage=stage, tissue=tissue, msi_flag=False)
	feature_matrix = get_LR_feature(element=element, matrix=whole_feature_matrix, scale=scale)

	df = pandas.concat([response_matrix, covariate_matrix, feature_matrix], axis=1)

	return df


def compare_mono_combo_availability(element, drug_info, feature, level, tissue, estimate_lr,
									min_cl_criteria, selection_criteria):

	if feature == "transcription": title, scale, data_type = "log2_filtered_" + str(selection_criteria), True, "continuous"
	elif feature == "proteomics": title, scale, data_type = "averaged_" + str(selection_criteria), True, "continuous"
	#elif feature == "cnv": title, scale, data_type = "pureCN_" + str(selection_criteria), True, "continuous"
	#elif feature == "msi": title, level, scale, data_type = "positive_" + str(selection_criteria), "level", False, "binary"
	else: title, scale, data_type = str(selection_criteria), False, "binary"

	all_cls = list()
	cls = dict()
	for stage in ["mono", "combo", "delta"]:
		if stage != "mono":
			drug_col = "DrugComb"
			combi_df = combine_combi(estimate_data="XMID", treatment="combination")
		else:
			drug_col = "library_name"
			combi_df = combine_combi(estimate_data="XMID", treatment="mono")

		whole_response_df = get_response_data(tissue=tissue, stage=stage, estimate_lr=estimate_lr)
		whole_feature_matrix = create_feature_matrix(feature=feature, level=level, selection_criteria=selection_criteria,
													 tissue=tissue, min_cl_criteria = min_cl_criteria)

		response_matrix = get_LR_response(drug_info=drug_info, response_df=whole_response_df, drug_col=drug_col)
		covariate_matrix = get_LR_covariate(whole_combi=combi_df, stage=stage, tissue=tissue, msi_flag=False)
		feature_matrix = get_LR_feature(element=element, matrix=whole_feature_matrix, scale=scale)

		all_cls.extend(response_matrix.index)

		covariate_matrix = covariate_matrix.drop_duplicates()
		covariate_matrix = covariate_matrix.dropna()
		cov_models = covariate_matrix.index.unique()

		feature_matrix = feature_matrix.dropna()
		feature_matrix2 = feature_matrix.reset_index()
		feature_matrix2 = feature_matrix2.drop_duplicates()

		if "SIDM" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("SIDM")
		elif "sanger_id" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("sanger_id")
		elif "index" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("index")
		elif "0" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("0")
		elif 0 in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index(0)
		elif "uniprot_id" in feature_matrix2.columns:
			feature_matrix = feature_matrix2.set_index("uniprot_id")
		else:
			feature_matrix = feature_matrix2.set_index("Unnamed: 1")
		fet_models = feature_matrix.index.unique()

		response_matrix = response_matrix.drop_duplicates()
		response_matrix = response_matrix.dropna()
		res_models = response_matrix.index.unique()

		intersected_models = set(cov_models).intersection(set(fet_models).intersection(set(res_models)))
		feature_matrix = feature_matrix.loc[intersected_models]
		feature_matrix = feature_matrix.sort_index()
		covariate_matrix = covariate_matrix.loc[intersected_models]
		covariate_matrix = covariate_matrix.drop_duplicates()
		covariate_matrix = covariate_matrix.sort_index()

		response_matrix = response_matrix.loc[intersected_models]
		response_matrix = response_matrix.sort_index()

		cl = list(pandas.concat([response_matrix, covariate_matrix, feature_matrix], axis=1).index)
		cls[stage] = cl

	df = pandas.DataFrame(0, columns=["mono", "combo", "delta"], index = list(set(all_cls)))
	for stage in ["mono", "combo", "delta"]:
		df.loc[cls[stage], stage] = 1

	return df


def annotate_drug_target(drug_name, annotation_type):
	"""
	Annotate the type, gene and pathway targets of the drugs
	:param drug_name: Name of the drug
	:param annotation_type: gene_target / pathway_target / drug_type
	:return:
	"""

	if annotation_type == "gene_target":
		if Drug(drug_name).targets is not None:
			return ",".join(Drug(drug_name).targets)
		else: return None
	elif annotation_type == "pathway_target":
		if Drug(drug_name).target_pathways is not None:
			return ",".join(Drug(drug_name).target_pathways)
		else:
			return None
	elif annotation_type == "drug_type":
		if Drug(drug_name).drug_type is not None:
			return ",".join(Drug(drug_name).drug_type)
		else:
			return None


def x_biomarker(feature, level, tissue, stage, estimate_lr, min_cl_criteria, selection_criteria,
				msi_flag, fdr_limit, random):
	"""
	Running biomarker analysis
	:param feature: The feature that will be test for association with drug data
	:param level: Level of feature selection
	:param selection_criteria: Binary feature = 3 / continuous feature = variance
	:param level: Level of feature selection (genes / mutations)
	:param tissue: Which tissue the data will be separated into
	:param stage: LIBRARY / COMBO / DELTA
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:param min_cl_criteria: The minimum number of cell lines having the feature value
	:param msi_flag: Boolean - If MSI status will be used as covariate or not
	:param fdr_limit: FDR limit
	:param random: Boolean if random effect will be added or not
	:return: Biomarker Data Frame
	"""

	if tissue is not None: t_title = "_" + "_".join(tissue.split(" "))
	else: t_title = ""

	if msi_flag: msi_title = "_msi"
	else: msi_title = ""

	level_text = "_".join(level.split(" "))

	fdr_limit = int(fdr_limit)

	if random:
		random_text = "_randomeffect"
	else:
		random_text = ""

	if feature == "transcription": title, scale, data_type = "log2_filtered_" + str(selection_criteria), True, "continuous"
	elif feature == "proteomics": title, scale, data_type = "averaged_" + str(selection_criteria), True, "continuous"
	else: title, scale, data_type = str(selection_criteria), False, "binary"

	if "LR_%s_%s_%s_%s%s_%s%s%s.csv" % (stage, estimate_lr, feature, level_text, t_title, title, msi_title, random_text) not in \
			os.listdir(output_path + "biomarker/LR/run_LR/"):

		if stage != "mono":
			drug_col = "DrugComb"
			combi_df = combine_combi(estimate_data="XMID", treatment="combination")
		else:
			drug_col = "library_name"
			combi_df = combine_combi(estimate_data="XMID", treatment="mono")

		whole_response_df = get_response_data(tissue=tissue, stage=stage, estimate_lr=estimate_lr)
		biomarker_df = pandas.DataFrame(columns=["Gene", drug_col, "Pvalue", "Coefficient", "Effect",
												 "# Sample", "is_random"])

		# Take the feature matrix
		whole_feature_matrix = create_feature_matrix(feature=feature, level=level, selection_criteria=selection_criteria,
													 tissue=tissue, min_cl_criteria = min_cl_criteria)

		# Take the random effect data frame
		random_effect_matrix = create_raneff_matrix(combi_df=combi_df, drug_col=drug_col, response_df=whole_response_df)
		print("""
	------------------------------------------------------  
		LR is running...\n""")

		i, t = 0, len(whole_response_df.groupby([drug_col]))
		for dp, _ in whole_response_df.groupby([drug_col]):
			# For each drug combination we will regress each feature --> each drug combination will be corrected separately
			if len(dp.split("/")) > 1: d_title = dp.split("/")[0] + "_" + dp.split("/")[1]
			else: d_title = dp
			print(dp)
			response_m = get_LR_response(drug_info =dp , response_df= whole_response_df, drug_col= drug_col)
			covariate_m = get_LR_covariate(whole_combi= combi_df, stage=stage, tissue=tissue, msi_flag=msi_flag)
			if random:
				random_m = get_LR_raneff(drug_info=dp, drug_col=drug_col, matrix=random_effect_matrix)
			else:
				random_m = None
			dp_bio_df = pandas.DataFrame(columns=["Gene", drug_col, "Feature", "Pvalue", "Coefficient", "Effect",
												  "# Sample", "is_random"])
			if len(response_m.index) >= int(min_cl_criteria):
				if whole_feature_matrix is not None and len(whole_feature_matrix.columns) > 0:
					for g in whole_feature_matrix.columns:
						print(g)
						# For each feature --> feature array
						feature_m = get_LR_feature(element = g, matrix = whole_feature_matrix, scale = scale)
						dim = check_dimension(drug_name=dp, stage=stage, feature=feature, level=level, element=g,
											  data_type=data_type, covariate_matrix=covariate_m,
											  feature_matrix=feature_m, response_matrix=response_m,
											  random_matrix=random_m, random=random,
											  min_cl_criteria=min_cl_criteria,
											  selection_criteria=selection_criteria,
											  t_title=t_title, title=title, msi_title=msi_title, d_title=d_title)

						if dim is not None:
							if random:
								lmm = get_lmm(response=dim[["response"]], feature=dim[[g]],
											  covariate=dim[[i for i in dim.columns if i not in ["response", g]]],
											  random=dim[["screen"]])
							else:
								lmm = get_lmm(response=dim[["response"]], feature=dim[[g]],
											  covariate=dim[[i for i in dim.columns if i not in ["response", g]]],
											  random=None)

							pvalue = get_pvalue(lmm)
							coeff = get_coefficient(lmm, g)
							effect = get_effect(coeff=coeff, stage=stage)

							d = {"Gene": [g], drug_col: [dp], "Pvalue": [pvalue], "Coefficient": [coeff],
								 "Effect": [effect],
								 "# Sample" : [len(dim[["response"]].index)],
								 "# initial response": [len(response_m)], "# initial covariate": [len(covariate_m)],
								 "# initial feature": [len(feature_m)], "is_random": random,
								 "# initial random effect": [len(random_m_m)] if random else None}

							df = pandas.DataFrame.from_dict(d)
							dp_bio_df = pandas.concat([dp_bio_df, df], ignore_index=True)
				else: print("No Feature Matrix")
			else:
				print("Response for this dp is less than %s" % str(min_cl_criteria))

			if len(dp_bio_df.index) > 0:
				if len(dp_bio_df["Pvalue"]) > 0:
					dp_bio_df["Adj_p"] = multipletests(dp_bio_df["Pvalue"], alpha=0.05, method="fdr_bh",
													   is_sorted=False, returnsorted=False)[1]
					dp_bio_df["FDR"] = dp_bio_df.apply(lambda x: x.Adj_p * 100, axis=1)
				else:
					dp_bio_df["Adj_p"] = None
					dp_bio_df["FDR"] = None
				biomarker_df = pandas.concat([biomarker_df, dp_bio_df])
				biomarker_df["Feature"] = feature
				biomarker_df["Feature_level"] = level
				biomarker_df["Stage"] = stage
				biomarker_df["Estimate_LR"] = estimate_lr
				biomarker_df["Selection_Criteria"] = selection_criteria
				biomarker_df["Min_CLs"] = min_cl_criteria
				biomarker_df["Tissue"] = tissue
				biomarker_df["MSI_cov"] = True if msi_flag else False
			else:
				print("dp bio df is empty")
			i += 1
			if int(i * 100.0 / t) % 10 == 0:
				print(i * 100.0 / t)

		if len(biomarker_df.index) != 0:
			print("Drug targets are annotating.\n")
			# Annotations
			if stage != "mono":
				biomarker_df = biomarker_df.reset_index()

				for g, g_df in biomarker_df.groupby(["DrugComb"]):
					inds = list(biomarker_df[biomarker_df.DrugComb == g].index)
					biomarker_df.loc[inds, "gene_target_1"] = annotate_drug_target(
						drug_name=g.split("/")[0], annotation_type="gene_target")
					biomarker_df.loc[inds, "gene_target_2"] = annotate_drug_target(
						drug_name=g.split("/")[1], annotation_type="gene_target")
					biomarker_df.loc[inds, "pathway_target_1"] = annotate_drug_target(
						drug_name=g.split("/")[0], annotation_type="pathway_target")
					biomarker_df.loc[inds, "pathway_target_2"] = annotate_drug_target(
						drug_name=g.split("/")[1], annotation_type="pathway_target")
					biomarker_df.loc[inds, "drug_type_1"] = annotate_drug_target(
						drug_name=g.split("/")[0], annotation_type="drug_type")
					biomarker_df.loc[inds, "drug_type_2"] = annotate_drug_target(
						drug_name=g.split("/")[1], annotation_type="drug_type")

				biomarker_df["Association"] = biomarker_df.apply(lambda x: x.Effect if x.FDR < fdr_limit else "not significant",axis=1)
				biomarker_df["Association_name"] = biomarker_df.apply(lambda x: x.DrugComb + ":" + x.Gene, axis=1)


			else:
				biomarker_df = biomarker_df.reset_index()
				for g, g_df in biomarker_df.groupby(["library_name"]):
					inds = list(biomarker_df[biomarker_df.library_name == g].index)
					biomarker_df.loc[inds, "gene_target"] = annotate_drug_target(
						drug_name=g, annotation_type="gene_target")
					biomarker_df.loc[inds, "pathway_target"] = annotate_drug_target(
						drug_name=g, annotation_type="pathway_target")
					biomarker_df.loc[inds, "drug_type"] = annotate_drug_target(
						drug_name=g, annotation_type="drug_type")

				biomarker_df["Association"] = biomarker_df.apply(lambda x: x.Effect if x.FDR < fdr_limit else "not significant",axis=1)
				biomarker_df["Association_name"] = biomarker_df.apply(lambda x: x.library_name + ":" + x.Gene, axis=1)

			print("LR file is writing.\n")
			biomarker_df.to_csv(output_path + "biomarker/LR/run_LR/LR_%s_%s_%s_%s%s_%s%s%s.csv"
								% (stage, estimate_lr, feature, level_text, t_title, title, msi_title, random_text), index=False)

			print("LR analysis is finished.\n")

		else:
			biomarker_df.to_csv(output_path + "biomarker/LR/run_LR/LR_%s_%s_%s_%s%s_%s%s%s.csv"
								% (stage, estimate_lr, feature, level_text, t_title, title, msi_title, random_text), index=False)
			print("Empty Data frame")
		print("""
	------------------------------------------------------""")
	else:
		biomarker_df = pandas.read_csv(output_path + "biomarker/LR/run_LR/LR_%s_%s_%s_%s%s_%s%s%s.csv"
									   % (stage, estimate_lr, feature, level_text, t_title, title, msi_title, random_text),
									   low_memory=False)
		print("LR analysis is read.\n")
		print("""
	------------------------------------------------------""")

	return biomarker_df



# ---------------------------------------------------------------------------#
#                          Biomarker Analysis		                         #
# ---------------------------------------------------------------------------#

def run_LR(args):
	print("""
	------------------------------------------------------                                                                                         
	      D R U G  C O M B - Biomarker Analysis
	           PhD Project of Cansu Dincer                                      		   
	            Wellcome Sanger Institute                                  
	------------------------------------------------------
	    """)

	if args["TISSUE"] is not None:
		t_text = " " + args["TISSUE"]
		t_title = "_" + "_".join(args["TISSUE"].split(" "))
	else: t_text, t_title = "", ""

	if args["MSI_COV"]: msi_title = "_msi"
	else: msi_title = ""

	if args["RANDOM"]: random_text = "_randomeffect"
	else: random_text = ""

	level_text = "_".join(args["LEVEL"].split(" "))

	if args["FEATURE"] == "transcription":
		feature_text = "Transcription (log2 filtered) - Genes were selected by " + str(args["CRITERIA"])
		title, scale = "log2_filtered_" + str(args["CRITERIA"]), True
	elif args["FEATURE"] == "proteomics":
		feature_text = "Proteomics (Averaged) - Genes were selected by " + str(args["CRITERIA"])
		title, scale = "averaged_" + str(args["CRITERIA"]), True
	elif args["FEATURE"] == "msi":
		feature_text = "MSI Positive"
		title, level, scale = "positive", "level", False
	elif args["FEATURE"] == "mutation":
		if args["LEVEL"] == "genes_driver_mut":
			feature_text = "Genes with Driver Mutation - Minimum model number with mutation as " + str(args["CRITERIA"])
		elif args["LEVEL"] == "mutations_driver":
			feature_text = "Driver Mutation - Minimum model number with mutation as " + str(args["CRITERIA"])
		title, scale = str(args["CRITERIA"]), False
	elif args["FEATURE"] == "amplification":
		feature_text = "Amplification - Minimum model number with amplification as " + str(args["CRITERIA"])
		title, scale = str(args["CRITERIA"]), False
	elif args["FEATURE"] == "deletion":
		feature_text = "Deletion - Minimum model number with deletion as " + str(args["CRITERIA"])
		title, scale = str(args["CRITERIA"]), False
	elif args["FEATURE"] == "hypermethylation":
		feature_text = "Hypermethylation - Minimum model number with hypermethylation as " + str(args["CRITERIA"])
		title, scale = str(args["CRITERIA"]), False
	elif args["FEATURE"] == "gain":
		feature_text = "Gain of Function - Minimum model number with GoF as " + str(args["CRITERIA"])
		title, scale = str(args["CRITERIA"]), False
	elif args["FEATURE"] == "loss":
		feature_text = "Loss of Function - Minimum model number with LoF as " + str(args["CRITERIA"])
		title, scale = str(args["CRITERIA"]), False
	elif args["FEATURE"] == "clinicalsubtype":
		feature_text = "Clinical subtype - %s - Minimum model number with LoF as %s" % (args["LEVEL"], str(args["CRITERIA"]))
		title, scale = str(args["CRITERIA"]), False

	if args["EST_LR"] == "XMID": est_lr = "Scaled IC50"
	elif args["EST_LR"] == "EMAX": est_lr = "Emax"

	if args["STAGE"] == "mono":
		stage_text = "Library " + est_lr
	elif args["STAGE"] == "combo":
		stage_text = "Combination " + est_lr
	elif args["STAGE"] == "delta":
		stage_text = "Delta " + est_lr
	"""
	print("The Mixed Effect Linear Regression will be run by:\n- Models : %s\n- Feature: %s\n- Metrics: %s\n- Min Model Criteria: %s%s%s " % (t_text, feature_text, stage_text, args["MIN_CL"]), "\n- MSI as covariate" if args["MSI_COV"] else "","\n- Screen types as random effects" if args["RANDOM"] else "")
	"""
	print("""
	------------------------------------------------------  """)
	f = open(output_path + "biomarker/LR/setup_LR/LR_%s_%s_%s_%s%s_%s%s%s.txt"
									   % (args["STAGE"], args["EST_LR"], args["FEATURE"], level_text,
										  t_title, title, msi_title, random_text), "w")
	f.write("File " + output_path + "biomarker/LR/run_LR/LR_%s_%s_%s_%s%s_%s%s%s.txt\n"
			% (args["STAGE"], args["EST_LR"], args["FEATURE"], level_text, t_title, title, msi_title, random_text))
	f.write("The Mixed Effect Linear Regression:\n")
	f.write("- Models : %s\n- Feature: %s\n- Metrics: %s\n- Min Model Criteria: %s\n- FDR: %s\n-MSI as covariate:%s\n-Screen type as random effect:%s"""
			% (t_text, feature_text, stage_text, args["MIN_CL"], args["FDR"], True if args["MSI_COV"] else False, True if args["RANDOM"] else False))

	df = x_biomarker(feature= args["FEATURE"], level = args["LEVEL"], selection_criteria = args["CRITERIA"],
					 tissue=args["TISSUE"], stage=args["STAGE"], estimate_lr= args["EST_LR"],
					 min_cl_criteria=int(args["MIN_CL"]), msi_flag=args["MSI_COV"],
					 fdr_limit=int(args["FDR"]), random=None)

	f.write("# of total tests: %d" % len(df.index))
	f.close()

	return 1




if __name__ == '__main__':
	args = take_input()
	print(args)

	_ = run_LR(args)

