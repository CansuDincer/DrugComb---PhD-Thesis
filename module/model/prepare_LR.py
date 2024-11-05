"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 16 January 2023
Last Update : 8 August 2023
Input : Biomarker Analysis
Output : Input files
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, argparse, scipy, sklearn, sys, random
from scipy.stats import mannwhitneyu, normaltest

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

from statannotations.Annotator import Annotator
from sklearn.preprocessing import quantile_transform
import statsmodels.api as sm
from sklearn.utils import shuffle
from utils import *
import warnings
warnings.filterwarnings('ignore')


from CombDrug.module.path import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.responses import *
from CombDrug.module.data.omics import *


# ---------------------------------------------------------------------------#
#                               	Inputs                                   #
# ---------------------------------------------------------------------------#

def take_input():
	parser = argparse.ArgumentParser(prog="CombDrug Linear Regression Input",
									 usage="%(prog)s [inputs]",
									 description="""
                                     **********************************
                                     		   Prepare files
                                     **********************************""")

	for group in parser._action_groups:
		if group.title == "optional arguments":
			group.title = "Inputs"
		elif "positional arguments":
			group.title = "Mandatory Inputs"

	# Feature
	parser.add_argument("-tissue", dest="TISSUE", required=True)
	parser.add_argument("-stage", dest="STAGE", required=False)
	parser.add_argument("-feature", dest="FEATURE", required=False)
	parser.add_argument("-level", dest="LEVEL", required=False)
	parser.add_argument("-msi_cov", dest="MSI_COV", action="store_true")
	parser.add_argument("-run", dest="RUN_FOR", required=True)
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                                Functions                                   #
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
		print(combi)
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



def prepare_responses(tissue, stage):
	_ = get_response_data(tissue=tissue, stage=stage, estimate_lr = "XMID")
	return True




def prepare_covariates(tissue, stage):

	if stage == "mono":
		combi_df = combine_combi(estimate_data="XMID", treatment=stage)
		print(tissue)
		print(stage)
		_ = get_LR_covariate(whole_combi=combi_df, tissue=tissue, stage=stage, msi_flag=False)
	else:
		combi_df = combine_combi(estimate_data="XMID", treatment="combination")
		print(stage)
		_ = get_LR_covariate(whole_combi=combi_df, tissue=tissue, stage=stage, msi_flag=False)
	return True



def prepare_features(tissue, feature, level):

	if feature == "transcription": selection_criteria="variance"
	elif feature == "proteomics": selection_criteria="variance"
	else: selection_criteria =5

	_ = create_feature_matrix(feature=feature, level=level, tissue=tissue,
							  selection_criteria=selection_criteria, min_cl_criteria=15)
	return True



def prepare(args):
	if args["RUN_FOR"] == "response":
		_ = prepare_responses(tissue=args["TISSUE"], stage =args["STAGE"])
	elif args["RUN_FOR"] == "covariate":
		_ = prepare_covariates(tissue=args["TISSUE"], stage =args["STAGE"])
	else:
		 _ = prepare_features(tissue=args["TISSUE"], feature =args["FEATURE"], level =args["LEVEL"])
	return True



def get_cov_gr(response):
	growth_rate = get_growth()
	growth_rate.columns = ["SIDM", "growth_rate"]
	growth_rate_df = growth_rate[growth_rate.SIDM.isin(response.SIDM.unique())]
	return growth_rate_df, ["growth_rate"]


def get_cov_gp(response):
	# Model info from CMP
	model_info = cancer_models()
	# Growth Properties
	media_type = model_info[["sanger_id", "growth_properties"]].drop_duplicates()
	media_type = media_type.set_index(["sanger_id"])
	media_type = pandas.get_dummies(media_type["growth_properties"])
	media_cols = list(media_type.columns)
	media_type = media_type.reset_index()
	media_type = media_type.rename({"sanger_id": "SIDM"}, axis=1)
	media_type = media_type[media_type.SIDM.isin(response.SIDM.unique())]
	media_type = media_type.drop_duplicates()
	return media_type, media_cols


def get_cov_tissue(response):
	# Combi data
	combi = combine_combi(estimate_data="XMID", treatment="combination")
	# Tissue
	tissue_df = combi[["SIDM", "tissue_type"]]
	tissue_df = tissue_df.set_index(["SIDM"])
	tissue_df = pandas.get_dummies(tissue_df["tissue_type"])
	tissue_columns = list(tissue_df.columns)
	tissue_df = tissue_df.reset_index()
	tissue_df = tissue_df[tissue_df.SIDM.isin(response.SIDM.unique())]
	tissue_df = tissue_df.fillna(value=numpy.nan)
	tissue_df = tissue_df.drop_duplicates()
	return tissue_df, tissue_columns


def get_cov_msi(response):
	# Model info from CMP
	model_info = cancer_models()
	# MSI
	msi = model_info[["sanger_id", "msi_status"]].drop_duplicates()
	msi = msi.set_index(["sanger_id"])
	msi = pandas.get_dummies(msi["msi_status"])
	msi_columns = list(msi.columns)
	msi = msi.reset_index()
	msi = msi.rename({"sanger_id": "SIDM"}, axis=1)
	msi = msi[msi.SIDM.isin(response.SIDM.unique())]
	msi = msi.drop_duplicates()
	return msi, msi_columns


def get_cov_expt(response, combi):
	# Exp type
	screen = combi[["SIDM", "screen_type", "DrugComb"]]
	screen = screen.set_index(["SIDM", "DrugComb"])
	screen = pandas.get_dummies(screen["screen_type"])
	screen_columns = list(screen.columns)
	screen = screen.reset_index()
	screen = screen[screen.SIDM.isin(response.SIDM.unique())]
	screen = screen.drop_duplicates()
	return screen, screen_columns


def select_random_features(tissue, msi_specific):

	if tissue is not None:
		t_title = "_" + "_".join(tissue.split(" "))
	elif tissue in ["panliquid", "pansolid"]:
		t_title = "_" + ""

	if msi_specific:
		msi_title = "_msi"
	else: msi_title = ""

	if "AIC_comparison%s%s.csv" % (t_title, msi_title) not in os.listdir(output_path + "biomarker/covariates/test/"):
		print("features collecting")
		mut_features = get_mutations(level="genes_driver_mut", tissue=tissue, selection_criteria=5, min_cl_criteria=15)
		gex_features = get_transcriptomics(level="genes_cancer", tissue=tissue, plotting=False,
										   selection_criteria="variance", min_cl_criteria=15)
		amp_features = get_cnv(feature="amplification", level="genes_cancer", tissue=tissue,
							   selection_criteria=5, min_cl_criteria=15)
		del_features = get_cnv(feature="deletion", level="genes_cancer", tissue=tissue,
							   selection_criteria=5, min_cl_criteria=15)
		pex_features = get_proteomics(level="genes_cancer", tissue=tissue,
									  selection_criteria="variance", plotting=False, min_cl_criteria=15)
		met_features = get_methylation(level="genes_cancer", tissue=tissue,
									   selection_criteria=5, min_cl_criteria=15)
		gain_features = get_gain(level="genes_cancer", tissue=tissue,
								 selection_criteria=5, min_cl_criteria=15)
		loss_features = get_loss(level="genes_cancer", tissue=tissue,
								 selection_criteria=5, min_cl_criteria=15)
		whole_response = get_response_data(tissue=tissue, stage="combo", estimate_lr="XMID")

		# Combi data
		combi = combine_combi(estimate_data="XMID", treatment="combination")

		if tissue is not None:
			models = get_sidm_tissue(tissue_type=tissue)
		else:
			models = all_screened_SIDMs(project_list=None, integrated=True)

		response = whole_response[whole_response.index.isin(models)]
		all_drug_combinations = response.DrugComb.unique()

		total_df = pandas.DataFrame(columns = ["iteration", "hypothesis", "dp", "feature", "AIC", "BIC"])
		total_count = 0
		for i in range(100):
			if msi_specific:
				# Mutation
				t0 = mut_features.columns
				if len(t0) > 9:
					random_mut_features = random.choices(t0, k=10)
					random_features = mut_features[random_mut_features]
				else:
					random_features = mut_features.copy()
			else:
				# Mutation
				t0 = mut_features.columns
				if len(t0) > 1:
					random_mut_features = random.choices(t0, k=2)
					random_mut_features = list(set(random_mut_features))
					ran_mut_features = mut_features[random_mut_features]
				else:
					random_mut_features = list()
					ran_mut_features = pandas.DataFrame()
				# Gex
				t1 = list(set(gex_features.columns).difference(set(random_mut_features)))
				if len(t1) > 1:
					random_gex_features = random.choices(t1, k=2)
					random_gex_features = list(set(random_gex_features))
					ran_gex_features = gex_features[random_gex_features]
				else:
					random_gex_features = list()
					ran_gex_features = pandas.DataFrame()
				# Amplification
				t2 = list(set(amp_features.columns).difference(set(random_gex_features + random_mut_features)))
				if len(t2) > 1:
					random_amp_features = random.choices(t2, k=2)
					random_amp_features = list(set(random_amp_features))
					ran_amp_features = amp_features[random_amp_features]
				else:
					random_amp_features = list()
					ran_amp_features = pandas.DataFrame()
				# Deletion
				t3 = list(set(del_features.columns).difference(
					set(random_gex_features + random_mut_features + random_amp_features)))
				if len(t3) > 1:
					random_del_features = random.choices(t3, k=2)
					random_del_features = list(set(random_del_features))
					ran_del_features = del_features[random_del_features]
				else:
					random_del_features = list()
					ran_del_features = pandas.DataFrame()
				# Pex
				t4 = list(set(pex_features.columns).difference(
					set(random_gex_features + random_mut_features + random_amp_features+ random_del_features)))
				if len(t4) > 1:
					random_pex_features = random.choices(t4, k=2)
					random_pex_features = list(set(random_pex_features))
					ran_pex_features = pex_features[random_pex_features]
				else:
					random_pex_features = list()
					ran_pex_features = pandas.DataFrame()
				# Gain
				t5 = list(set(gain_features.columns).difference(
					set(random_gex_features + random_mut_features + random_amp_features+
						random_del_features+ random_pex_features)))
				if len(t5) > 1:
					random_gain_features = random.choices(t5, k=2)
					random_gain_features = list(set(random_gain_features))
					ran_gain_features = gain_features[random_gain_features]
				else:
					random_gain_features = list()
					ran_gain_features = pandas.DataFrame()
				# Loss
				t6 = list(set(loss_features.columns).difference(
					set(random_gex_features + random_mut_features + random_amp_features+
						random_del_features+ random_pex_features + random_gain_features)))
				if len(t6) > 1:
					random_loss_features = random.choices(t6, k=2)
					random_loss_features = list(set(random_loss_features))
					ran_loss_features = loss_features[random_loss_features]
				else:
					random_loss_features = list()
					ran_loss_features = pandas.DataFrame()
				# Methylation
				t7 = list(set(met_features.columns).difference(
					set(random_gex_features + random_mut_features + random_amp_features+ random_del_features+
						random_pex_features + random_gain_features + random_loss_features)))
				if len(t7) > 1:
					random_met_features = random.choices(t7, k=2)
					random_met_features = list(set(random_met_features))
					ran_met_features = met_features[random_met_features]
				else:
					ran_met_features = pandas.DataFrame()

				random_features = pandas.concat([ran_mut_features, ran_gex_features, ran_amp_features,
												 ran_del_features, ran_pex_features, ran_met_features,
												 ran_gain_features, ran_loss_features], axis=1)

			random_combinations = random.choices(all_drug_combinations, k=100)
			for dp in random_combinations:
				r = response[response.DrugComb ==dp].reset_index()
				gp, gp_cols = get_cov_gp(r)
				gr, gr_cols = get_cov_gr(r)
				exp_t, exp_t_cols = get_cov_expt(r, combi)

				if tissue in ["panliquid", "pansolid"]:
					x, x_cols = get_cov_tissue(r)
				elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
					if msi_specific:
						x, x_cols = get_cov_msi(r)
					else:
						x = pandas.DataFrame(index=r.SIDM.unique())
						x= x.reset_index()
						x.columns = ["SIDM"]
						x_cols = ["SIDM"]
				# Make it empty
				else:
					x = pandas.DataFrame(index=r.SIDM.unique())
					x= x.reset_index()
					x.columns = ["SIDM"]
					x_cols = ["SIDM"]

				random_feature_list = random_features.columns
				for feature in random_feature_list:
					f = random_features[[feature]].reset_index()

					if "SIDM" not in f.columns:
						col_name = list(set(f.columns).difference(set(random_feature_list)))[0]
						f = f.rename({col_name: "SIDM"}, axis=1)

					r2 = pandas.merge(r, f, on=["SIDM"])
					r2 = pandas.merge(r2, gp, on = ["SIDM"])
					r2 = pandas.merge(r2, gr, on=["SIDM"])
					r2 = pandas.merge(r2, exp_t, on=["SIDM", "DrugComb"])
					r2 = pandas.merge(r2, x, on=["SIDM"], how="outer")
					r2 = r2.dropna()
					r2 = r2.drop_duplicates()

					f_r2_shuffled = r2[[col for col in r2.columns if col not in [feature, "SIDM", "DrugComb", "Direction", "response"]]]
					r2_shuffled = r2[[col for col in r2.columns if col in [feature, "SIDM", "DrugComb", "Direction", "response"]]]
					for col in f_r2_shuffled.columns:
						s = sklearn.utils.shuffle(f_r2_shuffled[[col]]).reset_index()[[col]]
						r2_shuffled[col] = s

					r2_shuffled = r2_shuffled.dropna()
					r2_shuffled = r2_shuffled.drop_duplicates()

					# H0 : y = β0 + β1 . F
					h0_col = [feature]

					# H1 : y = β0 . C[GR] + β1 . F
					h1_col = [feature] + gr_cols

					# H2 : y = β0 . C[Tissue/MSI] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h2_col = [feature] + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h2_col = [feature] + x_cols
						else:
							h2_col = [feature]
					else:
						h2_col = [feature]

					# H3 : y = β0 . C[GP] + β1 . F
					h3_col = [feature] + gp_cols

					# H4 : y = β0 . C[Exp Type] + β1 . F
					h4_col = [feature] + exp_t_cols

					# H5 : y = β0 . C[GR + Tissue/MSI] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h5_col = [feature] + gr_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h5_col = [feature] + gr_cols + x_cols
						else:
							h5_col = [feature] + gr_cols
					else:
						h5_col = [feature] + gr_cols

					# H6 : y = β0 . C[GR + GP] + β1 . F
					h6_col = [feature] + gr_cols + gp_cols

					# H7 : y = β0 . C[GR + Exp Type] + β1 . F
					h7_col = [feature] + gr_cols + exp_t_cols

					# H8 : y = β0 . C[Tissue/MSI + GP] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h8_col = [feature] + gp_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h8_col = [feature] + gp_cols + x_cols
						else:
							h8_col = [feature] + gp_cols
					else:
						h8_col = [feature] + gp_cols

					# H9 : y = β0 . C[Tissue/MSI + Exp Type] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h9_col = [feature] + exp_t_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h9_col = [feature] + exp_t_cols + x_cols
						else:
							h9_col = [feature] + exp_t_cols
					else:
						h9_col = [feature] + exp_t_cols

					# H10 : y = β0 . C[GP + Exp Type] + β1 . F
					h10_col = [feature] + gp_cols + exp_t_cols

					# H11 : y = β0 . C[GR + Tissue/MSI + GP] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h11_col = [feature] + gr_cols + gp_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h11_col = [feature] + gr_cols + gp_cols + x_cols
						else:
							h11_col = [feature] + gr_cols + gp_cols
					else:
						h11_col = [feature] + gr_cols + gp_cols

					# H12 : y = β0 . C[GR + Tissue/MSI + Exp Type] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h12_col = [feature] + gr_cols + exp_t_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h12_col = [feature] + gr_cols + exp_t_cols + x_cols
						else:
							h12_col = [feature] + gr_cols + exp_t_cols
					else:
						h12_col = [feature] + gr_cols + exp_t_cols

					# H13 : y = β0 . C[GR + GP + Exp Type] + β1 . F
					h13_col = [feature] + gr_cols + gp_cols + exp_t_cols

					# H14 : y = β0 . C[Tissue/MSI + GP + Exp Type] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h14_col = [feature] + gp_cols + exp_t_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h14_col = [feature] + gp_cols + exp_t_cols + x_cols
						else:
							h14_col = [feature] + gp_cols + exp_t_cols
					else:
						h14_col = [feature] + gp_cols + exp_t_cols

					# H15 : y = β0 . C[GR, Tissue/MSI + GP + Exp Type] + β1 . F
					if tissue in ["panliquid", "pansolid"]:
						h15_col = [feature] + gr_cols + gp_cols + exp_t_cols + x_cols
					elif tissue in ["Ovary", "Stomach", "Endometrium", "Large Intestine"]:
						if msi_specific:
							h15_col = [feature] + gr_cols + gp_cols + exp_t_cols + x_cols
						else:
							h15_col = [feature] + gr_cols +  gp_cols + exp_t_cols
					else:
						h15_col = [feature] + gr_cols + gp_cols + exp_t_cols

					cols = {"H0": h0_col, "H1": h1_col, "H2": h2_col, "H3": h3_col,
							"H4": h4_col, "H5": h5_col, "H6": h6_col, "H7": h7_col,
							"H8": h8_col, "H9": h9_col, "H10": h10_col, "H11": h11_col,
							"H12": h12_col, "H13": h13_col, "H14": h14_col, "H15": h15_col}
					for c in range(16):
						# Take response
						y = r2["response"]
						shuffled_y = r2_shuffled["response"]

						# Take feature and covariate according to the hypothesis above
						X = r2[cols["H%d" % c]]
						X = sm.add_constant(X)

						# Take feature and shuffled covariate according to the hypothesis above
						shuffled_X = r2_shuffled[cols["H%d" % c]]
						shuffled_X = sm.add_constant(shuffled_X)

						# Run model both for shuffled and normal
						if len(y) >= 15 and len(X) >= 15 and len(shuffled_X) >= 15 and len(shuffled_y) >= 15:
							model = sm.OLS(y.astype(float), X.astype(float)).fit()
							shuffled_model = sm.OLS(shuffled_y.astype(float), shuffled_X.astype(float)).fit()

							aic, shuffled_aic = model.aic, shuffled_model.aic
							bic, shuffled_bic = model.bic, shuffled_model.bic

							if msi_specific:
								fet =  "Mutation_%s" % feature
							else:
								if feature in ran_mut_features: fet = "Mutation_%s" % feature
								elif feature in ran_gex_features: fet = "GEx_%s" % feature
								elif feature in ran_amp_features: fet = "Amp_%s" % feature
								elif feature in ran_del_features: fet = "Del_%s" % feature
								elif feature in ran_pex_features: fet = "PEx_%s" % feature
								elif feature in ran_met_features: fet = "Met_%s" % feature
								elif feature in ran_gain_features: fet = "Gain_%s" % feature
								elif feature in ran_loss_features: fet = "Loss_%s" % feature

							total_x = pandas.DataFrame([{"iteration" : i, "hypothesis": "H%d" % c,
														 "dp": dp, "feature": fet,
														 "AIC": aic, "AIC_shuffled": shuffled_aic,
														 "BIC": bic, "BIC_shuffled": shuffled_bic}])
							total_df = pandas.concat([total_df, total_x], ignore_index=True)
				if (total_count / 100.0) % 10 == 0:
					print(total_count / 100.0)

		total_df.to_csv(output_path + "biomarker/covariates/test/AIC_comparison%s%s.csv" % (t_title, msi_title))

	else:
		total_df = pandas.read_csv(output_path + "biomarker/covariates/test/AIC_comparison%s%s.csv"
								   % (t_title, msi_title), index_col=0)

	return total_df


def test_AIC(tissue, msi_specific):

	if tissue not in ["pansolid", "panliquid"]: t_title, text_title = "_" + "_".join(tissue.split(" ")), tissue
	else: t_title, text_title = "_" + tissue, tissue

	if msi_specific: msi_title = "_msi"
	else: msi_title = ""

	flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
					  markeredgecolor="none")
	medianprops = dict(linestyle="-", linewidth=1.5, color="red")
	boxprops = dict(facecolor="white", edgecolor="darkgrey")
	whiskerprops = dict(color="darkgrey")

	total_df = select_random_features(tissue, msi_specific)

	# Check if covariates are different then random
	if total_df.empty is False:
		x = total_df.melt(id_vars=["hypothesis", "iteration"], value_vars=["AIC", "AIC_shuffled"])

		mw_pvalues = list()
		for h in range(0, 16):
			hyp = "H%d" % h
			aic = x[(x.hypothesis == hyp) & (x.variable == "AIC")][["value"]]
			aic_shuf = x[(x.hypothesis == hyp) & (x.variable == "AIC_shuffled")][["value"]]
			if aic is not None and aic_shuf is not None:
				mw_pvalues.append(scipy.stats.mannwhitneyu(aic, aic_shuf, alternative = "less").pvalue)

		if mw_pvalues:
			formatted_mw_pvals = [f'%.2e' %pvalue for pvalue in mw_pvalues]

			pairs = list()
			for h in range(0, 16):
				hyp = "H%d" % h
				pairs.append(((hyp, "AIC"), (hyp, "AIC_shuffled")))


			plt.figure(figsize=(18,12))
			ax=sns.stripplot(data=x, x="hypothesis", y="value", hue="variable", size=0.8,
						  palette={"AIC":"red", "AIC_shuffled":"navy"}, alpha=0.3, dodge=True)
			handles, labels = ax.get_legend_handles_labels()
			sns.boxplot(ax=ax, data=x, x="hypothesis", y="value", hue="variable",
						flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
						whiskerprops=whiskerprops, dodge=True)
			ax.legend_.remove()
			ax.legend(handles, labels, loc=4)

			annotator = Annotator(ax=ax, pairs=pairs, data=x, x="hypothesis", y="value", hue="variable")
			annotator.set_custom_annotations(formatted_mw_pvals)
			annotator.annotate()
			plt.xlabel("Hypothesis", fontsize=20)
			plt.xticks(fontsize=16)
			plt.ylabel("AIC", fontsize=20)
			plt.yticks(fontsize=16)
			plt.title("AIC Test for Random vs Empiric Covariates %s\n Mann Whitney Test" % text_title)
			plt.tight_layout()
			plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_randomisation_MW_comparison%s%s.pdf" % (t_title, msi_title))
			plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_randomisation_MW_comparison%s%s.jpg" % (t_title, msi_title))
			plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_randomisation_MW_comparison%s%s.png" % (t_title, msi_title))
			plt.close()

		# Check if covariates are adding effect on Null
		y = x[x.variable == "AIC"]

		hyp_pairs = list()
		aic_mw_pvalues = list()
		aic_null = y[(y.hypothesis == "H0")][["value"]]
		for h in range(1, 16):
			hyp = "H%d" % h
			hyp_pairs.append(("H0", hyp))
			aic = y[(y.hypothesis == hyp)][["value"]]
			if aic is not None and aic_null is not None:
				aic_mw_pvalues.append(scipy.stats.mannwhitneyu(aic, aic_null, alternative = "less").pvalue)

		if aic_mw_pvalues:
			formatted_aic_mw_pvals = [f'%.2e' %pvalue for pvalue in aic_mw_pvalues]

			plt.figure(figsize=(16,18))
			ax=sns.stripplot(data=y, x="hypothesis", y="value", size=0.8,
							 color="red", alpha=0.3, dodge=True)
			handles, labels = ax.get_legend_handles_labels()
			sns.boxplot(ax=ax, data=y, x="hypothesis", y="value",
						flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
						whiskerprops=whiskerprops, dodge=True)
			ax.legend(handles, labels, loc=4)

			annotator = Annotator(ax=ax, pairs=hyp_pairs, data=y, x="hypothesis", y="value")
			annotator.set_custom_annotations(formatted_aic_mw_pvals)
			annotator.annotate()
			plt.xlabel("Hypothesis", fontsize=20)
			plt.xticks(fontsize=16)
			plt.ylabel("AIC", fontsize=20)
			plt.yticks(fontsize=16)
			plt.title("AIC Test for Covariates vs Null %s\n Mann Whitney Test" % text_title)
			plt.tight_layout()
			plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_null_MW_comparison%s%s.pdf" % (t_title, msi_title))
			plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_null_MW_comparison%s%s.jpg" % (t_title, msi_title))
			plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_null_MW_comparison%s%s.png" % (t_title, msi_title))
			plt.close()

	return True


def test_AIC_pairs(tissue, msi_specific, hyp1, hyp2, plotting):
	"""
	Test if extra covariate in HYP2 makes fit statistically better
	:param tissue: Tissue name None if pancancer
	:param hyp1: The Null
	:param hyp2: The alternative
	:param plotting
	:return: P value from Mann Whitney Test
	"""

	flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
					  markeredgecolor="none")
	medianprops = dict(linestyle="-", linewidth=1.5, color="red")
	boxprops = dict(facecolor="white", edgecolor="darkgrey")
	whiskerprops = dict(color="darkgrey")

	if tissue not in ["pansolid", "panliquid"]: t_title, text_title = "_" + "_".join(tissue.split(" ")), tissue
	else: t_title, text_title = tissue, tissue

	if msi_specific: msi_title = "_msi"
	else: msi_title = ""

	total_df = select_random_features(tissue=tissue, msi_specific=msi_specific)

	# Check if covariates are different then random
	x = total_df.melt(id_vars=["hypothesis", "iteration"], value_vars=["AIC", "AIC_shuffled"])

	# Check if hyp2 covariates are adding effect on hyp1
	y = x[x.variable == "AIC"]
	aic_1 = y[(y.hypothesis == hyp1)][["value"]]
	aic_2 = y[(y.hypothesis == hyp2)][["value"]]
	mw_pval = scipy.stats.mannwhitneyu(aic_2, aic_1, alternative="less").pvalue
	formatted_aic_mw_pval = [f'%.2e' % pvalue for pvalue in [mw_pval]]

	if plotting:
		y2 = y[y.hypothesis.isin([hyp1, hyp2])]

		plt.figure(figsize=(6,8))
		ax=sns.stripplot(data=y2, x="hypothesis", y="value", size=0.8,
						 color="navy", alpha=0.3, dodge=True)
		handles, labels = ax.get_legend_handles_labels()
		sns.boxplot(ax=ax, data=y2, x="hypothesis", y="value",
					flierprops=flierprops, medianprops=medianprops, boxprops=boxprops,
					whiskerprops=whiskerprops, dodge=True)
		ax.legend(handles, labels, loc=4)

		annotator = Annotator(ax=ax, pairs=[(hyp1, hyp2)], data=y2,
							  x="hypothesis", y="value")
		annotator.set_custom_annotations(formatted_aic_mw_pval)
		annotator.annotate()

		plt.xlabel("Hypothesis", fontsize=20)
		plt.xticks(fontsize=16)
		plt.ylabel("AIC", fontsize=20)
		plt.yticks(fontsize=16)
		plt.title("AIC Test H0: %s < %s \n Mann Whitney Test" % (hyp1, hyp2))
		plt.tight_layout()
		plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_%s_%s_MW_comparison%s%s.pdf" % (hyp1, hyp2, t_title, msi_title))
		plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_%s_%s_MW_comparison%s%s.jpg" % (hyp1, hyp2, t_title, msi_title))
		plt.savefig(output_path + "biomarker/covariates/test/figures/AIC_%s_%s_MW_comparison%s%s.png" % (hyp1, hyp2, t_title, msi_title))
		plt.close()

	return mw_pval


if __name__ == '__main__':
	args = take_input()
	if args["TISSUE"] == "None": args["TISSUE"] = None
	print(args)

	if args["RUN_FOR"] != "covariate_testing":
		_ = prepare(args)
	else:
		_ = select_random_features(tissue=args["TISSUE"], msi_specific=args["MSI_COV"])

