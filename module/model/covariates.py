"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 12 July 2023
Last Update : 15 April 2024
Input : Covariate Detection - PCA and ACI Analaysis
Output : Selected covariates
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, scipy, sklearn, sys, matplotlib, random, argparse
from scipy import stats
from scipy.stats import mannwhitneyu, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
from statannotations.Annotator import Annotator
from sklearn.preprocessing import quantile_transform
import statsmodels.api as sm
from sklearn.utils import shuffle
from utils import *
import warnings
warnings.filterwarnings('ignore')

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
	parser = argparse.ArgumentParser(prog="CombDrug Covariate Analysis",
									 usage="%(prog)s [inputs]",
									 description="""
                                     **********************************
                                     		   Test covariates
                                     **********************************""")

	for group in parser._action_groups:
		if group.title == "optional arguments":
			group.title = "Inputs"
		elif "positional arguments":
			group.title = "Mandatory Inputs"

	# Feature
	parser.add_argument("-tissue", dest="TISSUE", default=None)
	parser.add_argument("-msi_cov", dest="MSI_COV", action="store_true")
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


args = take_input()
if args["TISSUE"] == "None": args["TISSUE"] = None
# ---------------------------------------------------------------------------#
#                                 Analysis                                   #
# ---------------------------------------------------------------------------#

def do_cov_PCA(tissue, stage, estimate_lr, covariate_type):

	# Response data
	whole_response = get_response_data(tissue=tissue, stage=stage, estimate_lr=estimate_lr)
	if tissue is not None:
		models = get_sidm_tissue(tissue=tissue)
	else:
		models = all_screened_SIDMs(project_list=None, integrated=True)
	response = whole_response.loc[[i for i in models if i in whole_response.index]]
	response = response.reset_index()
	r = response[["SIDM", "DrugComb", "response"]]
	m = pandas.pivot(r, columns="DrugComb", index="SIDM", values="response").reset_index().rename_axis(None)
	m.columns.name = ' '
	m = m.set_index(["SIDM"])

	# Model info from CMP
	model_info = cancer_models()

	# Combi information from screens
	if stage == "mono":
		combi = combo_combine_all(estimate_data="XMID", treatment="mono")
	else:
		combi = combo_combine_all(estimate_data="XMID", treatment="combination")

	if covariate_type == "growth_rate":
		cov_data_type = "continuous"
		growth_rate = get_growth()
		growth_rate.columns = ["SIDM", "growth_rate"]
		growth_rate = growth_rate[growth_rate.SIDM.isin(whole_response.index.unique())]
		growth_rate = growth_rate.rename({"growth_rate": "cov"}, axis=1)
		df = pandas.merge(m, growth_rate, on=["SIDM"])

	elif covariate_type == "tissue":
		cov_data_type = "binary"
		tissue_df = combi[["SIDM", "tissue"]]
		tissue_df = tissue_df[tissue_df.SIDM.isin(whole_response.index.unique())]
		tissue_df = tissue_df.rename({"tissue": "cov"}, axis=1)
		df = pandas.merge(m, tissue_df, on=["SIDM"])

	elif covariate_type == "growth_properties":
		cov_data_type = "binary"
		media_type = model_info[["sanger_id", "growth_properties"]]
		media_type = media_type.rename({"sanger_id": "SIDM"}, axis=1)
		media_type = media_type[media_type.SIDM.isin(whole_response.index.unique())]
		media_type = media_type.rename({"growth_properties": "cov"}, axis=1)
		df = pandas.merge(m, media_type, on=["SIDM"])

	elif covariate_type == "msi":
		cov_data_type = "binary"
		msi_cov = model_info[["sanger_id", "msi_status"]].drop_duplicates()
		msi_cov = msi_cov.rename({"sanger_id": "SIDM"}, axis=1)
		msi_cov = msi_cov[msi_cov.SIDM.isin(whole_response.index.unique())]
		msi_cov = msi_cov.rename({"msi_status": "cov"}, axis=1)
		df = pandas.merge(m, msi_cov, on=["SIDM"])


	df = df.drop_duplicates()
	df = df.dropna(axis="index")
	wout_cov_df = df[[c for c in df.columns if c != "cov"]]
	wout_cov_df = wout_cov_df.set_index(["SIDM"])
	cov_df = df[[c for c in df.columns if c == "cov"]]
	cov_df = cov_df.reset_index()[["cov"]]

	trans_wout_cov_df = quantile_transform(wout_cov_df)
	trans_wout_cov_df =pandas.DataFrame(trans_wout_cov_df)
	trans_wout_cov_df.index = wout_cov_df.index
	trans_wout_cov_df.columns = wout_cov_df.columns

	pca = PCA(n_components=2)
	pc = pca.fit_transform(trans_wout_cov_df)
	pc_df = pandas.DataFrame(data=pc, columns=["PC1", "PC2"])
	pc_label_df = pandas.concat([pc_df, cov_df], axis=1)
	pc1_evr = pca.explained_variance_ratio_[0]
	pc2_evr = pca.explained_variance_ratio_[1]

	plt.figure(figsize=(10, 10))
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlabel("PC 1 %.3f" % pc1_evr, fontsize=12)
	plt.ylabel("PC 2 %.3f" % pc2_evr, fontsize=12)
	plt.title("Principal Component Analysis\n%s | %s" % (tissue, covariate_type), fontsize=16)

	targets = list(df["cov"].unique())

	if cov_data_type == "continuous":
		sns.scatterplot(data=pc_label_df, x = "PC1", y = "PC2", hue="cov", palette="Reds")

	else:
		cmap = get_cmap("tab10")
		colors = cmap.colors
		for target, color in zip(targets, colors):
			inxs = list(pc_label_df[pc_label_df["cov"] == target].index)
			plt.scatter(pc_label_df.loc[inxs, "PC1"], pc_label_df.loc[inxs, "PC2"], c=color, s=50)

	for i in pc_label_df.index:
		plt.annotate(df.loc[i, "SIDM"],
					 xy=(pc_label_df.loc[i, "PC1"] + 0.01,
						 pc_label_df.loc[i, "PC2"] + 0.01),
					 xytext=(pc_label_df.loc[i, "PC1"]- 0.2,
							 pc_label_df.loc[i, "PC2"]+ 0.06),
					 arrowprops=dict(color='black', arrowstyle="->", connectionstyle="arc3"),
					 fontsize = 8)
	plt.legend(targets, prop={'size': 15})
	plt.show()

	return 1


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

#_ = select_random_features(tissue=args["TISSUE"], msi_specific=args["MSI_COV"])


def check_growth_rate_distributions():
	"""

	:return:
	"""

	tissues = ["Large Intestine", "Ovary", "Stomach", "Breast", "Skin", "Lung", "Pancreas", "Bone",
			   "Head and Neck", "Esophagus", "Bladder", "Peripheral Nervous System", "Kidney",
			   "Soft Tissue", "Central Nervous System", "Leukemia", "Lymphoma"]

	no_gr_tissues = ["Lung", "Esophagus", "Head and Neck", "Peripheral Nervous System", "Leukemia"]
	models = all_screened_SIDMs(project_list=None, integrated=True)

	growth_rate = get_growth()
	growth_rate = growth_rate[growth_rate.model_id.isin(models)].reset_index()[["model_id", "day4_day1_ratio"]]
	growth_rate["tissue"] = growth_rate.apply(lambda x: CellLine(sanger2model(x.model_id)).get_tissue_type(), axis=1)

	growth_rate=growth_rate[growth_rate.tissue.isin(tissues)]
	growth_rate = growth_rate.sort_values(by=["tissue"])
	growth_rate.to_csv(output_path + "biomarker/covariates/tissue_growth_rates.csv")

	tissue_ranked = growth_rate.tissue.unique()

	median_df = pandas.DataFrame(index=tissues, columns=["Median", "Max", "Min", "sd"])
	for t in tissues:
		x = growth_rate[growth_rate.tissue == t]["day4_day1_ratio"]
		median_df.loc[t, "Median"] = x.median()
		median_df.loc[t, "Max"] = x.max()
		median_df.loc[t, "Min"] = x.min()
		median_df.loc[t, "sd"] = x.std()

	median_df.to_csv(output_path + "biomarker/covariates/growth_rate_median_sd.csv", index=True)

	#pal = sns.color_palette(palette='coolwarm', n_colors=17)

	g = sns.FacetGrid(growth_rate, row="tissue", aspect=15, height=0.75)
	g.map(sns.kdeplot, "day4_day1_ratio", bw_adjust=1, clip_on=False, fill=True, alpha=0.6,
		  linewidth=1.5,color="navy")
	# Contour between plots
	g.map(sns.kdeplot, 'day4_day1_ratio', bw_adjust=1, clip_on=False, color="w", lw=2)
	g.map(plt.axhline, y=0, lw=2, clip_on=False, color= "darkgrey")
	g.map(plt.axvline, x=2.5, lw=2, linestyle=":",  clip_on=False, color="darkgrey")

	for i, ax in enumerate(g.axes.flat):
		ax.text(12, 0.02, tissue_ranked[i], fontweight='bold', fontsize=12, color="black")
		ax.set_ylabel("")

	g.fig.subplots_adjust(hspace=0)
	g.set_titles("")
	g.set(yticks=[])
	g.despine(bottom=True, left=True)

	plt.setp(ax.get_yticklabels(), fontsize=0)
	plt.setp(ax.get_xticklabels(), fontsize=12)
	plt.xlabel("Growth Rate", fontsize=16)
	g.fig.suptitle("Growth Rate distributions of tissues", ha='center', fontsize=16,fontweight="bold")
	plt.tight_layout()
	plt.savefig(output_path + "biomarker/covariates/figures/growth_rate_distributions.pdf", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_rate_distributions.jpg", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_rate_distributions.png", dpi=300)
	plt.close()


	plt.figure(figsize=(12,10))
	ax = sns.stripplot(data=growth_rate, x="tissue", y="day4_day1_ratio", size=4,
					   color="navy", alpha=0.3, dodge=True)
	handles, labels = ax.get_legend_handles_labels()
	sns.violinplot(ax=ax, data=growth_rate, x="tissue", y="day4_day1_ratio",color="white", alpha=0)

	ax.legend(handles, labels, loc=4)

	plt.xlabel("Tissues", fontsize=16)
	plt.xticks(fontsize=10, rotation=90)
	plt.ylabel("Growth Rate", fontsize=16)
	plt.yticks(fontsize=10)
	plt.title("Growth Rate distributions of tissues", ha='center', fontsize=16,fontweight="bold")
	plt.tight_layout()
	plt.savefig(output_path + "biomarker/covariates/figures/growth_rate_distributions_box.pdf", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_rate_distributions_box.jpg", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_rate_distributions_box.png", dpi=300)
	plt.close()

	return growth_rate


def check_growth_properties():

	tissues = ["Large Intestine", "Ovary", "Stomach", "Breast", "Skin", "Lung", "Pancreas", "Bone",
			   "Head and Neck", "Esophagus", "Bladder", "Peripheral Nervous System", "Kidney",
			   "Soft Tissue", "Central Nervous System", "Leukemia", "Lymphoma"]

	model_info = cancer_models()
	media_type = model_info[["sanger_id", "growth_properties"]].drop_duplicates()

	media_type["tissue"] = None
	model_tissue = dict()
	for t in tissues:
		cls = get_sidm_tissue(tissue_type=t)
		model_tissue[t] = cls
		media_type.loc[list(media_type[media_type.sanger_id.isin(cls)].index), "tissue"] = t
	media_type=media_type[media_type.tissue.isin(tissues)]
	media_type = media_type.sort_values(by=["tissue"])
	media_freq_df = pandas.DataFrame(media_type.groupby(["tissue", "growth_properties"]).size()).reset_index()
	media_freq_df.columns = ["tissue", "growth_properties", "number_of_models"]
	media_freq_df["percent"] = 100 * media_freq_df['number_of_models'] / media_freq_df.groupby('tissue')[
		'number_of_models'].transform('sum')

	media_freq_df.to_csv(output_path + "biomarker/covariates/media_type_counts_percents.csv")

	hex_dict = dict()
	for name, hex in matplotlib.colors.cnames.items():
		hex_dict[name] = hex

	custom_palette = {"Unknown": hex_dict["grey"],
					  "Adherent": hex_dict["salmon"],
					  "Semi-Adherent": hex_dict["lightsalmon"],
					  "Suspension": hex_dict["lightseagreen"]}
	plt.figure(figsize=(10,8))
	ax = sns.barplot(data=media_freq_df, x="tissue", y="number_of_models", hue="growth_properties",
					 palette=custom_palette)
	ax.legend(title="Growth Property", loc=1)
	plt.xlabel("Tissues", fontsize=14)
	plt.xticks(fontsize=10, rotation=90)
	plt.ylabel("Number of Cell Lines", fontsize=14)
	plt.yticks(fontsize=10)
	plt.title("Growth Property Landscape", ha='center', fontsize=16,fontweight="bold")
	plt.tight_layout()
	plt.savefig(output_path + "biomarker/covariates/figures/growth_property_frequency.pdf", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_property_frequency.jpg", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_property_frequency.png", dpi=300)
	plt.close()

	plt.figure(figsize=(10,8))
	ax = sns.barplot(data=media_freq_df, x="tissue", y="percent", hue="growth_properties",
					 palette=custom_palette)
	ax.legend(title="Growth Property", loc=1)
	plt.xlabel("Tissues", fontsize=14)
	plt.xticks(fontsize=10, rotation=90)
	plt.ylabel("Percent of Cell Lines", fontsize=14)
	plt.yticks(fontsize=10)
	plt.title("Growth Property Landscape", ha='center', fontsize=16,fontweight="bold")
	plt.tight_layout()
	plt.savefig(output_path + "biomarker/covariates/figures/growth_property_percentage.pdf", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_property_percentage.jpg", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/growth_property_percentage.png", dpi=300)
	plt.close()


	return media_freq_df


def check_experiment_type():

	tissues = ["Large Intestine", "Ovary", "Stomach", "Breast", "Skin", "Lung", "Pancreas", "Bone",
			   "Head and Neck", "Esophagus", "Bladder", "Peripheral Nervous System", "Kidney",
			   "Soft Tissue", "Central Nervous System", "Leukemia", "Lymphoma"]

	combi = combine_combi(estimate_data="XMID", treatment="combination")

	screen = combi[["SIDM", "screen_type", "DrugComb", "tissue_type", "SYNERGY_XMID"]]
	screen = screen[screen.tissue_type.isin(tissues)]
	screen = screen.drop_duplicates()

	screen_freq_df = pandas.DataFrame(screen.groupby(["tissue_type", "screen_type"]).size()).reset_index()
	screen_freq_df.columns = ["tissue_type", "screen_type", "number_of_models"]
	screen_freq_df["percent"] = 100 * screen_freq_df['number_of_models'] / screen_freq_df.groupby('tissue_type')['number_of_models'].transform('sum')
	anchor_df = screen_freq_df[screen_freq_df.screen_type == "anchor"]
	all_df = screen_freq_df.copy()
	all_df["percent"] = 100
	all_df[all_df.screen_type == "matrix"]

	screen_freq_df.to_csv(output_path + "biomarker/covariates/screen_type_counts_percetages.csv")

	plt.figure(figsize=(8,8))
	ax1 = sns.barplot(data=all_df, x="tissue_type", y="percent", color='lightgrey')
	ax2 = sns.barplot(data=anchor_df, x="tissue_type", y="percent", color='navy')

	top_bar = mpatches.Patch(color='lightgrey', label="Matrix")
	bottom_bar = mpatches.Patch(color='navy', label="Acnhor")
	plt.legend(handles=[top_bar, bottom_bar], loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)

	plt.xlabel("")
	plt.xticks(fontsize=10, rotation=90)
	plt.ylabel("Percentage % of Experiment Format", fontsize=14)
	plt.yticks(fontsize=10)
	plt.title("Experiment format of the selected representative curve", ha='center', fontsize=16,fontweight="bold")
	plt.tight_layout()
	plt.savefig(output_path + "biomarker/covariates/figures/experiment_type_percent.pdf", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/experiment_type_percent.jpg", dpi=300)
	plt.savefig(output_path + "biomarker/covariates/figures/experiment_type_percent.png", dpi=300)
	plt.close()

	return screen_freq_df


def plt_experiment_type_response():

	combi = combine_combi(estimate_data="XMID", treatment="combination")
	screen = combi[["SIDM", "screen_type", "DrugComb", "tissue_type", "SYNERGY_XMID"]]


	tissues = ["Large Intestine", "Ovary", "Stomach", "Breast", "Skin", "Lung", "Pancreas", "Bone",
			   "Head and Neck", "Esophagus", "Bladder", "Peripheral Nervous System", "Kidney",
			   "Soft Tissue", "Central Nervous System", "Leukemia", "Lymphoma"]

	for tissue in tissues:
		df = screen[screen.tissue_type == tissue]

		mw_pval = scipy.stats.mannwhitneyu(df[df.screen_type=="anchor"]["SYNERGY_XMID"],
										   df[df.screen_type=="matrix"]["SYNERGY_XMID"]).pvalue
		formatted_aic_mw_pval = [f'%.2e' % pvalue for pvalue in [mw_pval]]

		plt.figure(figsize=(3,5))
		ax = sns.stripplot(data=df, x="screen_type", y="SYNERGY_XMID", size=2,
						   color="navy", alpha=0.4, dodge=True, legend=False)

		sns.violinplot(ax=ax, data=df, x="screen_type", y="SYNERGY_XMID", color="white",
					   alpha=0, legend=False)

		annotator = Annotator(ax=ax, pairs=[("anchor", "matrix")], data=df,
							  x="screen_type", y="SYNERGY_XMID")
		annotator.set_custom_annotations(formatted_aic_mw_pval)
		annotator.annotate()

		plt.xlabel("Experiment Type", fontsize=10)
		plt.xticks(fontsize=8)
		plt.ylabel("Scaled Combo IC50", fontsize=10)
		plt.yticks(fontsize=8)
		plt.tight_layout()
		plt.savefig(output_path + "biomarker/covariates/figures/%s_experiment_type_response.pdf" % tissue, dpi=300)
		plt.savefig(output_path + "biomarker/covariates/figures/%s_experiment_type_response.jpg" % tissue, dpi=300)
		plt.savefig(output_path + "biomarker/covariates/figures/%s_experiment_type_response.png" % tissue, dpi=300)
		plt.close()

	return True

