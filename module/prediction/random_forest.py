"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 8 May 2024
Last Update : 14 May 2024
Input : Features and Variables
Output : Random Forest Regressor
#------------------------------------------------------------------------#
"""
import os

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import pandas, argparse
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from sklearn.model_selection import *
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline
from sklearn.inspection import partial_dependence, plot_partial_dependence, permutation_importance

from CombDrug.module.path import output_path
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.omics import *
from CombDrug.module.data.responses import get_response_data
from CombDrug.module.prediction.features import x_omics_features
from CombDrug.module.prediction.drugs import x_drugs


# ---------------------------------------------------------------------------#
#                                    Input                                   #
# ---------------------------------------------------------------------------#

def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-run_for", dest="RUN_FOR")
	parser.add_argument("-stage", dest="STAGE")
	parser.add_argument("-kf", dest="FOLD", required=False)
	parser.add_argument("-njob", dest="NJOB", required=False)
	parser.add_argument("-niter", dest="NITER", required=False)
	parser.add_argument("-tuning", dest="TUNING", required=False)
	parser.add_argument("-gex", dest="GEX", action="store_true")
	parser.add_argument("-mut", dest="MUT", action="store_true")
	parser.add_argument("-pex", dest="PEX", action="store_true")
	parser.add_argument("-gain", dest="GAIN", action="store_true")
	parser.add_argument("-loss", dest="LOSS", action="store_true")
	parser.add_argument("-amp", dest="AMP", action="store_true")
	parser.add_argument("-del", dest="DEL", action="store_true")
	parser.add_argument("-met", dest="MET", action="store_true")
	parser.add_argument("-fp", dest="FP", action="store_true")
	parser.add_argument("-modsim", dest="MOD_SIM", action="store_true")
	parser.add_argument("-target", dest="TARGET", action="store_true")
	parser.add_argument("-modprob", dest="MOD_PROB", action="store_true")
	parser.add_argument("-strsim", dest="STR_SIM", action="store_true")
	parser.add_argument("-drop_cl", dest="DROP_CL", action="store_true")
	parser.add_argument("-drop_dc", dest="DROP_DC", action="store_true")
	parser.add_argument("-drop_dc_comp", dest="DROP_DC_COMPLETE", action="store_true")
	parser.add_argument("-frandom", dest="RANDOM", action="store_true")
	parser.add_argument("-lasso", dest="LASSO", action="store_true")
	parser.add_argument("-param", dest="PARAM", action="store_true")
	parser.add_argument("-cluster", dest="CLUSTER", action="store_true")
	parser.add_argument("-clustering_t", dest="THRESHOLD", required=False, default=0.7)
	parser.add_argument("-simt", dest="SIM_TYPE", required=False)
	parser.add_argument("-fpt", dest="FP_TYPE", required=False)
	parser.add_argument("-file", dest="FILE", required=False)
	parser.add_argument("-seed", dest="SEED", required=False)
	parser.add_argument("-shuffle", dest="SHUFFLE", action="store_true")

	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                                  Functions                                 #
# ---------------------------------------------------------------------------#

def hierarchical_clustering(clustered_features, data_name, threshold, file_title):
	"""
	Perform hierarchical clustering on Spearman rank-order correlations and
	select a single feature from each cluster based on a given threshold.
	:param clustered_features: Feature data frame
	:param data_name: Name of the data type (Gene Expression/Protein Expression/CRISPR-KO)
	:param threshold: Threshold for clustering (default -0.7)
	:param file_title: File initials
	"""
	names = clustered_features.columns
	new_names = [i.split("_GEx")[0] for i in names]
	clustered_features.columns = new_names

	# Calculate correlation matrix
	correlation_matrix = clustered_features.corr(method="spearman")

	plt.figure(figsize=(10, 12))
	sns.clustermap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True, center=0,
				   linewidths=.5, yticklabels=True, xticklabels=True)
	plt.title("Correlation Matrix for %s" % data_name)
	plt.savefig(output_path + "prediction/random_forest/figures/%s_spearman_correlations.pdf" % file_title, dpi=300)
	plt.savefig(output_path + "prediction/random_forest/figures/%s_spearman_correlations.png" % file_title, dpi=300)
	plt.savefig(output_path + "prediction/random_forest/figures/%s_spearman_correlations.jpg" % file_title, dpi=300)
	plt.close()

	linkage = hierarchy.ward(correlation_matrix)

	plt.figure(figsize=(8, 8))
	hierarchy.set_link_color_palette(["orangered", "forestgreen"])
	dendrogram = hierarchy.dendrogram(linkage, labels=correlation_matrix.columns, orientation="top",
									  show_leaf_counts=True)
	hierarchy.set_link_color_palette(None)
	plt.tight_layout()
	plt.savefig(output_path + "prediction/random_forest/figures/%s_spearman_correlations_dendongrams.pdf" % file_title, dpi=300)
	plt.savefig(output_path + "prediction/random_forest/figures/%s_spearman_correlations_dendongrams.png" % file_title, dpi=300)
	plt.savefig(output_path + "prediction/random_forest/figures/%s_spearman_correlations_dendongrams.jpg" % file_title, dpi=300)
	plt.close()

	clusters = hierarchy.fcluster(linkage, threshold, criterion='distance')

	# Select a single feature from each cluster
	selected_features = []
	for cluster_id in range(1, max(clusters) + 1):
		cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
		selected_feature = correlation_matrix.index[cluster_indices[0]]
		selected_features.append(selected_feature)

	return selected_features



def x_drop_drug(stage, fully_random, is_complete, seed):

	if is_complete: text_complete = "_complete"
	else: text_complete = ""

	if fully_random is False:
		if "randomly_selected%s_drop_drugs_%s.npy" % (text_complete, seed) not in os.listdir(output_path + "prediction/drugs/"):

			# Drug with modules
			empiric_output_folder = "/nfs/team215_pipeline/Cansu/CombDrug/output/network/modelling/PPR/intact/drugs/"
			drug_w_modules = list()
			for d in os.listdir(empiric_output_folder + "empiric/"):
				p = empiric_output_folder + "empiric/%s/interactome/network/" % d
				for f in os.listdir(p):
					if f.split("_")[-3] == "55" and f.split("_")[1] == "network":
						df = pandas.read_csv(p + f)
						if len(df.index) > 0:
							drug = f.split("_")[2]
							drug_w_modules.append(drug)

			# Pathway target groups
			df = pandas.read_csv(output_path + "data/drug_info/manually_pathway_target_grouped_drugs.csv")

			selected_drugs = list()
			for ind, row in df.iterrows():
				drug_list = row.drug_list.split(";")
				if len(set(drug_w_modules).intersection(set(drug_list))) >= 4:
					selectable_drug_list = list(set(drug_w_modules).intersection(set(drug_list)))
					r_drug = selectable_drug_list[numpy.random.choice(len(selectable_drug_list), 1)[0]]
					selected_drugs.append(r_drug)

			numpy.save(output_path + "prediction/drugs/randomly_selected%s_drop_drugs_%s.npy" % (text_complete, seed), selected_drugs)
		else:
			selected_drugs = numpy.load(output_path + "prediction/drugs/randomly_selected%s_drop_drugs_%s.npy" % (text_complete, seed))
	else:
		if "full_randomly_selected_drop_drugs_%s.npy" % seed not in os.listdir(output_path + "prediction/drugs/"):

			# Drug with modules
			empiric_output_folder = "/nfs/team215_pipeline/Cansu/CombDrug/output/network/modelling/PPR/intact/drugs/"
			drug_w_modules = list()
			for d in os.listdir(empiric_output_folder + "empiric/"):
				p = empiric_output_folder + "empiric/%s/interactome/network/" % d
				for f in os.listdir(p):
					if f.split("_")[-3] == "55" and f.split("_")[1] == "network":
						df = pandas.read_csv(p + f)
						if len(df.index) > 0:
							drug = f.split("_")[2]
							drug_w_modules.append(drug)

			df = make_drug_df()[["drug_name"]].dropna().drop_duplicates().reset_index().drop(columns=["index"])
			drug_list = list(set(list(df.drug_name.unique())).intersection(set(drug_w_modules)))
			selected_drugs = drug_list[numpy.random.choice(len(drug_list), 10)[0]]
			numpy.save(output_path + "prediction/drugs/full_randomly_selected_drop_drugs_%s.npy" % seed, selected_drugs)
		else:
			selected_drugs = numpy.load(output_path + "prediction/drugs/full_randomly_selected_drop_drugs_%s.npy" % seed)

	if stage == "combo":
		if fully_random is False:
			file = "randomly_selected%s_drop_drugcomb_%s.npy" % (text_complete, seed)
			save_file = "randomly_selected%s_drop_drugcomb_%s.npy" % (text_complete, seed)
		else:
			file = "randomly_selected_drop_drugcomb_%s.npy" % seed
			save_file = "randomly_selected_drop_drugcomb_%s.npy" % seed

		if file not in os.listdir(output_path + "prediction/drugs/"):
			drugcomb_w_module = list()
			all_combs = all_screened_combinations(project_list=None, integrated=True)
			for comb in all_combs:
				d1, d2 = comb.split("/")[0], comb.split("/")[1]
				if d1 in drug_w_modules and d2 in drug_w_modules:
					drugcomb_w_module.append(comb)

			if is_complete:
				selected_combs = list()
				for i in list(itertools.combinations(selected_drugs, 2)):
					m = [i[0], i[1]]
					m.sort()
					t = "/".join([m[0], m[1]])
					selected_combs.append(t)
				selected_drugcombs = set(drugcomb_w_module).intersection(set(selected_combs))
			else:
				selected_drugcombs = list()
				for comb in drugcomb_w_module:
					if len(set(selected_drugs).intersection(set(comb.split("/")))) != 0:
						selected_drugcombs.append(comb)

			numpy.save(output_path + "prediction/drugs/" + save_file, selected_drugcombs)
		else:
			selected_drugcombs = numpy.load(output_path + "prediction/drugs/" + save_file)

		return selected_drugcombs
	else:
		return selected_drugs


def x_drugs_w_modules():

	empiric_output_folder = "/nfs/team215_pipeline/Cansu/CombDrug/output/network/modelling/PPR/intact/drugs/"
	drug_w_modules = list()
	for d in os.listdir(empiric_output_folder + "empiric/"):
		p = empiric_output_folder + "empiric/%s/interactome/network/" % d
		for f in os.listdir(p):
			if f.split("_")[-3] == "55" and f.split("_")[1] == "network":
				df = pandas.read_csv(p + f)
				if len(df.index) > 0:
					drug = f.split("_")[2]
					drug_w_modules.append(drug)

	drug_w_modules = list(set(drug_w_modules))
	drugcomb_w_module = list()
	all_combs = all_screened_combinations(project_list=None, integrated=True)
	for comb in all_combs:
		d1, d2 = comb.split("/")[0], comb.split("/")[1]
		if d1 in drug_w_modules and d2 in drug_w_modules:
			drugcomb_w_module.append(comb)
	drugcomb_w_module = list(set(drugcomb_w_module))

	return drugcomb_w_module


def x_drop_model():
	if "randomly_selected_drop_models.npy" not in os.listdir(output_path + "prediction/features/"):
		tissues = get_tissue_types()
		selected_models = list()
		for tissue in tissues:
			models = get_sidm_tissue(tissue_type=tissue)
			if len(models) > 20:
				r_model = [models[ind] for ind in numpy.random.choice(len(models), 1)]
				selected_models.extend(r_model)

		numpy.save(output_path + "prediction/features/randomly_selected_drop_models.npy", selected_models)
	else:
		selected_models = numpy.load(output_path + "prediction/features/randomly_selected_drop_models.npy")

	return selected_models



def x_whole_perturbation(stage):
	if stage == "mono":
		drug_col = "library_name"
	else:
		drug_col = "DrugComb"

	data_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/prediction/random_forest/prepared_data/"
	if "perturbation_%s_all.csv" % stage not in os.listdir(data_path):

		# Get drug responses
		response = get_response_data(tissue=None, stage=stage, estimate_lr="XMID")

		# Take sensitive and resistance (get rid of inbalance splitting)
		response["sensitive"] = response.apply(lambda x: 1 if x.response <= 9 else 0, axis=1)
		response["sensitive"] = response["sensitive"].astype('category')

		# Get cell line features
		omics_features, num_omics, cat_omics = x_omics_features(
			treatment=stage if stage == "mono" else "combination",tissue=None, tissue_flag=False, msi_flag=False,
			media_flag=False, growth_flag=False, mutation_gene_flag=False, mutation_mut_flag=False, gex_flag=True,
			pex_flag=False, met_flag=False, amp_flag=False, del_flag=False, gain_flag=True, loss_flag=True,
			clinical_flag=False)

		# Get drug fingerprints
		drug_features, num_drugs, cat_drugs = x_drugs(
			stage=stage, fingerprint_flag=True, fingerprint_type="morgan", mod_prob_flag=True, target_flag=True,
			mod_sim_flag=True, str_sim_flag=True, sim_type="dice")


		# Perturbation
		perturbation_df = response.reset_index()
		for c in omics_features.columns:
			perturbation_df[c] = None

		for c in drug_features.columns:
			perturbation_df[c] = None

		#perturbation_df = perturbation_df[perturbation_df.SIDM.isin(list(omics_features.index))]
		count, total = 0, len(perturbation_df.groupby(["SIDM"]).groups.keys()) + len(perturbation_df.groupby([drug_col]).groups.keys())
		for g, g_df in perturbation_df.groupby(["SIDM"]):
			for c in omics_features.columns:
				if g in omics_features.index:
					perturbation_df.loc[list(g_df.index), c] = omics_features.loc[g][c]
			count += 1
			print(count * 100.0 /total)

		for g, g_df in perturbation_df.groupby([drug_col]):
			inds = list(perturbation_df[perturbation_df[drug_col] == g].index)
			for c in drug_features.columns:
				if g in drug_features.index:
					perturbation_df.loc[inds, c] = drug_features.loc[g][c]
			count += 1
			print(count * 100.0 /total)

		perturbation_df = perturbation_df.reset_index().drop(["index"], axis=1)

		# Preparation
		perturbation_df.to_csv(data_path + "perturbation_%s_all.csv" % stage, index=True)

	else:
		perturbation_df = pandas.read_csv(data_path + "perturbation_%s_all.csv" % stage, index_col=0)

	return perturbation_df



def extract_columns(column, stage, mutation_gene_flag, mutation_mut_flag, gex_flag, pex_flag, met_flag,
					amp_flag, del_flag, gain_flag, loss_flag , fingerprint_flag, mod_sim_flag,
					mod_prob_flag, str_sim_flag, fingerprint_type, sim_type):

	selected = False
	if mutation_gene_flag:
		if len(re.findall("^\S+_M$", column)) > 0:
			selected = True

	elif mutation_mut_flag:
		if len(re.findall("^\S+_dM$", column)) > 0:
			selected = True

	elif gex_flag:
		if len(re.findall("^\S+_GEx$", column)) > 0:
			selected = True

	elif pex_flag:
		if len(re.findall("^\S+_PEx$", column)) > 0:
			selected = True

	elif met_flag:
		if len(re.findall("^\S+_MET$", column)) > 0:
			selected = True

	elif amp_flag:
		if len(re.findall("^\S+_AMP$", column)) > 0:
			selected = True

	elif del_flag:
		if len(re.findall("^\S+_DEL$", column)) > 0:
			selected = True

	elif gain_flag:
		if len(re.findall("^\S+[-(+)]$", column)) > 0:
			selected = True

	elif loss_flag:
		if len(re.findall("^\S+[-(-)]$", column)) > 0:
			selected = True

	elif fingerprint_flag:
		if stage == "mono":
			if len(re.findall("^FP_[0-9]+$", column)) > 0:
				selected = True
		else:
			if len(re.findall("^Drug{1}[1,2]_FP_[0-9]+$", column)) > 0:
				selected = True

	elif target_flag:
		if len(re.findall("^DT[_]\S+$", column)) > 0:
			selected = True

	elif mod_prob_flag:
		if len(re.findall("^MP[_]\S+$", column)) > 0:
			selected = True

	elif mod_sim_flag:
		if column == "similarity_score":
			selected = True

	elif str_sim_flag:
		if column == "%s_%s" % (fingerprint_type, sim_type):
			selected = True
	return selected


def x_train_validation(stage, mutation_gene_flag, mutation_mut_flag, gex_flag, pex_flag, met_flag, amp_flag,
					   del_flag, gain_flag, loss_flag , fingerprint_flag, fingerprint_type, target_flag, mod_sim_flag,
					   mod_prob_flag, str_sim_flag, sim_type, drop_cl, drop_dc, is_complete, fully_random, file_title, seed):
	if stage == "mono":
		drug_col = "library_name"
	else:
		drug_col = "DrugComb"

	if drop_dc or drop_cl:
		if fully_random:
			drop_text = "_fully_random"
		else:
			drop_text = "_selectively_random"
	else: drop_text = ""

	if is_complete: text_complete = "_complete"
	else: text_complete = ""

	data_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/prediction/random_forest/prepared_data/"
	if "perturbation_%s.csv" % file_title not in os.listdir(data_path):

		perturbation_df = x_whole_perturbation(stage=stage)

		# Get cell line features
		omics_features, _, _ = x_omics_features(treatment=stage if stage == "mono" else "combination",tissue=None, tissue_flag=False, msi_flag=False,
												media_flag=False, growth_flag=False, mutation_gene_flag=mutation_gene_flag, mutation_mut_flag=mutation_mut_flag,
												gex_flag=gex_flag, pex_flag=pex_flag, met_flag=met_flag, amp_flag=amp_flag, del_flag=del_flag, gain_flag=gain_flag,
												loss_flag=loss_flag, clinical_flag=False)

		# Get drug fingerprints
		drug_features, _, _ = x_drugs(stage=stage, fingerprint_flag=fingerprint_flag, fingerprint_type=fingerprint_type, mod_prob_flag=mod_prob_flag,
									  mod_sim_flag=mod_sim_flag, str_sim_flag=str_sim_flag, sim_type=sim_type, target_flag=target_flag)

		selected_cols = ["SIDM", "DrugComb", "response", "sensitive"] + list(omics_features.columns) + list(drug_features.columns)

		perturbation_df = perturbation_df[selected_cols]
		perturbation_df = perturbation_df.reset_index().drop(["index"], axis=1).dropna().drop_duplicates()
		perturbation_df.to_csv(data_path + "perturbation_%s.csv" % file_title, index=True)

	else:
		perturbation_df = pandas.read_csv(data_path + "perturbation_%s.csv" % file_title, index_col=0)

	modules_drugcomb = x_drugs_w_modules()
	perturbation_df = perturbation_df[perturbation_df.DrugComb.isin(modules_drugcomb)]

	if drop_dc or drop_cl:
		if "train_test_perturbation_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed) not in os.listdir(data_path):

			if drop_dc and drop_cl is False:
				dropped_drugs = x_drop_drug(stage=stage, fully_random=fully_random, seed=seed, is_complete=is_complete)

				remaining_df = perturbation_df[~perturbation_df[drug_col].isin(dropped_drugs)]

				dropped_df = perturbation_df[perturbation_df[drug_col].isin(dropped_drugs)]

			elif drop_dc is False and drop_cl:
				dropped_models = x_drop_model()

				remaining_df = perturbation_df[~perturbation_df.SIDM.isin(dropped_models)]

				dropped_df = perturbation_df[perturbation_df.SIDM.isin(dropped_models)]

			elif drop_dc and drop_cl:
				dropped_drugs = x_drop_drug(stage=stage, fully_random=fully_random, is_complete=is_complete)
				dropped_models = x_drop_model()

				remaining_df = perturbation_df[(~perturbation_df.SIDM.isin(dropped_models)) &
											   (~perturbation_df[drug_col].isin(dropped_drugs))]

				dropped_df = perturbation_df[(perturbation_df.SIDM.isin(dropped_models)) &
											 (perturbation_df[drug_col].isin(dropped_drugs))]

			dropped_df.to_csv(data_path + "generalisation_perturbation_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed), index=True)

			remaining_df.to_csv(data_path + "train_test_perturbation_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed), index=True)

			sns.distplot(dropped_df["response"], kde=True, hist=False, rug=False, color="red", label ="Unseen Data")
			sns.distplot(remaining_df["response"], kde=True, hist=False, rug=False, color="navy", label = "Train/Test Data")
			plt.xlabel("Scaled XMID")
			plt.ylabel("Density")
			plt.title("Drug response distribution", ha='center', fontsize=12, fontweight="bold")
			plt.legend(loc="upper right")
			plt.tight_layout()
			plt.savefig(output_path + "prediction/random_forest/figures/seen_unseen_distribution_%s%s%s_%s.pdf" % (file_title, drop_text, text_complete, seed), dpi=300)
			plt.savefig(output_path + "prediction/random_forest/figures/seen_unseen_distribution_%s%s%s_%s.jpg" % (file_title, drop_text, text_complete, seed), dpi=300)
			plt.savefig(output_path + "prediction/random_forest/figures/seen_unseen_distribution_%s%s%s_%s.png" % (file_title, drop_text, text_complete, seed), dpi=300)
			plt.close()

		else:
			dropped_df = pandas.read_csv(data_path + "generalisation_perturbation_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed), index_col=0)

			remaining_df = pandas.read_csv(data_path + "train_test_perturbation_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed), index_col=0)
	else:
		remaining_df = perturbation_df.copy()
		dropped_df = None

	# Labels are the values we want to predict
	labels = numpy.array(remaining_df["response"])
	classes = pandas.Series(remaining_df["sensitive"])

	# Features are the values we want to predict from
	features = remaining_df.drop(["response", "sensitive"], axis=1)

	features = features.drop("SIDM", axis=1)
	features = features.drop(drug_col, axis=1)

	# Saving feature names for later use
	feature_list = list(features.columns)

	# Train and Test
	# Separate the validation set

	X = pandas.DataFrame(features, columns=feature_list)
	y = pandas.Series(labels)

	# Balanced split of train and final validation (20-80%)
	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size = 0.20, random_state = 0, stratify=classes)

	training_inx, test_inx = list(X_train.index), list(X_test.index)
	numpy.save(output_path + "prediction/random_forest/statistics/training_inx_%s%s%s_%s.npy" % (file_title, drop_text, text_complete, seed), training_inx)
	numpy.save(output_path + "prediction/random_forest/statistics/text_inx_%s%s%s_%s.npy" % (file_title, drop_text, text_complete, seed), test_inx)

	# Plotting the density of the responses
	dist_df1 = pandas.DataFrame()
	dist_df1["values"] = pandas.Series(y_train)
	dist_df1["type"] = "Train"
	dist_df2 = pandas.DataFrame()
	dist_df2["values"] = pandas.Series(y_test)
	dist_df2["type"] = "Test"

	dist_df = pandas.concat([dist_df1, dist_df2])
	sns.displot(data=dist_df, x="values", hue="type", kde=True, palette="vlag")
	plt.xlabel("Scaled XMID")
	plt.ylabel("Density")
	plt.title("Drug response distribution of Train and Test Sets", ha='center', fontsize=12, fontweight="bold")
	plt.tight_layout()
	plt.savefig(output_path + "prediction/random_forest/figures/train_test_split_distribution_%s%s%s_%s.pdf" % (file_title, drop_text, text_complete, seed), dpi=300)
	plt.savefig(output_path + "prediction/random_forest/figures/train_test_split_distribution_%s%s%s_%s.jpg" % (file_title, drop_text, text_complete, seed), dpi=300)
	plt.savefig(output_path + "prediction/random_forest/figures/train_test_split_distribution_%s%s%s_%s.png" % (file_title, drop_text, text_complete, seed), dpi=300)
	plt.close()

	return training_inx, test_inx, X_train, X_test, y_train, y_test, remaining_df, dropped_df


def x_RF(remaining_df, stage, training_indices, kfold, n_iter, plotting, file_title, n_jobs, tuning, drop, is_complete, fully_random, lasso_tag, param, seed, shuffle):

	if stage == "mono":
		drug_col = "library_name"
	else:
		drug_col = "DrugComb"

	if drop:
		if fully_random:
			drop_text = "_fully_random"
		else:
			drop_text = "_selectively_random"
	else:
		drop_text = ""

	if lasso_tag:
		lasso_text = "_lasso"
	else:
		lasso_text = ""

	if is_complete: text_complete = "_complete"
	else: text_complete = ""

	if "model_%s%s%s%s_%s.p" % (file_title, drop_text, text_complete, lasso_text, seed) not in os.listdir(output_path + "prediction/random_forest/statistics/"):
		# Get only the training sample
		df = remaining_df.loc[training_indices]

		# Save feature names
		feature_list = [c for c in df.columns if c not in ["response", "sensitive", "index", "SIDM", drug_col]]

		# Get features and responses
		X = df[feature_list].values
		y = df["response"].values

		# Cross Validation
		cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
		if lasso_tag:
			if "lasso_best_params_%s%s%s_%s.txt" % (file_title, drop_text, text_complete, seed) not in os.listdir(output_path + "prediction/random_forest/statistics/"):
				# Feature selection with Lasso (L1)
				lasso_params = {"alpha": numpy.linspace(0.00001, 10, 500)}

				# Initializing the Model
				lasso = Lasso()

				# GridSearchCV with model, params and folds.
				lasso_cv = GridSearchCV(lasso, param_grid=lasso_params, cv=cv)
				lasso_cv.fit(X, y)
				lasso_best_params = lasso_cv.best_params_
				lasso_parameter_results = pandas.DataFrame(lasso_cv.cv_results_)

				lasso_parameter_results.to_csv(
					output_path + "prediction/random_forest/statistics/lasso_parameters_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed), index=True)
				f = open(output_path + "prediction/random_forest/statistics/lasso_best_params_%s%s%s_%s.txt" % (file_title, drop_text, text_complete, seed), "w")
				for b, p in lasso_best_params.items():
					f.write(b + str(p) + "\n")
				f.close()

			else:
				f = open(output_path + "prediction/random_forest/statistics/lasso_best_params_%s%s%s_%s.txt" % (file_title, drop_text, text_complete, seed), "r")
				line = f.readlines()
				lasso_best_params = dict()
				lasso_best_params["alpha"] = float(line[0].strip().split("alpha")[1])
				f.close()

			if plotting:
				lasso_actual = Lasso(alpha=lasso_best_params["alpha"])
				lasso_actual.fit(X, y)
				lasso_actual_coef = numpy.abs(lasso_actual.coef_)
				lasso_df = pandas.DataFrame(lasso_actual_coef, index=feature_list, columns = ["Lasso_coeff"])
				lasso_df_sorted = lasso_df.sort_values(by="Lasso_coeff", ascending=False)
				lasso_df_sorted.to_csv(output_path + "prediction/random_forest/statistics/lasso_best_model_coefficients_%s%s%s_%s.csv" % (file_title, drop_text, text_complete, seed), index=True)

				lasso_df_sorted = lasso_df_sorted.reset_index()
				lasso_df_sorted.columns = ["Features", "Lasso_coeff"]

				selected_lasso_df = lasso_df_sorted[lasso_df_sorted.Lasso_coeff > 0.001]

				plt.bar(selected_lasso_df["Features"], selected_lasso_df["Lasso_coeff"])
				plt.xticks(rotation=90, fontsize=5)
				plt.title("Feature Selection Based on Lasso (coefficient > 0.001)")
				plt.xlabel("Features")
				plt.ylabel("Importance")
				plt.tight_layout()
				plt.savefig(output_path + "prediction/random_forest/figures/lasso_importances_%s%s%s_%s.pdf" % (file_title, drop_text, text_complete, seed), dpi=300)
				plt.savefig(output_path + "prediction/random_forest/figures/lasso_importances_%s%s%s_%s.jpg" % (file_title, drop_text, text_complete, seed), dpi=300)
				plt.savefig(output_path + "prediction/random_forest/figures/lasso_importances_%s%s%s_%s.png" % (file_title, drop_text, text_complete, seed), dpi=300)
				plt.close()

			# Select the important features

			selected_features = numpy.array(lasso_df_sorted[lasso_df_sorted.Lasso_coeff > 0.001]["Features"])
			numpy.save(output_path + "prediction/random_forest/statistics/lasso_best_selected_features_coeff_001_%s%s%s_%s.npy" % (file_title, drop_text, text_complete, seed), selected_features)

			# Get features and responses
			new_X = df[selected_features].values
			new_y = df["response"].values
		else:
			new_X = df[feature_list].values
			new_y = df["response"].values

		if param:
			if "best_params_%s_%s%s%s%s_%s.txt" % (file_title, tuning, drop_text, text_complete, lasso_text, seed) not in os.listdir(output_path + "prediction/random_forest/statistics/"):

				# Random Forest Regressor
				rf_regressor = RandomForestRegressor(random_state=0, oob_score=True, n_jobs=n_jobs, verbose=1)

				# Hyperparameter tuning

				param_distributions = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30],
									   "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 0.2],
									   "max_features": [0.25, 0.5,  1, "sqrt", "log2"],
									   "max_leaf_nodes": [2, 10, 50, None], "max_samples": [0.5, 0.8, None]}

				if tuning == "random":
					randomized_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_distributions,
														   n_iter=n_iter, cv=cv, scoring='neg_mean_squared_error',
														   random_state=0)
					randomized_search.fit(new_X, new_y)
					best_params = randomized_search.best_params_
					parameter_results = pandas.DataFrame(randomized_search.cv_results_)

					# Extract hyperparameters and performance metrics
					params = randomized_search.cv_results_['params']
					mean_test_scores = randomized_search.cv_results_['mean_test_score']

				elif tuning == "grid":
					grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_distributions, cv=cv,
											   scoring='neg_mean_squared_error')
					grid_search.fit(new_X, new_y)
					best_params = grid_search.best_params_
					parameter_results = pandas.DataFrame(grid_search.cv_results_)

					# Extract hyperparameters and performance metrics
					params = grid_search.cv_results_['params']
					mean_test_scores = grid_search.cv_results_['mean_test_score']


				numpy.save(output_path + "prediction/random_forest/statistics/params_%s_%s%s%s%s_%s.npy" % (file_title, tuning, drop_text,text_complete, lasso_text, seed), params)
				numpy.save(output_path + "prediction/random_forest/statistics/mean_test_scores_%s_%s%s%s%s_%s.npy" % (file_title, tuning, drop_text, text_complete, lasso_text, seed), mean_test_scores)
				parameter_results.to_csv(output_path + "prediction/random_forest/statistics/parameters_%s_%s%s%s%s_%s.csv" % (file_title, tuning, drop_text, text_complete, lasso_text, seed), index=True)
				f = open(output_path + "prediction/random_forest/statistics/best_params_%s_%s%s%s%s_%s.txt" % (file_title, tuning, drop_text, text_complete, lasso_text, seed), "w")
				for b, p in best_params.items():
					f.write(b + ":" + str(p) + "\n")
				f.close()

			else:
				params = numpy.load(output_path + "prediction/random_forest/statistics/params_%s_%s%s%s%s_%s.npy" % (file_title, tuning, drop_text, text_complete, lasso_text, seed), allow_pickle=True)
				mean_test_scores = numpy.load(output_path + "prediction/random_forest/statistics/mean_test_scores_%s_%s%s%s%s_%s.npy" % (file_title, tuning, drop_text, text_complete, lasso_text, seed), allow_pickle=True)

				best_params = dict()
				f = open(output_path + "prediction/random_forest/statistics/best_params_%s_%s%s%s%s_%s.txt" % (file_title, tuning, drop_text, text_complete, lasso_text, seed), "r")
				lines = f.readlines()
				for line in lines:
					t = line.strip().split(":")[1]
					if t is None or t == "None":
						tt = None
					else:
						print(line.strip().split(":")[1])
						tt = float(line.strip().split(":")[1])
					best_params[line.strip().split(":")[0]] = tt
				f.close()

			"""
			if plotting:
				# Plot performance metric against each hyperparameter
				fig, axs = plt.subplots(1, len(params[0]), figsize=(15, 5), sharey=True)
				for i, param in enumerate(params[0]):
					param_values = [params[j][param] for j in range(len(params))]
					print(param_values)
					print(mean_test_scores)
					axs[i].scatter(x=param_values, y=mean_test_scores, marker='o', color='b')
					axs[i].set_xlabel(param)
					axs[i].set_ylabel('Negative Mean Squared Error')
					axs[i].set_title(f'{param} vs. Performance')
				plt.tight_layout()
				plt.savefig(output_path + "prediction/random_forest/figures/parameters_%s_%s_%s%s.pdf" % (file_title, tuning, drop_text, lasso_text), dpi=300)
				plt.savefig(output_path + "prediction/random_forest/figures/parameters_%s_%s_%s%s.jpg" % (file_title, tuning, drop_text, lasso_text), dpi=300)
				plt.savefig(output_path + "prediction/random_forest/figures/parameters_%s_%s_%s%s.png" % (file_title, tuning, drop_text, lasso_text), dpi=300)
				plt.close()
			"""

			# Random Forest Regressor with the best parameters

			best_params_new = dict()
			for i,l in best_params.items():
				if i in ["max_features", "max_samples"]:
					if l is not None:
						best_params_new[i] = l
					else:
						best_params_new[i] = None
				else:
					if l is not None:
						best_params_new[i] = int(l)
					else:
						best_params_new[i] = None

		else:
			best_params_new = {"n_estimators": 200, "max_depth": None, "min_samples_split": 5,
							   "min_samples_leaf": 2, "max_features": 0.25,
							   "max_leaf_nodes": None, "max_samples": None}

		if shuffle:
			# Randomisation
			new_X = sklearn.utils.shuffle(new_X, random_state=int(seed))
			shuffle_text = "_shuffled_%s" % seed
		else:
			shuffle_text = ""

		if "cv_scores_%s%s%s%s%s_%s.txt" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed) not in os.listdir(output_path + "prediction/random_forest/statistics/"):
			best_rf_regressor = RandomForestRegressor(random_state=0, oob_score=True, n_jobs=n_jobs, verbose=1, **best_params_new)

			# CV on RFR
			cv_scores = cross_val_score(best_rf_regressor, new_X, new_y, cv=cv)
			mean_cv_score = cv_scores.mean()
			f = open(output_path + "prediction/random_forest/statistics/cv_scores_%s%s%s%s%s_%s.txt" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), "w")
			for i in cv_scores:
				f.write(str(i) + "\n")
			f.write(str(mean_cv_score))
			f.close()

		# Again a rf
		rf = RandomForestRegressor(random_state=0, oob_score=True, n_jobs=n_jobs, verbose=1, **best_params_new)

		# Train using all dataset
		rf.fit(new_X, new_y)

		# Feature importance
		if lasso_tag: new_X_df = df[selected_features]
		else: new_X_df = df[feature_list]

		feature_importance = rf.feature_importances_
		sorted_indices = feature_importance.argsort()
		sorted_features = new_X_df.columns[sorted_indices]
		sorted_importance = feature_importance[sorted_indices]
		feature_importance_df = pandas.DataFrame(sorted_importance)
		feature_importance_df["Features"] = sorted_features
		feature_importance_df.to_csv(output_path + "prediction/random_forest/statistics/feature_inportance_%s%s%s%s%s_%s.csv" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), index=False)

		if plotting:
			plt.figure(figsize=(10, 6))
			#fplt.barh(range(len(sorted_features)), sorted_importance, align='center')
			plt.barh(50, sorted_importance[:50], align='center')
			#plt.yticks(50, sorted_features[:50], fontsize=5)
			plt.xlabel('Feature Importance')
			plt.ylabel("")
			plt.title('Feature Importance with MDI (Top 50)')
			plt.savefig(output_path + "prediction/random_forest/figures/RF_model_evaluations_%s%s%s%s%s_%s.pdf" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
			plt.savefig(output_path + "prediction/random_forest/figures/RF_model_evaluations_%s%s%s%s%s_%s.png" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
			plt.savefig(output_path + "prediction/random_forest/figures/RF_model_evaluations_%s%s%s%s%s_%s.jpg" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
			plt.close()

		d = {"MODEL": rf, "best_parameters": best_params_new}
		pickle.dump(d, open(output_path + "prediction/random_forest/statistics/model_%s%s%s%s%s_%s.p" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), "wb"))

	else:
		d = pickle.load(open(output_path + "prediction/random_forest/statistics/model_%s%s%s%s%s_%s.p" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), "rb"))
		rf, best_params_new = d["MODEL"], d["best_parameters"]

	return rf, best_params_new



def test_RF(whole_df, test_indices, stage, model, file_title, drop, is_complete, fully_random, lasso_tag, seed, shuffle):

	if stage == "mono":
		drug_col = "library_name"
	else:
		drug_col = "DrugComb"

	if drop:
		if fully_random:
			drop_text = "_fully_random"
		else:
			drop_text = "_selectively_random"
	else:
		drop_text = ""

	if is_complete: text_complete = "_complete"
	else: text_complete = ""

	if lasso_tag:
		lasso_text = "_lasso"
		features = numpy.load(output_path + "prediction/random_forest/statistics/lasso_best_selected_features_coeff_001_%s%s%s_%s.npy" % (file_title, drop_text, text_complete, seed), allow_pickle=True)
	else:
		lasso_text = ""
		features = [c for c in whole_df.columns if c not in ["response", "sensitive", "index", "SIDM", drug_col]]

	if shuffle:
		shuffle_text = "_shuffled_%s" % seed
	else:
		shuffle_text = ""

	test_df = whole_df.loc[test_indices]
	test_features = test_df[features].values
	test_results = test_df["response"].values
	test_predictions = model.predict(test_features)

	# Test comparison df
	comparison_df = pandas.DataFrame(index= test_df["response"].index, columns=["empiric", "prediction"])
	comparison_df["empiric"], comparison_df["prediction"] = test_results, test_predictions
	comparison_df = pandas.concat([comparison_df, test_df[["SIDM", drug_col]]], axis=1)
	comparison_df.to_csv(output_path + "prediction/random_forest/statistics/test_comparison_%s%s%s%s%s_%s.csv" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed))

	pcc, p_val = pearsonr(test_results, test_predictions)
	rho, p_val_rho = spearmanr(test_results, test_predictions)

	# Test results
	f = open(output_path + "prediction/random_forest/statistics/test_RF_model_%s%s%s%s%s_%s.txt" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), "w")
	f.write("MSE: %.3f\n" % metrics.mean_squared_error(test_results, test_predictions, squared=True))
	f.write("RMSE: %.3f\n" % metrics.mean_squared_error(test_results, test_predictions, squared=False))
	f.write("R2: %.3f\n" % metrics.r2_score(test_results, test_predictions))
	f.write("PCC: %.3f\n" % pcc)
	f.write("PCC P-valus: %.3f\n" % p_val)
	f.write("ρ: %.3f\n" % rho)
	f.write("ρ P-valus: %.3f\n" % p_val_rho)
	f.close()

	fig, ax = plt.subplots(1, 1)

	onetoone = numpy.linspace(min(test_results), max(test_predictions), 10)
	ax.plot(onetoone, onetoone, label="1:1 Fit", linestyle="dotted", color="silver")
	ax.set_ylabel("Predicted IC50s")
	ax.set_xlabel("Empiric IC50s")
	ax.set_title("Validation of the RF Model\nRMSE: %.2f | $R^2$: %.3f\nPCC: %.3f | ρ: %.3f"
				 % (metrics.mean_squared_error(test_results, test_predictions, squared=False),
					metrics.r2_score(test_results, test_predictions), pcc, rho), fontsize=10)

	sns.scatterplot(ax=ax, data=comparison_df, x=comparison_df["empiric"],
					y=comparison_df["prediction"], alpha=0.6, color="navy")

	slope, intercept, r_value, p_value, _ = stats.linregress(test_results, test_predictions)
	ax.plot(test_results, intercept + slope * test_results, 'r',
			label=f'$y = {slope:.1f}x {intercept:+.1f}$')

	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
			  fancybox=True, shadow=True, ncol=20)

	fig.savefig(output_path + "prediction/random_forest/figures/test_RF_model_%s%s%s%s%s_%s.pdf" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
	fig.savefig(output_path + "prediction/random_forest/figures/test_RF_model_%s%s%s%s%s_%s.png" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
	fig.savefig(output_path + "prediction/random_forest/figures/test_RF_model_%s%s%s%s%s_%s.jpg" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
	plt.close()

	return comparison_df



def generalisation_RF(dropped_df, stage, model, file_title, fully_random, is_complete, lasso_tag, seed, shuffle):

	if stage == "mono":
		drug_col = "library_name"
	else:
		drug_col = "DrugComb"

	if fully_random:
		drop_text = "fully_random"
	else:
		drop_text = "selectively_random"

	if is_complete: text_complete = "_complete"
	else: text_complete = ""

	if lasso_tag:
		lasso_text = "_lasso"
		features = numpy.load(output_path + "prediction/random_forest/statistics/lasso_best_selected_features_coeff_001_%s%s%s_%s.npy" % (file_title, drop_text, text_complete, seed), allow_pickle=True)
	else:
		lasso_text = ""
		features = [c for c in dropped_df.columns if c not in ["response", "sensitive", "index", "SIDM", drug_col]]

	if shuffle:
		shuffle_text = "_shuffled_%s" % seed
	else:
		shuffle_text = ""

	test_features = dropped_df[features].values
	test_results = dropped_df["response"].values

	test_predictions = model.predict(test_features)

	# Test comparison df
	comparison_df = pandas.DataFrame(index= dropped_df["response"].index, columns=["empiric", "prediction"])
	comparison_df["empiric"], comparison_df["prediction"] = test_results, test_predictions
	comparison_df = pandas.concat([comparison_df, dropped_df[["SIDM", drug_col]]], axis=1)
	comparison_df.to_csv(output_path + "prediction/random_forest/statistics/generalisation_%s_%s%s%s%s_%s.csv" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed))

	pcc, p_val = pearsonr(test_results, test_predictions)
	rho, p_val_rho = spearmanr(test_results, test_predictions)

	# Test results
	f = open(output_path + "prediction/random_forest/statistics/generalisation_RF_model_%s_%s%s%s%s_%s.txt" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), "w")
	f.write("MSE: %.3f\n" % metrics.mean_squared_error(test_results, test_predictions, squared=True))
	f.write("RMSE: %.3f\n" % metrics.mean_squared_error(test_results, test_predictions, squared=False))
	f.write("R2: %.3f\n" % metrics.r2_score(test_results, test_predictions))
	f.write("PCC: %.3f\n" % pcc)
	f.write("PCC P-valus: %.3f\n" % p_val)
	f.write("ρ: %.3f\n" % rho)
	f.write("ρ P-valus: %.3f\n" % p_val_rho)
	f.close()

	fig, ax = plt.subplots(1, 1)

	onetoone = numpy.linspace(min(test_results), max(test_predictions), 10)
	ax.plot(onetoone, onetoone, label="1:1 Fit", linestyle="dotted", color="silver")
	ax.set_ylabel("Predicted IC50s")
	ax.set_xlabel("Empiric IC50s")
	ax.set_title("Generalisation of the RF Model\nRMSE: %.2f | $R^2$: %.3f\nPCC: %.3f | ρ: %.3f"
				 % (metrics.mean_squared_error(test_results, test_predictions, squared=False),
					metrics.r2_score(test_results, test_predictions), pcc, rho), fontsize=10)

	sns.scatterplot(ax=ax, data=comparison_df, x=comparison_df["empiric"],
					y=comparison_df["prediction"], alpha=0.6, color="navy")

	slope, intercept, r_value, p_value, _ = stats.linregress(test_results, test_predictions)
	ax.plot(test_results, intercept + slope * test_results, 'r',
			label=f'$y = {slope:.1f}x {intercept:+.1f}$')

	ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05),
			  fancybox=True, shadow=True, ncol=20)

	fig.savefig(output_path + "prediction/random_forest/figures/generalisation_RF_model_%s_%s%s%s%s_%s.pdf" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
	fig.savefig(output_path + "prediction/random_forest/figures/generalisation_RF_model_%s_%s%s%s%s_%s.png" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
	fig.savefig(output_path + "prediction/random_forest/figures/generalisation_RF_model_%s_%s%s%s%s_%s.jpg" % (file_title, drop_text, text_complete, lasso_text, shuffle_text, seed), dpi=300)
	plt.close()

	return comparison_df



def run_rf(stage, mutation_gene_flag, mutation_mut_flag, gex_flag, pex_flag, met_flag, amp_flag, del_flag, gain_flag,
		   loss_flag, fingerprint_flag, fingerprint_type, target_flag, mod_sim_flag, mod_prob_flag, str_sim_flag, sim_type,
		   file_title, drop_cl, drop_dc, is_complete, kfold, n_iter, n_jobs, tuning, fully_random, lasso_tag, param, seed, shuffle):


	training_inx, test_inx, _, _, _, _, remaining_df, dropped_df = x_train_validation(
		stage=stage, mutation_gene_flag=mutation_gene_flag, mutation_mut_flag=mutation_mut_flag,
		gex_flag=gex_flag, pex_flag=pex_flag, met_flag=met_flag, amp_flag=amp_flag, del_flag=del_flag,
		gain_flag=gain_flag, loss_flag=loss_flag, fingerprint_flag=fingerprint_flag, fingerprint_type=fingerprint_type,
		target_flag=target_flag, mod_sim_flag=mod_sim_flag, mod_prob_flag=mod_prob_flag, str_sim_flag=str_sim_flag, sim_type=sim_type,
		drop_cl=drop_cl, drop_dc=drop_dc, is_complete=is_complete, file_title=file_title, fully_random=fully_random, seed=seed)

	drop = True if drop_cl or drop_dc else False

	rf_model, best_params = x_RF(remaining_df=remaining_df, stage=stage, training_indices=training_inx,
								 kfold=kfold, tuning=tuning, n_iter=n_iter, plotting=True, param=param,
								 file_title=file_title, n_jobs=n_jobs, drop=drop, fully_random=fully_random,
								 lasso_tag=lasso_tag, seed=seed, is_complete=is_complete, shuffle=shuffle)

	_ = test_RF(whole_df=remaining_df, test_indices=test_inx, stage=stage, model=rf_model, file_title=file_title,
				drop=drop, is_complete=is_complete, fully_random=fully_random, lasso_tag=lasso_tag, seed=seed, shuffle=shuffle)

	if drop_dc or drop_cl:
		_ = generalisation_RF(dropped_df=dropped_df, stage=stage, model=rf_model, file_title=file_title, is_complete=is_complete,
							  fully_random=fully_random, lasso_tag=lasso_tag, seed=seed, shuffle=shuffle)

	return True


if __name__ == "__main__":
	args = take_input()

	if "DROP_CL" in args.keys():
		if args["DROP_CL"] in [True, "True"]:
			args["DROP_CL"] = True
		elif args["DROP_CL"] in [False, None, "False", "None"]:
			args["DROP_CL"] = False
	if "DROP_DC" in args.keys():
		if args["DROP_DC"] in [True, "True"]:
			args["DROP_DC"] = True
		elif args["DROP_DC"] in [False, None, "False", "None"]:
			args["DROP_DC"] = False

	if "DROP_DC_COMPLETE" in args.keys():
		if args["DROP_DC_COMPLETE"] in [True, "True"]:
			args["DROP_DC_COMPLETE"] = True
		elif args["DROP_DC_COMPLETE"] in [False, None, "False", "None"]:
			args["DROP_DC_COMPLETE"] = False

	if "SHUFFLE" in args.keys():
		if args["SHUFFLE"] in [True, "True"]:
			args["SHUFFLE"] = True
		elif args["SHUFFLE"] in [False, None, "False", "None"]:
			args["SHUFFLE"] = False

	if "RANDOM" in args.keys():
		if args["RANDOM"] in [True, "True"]:
			args["RANDOM"] = True
		elif args["RANDOM"] in [False, None, "False", "None"]:
			args["RANDOM"] = False

	if "LASSO" in args.keys():
		if args["LASSO"] in [True, "True"]:
			args["LASSO"] = True
		elif args["LASSO"] in [False, None, "False", "None"]:
			args["LASSO"] = False

	if "PARAM" in args.keys():
		if args["PARAM"] in [True, "True"]:
			args["PARAM"] = True
		elif args["PARAM"] in [False, None, "False", "None"]:
			args["PARAM"] = False

	args["FOLD"] = int(args["FOLD"]) if args["FOLD"] is not None else None
	args["NITER"] = int(args["NITER"]) if args["NITER"] is not None else None
	args["NJOB"] = int(args["NJOB"]) if args["NJOB"] is not None else None

	print(args)

	if args["RUN_FOR"] == "omics_preparation":

		_, _, _ = x_omics_features(treatment= "combination" if args["stage"] == "combo" else "mono", tissue=None,
								   tissue_flag=True, msi_flag=True, media_flag=True, growth_flag=True,
								   mutation_mut_flag=True, mutation_gene_flag=True, gex_flag=True, pex_flag=True,
								   met_flag=True, amp_flag=True, del_flag=True, gain_flag=True, loss_flag=True,
								   clinical_flag=True)

	elif args["RUN_FOR"] == "drug_preparation":

		_, _, _ = x_drugs(stage=args["STAGE"], fingerprint_flag=True, fingerprint_type="morgan", target_flag=True,
						  mod_prob_flag=True, mod_sim_flag=True, str_sim_flag=True, sim_type="dice")


	elif args["RUN_FOR"] == "perturbation_preparation":

		_ = x_whole_perturbation(stage=args["STAGE"])

	elif args["RUN_FOR"] == "RF":

		_ = run_rf(stage=args["STAGE"], mutation_gene_flag=args["MUT"],mutation_mut_flag=False, gex_flag=args["GEX"],
				   pex_flag=args["PEX"], met_flag=args["MET"], amp_flag=args["AMP"], del_flag=args["DEL"],
				   gain_flag=args["GAIN"], loss_flag=args["LOSS"], fingerprint_flag=args["FP"], fingerprint_type="morgan" if args["FP"] else None,
				   mod_sim_flag=args["MOD_SIM"], target_flag = args["TARGET"], mod_prob_flag=args["MOD_PROB"], str_sim_flag=args["STR_SIM"],
				   sim_type="dice" if args["STR_SIM"] else None, drop_cl=args["DROP_CL"], drop_dc=args["DROP_DC"], tuning=args["TUNING"],
				   file_title=args["FILE"], kfold=args["FOLD"], n_iter=args["NITER"], n_jobs=args["NJOB"], fully_random=args["RANDOM"],
				   lasso_tag=args["LASSO"], param=args["PARAM"], seed=args["SEED"], is_complete=args["DROP_DC_COMPLETE"], shuffle=args["SHUFFLE"])

