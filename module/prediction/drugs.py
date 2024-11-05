"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 7 November 2023
Last Update : 16 November 2023
Input : Drug
Output : Drug fingerprints and descriptors
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas
"""
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Chem import MACCSkeys
morgan = AllChem.GetMorganGenerator(radius=2)
"""
#from pysmiles import read_smiles
from CombDrug.module.path import output_path
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.drug import *
from CombDrug.module.network.drug_modulation import *

# ---------------------------------------------------------------------------#
#                                  Functions                                 #
# ---------------------------------------------------------------------------#

def d_smiles():
	if "smiles.csv" not in os.listdir(output_path + "prediction/drugs/"):
		drug_df = make_drug_df()[["drug_name", "canonical_smiles"]]
		drug_df = drug_df.drop_duplicates()
		drug_df= drug_df.set_index(["drug_name"])
		drug_df.to_csv(output_path + "prediction/drugs/smiles.csv", index=True)
	else:
		drug_df = pandas.read_csv(output_path + "prediction/drugs/smiles.csv", index_col=0)

	return drug_df



def d_fingerprint(stage, fingerprint):
	if "feature_%s_fingerprints_combo.csv" % fingerprint not in os.listdir(output_path + "prediction/drugs/"):

		all_drugs = all_screened_compounds(project_list=None, integrated=True)
		f_df = pandas.DataFrame(index=all_drugs, columns = ["FP_%d" % i for i in list(range(167))])

		for drug in all_drugs:
			drug_smiles = Drug(drug).canonical_smiles

			if pandas.isna(drug_smiles) is False:
				molecule = Chem.MolFromSmiles(drug_smiles)

				if fingerprint == "maccs":
					# Maccs
					maccs_fingerprint = MACCSkeys.GenMACCSKeys(molecule)
					maccs_fingerprint_bits = maccs_fingerprint.ToList()
					for i in range(len(maccs_fingerprint_bits)):
						f_df.loc[drug, "FP_%d" % i] = maccs_fingerprint_bits[i]

				elif fingerprint == "morgan":
					global morgan
				# Morgan
					morgan_fingerprint = morgan.GetFingerprint(molecule)
					morgan_fingerprint_bits = morgan_fingerprint.ToList()
					for i in range(len(morgan_fingerprint_bits)):
						f_df.loc[drug, "FP_%d" % i] = morgan_fingerprint_bits[i]

		f_df.to_csv(output_path + "prediction/drugs/feature_%s_fingerprints_mono.csv" % fingerprint, index=True)

		all_combs = all_screened_combinations(project_list=None, integrated=True)
		new_columns = ["drug%d_%s" % (i, c) for c in f_df.columns for i in [1, 2]]
		f_df2 = pandas.DataFrame(index=all_combs, columns=new_columns)

		for drugcomb in all_combs:
			drug1, drug2 = drugcomb.split("/")[0], drugcomb.split("/")[1]
			for c in f_df.columns:
				c_name1 = "drug1_" + c
				c_name2 = "drug2_" + c
				f_df2.loc[drugcomb, c_name1] = f_df.loc[drug1][c]
				f_df2.loc[drugcomb, c_name2] = f_df.loc[drug2][c]

		f_df2.to_csv(output_path + "prediction/drugs/feature_%s_fingerprints_combo.csv" % fingerprint, index=True)

	else:
		f_df = pandas.read_csv(output_path + "prediction/drugs/feature_%s_fingerprints_mono.csv" % fingerprint, index_col=0)
		f_df2 = pandas.read_csv(output_path + "prediction/drugs/feature_%s_fingerprints_combo.csv" % fingerprint, index_col=0)

	if stage == "mono":
		return f_df
	else:
		return f_df2




def d_drug_targets(stage):

	if "feature_%s_drug_targets.csv" % stage not in os.listdir(output_path + "prediction/drugs/"):

		targeted_drugs = dict()
		all_drugs = all_screened_compounds(project_list=None, integrated=True)
		for drug in all_drugs:
			targets = Drug(drug).targets
			if targets is not None:
				targeted_drugs[drug] = targets

		interactome = get_norm_interactome(interactome_name="intact", weight=True, filter_mi=0.4)
		nodes = ["DT_%s" % i for i in interactome.nodes]
		target_df = pandas.DataFrame(columns= nodes, index= targeted_drugs.keys())

		for drug in targeted_drugs:
			for c in target_df.columns:
				g = c.split("_")[1]
				if g in targeted_drugs[drug]:
					target_df.loc[drug, c] = 1
				else:
					target_df.loc[drug, c] = 0

		target_df.to_csv(output_path + "prediction/drugs/feature_mono_drug_targets.csv", index=True)

		if stage == "combo":
			drug_combinations = all_screened_combinations(project_list=None, integrated=True)
			comb_target_df = pandas.DataFrame(columns= nodes, index= drug_combinations)

			comb_avail = list()
			for drugcomb in drug_combinations:
				drug1, drug2 = drugcomb.split("/")[0], drugcomb.split("/")[1]
				if drug1 in target_df.index and drug2 in target_df.index:
					comb_avail.append(drugcomb)
					for c in comb_target_df.columns:
						comb_target_df.loc[drugcomb, c] = target_df.loc[drug1, c] + target_df.loc[drug2, c] if target_df.loc[drug1, c] + target_df.loc[drug2, c] <=1 else 1
				else:
					comb_target_df.loc[drugcomb, c] = 0

			# Take only combinations having modules
			comb_target_df = comb_target_df[comb_target_df.index.isin(comb_avail)]
			comb_target_df.to_csv(output_path + "prediction/drugs/feature_combo_drug_targets.csv", index=True)
			return comb_target_df
		else:
			return target_df
	else:
		if stage == "mono":
			target_df = pandas.read_csv(output_path + "prediction/drugs/feature_mono_drug_targets.csv", index_col=0)
		else:
			target_df = pandas.read_csv(output_path + "prediction/drugs/feature_combo_drug_targets.csv",
										index_col=0)
		return target_df


def d_module_probability(stage):

	if "feature_%s_module_probabilities.csv" % stage not in os.listdir(output_path + "prediction/drugs/"):
		modelled_drugs = get_drug_networks(graph_type="interactome", interactome_name="intact",
										   stage="mono", alpha=0.55, weight=True,
										   network_name="interactome", value_type=None, filter_mi=0.4,
										   random_tag=False, seed=None).keys()

		interactome = get_norm_interactome(interactome_name="intact", weight=True, filter_mi=0.4)
		nodes = ["MP_%s" % i for i in interactome.nodes]
		module_df = pandas.DataFrame(columns= nodes, index= modelled_drugs)
		for drug in modelled_drugs:
			if os.path.exists(output_path + "network/modelling/PPR/intact/drugs/empiric/%s/" % drug):
				try:
					df = pandas.read_csv(output_path + "network/modelling/PPR/intact/drugs/empiric/%s/"
													   "interactome/network/ppr_node_attributes_%s_55_empiric_weighted.csv"
										 % (drug, drug)).set_index(["node"])
				except FileNotFoundError:
					df = None

			if df is not None:
				for c in module_df.columns:
					g = c.split("_")[1]
					if g in df.index:
						module_df.loc[drug, c] = df.loc[g, "ppr_probability"]
					else:
						module_df.loc[drug, c] = 0
		module_df.to_csv(output_path + "prediction/drugs/feature_mono_module_probabilities.csv", index=True)

		if stage == "combo":
			drug_combinations = all_screened_combinations(project_list=None, integrated=True)
			comb_module_df = pandas.DataFrame(columns= nodes, index= drug_combinations)
			module_avail = list()
			for drugcomb in drug_combinations:
				drug1, drug2 = drugcomb.split("/")[0], drugcomb.split("/")[1]
				if drug1 in module_df.index and drug2 in module_df.index:
					module_avail.append(drugcomb)
					for c in comb_module_df.columns:
						comb_module_df.loc[drugcomb, c] = module_df.loc[drug1, c] + module_df.loc[drug2, c] if module_df.loc[drug1, c] + module_df.loc[drug2, c] <=1 else 1
				else:
					comb_module_df.loc[drugcomb, c] = 0

			# Take only combinations having modules
			comb_module_df = comb_module_df[comb_module_df.index.isin(module_avail)]
			comb_module_df.to_csv(output_path + "prediction/drugs/feature_combo_module_probabilities.csv", index=True)
			return comb_module_df
		else:
			return module_df
	else:
		if stage == "mono":
			module_df = pandas.read_csv(output_path + "prediction/drugs/feature_mono_module_probabilities.csv", index_col=0)
		else:
			module_df = pandas.read_csv(output_path + "prediction/drugs/feature_combo_module_probabilities.csv",
										index_col=0)
		return module_df



def d_adjacency_matrix(interactome):
	nodes = interactome.nodes
	adj_df = pandas.DataFrame(networkx.adjacency_matrix(interactome, weight="Weight").toarray(), index=nodes, columns=nodes)
	return adj_df



def d_module_similarity():
	if "feature_module_similarity.csv" not in os.listdir(output_path + "prediction/drugs/"):
		df = collect_drug_network_similarity(
			graph_type="interactome", interactome_name="intact", weight=True, alpha=0.55,
			network_name="interactome", value_type=None, random_tag=False, seed=None)[["similarity_score"]]

		df.to_csv(output_path + "prediction/drugs/feature_module_similarity.csv", index=True)

	else:
		df = pandas.read_csv(output_path + "prediction/drugs/feature_module_similarity.csv", index_col=0)

	return df


def d_module_overlap():
	if "feature_module_similarity.csv" not in os.listdir(output_path + "prediction/drugs/"):
		df = collect_drug_network_similarity(
			graph_type="interactome", interactome_name="intact", weight=True, alpha=0.55,
			network_name="interactome", value_type=None, random_tag=False, seed=None)[["jaccard_index", "overlap_coefficient"]]

		df["overlap"] = df.apply(lambda x: "identical" if x.jaccard_index == 1 and x.overlap_coefficient == 1 else (
			"subset" if x.jaccard_index < 1 and x.jaccard_index > 0 and x.overlap_coefficient == 1 else (
				"overlap" if x.jaccard_index < 1 and x.jaccard_index > 0 and x.overlap_coefficient < 1 else (
					"independent" if x.jaccard_index == 0 and x.overlap_coefficient == 0 else None))), axis=1)

		df.to_csv(output_path + "prediction/drugs/feature_module_overlap.csv", index=True)

	else:
		df = pandas.read_csv(output_path + "prediction/drugs/feature_module_overlap.csv", index_col=0)

	return df


def d_structure_similarity(fingerprint, similarity):
	"""
	Drug similarity calculation
	:return:
	"""

	if "feature_drug_structure_similarity.csv" not in os.listdir(output_path + "prediction/drugs/"):

		global morgan
		all_drugs = all_screened_compounds(project_list=None, integrated=True)

		df = pandas.DataFrame(columns=[ "maccs_tanimato", "maccs_dice", "morgan_tanimato", "morgan_dice"],
							  index = [sort_drug_pairs(i[0], i[1])["sorted_pair"] for i in list(itertools.permutations(all_drugs, 2))])

		for drug_pair in list(itertools.permutations(all_drugs, 2)):
			drug1, drug2 = drug_pair[0], drug_pair[1]
			drugs = sort_drug_pairs(drug1, drug2)["sorted_pair"]

			drug1_smiles, drug2_smiles = Drug(drug1).canonical_smiles, Drug(drug2).canonical_smiles

			if pandas.isna(drug1_smiles) is False and pandas.isna(drug2_smiles) is False:
				molecules = [Chem.MolFromSmiles(drug1_smiles), Chem.MolFromSmiles(drug2_smiles)]

				maccs_fingerprints = [MACCSkeys.GenMACCSKeys(molecule) for molecule in molecules]
				morgan_fingerprints = [morgan.GetFingerprint(molecule) for molecule in molecules]

				df.loc[drugs, "maccs_tanimato"] = DataStructs.TanimotoSimilarity(maccs_fingerprints[0], maccs_fingerprints[1])
				df.loc[drugs, "maccs_dice"] = DataStructs.DiceSimilarity(maccs_fingerprints[0], maccs_fingerprints[1])
				df.loc[drugs, "morgan_tanimato"] = DataStructs.TanimotoSimilarity(morgan_fingerprints[0], morgan_fingerprints[1])
				df.loc[drugs, "morgan_dice"] = DataStructs.DiceSimilarity(morgan_fingerprints[0], morgan_fingerprints[1])

		df.to_csv(output_path + "prediction/drugs/feature_drug_structure_similarity.csv", index=True)

	else:
		df = pandas.read_csv(output_path + "prediction/drugs/feature_drug_structure_similarity.csv", index_col=0)

	df = df[["%s_%s" % (fingerprint, similarity)]]

	return df


def get_3d_drug_network(drug):

	smiles = Drug(srug).canonical_smiles
	if smiles is not None:
		molecule = read_smiles(smiles)
		elements = networkx.get_node_attributes(molecule, name="element")

		# wirh rdKit
		mol = Chem.MolFromSmiles(smiles)
		adj_mat = Chem.GetAdjacencyMatrix(mol, useBO=True)
		G = networkx.from_numpy_array(adj_mat)

		# Get adjancency matrix
		adj_df = pandas.DataFrame(networkx.adjacency_matrix(molecule).A)
		adj_df.columns = elements.values()
		adj_df.index = elements.values()

	return adj_df, G



def x_drugs(stage, fingerprint_flag, fingerprint_type, target_flag, mod_prob_flag, mod_sim_flag, str_sim_flag, sim_type):
	"""
	:param stage: List of drug names
	:param fingerprint_flag: Boolean - If PubMed fingerprints will be used
	:param fingerprint_type: Type of fingerprint / MACCS or Morgan
	:param target_flag: Boolean - If drug targets will be used
	:param mod_prob_flag: Boolean - If drug module probabilities will be used
	:param mod_sim_flag: Boolean - If drug module similairty will be used
	:param str_sim_flag: Boolean - If drug structure similairty will be used
	:param sim_type: Type of structural similarity / Tanimato or Dice
	:return: Drug input for AI prediction
	"""

	if stage == "mono":
		drugs = all_screened_compounds(project_list=None, integrated=True)
	else:
		drugs = all_screened_combinations(project_list=None, integrated=True)

	drug_df = pandas.DataFrame(index=list(set(drugs)))
	num_drugs, cat_drugs = list(), list()
	if fingerprint_flag:
		df = d_fingerprint(stage=stage, fingerprint=fingerprint_type)
		drug_df = pandas.concat([drug_df, df], axis=1)
		num_drugs.extend(list(df.columns))

	if target_flag:
		df = d_drug_targets(stage=stage)
		drug_df = pandas.concat([drug_df, df], axis=1)
		num_drugs.extend(list(df.columns))

	if mod_prob_flag:
		df = d_module_probability(stage=stage)
		drug_df = pandas.concat([drug_df, df], axis=1)
		num_drugs.extend(list(df.columns))

	if mod_sim_flag:
		df = d_module_similarity()
		drug_df = pandas.concat([drug_df, df], axis=1)
		num_drugs.extend(list(df.columns))

	if str_sim_flag:
		df = d_structure_similarity(fingerprint=fingerprint_type, similarity=sim_type)
		inds = list(set(df.index).intersection(set(drug_df.index)))
		df2 = df.loc[inds].drop_duplicates()
		drug_df = pandas.concat([drug_df, df2], axis=1)
		num_drugs.extend(list(df.columns))

	drug_df = drug_df.loc[[drug for drug in drugs if drug in list(drug_df.index)]]
	drug_df = drug_df.reset_index()
	drug_df = drug_df.drop_duplicates()
	#drug_df = drug_df.dropna()
	drug_df = drug_df.set_index(["index"])

	return drug_df, num_drugs, cat_drugs



# ---------------------------------------------------------------------------#



def get_drug_structure_similarity_array(similarity_type):
	"""
	:param similarity_type:
	:return:
	"""

	if "drug_%s_similariy_array.csv" % similarity_type not in os.listdir(output_path + "network/statistics/drugs/"):
		df = get_drug_structure_similarity()
		df2 = df.dropna(axis=0)
		df2 = df2.drop_duplicates()

		x = pandas.DataFrame(index=all_drugs, columns=all_drugs)

		for ind, row in x.iterrows():
			for d in x.columns:
				drugs = sort_drug_pairs(ind, d)["sorted_pair"]
				if drugs in df2.index:
					t = df2.loc[drugs, similarity_type]
				else:
					t = numpy.nan
				if ind == d:
					t = 1
				x.loc[ind, d] = t
		x2 = x.astype(numpy.float32)
		all_drugs_wout_smiles = list()
		for d in all_drugs:
			if pandas.isna(Drug(d).canonical_smiles):
				all_drugs_wout_smiles.append(d)

		x2 = x2.drop(all_drugs_wout_smiles, axis=1)
		x2 = x2.drop(all_drugs_wout_smiles, axis=0)

		x2.to_csv(output_path + "network/statistics/drugs/drug_%s_similariy_array.csv"
				  % similarity_type, index=True)

	else:
		x2 = pandas.read_csv(output_path + "network/statistics/drugs/drug_%s_similariy_array.csv"
							 % similarity_type, index_col=0)

	return x2
