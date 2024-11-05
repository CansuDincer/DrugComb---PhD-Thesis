# ---------------------------------------------------------------------------#
#          		 C o m b D r u g - D R U G   R E S P O N S E  			     #
# ---------------------------------------------------------------------------#

"""
# ---------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 21 March 2023
Last Update : 13 May 2024
Output : Combined drug response
# ---------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy

from CombDrug.module.path import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.reproducibility import prepare_combi, get_all_combi
from CombDrug.module.data.cancer_model import *


# ---------------------------------------------------------------------------#
#        		 			      Combining		 			 		         #
# ---------------------------------------------------------------------------#

def combine_combi(estimate_data, treatment):
	"""
	Retrieving the rows having the highest synergy estimates
	:param estimate_data: The estimate that has been used in data filtration - XMID / EMAX
	:param treatment: Perturbation - mono / combination
	:return: Data Frame
	"""

	output_folder = output_path + "data/curves/combined_response/"
	if treatment == "combination":
		if "all_combo_combined_%s.csv" % estimate_data not in os.listdir(output_folder):

			# Take the most sensitive fit from each screen

			min_xmid_merged_combi = prepare_combi(stage="combo", anchor_dose=False, estimate_data="XMID",
												  m_type="min", subproject=True, del_all=False)
			min_xmid_merged_combi = min_xmid_merged_combi.reset_index()
			min_xmid_merged_combi = min_xmid_merged_combi.drop(["level_0", "index"], axis=1)
			max_indices = list(
				min_xmid_merged_combi.groupby(["SIDM", "DrugComb"])["SYNERGY_DELTA_%s" % estimate_data].idxmax())

			all_combi = min_xmid_merged_combi.loc[max_indices]
			all_combi = all_combi[[col for col in all_combi.columns if col != "index"]]
			all_combi = all_combi.drop_duplicates()

			all_combi["screen_type"] = all_combi.apply(
				lambda x: "anchor" if x.RESEARCH_PROJECT in anchor_projects else "matrix", axis=1)

			all_combi["tissue_type"]= None
			all_combi["pan_type"] = None
			for g, g_df in all_combi.groupby(["SIDM"]):
				inds = list(all_combi[all_combi.SIDM == g].index)
				obj = CellLine(sanger2model(g))
				all_combi.loc[inds, "tissue_type"] = obj.get_tissue_type()
				all_combi.loc[inds, "pan_type"] = obj.get_pan_type()

			all_combi.to_csv(output_folder + "all_combo_combined_%s.csv" % estimate_data, index=False)

		else:
			all_combi = pandas.read_csv(output_folder + "all_combo_combined_%s.csv" % estimate_data)
	else:
		if "all_mono_combined_%s.csv" % estimate_data not in os.listdir(output_folder):
			combi = get_all_combi()
			combi = combi.reset_index()
			combi = combi.drop(["Unnamed: 0"], axis=1)

			min_indices = list(combi.groupby(["SIDM", "library_name"])["LIBRARY_%s" % estimate_data].idxmin())
			all_combi = combi.loc[min_indices]
			all_combi = all_combi[[col for col in all_combi.columns if col != "index"]]
			all_combi = all_combi.drop_duplicates()
			all_combi["screen_type"] = all_combi.apply(
				lambda x: "anchor" if x.RESEARCH_PROJECT in anchor_projects else "matrix", axis=1)

			all_combi["tissue_type"]= None
			all_combi["pan_type"] = None
			for g, g_df in all_combi.groupby(["SIDM"]):
				inds = list(all_combi[all_combi.SIDM == g].index)
				obj = CellLine(sanger2model(g))
				all_combi.loc[inds, "tissue_type"] = obj.get_tissue_type()
				all_combi.loc[inds, "pan_type"] = obj.get_pan_type()

			all_combi.to_csv(output_folder + "all_mono_combined_%s.csv" % estimate_data, index=False)
		else:
			all_combi = pandas.read_csv(output_folder + "all_mono_combined_%s.csv" % estimate_data)

	return all_combi


def which_screen_synergy():
	"""
	Calculating the screen layout effect on synergy
	:return:
	"""
	combi = combine_combi(estimate_data="XMID", treatment="combination")
	project = pandas.get_dummies(combi["RESEARCH_PROJECT"])
	project_names = list(combi["RESEARCH_PROJECT"].unique())
	combi = pandas.concat([combi, project], axis=1)
	combi = combi.set_index(["SIDM"])
	combi = combi[["DrugComb"] + project_names]
	combi = combi.reset_index()
	combi = combi.drop_duplicates()
	combi = combi.set_index(["SIDM"])
	combi["screen"] = combi.apply(
		lambda x: "matrix" if (x["GDSC_010-B"] == 1) or (x["GDSC_007-B"] == 1) or (x["GDSC_007-A"] == 1) or
							  (x["GDSC_008-A"] == 1) or (x["GDSC_008-B"] == 1) or (x["GDSC_009-A"] == 1) or
							  (x["GDSC_009-B"] == 1)  else "anchor", axis=1)

	combi2 = combi.copy()
	anchor_dps = all_screened_combinations(project_list=project_name_p_layout["anchor"], integrated=False)
	matrix_dps = all_screened_combinations(project_list=project_name_p_layout["matrix"], integrated=False)
	combi2["layout_existance"] = combi2.apply(
		lambda x: True if x.DrugComb in anchor_dps and x.DrugComb in matrix_dps else False, axis=1)
	combi2 = combi2[combi2.layout_existance]

	print("# Anchor selected - the most synergistic curve: %d" % combi2.groupby(["screen"]).size()["anchor"])
	print("# Matrix selected - the most synergistic curve: %d" % combi2.groupby(["screen"]).size()["matrix"])

	return combi, combi2



# ---------------------------------------------------------------------------#
#                               Response Matrix                              #
# ---------------------------------------------------------------------------#

def get_tissue_types():
	response_df = combine_combi(estimate_data="XMID", treatment="mono")
	# Some HL cell lines are not leukemia or lymphoma or myeloma so other liquid --> they are in panliquid
	return [i for i in list(response_df["tissue_type"].unique()) if pandas.isna(i) == False]


def get_response_data(tissue, stage, estimate_lr):
	"""
	Retrieving the response data
	:param tissue: Which tissue the data will be separated into
	:param stage: LIBRARY / COMBO / DELTA
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:return: Response Data Frame
	"""

	print("\n	Collecting response data")

	if tissue is not None:
		title = "_" + "_".join(tissue.split(" "))
	else:
		title = ""

	if "%s_%s_response%s.csv" % (stage, estimate_lr, title) not in os.listdir(output_path + "biomarker/responses/"):
		if stage in ["combo", "delta"]:
			res_df = combine_combi(estimate_data="XMID", treatment="combination")

			if stage == "combo":
				col = "SYNERGY_%s" % estimate_lr
			elif stage == "delta":
				col = "SYNERGY_DELTA_%s" % estimate_lr

			if tissue in ["pansolid", "panliquid"]:
				response_df = res_df[["SIDM", "DrugComb", col, "pan_type"]]
				response_df.columns = ["SIDM", "DrugComb", "response", "group"]
				response_df = response_df.set_index("SIDM")
				response_df = response_df[response_df.group == tissue]
				response_df = response_df[[col for col in response_df.columns if col != "group"]]
			elif tissue is not None:
				response_df = res_df[["SIDM", "DrugComb", col, "tissue_type"]]
				response_df.columns = ["SIDM", "DrugComb", "response", "group"]
				response_df = response_df.set_index("SIDM")
				response_df = response_df[response_df.group == tissue]
				response_df = response_df[[col for col in response_df.columns if col != "group"]]
			else:
				response_df = res_df[["SIDM", "DrugComb", col]]
				response_df.columns = ["SIDM", "DrugComb", "response"]
				response_df = response_df.set_index("SIDM")

		elif stage == "mono":

			res_df = combine_combi(estimate_data="XMID", treatment="mono")

			col = "LIBRARY_%s" % estimate_lr

			if tissue in ["pansolid", "panliquid"]:
				response_df = res_df[["SIDM", "library_name", col, "pan_type"]]
				response_df.columns = ["SIDM", "library_name", "response", "group"]
				response_df = response_df.set_index("SIDM")
				response_df = response_df[response_df.group == tissue]
				response_df = response_df[[col for col in response_df.columns if col != "group"]]
			elif tissue is not None:
				response_df = res_df[["SIDM", "library_name", col, "tissue_type"]]
				response_df.columns = ["SIDM", "library_name", "response", "group"]
				response_df = response_df.set_index("SIDM")
				response_df = response_df[response_df.group == tissue]
				response_df = response_df[[col for col in response_df.columns if col != "group"]]
			else:
				response_df = res_df[["SIDM", "library_name", col]]
				response_df.columns = ["SIDM", "library_name", "response"]
				response_df = response_df.set_index("SIDM")

		response_df.to_csv(output_path + "biomarker/responses/%s_%s_response%s.csv"
						   % (stage, estimate_lr, title), index=True)

	else:
		response_df = pandas.read_csv(output_path + "biomarker/responses/%s_%s_response%s.csv"
									  % (stage, estimate_lr, title), index_col=0)

	return response_df


def get_sidm_tissue(tissue_type):

	if "sidms_%s.p" % tissue_type not in os.listdir(output_path + "data/objects/"):
		models = all_screened_SIDMs(project_list = None, integrated=True)
		response_df = combine_combi(estimate_data="XMID", treatment="mono")
		if tissue_type is None:
			sidms = response_df.SIDM.unique()
		elif tissue_type not in ["pansolid", "panliquid"]:
			sidms = response_df[response_df.tissue_type == tissue_type].SIDM.unique()
		else:
			sidms = response_df[response_df.pan_type == tissue_type].SIDM.unique()

		tissue_sidms = list(set(sidms).intersection(set(models)))
		pickle.dump(tissue_sidms, open(output_path + "data/objects/sidms_%s.p" % tissue_type, "wb"))

	else:
		tissue_sidms = pickle.load(open(output_path + "data/objects/sidms_%s.p" % tissue_type, "rb"))

	return tissue_sidms


def get_models(tissue, stage, estimate_lr):
	"""
	Retrieving all the models in the input drug combination data
	:param stage: LIBRARY / COMBO / DELTA
	:param tissue: Which tissue the data will be separated into
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:return:
	"""
	all_models = get_response_data(tissue=tissue, stage=stage, estimate_lr=estimate_lr).index.unique()
	return all_models


def get_compounds(model_id, estimate_lr):
	"""
	Retrieving drug have been screened on specified cancer cell line
	:param model_id: The SANGER ID of the cancer cell line
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:return:
	"""
	compounds = list(get_response_data(tissue=None, stage="mono", estimate_lr=estimate_lr).loc[model_id].library_name.unique())
	return compounds


def get_targeted_compounds(model_id, estimate_lr):
	"""
	Retrieving targeted drug have been screened on specified cancer cell line
	:param model_id: The SANGER ID of the cancer cell line
	:param estimate_lr: The estimate that will be used in the LR response - XMID / EMAX
	:return:
	"""
	compounds = list(get_response_data(tissue=None, stage="mono", estimate_lr=estimate_lr).loc[model_id].library_name.unique())
	targeted = list()
	for drug in compounds:
		if "targeted" in Drug(drug).drug_type or "Targeted" in Drug(drug).drug_type:
			targeted.append(drug)
	return targeted


def get_targeted_combinations(model_id):
	"""
	Retrieving targeted drug combinations have been screened on specified cancer cell line
	:param model_id: The SANGER ID of the cancer cell line
	:return:
	"""
	compounds = list(get_response_data(tissue=None, stage="combo", estimate_lr="XMID").loc[model_id].DrugComb.unique())
	targeted = list()
	for drug_comb in compounds:
		if Drug(drug_comb.split("/")[0]).drug_type is not None and Drug(drug_comb.split("/")[1]).drug_type is not None:
			if "targeted" in Drug(drug_comb.split("/")[0]).drug_type or "Targeted" in Drug(drug_comb.split("/")[0]).drug_type:
				if "targeted" in Drug(drug_comb.split("/")[1]).drug_type or "Targeted" in Drug(drug_comb.split("/")[1]).drug_type:
					targeted.append(drug_comb)
	return targeted


# ---------------------------------------------------------------------------#
#                             Drugs without XMID                             #
# ---------------------------------------------------------------------------#

def check_mono_XMID():

	if "SIDM_Drug_group_wout_mono.p" not in os.listdir(output_path + "drug_info/"):
		xmid_possible = list()
		xmid_not_possible = list()

		combo_response_df = get_response_data(tissue=None, stage="combo", estimate_lr="XMID")
		mono_response_df = get_response_data(tissue=None, stage="mono", estimate_lr="XMID")

		mono_viability_df = get_all_viabilities(treatment="mono")

		mono_xmid_groups = mono_response_df.groupby(["SIDM", "library_name"]).groups.keys()
		for group, group_df in combo_response_df.groupby(["SIDM", "DrugComb"]):
			sidm = group[0]
			for drug in group[1].split("/"):
				if (sidm, drug) not in mono_xmid_groups:
					x = len(mono_viability_df[(mono_viability_df.SIDM == sidm) &
											  (mono_viability_df.Drug == drug)].library_dose.unique())
					if x > 2:
						xmid_possible.append((sidm, drug))
					else:
						xmid_not_possible.append((sidm, drug))

		ds = list()
		for i in list(set(xmid_not_possible)):
			if i[1] not in ds:
				ds.append(i[1])

		cls = list()
		for i in list(set(xmid_not_possible)):
			if i[0] not in cls:
				cls.append(i[0])

		drug_cls =dict()

		for i in list(set(xmid_not_possible)):
			if i[1] not in drug_cls.keys():
				drug_cls[i[1]] = [i[0]]
			else:
				if i[0] not in drug_cls[i[1]]:
					drug_cls[i[1]].append(i[0])

		pickle.dump(drug_cls, open(output_path + "drug_info/SIDM_Drug_group_wout_mono.p", "wb"))
	else:
		drug_cls = pickle.load(open(output_path + "drug_info/SIDM_Drug_group_wout_mono.p", "rb"))

	for i, l in d_cl.items():
		print(i, len(l))

	return drug_cls
