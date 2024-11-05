#---------------------------------------------------------------------------#
#    				  C o m b D r u g - D A T A  S E T   					#
#---------------------------------------------------------------------------#

"""
# ---------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 17 March 2022
Last Update : 21 March 2024
# ---------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, requests
import pickle
import itertools
from CombDrug.module.path import *
from CombDrug.module.data.cancer_model import *
"""
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

readRDS = robjects.r['readRDS']
readRData = robjects.r['load']
"""
# ---------------------------------------------------------------------------#
#                                    Paths                                   #
# ---------------------------------------------------------------------------#

anchor_projects = ["GDSC_002-A","GDSC_002-B", "GDSC_004-A", "GDSC_004-B", "GDSC_005-A",
                   "GDSC_005-B", "GDSC_Panc", "GDSC_Colo", "GDSC_Colo-2", "GDSC_Breast-2", "GDSC_Breast"]

pancancer_anchor_projects = ["GDSC_002-A","GDSC_002-B", "GDSC_004-A", "GDSC_004-B", "GDSC_005-A", "GDSC_005-B"]
percancer_anchor_projects = ["Panc", "Colo", "Breast"]

matrix_projects = ["GDSC_007-A", "GDSC_007-B", "GDSC_008-A", "GDSC_008-B",
				   "GDSC_009-A", "GDSC_009-B", "GDSC_010-B"]
pancancer_matrix_project = ["GDSC_007", "GDSC_008", "GDSC_009", "GDSC_010"]

curve_folder_name = {"pancancer": ["combo_fit_Pancan_combi_12Jan21"],
                     "breast": ["combo_fit_GDSC_Breast_09Sep21", "combo_fit_GDSC_Breast-2_09Sep21"],
                     "colo": ["combo_fit_GDSC_Colo_13Sep21", "combo_fit_GDSC_Colo-2_13Sep21"],
                     "pancreas": ["combo_fit_GDSC_Panc_09Sep21"],
                     "gdsc7": ["combo_fit_GDSC_007_to_AL_18Mar22"],
                     "gdsc8": ["combo_fit_GDSC_008_to_AL_18Mar22"],
                     "gdsc9": ["combo_fit_GDSC_009_to_AL_26Apr22"],
					 "gdsc10": ["combo_fit_GDSC_010-B_to_AL_22Mar22"]}

project_base_name = {"pancancer": "Pancan_combi_12Jan21", "breast": "GDSC_Breast_09Sep21",
                     "colo": "GDSC_Colo_13Sep21", "pancreas": "GDSC_Panc_09Sep21",
                     "gdsc7": "GDSC_007_to_AL_18Mar22",
					 "gdsc8": "GDSC_008_to_AL_18Mar22",
                     "gdsc9": "GDSC_009_to_AL_26Apr22",
					 "gdsc10": "GDSC_010-B_to_AL_22Mar22"}

project_title_name = {"breast": "Anchor Breast", "colo": "Anchor Colorectal", "pancreas": "Anchor Pancreas",
					  "pancancer": "Anchor Pan-cancer", "gdsc7": "Matrix GDSC-7", "gdsc8": "Matrix GDSC-8",
					  "gdsc9": "Matrix GDSC-9", "gdsc10": "Matrix GDSC-10"}

project_name_p_layout = {"anchor": ["pancancer", "breast", "colo", "pancreas"],
						 "matrix": ["gdsc7", "gdsc8", "gdsc9", "gdsc10"]}

project_pairs = [i for i in itertools.combinations(project_title_name.keys(), 2)]

pancancer_subgroups = pancancer_anchor_projects.copy()
pancancer_subproject_pairs = [i for i in itertools.combinations(pancancer_subgroups, 2)]

breast_subgroups = ["GDSC_Breast", "GDSC_Breast-2"]
breast_subproject_pairs = [i for i in itertools.combinations(breast_subgroups, 2)]

pancreas_subgroups = ["GDSC_Panc"]

colo_subgroups = ["GDSC_Colo", "GDSC_Colo-2"]
colo_subproject_pairs = [i for i in itertools.combinations(colo_subgroups, 2)]

gdsc_7_subgroups = ["GDSC_007-B", "GDSC_007-A"]
gdsc_7_subproject_pairs = [i for i in itertools.combinations(gdsc_7_subgroups, 2)]

gdsc_8_subgroups = ["GDSC_008-B", "GDSC_008-A"]
gdsc_8_subproject_pairs = [i for i in itertools.combinations(gdsc_8_subgroups, 2)]

gdsc_9_subgroups = ["GDSC_009-B", "GDSC_009-A"]
gdsc_9_subproject_pairs = [i for i in itertools.combinations(gdsc_9_subgroups, 2)]

gdsc_10_subgroups = ["GDSC_010-B"]


project_subgroup_dict = {"pancancer": pancancer_subgroups,
						 "breast": breast_subgroups, "colo": colo_subgroups,
						 "pancreas": pancreas_subgroups, "gdsc7": gdsc_7_subgroups,
						 "gdsc8": gdsc_8_subgroups, "gdsc9": gdsc_9_subgroups,
						 "gdsc10": gdsc_10_subgroups}


project_subgroup_pairs_dict = {"pancancer": pancancer_subproject_pairs,
							   "breast": breast_subproject_pairs, "colo": colo_subproject_pairs,
							   "pancreas": pancancer_subproject_pairs, "gdsc7": gdsc_7_subproject_pairs,
							   "gdsc8": gdsc_8_subproject_pairs, "gdsc9": gdsc_9_subproject_pairs, "gdsc10": None}


# ---------------------------------------------------------------------------#
#                                 Functions                                  #
# ---------------------------------------------------------------------------#


def sort_drug_pairs(anchor_name, library_name):

	dp = [anchor_name, library_name]
	dp_new = sorted(dp, key=str.casefold)
	if dp_new == dp: direction = "A-L"
	else: direction = "L-A"
	return {"sorted_pair": dp_new[0] + "/" +dp_new[1], "direction": direction}


# ---------------------------------------------------------------------------#
#                                   Files                                    #
# ---------------------------------------------------------------------------#


def get_combi(project):
	global project_base_name
	output_folder = output_path + "data/curves/combi_files/projects/"
	if project_base_name[project] + "_combi_df.csv" not in os.listdir(output_folder):
		if project in ["pancancer", "breast", "colo", "pancreas"]:
			if project not in ["breast", "colo"]:
				df = readRDS(curve_path + curve_folder_name[project][0] +
							 "/combiexplore_" +
							 "_".join(project_base_name[project].split("_")[:2]) + "_nlminb_" +
							 project_base_name[project].split("_")[-1] + ".rds")
				for i, l in df.items():
					if i == "combi":
						initial_combi_df = l

				initial_combi_df.to_csvfile(output_folder + project_base_name[project] + "_combi_initial_df.csv")

				combi_df_new = pandas.read_csv(output_folder + project_base_name[project] + "_combi_initial_df.csv")

			else:
				combi_dfs = list()
				for k in curve_folder_name[project]:
					df = readRDS(curve_path + k +
								 "/combiexplore_" + "_".join(k.split("_")[-3:-1]) + "_nlminb_" +
								 k.split("_")[-1] +  ".rds")
					for i, l in df.items():
						if i == "combi":
							l.to_csvfile(output_folder + project_base_name[project] + "_combi_initial_sub.csv")
							l2 = pandas.read_csv(output_folder + project_base_name[project] + "_combi_initial_sub.csv")
							os.system("rm %s*_combi_initial_sub.csv" % output_folder)
							combi_dfs.append(l2)

				initial_combi_df = pandas.concat(combi_dfs)

				initial_combi_df.to_csv(output_folder  + project_base_name[project] + "_combi_initial_df.csv")
				combi_df_new = initial_combi_df.copy()

		else:
			df = readRDS(curve_path + curve_folder_name[project][0] +
						 "/combiexplore_" +
						 "_".join(project_base_name[project].split("_")[:4]) + "_nlminb_" +
						 project_base_name[project].split("_")[-1] + ".rds")

			for i, l in df.items():
				if i == "combi":
					initial_combi_df = l

			initial_combi_df.to_csvfile(output_folder + project_base_name[project] + "_combi_initial_df.csv")

			combi_df_new = pandas.read_csv(output_folder + project_base_name[project] + "_combi_initial_df.csv")

		combi_df_new = combi_df_new.reset_index()
		combi_df_new = combi_df_new[[i for i in combi_df_new.columns if i != "index"]]

		# Remove specific combinations (AZ-specific)
		if project in project_name_p_layout["matrix"]:
			removed_inds = list(combi_df_new[combi_df_new.ANCHOR_ID.isin(removed_drug_ids_matrix) |
											 combi_df_new.LIBRARY_ID.isin(removed_drug_ids_matrix)].index)

			combi_df_new = combi_df_new.drop(index=removed_inds)

		# Remove >2 drug combinations
		combi_df_new["two_comb"] = True

		for comb, comb_df in combi_df_new.groupby(["ANCHOR_ID", "LIBRARY_ID"]):
			inds = combi_df_new[(combi_df_new.ANCHOR_ID == comb[0]) & (combi_df_new.LIBRARY_ID == comb[1])].index
			if len(str(comb[0]).split("|")) != 1 or len(str(comb[1]).split("|")) != 1:
				combi_df_new.loc[inds, "two_comb"] = False

		combi_df = combi_df_new[combi_df_new.two_comb]

		combi_df.to_csv(output_folder + project_base_name[project] + "_combi_df.csv")
		os.system("rm %s*_combi_initial_df.csv" % output_folder)
	else:
		combi_df = pandas.read_csv(output_folder + project_base_name[project] + "_combi_df.csv", low_memory=False, index_col=0)

	return combi_df


def get_synergy(project):

	output_folder = output_path + "data/curves/synergy_files/projects/"
	if project_base_name[project] + "_synergy_df.csv" not in os.listdir(output_path + "data/curves/synergy_files/"):
		if project in ["pancancer", "breast", "colo", "pancreas"]:
			if project not in ["breast", "colo"]:
				df = readRDS(curve_path + curve_folder_name[project][0] +
							 "/combiexplore_" +
							 "_".join(project_base_name[project].split("_")[:2]) + "_nlminb_" +
							 project_base_name[project].split("_")[-1] + ".rds")
				for i, l in df.items():
					if i == "synergy_calls":
						initial_synergy_df = l

				initial_synergy_df.to_csvfile(output_folder + project_base_name[project] + "_synergy_initial_df.csv")

				synergy_df_new = pandas.read_csv(output_folder + project_base_name[project] + "_synergy_initial_df.csv")

			else:
				synergy_dfs = list()
				for k in curve_folder_name[project]:
					df = readRDS(curve_path + k +
								 "/combiexplore_" + "_".join(k.split("_")[-3:-1]) + "_nlminb_" +
								 k.split("_")[-1] +  ".rds")
					for i, l in df.items():
						if i == "synergy_calls":
							l.to_csvfile(output_path + "curves/synergy_files/" + project_base_name[
								project] + "_synergy_initial_sub.csv")
							l2 = pandas.read_csv(
								output_path + "curves/synergy_files/" + project_base_name[
									project] + "_synergy_initial_sub.csv")
							os.system("rm *_synergy_initial_sub.csv" % output_path)
							synergy_dfs.append(l2)
				initial_synergy_df = pandas.concat(synergy_dfs)
				initial_synergy_df.to_csv(output_folder + project_base_name[project] + "_synergy_initial_df.csv")
				synergy_df_new = initial_synergy_df.copy()

		else:
			df = readRDS(curve_path + curve_folder_name[project][0] +
						 "/combiexplore_" +
						 "_".join(project_base_name[project].split("_")[:4]) + "_nlminb_" +
						 project_base_name[project].split("_")[-1] + ".rds")
			for i, l in df.items():
				if i == "synergy_calls":
					initial_synergy_df = l

			initial_synergy_df.to_csvfile(output_folder + project_base_name[project] + "_synergy_initial_df.csv")

			synergy_df_new = pandas.read_csv(output_folder + project_base_name[project] + "_synergy_initial_df.csv")


		synergy_df_new = synergy_df_new.reset_index()
		synergy_df_new = synergy_df_new[[i for i in synergy_df_new.columns if i != "index"]]

		if project in project_name_p_layout["matrix"]:

			removed_inds = list(synergy_df_new[synergy_df_new.ANCHOR_ID.isin(removed_drug_ids_matrix) |
											   synergy_df_new.LIBRARY_ID.isin(removed_drug_ids_matrix)].index)

			synergy_df_new = synergy_df_new.drop(index=removed_inds)

		# Remove >2 drug combinations
		synergy_df_new["two_comb"] = True

		for comb, comb_df in synergy_df_new.groupby(["ANCHOR_ID", "LIBRARY_ID"]):
			inds = synergy_df_new[(synergy_df_new.ANCHOR_ID == comb[0]) & (synergy_df_new.LIBRARY_ID == comb[1])].index
			if len(str(comb[0]).split(" | ")) != 1 or len(str(comb[1]).split(" | ")) != 1:
				synergy_df_new.loc[inds, "two_comb"] = False

		synergy_df = synergy_df_new[synergy_df_new.two_comb]

		synergy_df.to_csv(output_path + "data/curves/synergy_files/" + project_base_name[project] + "_synergy_df.csv")

	else:
		synergy_df = pandas.read_csv(output_path + "data/curves/synergy_files/" + project_base_name[project] + "_synergy_df.csv")

	return synergy_df


def get_viability(project):

	output_folder = output_path + "data/curves/viability_files/projects/"
	if project_base_name[project] + "_viability_df.csv" not in os.listdir(output_folder):
		if project in ["pancancer", "breast", "colo", "pancreas"]:
			if project not in ["breast", "colo"]:
				df = readRDS(curve_path + curve_folder_name[project][0] +
							 "/combiexplore_" +
							 "_".join(project_base_name[project].split("_")[:2]) + "_nlminb_" +
							 project_base_name[project].split("_")[-1] + ".rds")
				for i, l in df.items():
					if i == "viability":
						initial_viability_df = l

				initial_viability_df.to_csvfile(output_folder + project_base_name[project] + "_viability_initial_df.csv")

				viability_df_new = pandas.read_csv(output_folder + project_base_name[project] + "_viability_initial_df.csv", index_col=0)

			else:
				viability_dfs = list()
				for k in curve_folder_name[project]:
					df = readRDS(curve_path + k +
								 "/combiexplore_" + "_".join(k.split("_")[-3:-1]) + "_nlminb_" +
								 k.split("_")[-1] +  ".rds")
					for i, l in df.items():
						if i == "viability":
							l.to_csvfile(output_folder + project_base_name[project] + "_viability_initial_sub.csv")
							l2 = pandas.read_csv(output_folder + project_base_name[project] + "_viability_initial_sub.csv")
							os.system("rm %s*_viability_initial_sub.csv" % output_folder)
							viability_dfs.append(l2)

				initial_viability_df = pandas.concat(viability_dfs)
				initial_viability_df.to_csv(output_folder + project_base_name[project] + "_viability_initial_df.csv")
				viability_df_new = initial_viability_df.copy()

		else:
			df = readRDS(curve_path + curve_folder_name[project][0] +
						 "/combiexplore_" +
						 "_".join(project_base_name[project].split("_")[:4]) + "_nlminb_" +
						 project_base_name[project].split("_")[-1] + ".rds")
			for i, l in df.items():
				if i == "viability":
					initial_viability_df = l

			initial_viability_df.to_csvfile(output_folder + project_base_name[project] + "_viability_initial_df.csv")

			viability_df_new = pandas.read_csv(output_folder + project_base_name[project] + "_viability_initial_df.csv", index_col=0)

		viability_df_new = viability_df_new.reset_index()
		viability_df_new = viability_df_new[[i for i in viability_df_new.columns if i != "index"]]

		if project in project_name_p_layout["matrix"]:

			removed_inds = list(viability_df_new[viability_df_new.ANCHOR_ID.isin(removed_drug_ids_matrix) |
												 viability_df_new.LIBRARY_ID.isin(removed_drug_ids_matrix)].index)

			viability_df_new = viability_df_new.drop(index=removed_inds)

		# Remove >2 drug combinations
		viability_df_new["two_comb"] = True
		for comb, comb_df in viability_df_new.groupby(["ANCHOR_ID", "LIBRARY_ID"]):
			inds = viability_df_new[(viability_df_new.ANCHOR_ID == comb[0]) & (viability_df_new.LIBRARY_ID == comb[1])].index
			if len(str(comb[0]).split("|")) != 1 or len(str(comb[1]).split("|")) != 1:
				viability_df_new.loc[inds, "two_comb"] = False
		viability_df = viability_df_new[viability_df_new.two_comb]

		viability_df.to_csv(output_folder + project_base_name[project] + "_viability_df.csv")
		os.system("rm %s*_viability_initial_df.csv" % output_folder)
	else:
		viability_df = pandas.read_csv(output_folder + project_base_name[project] + "_viability_df.csv", index_col=0)

	return viability_df



# ---------------------------------------------------------------------------#
#                              Curated Files                                 #
# ---------------------------------------------------------------------------#

def annotate_viabilities(treatment, project):
	"""
	Concatenate all viability files
	:param treatment: Combination or single
	:param project
	"""

	output_folder = output_path + "data/curves/viability_files/%s/" % treatment
	if "annotated_%s_%s_viability_df.csv" % (project, treatment) not in os.listdir(output_folder):
		if project != "colo":
			df = get_viability(project)[["RESEARCH_PROJECT", "BARCODE", "DATE_CREATED",
										 "CELL_LINE_NAME", "maxc", "CELL_ID", "POSITION",
										 "treatment", "ANCHOR_ID", "LIBRARY_ID",
										 "ANCHOR_CONC", "LIBRARY_CONC", "VIAB_COMBI_UNCAPPED"]]
		else:
			df = get_viability(project)[["RESEARCH_PROJECT", "BARCODE", "DATE_CREATED","SCAN_DATE",
										 "CELL_LINE_NAME", "maxc", "CELL_ID", "POSITION",
										 "treatment", "ANCHOR_ID", "LIBRARY_ID",
										 "ANCHOR_CONC", "LIBRARY_CONC", "VIAB_COMBI_UNCAPPED"]]
			df2 = df[df.RESEARCH_PROJECT == "GDSC_Colo-2"]
			df1 = df[df.RESEARCH_PROJECT == "GDSC_Colo"]
			df1["DATE_CREATED"] = df1.apply(
				lambda x: "-".join(x.SCAN_DATE.split("T")[0].split("-")[:2]) + "-" +
						  str(int(x.SCAN_DATE.split("T")[0].split("-")[-1])-3), axis=1)

			df = pandas.concat([df1, df2])[["RESEARCH_PROJECT", "BARCODE", "DATE_CREATED",
											"CELL_LINE_NAME", "maxc", "CELL_ID", "POSITION",
											"treatment", "ANCHOR_ID", "LIBRARY_ID",
											"ANCHOR_CONC", "LIBRARY_CONC", "VIAB_COMBI_UNCAPPED"]]

		df = df.reset_index()
		df = df[[i for i in df.columns if i != "index"]]

		df["mono_drug"] = True
		for g, g_df in df.groupby(["ANCHOR_ID"]):
			if pandas.isna(g) is False and len(str(g).split("|")) > 1:
				df.loc[list(g_df.index), "mono_drug"] = False

		for g, g_df in df.groupby(["LIBRARY_ID"]):
			if pandas.isna(g) is False and len(str(g).split("|")) > 1:
				df.loc[list(g_df.index), "mono_drug"] = False

		df = df[df.mono_drug]
		df[["ANCHOR_ID", "LIBRARY_ID"]] = df[["ANCHOR_ID", "LIBRARY_ID"]].apply(pandas.to_numeric)

		# Remove the combinations/drugs which do not have a IC50 curve fitting
		combi_df = get_combi(project)
		combi_df["mono_drug"] = True
		for g, g_df in combi_df.groupby(["ANCHOR_ID"]):
			if pandas.isna(g) is False and len(str(g).split("|")) > 1:
				combi_df.loc[list(g_df.index), "mono_drug"] = False

		for g, g_df in combi_df.groupby(["LIBRARY_ID"]):
			if pandas.isna(g) is False and len(str(g).split("|")) > 1:
				combi_df.loc[list(g_df.index), "mono_drug"] = False

		combi_df = combi_df[combi_df.mono_drug]
		combi_df[["ANCHOR_ID", "LIBRARY_ID"]] = combi_df[["ANCHOR_ID", "LIBRARY_ID"]].apply(pandas.to_numeric)

		combi_pairs = list(combi_df.groupby(["ANCHOR_ID", "LIBRARY_ID"]).groups.keys())
		rev_combi_pairs = list(combi_df.groupby(["LIBRARY_ID", "ANCHOR_ID"]).groups.keys())
		all_combi_pairs = set(combi_pairs).union(set(rev_combi_pairs))

		via_pairs = list(df.groupby(["ANCHOR_ID", "LIBRARY_ID"]).groups.keys())
		rev_via_pairs = list(df.groupby(["LIBRARY_ID", "ANCHOR_ID"]).groups.keys())
		all_via_pairs = set(via_pairs).union(set(rev_via_pairs))

		missing_combs = [i for i in list(set(all_via_pairs).difference(set(all_combi_pairs))) if
						 pandas.isna(i[0]) is False and pandas.isna(i[1]) is False]
		missing_combs = list(set(missing_combs))

		combi_anchors = list(combi_df.groupby(["ANCHOR_ID"]).groups.keys())
		combi_libraries = list(combi_df.groupby(["LIBRARY_ID"]).groups.keys())
		all_combi_drugs = list(set(combi_anchors).union(set(combi_libraries)))

		via_anchors = list(df.groupby(["ANCHOR_ID"]).groups.keys())
		via_libraries = list(df.groupby(["LIBRARY_ID"]).groups.keys())
		all_via_drugs = list(set(via_anchors).union(set(via_libraries)))

		missing_drugs = [i for i in list(set(all_via_drugs).difference(set(all_combi_drugs))) if pandas.isna(i) is False]
		missing_drugs = list(set(missing_drugs))

		pickle.dump(missing_combs, open(output_path + "data/curves/combi_files/missing_drugs_combinations/missing_combs_%s.p"
										% project, "wb"))
		pickle.dump(missing_drugs, open(output_path + "data/curves/combi_files/missing_drugs_combinations/missing_drugs_%s.p"
										% project, "wb"))

		df["missing_combination"] = False
		for g, g_df in df.groupby(["ANCHOR_ID", "LIBRARY_ID"]):
			if g in missing_combs:
				df.loc[list(g_df.index), "missing_combination"] = True

		df = df[df.missing_combination == False]

		df["missing_drug"] = False
		for g, g_df in df[df.treatment == "S"].groupby(["ANCHOR_ID"]):
			if pandas.isna(g) is False and g in missing_drugs:
				df.loc[list(g_df.index), "missing_drug"] = True

		for g, g_df in df[df.treatment == "S"].groupby(["LIBRARY_ID"]):
			if pandas.isna(g) is False and g in missing_drugs:
				df.loc[list(g_df.index), "missing_drug"] = True

		df = df[df.missing_combination == False]

		df["SIDM"] = None
		for cl, cl_df in df.groupby(["CELL_LINE_NAME"]):
			inds = df[df.CELL_LINE_NAME == cl].index
			df.loc[inds, "SIDM"] = CellLine(cl).id

		if treatment == "combination":
			df = df[df.treatment == "C"]

			df["anchor_name"] = None
			for anchor, anchor_df in df.groupby(["ANCHOR_ID"]):
				df.loc[list(anchor_df.index), "anchor_name"] = drug_id2name(float(anchor))

			df["library_name"] = None
			for library, library_df in df.groupby(["LIBRARY_ID"]):
				df.loc[list(library_df.index), "library_name"] = drug_id2name(float(library))

			df["tissue"] = None
			for model, model_df in df.groupby(["CELL_LINE_NAME"]):
				if model in all_models():
					df.loc[list(model_df.index), "tissue"] = CellLine(model).tissue
				else:
					df.loc[list(model_df.index), "tissue"] = None

			df["barcode"] = df.apply(lambda x: int(x.BARCODE), axis=1)
			df["exp_date"] = df.apply(lambda x: str(x.DATE_CREATED), axis=1)
			df = df[[col for col in df.columns if col not in ["BARCODE", "DATE_CREATED"]]]

			df.rename(columns={"RESEARCH_PROJECT": "subproject", "barcode": "barcode",
							   "exp_date": "exp_date", "CELL_LINE_NAME": "cell_line",
							   "SIDM": "SIDM", "CELL_ID": "cell_id", "POSITION": "position",
							   "maxc": "maxc", "treatment": "treatment",
							   "ANCHOR_ID": "anchor_id", "LIBRARY_ID": "library_id",
							   "ANCHOR_CONC": "anchor_dose", "LIBRARY_CONC": "library_dose",
							   "VIAB_COMBI_UNCAPPED": "viability",
							   "anchor_name": "anchor_name", "library_name": "library_name",
							   "tissue": "tissue"}, inplace=True)

		else:
			df = df[df.treatment == "S"]

			anchor_df = df[~pandas.isna(df.ANCHOR_ID)]
			anchor_df["only_2_anchor"] = anchor_df.apply(
				lambda x: True if len(str(x.ANCHOR_ID).split("|")) == 1 else False, axis=1)
			anchor_df = anchor_df[anchor_df.only_2_anchor]

			library_df = df[~pandas.isna(df.LIBRARY_ID)]
			library_df["only_2_library"] = library_df.apply(
				lambda x: True if len(str(x.LIBRARY_ID).split("|")) == 1 else False, axis=1)
			library_df = library_df[library_df.only_2_library]

			anchor_df["library_name"] = None
			anchor_df["library_id"] = anchor_df.apply(lambda x: x.ANCHOR_ID, axis=1)
			anchor_df["library_dose"] = anchor_df.apply(lambda x: x.ANCHOR_CONC, axis=1)
			for anchor, a_df in anchor_df.groupby(["ANCHOR_ID"]):
				anchor_df.loc[list(a_df.index), "library_name"] = drug_id2name(float(anchor))

			library_df["library_name"] = None
			library_df["library_id"] = library_df.apply(lambda x: x.LIBRARY_ID, axis=1)
			library_df["library_dose"] = library_df.apply(lambda x: x.LIBRARY_CONC, axis=1)
			for library, l_df in library_df.groupby(["LIBRARY_ID"]):
				library_df.loc[list(l_df.index), "library_name"] = drug_id2name(float(library))

			df = pandas.concat([anchor_df, library_df], ignore_index=True)

			df["tissue"] = None
			for model, model_df in df.groupby(["CELL_LINE_NAME"]):
				if model in all_models():
					df.loc[list(model_df.index), "tissue"] = CellLine(model).tissue
				else:
					df.loc[list(model_df.index), "tissue"] = None

			df.rename(columns={"RESEARCH_PROJECT": "subproject", "BARCODE": "barcode",
							   "DATE_CREATED": "exp_date", "CELL_LINE_NAME": "cell_line",
							   "SIDM": "SIDM", "CELL_ID": "cell_id", "POSITION": "position",
							   "maxc": "maxc", "treatment": "treatment",
							   "library_id": "library_id", "library_dose": "library_dose",
							   "VIAB_COMBI_UNCAPPED": "viability",
							   "library_name": "library_name", "tissue": "tissue"}, inplace=True)

		df.to_csv(output_folder + "annotated_%s_%s_viability_df.csv" % (project, treatment), index=False)

	else:
		df = pandas.read_csv(output_folder + "annotated_%s_%s_viability_df.csv" % (project, treatment))

	return df


# ---------------------------------------------------------------------------#
#                                   Model                                    #
# ---------------------------------------------------------------------------#


def all_screened_models(project_list, integrated):
	"""
	List of screened cancer cell models
	:param project_list: The name of the project(s) in a list
	:param integrated: Usage of the integrated dataset (T/F)
	:return: A list of screened cancer cell models
	"""

	if "all_screen_models_project.p" not in os.listdir(output_path + "data/objects/"):
		models_dict = dict()
		models_list = list()
		for project in curve_folder_name.keys():
			combi = annotate_viabilities(treatment="combination", project=project)
			models_dict[project] = list(combi.cell_line.unique())
			models_list.extend(list(combi.cell_line.unique()))
		pickle.dump(models_dict, open(output_path + "data/objects/all_screen_models_project.p", "wb"))
		models_list = list(set(models_list))
		pickle.dump(models_list, open(output_path + "data/objects/all_screen_models.p", "wb"))

	else:
		models_dict = pickle.load(open(output_path + "data/objects/all_screen_models_project.p", "rb"))
		models_list = pickle.load(open(output_path + "data/objects/all_screen_models.p", "rb"))

	if integrated:
		return models_list

	else:
		all_models = list()
		for project, model_list in models_dict.items():
			if project in project_list:
				all_models.extend(model_list)

		return list(set(all_models))


def all_screened_SIDMs(project_list, integrated):
	"""
	List of screened cancer cell models
	:param project_list: The name of the project(s) in a list
	:param integrated: Usage of the integrated dataset (T/F)
	:return: A list of screened cancer cell models with Sanger IDs
	"""

	if "all_screen_SIDMs_project.p" not in os.listdir(output_path + "data/objects/"):
		sidm_d = dict()
		sidm_l = list()
		for project in curve_folder_name.keys():
			combi = annotate_viabilities(treatment="combination", project=project)
			sidms = list(combi.SIDM.unique())
			sidm_d[project] =sidms
			sidm_l.extend(sidms)
		pickle.dump(sidm_d, open(output_path + "data/objects/all_screen_SIDMs_project.p", "wb"))
		sidm_l = list(set(sidm_l))
		pickle.dump(sidm_l, open(output_path + "data/objects/all_screen_SIDMs.p", "wb"))

	else:
		sidm_l = pickle.load(open(output_path + "data/objects/all_screen_SIDMs.p", "rb"))
		sidm_d = pickle.load(open(output_path + "data/objects/all_screen_SIDMs_project.p", "rb"))

	if integrated:
		return sidm_l

	else:
		all_sidms = list()
		for project, sidm_ids in sidm_d.items():
			if project in project_list:
				all_sidms.extend(sidm_ids)

		return list(set(all_sidms))


def get_models_tissue(tissue):
	models = all_screened_models(project_list=None, integrated=True)
	df = cancer_models()
	df = df[df.tissue == tissue]
	df = df[df.model_name.isin(models)]
	return df.model_name.unique()


def get_sidm_tissue(tissue):
	models = all_screened_SIDMs(project_list=None, integrated=True)
	df = cancer_models()
	df = df[df.tissue == tissue]
	df = df[df.sanger_id.isin(models)]
	return df.sanger_id.unique()


# ---------------------------------------------------------------------------#
#                                Compounds                                   #
# ---------------------------------------------------------------------------#

def all_screened_compound_ids(project_list, integrated):
	"""
	List of screened compounds
	:param project_list: The name of the project(s) in a list
	:param integrated: Usage of the integrated dataset (T/F)
	:return: A list of screened compounds
	"""

	if "all_screen_compound_ids_project.p" not in os.listdir(output_path + "data/objects/"):
		compounds_dict = dict()
		compounds_list = list()
		for project in project_base_name.keys():
			combi = annotate_viabilities(treatment="combination", project=project)
			anchors = [i for i in list(combi.anchor_id.unique())]
			libraries = [i for i in list(combi.library_id.unique())]
			compounds = list(set(anchors).union(libraries))
			compounds_dict[project] = compounds
			compounds_list.extend(compounds)
			compounds_list = list(set(compounds_list))

		pickle.dump(compounds_dict, open(output_path + "data/objects/all_screen_compound_ids_project.p", "wb"))
		compounds_list = list(set(compounds_list))
		pickle.dump(compounds_list, open(output_path + "data/objects/all_screen_compound_ids.p", "wb"))

	else:
		compounds_dict = pickle.load(open(output_path + "data/objects/all_screen_compound_ids_project.p", "rb"))
		compounds_list = pickle.load(open(output_path + "data/objects/all_screen_compound_ids.p", "rb"))

	if integrated:
		return compounds_list
	else:
		all_compounds = list()
		for project, compound_list in compounds_dict.items():
			if project in project_list:
				all_compounds.extend(compound_list)

		return list(set(all_compounds))


def all_screened_compounds(project_list, integrated):
	"""
	List of screened compounds
	:param project_list: The name of the project(s) in a list
	:param integrated: Usage of the integrated dataset (T/F)
	:return: A list of screened compounds
	"""

	if "all_screen_compounds_project.p" not in os.listdir(output_path + "data/objects/"):
		compounds_dict = dict()
		compounds_list = list()
		for project in project_base_name.keys():
			combi = annotate_viabilities(treatment="combination", project=project)
			anchors = [i for i in list(combi.anchor_name.unique())]
			libraries = [i for i in list(combi.library_name.unique())]
			compounds = list(set(anchors).union(libraries))
			compounds_dict[project] = compounds
			compounds_list.extend(compounds)

		pickle.dump(compounds_dict, open(output_path + "data/objects/all_screen_compounds_project.p", "wb"))
		compounds_list = list(set(compounds_list))
		pickle.dump(compounds_list, open(output_path + "data/objects/all_screen_compounds.p", "wb"))

	else:
		compounds_dict = pickle.load(open(output_path + "data/objects/all_screen_compounds_project.p", "rb"))
		compounds_list = pickle.load(open(output_path + "data/objects/all_screen_compounds.p", "rb"))

	if integrated:
		return compounds_list
	else:
		all_compounds = list()
		for project, compound_list in compounds_dict.items():
			if project in project_list:
				all_compounds.extend(compound_list)

		return list(set(all_compounds))


def all_screened_combinations(project_list, integrated):
	"""
	List of screened combinations
	:param project_list: The name of the project(s) in a list
	:param integrated: Usage of the integrated dataset (T/F)
	:return: A list of screened combinations
	"""

	if "all_screen_drugcombs_project.p" not in os.listdir(output_path + "data/objects/"):
		combo_dict, combo_list = dict(), list()
		for project in project_base_name.keys():
			combi = annotate_viabilities(treatment="combination", project=project)
			combi["DrugComb"] = combi.apply(
				lambda x: sort_drug_pairs(x.anchor_name, x.library_name)["sorted_pair"], axis=1)
			drugcombs = list(combi.DrugComb.unique())
			print(drugcombs)
			combo_dict[project] = drugcombs
			combo_list.extend(drugcombs)
			combo_list = list(set(combo_list))

		pickle.dump(combo_dict, open(output_path + "data/objects/all_screen_drugcombs_project.p", "wb"))
		combo_list = list(set(combo_list))
		pickle.dump(combo_list, open(output_path + "data/objects/all_screen_drugcombs.p", "wb"))

	else:
		combo_dict = pickle.load(open(output_path + "data/objects/all_screen_drugcombs_project.p", "rb"))
		combo_list = pickle.load(open(output_path + "data/objects/all_screen_drugcombs.p", "rb"))

	if integrated:
		return combo_list
	else:
		drugcombs = list()
		for project, comb_list in combo_dict.items():
			if project in project_list:
				drugcombs.extend(comb_list)

		return list(set(drugcombs))


# ---------------------------------------------------------------------------#
#                              Perturbations                                 #
# ---------------------------------------------------------------------------#

def all_screened_perturbations(project_list, integrated):
	"""
	List of screened perturbations
	:param project_list: The name of the project(s) in a list
	:param integrated: Usage of the integrated dataset (T/F)
	:return: A list of screened perturbations
	"""

	if "all_screen_perturbations_project.p" not in os.listdir(output_path + "data/objects/"):
		perturbations_dict = dict()
		perturbations_list = list()
		for project in project_base_name.keys():
			combi = annotate_viabilities(treatment="combination", project=project)
			combi["DrugComb"] = combi.apply(
				lambda x: sort_drug_pairs(x.anchor_name, x.library_name)["sorted_pair"], axis=1)
			perturbations = list(combi.groupby(["SIDM", "DrugComb"]).groups.keys())
			perturbations_dict[project] = perturbations
			perturbations_list.extend(perturbations)

		pickle.dump(perturbations_dict, open(output_path + "data/objects/all_screen_perturbations_project.p", "wb"))
		perturbations_list = list(set(perturbations_list))
		pickle.dump(perturbations_list, open(output_path + "data/objects/all_screen_perturbations.p", "wb"))

	else:
		perturbations_dict = pickle.load(open(output_path + "data/objects/all_screen_perturbations_project.p", "rb"))
		perturbations_list = pickle.load(open(output_path + "data/objects/all_screen_perturbations.p", "rb"))

	if integrated:
		return perturbations_list
	else:
		all_perturbations = list()
		for project, perturbations in perturbations_dict:
			if project in project_list:
				all_perturbations.extend(perturbations)

		return list(set(all_perturbations))


def get_perturbation_adj_matrix(screen, integrated):
	"""
	Adjancency matrix of the perturbations
	:param screen: The name of the screen
	:param integrated: Usage of the integrated dataset (T/F)
	:return:
	"""
	if integrated:
		title = "integrated"
	else:
		title = screen

	if "%s_perturbation_matrix.csv" % title is not os.listdir(output_path + "data/"):
		perturbations = all_screened_perturbations(project_list=project_list, integrated=integrated)
		combs = all_screened_combinations(project_list=project_list, integrated=integrated)
		models = all_screened_SIDMs(project_list=project_list, integrated=integrated)

		matrix = pandas.DataFrame(0, index = models, columns=combs)
		for p in perturbations:
			matrix.loc[p[0], p[1]] = 1

		matrix["Tissue"] = matrix.apply(lambda x: CellLine(sanger2model(x.name)).tissue, axis=1)

		matrix.to_csv(output_path + "data/%s_perturbation_matrix.csv" % title, index=True)
	else:
		matrix = pandas.read_csv(output_path + "data/%s_perturbation_matrix.csv" % title, index_col=0)
	return matrix


# ---------------------------------------------------------------------------#
#                                 Tissues                                    #
# ---------------------------------------------------------------------------#

def all_tissues():
	"""
	List of screened tissues
	:return:
	"""
	if "all_screen_tissue.p" not in os.listdir(output_path + "data/objects/"):
		models = all_screened_SIDMs(project_list=None, integrated=True)
		tissues = list()
		for model in models:
			t = CellLine(sanger2model(model)).tissue
			if t not in tissues:
				tissues.append(t)
		pickle.dump(tissues, open(output_path + "data/objects/all_screen_tissue.p", "wb"))
	else:
		tissues = pickle.load(open(output_path + "data/objects/all_screen_tissue.p", "rb"))
	return tissues


# ---------------------------------------------------------------------------#
#                                  XData                                     #
# ---------------------------------------------------------------------------#


def get_uniprot_conversion():
	"""
	uniprot = pandas.read_csv(input_path + "uniprot/uniprot_swissprots_2023_02_23.csv")
	uniprot.columns = ["uniprot", "entry", "gene_name"]
	"""

	if "swissprot_labels.csv" not in os.listdir(input_path + "uniprot/"):

		if "human_idmapping.csv" not in os.listdir(input_path + "uniprot/"):
			# Retrieved in 12/04/24
			uniprot = pandas.read_csv(input_path + "uniprot/HUMAN_9606_idmapping.dat", index_col=0,
									  header=None, sep="\t")

			uniprot = uniprot[uniprot[1] == "Gene_Name"]
			uniprot = uniprot[[2]]
			uniprot.columns = ["gene_name"]
			uniprot = uniprot.rename_axis(None, axis=0)
			uniprot.to_csv(input_path + "uniprot/human_idmapping.csv", index=True)

		else:
			uniprot = pandas.read_csv(input_path + "uniprot/human_idmapping.csv", index_col=0)
			uniprot["reviewed"] = None
			server = "https://www.ebi.ac.uk/proteins/api/"
			for u in list(x.index.unique()):
				print(u)
				uniprot_api = "proteins?offset=0&size=-1&accession=%s" % u
				api_request = requests.get(server + uniprot_api, headers={"Accept": "application/json"})
				if api_request.status_code == 200:
					for i in api_request.json():
						reviewed = False if i["info"]["type"] == "TrEMBL" else True
						print(reviewed)
						uniprot.loc[u, "reviewed"] = reviewed

		uniprot = uniprot[uniprot.reviewed]
		uniprot.to_csv(input_path + "uniprot/swissprot_labels.csv", index=True)

	else:
		uniprot = pandas.read_csv(input_path + "uniprot/swissprot_labels.csv", index_col=0)

	return uniprot







