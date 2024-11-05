#---------------------------------------------------------------------------#
#				  C o m b D r u g - C A N C E R M O D E L					#
#---------------------------------------------------------------------------#

"""
#---------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 29 October 2021
Last Update : 21 March 2024
Input : Cell Line Passports dataset
Output: CellLine Object
#---------------------------------------------------------------------------#
"""
import os

#---------------------------------------------------------------------------#

# Import
import pandas, numpy
from CombDrug.module.path import *

"""
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

readRDS = robjects.r['readRDS']
readRData = robjects.r['load']
"""

project_base_name = {"pancancer": "Pancan_combi_12Jan21", "breast": "GDSC_Breast_09Sep21",
                     "colo": "GDSC_Colo_13Sep21", "pancreas": "GDSC_Panc_09Sep21",
                     "gdsc7": "GDSC_007_to_AL_18Mar22",
					 "gdsc8": "GDSC_008_to_AL_18Mar22",
                     "gdsc9": "GDSC_009_to_AL_26Apr22",
					 "gdsc10": "GDSC_010-B_to_AL_22Mar22"}


#---------------------------------------------------------------------------#

# Cell Model Passports

# Cancer Cell Line Information

def cell_model_passport():
	if "cmp_model_info.csv" not in os.listdir(output_path + "data/model_info/"):
		cmp_df = pandas.read_csv(input_path + "cell_model_passports/model_list_20240110.csv")
		cmp_df = cmp_df[cmp_df.model_type == "Cell Line"]
		cmp_df = cmp_df[["model_id", "model_name", "growth_properties", "cancer_type_detail",
						 "COSMIC_ID", "BROAD_ID", "CCLE_ID", "msi_status"]]
		cmp_df["broad"] = cmp_df.apply(
			lambda x: ";".join(list(set(x.BROAD_ID.split(";"))))
			if pandas.isna(x.BROAD_ID) == False and len(x.BROAD_ID.split(";")) > 1 else x.BROAD_ID, axis=1)
		# Manual curation due to broad information deleted.
		# ACH-002303 is not in Broad anymore
		cmp_df.loc[cmp_df[cmp_df.broad == "ACH-002392;ACH-002303"].index, "broad"] = "ACH-002392"
		# ACH-001741 is not in Broad anymore
		# ACH-000833’s name is RH30
		cmp_df.loc[cmp_df[cmp_df.broad == "ACH-000833;ACH-001741;ACH-001189"].index, "broad"] = "ACH-001189"
		# NCI-H2369 --> CMP (None) and DepMap (ACH-002123)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM00104"].index, "broad"] = "ACH-002123"
		# NCI-H3118 --> CMP (None) and DepMap (ACH-002239)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM00517"].index, "broad"] = "ACH-002239"
		# COLO-320-HSR--> CMP (None) and DepMap (ACH-002219)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM00842"].index, "broad"] = "ACH-002219"
		# LU-99A--> CMP (None) and DepMap (ACH-002158)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM01219"].index, "broad"] = "ACH-002158"
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM01219"].index, "growth_properties"] = "Adherent"
		# SJRH30--> CMP (ACH-001741, ACH-001189, ACH-000833) and DepMap (ACH-001189)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM01095"].index, "broad"] = "ACH-001189"
		# SCC90 --> CMP (ACH-002315 - not included in DepMap) and DepMap (ACH-001227)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM00399"].index, "broad"] = "ACH-001227"
		# DOV13--> CMP (None) and DepMap (ACH-001063)
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM00969"].index, "broad"] = "ACH-001063"
		# NCI-H322M --> CMP(ACH-000837 and ACH-002172) but ACH-000837 is NCI-H322
		cmp_df.loc[cmp_df[cmp_df.model_id == "SIDM00117"].index, "broad"] = "ACH-002172"

		cmp_df.columns = ["sanger_id", "model_name", "growth_properties", "cancer_type_detail",
						  "cosmic", "BROAD_ID", "ccle", "msi_status", "broad"]
		cmp_df = cmp_df[["sanger_id", "model_name", "growth_properties", "cancer_type_detail",
						 "cosmic", "ccle", "msi_status", "broad"]]
		cmp_df.to_csv(output_path + "model_info/cmp_model_info.csv", index=False)
	else:
		cmp_df = pandas.read_csv(output_path + "data/model_info/cmp_model_info.csv")

	return cmp_df


def depmap():
	if "depmap_model_info.csv" not in os.listdir(output_path + "data/model_info/"):
		dm_df = pandas.read_csv(input_path + "DepMap/Model_23Q4.csv")
		dm_df = dm_df[["ModelID", "SangerModelID", "OncotreeLineage",
					   "LegacyMolecularSubtype", "LegacySubSubtype"]]
		dm_df.columns = ["broad", "sanger_id", "lineage",
						 "molecular_subtype", "sub_subtype"]
		dm_df.loc[list(dm_df[dm_df.sanger_id == "SIDM00117"].index), "broad"] = "ACH-002172"
		dm_df = dm_df.drop_duplicates()
		dm_df.to_csv(output_path + "data/model_info/depmap_model_info.csv", index=False)
	else:
		dm_df = pandas.read_csv(output_path + "data/model_info/depmap_model_info.csv")
	return dm_df


"""
def check_cell_line(model_name):
    df = cancer_model()
    if model_name not in list(df.model_name):
        if len(model_name.split("-")) > 1 and len(model_name.split(".")) > 1:
            if "".join("".join(model_name.split("-")).split(".")) in list(df.model_name):
                return "".join("".join(model_name.split("-")).split("."))
            else:
                return model_name
        elif len(model_name.split("-")) > 1:
            if "".join(model_name.split("-")) in list(df.model_name):
                return "".join(model_name.split("-"))
            else: return model_name
        elif len(model_name.split(".")) > 1:
            if "".join(model_name.split(".")) in list(df.model_name):
                return "".join(model_name.split("."))
            else: return model_name
        else: return model_name
    else: return model_name

"""



# Screens

def get_combi(project):

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
		combi_df = pandas.read_csv(output_folder + project_base_name[project] + "_combi_df.csv", low_memory=False)

	return combi_df


def collect_model_info():
	if "model_info_combi.csv" not in os.listdir(output_path + "data/model_info/"):
		cl_df = pandas.DataFrame(columns = ["model_name", "cell_id", "sanger_id", "master_cell_id",
											"tcga", "msi_status", "tissue", "cancer_type"])
		for project in project_base_name.keys():
			df = get_combi(project)
			if "CELL_ID" in list(df.CL_SPEC.unique()):
				df2 = df[["CELL_LINE_NAME", "CL", "SIDM", "MASTER_CELL_ID", "TCGA_DESC",
						 "msi_status", "tissue", "cancer_type"]].drop_duplicates()
				df2.columns = ["model_name", "cell_id", "sanger_id", "master_cell_id",
							  "tcga", "msi_status", "tissue", "cancer_type"]
			if "MASTER_CELL_ID" in list(df.CL_SPEC.unique()):
				df2 = df[["CELL_LINE_NAME", "CL", "SIDM", "TCGA_DESC",
						 "msi_status", "tissue", "cancer_type"]].drop_duplicates()
				df2.columns = ["model_name", "master_cell_id", "sanger_id",
							  "tcga", "msi_status", "tissue", "cancer_type"]

			cl_df = pandas.concat([cl_df, df2])
			cl_df = cl_df.drop_duplicates()

		cl_df.columns = ["model_name", "cell_id", "sanger_id", "master_cell_id",
						 "tcga", "msi_status", "tissue", "cancer_type"]

		curated_cl_df = pandas.DataFrame(columns = ["model_name", "cell_id", "sanger_id", "master_cell_id",
													"tcga", "msi_status", "tissue", "cancer_type"])
		for cl, x in cl_df.groupby(["model_name"]):
			# Already checked the master IDs and for all they are one specific.
			if len(x.sanger_id.unique()) == 1:
				if len(x.master_cell_id.unique()) == 1:
					x2 = x[x.master_cell_id.isin([i for i in x.master_cell_id.unique() if pandas.notna(i)])]
					curated_cl_df = pandas.concat([curated_cl_df, x2])

		curated_cl_df.to_csv(output_path + "data/model_info/model_info_combi.csv", index=False)
	else:
		curated_cl_df = pandas.read_csv(output_path + "data/model_info/model_info_combi.csv")
	return curated_cl_df


# Combine all information
def cancer_models():
	if "cancer_models.csv" not in os.listdir(output_path + "data/model_info/"):
		cmp = cell_model_passport()
		dm_df = depmap()
		cl_df = collect_model_info()

		# Only LS-1034 name is different as LS1034 --> GDSC name will be used

		model_df = cl_df.merge(cmp, how="left", on= ["sanger_id"])[[
			"model_name_x", "cell_id", "sanger_id", "tcga", "master_cell_id", "msi_status_y", "tissue", "cancer_type",
			"growth_properties", "cancer_type_detail", "cosmic", "broad", "ccle"]]
		model_df.columns = ["model_name", "cell_id", "sanger_id", "tcga", "master_cell_id", "msi_status", "tissue",
							"cancer_type", "growth_properties", "cancer_type_detail", "cosmic", "broad", "ccle"]

		# Found discrepancies of the BROAD IDs - No Broad_x is taken
		# Broad ID
		# SCC90 --> CMP (ACH-002315 - not included in DepMap) and DepMap (ACH-001227)
		# SOLVED | NCI-H2369 --> CMP (None) and DepMap (ACH-002123)
		# NCI-H3118 --> CMP (None) and DepMap (ACH-002239)
		# DOV13--> CMP (None) and DepMap (ACH-001063)
		# COLO-320-HSR--> CMP (None) and DepMap (ACH-002219)
		# LU-99A--> CMP (None) and DepMap (ACH-002158)
		# NCI-H513--> CMP (ACH-002335 - not included in DepMap) and DepMap (ACH-002138)
		# SJRH30--> CMP (ACH-001741, ACH-001189, ACH-000833) and DepMap (ACH-001189)

		model_df = model_df.merge(dm_df, how="left", on =["sanger_id"])[[
			"model_name", "cell_id", "sanger_id", "tcga", "master_cell_id", "msi_status", "tissue", "cancer_type",
			"growth_properties", "cancer_type_detail", "cosmic", "ccle", "broad_y", "lineage",
			"molecular_subtype", "sub_subtype"]]

		model_df.columns = ["model_name", "cell_id", "sanger_id", "tcga", "master_cell_id", "msi_status", "tissue",
							"cancer_type", "growth_properties", "cancer_type_detail", "cosmic", "ccle", "broad",
							"lineage", "molecular_subtype", "sub_subtype"]

		model_df = model_df.drop_duplicates()

		# MSI discrepancies
		# JIMT-1 has four cell ids and only one of them MSS others are NaN --> Control CMP and manually corrected as MSS
		# A2780 has four cell ids and only one of them MSI others are MSS --> Control CMP and manually corrected as MSI
		# CAL-51 has four cell ids and only one of them MSI others are MSS --> Control CMP and manually corrected as MSI
		# LC-1-sq has four cell ids and only one of them MSI others are MSS --> Control CMP and manually corrected as MSI
		model_df.loc[model_df[model_df.model_name == "JIMT-1"].index, "msi_status"] = "MSS"
		model_df.loc[model_df[model_df.model_name == "A2780"].index, "msi_status"] = "MSI"
		model_df.loc[model_df[model_df.model_name == "CAL-51"].index, "msi_status"] = "MSI"
		model_df.loc[model_df[model_df.model_name == "LC-1-sq"].index, "msi_status"] = "MSS"
		model_df.loc[model_df[model_df.model_name == "NCI-H1395"].index, "msi_status"] = "MSS"
		model_df.loc[model_df[model_df.model_name == "NCI-H1417"].index, "msi_status"] = "MSS"
		model_df.loc[model_df[model_df.model_name == "NCI-H1876"].index, "msi_status"] = "MSS"
		model_df.loc[model_df[model_df.model_name == "OCUB-M"].index, "msi_status"] = "MSI"
		model_df.loc[model_df[model_df.model_name == "VMRC-LCD"].index, "msi_status"] = "MSS"

		model_df.to_csv(output_path + "data/model_info/cancer_models.csv", index = False)
	else:
		model_df = pandas.read_csv(output_path + "data/model_info/cancer_models.csv")
	return model_df


def all_models():
	df = cancer_models()
	list_of_models = list(df["model_name"].unique())
	return list_of_models


def all_sanger_ids():
	df = cancer_models()
	list_of_sanger = list(df["sanger_id"].unique())
	return list_of_sanger


def all_cell_ids():
	df = cancer_models()
	list_of_ids = list(df["cell_id"].unique())
	return list_of_ids


def gdsc2model(cell_id):
	model_name = None
	df = cancer_models()
	if cell_id in df.cell_id.unique():
		model_name = ",".join([str(i) for i in df[df.cell_id == cell_id]["model_name"].unique() if pandas.isna(i) == False])
	if model_name is not None and model_name != "":
		return model_name
	else: return None


def model2gdsc(model_name):
	gdsc_id = None
	df = cancer_models()
	if model_name in df.model_name.unique():
		if True in pandas.isna(df[df.model_name == model_name]["cell_id"].unique()):
			gdsc_id = None
		else:
			gdsc_id = ",".join([str(i) for i in df[df.model_name == model_name]["cell_id"].unique() if pandas.isna(i) == False])
		return gdsc_id
	else: return None


def model2sanger(model_name):
	sanger_id = None
	df = cancer_models()
	if model_name in df.model_name.unique():
		if True in pandas.isna(df[df.model_name == model_name]["sanger_id"].unique()):
			sanger_id = None
		else:
			sanger_id = ",".join([str(i) for i in df[df.model_name == model_name]["sanger_id"].unique()])
		return sanger_id
	else: return None


def model2ccle(model_name):
	ccle = None
	df = cancer_models()
	if model_name in df.model_name.unique():
		if True in pandas.isna(df[df.model_name == model_name]["ccle"].unique()):
			ccle = None
		else:
			ccle = ",".join([str(i) for i in df[df.model_name == model_name]["ccle"].unique()])
		return ccle
	else: return None


def model2cosmic(model_name):
	cosmic = None
	df = cancer_models()
	if model_name in df.model_name.unique():
		if True in pandas.isna(df[df.model_name == model_name]["cosmic"].unique()):
			cosmic = None
		else:
			cosmic = ",".join([str(i) for i in df[df.model_name == model_name]["cosmic"].unique()])
		return cosmic
	else: return None


def model2broad(model_name):
	broad = None
	df = cancer_models()
	if model_name in df.model_name.unique():
		if True in pandas.isna(df[df.model_name == model_name]["broad"].unique()):
			broad = None
		else:
			broad = ",".join([str(i) for i in df[df.model_name == model_name]["broad"].unique()])
		return broad
	else:
		return None


def broad2model(broad):
	model_name = None
	df = cancer_models()
	if broad in df.broad.unique():
		if True in pandas.isna(df[df.broad == broad]["model_name"].unique()):
			model_name = None
		else:
			model_name = ",".join([str(i) for i in df[df.broad == broad]["model_name"].unique()])
		return model_name
	else:
		return None


def sanger2model(sanger_id):
	model = None
	df = cancer_models()
	if sanger_id in df.sanger_id.unique():
		if True in pandas.isna(df[df.sanger_id == sanger_id]["model_name"].unique()):
			model_name = None
		else:
			model_name = ",".join([str(i) for i in df[df.sanger_id == sanger_id]["model_name"].unique()])
		return model_name
	else: return None


def get_tissue_names():
	df = cancer_models()
	return df.tissue.unique()


def get_growth():

	if "cmp_growth_rate.csv" not in os.listdir(input_path + "cell_model_passports/"):
		df = pandas.read_csv(input_path + "cell_model_passports/growth_rate_20220907.csv")
		size_df = pandas.DataFrame(df.groupby(["model_id"]).size())
		models_one = list(size_df[size_df[0] == 1].index)
		models_more = list(size_df[size_df[0] > 1].index)

		more_df1 = df[df.model_id.isin(models_more)][["model_id", "day4_day1_ratio", "replicates"]]
		indices = more_df1.groupby(["model_id"])["replicates"].idxmax().values
		more_df2 = more_df1.loc[indices]

		one_df = df[df.model_id.isin(models_one)][["model_id", "day4_day1_ratio", "replicates"]]
		last_df = pandas.concat([one_df, more_df2]).drop_duplicates()[["model_id", "day4_day1_ratio"]]
		last_df.to_csv(input_path + "cell_model_passports/cmp_growth_rate.csv")
	else:
		last_df = pandas.read_csv(input_path + "cell_model_passports/cmp_growth_rate.csv", index_col=0)
	return last_df


def get_colon_subgroups():
	if "cris_cms_subgrouping.csv" not in os.listdir(input_path + "cell_model_passports/"):

		df = pandas.read_csv(input_path + "cell_model_passports/summary_classifiers_organoids-celllines_shriram.csv")
		df["SIDM"] = df.apply(lambda x: CellLine(x.CELL_LINE_NAME).id, axis=1)
		df = df[~pandas.isna(df["SIDM"])]

		df.to_csv(input_path + "cell_model_passports/cris_cms_subgrouping.csv", index=True)
	else:
		df = pandas.read_csv(input_path + "cell_model_passports/cris_cms_subgrouping.csv", index_col=0)
	return df


def get_pam50():
	df = pandas.read_csv(input_path + "cell_model_passports/nature_paper_pam50.csv", index_col=0)
	return df



#---------------------------------------------------------------------------#

# CellLine Object

class CellLine:

    def __init__(self, model_name):
        df = cancer_models()
        self.model_name = model_name
        if self.model_name in df.model_name.unique():
            x = df[df.model_name == self.model_name]
            self.id = model2sanger(self.model_name)
            self.master_id = ",".join([str(i) for i in x["master_cell_id"].unique()]) \
                if True not in list(pandas.isna(x["master_cell_id"])) and len(x["master_cell_id"]) != 0 else None
            self.broad = model2broad(self.model_name)
            self.cosmic = model2cosmic(self.model_name)
            self.ccle = model2ccle(self.model_name)
            self.tissue = ",".join(x["tissue"].unique()) \
                if True not in list(pandas.isna(x["tissue"])) and len(x["tissue"]) != 0 else None
            self.cancer = ",".join(x["cancer_type"].unique()) \
                if True not in list(pandas.isna(x["cancer_type"])) and len(x["cancer_type"]) != 0 else None
            self.cancer_detail = ",".join(x["cancer_type_detail"].unique()) \
                if True not in list(pandas.isna(x["cancer_type_detail"])) and len(x["cancer_type_detail"]) != 0 else None
            self.growth_properties = ",".join(x["growth_properties"].unique()) \
                if True not in list(pandas.isna(x["growth_properties"])) and len(x["growth_properties"]) != 0 else None
            self.msi = ",".join(x["msi_status"].unique()) \
                if True not in list(pandas.isna(x["msi_status"])) and len(x["msi_status"]) != 0 else None
            self.cell_id = ",".join([str(i) for i in x["cell_id"].unique()]) \
                if True not in list(pandas.isna(x["cell_id"])) and len(x["cell_id"]) != 0 else None
            self.cancer_depmap = ",".join([str(i) for i in x["lineage"].unique()]) \
                if True not in list(pandas.isna(x["lineage"])) and len(x["lineage"]) != 0 else None
            self.cancer_subtype_depmap = ",".join([str(i) for i in x["sub_subtype"].unique()]) \
                if True not in list(pandas.isna(x["sub_subtype"])) and len(x["sub_subtype"]) != 0 else None
            self.cancer_molecular_subtype_depmap = ",".join([str(i) for i in x["molecular_subtype"].unique()]) \
                if True not in list(pandas.isna(x["molecular_subtype"])) and len(x["molecular_subtype"]) != 0 else None
        else:
            self.id, self.master_id, self.cell_id = None, None, None
            self.broad, self.cosmic, self.ccle =None, None, None
            self.tissue, self.cancer, self.cancer_detail = None, None, None
            self.growth_properties, self.msi = None, None
            self.cancer_depmap, self.cancer_subtype_depmap = None, None
            self.cancer_molecular_subtype_depmap = None

        self.tissue_type = None
        self.pan_type = None
        self.breast_subgroup = None
        self.cris, self.cms = None, None

    def get_tissue_type(self):

        if self.tissue == "Haematopoietic and Lymphoid":
            if "Leukemia" in self.cancer_detail.split(" "):
                self.tissue_type = "Leukemia"
            elif "Lymphoma" in self.cancer_detail.split(" "):
                self.tissue_type = "Lymphoma"
            elif "Myeloma" in self.cancer_detail.split(" "):
                self.tissue_type = "Myeloma"
            else:
                self.tissue_type = "Other liquid tumours"
        else:
            self.tissue_type = self.tissue

        return self.tissue_type
    def get_pan_type(self):

        if self.tissue == "Haematopoietic and Lymphoid":
            self.pan_type = "panliquid"
        else:
            self.pan_type = "pansolid"

        return self.pan_type

    def get_breast_subgroup(self, breast_group_source):
        if self.tissue == "Breast":
            if breast_group_source == "broad":
                if self.cancer_molecular_subtype_depmap == "luminal":
                    self.breast_subgroup = "luminal A"
                elif self.cancer_molecular_subtype_depmap == "luminal_HER2_amp":
                    self.breast_subgroup = "luminal B"
                elif self.cancer_molecular_subtype_depmap == "HER2_amp":
                    self.breast_subgroup = "HER2+"
                elif self.cancer_molecular_subtype_depmap in ["basal_A", "basal_B"]:
                    self.breast_subgroup = "TRNB"
            elif breast_group_source == "sanger":
                sgorup = get_pam50().loc[self.id, "PAM50"]
                self.breast_subgroup = sgorup
        return self.breast_subgroup

    def get_colon_subgroup(self):

        if self.tissue == "Large Intestine":
            colo_groups = get_colon_subgroups().set_index(["SIDM"])
            self.cris = colo_groups.loc[self.id]["CMScaller_CRIS"] if self.id in colo_groups.index else numpy.nan
            self.cms = colo_groups.loc[self.id]["CMScaller_CMS"] if self.id in colo_groups.index else numpy.nan

        return self.cms, self.cris