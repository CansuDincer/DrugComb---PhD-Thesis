# ---------------------------------------------------------------------------#
#						C o m b D r u g - D R U G							#
# ---------------------------------------------------------------------------#

"""
#---------------------------------------------------------------------------#
data - drug
Author : Cansu Dincer
Date : 29 October 2021
Last Update : 19 March 2024
Input : Drug information from matrix explorer and GDSC dataset
Output: Drug Object
#---------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#

# Import
import os, pandas, json, pubchempy, requests, numpy
from CombDrug.module.path import *
from CombDrug.module.data.dataset_info import *


# from rpy2.robjects import pandas2ri
# pandas2ri.activate()

# ---------------------------------------------------------------------------#

# Combi Data

def get_combi_drug():
	if "combi_drug_df.csv" not in os.listdir(output_path + "data/drug_info/"):
		drug = pandas.DataFrame(columns=["drug_id", "drug_name", "drug_target", "drug_gene_target",
										 "drug_type", "drug_pathway_target"])

		for project in project_base_name:
			combi_df = get_combi(project)
			a = combi_df[["ANCHOR_ID", "ANCHOR_NAME", "ANCHOR_TARGET", "ANCHOR_GENE_TARGET",
						  "ANCHOR_DRUG_TYPE", "ANCHOR_TARGET_PATHWAY"]]
			a.columns = ["drug_id", "drug_name", "drug_target", "drug_gene_target",
						 "drug_type", "drug_pathway_target"]
			l = combi_df[["LIBRARY_ID", "LIBRARY_NAME", "LIBRARY_TARGET", "LIBRARY_GENE_TARGET",
						  "LIBRARY_DRUG_TYPE", "LIBRARY_TARGET_PATHWAY"]]
			l.columns = ["drug_id", "drug_name", "drug_target", "drug_gene_target",
						 "drug_type", "drug_pathway_target"]

			al = pandas.concat([a, l], ignore_index=True)

			mult_drugs = list()
			mult_drug_row = list()
			for drug_id in al.drug_id.unique():
				if len(str(drug_id).split("|")) > 1:
					x = al[al.drug_id == drug_id]
					if len(list(x.index)) == 1:
						mult_drug_row.append(list(x.index)[0])
					else:
						for k in list(x.index):
							mult_drug_row.append(k)
					x = x.drop_duplicates()
					for i in range(len(str(drug_id).split("|"))):
						if str(drug_id).split("|")[i] not in al.drug_id.unique():
							d = {"drug_id": x.drug_id.unique().split("|")[i],
								 "drug_name": x.drug_name.unique().split(" | ")[i],
								 "drug_target": x.drug_target.unique().split(" | ")[i],
								 "drug_gene_target": x.drug_gene_target.unique().split(" | ")[i],
								 "drug_type": x.drug_type.unique().split(" | ")[i],
								 "drug_pathway_target": x.drug_pathway_target.unique().split(" | ")[i]}

							d_df = pandas.DataFrame.from_dict(d)
							mult_drugs.append(d_df)

			if mult_drugs:
				m_df = pandas.concat(mult_drugs)
				al = pandas.concat([al, m_df])
			al = al.drop(mult_drug_row)
			al = al.drop_duplicates()
			drug = pandas.concat([drug, al], ignore_index=True)

		drug.astype({"drug_id": "int32"}).dtypes
		drug = drug.drop_duplicates()
		"""
		for d, df in drug.groupby(["drug_id"]):
			if len(df.index) > 1:
				# Capivasertib and AZD5363 --> same thing
				drug_names = df["drug_name"].unique()
				for drug_name in drug_names:
					if drug_name[:3] == "AZD":
						drug.drop(drug[drug.drug_name == drug_name].index, inplace=True)
		"""

		drug.to_csv(output_path + "data/drug_info/combi_drug_df.csv", index=False)

	else:
		drug = pandas.read_csv(output_path + "data/drug_info/combi_drug_df.csv")
		drug = drug.drop_duplicates()

	return drug


def get_ot_target(row):
	targets = list()
	if row is not None and row != []:
		for i in row:
			if "targets" in i.keys():
				for l in i["targets"]:
					if "approvedSymbol" in l.keys():
						targets.append(l["approvedSymbol"])
	if targets:
		t = ", ".join(list(set(targets)))
	else:
		t = None
	return t


def get_ot_xref(xref_list, xref_type):
	t = ""
	if xref_list is not None:
		for i in xref_list:
			if i["source"] == xref_type:
				t = ", ".join(i["reference"])
		if t == "":
			t = None
	return t


def x_opentargets(chembl):
	"""

	:param chembl:
	:return:
	"""
	query = """
	query drugApprovalWithdrawnWarningData($chemblId: String!) {
	  drug(chemblId: $chemblId) {
	  	name
		tradeNames
		id
		isApproved
		hasBeenWithdrawn
		maximumClinicalTrialPhase
		mechanismsOfAction{
		  uniqueActionTypes
		  uniqueTargetTypes
		  rows{
			targets {
			  approvedSymbol
			}
		  }
		}
		crossReferences{
		  source
		  reference
		}
	  }
	}

	"""

	ot_url = "https://api.platform.opentargets.org/api/v4/graphql"
	r = requests.post(ot_url, json={"query": query, "variables": {"chemblId": chembl}})

	ot_dict = None
	if r.status_code == 200:
		response = json.loads(r.text)
		d = response["data"]["drug"]
		if d is not None:
			ot_dict = dict()
			ot_dict["name"] = d["name"].lower() if d["name"] is not None else None
			ot_dict["tradeNames"] = ", ".join(d["tradeNames"]) if d["tradeNames"] != [] and d[
				"tradeNames"] is not None else ""
			if ot_dict["tradeNames"] == "" or ot_dict["tradeNames"] == ", ":
				ot_dict["tradeNames"] = None
			ot_dict["approval"] = d["isApproved"] if d["isApproved"] is not None else False
			ot_dict["withdrawn"] = d["hasBeenWithdrawn"]
			ot_dict["max_clinical_trials"] = d[
				"maximumClinicalTrialPhase"] if "maximumClinicalTrialPhase" in d.keys() else False
			ot_dict["mechanismsOfAction"] = ", ".join(d["mechanismsOfAction"]["uniqueActionTypes"]) \
				if d["mechanismsOfAction"]["uniqueActionTypes"] != [] else None
			ot_dict["targets"] = get_ot_target(row=d["mechanismsOfAction"]["rows"])
			ot_dict["pubchem"] = get_ot_xref(xref_list=d["crossReferences"], xref_type="PubChem")
			ot_dict["drugbank"] = get_ot_xref(xref_list=d["crossReferences"], xref_type="drugbank")

	return ot_dict


def make_drug_df():
	if "annotated_drug_df.csv" not in os.listdir(output_path + "data/drug_info/"):
		# GET PUBMED IDS
		df = get_combi_drug()[["drug_id", "drug_name", "drug_gene_target", "drug_type", "drug_pathway_target"]]
		# Retrieved on 15th of March 2024
		canapps_df = pandas.read_csv(output_path + "data/drug_info/canapps_all.csv")
		df["cid"] = df.apply(
			lambda x: canapps_df[canapps_df["Drug ID"] == x.drug_id].Pubchem.values[0] if
			canapps_df[canapps_df["Drug ID"] == x.drug_id].Pubchem.values[0] is not None and pandas.isna(
				canapps_df[canapps_df["Drug ID"] == x.drug_id].Pubchem.values[0]) != True else (
				int(pubchempy.get_compounds(x.drug_name, "name")[0].cid) if len(
					pubchempy.get_compounds(x.drug_name, "name")) == 1 else None), axis=1)

		# GET CHEMBL IDS
		df["chembl"] = df.apply(lambda x: canapps_df[canapps_df["Drug ID"] == x.drug_id].Chembl.values[0], axis=1)
		df["chembl"] = df["chembl"].replace(numpy.nan, None)
		df["chembl"] = df["chembl"].replace("None", None)
		for g, g_df in df.groupby(["drug_name"]):
			if len(g_df.index) > 1:
				if len(g_df.chembl.unique()) > 1:
					chembl = [i for i in list(g_df.chembl.unique()) if i is not None]
					if len(chembl) > 1:
						if "SCHEMBL34018" in chembl and g == "SN-38":
							chembl = ["CHEMBL837"]
						if "CHEMBL2364611" in chembl and g == "AZD1775":
							chembl = ["CHEMBL1976040"]
						if "SCHEMBL34018" in chembl and g == "Venetoclax":
							chembl = ["CHEMBL3137309"]
						if "CHEMBL2387080" in chembl and g == "Palbociclib":
							chembl = ["CHEMBL189963"]
						if "CHEMBL2364611" in chembl and g == "Olaparib":
							chembl = ["CHEMBL521686"]
						if "CHEMBL2364611" in chembl and g == "AZD7648":
							chembl = ["CHEMBL4650446"]
					df.loc[df[df.drug_name == g].index, "chembl"] = chembl[0]

		# MANUAL CURATION
		df["sid"] = None
		df.loc[df[df.drug_name == "Doxorubicin"].index, "chembl"] = "CHEMBL53463"
		df.loc[df[df.drug_name == "Prexasertib"].index, "chembl"] = "CHEMBL3544911"
		df.loc[df[df.drug_name == "AZD6738"].index, "cid"] = 54761306
		df.loc[df[df.drug_name == "AZD6738"].index, "chembl"] = "CHEMBL4285417"
		df.loc[df[df.drug_name == "AZD1775"].index, "chembl"] = "CHEMBL1976040"
		df.loc[df[df.drug_name == "AZD0156"].index, "chembl"] = "CHEMBL3960662"
		df.loc[df[df.drug_name == "AZD2811"].index, "chembl"] = "CHEMBL215152"
		df.loc[df[df.drug_name == "SG3199"].index, "chembl"] = "SCHEMBL15686377"
		df.loc[df[df.drug_name == "AZD6244"].index, "chembl"] = "CHEMBL1614701"
		df.loc[df[df.drug_name == "AGI-25696"].index, "chembl"] = "CHEMBL4552844"
		df.loc[df[df.drug_name == "GSK3326595"].index, "chembl"] = "CHEMBL4466233"
		df.loc[df[df.drug_name == "GSK3368715"].index, "chembl"] = "SCHEMBL16121956"
		df.loc[df[df.drug_name == "Capivasertib"].index, "cid"] = 25227436
		df.loc[df[df.drug_name == "Capivasertib"].index, "chembl"] = "CHEMBL2325741"
		df.loc[df[df.drug_name == "AZ-6197"].index, "cid"] = 122604354
		df.loc[df[df.drug_name == "AZ-6197"].index, "chembl"] = "CHEMBL4075638"
		df.loc[df[df.drug_name == "Gefitinib"].index, "cid"] = 123631
		df.loc[df[df.drug_name == "Gefitinib"].index, "chembl"] = "CHEMBL939"
		df.loc[df[df.drug_name == "Savolitinib"].index, "cid"] = 68289010
		df.loc[df[df.drug_name == "Savolitinib"].index, "chembl"] = "CHEMBL3334567"
		df.loc[df[df.drug_name == "AZD1390"].index, "cid"] = 126689157
		df.loc[df[df.drug_name == "AZD1390"].index, "chembl"] = "CHEMBL4594429"
		df.loc[df[df.drug_name == "AZD5153"].index, "cid"] = 118693659
		df.loc[df[df.drug_name == "AZD5153"].index, "chembl"] = "CHEMBL4078100"
		df.loc[df[df.drug_name == "Selumetinib"].index, "cid"] = 10127622
		df.loc[df[df.drug_name == "Selumetinib"].index, "chembl"] = "CHEMBL1614701"
		df.loc[df[df.drug_name == "AZD8186"].index, "cid"] = 52913813
		df.loc[df[df.drug_name == "AZD8186"].index, "chembl"] = "CHEMBL3545424"
		df.loc[df[df.drug_name == "AZD4320"].index, "cid"] = 86661883
		df.loc[df[df.drug_name == "AZD4320"].index, "chembl"] = "CHEMBL3703600"
		df.loc[df[df.drug_name == "AZD5991"].index, "cid"] = 131634760
		df.loc[df[df.drug_name == "AZD5991"].index, "chembl"] = "CHEMBL4297482"
		df.loc[df[df.drug_name == "Venetoclax"].index, "cid"] = 49846579
		df.loc[df[df.drug_name == "XAV939"].index, "cid"] = 135418940
		df.loc[df[df.drug_name == "Cetuximab"].index, "cid"] = None
		df.loc[df[df.drug_name == "Cetuximab"].index, "sid"] = 103771411
		df.loc[df[df.drug_name == "Lapatinib"].index, "chembl"] = "CHEMBL554"
		df.loc[df[df.drug_name == "Cisplatin"].index, "chembl"] = "CHEMBL11359"
		df.loc[df[df.drug_name == "Gemcitabine"].index, "chembl"] = "CHEMBL888"
		df.loc[df[df.drug_name == "Crizotinib"].index, "chembl"] = "CHEMBL601719"
		df.loc[df[df.drug_name == "Dasatinib"].index, "chembl"] = "CHEMBL1421"
		df.loc[df[df.drug_name == "SCH772984"].index, "chembl"] = "CHEMBL3590107"
		df.loc[df[df.drug_name == "Ruxolitinib"].index, "chembl"] = "CHEMBL1789941"
		df.loc[df[df.drug_name == "Luminespib"].index, "cid"] = 135539077
		df.loc[df[df.drug_name == "OSI-027"].index, "cid"] = 135398516
		df.loc[df[df.drug_name == "5-Fluorouracil"].index, "chembl"] = "CHEMBL185"
		df.loc[df[df.drug_name == "Alpelisib"].index, "chembl"] = "CHEMBL2396661"
		df.loc[df[df.drug_name == "Alisertib"].index, "chembl"] = "CHEMBL483158"
		df.loc[df[df.drug_name == "AZD4547"].index, "chembl"] = "CHEMBL3348846"
		df.loc[df[df.drug_name == "AZD7648"].index, "chembl"] = "CHEMBL4439259"
		df.loc[df[df.drug_name == "AZD8186"].index, "chembl"] = "CHEMBL3408248"
		df.loc[df[df.drug_name == "RO-3306"].index, "cid"] = 135400873
		df.loc[df[df.drug_name == "Paclitaxel"].index, "chembl"] = "CHEMBL428647"
		df.loc[df[df.drug_name == "Trametinib"].index, "chembl"] = "CHEMBL2103875"
		df.loc[df[df.drug_name == "Taselisib"].index, "chembl"] = "CHEMBL2387080"
		df.loc[df[df.drug_name == "SRA737"].index, "chembl"] = "CHEMBL4169078"
		df.loc[df[df.drug_name == "Palbociclib"].index, "cid"] = 5330286
		df.loc[df[df.drug_name == "Carboplatin"].index, "cid"] = None
		df.loc[df[df.drug_name == "AZD7648"].index, "cid"] = 135151360
		df.loc[df[df.drug_name == "AZD5363"].index, "cid"] = 25227436
		df.loc[df[df.drug_name == "AZD5363"].index, "chembl"] = "CHEMBL2325741"
		df.loc[df[df.drug_name == "AZ-3202"].index, "chembl"] = None
		df.loc[df[df.chembl == "CHEMBL1976040"].index, "drug_name"] = "AZD1775"
		df.loc[df[df.chembl == "CHEMBL1976040"].index, "drug_gene_target"] = "WEE1;PLK1"

		# ANNOTATION

		df["molecular_formula"] = None
		df["molecular_weight"] = None
		df["exact_mass"] = None
		df["canonical_smiles"] = None
		df["inchikey"] = None
		df["fingerprint"] = None
		df["cactvs_fingerprint"] = None
		for g, g_df in df.groupby(["cid"]):
			if g is not None and pandas.isna(g) is False and g != "None" and g != "several":
				g = int(g)
				x = pubchempy.Compound.from_cid(g).to_dict()
				df.loc[list(g_df.index), "molecular_formula"] = x["molecular_formula"]
				df.loc[list(g_df.index), "molecular_weight"] = x["molecular_weight"]
				df.loc[list(g_df.index), "exact_mass"] = x["exact_mass"]
				df.loc[list(g_df.index), "canonical_smiles"] = x["canonical_smiles"]
				df.loc[list(g_df.index), "inchikey"] = x["inchikey"]
				df.loc[list(g_df.index), "fingerprint"] = x["fingerprint"]
				df.loc[list(g_df.index), "cactvs_fingerprint"] = x["cactvs_fingerprint"]

		df["ot_name"] = None
		df["trade_names"] = None
		df["approval"] = None
		df["withdrawn"] = None
		df["clinical_phase"] = None
		df["clinical_trials"] = None
		df["MoA"] = None
		df["ot_targets"] = None
		df["ot_pubchem"] = None
		df["ot_drugbank"] = None
		for g, g_df in df.groupby(["chembl"]):
			if g is not None and pandas.isna(g) is False and g != "None" and g != "several":
				d = x_opentargets(chembl=g)
				if d is not None:
					df.loc[list(g_df.index), "ot_name"] = d["name"]
					df.loc[list(g_df.index), "trade_names"] = d["tradeNames"]
					df.loc[list(g_df.index), "approval"] = d["approval"]
					df.loc[list(g_df.index), "withdrawn"] = d["withdrawn"]
					df.loc[list(g_df.index), "clinical_phase"] = d["max_clinical_trials"]
					df.loc[list(g_df.index), "clinical_trials"] = True if d["approval"] is False and d[
						"max_clinical_trials"] is not None and d["max_clinical_trials"] is not False else False
					df.loc[list(g_df.index), "MoA"] = d["mechanismsOfAction"]
					df.loc[list(g_df.index), "ot_targets"] = d["targets"]
					df.loc[list(g_df.index), "ot_pubchem_sid"] = d["pubchem"] if "pubchem" in d.keys() else None
					df.loc[list(g_df.index), "ot_drugbank"] = d["drugbank"] if "drugbank" in d.keys() else None

		# Manual curation
		# Chembl
		manual_selected_drug_names = ["Nutlin-3a (-)", "PD173074", "RO-3306", "PF-4708671", "GSK269962A", "NU7441",
									  "ZM447439", "SB505124", "(5Z)-7-Oxozeaenol", "AZD0156", "AZD2811", "SRA737",
									  "AZD4320", "AZ-6197", "AGI-25696"]
		for n in manual_selected_drug_names:
			df.loc[df[df.drug_name == n].index, "approval"] = False
			df.loc[df[df.drug_name == n].index, "withdrawn"] = None
			df.loc[df[df.drug_name == n].index, "clinical_phase"] = None
			df.loc[df[df.drug_name == n].index, "clinical_trials"] = False

		df.to_csv(output_path + "data/drug_info/annotated_drug_df.csv", index=False)
	else:
		df = pandas.read_csv(output_path + "data/drug_info/annotated_drug_df.csv")

	return df


def get_drug_ids():
	drug = make_drug_df()
	return drug.drug_id.unique()


def get_drug_names():
	drug = make_drug_df()
	return drug.drug_name.unique()


def drug_name2id(drug_name):
	drug = make_drug_df()
	if drug_name in drug.drug_name.unique():
		drug_ids = ",".join([str(i) for i in drug[drug.drug_name == drug_name].drug_id.unique()])
		return drug_ids
	else:
		return None


def drug_id2name(drug_id):
	drug = make_drug_df()
	if drug_id in drug.drug_id.unique():
		drug_name = drug[drug.drug_id == drug_id].drug_name.unique()[0]
		return drug_name
	else:
		return None


def get_clinical_df():
	all_drugs = list(make_drug_df()["drug_name"].unique())
	df = pandas.DataFrame(index=all_drugs, columns=["approved", "in_clinical", "clinical_phase", "summary"])

	for drug in all_drugs:
		x = Drug(drug)
		if x.approval is not None and pandas.isna(x.approval) is False and x.approval:
			df.loc[drug, "approved"] = True
			df.loc[drug, "summary"] = "approved"
		else:
			df.loc[drug, "approved"] = False

		if x.clinical_trials is not None and pandas.isna(x.clinical_trials) is False and x.clinical_trials:
			df.loc[drug, "in_clinical"] = True
			if int(x.clinical_phase) != -1:
				df.loc[drug, "clinical_phase"] = x.clinical_phase
				df.loc[drug, "summary"] = "clinical_phase_%d" % int(x.clinical_phase)
			else:
				df.loc[drug, "in_clinical"] = "Unknown"
				df.loc[drug, "clinical_phase"] = "Unknown"
				df.loc[drug, "summary"] = "Unknown"
		else:
			df.loc[drug, "in_clinical"] = False
			df.loc[drug, "clinical_phase"] = None
			if df.loc[drug, "approved"] is False:
				df.loc[drug, "summary"] = "pre_clinical"

	all_combinations = [i for i in all_screened_combinations(project_list=None, integrated=True)
						if len(i.split(" | ")) == 1]

	comb_df = pandas.DataFrame(index=all_combinations, columns=["summary", "priority"])

	for drug_comb in all_combinations:
		drug1, drug2 = drug_comb.split("/")[0], drug_comb.split("/")[1]

		if df.loc[drug1, "summary"] == "approved" and df.loc[drug2, "summary"] == "approved":
			comb_df.loc[drug_comb, "summary"] = "approved-approved"
			comb_df.loc[drug_comb, "priority"] = 1

		elif df.loc[drug1, "summary"] == "approved" and df.loc[drug2, "in_clinical"]:
			comb_df.loc[drug_comb, "summary"] = "approved-clinical"
			comb_df.loc[drug_comb, "priority"] = 2

		elif df.loc[drug1, "in_clinical"] and df.loc[drug2, "summary"] == "approved":
			comb_df.loc[drug_comb, "summary"] = "approved-clinical"
			comb_df.loc[drug_comb, "priority"] = 2

		elif df.loc[drug1, "summary"] == "approved" and df.loc[drug2, "summary"] == "Unknown":
			comb_df.loc[drug_comb, "summary"] = "approved-unknown"
			comb_df.loc[drug_comb, "priority"] = 3

		elif df.loc[drug2, "summary"] == "approved" and df.loc[drug1, "summary"] == "Unknown":
			comb_df.loc[drug_comb, "summary"] = "approved-unknown"
			comb_df.loc[drug_comb, "priority"] = 3

		elif df.loc[drug1, "summary"] == "approved" and df.loc[drug2, "summary"] == "pre_clinical":
			comb_df.loc[drug_comb, "summary"] = "approved-preclinical"
			comb_df.loc[drug_comb, "priority"] = 4

		elif df.loc[drug2, "summary"] == "approved" and df.loc[drug2, "summary"] == "pre_clinical":
			comb_df.loc[drug_comb, "summary"] = "approved-preclinical"
			comb_df.loc[drug_comb, "priority"] = 4

		elif df.loc[drug1, "in_clinical"] and df.loc[drug2, "in_clinical"]:
			comb_df.loc[drug_comb, "summary"] = "clinical-clinical"
			comb_df.loc[drug_comb, "priority"] = 5

		elif df.loc[drug1, "in_clinical"] and df.loc[drug2, "summary"] == "Unknown":
			comb_df.loc[drug_comb, "summary"] = "clinical-unknown"
			comb_df.loc[drug_comb, "priority"] = 6

		elif df.loc[drug2, "in_clinical"] and df.loc[drug1, "summary"] == "Unknown":
			comb_df.loc[drug_comb, "summary"] = "clinical-unknown"
			comb_df.loc[drug_comb, "priority"] = 6

		elif df.loc[drug1, "in_clinical"] and df.loc[drug2, "summary"] == "pre_clinical":
			comb_df.loc[drug_comb, "summary"] = "clinical-preclinical"
			comb_df.loc[drug_comb, "priority"] = 7

		elif df.loc[drug2, "in_clinical"] and df.loc[drug1, "summary"] == "pre_clinical":
			comb_df.loc[drug_comb, "summary"] = "clinical-preclinical"
			comb_df.loc[drug_comb, "priority"] = 7

		elif df.loc[drug1, "summary"] == "pre_clinical" and df.loc[drug2, "summary"] == "pre_clinical":
			comb_df.loc[drug_comb, "summary"] = "preclinical-preclinical"
			comb_df.loc[drug_comb, "priority"] = 8

	return df, comb_df


def x_3d(cid):
	if "%s.sdf" % cid not in os.listdir(output_path + "drug_info/3D/"):
		cid_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/SDF" % str(int(cid))
		sdf_file = requests.get(cid_url)
		if sdf_file.status_code == 200:
			if sdf_file is not None:
				f = open(output_path + "data/drug_info/3D/%s.sdf" % str(int(cid)), "w")
				f.write(sdf_file.text)
				f.close()
	return True


# ---------------------------------------------------------------------------#

# Drug Object

class Drug:
	def __init__(self, drug_name):
		drug_df = make_drug_df()
		self.drug_name = drug_name
		self.drug_id = drug_df[drug_df.drug_name == self.drug_name]["drug_id"].unique()
		self.targets = None
		self.target_pathways = None
		self.drug_type = None
		if self.drug_name in drug_df.drug_name.unique():
			if type(drug_df[drug_df.drug_name == self.drug_name]["drug_gene_target"].unique()[0]) != float:
				self.targets = drug_df[drug_df.drug_name == self.drug_name]["drug_gene_target"].unique()[0].split(";")
			if type(drug_df[drug_df.drug_name == self.drug_name]["drug_pathway_target"].unique()[0]) != float:
				self.target_pathways = drug_df[drug_df.drug_name == self.drug_name]["drug_pathway_target"].unique()
			if type(drug_df[drug_df.drug_name == self.drug_name]["drug_type"].unique()) != float:
				self.drug_type = drug_df[drug_df.drug_name == self.drug_name]["drug_type"].unique()
		try:
			self.cid = drug_df[drug_df.drug_name == self.drug_name]["cid"].unique()[0]
			self.sid = drug_df[drug_df.drug_name == self.drug_name]["sid"].unique()[0]
			self.chembl = drug_df[drug_df.drug_name == self.drug_name]["chembl"].unique()[0]
			self.molecular_formula = drug_df[drug_df.drug_name == self.drug_name]["molecular_formula"].unique()[0]
			self.molecular_weight = drug_df[drug_df.drug_name == self.drug_name]["molecular_weight"].unique()[0]
			self.exact_mass = drug_df[drug_df.drug_name == self.drug_name]["exact_mass"].unique()[0]
			self.canonical_smiles = drug_df[drug_df.drug_name == self.drug_name]["canonical_smiles"].unique()[0]
			self.inchikey = drug_df[drug_df.drug_name == self.drug_name]["inchikey"].unique()[0]
			self.fingerprint = drug_df[drug_df.drug_name == self.drug_name]["fingerprint"].unique()[0]
			self.cactvs_fingerprint = drug_df[drug_df.drug_name == self.drug_name]["cactvs_fingerprint"].unique()[0]
			self.approval = drug_df[drug_df.drug_name == self.drug_name]["approval"].unique()[0]
			self.withdrawn = drug_df[drug_df.drug_name == self.drug_name]["withdrawn"].unique()[0]
			self.clinical_phase = drug_df[drug_df.drug_name == self.drug_name]["clinical_phase"].unique()[0]
			self.clinical_trials = drug_df[drug_df.drug_name == self.drug_name]["clinical_trials"].unique()[0]
			self.MoA = drug_df[drug_df.drug_name == self.drug_name]["MoA"].unique()[0]
		except IndexError:
			self.cid, self.sid, self.chembl = None, None, None
			self.molecular_formula, self.molecular_weight = None, None
			self.exact_mass, self.canonical_smiles, self.inchikey = None, None, None
			self.fingerprint, self.cactvs_fingerprint = None, None
			self.approval, self.withdrawn, self.clinical_phase, self.clinical_trials = None, None, None, None
			self.MoA = None