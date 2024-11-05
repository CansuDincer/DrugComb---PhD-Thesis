# ---------------------------------------------------------------------------#
#       		 C o m b D r u g - R E P R O D U C I B I L I T Y			 #
# ---------------------------------------------------------------------------#

"""
# ---------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 15 September 2022
Last Update : 2 May 2024
Input : Replicate and screen files
Output : Merged viability and combination files
# ---------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, pickle, numpy, scipy, itertools, argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

from CombDrug.module.path import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.drug import get_drug_ids, drug_id2name
from CombDrug.module.data.cancer_model import *


def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-rep_type", dest="REP_TYPE")
	parser.add_argument("-treatment", dest="TREATMENT")
	parser.add_argument("-merge", dest="MERGE")
	parser.add_argument("-trdel", dest="TR_DEL")
	parser.add_argument("-brdel", dest="BR_DEL")
	parser.add_argument("-run", dest="RUN_FOR")
	parser.add_argument("-anchor", dest="ANCHOR_DOSE")
	parser.add_argument("-subproject", dest="SUBPROJECT")
	parser.add_argument("-combo_del", dest="COMBI_DEL")
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


project_base_name = {"pancancer": "Pancan_combi_12Jan21", "breast": "GDSC_Breast_09Sep21",
					 "colo": "GDSC_Colo_13Sep21", "pancreas": "GDSC_Panc_09Sep21",
					 "gdsc7": "GDSC_007_to_AL_18Mar22",
					 "gdsc8": "GDSC_008_to_AL_18Mar22",
					 "gdsc9": "GDSC_009_to_AL_26Apr22",
					 "gdsc10": "GDSC_010-B_to_AL_22Mar22"}


# ---------------------------------------------------------------------------#
#                              HELPER FUNCTIONS		 			             #
# ---------------------------------------------------------------------------#


def plot_density_reg(df, col1, col2, data_point_name, num_data_points, r2, axs):
	data = numpy.vstack([df[col1], df[col2]])
	gauss = scipy.stats.gaussian_kde(data)(data)
	idx = gauss.argsort()
	x, y, z = df[col1].values[idx], df[col2].values[idx], gauss[idx]

	axs.scatter(x, y, c=z, s=8, label="%s (%d)" % (data_point_name, num_data_points))
	axs.set_xlabel(col1)
	axs.set_ylabel(col2)
	axs.legend(title="$R^2$ %.3f" % r2, loc="upper left")
	return axs


def test_Z(values, pvalue, selected_side):
	x = numpy.array(values)
	x = x[~numpy.isnan(x)]
	min_outlier = max(x)
	m, sd = scipy.stats.norm.fit(x)
	for v in x:
		z_score = (v - m) / sd
		p_value = scipy.stats.norm.sf(abs(z_score)) * 2
		if p_value < pvalue:
			if v < min_outlier:
				if selected_side == "right":
					if v > m:
						min_outlier = v
				else:
					min_outlier = v
	return min_outlier


# ---------------------------------------------------------------------------#
#        		Replicate analysis functions - Viabilities		 			 #
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
			df = get_viability(project)[["RESEARCH_PROJECT", "BARCODE", "DATE_CREATED", "SCAN_DATE",
										 "CELL_LINE_NAME", "maxc", "CELL_ID", "POSITION",
										 "treatment", "ANCHOR_ID", "LIBRARY_ID",
										 "ANCHOR_CONC", "LIBRARY_CONC", "VIAB_COMBI_UNCAPPED"]]
			df2 = df[df.RESEARCH_PROJECT == "GDSC_Colo-2"]
			df1 = df[df.RESEARCH_PROJECT == "GDSC_Colo"]
			df1["DATE_CREATED"] = df1.apply(
				lambda x: "-".join(x.SCAN_DATE.split("T")[0].split("-")[:2]) + "-" +
						  str(int(x.SCAN_DATE.split("T")[0].split("-")[-1]) - 3), axis=1)

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

		missing_drugs = [i for i in list(set(all_via_drugs).difference(set(all_combi_drugs))) if
						 pandas.isna(i) is False]
		missing_drugs = list(set(missing_drugs))

		pickle.dump(missing_combs,
					open(output_path + "data/curves/combi_files/missing_drugs_combinations/missing_combs_%s.p"
						 % project, "wb"))
		pickle.dump(missing_drugs,
					open(output_path + "data/curves/combi_files/missing_drugs_combinations/missing_drugs_%s.p"
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

		df = df[df.missing_drug == False]

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
			df["plate_num"] = df.apply(
				lambda x: str(x["BARCODE"])[:-1] if x.RESEARCH_PROJECT in matrix_projects else str(x["BARCODE"]),
				axis=1)

			df["exp_date"] = df.apply(lambda x: str(x.DATE_CREATED), axis=1)
			df = df[[col for col in df.columns if col not in ["BARCODE", "DATE_CREATED"]]]

			df.rename(columns={"RESEARCH_PROJECT": "subproject", "barcode": "barcode",
							   "plate_num": "plate_num", "exp_date": "exp_date", "CELL_LINE_NAME": "cell_line",
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

			df["barcode"] = df.apply(lambda x: int(x.BARCODE), axis=1)
			df["plate_num"] = df.apply(
				lambda x: str(x["BARCODE"])[:-1] if x.RESEARCH_PROJECT in matrix_projects else str(x["BARCODE"]),
				axis=1)
			df = df[[col for col in df.columns if col not in ["BARCODE"]]]

			df.rename(columns={"RESEARCH_PROJECT": "subproject", "barcode": "barcode",
							   "plate_num": "plate_num", "DATE_CREATED": "exp_date",
							   "CELL_LINE_NAME": "cell_line", "SIDM": "SIDM", "CELL_ID": "cell_id",
							   "POSITION": "position", "maxc": "maxc", "treatment": "treatment",
							   "library_id": "library_id", "library_dose": "library_dose",
							   "VIAB_COMBI_UNCAPPED": "viability",
							   "library_name": "library_name", "tissue": "tissue"}, inplace=True)

		df.to_csv(output_folder + "annotated_%s_%s_viability_df.csv" % (project, treatment), index=False)

	else:
		df = pandas.read_csv(output_folder + "annotated_%s_%s_viability_df.csv" % (project, treatment))

	return df


def get_all_viabilities(treatment):
	"""
	:param treatment:
	:return:
	"""
	output_folder = output_path + "data/curves/viability_files/%s/" % treatment
	if "all_%s_normalised_viability.csv" % treatment not in os.listdir(output_path + "data/curves/viability_files/"):
		viability_dfs = list()
		for project in project_base_name.keys():
			print(project)
			df = annotate_viabilities(treatment=treatment, project=project)

			if "paired_annotated_%s_%s_viability_df.csv" % (project, treatment) not in os.listdir(output_folder):
				if treatment == "combination":
					matrix = pandas.DataFrame(
						columns=["subproject", "cell_line", "SIDM", "cell_id", "position", "barcode", "exp_date",
								 "D1", "D1_id", "D2", "D2_id", "Do1", "Do2", "viability", "tissue"])

					for pair in all_screened_combinations(project_list=[project], integrated=False):
						pair1, pair2 = pair.split("/")[0], pair.split("/")[1]
						if pair1 in list(df["anchor_name"]) and pair2 in list(df["library_name"]):
							pair_df_1 = df[(df.anchor_name == pair1) & (df.library_name == pair2)]
							pair_df_1["D1"], pair_df_1["D2"] = pair1, pair2
							pair_df_1["D1_id"], pair_df_1["D2_id"] = pair_df_1["anchor_id"], pair_df_1["library_id"]
							pair_df_1["Do1"] = pair_df_1["anchor_dose"]
							pair_df_1["Do2"] = pair_df_1["library_dose"]
							pair_df_1["orientation"] = "A-L"
							pair_df_1 = pair_df_1[["subproject", "cell_line", "SIDM", "cell_id", "position",
												   "barcode", "exp_date", "D1", "D1_id", "D2", "D2_id", "Do1",
												   "Do2", "viability", "tissue", "orientation"]]
							matrix = pandas.concat([matrix, pair_df_1])

						if pair2 in list(df["anchor_name"]) and pair1 in list(df["library_name"]):
							pair_df_2 = df[(df.anchor_name == pair2) & (df.library_name == pair1)]
							pair_df_2["D1"], pair_df_2["D2"] = pair1, pair2
							pair_df_2["D1_id"], pair_df_2["D2_id"] = pair_df_2["library_id"], pair_df_2["anchor_id"]
							pair_df_2["Do1"] = pair_df_2["library_dose"]
							pair_df_2["Do2"] = pair_df_2["anchor_dose"]
							pair_df_2["orientation"] = "L-A"
							pair_df_2 = pair_df_2[["subproject", "cell_line", "SIDM", "cell_id", "position",
												   "barcode", "exp_date", "D1", "D1_id", "D2", "D2_id", "Do1",
												   "Do2", "viability", "tissue", "orientation"]]
							matrix = pandas.concat([matrix, pair_df_2])

					matrix["project"] = project

				else:
					matrix = pandas.DataFrame(
						columns=["subproject", "cell_line", "SIDM", "cell_id", "position", "barcode", "exp_date",
								 "Drug", "Drug_id", "Drug_dose", "viability", "tissue"])
					for drug in all_screened_compounds(project_list=[project], integrated=False):
						if drug in list(df["library_name"]):
							drug_df = df[df.library_name == drug]
							drug_df["Drug"] = drug
							drug_df["Drug_id"] = drug_df["library_id"]
							drug_df["Drug_dose"] = drug_df["library_dose"]

							matrix = pandas.concat([matrix, drug_df])

					matrix["project"] = project

				if matrix is not None:
					matrix = matrix.reset_index()[[i for i in matrix.columns if i != "index"]]
					matrix.to_csv(output_folder + "paired_annotated_%s_%s_viability_df.csv"
								  % (project, treatment), index=True)
			else:
				matrix = pandas.read_csv(output_folder + "paired_annotated_%s_%s_viability_df.csv"
										 % (project, treatment), index_col=0)

			viability_dfs.append(matrix)

		viability_df = pandas.concat(viability_dfs)
		viability_df = viability_df.reset_index()[[i for i in viability_df.columns if i != "index"]]
		viability_df["plate_num"] = viability_df.apply(
			lambda x: str(x["barcode"])[:-1] if x.subproject in matrix_projects else str(x["barcode"]), axis=1)

		viability_df = viability_df.drop(["barcode"], axis=1)
		viability_df.to_csv(output_path + "data/curves/viability_files/all_%s_normalised_viability.csv"
							% treatment, index=True)

	else:
		viability_df = pandas.read_csv(output_path + "data/curves/viability_files/all_%s_normalised_viability.csv"
									   % treatment, index_col=0)

	return viability_df


def count_replicates(replicate_type, treatment, type_of_merge, answer_t):
	"""
	Counting and extracting statistical variances across technical/biological replicates
	:param replicate_type: technical/biological
	:param treatment: Combination or single
	:param type_of_merge: median/mean merging replicates after analysis
	:param answer_t: If the replicate_type is biological, then it is the technical replicate merging method.
	:return: Data Frame with the number of replicates for each drug-dose pairs
	"""

	if answer_t:
		del_all_t = True
	else:
		del_all_t = False

	if "%s_%s_replicate_count.csv" % (replicate_type, treatment) not in os.listdir(
			replicate_path + "%s_replicates/statistics/" % replicate_type):

		if replicate_type == "technical":
			# For technical replicate, read the normalised viability scores with get_all_viabilities()
			viability_df = get_all_viabilities(treatment=treatment)

			if treatment == "combination":

				# Number of barcodes representing drug and dose pairs
				# Standard deviation of the viability of the same drug and dose pairs across barcodes
				stat_df = pandas.DataFrame(columns=["cell_line", "SIDM", "cell_id", "D1", "D2",
													"Do1", "Do2", "mean_viability", "median_viability",
													"max_viability", "min_viability", "resistant_response", "SD", "CV",
													"#barcodes", "project", "subproject", "exp_date", "tissue"])

				# Check the number of barcodes per project per exp_date
				rows = list()
				count, total = 0, len(
					viability_df.groupby(["SIDM", "exp_date", "subproject", "D1", "D2", "Do1", "Do2"]))
				for rep_name, rep_df in viability_df.groupby(["SIDM", "exp_date", "subproject",
															  "D1", "D2", "Do1", "Do2"]):

					if len(rep_df.index) > 1:
						sd = numpy.std(rep_df["viability"])
						if numpy.mean(rep_df["viability"]) == 0:
							cv = 0
						else:
							cv = (sd / numpy.mean(rep_df["viability"]))
					else:
						sd, cv = None, None
					rows.append({"cell_line": sanger2model(rep_name[0]), "SIDM": rep_name[0],
								 "cell_id": CellLine(rep_name[0]).cell_id,
								 "exp_date": rep_name[1], "subproject": rep_name[2],
								 "project": rep_df["project"].unique()[0],
								 "D1": rep_name[3], "D2": rep_name[4], "Do1": rep_name[5], "Do2": rep_name[6],
								 "#replicates": len(rep_df.index),
								 "#barcodes": len(rep_df["plate_num"].unique()),
								 "tissue": CellLine(sanger2model(rep_name[0])).tissue,
								 "mean_viability": numpy.mean(rep_df["viability"]),
								 "median_viability": numpy.median(rep_df["viability"]),
								 "min_viability": numpy.min(rep_df["viability"]),
								 "max_viability": max(rep_df["viability"]), "SD": sd, "CV": cv,
								 "resistant_response": len([i for i in rep_df.viability.values if i > 0.7]) * 100.0 /
													   len(rep_df.viability.values)})
					count += 1
					print(count * 100.0 / total)

				all_rep_df = pandas.DataFrame.from_dict(rows, orient="columns")
				stat_df = pandas.concat([stat_df, all_rep_df])

			else:
				# Number of barcodes representing drug and dose pairs
				# Standard deviation of the viability of the same drug and dose pairs across barcodes
				stat_df = pandas.DataFrame(columns=["cell_line", "SIDM", "cell_id", "Drug", "Drug_dose",
													"mean_viability", "median_viability", "resistant_response",
													"max_viability", "min_viability", "SD", "CV", "#barcodes",
													"project", "subproject", "exp_date", "tissue"])

				# Check the number of barcodes per project per exp_date
				rows = list()
				for rep_name, rep_df in viability_df.groupby(["SIDM", "exp_date", "subproject",
															  "Drug", "Drug_dose"]):
					if len(rep_df.index) > 1:
						sd = numpy.std(rep_df["viability"])
						if numpy.mean(rep_df["viability"]) == 0:
							cv = 0
						else:
							cv = (sd / numpy.mean(rep_df["viability"]))
					else:
						sd, cv = None, None

					rows.append({"cell_line": sanger2model(rep_name[0]), "SIDM": rep_name[0],
								 "cell_id": CellLine(rep_name[0]).cell_id,
								 "exp_date": rep_name[1], "subproject": rep_name[2],
								 "project": rep_df["project"].unqiue()[0],
								 "Drug": rep_name[3], "Drug_dose": rep_name[4],
								 "#replicates": len(rep_df.index),
								 "#barcodes": len(rep_df["plate_num"].unique()),
								 "tissue": CellLine(sanger2model(rep_name[0])).tissue,
								 "mean_viability": numpy.mean(rep_df["viability"]),
								 "median_viability": numpy.median(rep_df["viability"]),
								 "min_viability": numpy.min(rep_df["viability"]),
								 "max_viability": max(rep_df["viability"]), "SD": sd, "CV": cv,
								 "resistant_response": len([i for i in rep_df.viability.values if i > 0.7]) * 100.0 /
													   len(rep_df.viability.values)})

				all_rep_df = pandas.DataFrame.from_dict(rows, orient="columns")
				stat_df = pandas.concat([stat_df, all_rep_df])

		if replicate_type == "biological":

			from CombDrug.module.data.viability import Viability, deserialise_viability_object

			# For biological replicate, read the call_mid_viability
			viability_df = deserialise_viability_object(treatment=treatment, status="mid",
														type_of_merge=type_of_merge, del_rep_t=del_all_t,
														del_rep_b=False)["combination"].mid_viability
			print(viability_df)
			if treatment == "combination":

				# Number of barcodes representing drug and dose pairs
				# Standard deviation of the viability of the same drug and dose pairs across barcodes
				stat_df = pandas.DataFrame(columns=["cell_line", "SIDM", "cell_id", "D1", "D2",
													"Do1", "Do2", "mean_viability", "median_viability",
													"max_viability", "min_viability", "resistant_response", "SD", "CV",
													"#exp_date", "project", "subproject", "tissue"])

				# Check the number of barcodes per project per exp_date
				rows = list()
				count, total = 0, len(
					viability_df.groupby(["SIDM", "subproject", "D1", "D2", "Do1", "Do2"]))
				for rep_name, rep_df in viability_df.groupby(["SIDM", "subproject", "D1", "D2", "Do1", "Do2"]):
					if len(rep_df.index) > 1:
						sd = numpy.std(rep_df["viability"])
						if numpy.mean(rep_df["viability"]) == 0:
							cv = 0
						else:
							cv = (sd / numpy.mean(rep_df["viability"]))
					else:
						sd, cv = None, None
					rows.append({"cell_line": sanger2model(rep_name[0]), "SIDM": rep_name[0],
								 "cell_id": CellLine(rep_name[0]).cell_id,
								 "subproject": rep_name[1],
								 "project": rep_df["project"].unique()[0],
								 "D1": rep_name[2], "D2": rep_name[3], "Do1": rep_name[4], "Do2": rep_name[5],
								 "#replicates": len(rep_df.index),
								 "#exp_date": len(rep_df["exp_date"].unique()),
								 "tissue": CellLine(rep_name[0]).tissue,
								 "mean_viability": numpy.mean(rep_df["viability"]),
								 "median_viability": numpy.median(rep_df["viability"]),
								 "min_viability": numpy.min(rep_df["viability"]),
								 "max_viability": max(rep_df["viability"]), "SD": sd, "CV": cv,
								 "resistant_response": len(
									 [i for i in rep_df.viability.values if i > 0.7]) * 100.0 / len(
									 rep_df.viability.values)})
					count += 1
					print(count * 100.0 / total)

				all_rep_df = pandas.DataFrame.from_dict(rows, orient="columns")
				stat_df = pandas.concat([stat_df, all_rep_df])

			else:
				# Number of exp dates representing drug and dose pairs
				# Standard deviation of the viability of the same drug and dose pairs across exp dates
				stat_df = pandas.DataFrame(columns=["cell_line", "SIDM", "cell_id", "Drug", "Drug_dose",
													"mean_viability", "median_viability", "resistant_response",
													"max_viability", "min_viability", "SD", "CV", "#exp_date",
													"project", "subproject", "tissue"])

				# Check the number of barcodes per project per exp_date
				rows = list()
				for rep_name, rep_df in viability_df.groupby(["SIDM", "subproject", "Drug", "Drug_dose"]):
					if len(rep_df.index) > 1:
						sd = numpy.std(rep_df["viability"])
						if numpy.mean(rep_df["viability"]) == 0:
							cv = 0
						else:
							cv = (sd / numpy.mean(rep_df["viability"]))
					else:
						sd, cv = None, None
					rows.append({"cell_line": sanger2model(rep_name[0]), "SIMD": rep_name[0],
								 "cell_id": CellLine(rep_name[0]).cell_id,
								 "subproject": rep_name[1],
								 "project": rep_df["project"].unqiue()[0],
								 "Drug": rep_name[2], "Drug_dose": rep_name[3],
								 "#replicates": len(rep_df.index),
								 "#exp_date": len(rep_df["exp_date"].unique()),
								 "tissue": CellLine(rep_name[0]).tissue,
								 "mean_viability": numpy.mean(rep_df["viability"]),
								 "median_viability": numpy.median(rep_df["viability"]),
								 "min_viability": numpy.min(rep_df["viability"]),
								 "max_viability": max(rep_df["viability"]), "SD": sd, "CV": cv,
								 "resistant_response": len([i for i in rep_df.viability.values if i > 0.7]) * 100.0 /
													   len(rep_df.viability.values)})

				all_rep_df = pandas.DataFrame.from_dict(rows, orient="columns")
				stat_df = pandas.concat([stat_df, all_rep_df])

		stat_df.to_csv(replicate_path + "%s_replicates/statistics/%s_%s_replicate_count.csv"
					   % (replicate_type, replicate_type, treatment), index=False)

	else:
		stat_df = pandas.read_csv(replicate_path + "%s_replicates/statistics/%s_%s_replicate_count.csv"
								  % (replicate_type, replicate_type, treatment))
	return stat_df


def plot_variance(replicate_type, treatment, type_of_merge, answer_t):
	"""
	Plot the variance across replicates
	:param replicate_type: technical/biological
	:param treatment: Combination or single
	:param type_of_merge: median/mean merging replicates after analysis
	:param answer_t: If the replicate_type is biological, then it is the technical replicate merging method.
	:return:
	"""

	if replicate_type == "biological":
		title = "_" + type_of_merge + "_merged"
	else:
		title = ""

	stat_df = count_replicates(replicate_type, treatment, type_of_merge, answer_t)

	output_folder = output_path + "data/replicates/%s_replicates/figures/" % replicate_type
	# CV BoxPlot and Density Distribution
	cv_project_dict = {}
	for subproject, subproject_df in stat_df.groupby(["subproject"]):
		cv_project_dict[subproject] = list(subproject_df[~numpy.isnan(subproject_df["CV"])]["CV"])

	flierprops = dict(marker=".", markerfacecolor="black", markersize=1.7,
					  markeredgecolor="none")
	medianprops = dict(linestyle="-", linewidth=2.0, color="red")

	plt.figure(facecolor="white")
	plt.title("Coefficient of variance of the %s replicates" % replicate_type)

	plt.boxplot(cv_project_dict.values(), medianprops=medianprops, flierprops=flierprops,
				labels=cv_project_dict.keys())

	plt.xticks(rotation=90, fontsize=8)
	plt.xlabel("Sub-projects")
	plt.ylabel("Coefficient of variance %")
	plt.tight_layout()
	plt.savefig(output_folder + "%s_replicate_%s%s_CV_by_subproject.pdf"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_CV_by_subproject.jpg"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_CV_by_subproject.png"
				% (replicate_type, treatment, title), dpi=300)
	plt.close()

	g = sns.displot(data=stat_df, x="CV", col="subproject", kde=True, stat="density", common_bins=False,
					common_norm=False, height=4, facet_kws={'sharey': False, 'sharex': False}, col_wrap=5)
	g.set_titles("{col_name}")
	g.set_axis_labels(x_var="CV %", y_var="Density")
	plt.suptitle("Coefficient of Variance - %s Replicates" % replicate_type)
	plt.tight_layout()
	plt.savefig(output_folder + "%s_replicate_%s%s_density_CV_by_subproject.pdf"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_density_CV_by_subproject.jpg"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_density_CV_by_subproject.png"
				% (replicate_type, treatment, title), dpi=300)
	plt.close()

	# SD BoxPlot and Density Distribution
	sd_project_dict = {}
	for subproject, subproject_df in stat_df.groupby(["subproject"]):
		sd_project_dict[subproject] = list(subproject_df[~numpy.isnan(subproject_df["SD"])]["SD"])

	plt.figure(facecolor="white")
	plt.title("Standard Deviation of the %s replicates" % replicate_type)

	plt.boxplot(sd_project_dict.values(), medianprops=medianprops, flierprops=flierprops,
				labels=sd_project_dict.keys())

	plt.xticks(rotation=90, fontsize=8)
	plt.xlabel("Sub-projects")
	plt.ylabel("Standard Deviation")
	plt.tight_layout()
	plt.savefig(output_folder + "%s_replicate_%s%s_SD_by_subproject.pdf"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_SD_by_subproject.jpg"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_SD_by_subproject.png"
				% (replicate_type, treatment, title), dpi=300)
	plt.close()

	g = sns.displot(data=stat_df, x="SD", col="subproject", kde=True, stat="density", common_bins=False,
					common_norm=False, height=4, facet_kws={'sharey': False, 'sharex': False}, col_wrap=5)
	g.set_titles("{col_name}")
	g.set_axis_labels(x_var="SD", y_var="Density")
	plt.suptitle("Standard Deviation - %s Replicates" % replicate_type)
	plt.tight_layout()
	plt.savefig(output_folder + "%s_replicate_%s%s_density_SD_by_subproject.pdf"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_density_SD_by_subproject.jpg"
				% (replicate_type, treatment, title), dpi=300)
	plt.savefig(output_folder + "%s_replicate_%s%s_density_SD_by_subproject.png"
				% (replicate_type, treatment, title), dpi=300)
	plt.close()

	return True


def regression_replicates(replicate_type, treatment, plotting, type_of_merge, answer_t):
	"""
	Calculate the similarity across replicates
	:param replicate_type: technical/biological
	:param treatment: Combination or single
	:param plotting: Boolean
	:param type_of_merge: median/mean merging replicates after analysis
	:param answer_t: If the replicate_type is biological, then it is the technical replicate merging method.
	:return:
	"""

	if answer_t:
		del_all_t = True
	else:
		del_all_t = False

	if replicate_type == "biological":
		title = "_" + type_of_merge + "_merged"
	else:
		title = ""

	output_folder = output_path + "data/replicates/%s_replicates/" % replicate_type
	if "%s_replicate_%s%s_regression.p" % (replicate_type, treatment, title) not in os.listdir(
			output_folder + "statistics/"):

		if replicate_type == "technical":

			viability_df = get_all_viabilities(treatment=treatment)

			if "pre_%s_replicate_%s_regression.p" % (replicate_type, treatment) not in os.listdir(
					output_folder + "statistics/"):
				rep_regression_stat = {}
				for group, group_df in viability_df.groupby(["subproject", "project", "exp_date"]):
					subproject, project, exp_date = group[0], group[1], group[2]

					if len(set(group_df["plate_num"])) > 1:
						pivot_df = group_df.pivot(columns="plate_num", values="viability",
												  index=[col for col in group_df.columns
														 if col not in ["plate_num", "viability"]])
						all_plates = pivot_df.columns
						plate_pair_df = {}
						for bc in itertools.combinations(all_plates, 2):
							t = pivot_df[list(bc)]
							q = t.dropna(axis=0)
							if len(q.index) != 0:
								plate_pair_df[bc] = q.reset_index()

						plate_combs = list(plate_pair_df.keys())
						if len(plate_combs) < 4 and len(plate_combs) != 0:
							if plotting:
								if len(plate_pair_df.keys()) == 1:
									fig, ax = plt.subplots(1, len(plate_pair_df.keys()), sharex=True,
														   sharey=True, squeeze=False, figsize=(5, 5))

								else:
									fig, ax = plt.subplots(1, len(plate_pair_df.keys()), sharex=True,
														   sharey=True, squeeze=False,
														   gridspec_kw={
															   "width_ratios": [1] * len(plate_pair_df.keys())},
														   figsize=(5 * len(plate_pair_df.keys()), 5))

							for i in range(len(plate_pair_df.keys())):
								df = plate_pair_df[plate_combs[i]]
								if len(df.index) != 0:
									plates = [plate_combs[i][0], plate_combs[i][1]]
									num_data_points = len(df.index)
									slope, intercept, r_value, p_value, _ = \
										scipy.stats.linregress(df[plates[0]], df[plates[1]])

									if subproject not in rep_regression_stat.keys():
										rep_regression_stat[subproject] = {
											exp_date: {plate_combs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}}
									else:
										if exp_date not in rep_regression_stat[subproject].keys():
											rep_regression_stat[subproject][exp_date] = {
												plate_combs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}
										else:
											if plate_combs[i] not in rep_regression_stat[subproject][exp_date].keys():
												rep_regression_stat[subproject][exp_date][plate_combs[i]] = {
													"rvalue": r_value ** 2, "r2pvalue": p_value}
									if plotting:
										ax[0, i].plot(df[plates[0]], df[plates[1]], 'o', alpha=0.4, markersize=3,
													  label="Viability # data point %d" % num_data_points)
										ax[0, i].plot(df[plates[0]], intercept + slope * df[plates[0]],
													  'r', label="Fitted Line")

										onetoone = numpy.linspace(min(df[plates[0]]), max(df[plates[0]]), 10)
										ax[0, i].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")
										ax[0, i].set_xlabel(plates[0])
										ax[0, i].set_ylabel(plates[1])
										legnd = ax[0, i].legend(loc='upper left',
																title="y = %.2f + %.2f x\n$R^2$: %.2f" % (
																intercept, slope, r_value ** 2))
										"""
										fig.canvas.draw()
										h = ax[0, i].transData.inverted().transform(legnd.get_window_extent())
										ax[0, i].text(x=h[0][0], y=(h[0][1]) - 0.05,
													  s=r"y = %.2f + %.2f x" % (intercept, slope), transform=ax[0, i].transAxes)
										ax[0, i].text(x=h[0][0], y=(h[0][1]) - 0.1, s=r"R square: %.2f" % r_value ** 2,
													  transform=ax[0, i].transAxes)
										"""

								else:
									fig.delaxes(ax[0, i])

						elif len(plate_combs) >= 4 and len(plate_combs) != 0:
							if plotting:
								s = (len(plate_pair_df.keys()) // 4) + 1

								fig, ax = plt.subplots((len(plate_pair_df.keys()) // 4) + 1, 4,
													   sharey=True, sharex=True, figsize=(16, 4 * s))

							plate_combs = list(plate_pair_df.keys())
							for i in range(len(plate_pair_df.keys())):
								df = plate_pair_df[plate_combs[i]]
								if len(df.index) != 0:
									plates = [plate_combs[i][0], plate_combs[i][1]]
									num_data_points = len(df.index)
									slope, intercept, r_value, p_value, _ = \
										scipy.stats.linregress(df[plates[0]], df[plates[1]])

									if subproject not in rep_regression_stat.keys():
										rep_regression_stat[subproject] = {
											exp_date: {plate_combs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}}
									else:
										if exp_date not in rep_regression_stat[subproject].keys():
											rep_regression_stat[subproject][exp_date] = {
												plate_combs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}
										else:
											if plate_combs[i] not in rep_regression_stat[subproject][exp_date].keys():
												rep_regression_stat[subproject][exp_date][plate_combs[i]] = {
													"rvalue": r_value ** 2, "r2pvalue": p_value}
								if plotting:
									a = i // 4
									if i < 4:
										b = i
									else:
										b = i - (4 * (i // 4))
									if len(df.index) != 0:
										ax[a, b].set_xlim([0.1, 1])
										ax[a, b].set_ylim([0.1, 1])

										ax[a, b].plot(df[plates[0]], df[plates[1]], 'o', alpha=0.4, markersize=3,
													  label="Viability # data point %d" % num_data_points)
										ax[a, b].plot(df[plates[0]], intercept + slope * df[plates[0]],
													  'r', label="Fitted Line")

										onetoone = numpy.linspace(min(df[plates[0]]), max(df[plates[0]]), 10)
										ax[a, b].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")
										ax[a, b].set_xlabel(plates[0])
										ax[a, b].set_ylabel(plates[1])
										legnd = ax[a, b].legend(loc='upper left',
																title="y = %.2f + %.2f x\n$R^2$: %.2f" % (
																intercept, slope, r_value ** 2))
										"""
										fig.canvas.draw()
										h = ax[a, b].transData.inverted().transform(legnd.get_window_extent())
										ax[a, b].text(x=h[0][0], y=(h[0][1]) - 0.05,
													  s=r"y = %.2f + %.2f x" % (intercept, slope), transform=ax[a, b].transAxes)
										ax[a, b].text(x=h[0][0], y=(h[0][1]) - 0.1, s=r"R square: %.2f" % r_value ** 2,
													  transform=ax[a, b].transAxes)
										"""
									else:
										fig.delaxes(ax[a, b])

						if len(plate_combs) != 0 and plotting:
							fig.tight_layout()
							fig.subplots_adjust(top=0.90)
							plt.suptitle("%s - LR of Technical Replicates (Barcodes)\nExperiment Date %s"
										 % (subproject, exp_date))
							try:
								plt.savefig(
									output_folder + "figures/regression_plots/barcode_pair_regressions_%s_%s_%s.pdf"
									% (treatment, subproject, exp_date), dpi=300)
								plt.savefig(
									output_folder + "figures/regression_plots/barcode_pair_regressions_%s_%s_%s.jpg"
									% (treatment, subproject, exp_date), dpi=300)
								plt.savefig(
									output_folder + "figures/regression_plots/barcode_pair_regressions_%s_%s_%s.png"
									% (treatment, subproject, exp_date), dpi=300)
								plt.close()
							except ValueError:
								plt.savefig(
									output_folder + "figures/regression_plots/barcode_pair_regressions_%s_%s_%s.pdf"
									% (treatment, subproject, exp_date), dpi=150)
								plt.savefig(
									output_folder + "figures/regression_plots/barcode_pair_regressions_%s_%s_%s.jpg"
									% (treatment, subproject, exp_date), dpi=150)
								plt.savefig(
									output_folder + "figures/regression_plots/barcode_pair_regressions_%s_%s_%s.png"
									% (treatment, subproject, exp_date), dpi=150)
								plt.close()

				pickle.dump(rep_regression_stat, open(output_folder + "statistics/pre_%s_replicate_%s_regression.p"
													  % (replicate_type, treatment), "wb"))
			else:
				rep_regression_stat = pickle.load(open(output_folder + "statistics/pre_%s_replicate_%s_regression.p"
													   % (replicate_type, treatment), "rb"))

			# R^2 distribution -- sub-project based
			subproject_based_rsquares = {}
			for subproject, d1 in rep_regression_stat.items():
				rs = []
				for exp_date, d2 in d1.items():
					for b, rsquare in d2.items():
						rs.append(rsquare["rvalue"])
				if subproject not in subproject_based_rsquares.keys():
					subproject_based_rsquares[subproject] = rs
				else:
					subproject_based_rsquares[subproject].extend(rs)

			# R^2 distribution -- Whole
			all_rsquares = [rsquare["rvalue"] for subproject, d1 in rep_regression_stat.items()
							for exp_date, d2 in d1.items() for b, rsquare in d2.items()]

			m_all, sd_all = scipy.stats.norm.fit(all_rsquares)
			for subproject, subproject_rsquares in subproject_based_rsquares.items():
				if subproject_rsquares is not None:
					# Z score calculation of the r-squares
					m, sd = scipy.stats.norm.fit(subproject_rsquares)
					for exp_date, d2 in rep_regression_stat[subproject].items():
						for plate_pair, stat in d2.items():
							if sd != 0 or sd is not None:
								# Sub project level
								z_score = (stat["rvalue"] - m) / sd
								p_value = scipy.stats.norm.sf(abs(z_score)) * 2
								rep_regression_stat[subproject][exp_date][plate_pair].update({"zscore": z_score})
								rep_regression_stat[subproject][exp_date][plate_pair].update({"pvalue": p_value})
								# All
								z_score_all = (stat["rvalue"] - m_all) / sd_all
								p_value_all = scipy.stats.norm.sf(abs(z_score_all)) * 2
								rep_regression_stat[subproject][exp_date][plate_pair].update(
									{"all_zscore": z_score_all})
								rep_regression_stat[subproject][exp_date][plate_pair].update(
									{"all_pvalue": p_value_all})

					outliers = list()
					max_outlier_rsquare = 0
					for exp_date, d2 in rep_regression_stat[subproject].items():
						for plate_pair, stat in d2.items():
							if "pvalue" in stat.keys():
								if stat["pvalue"] < 0.05:
									if stat["rvalue"] > max_outlier_rsquare:
										max_outlier_rsquare = stat["rvalue"]
									outliers.append((subproject, exp_date, plate_pair, stat["rvalue"]))

					pickle.dump(outliers, open(output_folder + "statistics/%s_replicate_%s_%s_outliers.p"
											   % (replicate_type, treatment, subproject), "wb"))

					if plotting:
						# Density Plot and Histogram of all arrival delays
						sns.distplot(subproject_rsquares, hist=True, kde=True, bins=int(180 / 5), color='darkblue',
									 hist_kws={'edgecolor': 'black'},
									 kde_kws={'linewidth': 4})
						plt.title(
							"Density Plot of %d $R^2$ of all cell lines in %s" % (len(subproject_rsquares), subproject))
						plt.xlim(0, 1.1)
						plt.axvline(x=max_outlier_rsquare, label="Upper limit for p-value 0.05",
									color="gray", ls="--", ymax=0.95)
						plt.legend()
						plt.xlabel("R-squares")
						plt.ylabel("Density")
						plt.savefig(output_folder + "figures/%s_replicate_%s_%s_rsquare_distribution.pdf"
									% (replicate_type, treatment, subproject), dpi=300)
						plt.savefig(output_folder + "figures/%s_replicate_%s_%s_rsquare_distribution.jpg"
									% (replicate_type, treatment, subproject), dpi=300)
						plt.savefig(output_folder + "figures/%s_replicate_%s_%s_rsquare_distribution.png"
									% (replicate_type, treatment, subproject), dpi=300)
						plt.close()

			outliers_all = []
			max_outlier_rsquare_all = 0
			for subproject, d1 in rep_regression_stat.items():
				for exp_date, d2 in d1.items():
					for plate_pair, stat in d2.items():
						if "all_pvalue" in stat.keys():
							if stat["all_pvalue"] < 0.05:
								if stat["rvalue"] > max_outlier_rsquare_all:
									max_outlier_rsquare_all = stat["rvalue"]
								outliers_all.append((subproject, exp_date, plate_pair, stat["rvalue"]))

			pickle.dump(outliers_all, open(output_folder + "statistics/%s_replicate_%s_outliers.p"
										   % (replicate_type, treatment), "wb"))

			pickle.dump(rep_regression_stat, open(output_folder + "statistics/%s_replicate_%s_regression.p"
												  % (replicate_type, treatment), "wb"))

			if plotting:
				# Density Plot and Histogram of all arrival delays
				sns.distplot(all_rsquares, hist=True, kde=True, bins=int(180 / 5), color='darkblue',
							 hist_kws={'edgecolor': 'black'},
							 kde_kws={'linewidth': 4})
				plt.title("Density Plot of %d $R^2$ of %s cell lines in all projects" % (len(all_rsquares), "all"))
				plt.xlim(0, 1.1)
				plt.axvline(x=max_outlier_rsquare_all, label="Upper limit for p-value 0.05",
							color="gray", ls="--", ymax=0.95)
				plt.legend()
				plt.xlabel("R-squares")
				plt.ylabel("Density")
				plt.savefig(output_folder + "figures/%s_replicate_%s_rsquare_distribution.pdf"
							% (replicate_type, treatment), dpi=300)
				plt.savefig(output_folder + "figures/%s_replicate_%s_rsquare_distribution.jpg"
							% (replicate_type, treatment), dpi=300)
				plt.savefig(output_folder + "figures/%s_replicate_%s_rsquare_distribution.png"
							% (replicate_type, treatment), dpi=300)
				plt.close()

		if replicate_type == "biological":

			from CombDrug.module.data.viability import Viability, deserialise_viability_object

			# For biological replicate, read the call_mid_viability
			viability_df = deserialise_viability_object(treatment=treatment, status="mid",
														type_of_merge=type_of_merge, del_rep_t=del_all_t,
														del_rep_b=False)["combination"].mid_viability

			if "pre_%s_replicate_%s_regression.p" % (replicate_type, treatment) not in os.listdir(
					output_folder + "statistics/"):
				rep_regression_stat = {}
				for group, group_df in viability_df.groupby(["subproject", "project"]):
					subproject, project = group[0], group[1]

					if len(set(group_df["exp_date"])) > 1:
						pivot_df = group_df.pivot(columns="exp_date", values="viability",
												  index=[col for col in group_df.columns
														 if col not in ["exp_date", "viability"]])

						all_exp_dates = pivot_df.columns
						exp_date_pair_df = {}
						for ec in itertools.combinations(all_exp_dates, 2):
							t = pivot_df[list(ec)]
							q = t.dropna(axis=0)
							if len(q.index) != 0:
								exp_date_pair_df[ec] = q.reset_index()

						exp_date_combs = list(exp_date_pair_df.keys())
						if len(exp_date_combs) < 4 and len(exp_date_combs) != 0:
							if plotting:
								if len(exp_date_pair_df.keys()) == 1:
									fig, ax = plt.subplots(1, len(exp_date_pair_df.keys()), sharex=True,
														   sharey=True, squeeze=False, figsize=(5, 5))

								else:
									fig, ax = plt.subplots(1, len(exp_date_pair_df.keys()), sharex=True,
														   sharey=True, squeeze=False,
														   gridspec_kw={
															   "width_ratios": [1] * len(exp_date_pair_df.keys())},
														   figsize=(5 * len(exp_date_pair_df.keys()), 5))

							for i in range(len(exp_date_pair_df.keys())):
								df = exp_date_pair_df[exp_date_combs[i]]
								if len(df.index) != 0:
									exp_dates = [exp_date_combs[i][0], exp_date_combs[i][1]]
									num_data_points = len(df.index)
									slope, intercept, r_value, p_value, _ = \
										scipy.stats.linregress(df[exp_dates[0]], df[exp_dates[1]])

									if subproject not in rep_regression_stat.keys():
										rep_regression_stat[subproject] = {
											exp_date_combs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}
									else:
										if exp_date_combs[i] not in rep_regression_stat[subproject].keys():
											rep_regression_stat[subproject][exp_date_combs[i]] = {
												"rvalue": r_value ** 2, "r2pvalue": p_value}

									if plotting:
										ax[0, i].plot(df[exp_dates[0]], df[exp_dates[1]], 'o', alpha=0.4, markersize=3,
													  label="Viability # data point %d" % num_data_points)
										ax[0, i].plot(df[exp_dates[0]], intercept + slope * df[exp_dates[0]],
													  'r', label="Fitted Line")

										onetoone = numpy.linspace(min(df[exp_dates[0]]), max(df[exp_dates[0]]), 10)
										ax[0, i].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")
										ax[0, i].set_xlabel(exp_dates[0])
										ax[0, i].set_ylabel(exp_dates[1])
										legnd = ax[0, i].legend(loc='upper left',
																title="y = %.2f + %.2f x\n$R^2$: %.2f" % (
																intercept, slope, r_value ** 2))
										"""
										fig.canvas.draw()
										h = ax[0, i].transData.inverted().transform(legnd.get_window_extent())
										ax[0, i].text(x=h[0][0], y=(h[0][1]) - 0.05,
													  s=r"y = %.2f + %.2f x" % (intercept, slope),
													  transform=ax[0, i].transAxes)
										ax[0, i].text(x=h[0][0], y=(h[0][1]) - 0.1, s=r"R square: %.2f" % r_value ** 2,
													  transform=ax[0, i].transAxes)
										"""

								else:
									fig.delaxes(ax[0, i])

						elif len(exp_date_combs) >= 4 and len(exp_date_combs) != 0:
							if plotting:
								s = (len(exp_date_pair_df.keys()) // 4) + 1
								fig, ax = plt.subplots((len(exp_date_pair_df.keys()) // 4) + 1, 4,
													   sharey=True, sharex=True, figsize=(16, 4 * s))

							exp_date_combs = list(exp_date_pair_df.keys())
							for i in range(len(exp_date_pair_df.keys())):
								df = exp_date_pair_df[exp_date_combs[i]]
								if len(df.index) != 0:
									exp_dates = [exp_date_combs[i][0], exp_date_combs[i][1]]
									num_data_points = len(df.index)
									slope, intercept, r_value, p_value, _ = \
										scipy.stats.linregress(df[exp_dates[0]], df[exp_dates[1]])

									if subproject not in rep_regression_stat.keys():
										rep_regression_stat[subproject] = {
											exp_date_combs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}
									else:
										if exp_date_combs[i] not in rep_regression_stat[subproject].keys():
											rep_regression_stat[subproject][exp_date_combs[i]] = {
												"rvalue": r_value ** 2, "r2pvalue": p_value}

								if plotting:
									a = i // 4
									if i < 4:
										b = i
									else:
										b = i - (4 * (i // 4))
									if len(df.index) != 0:
										ax[a, b].set_xlim([0.1, 1])
										ax[a, b].set_ylim([0.1, 1])

										ax[a, b].plot(df[exp_dates[0]], df[exp_dates[1]], 'o', alpha=0.4, markersize=3,
													  label="Viability # data point %d" % num_data_points)
										ax[a, b].plot(df[exp_dates[0]], intercept + slope * df[exp_dates[0]],
													  'r', label="Fitted Line")

										onetoone = numpy.linspace(min(df[exp_dates[0]]), max(df[exp_dates[0]]), 10)
										ax[a, b].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")
										ax[a, b].set_xlabel(exp_dates[0])
										ax[a, b].set_ylabel(exp_dates[1])
										legnd = ax[a, b].legend(loc='upper left',
																title="y = %.2f + %.2f x\n$R^2$: %.2f" % (
																intercept, slope, r_value ** 2))
										"""
										fig.canvas.draw()
										h = ax[a, b].transData.inverted().transform(legnd.get_window_extent())
										ax[a, b].text(x=h[0][0], y=(h[0][1]) - 0.05,
													  s=r"y = %.2f + %.2f x" % (intercept, slope),
													  transform=ax[a, b].transAxes)
										ax[a, b].text(x=h[0][0], y=(h[0][1]) - 0.1, s=r"R square: %.2f" % r_value ** 2,
													  transform=ax[a, b].transAxes)
										"""
									else:
										fig.delaxes(ax[a, b])

						if len(exp_date_combs) != 0 and plotting:
							fig.tight_layout()
							fig.subplots_adjust(top=0.90)
							plt.suptitle("%s - LR of Biological Replicates (Experiment Dates)" % subproject)
							try:
								plt.savefig(output_folder + "figures/regression_plots/exp_pair_regressions_%s_%s.pdf"
											% (treatment, subproject), dpi=300)
								plt.savefig(output_folder + "figures/regression_plots/exp_pair_regressions_%s_%s.jpg"
											% (treatment, subproject), dpi=300)
								plt.savefig(output_folder + "figures/regression_plots/exp_pair_regressions_%s_%s.png"
											% (treatment, subproject), dpi=300)

							except ValueError:
								plt.savefig(output_folder + "figures/regression_plots/exp_pair_regressions_%s_%s.pdf"
											% (treatment, subproject))
								plt.savefig(output_folder + "figures/regression_plots/exp_pair_regressions_%s_%s.jpg"
											% (treatment, subproject))
								plt.savefig(output_folder + "figures/regression_plots/exp_pair_regressions_%s_%s.png"
											% (treatment, subproject))

							plt.close()

				pickle.dump(rep_regression_stat, open(output_folder + "statistics/pre_%s_replicate_%s_regression.p"
													  % (replicate_type, treatment), "wb"))
			else:
				rep_regression_stat = pickle.load(open(output_folder + "statistics/pre_%s_replicate_%s_regression.p"
													   % (replicate_type, treatment), "rb"))

			# R^2 distribution -- sub-project based

			subproject_based_rsquares = {}
			for subproject, d1 in rep_regression_stat.items():
				rs = []
				for e, rsquare in d1.items():
					rs.append(rsquare["rvalue"])

				if subproject not in subproject_based_rsquares.keys():
					subproject_based_rsquares[subproject] = rs
				else:
					subproject_based_rsquares[subproject].extend(rs)

			# R^2 distribution -- Whole
			all_rsquares = [rsquare["rvalue"] for subproject, d1 in rep_regression_stat.items()
							for e, rsquare in d1.items()]

			m_all, sd_all = scipy.stats.norm.fit(all_rsquares)
			for subproject, subproject_rsquares in subproject_based_rsquares.items():
				if subproject_rsquares is not None:
					# Z score calculation of the r-squares
					m, sd = scipy.stats.norm.fit(subproject_rsquares)
					for ep, stat in rep_regression_stat[subproject].items():
						if sd != 0 or sd is not None:
							# Sub project level
							z_score = (stat["rvalue"] - m) / sd
							p_value = scipy.stats.norm.sf(abs(z_score)) * 2
							rep_regression_stat[subproject][ep].update({"zscore": z_score})
							rep_regression_stat[subproject][ep].update({"pvalue": p_value})
							# All
							z_score_all = (stat["rvalue"] - m_all) / sd_all
							p_value_all = scipy.stats.norm.sf(abs(z_score_all)) * 2
							rep_regression_stat[subproject][ep].update({"all_zscore": z_score_all})
							rep_regression_stat[subproject][ep].update({"all_pvalue": p_value_all})

					outliers = list()
					max_outlier_rsquare = 0
					for ep, stat in rep_regression_stat[subproject].items():
						if "pvalue" in stat.keys():
							if stat["pvalue"] < 0.05:
								if stat["rvalue"] > max_outlier_rsquare:
									max_outlier_rsquare = stat["rvalue"]
									outliers.append((subproject, ep, stat["rvalue"]))

					pickle.dump(outliers, open(output_folder + "statistics/%s_replicate_%s_%s_outliers.p"
											   % (replicate_type, treatment, subproject), "wb"))

					if plotting:
						# Density Plot and Histogram of all arrival delays
						sns.distplot(subproject_rsquares, hist=True, kde=True, bins=int(180 / 5), color='darkblue',
									 hist_kws={'edgecolor': 'black'},
									 kde_kws={'linewidth': 4})
						plt.title(
							"Density Plot of %d $R^2$ of all cell lines in %s" % (len(subproject_rsquares), subproject))
						plt.xlim(0, 1.1)
						plt.axvline(x=max_outlier_rsquare, label="Upper limit for p-value 0.05",
									color="gray", ls="--", ymax=0.95)
						plt.legend()
						plt.xlabel("R-squares")
						plt.ylabel("Density")
						plt.savefig(output_folder + "figures/%s_replicate_%s_%s_rsquare_distribution.pdf"
									% (replicate_type, treatment, subproject), dpi=300)
						plt.savefig(output_folder + "figures/%s_replicate_%s_%s_rsquare_distribution.jpg"
									% (replicate_type, treatment, subproject), dpi=300)
						plt.savefig(output_folder + "figures/%s_replicate_%s_%s_rsquare_distribution.png"
									% (replicate_type, treatment, subproject), dpi=300)
						plt.close()

			outliers_all = []
			max_outlier_rsquare_all = 0
			for subproject, d1 in rep_regression_stat.items():
				for ep, stat in d1.items():
					if "all_pvalue" in stat.keys():
						if stat["all_pvalue"] < 0.05:
							if stat["rvalue"] > max_outlier_rsquare_all:
								max_outlier_rsquare_all = stat["rvalue"]
								outliers_all.append((subproject, ep, stat["rvalue"]))

			pickle.dump(outliers_all, open(output_folder + "statistics/%s_replicate_%s_outliers.p"
										   % (replicate_type, treatment), "wb"))

			pickle.dump(rep_regression_stat, open(output_folder + "statistics/%s_replicate_%s_regression.p"
												  % (replicate_type, treatment), "wb"))

			if plotting:
				# Density Plot and Histogram of all arrival delays
				sns.distplot(all_rsquares, hist=True, kde=True, bins=int(180 / 5), color='darkblue',
							 hist_kws={'edgecolor': 'black'},
							 kde_kws={'linewidth': 4})
				plt.title("Density Plot of %d $R^2$ of %s cell lines in all projects" % (len(all_rsquares), "all"))
				plt.xlim(0, 1.1)
				plt.axvline(x=max_outlier_rsquare_all, label="Upper limit for p-value 0.05",
							color="gray", ls="--", ymax=0.95)
				plt.legend()
				plt.xlabel("R-squares")
				plt.ylabel("Density")
				plt.savefig(output_folder + "figures/%s_replicate_%s_rsquare_distribution.pdf"
							% (replicate_type, treatment), dpi=300)
				plt.savefig(output_folder + "figures/%s_replicate_%s_rsquare_distribution.jpg"
							% (replicate_type, treatment), dpi=300)
				plt.savefig(output_folder + "figures/%s_replicate_%s_rsquare_distribution.png"
							% (replicate_type, treatment), dpi=300)
				plt.close()

	else:
		rep_regression_stat = pickle.load(open(output_folder + "statistics/%s_replicate_%s_regression.p"
											   % (replicate_type, treatment), "rb"))

	return rep_regression_stat


def get_replicate_outliers(replicate_type, treatment, type_of_merge, answer_t):
	"""
	:param replicate_type: technical/biological
	:param treatment: Combination or single
	:param type_of_merge: median/mean merging replicates after analysis
	:param answer_t: If the replicate_type is biological, then it is the technical replicate merging method.
	:return:
	"""

	output_folder = output_path + "data/replicates/%s_replicates/statistics/" % replicate_type
	if "%s_replicate_%s_outliers.p" % (replicate_type, treatment) not in os.listdir(output_folder):
		_ = regression_replicates(replicate_type=replicate_type, treatment=treatment,
								  plotting=False, type_of_merge=type_of_merge, answer_t=answer_t)

	outliers = pickle.load(open(output_folder + "%s_replicate_%s_outliers.p" % (replicate_type, treatment), "rb"))

	return outliers


def statistics_replicates(replicate_type, treatment, type_of_merge, answer_t):
	"""
	:param replicate_type: technical/biological
	:param treatment: Combination or single
	:param type_of_merge: median/mean merging replicates after analysis
	:param answer_t: If the replicate_type is biological, then it is the technical replicate merging method.
	:return:
	"""

	if answer_t:
		del_all_t = True
	else:
		del_all_t = False

	output_folder = output_path + "data/replicates/%s_replicates/statistics/" % replicate_type
	if "%s_replicate_%s_statistics.csv" % (replicate_type, treatment) not in os.listdir(output_folder):

		if replicate_type == "technical":

			viability_df = get_all_viabilities(treatment=treatment)
			rep_regression = regression_replicates(replicate_type=replicate_type, treatment=treatment,
												   plotting=False, type_of_merge=type_of_merge, answer_t=answer_t)
			outliers = get_replicate_outliers(replicate_type=replicate_type, treatment=treatment,
											  type_of_merge=type_of_merge, answer_t=answer_t)
			max_outlier_limit = max([o[3] for o in outliers])
			rep_variance = count_replicates(replicate_type=replicate_type, treatment=treatment,
											type_of_merge=type_of_merge, answer_t=answer_t)

			responsiveness_rsquare_count_df = pandas.DataFrame(0, columns=["unresponsive", "responsive"],
															   index=["correlated", "uncorrelated"])

			all_stat_df = pandas.DataFrame(columns=["cell_line", "SIDM" "cell_id", "subproject", "project", "tissue",
													"exp_date", "bp", "r_square", "r_square_p", "z_score_p",
													"bp_min", "bp_mean", "bp_median", "bp_max",
													"b1_min", "b1_mean", "b1_median", "b1_max",
													"b2_min", "b2_mean", "b2_median", "b2_max",
													"b1_responsive", "b2_responsive", "outlier",
													"correlated", "responsive", "CV_exp_mean", "CV_exp_median",
													"CV_exp_min", "CV_exp_max", "SD_exp_mean", "SD_exp_median",
													"SD_exp_min", "SD_exp_max", "CV_bp_mean", "CV_bp_median",
													"CV_bp_min", "CV_bp_max", "SD_bp_mean", "SD_bp_median", "SD_bp_min",
													"SD_bp_max"])

			for subproject, subproject_stat in rep_regression.items():
				for exp_date, subproject_exp_stat in subproject_stat.items():
					# Variation information
					exp_rep_variance_stat = rep_variance[(rep_variance.subproject == subproject) &
														 (rep_variance.exp_date == exp_date)]

					group_df = viability_df[(viability_df.subproject == subproject) &
											(viability_df.exp_date == exp_date)]

					pivot_df = group_df.pivot(columns="plate_num", values="viability",
											  index=[col for col in group_df.columns
													 if col not in ["plate_num", "viability"]])

					for bp, subproject_exp_bp_stat in subproject_exp_stat.items():
						response, correlated = False, False
						b1_responsive, b2_responsive = False, False
						if (subproject, exp_date, bp, subproject_exp_bp_stat["rvalue"]) in outliers:
							outlier = True
						else:
							outlier = False

						pivot_b1_df = pivot_df[bp[0]].dropna()
						pivot_b2_df = pivot_df[bp[1]].dropna()
						pivot_bp_df = pivot_df[[bp[0], bp[1]]].dropna()

						response_b1_percentage = (len(
							[v for v in pivot_b1_df.groupby(["D1", "D2", "SIDM"]).min().values
							 if v > 0.7]) * 100.0) / len(pivot_b1_df.groupby(["D1", "D2", "SIDM"]).min().values)
						if response_b1_percentage >= 80:
							b1_responsive = False
						else:
							b1_responsive = True

						response_b2_percentage = (len(
							[v for v in pivot_b2_df.groupby(["D1", "D2", "SIDM"]).min().values
							 if v > 0.7]) * 100.0) / len(pivot_b2_df.groupby(["D1", "D2", "SIDM"]).min().values)
						if response_b2_percentage >= 80:
							b2_responsive = False
						else:
							b2_responsive = True

						response_percentage = (len(
							[v for v in [k for i in pivot_bp_df.groupby(["D1", "D2", "SIDM"]).min().values for k in i]
							 if v > 0.7]) * 100.0) / len(
							[k for i in pivot_bp_df.groupby(["D1", "D2", "SIDM"]).min().values for k in i])
						if response_percentage >= 80:
							response = False
						else:
							response = True

						if subproject_exp_bp_stat["rvalue"] > max_outlier_limit:
							correlated = True
							if response:
								responsiveness_rsquare_count_df.loc["correlated", "responsive"] += 1
							else:
								responsiveness_rsquare_count_df.loc["correlated", "unresponsive"] += 1
						else:
							if response:
								responsiveness_rsquare_count_df.loc["uncorrelated", "responsive"] += 1
							else:
								responsiveness_rsquare_count_df.loc["uncorrelated", "unresponsive"] += 1

						pivot_bp_df["SD_bp"] = pivot_bp_df.apply(lambda x: numpy.std([x[bp[0]], x[bp[1]]]), axis=1)
						pivot_bp_df["CV_bp"] = pivot_bp_df.apply(
							lambda x: (numpy.std([x[bp[0]], x[bp[1]]]) / numpy.mean([x[bp[0]], x[bp[1]]])) * 100.0
							if numpy.mean([x[bp[0]], x[bp[1]]]) != 0 else 0, axis=1)

						row = {"cell_id": exp_rep_variance_stat["cell_id"].unique()[0],
							   "SIDM": exp_rep_variance_stat["SIDM"].unique()[0],
							   "cell_line": sanger2model(exp_rep_variance_stat["SIDM"].unique()[0]),
							   "project": exp_rep_variance_stat["project"].unique()[0], "subproject": subproject,
							   "exp_date": exp_date, "bp": bp,
							   "bp_min": numpy.min(pivot_bp_df.values),
							   "bp_max": numpy.max(pivot_bp_df.values),
							   "bp_mean": numpy.mean(pivot_bp_df.values),
							   "bp_median": numpy.median(pivot_bp_df.values),
							   "b1_min": numpy.min(pivot_b1_df.values),
							   "b1_max": numpy.max(pivot_b1_df.values),
							   "b1_mean": numpy.mean(pivot_b1_df.values),
							   "b1_median": numpy.median(pivot_b1_df.values),
							   "b2_min": numpy.min(pivot_b2_df.values),
							   "b2_max": numpy.max(pivot_b2_df.values),
							   "b2_mean": numpy.mean(pivot_b2_df.values),
							   "b2_median": numpy.median(pivot_b2_df.values),
							   "r_square": subproject_exp_bp_stat["rvalue"],
							   "r_square_p": subproject_exp_bp_stat["r2pvalue"],
							   "z_score_p": subproject_exp_bp_stat["all_pvalue"],
							   "outlier": outlier, "correlated": correlated,
							   "responsive": response, "b1_responsive": b1_responsive, "b2_responsive": b2_responsive,
							   "CV_exp_mean": numpy.mean(exp_rep_variance_stat["CV"]),
							   "CV_exp_median": numpy.median(exp_rep_variance_stat["CV"]),
							   "CV_exp_min": numpy.min(exp_rep_variance_stat["CV"]),
							   "CV_exp_max": numpy.max(exp_rep_variance_stat["CV"]),
							   "SD_exp_mean": numpy.mean(exp_rep_variance_stat["SD"]),
							   "SD_exp_median": numpy.median(exp_rep_variance_stat["SD"]),
							   "SD_exp_min": numpy.min(exp_rep_variance_stat["SD"]),
							   "SD_exp_max": numpy.max(exp_rep_variance_stat["SD"]),
							   "CV_bp_mean": numpy.mean(pivot_bp_df["CV_bp"]),
							   "CV_bp_median": numpy.median(pivot_bp_df["CV_bp"]),
							   "CV_bp_min": numpy.min(pivot_bp_df["CV_bp"]),
							   "CV_bp_max": numpy.max(pivot_bp_df["CV_bp"]),
							   "SD_bp_mean": numpy.mean(pivot_bp_df["SD_bp"]),
							   "SD_bp_median": numpy.median(pivot_bp_df["SD_bp"]),
							   "SD_bp_min": numpy.min(pivot_bp_df["SD_bp"]),
							   "SD_bp_max": numpy.max(pivot_bp_df["SD_bp"])}
						df = pandas.DataFrame.from_dict([row])
						all_stat_df = pandas.concat([all_stat_df, df])

			all_stat_df.to_csv(output_folder + "%s_replicate_%s_statistics.csv" % (replicate_type, treatment),
							   index=True)

			responsiveness_rsquare_count_df.to_csv(output_folder + "%s_replicate_%s_responsiveness_count.csv"
												   % (replicate_type, treatment), index=True)

		if replicate_type == "biological":

			# For biological replicate, read the call_mid_viability
			viability_df = Viability(treatment=treatment).call_mid_viability(type_of_merge=type_of_merge,
																			 del_rep=del_all_t)

			rep_regression = regression_replicates(replicate_type=replicate_type, treatment=treatment,
												   plotting=False, type_of_merge=type_of_merge, answer_t=answer_t)
			outliers = get_replicate_outliers(replicate_type=replicate_type, treatment=treatment,
											  type_of_merge=type_of_merge, answer_t=answer_t)
			max_outlier_limit = max([o[2] for o in outliers])

			rep_variance = count_replicates(replicate_type=replicate_type, treatment=treatment,
											type_of_merge=type_of_merge, answer_t=answer_t)

			responsiveness_rsquare_count_df = pandas.DataFrame(0, columns=["unresponsive", "responsive"],
															   index=["correlated", "uncorrelated"])

			all_stat_df = pandas.DataFrame(columns=["cell_line", "SIDM", "cell_id", "subproject", "project", "tissue",
													"ep", "r_square", "r_square_p", "z_score_p",
													"ep_min", "ep_mean", "ep_median", "ep_max",
													"e1_min", "e1_mean", "e1_median", "e1_max",
													"e2_min", "e2_mean", "e2_median", "e2_max",
													"outlier", "correlated",
													"e1_responsive", "e2_responsive", "responsive",
													"CV_mean", "CV_median", "CV_min", "CV_max",
													"SD_mean", "SD_median", "SD_min", "SD_max",
													"CV_ep_mean", "CV_ep_median", "CV_ep_min", "CV_ep_max",
													"SD_ep_mean", "SD_ep_median", "SD_ep_min", "SD_ep_max"])

			for subproject, subproject_stat in rep_regression.items():
				# Variation information
				rep_variance_stat = rep_variance[(rep_variance.subproject == subproject)]

				group_df = viability_df[(viability_df.subproject == subproject)]

				pivot_df = group_df.pivot(columns="exp_date", values="viability",
										  index=[col for col in group_df.columns
												 if col not in ["exp_date", "viability"]])

				for ep, subproject_ep_stat in subproject_stat.items():
					response, correlated = False, False
					e1_response, e2_response = False, False
					if (subproject, ep, subproject_ep_stat["rvalue"]) in outliers:
						outlier = True
					else:
						outlier = False

					pivot_e1_df = pivot_df[[ep[0]]].dropna()
					pivot_e2_df = pivot_df[[ep[1]]].dropna()
					pivot_ep_df = pivot_df[[ep[0], ep[1]]].dropna()

					response_e1_percentage = (len([v for v in pivot_e1_df.values
												   if min(v) > 0.7]) * 100.0) / len(pivot_e1_df.values)
					if response_e1_percentage >= 80:
						e1_response = False
					else:
						e1_response = True

					response_e2_percentage = (len([v for v in pivot_e2_df.values
												   if min(v) > 0.7]) * 100.0) / len(pivot_e2_df.values)
					if response_e2_percentage >= 80:
						e2_response = False
					else:
						e2_response = True

					response_percentage = (len([v for v in pivot_ep_df.values
												if min(v) > 0.7]) * 100.0) / len(pivot_ep_df.values)
					if response_percentage >= 80:
						response = False
					else:
						response = True

					if subproject_ep_stat["rvalue"] > max_outlier_limit:
						correlated = True
						if response:
							responsiveness_rsquare_count_df.loc["correlated", "responsive"] += 1
						else:
							responsiveness_rsquare_count_df.loc["correlated", "unresponsive"] += 1
					else:
						if response:
							responsiveness_rsquare_count_df.loc["uncorrelated", "responsive"] += 1
						else:
							responsiveness_rsquare_count_df.loc["uncorrelated", "unresponsive"] += 1

					pivot_ep_df["SD_ep"] = pivot_ep_df.apply(lambda x: numpy.std([x[ep[0]], x[ep[1]]]), axis=1)
					pivot_ep_df["CV_ep"] = pivot_ep_df.apply(
						lambda x: (numpy.std([x[ep[0]], x[ep[1]]]) / numpy.mean([x[ep[0]], x[ep[1]]])) * 100.0
						if numpy.mean([x[ep[0]], x[ep[1]]]) != 0 else 0, axis=1)

					row = {"cell_id": rep_variance_stat["cell_id"].unique()[0],
						   "SIDM": rep_variance_stat["SIDM"].unique()[0],
						   "cell_line": sanger2model(rep_variance_stat["SIDM"].unique()[0]),
						   "project": rep_variance_stat["project"].unique()[0], "subproject": subproject,
						   "ep": ep,
						   "ep_min": numpy.min(pivot_ep_df.values),
						   "ep_max": numpy.max(pivot_ep_df.values),
						   "ep_mean": numpy.mean(pivot_ep_df.values),
						   "ep_median": numpy.median(pivot_ep_df.values),
						   "e1_min": numpy.min(pivot_e1_df.values),
						   "e1_max": numpy.max(pivot_e1_df.values),
						   "e1_mean": numpy.mean(pivot_e1_df.values),
						   "e1_median": numpy.median(pivot_e1_df.values),
						   "e2_min": numpy.min(pivot_e2_df.values),
						   "e2_max": numpy.max(pivot_e2_df.values),
						   "e2_mean": numpy.mean(pivot_e2_df.values),
						   "e2_median": numpy.median(pivot_e2_df.values),
						   "r_square": subproject_ep_stat["rvalue"],
						   "r_square_p": subproject_ep_stat["r2pvalue"],
						   "z_score_p": subproject_ep_stat["all_pvalue"],
						   "outlier": outlier, "correlated": correlated, "responsive": response,
						   "e1_responsive": e1_response, "e2_responsive": e2_response,
						   "CV_mean": numpy.mean(rep_variance_stat["CV"]),
						   "CV_median": numpy.median(rep_variance_stat["CV"]),
						   "CV_min": numpy.min(rep_variance_stat["CV"]),
						   "CV_max": numpy.max(rep_variance_stat["CV"]),
						   "SD_mean": numpy.mean(rep_variance_stat["SD"]),
						   "SD_median": numpy.median(rep_variance_stat["SD"]),
						   "SD_min": numpy.min(rep_variance_stat["SD"]),
						   "SD_max": numpy.max(rep_variance_stat["SD"]),
						   "CV_ep_mean": numpy.mean(pivot_ep_df["CV_ep"]),
						   "CV_ep_median": numpy.median(pivot_ep_df["CV_ep"]),
						   "CV_ep_min": numpy.min(pivot_ep_df["CV_ep"]),
						   "CV_ep_max": numpy.max(pivot_ep_df["CV_ep"]),
						   "SD_ep_mean": numpy.mean(pivot_ep_df["SD_ep"]),
						   "SD_ep_median": numpy.median(pivot_ep_df["SD_ep"]),
						   "SD_ep_min": numpy.min(pivot_ep_df["SD_ep"]),
						   "SD_ep_max": numpy.max(pivot_ep_df["SD_ep"])}
					df = pandas.DataFrame.from_dict([row])
					all_stat_df = pandas.concat([all_stat_df, df])

			all_stat_df.to_csv(output_folder + "%s_replicate_%s_statistics.csv" % (replicate_type, treatment),
							   index=False)
			responsiveness_rsquare_count_df.to_csv(output_folder + "%s_replicate_%s_responsiveness_count.csv"
												   % (replicate_type, treatment), index=False)

	else:
		all_stat_df = pandas.read_csv(output_folder + "%s_replicate_%s_statistics.csv" % (replicate_type, treatment),
									  index_col=0)
		responsiveness_rsquare_count_df = pandas.read_csv(output_folder + "%s_replicate_%s_responsiveness_count.csv"
														  % (replicate_type, treatment), index_col=0)

	return all_stat_df, responsiveness_rsquare_count_df


def bad_replicate(replicate_type, treatment, type_of_merge, answer_t):
	"""
	:param replicate_type: technical/biological
	:param treatment: Combination or single
	:param type_of_merge: median/mean merging replicates after analysis
	:param answer_t: If the replicate_type is biological, then it is the technical replicate merging method.
	:return:
	"""

	output_folder = output_path + "data/replicates/%s_replicates/statistics/" % replicate_type
	if "bad_%s_replicate_%s.p" % (replicate_type, treatment) not in os.listdir(output_folder):
		if replicate_type == "technical":
			all_rep_stat, _ = statistics_replicates(replicate_type=replicate_type, treatment=treatment,
													type_of_merge=type_of_merge, answer_t=answer_t)
			outlier_df = all_rep_stat[all_rep_stat.outlier]

			bad_replicates = dict()
			for subproject, subproject_outlier in outlier_df.groupby(["subproject"]):
				for ind, row in subproject_outlier.iterrows():
					bp = row["bp"]
					if type(bp) == tuple:
						b1, b2 = bp[0], bp[1]
					else:
						b1 = bp.split("(")[1].split(", ")[0]
						b2 = bp.split(", ")[1].split(")")[0]
						if len(b1.split("'")) > 1:
							b1 = b1.split("'")[1]
						if len(b2.split("'")) > 1:
							b2 = b2.split("'")[1]
					for barcode in [b1, b2]:
						if subproject not in bad_replicates.keys():
							bad_replicates[subproject] = [{"exp_date": row["exp_date"], "barcode": barcode}]
						else:
							if {"exp_date": row["exp_date"], "barcode": barcode} not in bad_replicates[subproject]:
								bad_replicates[subproject].append({"exp_date": row["exp_date"], "barcode": barcode})

		if replicate_type == "biological":
			all_rep_stat, _ = statistics_replicates(replicate_type=replicate_type, treatment=treatment,
													type_of_merge=type_of_merge, answer_t=answer_t)
			outlier_df = all_rep_stat[all_rep_stat.outlier]

			bad_replicates = dict()
			for subproject, subproject_outlier in outlier_df.groupby(["subproject"]):
				for ind, row in subproject_outlier.iterrows():
					ep = row["ep"]
					if type(ep) == tuple:
						e1, e2 = ep[0], ep[1]
					else:
						e1 = ep.split("(")[1].split(", ")[0]
						e2 = ep.split(", ")[1].split(")")[0]
						if len(e1.split("'")) > 1:
							e1 = e1.split("'")[1]
						if len(e2.split("'")) > 1:
							e2 = e2.split("'")[1]
					for exp_date in [e1, e2]:
						if subproject not in bad_replicates.keys():
							bad_replicates[subproject] = [{"exp_date": exp_date}]
						else:
							if {"exp_date": exp_date} not in bad_replicates[subproject]:
								bad_replicates[subproject].append({"exp_date": exp_date})

		pickle.dump(bad_replicates, open(output_folder + "bad_%s_replicate_%s.p" % (replicate_type, treatment), "wb"))
	else:
		bad_replicates = pickle.load(open(output_folder + "bad_%s_replicate_%s.p" % (replicate_type, treatment), "rb"))

	return bad_replicates


def get_bad_pairs(replicate_type, treatment, type_of_merge, del_all_t, del_all_b):
	"""
	To obtain the viability scores of the CL-DP-DoP-(Exp Date) pair whose at least
	one of the observation is in one of the bad replicates
	:param replicate_type: technical/biological
	:param treatment: combination
	:param type_of_merge: median/mean
	:param del_all_t: True for del and False for all (technical replicate)
	:param del_all_b: True for del and False for all (biological replicate)
	:return: list of pair
	"""

	del_inc_pairs = list()

	if replicate_type == "technical":
		obj_status = "mid"
	else:
		obj_status = "post"

	output_folder = output_path + "data/replicates/%s_replicates/statistics/" % replicate_type
	if "%s_bad_pairs_merged_%s.p" % (replicate_type, type_of_merge) not in os.listdir(output_folder):

		bad_reps = bad_replicate(replicate_type=replicate_type, treatment=treatment,
								 type_of_merge=type_of_merge, answer_t=del_all_t)

		mid_obj = deserialise_viability_object(treatment=treatment, status=obj_status,
											   type_of_merge=type_of_merge,
											   del_rep_t=del_all_t, del_rep_b=del_all_b)

		if replicate_type == "technical":
			pre_df = mid_obj[treatment].pre_viability
			for p, bdl in bad_reps.items():
				for b in bdl:
					x = pre_df[(pre_df.subproject == p) & (pre_df.exp_date == b["exp_date"]) & (
								pre_df.plate_num == b["barcode"])]
					for i, l in x.groupby(["SIDM", "D1", "D2", "Do1", "Do2", "subproject", "exp_date"]):
						if i not in del_inc_pairs:
							del_inc_pairs.append(i)

		elif replicate_type == "biological":
			mid_df = mid_obj[treatment].mid_viability
			for p, bdl in bad_reps.items():
				for b in bdl:
					x = mid_df[(mid_df.subproject == p) & (mid_df.exp_date == b["exp_date"])]
					for i, l in x.groupby(["SIDM", "D1", "D2", "Do1", "Do2", "subproject"]):
						if i not in del_inc_pairs:
							del_inc_pairs.append(i)
		pickle.dump(del_inc_pairs,
					open(output_folder + "%s_bad_pairs_merged_%s.p" % (replicate_type, type_of_merge), "wb"))
	else:
		del_inc_pairs = pickle.load(
			open(output_folder + "%s_bad_pairs_merged_%s.p" % (replicate_type, type_of_merge), "rb"))
	return del_inc_pairs


def is_removal_worth(replicate_type, treatment, type_of_merge, plotting, t_answer):
	output_folder = output_path + "data/replicates/%s_replicates/statistics/" % replicate_type
	output_folder_fig = output_path + "data/replicates/%s_replicates/figures/" % replicate_type
	if "%s_del_all_%s_comparison_df.csv" % (replicate_type, type_of_merge) not in os.listdir(output_folder):

		if replicate_type == "technical":
			del_all_t = True
		elif replicate_type == "biological":
			if t_answer:
				del_all_t = True
			else:
				del_all_t = False

		del_inc_pairs = get_bad_pairs(replicate_type=replicate_type, treatment=treatment,
									  type_of_merge=type_of_merge, del_all_t=del_all_t, del_all_b=True)

		if replicate_type == "technical":

			del_obj = deserialise_viability_object(treatment=treatment, status="mid",
												   type_of_merge=type_of_merge, del_rep_t=True, del_rep_b=True)

			all_obj = deserialise_viability_object(treatment=treatment, status="mid",
												   type_of_merge=type_of_merge, del_rep_t=False, del_rep_b=False)

			del_df = del_obj[treatment].mid_viability
			all_df = all_obj[treatment].mid_viability

			merged_columns = ["SIDM", "D1", "D2", "Do1", "Do2", "subproject", "exp_date"]

		elif replicate_type == "biological":

			del_obj = deserialise_viability_object(treatment=treatment, status="post",
												   type_of_merge=type_of_merge, del_rep_t=t_answer, del_rep_b=True)

			all_obj = deserialise_viability_object(treatment=treatment, status="post",
												   type_of_merge=type_of_merge, del_rep_t=t_answer, del_rep_b=False)

			del_df = del_obj[treatment].post_viability
			all_df = all_obj[treatment].post_viability

			merged_columns = ["SIDM", "D1", "D2", "Do1", "Do2", "subproject"]

		del_df["selected"] = False
		all_df["selected"] = False

		del_lost = list()
		all_missing = list()
		i, t = 0, len(del_inc_pairs)
		del_df_groups = dict(del_df.groupby(merged_columns).groups)
		all_df_groups = dict(all_df.groupby(merged_columns).groups)
		for k in del_inc_pairs:
			del_indices = list()
			all_indices = list()
			if k in del_df_groups.keys():
				del_indices = list(del_df_groups[k])
				del_df.loc[del_indices, "selected"] = True
			else:
				del_lost.append(k)
			if k in all_df_groups.keys():
				all_indices = list(all_df_groups[k])
				all_df.loc[all_indices, "selected"] = True
			else:
				all_missing.append(k)
			i += 1
			print(i * 100.0 / t)

		del_df = del_df[del_df.selected]
		all_df = all_df[all_df.selected]

		all_data = list()
		for i in all_df.groupby(["SIDM", "D1", "D2"]).groups.keys():
			if i not in all_data:
				all_data.append(i)

		lost_data = list()
		for i in del_lost:
			if (i[0], i[1], i[2]) not in all_data:
				if (i[0], i[1], i[2]) not in lost_data:
					lost_data.append((i[0], i[1], i[2]))

		comparison_df = pandas.DataFrame(columns=merged_columns + ["viability_all", "viability_del"])

		mdf = all_df.merge(del_df, left_on=merged_columns, right_on=merged_columns,
						   suffixes=("_all", "_del"), how="left")
		comparison_df = pandas.concat([comparison_df, mdf])
		comparison_df.to_csv(output_folder + "%s_del_all_%s_comparison_df.csv" % (replicate_type, type_of_merge),
							 index=False)

		pickle.dump(lost_data, open(output_folder + "%s_del_lost_pairs.p" % replicate_type, "wb"))
		pickle.dump(all_data, open(output_folder + "%s_all_pairs.p" % replicate_type, "wb"))
		pickle.dump(all_missing, open(output_folder + "%s_all_missing.p" % replicate_type, "wb"))



	else:
		lost_data = pickle.load(open(output_folder + "%s_del_lost_pairs.p" % replicate_type, "rb"))
		all_data = pickle.load(open(output_folder + "%s_all_pairs.p" % replicate_type, "rb"))
		comparison_df = pandas.read_csv(
			output_folder + "%s_del_all_%s_comparison_df.csv" % (replicate_type, type_of_merge))

	if plotting:

		del_comparison_df = comparison_df[pandas.isna(comparison_df.selected_del)][
			["SIDM", "D1", "D2", "Do1", "Do2", "subproject", "viability_all"]]
		comparison_df = comparison_df[(~pandas.isna(comparison_df.selected_del)) & (comparison_df.selected_del)]

		# Distribution comparison of the averaged dropped and non-dropped viabilities

		melt_comparison_df = pandas.melt(
			comparison_df,
			id_vars=[col for col in comparison_df.columns if col not in ["viability_all", "viability_del"]],
			value_vars=["viability_all", "viability_del"])

		sns.displot(data=melt_comparison_df, x="value", hue="variable", kind="kde")
		plt.xlabel("Viability", fontsize=14)
		plt.ylabel("Density", fontsize=14)
		plt.title("Density Distribution of viabilities from affected pairs (# %d)"
				  % len(comparison_df.index))
		plt.savefig(output_folder_fig + "%s_del_all_%s_viability_distribution.pdf" % (replicate_type, type_of_merge),
					dpi=300)
		plt.savefig(output_folder_fig + "%s_del_all_%s_viability_distribution.jpg" % (replicate_type, type_of_merge),
					dpi=300)
		plt.savefig(output_folder_fig + "%s_del_all_%s_viability_distribution.png" % (replicate_type, type_of_merge),
					dpi=300)
		plt.close()

		# Distribution comparison of the averaged dropped viabilities in non-dropped

		del_melt_comparison_df = pandas.melt(
			del_comparison_df,
			id_vars=[col for col in del_comparison_df.columns if col != "viability_all"],
			value_vars=["viability_all"])

		sns.displot(data=del_melt_comparison_df, x="value", kind="kde")
		plt.xlabel("Viability", fontsize=14)
		plt.ylabel("Density", fontsize=14)
		plt.title(
			"Density Distribution of deleted viabilities (not in deleted subset) (# %d)" % len(del_comparison_df.index))
		plt.savefig(output_folder_fig + "%s_fully_del_%s_viability_distribution.pdf" % (replicate_type, type_of_merge),
					dpi=300)
		plt.savefig(output_folder_fig + "%s_fully_del_%s_viability_distribution.jpg" % (replicate_type, type_of_merge),
					dpi=300)
		plt.savefig(output_folder_fig + "%s_fully_del_%s_viability_distribution.png" % (replicate_type, type_of_merge),
					dpi=300)
		plt.close()

		# Check and plot the association
		num_data_points = len(comparison_df.index)
		print(comparison_df["viability_del"].values)
		print(comparison_df["viability_all"].values)
		slope, intercept, r_value, p_value, _ = \
			scipy.stats.linregress(comparison_df["viability_all"].values, comparison_df["viability_del"].values)
		"""
		xy = numpy.vstack([comparison_df["viability_all"], comparison_df["viability_del"]])
		z = gaussian_kde(xy)(xy)
		idx = z.argsort()
		x, y, z = comparison_df["viability_all"].values[idx], comparison_df["viability_del"].values[idx], z[idx]

		plt.scatter(x, y, c=z, s=16)
		plt.xlabel("Viability w/out dropping")
		plt.ylabel("Viability w/ dropping")
		plt.text(x=min(comparison_df["viability_all"]) - 0.02, y=0.99,
				 s=r"y = %.3f + %.3f x" % (intercept, slope))
		plt.text(x=min(comparison_df["viability_all"]) - 0.02, y=0.93,
				 s=r"R square: %.2f" % r_value ** 2)
		plt.text(x=min(comparison_df["viability_all"]) - 0.02, y=0.87,
				 s=r"P value: %.2f" % p_value)
		# fig.colorbar(density, label='Number of points per pixel')
		"""
		fig = plt.figure()
		ax = plot_density_reg(df=comparison_df, col1="viability_all", col2="viability_del",
							  data_point_name="Viability",
							  num_data_points=len(comparison_df.index), r2=r_value ** 2, axs=fig.subplots())
		plt.title("%s Replicates" % "".join(replicate_type[0].upper() + replicate_type[1:]))
		plt.xlabel("Viability without removal")
		plt.ylabel("Viability with removal")
		plt.savefig(output_folder_fig + "%s_del_all_%s_viability_comparison.pdf" % (replicate_type, type_of_merge),
					dpi=300)
		plt.savefig(output_folder_fig + "%s_del_all_%s_viability_comparison.jpg" % (replicate_type, type_of_merge),
					dpi=300)
		plt.savefig(output_folder_fig + "%s_del_all_%s_viability_comparison.png" % (replicate_type, type_of_merge),
					dpi=300)
		plt.close()

	else:
		num_data_points = len(comparison_df.index)
		slope, intercept, r_value, p_value, _ = \
			scipy.stats.linregress(comparison_df[["viability_all"]], comparison_df[["viability_del"]])

	print("R square: %.3f\nP value: %.3f\nPercentage of lost data: %d" %
		  (r_value ** 2, p_value, len(lost_data) * 100.0 / len(all_data)))
	if r_value ** 2 > 0.75 and p_value < 0.05:
		return False
	else:
		return True


# ---------------------------------------------------------------------------#
#        	Reproducibility analysis functions - Combi values                #
# ---------------------------------------------------------------------------#


def get_all_combi():
	"""
	Concatenate all combi files
	:return: Concatenated data frame
	"""

	output_folder = output_path + "data/curves/combi_files/"
	if "all_combi.csv" not in os.listdir(output_folder):
		combi_dfs = list()
		for project in project_base_name.keys():
			if "annotated_%s_combi_df.csv" % project not in os.listdir(output_folder + "combination/"):
				combi_df = get_combi(project=project)

				# Reset index to protect unique indexing
				combi_df = combi_df.reset_index()
				combi_df = combi_df[[i for i in combi_df.columns if i != "index"]]

				# Get combinations composed of only two drugs
				combi_df["mono_drug"] = True
				for g, g_df in combi_df.groupby(["ANCHOR_ID"]):
					if pandas.isna(g) is False and len(str(g).split("|")) > 1:
						df.loc[list(g_df.index), "mono_drug"] = False

				for g, g_df in combi_df.groupby(["LIBRARY_ID"]):
					if pandas.isna(g) is False and len(str(g).split("|")) > 1:
						df.loc[list(g_df.index), "mono_drug"] = False

				combi_df = combi_df[combi_df.mono_drug]

				# Make Drug IDs as numeric
				combi_df[["ANCHOR_ID", "LIBRARY_ID"]] = combi_df[["ANCHOR_ID", "LIBRARY_ID"]].apply(pandas.to_numeric)

				# SIDM conversion of viabilities have been done by me
				# Check if SIDM identification is concordant with the viability
				combi_df["SIDM_me"] = None
				combi_df["SIDM_different"] = False
				for cl, cl_df in combi_df.groupby(["CELL_LINE_NAME"]):
					inds = combi_df[combi_df.CELL_LINE_NAME == cl].index
					combi_df.loc[inds, "SIDM_me"] = CellLine(cl).id
					if combi_df.loc[inds, "SIDM"].unique() != combi_df.loc[inds, "SIDM_me"].unique():
						combi_df.loc[inds, "SIDM_different"] = True

				combi_df = combi_df[combi_df.SIDM_different == False]

				# Anchor name and Library name are the same above
				# Check if they are concordant
				combi_df["anchor_name"] = None
				combi_df["anchor_name_different"] = False
				for anchor, anchor_df in combi_df.groupby(["ANCHOR_ID"]):
					inds = combi_df[combi_df.ANCHOR_ID == anchor].index
					combi_df.loc[list(anchor_df.index), "anchor_name"] = drug_id2name(float(anchor))
					if combi_df.loc[inds, "ANCHOR_NAME"].unique() != combi_df.loc[inds, "anchor_name"].unique():
						combi_df.loc[inds, "anchor_name_different"] = True

				combi_df["library_name"] = None
				combi_df["library_name_different"] = False
				for library, library_df in combi_df.groupby(["LIBRARY_ID"]):
					inds = combi_df[combi_df.LIBRARY_ID == library].index
					combi_df.loc[list(library_df.index), "library_name"] = drug_id2name(float(library))
					if combi_df.loc[inds, "LIBRARY_NAME"].unique() != combi_df.loc[inds, "library_name"].unique():
						combi_df.loc[inds, "library_name_different"] = True

				# Only different one is MK-1775 and AZD1775 which are the same drug
				# but I selected to continue with AZD1775 as name
				# Continue with "anchor_name" and "library_name" columns

				# Check as well the tissue annotation
				combi_df["tissue_me"] = None
				combi_df["tissue_different"] = False
				for model, model_df in combi_df.groupby(["CELL_LINE_NAME"]):
					inds = combi_df[combi_df.CELL_LINE_NAME == model].index
					if model in all_models():
						combi_df.loc[inds, "tissue_me"] = CellLine(model).tissue
					else:
						combi_df.loc[inds, "tissue_me"] = None
					if combi_df.loc[inds, "tissue"].unique() != combi_df.loc[inds, "tissue_me"].unique():
						combi_df.loc[inds, "tissue_different"] = True

				# Sort the drug combination alphabetically and take the original orientation
				combi_df["DrugComb"] = combi_df.apply(
					lambda x: sort_drug_pairs(x.anchor_name, x.library_name)["sorted_pair"], axis=1)
				combi_df["Direction"] = combi_df.apply(
					lambda x: sort_drug_pairs(x.anchor_name, x.library_name)["direction"], axis=1)
				combi_df["project"] = project

				combi_df.to_csv(output_folder + "combination/annotated_%s_combi_df.csv" % project, index=False)

			else:
				combi_df = pandas.read_csv(output_folder + "combination/annotated_%s_combi_df.csv" % project,
										   index_col=0)

			combi_dfs.append(combi_df)

		whole_combi_df = pandas.concat(combi_dfs)
		whole_combi_df.to_csv(output_folder + "all_combi.csv", index=False)

	else:
		whole_combi_df = pandas.read_csv(output_folder + "all_combi.csv", index_col=0)

	return whole_combi_df


def get_bad_fits(stage, anchor_dose, estimate_data, subproject):
	"""
	Find the inconsistant fits with sd across
	:param stage: LIBRARY / COMBO / DELTA
	:param anchor_dose: Boolean, if comparison will be made with considering anchor doses or not
	:param estimate_data: The estimate that has been used in data filtration - XMID / EMAX
	:param subproject: boolean if the merging will be subproject level or project level
	:return:
	"""

	group_cols = ["SIDM", "DrugComb"]

	if anchor_dose:
		added_title = "_anchor_dose"
		group_cols.extend(["Direction", "ANCHOR_CONC", "maxc"])
	else:
		added_title = ""

	if subproject:
		p_title = "subproject_level"
		group_cols.extend(["RESEARCH_PROJECT", "project"])
	else:
		p_title = "project_level"
		group_cols.append("project")

	if stage == "combo":
		if estimate_data == "XMID":
			response_col = "SYNERGY_XMID"
		else:
			response_col = "SYNERGY_OBS_EMAX"
	elif stage == "delta":
		response_col = "SYNERGY_DELTA_%s" % estimate_data

	output_folder = output_path + "data/replicates/across_screens/"
	if "variance_%s_%s%s.csv" % (stage, estimate_data, added_title) not in os.listdir(
			output_folder + "%s/statistics/" % p_title):

		combi_df = get_all_combi().copy()
		if estimate_data == "XMID" and stage == "combo":
			combi_df["resistant"] = combi_df.apply(lambda x: True if x["SYNERGY_XMID"] > 9 else False, axis=1)
		else:
			combi_df["resistant"] = None

		variance_df = pandas.DataFrame()

		# Check the standard deviation
		for g, g_df in combi_df.groupby(group_cols):
			# If sample size is more than 2
			if len(g_df.index) > 2:
				if anchor_dose and subproject:
					d = {"SIDM": [g[0]], "DrugComb": [g[1]], "Direction": [g[2]], "ANCHOR_CONC": [g[3]],
						 "maxc": [g[4]], "RESEARCH_PROJECT": [g[5]], "project": [g[6]],
						 "sd": [numpy.std(g_df[response_col])],
						 "include_resistant": [True if pandas.isna(g_df.resistant) is False and
													   True in g_df.resistant.unique() else False]}
				elif anchor_dose and subproject is False:
					d = {"SIDM": [g[0]], "DrugComb": [g[1]], "Direction": [g[2]], "ANCHOR_CONC": [g[3]],
						 "maxc": [g[4]], "project": [g[5]], "sd": [numpy.std(g_df[response_col])],
						 "include_resistant": [True if pandas.isna(g_df.resistant) is False and
													   True in g_df.resistant.unique() else False]}
				elif anchor_dose is False and subproject:
					d = {"SIDM": [g[0]], "DrugComb": [g[1]], "RESEARCH_PROJECT": [g[2]], "project": [g[2]],
						 "sd": [numpy.std(g_df[response_col])],
						 "include_resistant": [True if pandas.isna(g_df.resistant) is False and
													   True in g_df.resistant.unique() else False]}
				else:
					d = {"SIDM": [g[0]], "DrugComb": [g[1]], "project": [g[2]],
						 "sd": [numpy.std(g_df[response_col])],
						 "include_resistant": [True if pandas.isna(g_df.resistant) is False and
													   True in g_df.resistant.unique() else False]}

				df = pandas.DataFrame.from_dict(d)
				variance_df = pandas.concat([variance_df, df])
		variance_df.to_csv(output_folder + "%s/statistics/variance_%s_%s%s.csv"
						   % (p_title, stage, estimate_data, added_title), index=True)
	else:
		variance_df = pandas.read_csv(output_folder + "%s/statistics/variance_%s_%s%s.csv"
									  % (p_title, stage, estimate_data, added_title), index_col=0)

	if "bad_fits_%s_%s%s.p" % (stage, estimate_data, added_title) not in os.listdir(
			output_folder + "%s/statistics/" % p_title):

		bad_fits = list()
		for r, row in variance_df[variance_df.sd >= 3].iterrows():
			t = list()
			for c in group_cols:
				t.append(row[c])
			bad_fits.append(t)

		if bad_fits:
			pickle.dump(bad_fits, open(output_folder + "%s/statistics/bad_fits_%s_%s%s.p"
									   % (p_title, stage, estimate_data, added_title), "wb"))
			return bad_fits
		else:
			return None

	else:
		bad_fits = pickle.load(open(output_folder + "%s/statistics/bad_fits_%s_%s%s.p"
									% (p_title, stage, estimate_data, added_title), "rb"))
		if bad_fits:
			return bad_fits
		else:
			return None


def prepare_combi(stage, anchor_dose, estimate_data, m_type, subproject, del_all):
	"""
	Retrieving the rows having the highest sensitivity
	:param anchor_dose: Boolean, if comparison will be made with considering anchor doses or not
	:param estimate_data: The estimate that has been used in data filtration - XMID / EMAX
	:param m_type: how data will be selected with- min, median, quantile or max
	:param subproject: boolean if the merging will be subproject level or project level
	:param del_all: boolean if the merging will be done after deletion or not
	:return: Data Frame
	"""

	if del_all:
		del_title = "del"
	else:
		del_title = "all"

	if estimate_data == "XMID":
		col = "SYNERGY_%s" % estimate_data
	else:
		col = "SYNERGY_OBS_EMAX"

	group_cols = ["SIDM", "DrugComb"]

	if anchor_dose:
		added_title = "_anchor_dose"
		group_cols.extend(["Direction", "ANCHOR_CONC", "maxc"])
	else:
		added_title = ""

	if subproject:
		title = "subproject_level"
		group_cols.extend(["RESEARCH_PROJECT", "project"])

	else:
		title = "project_level"
		group_cols.append("project")

	output_folder = output_path + "data/replicates/across_screens/%s/statistics/" % title
	if "%s_%s_%s_%s%s_merged_combi.csv" % (del_title, m_type, stage, estimate_data, added_title) not in os.listdir(
			output_folder):

		# Get all combi data
		whole_combi_df = get_all_combi()
		whole_combi_df = whole_combi_df.reset_index()
		combi_df = whole_combi_df.copy()

		if del_all:
			bad_fits = get_bad_fits(stage=stage, anchor_dose=anchor_dose,
									estimate_data=estimate_data, subproject=subproject)
			if bad_fits is not None:
				for fit in bad_fits:
					if anchor_dose and subproject:
						removed_indices = list(whole_combi_df[(whole_combi_df.SIDM == fit[0]) &
															  (whole_combi_df.DrugComb == fit[1]) &
															  (whole_combi_df.Direction == fit[2]) &
															  (whole_combi_df.ANCHOR_CONC == fit[3]) &
															  (whole_combi_df.maxc == fit[4]) &
															  (whole_combi_df.RESEARCH_PROJECT == fit[5]) &
															  (whole_combi_df.project == fit[6])].index)
					elif anchor_dose and subproject is False:
						removed_indices = list(whole_combi_df[(whole_combi_df.SIDM == fit[0]) &
															  (whole_combi_df.DrugComb == fit[1]) &
															  (whole_combi_df.Direction == fit[2]) &
															  (whole_combi_df.ANCHOR_CONC == fit[3]) &
															  (whole_combi_df.maxc == fit[4]) &
															  (whole_combi_df.project == fit[5])].index)
					elif anchor_dose is False and subproject:
						removed_indices = list(whole_combi_df[(whole_combi_df.SIDM == fit[0]) &
															  (whole_combi_df.DrugComb == fit[1]) &
															  (whole_combi_df.project == fit[2]) &
															  (whole_combi_df.RESEARCH_PROJECT == fit[3])].index)
					else:
						removed_indices = list(whole_combi_df[(whole_combi_df.SIDM == fit[0]) &
															  (whole_combi_df.DrugComb == fit[1]) &
															  (whole_combi_df.project == fit[2])].index)

				combi_df = combi_df.drop(index=removed_indices)

		indices = list()
		if subproject:
			if m_type == "min":
				if anchor_dose:
					indices = list(
						combi_df.groupby(
							["SIDM", "DrugComb", "anchor_name", "ANCHOR_CONC", "maxc", "RESEARCH_PROJECT", "project"])[
							col].idxmin())
				else:
					indices = list(
						combi_df.groupby(["SIDM", "DrugComb", "RESEARCH_PROJECT", "project"])[col].idxmin())
			elif m_type == "median":
				if anchor_dose:
					for group, df in combi_df.groupby(
							["SIDM", "DrugComb", "anchor_name", "ANCHOR_CONC", "maxc", "RESEARCH_PROJECT", "project"]):
						x = df[col].median()
						m_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) &
												(combi_df.anchor_name == group[2]) & (
															combi_df.ANCHOR_CONC == group[3]) &
												(combi_df.maxc == group[4]) & (combi_df.RESEARCH_PROJECT == group[5]) &
												(combi_df.project == group[6]) &
												(combi_df[col] == x)].index)
						indices.extend(m_index)
				else:
					for group, df in combi_df.groupby(["SIDM", "DrugComb", "RESEARCH_PROJECT", "project"]):
						x = df[col].median()
						m_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) &
												(combi_df.RESEARCH_PROJECT == group[2]) & (
															combi_df.project == group[3]) &
												(combi_df[col] == x)].index)
						indices.extend(m_index)
			elif m_type == "quantile":
				if anchor_dose:
					for group, df in combi_df.groupby(
							["SIDM", "DrugComb", "anchor_name", "ANCHOR_CONC", "maxc", "RESEARCH_PROJECT", "project"]):
						x = df[col].quantile(0.5, interpolation="nearest")
						q_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) &
												(combi_df.anchor_name == group[2]) & (
															combi_df.ANCHOR_CONC == group[3]) &
												(combi_df.maxc == group[4]) & (combi_df.RESEARCH_PROJECT == group[5]) &
												(combi_df.project == group[6]) &
												(combi_df[col] == x)].index)
						indices.extend(q_index)

				else:
					for group, df in combi_df.groupby(["SIDM", "DrugComb", "RESEARCH_PROJECT", "project"]):
						x = df[col].quantile(0.5, interpolation="nearest")
						q_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) &
												(combi_df.RESEARCH_PROJECT == group[2]) & (
															combi_df.project == group[3]) &
												(combi_df[col] == x)].index)
						indices.extend(q_index)

		else:
			if m_type == "min":
				if anchor_dose:
					indices = list(
						combi_df.groupby(["SIDM", "DrugComb", "anchor_name", "ANCHOR_CONC", "maxc", "project"])[
							col].idxmin())
				else:
					indices = list(combi_df.groupby(["SIDM", "DrugComb", "project"])[col].idxmin())
			elif m_type == "median":
				if anchor_dose:
					for group, df in combi_df.groupby(
							["SIDM", "DrugComb", "Direction", "ANCHOR_CONC", "maxc", "project"]):
						x = df[col].median()
						m_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) &
												(combi_df.anchor_name == group[2]) & (
															combi_df.ANCHOR_CONC == group[3]) &
												(combi_df.maxc == group[4]) & (combi_df.project == group[5]) &
												(combi_df[col] == x)].index)
					indices.extend(m_index)
				else:
					for group, df in combi_df.groupby(["SIDM", "DrugComb", "project"]):
						x = df[col].median()
						m_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) & (
									combi_df.project == group[2]) &
												(combi_df[col] == x)].index)
					indices.extend(m_index)
			elif m_type == "quantile":
				if anchor_dose:
					for group, df in combi_df.groupby(
							["SIDM", "DrugComb", "anchor_name", "ANCHOR_CONC", "maxc", "project"]):
						x = df[col].quantile(0.5, interpolation="nearest")
						q_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) &
												(combi_df.anchor_name == group[2]) & (
															combi_df.ANCHOR_CONC == group[3]) &
												(combi_df.maxc == group[4]) & (combi_df.project == group[5]) &
												(combi_df[col] == x)].index)
					indices.extend(q_index)
				else:
					for group, df in combi_df.groupby(["SIDM", "DrugComb", "project"]):
						x = df[col].quantile(0.5, interpolation="nearest")
						q_index = list(combi_df[(combi_df.SIDM == group[0]) & (combi_df.DrugComb == group[1]) & (
									combi_df.project == group[2]) &
												(combi_df[col] == x)].index)
					indices.extend(q_index)

		df = combi_df.loc[indices]
		df = df.drop_duplicates()
		df.to_csv(
			output_folder + "%s_%s_%s_%s%s_merged_combi.csv" % (del_title, m_type, stage, estimate_data, added_title),
			index=False)

	else:
		df = pandas.read_csv(output_folder + "%s_%s_%s_%s%s_merged_combi.csv"
							 % (del_title, m_type, stage, estimate_data, added_title))
	return df


def check_new_across_reproducibility(stage, anchor_dose, estimate_data, across_level, exp_type, plotting, m_type,
									 del_all):
	if across_level == 1:
		subproject = True
		if exp_type == "anchor":
			title = "anchor_level"
		else:
			title = "matrix_level"
	elif across_level == 2:
		title = "format_level"

	if stage == "combo":
		if estimate_data == "XMID":
			value_column = "SYNERGY_XMID"
		else:
			value_column = "SYNERGY_OBS_EMAX"
	elif stage == "delta":
		value_column = "SYNERGY_DELTA_%s" % estimate_data

	if anchor_dose:
		added_title = "_anchor_dose"
	else:
		added_title = ""

	if del_all:
		del_title = "del"
		del_all = True
	else:
		del_title = "all"
		del_all = False

	total_stats = None

	output_folder = output_path + "data/replicates/across_screens/%s/statistics/" % title
	output_folder_fig = output_path + "data/replicates/across_screens/%s/figures/" % title
	# Challenge --> different anchor concentration can give us exactly the same combo XMID
	# Example ('SIDM00671', 'Olaparib/Temozolomide', 'GDSC_005-B')
	# Continue with the lower anchor dose

	if across_level == 1:

		inside_total_stats = dict()
		# First check inside project
		# Number of groups inside projects

		all_combi_df = prepare_combi(stage=stage, anchor_dose=anchor_dose, estimate_data=estimate_data,
									 m_type=m_type, subproject=subproject, del_all=del_all)

		all_combi_df = all_combi_df[[col for col in all_combi_df.columns if col != "index"]]
		merged_combi_df = all_combi_df.drop_duplicates()

		if exp_type == "anchor":
			merged_combi_df = merged_combi_df[merged_combi_df.RESEARCH_PROJECT.isin(anchor_projects)]
		else:
			merged_combi_df = merged_combi_df[merged_combi_df.RESEARCH_PROJECT.isin(matrix_projects)]

		merged_combi_df = merged_combi_df.reset_index()

		for project, p_df in merged_combi_df.groupby(["project"]):
			subgroup_pair_dfs = dict()
			p_subgroups = p_df.groupby(["RESEARCH_PROJECT"]).groups.keys()
			if len(p_subgroups) > 1:
				try:
					if anchor_dose:
						sp_pivot_df = p_df.pivot(columns="RESEARCH_PROJECT", values=value_column,
												 index=["SIDM", "DrugComb", "ANCHOR_CONC", "Direction"])
					else:
						sp_pivot_df = p_df.pivot(columns="RESEARCH_PROJECT", values=value_column,
												 index=["SIDM", "DrugComb"])

				except ValueError:
					ind = list()
					for i, l in p_df.groupby(["SIDM", "DrugComb", "RESEARCH_PROJECT"]):
						if len(l[value_column].unique()) == 1:
							# Take lower anchor dose
							if len(l.index) > 1:
								if len(l.ANCHOR_CONC.unique()) > 1:
									ind.extend(list(l[l.ANCHOR_CONC == numpy.min(l.ANCHOR_CONC.unique())].index))
							elif len(l.index) == 1:
								ind.extend(list(l.index))
					p_df = p_df.loc[ind]
					sp_pivot_df = p_df.pivot(columns="RESEARCH_PROJECT", values=value_column,
											 index=["SIDM", "DrugComb"])

				for sp in itertools.combinations(p_subgroups, 2):
					t = sp_pivot_df[list(sp)]
					q = t.dropna(axis=0)
					if len(q.index) != 0:
						subgroup_pair_dfs[sp] = q.reset_index()
				subgroup_pairs = list(subgroup_pair_dfs.keys())
				if len(subgroup_pairs) > 0:
					if plotting:
						if len(subgroup_pairs) < 4:
							if len(subgroup_pairs) == 1:
								fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True, squeeze=False)

							else:
								fig, ax = plt.subplots(1, len(subgroup_pairs), figsize=(5 * len(subgroup_pairs), 5),
													   sharex=True, sharey=True, squeeze=False)
						elif len(subgroup_pairs) >= 4:
							s = (len(subgroup_pairs) // 4) + 1
							fig, ax = plt.subplots(s, 4, figsize=(20, 5 * s), sharey=True, sharex=True)

					for i in range(len(subgroup_pairs)):
						df = subgroup_pair_dfs[subgroup_pairs[i]]
						subprojects = [subgroup_pairs[i][0], subgroup_pairs[i][1]]
						num_data_points = len(df.index)

						slope, intercept, r_value, p_value, _ = \
							scipy.stats.linregress(df[subprojects[0]], df[subprojects[1]])

						if project not in inside_total_stats.keys():
							inside_total_stats[project] = {
								subgroup_pairs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}
						else:
							if subgroup_pairs[i] not in inside_total_stats[project].keys():
								inside_total_stats[project][subgroup_pairs[i]] = {"rvalue": r_value ** 2,
																				  "r2pvalue": p_value}

						if plotting:
							if len(subgroup_pairs) < 4:
								plot_density_reg(df=df, col1=subprojects[0], col2=subprojects[1],
												 data_point_name=estimate_data, num_data_points=num_data_points,
												 r2=r_value ** 2, axs=ax[0, i])
								ax[0, i].plot(df[subprojects[0]], intercept + slope * df[subprojects[0]], 'r',
											  label="Fitted Line")

								onetoone = numpy.linspace(min(df[subprojects[0]]), max(df[subprojects[0]]), 10)
								ax[0, i].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")

							elif len(subgroup_pairs) >= 4:
								a = i // 4
								if i < 4:
									b = i
								else:
									b = i - (4 * (i // 4))

								plot_density_reg(df=df, col1=subprojects[0], col2=subprojects[1],
												 data_point_name=estimate_data, num_data_points=num_data_points,
												 r2=r_value ** 2, axs=ax[a, b])
								ax[a, b].plot(df[subprojects[0]], intercept + slope * df[subprojects[0]], 'r',
											  label="Fitted Line")

								onetoone = numpy.linspace(min(df[subprojects[0]]), max(df[subprojects[0]]), 10)
								ax[a, b].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")

					if plotting:
						fig.tight_layout()
						fig.subplots_adjust(top=0.90)
						plt.suptitle("%s - %s merged LR of subprojects" % (project, m_type))
						plt.savefig(output_folder_fig + "%s/%s_inside_%s_%s_%s_%s_%s%s_regressions.pdf" % (
							m_type, title, project, del_title, m_type, stage, estimate_data, added_title), dpi=300)
						plt.close()

		pickle.dump(inside_total_stats, open(output_folder + "%s_inside_%s_%s_%s_%s%s_regression_stat.p"
											 % (title, estimate_data, del_title, m_type, stage, added_title), "wb"))

		# Second across project inside the format
		across_total_stats = dict()

		all_combi_df = prepare_combi(stage=stage, anchor_dose=anchor_dose, estimate_data=estimate_data,
									 m_type=m_type, subproject=False, del_all=del_all)

		all_combi_df = all_combi_df[[col for col in all_combi_df.columns if col != "index"]]
		merged_combi_df = all_combi_df.drop_duplicates()

		if exp_type == "anchor":
			merged_combi_df = merged_combi_df[merged_combi_df.RESEARCH_PROJECT.isin(anchor_projects)]
		else:
			merged_combi_df = merged_combi_df[merged_combi_df.RESEARCH_PROJECT.isin(matrix_projects)]

		merged_combi_df = merged_combi_df.reset_index()

		# Number of projects
		project_pair_dfs = dict()
		p_projects = merged_combi_df.groupby(["project"]).groups.keys()
		if len(p_projects) > 1:
			try:
				if anchor_dose:
					p_pivot_df = merged_combi_df.pivot(columns="project", values=value_column,
													   index=["SIDM", "DrugComb", "Direction", "ANCHOR_CONC"])
				else:
					p_pivot_df = merged_combi_df.pivot(columns="project", values=value_column,
													   index=["SIDM", "DrugComb"])
			except ValueError:
				ind = list()
				if len(merged_combi_df[value_column].unique()) != 1:
					# Take lower anchor dose
					for i, l in merged_combi_df.groupby(["SIDM", "DrugComb"]):
						if len(l.index) > 1:
							if len(l.ANCHOR_CONC.unique()) > 1:
								ind.extend(list(l[l.ANCHOR_CONC == numpy.min(l.ANCHOR_CONC.unique())].index))
						elif len(l.index) == 1:
							ind.extend(list(l.index))
				merged_combi_df = merged_combi_df.loc[ind]
				p_pivot_df = merged_combi_df.pivot_table(columns="project", values=value_column,
														 index=["SIDM", "DrugComb"])

			for pp in itertools.combinations(p_projects, 2):
				t = p_pivot_df[list(pp)]
				q = t.dropna(axis=0)
				if len(q.index) != 0:
					project_pair_dfs[pp] = q.reset_index()

			project_pairs = list(project_pair_dfs.keys())
			project_pairs
			if len(project_pairs) > 0:
				if len(project_pairs) < 4:
					if len(project_pair_dfs.keys()) == 1:
						fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True, squeeze=False)

					else:
						fig, ax = plt.subplots(1, len(project_pairs), figsize=(5 * len(project_pairs), 5), sharex=True,
											   sharey=True, squeeze=False)

				elif len(project_pairs) >= 4:
					s = (len(project_pairs) // 4) + 1
					fig, ax = plt.subplots(s, 4, figsize=(20, 5 * s), sharey=True, sharex=True)

				for i in range(len(project_pairs)):
					df = project_pair_dfs[project_pairs[i]]
					projects = (project_pairs[i][0], project_pairs[i][1])
					num_data_points = len(df.index)

					slope, intercept, r_value, p_value, _ = \
						scipy.stats.linregress(df[projects[0]], df[projects[1]])

					if projects not in across_total_stats.keys():
						across_total_stats[projects] = {"rvalue": r_value ** 2, "r2pvalue": p_value}

					if plotting:
						if len(project_pairs) < 4:
							plot_density_reg(df=df, col1=projects[0], col2=projects[1], data_point_name=estimate_data,
											 num_data_points=num_data_points, r2=r_value ** 2, axs=ax[0, i])

							ax[0, i].plot(df[projects[0]], intercept + slope * df[projects[0]], 'r',
										  label="Fitted Line")

							onetoone = numpy.linspace(min(df[projects[0]]), max(df[projects[0]]), 10)
							ax[0, i].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")

						elif len(project_pairs) >= 4:
							a = i // 4
							if i < 4:
								b = i
							else:
								b = i - (4 * (i // 4))

							plot_density_reg(df=df, col1=projects[0], col2=projects[1], data_point_name=estimate_data,
											 num_data_points=num_data_points, r2=r_value ** 2, axs=ax[a, b])
							ax[a, b].plot(df[projects[0]], intercept + slope * df[projects[0]], 'r',
										  label="Fitted Line")

							onetoone = numpy.linspace(min(df[projects[0]]), max(df[projects[0]]), 10)
							ax[a, b].plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")

				if plotting:
					fig.tight_layout()
					fig.subplots_adjust(top=0.90)
					plt.suptitle("%s - %s merged LR of projects"
								 % (projects[0] + "-" + projects[1], m_type))

					plt.savefig(output_folder_fig + "%s_across_%s_%s_%s_%s%s_regressions.pdf" % (
						title, del_title, estimate_data, m_type, stage, added_title), dpi=300)
					plt.close()

			pickle.dump(across_total_stats, open(output_folder + "%s_across_%s_%s_%s_%s%s_regression_stat.p"
												 % (title, estimate_data, del_title, m_type, stage, added_title), "wb"))

	else:

		if "format_%s_%s_%s%s_comparison_df.csv" % (stage, estimate_data, m_type, added_title) not in os.listdir(
				output_folder):
			group_cols = ["SIDM", "DrugComb"]

			if anchor_dose: group_cols.extend(["Direction", "ANCHOR_CONC", "maxc"])

			# Compare formats
			anchor_combi_df = combine_format_combi(estimate_data=estimate_data, stage=stage,
												   exp_type="anchor").reset_index()
			anchor_combi_df["format"] = "anchored"
			anchor_combi_df["common"] = False
			matrix_combi_df = combine_format_combi(estimate_data=estimate_data, stage=stage,
												   exp_type="matrix").reset_index()
			matrix_combi_df["format"] = "matrix"
			matrix_combi_df["common"] = False

			anchor_perturbations = set(list(anchor_combi_df.groupby(["SIDM", "DrugComb"]).groups.keys()))
			matrix_perturbations = set(list(matrix_combi_df.groupby(["SIDM", "DrugComb"]).groups.keys()))

			common_perturbations = anchor_perturbations.intersection(matrix_perturbations)

			print("Number of common perturbations: % d" % len(common_perturbations))

			for pair in list(common_perturbations):
				anc_ind = list(
					anchor_combi_df[(anchor_combi_df.SIDM == pair[0]) & (anchor_combi_df.DrugComb == pair[1])].index)
				mat_ind = list(
					matrix_combi_df[(matrix_combi_df.SIDM == pair[0]) & (matrix_combi_df.DrugComb == pair[1])].index)
				anchor_combi_df.loc[anc_ind, "common"] = True
				matrix_combi_df.loc[mat_ind, "common"] = True

			common_anchor = anchor_combi_df[anchor_combi_df.common]
			common_matrix = matrix_combi_df[matrix_combi_df.common]

			mdf = common_anchor.merge(common_matrix, left_on=group_cols, right_on=group_cols,
									  suffixes=("_anchor", "_matrix"), how="left")
			comparison_df = mdf[group_cols + ["%s_anchor" % value_column, "%s_matrix" % value_column]]

			comparison_df.to_csv(output_folder + "format_%s_%s_%s%s_comparison_df.csv"
								 % (stage, estimate_data, m_type, added_title), index=False)
		else:
			comparison_df = pandas.read_csv(output_folder + "format_%s_%s_%s%s_comparison_df.csv"
											% (stage, estimate_data, m_type, added_title), index_col=0)
		if plotting:

			melt_comparison_df = pandas.melt(
				comparison_df,
				id_vars=[col for col in comparison_df.columns
						 if col not in ["%s_anchor" % value_column, "%s_matrix" % value_column]],
				value_vars=["%s_anchor" % value_column, "%s_matrix" % value_column])

			print(len(melt_comparison_df.index))

			if len(comparison_df.index) != 0:
				num_data_points = len(comparison_df.index)
				slope, intercept, r_value, p_value, _ = \
					scipy.stats.linregress(comparison_df["%s_anchor" % value_column],
										   comparison_df["%s_matrix" % value_column])

				fig = plt.figure()

				# sns.scatterplot(data=comparison_df, x="%s_anchor" % value_column, y="%s_matrix" % value_column, color="navy", alpha=0.4)

				ax = plot_density_reg(df=comparison_df, col1="%s_anchor" % value_column,
									  col2="%s_matrix" % value_column,
									  data_point_name="%s %s" % (stage, estimate_data),
									  num_data_points=len(comparison_df.index), r2=r_value ** 2, axs=fig.subplots())

				print(scipy.stats.spearmanr(comparison_df["%s_anchor" % value_column],
											comparison_df["%s_matrix" % value_column])[0])

				onetoone = numpy.linspace(min(comparison_df["%s_anchor" % value_column]),
										  max(comparison_df["%s_anchor" % value_column]), 10)
				ax.plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")

				plt.title("Comparison of Anchored and Matrix Screens")

				plt.savefig(output_folder_fig + "%s_%s_%s%s_comparison.pdf"
							% (stage, estimate_data, m_type, added_title), dpi=300)
				plt.close()

	return True


def combine_format_combi(estimate_data, stage, exp_type):
	"""
	Retrieving the rows having the highest synergy estimates
	"""

	if exp_type == "anchor":
		title = "anchor_level"
	else:
		title = "matrix_level"

	output_folder = output_path + "data/curves/combined_response/"
	if "%s_combo_combined_%s.csv" % (title, estimate_data) not in os.listdir(output_folder):

		# Take the most sensitive fit from each screen in format

		min_xmid_merged_combi = prepare_combi(stage="combo", anchor_dose=False, estimate_data="XMID",
											  m_type="min", subproject=True, del_all=False)

		if exp_type == "anchor":
			min_xmid_merged_combi = min_xmid_merged_combi[min_xmid_merged_combi.RESEARCH_PROJECT.isin(anchor_projects)]
		elif exp_type == "matrix":
			min_xmid_merged_combi = min_xmid_merged_combi[min_xmid_merged_combi.RESEARCH_PROJECT.isin(matrix_projects)]

		min_xmid_merged_combi = min_xmid_merged_combi.reset_index()
		min_xmid_merged_combi = min_xmid_merged_combi.drop(["Unnamed: 0", "index"], axis=1)
		max_indices = list(
			min_xmid_merged_combi.groupby(["SIDM", "DrugComb"])["SYNERGY_DELTA_%s" % estimate_data].idxmax())

		all_combi = min_xmid_merged_combi.loc[max_indices]
		all_combi = all_combi[[col for col in all_combi.columns if col != "index"]]
		all_combi = all_combi.drop_duplicates()

		all_combi.to_csv(output_folder + "%s_combo_combined_%s.csv" % (title, estimate_data), index=False)

	else:
		all_combi = pandas.read_csv(output_folder + "%s_combo_combined_%s.csv" % (title, estimate_data))

	return all_combi


def check_across_reproducibility(stage, anchor_dose, estimate_data, subproject, plotting, m_type, del_all):
	"""
	Reproducibility control across screens
	:param stage: COMBO / DELTA
	:param anchor_dose: Boolean, if comparison will be made with considering anchor doses or not
	:param estimate_data: The estimate that has been used in data filtration - XMID / EMAX
	:param subproject: boolean if the merging will be subproject level or project level
	:param plotting: boolean if the regression plots will be plotted or not
	:param m_type: how data will be selected with- min, median, quantile or max
	:return:
	"""

	if subproject:
		title = "subproject_level"
	else:
		title = "project_level"

	if stage == "combo":
		if estimate_data == "XMID":
			value_column = "SYNERGY_XMID"
		else:
			value_column = "SYNERGY_OBS_EMAX"
	elif stage == "delta":
		value_column = "SYNERGY_DELTA_%s" % estimate_data

	if anchor_dose:
		added_title = "_anchor_dose"
	else:
		added_title = ""

	if del_all:
		del_title = "del"
		del_all = True
	else:
		del_title = "all"
		del_all = False

	total_stats = None

	output_folder = output_path + "data/replicates/across_screens/%s/statistics/" % title
	output_folder_fig = output_path + "data/replicates/across_screens/%s/figures/" % title
	if "%s_%s_%s_%s%s_regression_stat.p" % (estimate_data, del_title, m_type, stage, added_title) not in os.listdir(
			output_folder):

		# Challenge --> different anchor concentration can give us exactly the same combo XMID
		# Example ('SIDM00671', 'Olaparib/Temozolomide', 'GDSC_005-B')
		# Continue with the lower anchor dose
		all_combi_df = prepare_combi(stage=stage, anchor_dose=anchor_dose, estimate_data=estimate_data,
									 m_type=m_type, subproject=subproject, del_all=del_all)

		all_combi_df = all_combi_df[[col for col in all_combi_df.columns if col != "index"]]
		merged_combi_df = all_combi_df.drop_duplicates()

		total_stats = dict()
		if subproject:
			# Number of groups inside projects
			for project, p_df in merged_combi_df.groupby(["project"]):
				subgroup_pair_dfs = dict()
				p_subgroups = p_df.groupby(["RESEARCH_PROJECT"]).groups.keys()
				if len(p_subgroups) > 1:
					try:
						if anchor_dose:
							sp_pivot_df = p_df.pivot(columns="RESEARCH_PROJECT", values=value_column,
													 index=["SIDM", "DrugComb", "ANCHOR_CONC", "Direction"])
						else:
							sp_pivot_df = p_df.pivot(columns="RESEARCH_PROJECT", values=value_column,
													 index=["SIDM", "DrugComb"])
					except ValueError:
						ind = list()
						for i, l in p_df.groupby(["SIDM", "DrugComb", "RESEARCH_PROJECT"]):
							if len(l[value_column].unique()) == 1:
								# Take lower anchor dose
								if len(l.index) > 1:
									if len(l.ANCHOR_CONC.unique()) > 1:
										ind.extend(list(l[l.ANCHOR_CONC == numpy.min(l.ANCHOR_CONC.unique())].index))
								elif len(l.index) == 1:
									ind.extend(list(l.index))
						p_df = p_df.loc[ind]
						sp_pivot_df = p_df.pivot(columns="RESEARCH_PROJECT", values=value_column,
												 index=["SIDM", "DrugComb"])

					for sp in itertools.combinations(p_subgroups, 2):
						t = sp_pivot_df[list(sp)]
						q = t.dropna(axis=0)
						if len(q.index) != 0:
							subgroup_pair_dfs[sp] = q.reset_index()
					subgroup_pairs = list(subgroup_pair_dfs.keys())
					if len(subgroup_pairs) > 0:
						if plotting:
							if len(subgroup_pairs) < 4:
								if len(subgroup_pairs) == 1:
									fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True,
														   squeeze=False)

								else:
									fig, ax = plt.subplots(1, len(subgroup_pairs), figsize=(5 * len(subgroup_pairs), 5),
														   sharex=True, sharey=True, squeeze=False)
							elif len(subgroup_pairs) >= 4:
								s = (len(subgroup_pairs) // 4) + 1
								fig, ax = plt.subplots(s, 4, figsize=(20, 5 * s), sharey=True, sharex=True)

						for i in range(len(subgroup_pairs)):
							df = subgroup_pair_dfs[subgroup_pairs[i]]
							subprojects = [subgroup_pairs[i][0], subgroup_pairs[i][1]]
							num_data_points = len(df.index)

							slope, intercept, r_value, p_value, _ = \
								scipy.stats.linregress(df[subprojects[0]], df[subprojects[1]])

							if project not in total_stats.keys():
								total_stats[project] = {
									subgroup_pairs[i]: {"rvalue": r_value ** 2, "r2pvalue": p_value}}
							else:
								if subgroup_pairs[i] not in total_stats[project].keys():
									total_stats[project][subgroup_pairs[i]] = {"rvalue": r_value ** 2,
																			   "r2pvalue": p_value}

							if plotting:
								if len(subgroup_pairs) < 4:
									plot_density_reg(df=df, col1=subprojects[0], col2=subprojects[1],
													 data_point_name=estimate_data, num_data_points=num_data_points,
													 r2=r_value ** 2, axs=ax[0, i])

								elif len(subgroup_pairs) >= 4:
									a = i // 4
									if i < 4:
										b = i
									else:
										b = i - (4 * (i // 4))

									plot_density_reg(df=df, col1=subprojects[0], col2=subprojects[1],
													 data_point_name=estimate_data, num_data_points=num_data_points,
													 r2=r_value ** 2, axs=ax[a, b])
						"""
						if len(subgroup_pairs) >= 4:
							if (s * 4) / len(subgroup_pairs) > 1.0:
								empty = (s * 4) - len(subgroup_pairs)
								for k in range(empty):
									fig.delaxes(ax[a, b + k + 1])
						"""
						if plotting:
							fig.tight_layout()
							fig.subplots_adjust(top=0.90)
							plt.suptitle("%s - %s merged LR of subprojects" % (project, m_type))
							plt.savefig(output_folder_fig + "%s/%s_%s_%s_%s_%s%s_regressions.pdf" % (
								m_type, project, del_title, m_type, stage, estimate_data, added_title), dpi=300)
							plt.savefig(output_folder_fig + "%s/%s_%s_%s_%s_%s%s_regressions.png" % (
								m_type, project, del_title, m_type, stage, estimate_data, added_title), dpi=300)
							plt.savefig(output_folder_fig + "%s/%s_%s_%s_%s_%s%s_regressions.jpg" % (
								m_type, project, del_title, m_type, stage, estimate_data, added_title), dpi=300)
							plt.close()
		else:
			# Number of projects
			project_pair_dfs = dict()
			p_projects = merged_combi_df.groupby(["project"]).groups.keys()
			try:
				if anchor_dose:
					p_pivot_df = merged_combi_df.pivot(columns="project", values=value_column,
													   index=["SIDM", "DrugComb", "Direction", "ANCHOR_CONC"])
				else:
					p_pivot_df = merged_combi_df.pivot(columns="project", values=value_column,
													   index=["SIDM", "DrugComb"])
			except ValueError:
				ind = list()
				if len(merged_combi_df[value_column].unique()) != 1:
					# Take lower anchor dose
					for i, l in merged_combi_df.groupby(["SIDM", "DrugComb"]):
						if len(l.index) > 1:
							if len(l.ANCHOR_CONC.unique()) > 1:
								ind.extend(list(l[l.ANCHOR_CONC == numpy.min(l.ANCHOR_CONC.unique())].index))
						elif len(l.index) == 1:
							ind.extend(list(l.index))
				merged_combi_df = merged_combi_df.loc[ind]
				p_pivot_df = merged_combi_df.pivot(columns="project", values=value_column,
												   index=["SIDM", "DrugComb"])
			for pp in itertools.combinations(p_projects, 2):
				t = p_pivot_df[list(pp)]
				q = t.dropna(axis=0)
				if len(q.index) != 0:
					project_pair_dfs[pp] = q.reset_index()
			project_pairs = list(project_pair_dfs.keys())
			if len(project_pairs) > 0:
				if len(project_pairs) < 4:
					if len(project_pairs.keys()) == 1:
						fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True, squeeze=False)

					else:
						fig, ax = plt.subplots(1, len(project_pairs), figsize=(5 * len(project_pairs), 5), sharex=True,
											   sharey=True, squeeze=False)

				elif len(project_pairs) >= 4:
					s = (len(project_pairs) // 4) + 1
					fig, ax = plt.subplots(s, 4, figsize=(20, 5 * s), sharey=True, sharex=True)

				for i in range(len(project_pairs)):
					df = project_pair_dfs[project_pairs[i]]
					projects = (project_pairs[i][0], project_pairs[i][1])
					num_data_points = len(df.index)

					slope, intercept, r_value, p_value, _ = \
						scipy.stats.linregress(df[projects[0]], df[projects[1]])

					if projects not in total_stats.keys():
						total_stats[projects] = {"rvalue": r_value ** 2, "r2pvalue": p_value}

					if plotting:
						if len(project_pairs) < 4:
							plot_density_reg(df=df, col1=projects[0], col2=projects[1], data_point_name=estimate_data,
											 num_data_points=num_data_points, r2=r_value ** 2, axs=ax[0, i])

						elif len(project_pairs) >= 4:
							a = i // 4
							if i < 4:
								b = i
							else:
								b = i - (4 * (i // 4))

							plot_density_reg(df=df, col1=projects[0], col2=projects[1], data_point_name=estimate_data,
											 num_data_points=num_data_points, r2=r_value ** 2, axs=ax[a, b])
				"""
				if len(project_pairs) >= 4:
					if (s * 4) / len(project_pairs) > 1.0:
						empty = (s * 4) - len(project_pairs)
						for k in range(empty):
							print(a, b,  k)
							fig.delaxes(ax[a, b + k + 1])
				"""
				if plotting:
					fig.tight_layout()
					fig.subplots_adjust(top=0.90)
					plt.suptitle("%s - %s merged LR of projects"
								 % (projects[0] + "-" + projects[1], m_type))

					plt.savefig(output_folder_fig + "%s_%s_%s_%s%s_regressions.pdf" % (
						del_title, estimate_data, m_type, stage, added_title), dpi=300)
					plt.savefig(output_folder_fig + "%s_%s_%s_%s%s_regressions.png" % (
						del_title, estimate_data, m_type, stage, added_title), dpi=300)
					plt.savefig(output_folder_fig + "%s_%s_%s_%s%s_regressions.jpg" % (
						del_title, estimate_data, m_type, stage, added_title), dpi=300)
					plt.close()

		pickle.dump(total_stats, open(output_folder + "%s_%s_%s_%s%s_regression_stat.p"
									  % (estimate_data, del_title, m_type, stage, added_title), "wb"))

	else:
		total_stats = pickle.load(open(output_folder + "%s_%s_%s_%s%s_regression_stat.p"
									   % (estimate_data, del_title, m_type, stage, added_title), "rb"))

	return total_stats


def is_screen_removal_worth(stage, anchor_dose, estimate_data, subproject, m_type, plotting):
	"""

	:param stage: LIBRARY / COMBO / DELTA
	:param anchor_dose: Boolean, if comparison will be made with considering anchor doses or not
	:param estimate_data: The estimate that has been used in data filtration - XMID / EMAX
	:param subproject: boolean if the merging will be subproject level or project level
	:param plotting: boolean if the regression plots will be plotted or not
	:param m_type: how data will be selected with- min, median, quantile or max
	:return:
	"""
	group_cols = ["SIDM", "DrugComb"]

	if stage == "combo":
		if estimate_data == "XMID":
			value_column = "SYNERGY_XMID"
		else:
			value_column = "SYNERGY_OBS_EMAX"
	elif stage == "delta":
		value_column = "SYNERGY_DELTA_%s" % estimate_data

	if anchor_dose:
		added_title = "_anchor_dose"
		group_cols.extend(["Direction", "ANCHOR_CONC", "maxc"])
	else:
		added_title = ""

	if subproject:
		title = "subproject_level"
		group_cols.extend(["RESEARCH_PROJECT", "project"])
	else:
		title = "project_level"
		group_cols.append("project")

	output_folder = output_path + "data/replicates/across_screens/%s/statistics/" % title
	output_folder_fig = output_path + "data/replicates/across_screens/%s/figures/" % title

	if "del_all_%s_%s_%s%s_comparison_df.csv" % (stage, estimate_data, m_type, added_title) not in os.listdir(
			output_folder):

		bad_pairs = get_bad_fits(stage=stage, anchor_dose=anchor_dose, estimate_data=estimate_data,
								 subproject=subproject)

		all_combi = prepare_combi(stage=stage, anchor_dose=anchor_dose, estimate_data=estimate_data,
								  m_type=m_type, subproject=subproject, del_all=False)

		del_combi = prepare_combi(stage=stage, anchor_dose=anchor_dose, estimate_data=estimate_data,
								  m_type=m_type, subproject=subproject, del_all=True)

		del_combi["selected"] = False
		all_combi["selected"] = False

		affected_cl_dp = list()
		for k in bad_pairs:
			if (k[0], k[1]) not in affected_cl_dp:
				affected_cl_dp.append((k[0], k[1]))

		del_lost = list()
		all_missing = list()
		i, t = 0, len(affected_cl_dp)
		del_df_groups = dict(del_combi.groupby(["SIDM", "DrugComb"]).groups)
		all_df_groups = dict(all_combi.groupby(["SIDM", "DrugComb"]).groups)
		for pair in affected_cl_dp:
			del_indices = list()
			all_indices = list()

			if pair in del_df_groups.keys():
				del_indices = list(del_df_groups[pair])
				del_combi.loc[del_indices, "selected"] = True
			else:
				del_lost.append(k)
			if pair in all_df_groups.keys():
				all_indices = list(all_df_groups[pair])
				all_combi.loc[all_indices, "selected"] = True
			else:
				all_missing.append(k)
			i += 1
			print(i * 100.0 / t)

		del_combi = del_combi[del_combi.selected]
		all_combi = all_combi[all_combi.selected]

		all_data = list()
		for i in all_combi.groupby(group_cols).groups.keys():
			if i not in all_data:
				all_data.append(i)

		lost_data = list()
		for i in del_lost:
			if (i[0], i[1]) not in all_data:
				if (i[0], i[1]) not in lost_data:
					lost_data.append((i[0], i[1]))

		mdf = all_combi.merge(del_combi, left_on=group_cols, right_on=group_cols,
							  suffixes=("_all", "_del"), how="left")
		comparison_df = mdf[group_cols + ["%s_all" % value_column, "%s_del" % value_column]]
		comparison_df.to_csv(output_folder + "del_all_%s_%s_%s%s_comparison_df.csv"
							 % (stage, estimate_data, m_type, added_title), index=False)

		pickle.dump(lost_data, open(output_folder + "del_lost_pairs_%s_%s_%s%s.p"
									% (stage, estimate_data, m_type, added_title), "wb"))
		pickle.dump(all_data, open(output_folder + "all_pairs_%s_%s_%s%s.p"
								   % (stage, estimate_data, m_type, added_title), "wb"))
		pickle.dump(all_missing, open(output_folder + "all_missing_%s_%s_%s%s.p"
									  % (stage, estimate_data, m_type, added_title), "wb"))

	else:
		lost_data = pickle.load(open(output_folder + "del_lost_pairs_%s_%s_%s%s.p"
									 % (stage, estimate_data, m_type, added_title), "rb"))
		all_data = pickle.load(open(output_folder + "all_pairs_%s_%s_%s%s.p"
									% (stage, estimate_data, m_type, added_title), "rb"))
		comparison_df = pandas.read_csv(output_folder + "del_all_%s_%s_%s%s_comparison_df.csv"
										% (stage, estimate_data, m_type, added_title))

	if plotting:
		# comparison_df = comparison_df[(~pandas.isna(comparison_df.selected_del)) & (comparison_df.selected_del)]

		# Distribution comparison of the averaged dropped and non-dropped viabilities

		melt_comparison_df = pandas.melt(
			comparison_df,
			id_vars=[col for col in comparison_df.columns
					 if col not in ["%s_all" % value_column, "%s_del" % value_column]],
			value_vars=["%s_all" % value_column, "%s_del" % value_column])

		sns.displot(data=melt_comparison_df, x="value", col="variable", kind="kde")
		plt.xlabel("%s %s" % (stage, estimate_data), fontsize=14)
		plt.ylabel("Density", fontsize=14)
		plt.title("Density Distribution of %s %s from affected pairs (# %d)"
				  % (stage, estimate_data, len(comparison_df.index)))
		plt.savefig(output_folder_fig + "del_all_%s_%s_%s%s_distribution.pdf"
					% (stage, estimate_data, m_type, added_title), dpi=300)
		plt.savefig(output_folder_fig + "del_all_%s_%s_%s%s_distribution.jpg"
					% (stage, estimate_data, m_type, added_title), dpi=300)
		plt.savefig(output_folder_fig + "del_all_%s_%s_%s%s_distribution.png"
					% (stage, estimate_data, m_type, added_title), dpi=300)
		plt.close()

		# Check and plot the association

		if len(comparison_df.index) != 0:
			num_data_points = len(comparison_df.index)
			print(comparison_df["%s_del" % value_column].values)
			slope, intercept, r_value, p_value, _ = \
				scipy.stats.linregress(comparison_df["%s_all" % value_column].values,
									   comparison_df["%s_del" % value_column].values)

			fig = plt.figure()

			sns.scatterplot(data=comparison_df, x="%s_all" % value_column, y="%s_del" % value_column,
							color="navy", alpha=0.4)
			plt.xlabel("%s without removal" % " ".join(value_column.split("_")))
			plt.ylabel("%s with removal" % " ".join(value_column.split("_")))

			plt.legend(title="$R^2$ %.3f\n# data %d" % (r_value, len(comparison_df.index)), loc="upper left")

			ax = plot_density_reg(df=comparison_df, col1="%s_all" % value_column, col2="%s_del" % value_column,
								  data_point_name="%s %s" % (stage, estimate_data),
								  num_data_points=len(comparison_df.index), r2=r_value ** 2, axs=fig.subplots())

			plt.title("Comparison of combined %s %s %s" % (m_type, stage, estimate_data))

			plt.savefig(output_folder_fig + "del_all_%s_%s_%s%s_comparison.pdf"
						% (stage, estimate_data, m_type, added_title), dpi=300)
			plt.savefig(output_folder_fig + "del_all_%s_%s_%s%s_comparison.jpg"
						% (stage, estimate_data, m_type, added_title), dpi=300)
			plt.savefig(output_folder_fig + "del_all_%s_%s_%s%s_comparison.png"
						% (stage, estimate_data, m_type, added_title), dpi=300)
			plt.close()

	else:
		if len(comparison_df.index) != 0:
			num_data_points = len(comparison_df.index)
			slope, intercept, r_value, p_value, _ = \
				scipy.stats.linregress(comparison_df["%s_all" % value_column].values,
									   comparison_df["%s_del" % value_column].values)

	if len(comparison_df.index) != 0:
		print("R square: %.3f\nP value: %.3f\nPercentage of lost data: %d" %
			  (r_value ** 2, p_value, len(lost_data) * 100.0 / len(all_data)))
		if r_value ** 2 > 0.75 and p_value < 0.05:
			return False
		else:
			return True
	else:
		return None


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
												  m_type="min", subproject=False, del_all=False)
			min_xmid_merged_combi = min_xmid_merged_combi.reset_index()
			min_xmid_merged_combi = min_xmid_merged_combi.drop(["index"], axis=1)
			max_indices = list(
				min_xmid_merged_combi.groupby(["SIDM", "DrugComb"])["SYNERGY_DELTA_%s" % estimate_data].idxmax())

			all_combi = min_xmid_merged_combi.loc[max_indices]
			all_combi = all_combi[[col for col in all_combi.columns if col != "index"]]
			all_combi = all_combi.drop_duplicates()

			all_combi["screen_type"] = all_combi.apply(
				lambda x: "anchor" if x.RESEARCH_PROJECT in anchor_projects else "matrix", axis=1)

			all_combi["tissue_type"] = None
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

			all_combi["tissue_type"] = None
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


# ---------------------------------------------------------------------------#
#                                   R U N                                    #
# ---------------------------------------------------------------------------#


def main(args):
	if args["RUN_FOR"] == "viability":
		# Collect and annotate viability files
		_ = get_all_viabilities(treatment=args["TREATMENT"])

	# Replicates

	if args["RUN_FOR"] == "count":
		# Count replicates and calculate sd
		_ = count_replicates(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"], type_of_merge=args["MERGE"],
							 answer_t=args["TR_DEL"])

	if args["RUN_FOR"] == "variance":
		# Plot variance
		_ = plot_variance(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"], type_of_merge=args["MERGE"],
						  answer_t=args["TR_DEL"])

	if args["RUN_FOR"] == "regress":
		# Replicate regression across subprojects and experiment dates
		_ = regression_replicates(replicate_type=args["REP_TYPE"], plotting=True, treatment=args["TREATMENT"],
								  type_of_merge=args["MERGE"], answer_t=args["TR_DEL"])

		# Get outliers
		_ = get_replicate_outliers(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"],
								   type_of_merge=args["MERGE"], answer_t=args["TR_DEL"])

	if args["RUN_FOR"] == "statistics":
		# Collect all statistics
		_ = statistics_replicates(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"],
								  type_of_merge=args["MERGE"], answer_t=args["TR_DEL"])

	if args["RUN_FOR"] == "bad_replicates":
		# Decide bad replicates
		_ = bad_replicate(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"], type_of_merge=args["MERGE"],
						  answer_t=args["TR_DEL"])

		# Get bad replicate pairs
		_ = get_bad_pairs(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"], type_of_merge=args["MERGE"],
						  del_all_t=args["TR_DEL"],
						  del_all_b=args["BR_DEL"])

	if args["RUN_FOR"] == "worth":
		# Check if remowal worth
		_ = is_removal_worth(replicate_type=args["REP_TYPE"], treatment=args["TREATMENT"], type_of_merge=args["MERGE"],
							 plotting=True, t_answer=args["TR_DEL"])

	# Combi

	if args["RUN_FOR"] == "combi":
		# Collect and annotate combi files
		_ = get_all_combi()

	if args["RUN_FOR"] == "bad_fits":
		# Decide bad fits
		_ = get_bad_fits(stage="combo", anchor_dose=args["ANCHOR_DOSE"], estimate_data="XMID",
						 subproject=args["SUBPROJECT"])

	if args["RUN_FOR"] == "select_fits":
		# Select the most sensitive fit inside each screen
		_ = prepare_combi(stage="combo", anchor_dose=args["ANCHOR_DOSE"], estimate_data="XMID",
						  m_type="min", subproject=args["SUBPROJECT"], del_all=args["COMBI_DEL"])

	if args["RUN_FOR"] == "regress_combi":
		_ = check_across_reproducibility(stage="combo", anchor_dose=args["ANCHOR_DOSE"], estimate_data="XMID",
										 subproject=args["SUBPROJECT"], plotting=True, m_type=args["MERGE"],
										 del_all=args["COMBI_DEL"])

	if args["RUN_FOR"] == "worth_combi":
		_ = is_screen_removal_worth(stage="combo", anchor_dose=args["ANCHOR_DOSE"], estimate_data="XMID",
									subproject=args["SUBPROJECT"], m_type=args["MERGE"], plotting=True)

	if args["RUN_FOR"] == "combine":
		_ = combine_combi(estimate_data="XMID", treatment=args["TREATMENT"])

	return True


if __name__ == '__main__':
	args = take_input()

	from CombDrug.module.data.viability import Viability, deserialise_viability_object

	if args["TR_DEL"] in ["False", "None", None, False]:
		args["TR_DEL"] = False
	elif args["TR_DEL"] in ["True", True]:
		args["TR_DEL"] = True

	if args["BR_DEL"] in ["False", "None", None, False]:
		args["BR_DEL"] = False
	elif args["BR_DEL"] in ["True", True]:
		args["BR_DEL"] = True

	if args["ANCHOR_DOSE"] in ["False", "None", None, False]:
		args["ANCHOR_DOSE"] = False
	elif args["ANCHOR_DOSE"] in ["True", True]:
		args["ANCHOR_DOSE"] = True

	if args["SUBPROJECT"] in ["False", "None", None, False]:
		args["SUBPROJECT"] = False
	elif args["SUBPROJECT"] in ["True", True]:
		args["SUBPROJECT"] = True

	if args["COMBI_DEL"] in ["False", "None", None, False]:
		args["COMBI_DEL"] = False
	elif args["COMBI_DEL"] in ["True", True]:
		args["COMBI_DEL"] = True

	print(args)
	_ = main(args=args)

