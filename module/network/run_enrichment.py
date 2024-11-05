"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 20 February 2023
Last Update : 11 June 2024
Input : Pathway Analysis
Output : Pathway results
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, networkx, pickle, argparse, re
from CombDrug.module.path import output_path
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *

# ---------------------------------------------------------------------------#
#	      			D R U G  C O M B - Pathway Analysis
#	           			PhD Project of Cansu Dincer
#	            		 Wellcome Sanger Institute
# ---------------------------------------------------------------------------#


# -----------------------------------------------------------------------------------------#
# Take inputs

def take_input():
	parser = argparse.ArgumentParser()

	for group in parser._action_groups:
		if group.title == "optional arguments":
			group.title = "Inputs"
		elif "positional arguments":
			group.title = "Mandatory Inputs"

	# BASIC INFORMATION

	parser.add_argument("-run", dest="RUN_FOR")
	parser.add_argument("-drug", dest="DRUG")
	parser.add_argument("-pathway_tool", dest="PT")
	parser.add_argument("-pathway_db", dest="PDB")

	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict



# ---------------------------------------------------------------------------#
#	      			D R U G  C O M B - Pathway Analysis
#	           			PhD Project of Cansu Dincer
#	            		 Wellcome Sanger Institute
# ---------------------------------------------------------------------------#

def create_pathway_matrix(pathway_list, pathway_dict):
	matrix = dict()
	for drug, df in pathway_dict.items():
		d = dict()
		for p in pathway_list:
			d[p] = 0.0
		for i, row in df.iterrows():
			if row.pathways in d.keys():
				d[row.pathways] = row.ratio
		matrix[drug] = d
	pathway_matrix = pandas.DataFrame(matrix)
	return pathway_matrix


def extract_enrichr_pathway_terms(pathway_term, db):

	if db == "GOBP":
		return pathway_term.split("(GO")[0]
	elif db == "KEGG":
		return pathway_term
	elif db == "Wiki":
		return " ".join(pathway_term.split(" ")[:-1])
	elif db == "Reactome":
		return pathway_term.split(" R-HSA-")[0]



def run_PA(pathway_tool, pathway_database, interactome_name, random_tag, seed):
	"""
	Run R Python script for pathway analysis
	:param pathway_tool: enrichr / webgestaltR
	:param interactome_name: Name of the used interactome
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	"""
	alpha = 0.55

	samples = list()
	for fold in os.listdir(output_path + "network/modelling/PPR/%s/drugs/empiric/" % interactome_name):
		if "ppr_network_%s_%d_empiric_weighted.csv" % (fold, alpha *100) in os.listdir(output_path + "network/modelling/PPR/%s/drugs/empiric/%s/interactome/network/" % (interactome_name, fold)):
			df = pandas.read_csv(output_path + "network/modelling/PPR/%s/drugs/empiric/%s/interactome/network/ppr_network_%s_%d_empiric_weighted.csv"
								 % (interactome_name, fold, fold, alpha *100))
			if len(df.index) > 0:
				samples.append(fold)

	if random_tag:
		random_text = "_random_" + str(seed)
		random_col = "/random/"
	else:
		random_text = "_empiric"
		random_col = "/empiric/"

	if pathway_database == "GO_Biological_Process_2023":
		pathway_db = "GOBP"
	elif pathway_database == "KEGG_2021_Human":
		pathway_db = "KEGG"
	elif pathway_database == "Reactome_2022":
		pathway_db = "Reactome"
	elif pathway_database == "WikiPathway_2023_Human":
		pathway_db = "Wiki"
	else: pathway_db = ""


	oe_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/network/modelling/ppr/pathway%s" % random_col
	run_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/CombDrug/module"

	for sample in samples:
		sample_col_text = "_".join(sample.split(" "))
		sample_col_text = sample_col_text.replace("(", "")
		sample_col_text = sample_col_text.replace(")", "")

		if pathway_tool == "webgestaltr":
			os.system("bsub -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
					  "-o %s%s_pathway_%s_%d.o -e %s%s_pathway_%s_%d.e -q normal -J '%s_%d' "
					  "Rscript %s/network/R_webgestaltR.R --interactome '%s' --drug '%s' --random '%s' --s '%s' --enrichment_method '%s' --a '%s' "
					  % (oe_path, pathway_tool, sample_col_text, int(alpha*100), oe_path, pathway_tool, sample_col_text, int(alpha*100),
						 sample, int(alpha*100), run_path, interactome_name, sample, "yes" if random_tag else "no",
						 str(seed) if random_tag else "no", "ORA", str(alpha)))

		elif pathway_tool == "enrichr":
			os.system("bsub -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
					  "-o %s%s_pathway_%s_%s_%s_%d.o -e %s%s_pathway_%s_%s_%s_%d.e -q normal -J '%s_%s_%d' "
					  "python3 %s/network/py_enrichr.py -drug '%s' -interactome_name '%s' %s -pathway_database '%s' -a '%s' "
					  % (oe_path, pathway_tool, pathway_db, interactome_name, sample_col_text, int(alpha*100),
						 oe_path, pathway_tool, pathway_db, interactome_name, sample_col_text, int(alpha*100),
						 pathway_db, sample_col_text, int(alpha*100), run_path, sample,  interactome_name,
						 "-random -s " + seed if random_tag else "", pathway_database, str(alpha)))

	return True



def refine_pathways(interactome_name, alpha, sample, pathway_tool, pathway_database, random_tag):

	if random_tag: random_col = "/random/"
	else: random_col = "/empiric/"
	alpha = 0.55

	if pathway_database == "GO_Biological_Process_2023":
		pathway_db = "GOBP"
	elif pathway_database == "KEGG_2021_Human":
		pathway_db = "KEGG"
	elif pathway_database == "Reactome_2022":
		pathway_db = "Reactome"
	elif pathway_database == "WikiPathway_2023_Human":
		pathway_db = "Wiki"
	else: pathway_db = ""


	pathway_path = output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/analysis/%s/" % (interactome_name, pathway_db)

	if sample in os.listdir(pathway_path):
		for file in os.listdir(pathway_path + "%s/" % sample):
			if file.split("_")[-3] == str(int(alpha * 100)) and file.split("_")[-2] == pathway_db:
				new_file_name = "Refined_" + file

				if pathway_tool == "enrichr":
					df = pandas.read_csv(pathway_path + "%s/%s" %(sample, file), sep ="\t")
					df = df[df["Adjusted P-value"] < 0.05]
					if len(df.index) > 0:
						df["Enriched_Pathways"] = df.apply(lambda x: extract_enrichr_pathway_terms(x["Term"], pathway_db), axis=1)

						# Eliminate diseases including infections, cancer, addiction related pathways from results
						df["selected"] = df.apply(lambda x: False if re.search(r"\bhepatitis\b|\bcancer\b|\bvirus\b|cardio|african|malaria|addiction|influenza|Shigellosis|huntington|infection|disease|Measles|tuberculosis|carcinoma|Leishmaniasis|carcinogenesis|viral|leukemia|melano|cocaine|Legionellosis|Amoebiasis|Alcoholism|diabetic|rejection|Bacterial|syndrome|Glioma|depression|deficiency|obesity|Metastatic|Disorders|Glioblastoma|toxin|diphtheria|anthrax|HIV|Anemia|sclerosis|Pertussis|immunodeficiency|arthritis|Toxoplasmosis|Infertility|anxiety|Listeria|Ovarian tumor|lupus|aging|Injury|Glomerulosclerosis|Pathogen|sepsis|Host|botulinum|provirus|tumoral|cystic fibrosis|diabetes|ataxia|neurodegeneration|SARS-CoV|potentiation|differentiation|Thermogenesis", x.Enriched_Pathways, re.IGNORECASE) is not None else True, axis=1)

						df = df[df.selected]
						df = df[["Enriched_Pathways", "Adjusted P-value", "Odds Ratio"]]
						df.columns = ["pathways", "adjpvalue", "ratio"]

				elif pathway_tool == "webgestaltr":
					df = pandas.read_csv(pathway_path + "%s/%s" % (sample, file), sep ="\t")
					df = df[df["FDR"] < 0.05]

					# Eliminate diseases including infections, cancer, addiction related pathways from results
					df["selected"] = df.apply(lambda x: False if re.search(r"\bhepatitis\b|\bcancer\b|\bvirus\b|cardio|african|malaria|addiction|influenza|Shigellosis|huntington|infection|disease|Measles|tuberculosis|carcinoma|Leishmaniasis|carcinogenesis|viral|leukemia|melano|cocaine|Legionellosis|Amoebiasis|Alcoholism|diabetic|rejection|Bacterial|syndrome|Glioma|depression|deficiency|obesity|Metastatic|Disorders|Glioblastoma|toxin|diphtheria|anthrax|HIV|Anemia|sclerosis|Pertussis|immunodeficiency|arthritis|Toxoplasmosis|Infertility|anxiety|Listeria|Ovarian tumor|lupus|aging|Injury|Glomerulosclerosis|Pathogen|sepsis|Host|botulinum|provirus|tumoral|cystic fibrosis|diabetes|ataxia|neurodegeneration|SARS-CoV|potentiation|differentiation|Thermogenesis", x.description, re.IGNORECASE) is not None else True, axis=1)

					df = df[df.selected]
					df = df[["description", "FDR", "enrichmentRatio"]]
					df.columns = ["pathways", "adjpvalue", "ratio"]

				df.to_csv(pathway_path + "%s/%s" % (sample, new_file_name), sep="\t")

	return True



def main(args):

	if args["RUN_FOR"] == "pathway":
		_ = run_PA(pathway_tool=args["PT"], interactome_name="intact", random_tag=False, seed=None, pathway_database=args["PDB"])

	if args["RUN_FOR"] == "refine_pathway":
		_ = refine_pathways(interactome_name="intact", alpha=0.55, pathway_tool=args["PT"], pathway_database=args["PDB"], random_tag=False, sample=args["DRUG"])


if __name__ == '__main__':

	args = take_input()
	print(args)
	_ = main(args=args)
