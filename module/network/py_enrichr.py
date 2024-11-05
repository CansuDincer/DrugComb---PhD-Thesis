"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 13 June 2023
Last Update : 12 February 2024
Input : Pathway Analysis
Output : Pathway results
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, pickle, networkx, argparse
import gseapy
from gseapy.plot import barplot, dotplot
from CombDrug.module.path import output_path
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.network.pcsf_statistics import check_file_exist

# ---------------------------------------------------------------------------#
#	      			D R U G  C O M B - Pathway Analysis
#	           			PhD Project of Cansu Dincer
#	            		 Wellcome Sanger Institute
# ---------------------------------------------------------------------------#

# ---------------------------------------------------------------------------#
#                               	Inputs                                   #
# ---------------------------------------------------------------------------#

def take_input():
	parser = argparse.ArgumentParser(prog="CombDrug Pathway Analysis - EnrichR",
									 usage="%(prog)s [inputs]",
									 description="""
                                     **********************************
                                     		   Find pathways
                                     **********************************""")

	for group in parser._action_groups:
		if group.title == "optional arguments":
			group.title = "Inputs"
		elif "positional arguments":
			group.title = "Mandatory Inputs"


	# Feature
	parser.add_argument("-drug", dest="DRUG", default=None)
	parser.add_argument("-interactome_name", dest="INTERACTOME", required=True)
	parser.add_argument("-pathway_database", dest="PATHWAY", required=True)
	parser.add_argument("-a", dest="ALPHA", required=True)
	parser.add_argument("-random", dest="RANDOM", action="store_true")
	parser.add_argument("-s", dest="SEED")
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                                  Functions                                 #
# ---------------------------------------------------------------------------#


def run_enrichr(drug_name, interactome_name, pathway_database, alpha, random_tag, seed):
	"""
	Run EnrichR function from gseapy package
	:param drug_name: Name of single-egnt or drug combination name
	:param interactome_name : Name of the used interactome
	:param pathway_database: 'GO_Biological_Process_2021', 'KEGG_2021_Human', 'Reactome_2022'
	:param alpha: Damping parameter for PageRank, default=0.85
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:return:
	"""

	if random_tag:
		random_text = "_random_%d" % seed
		folder = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/network/modelling/PPR/drugs/random/modules/%s/" % drug_name
	else:
		random_text = "_empiric"
		folder = output_path + "network/modelling/PPR/%s/drugs/empiric/%s/" % (interactome_name, drug_name)

	if pathway_database == "GO_Biological_Process_2023":
		pathway_db = "GOBP"
	elif pathway_database == "KEGG_2021_Human":
		pathway_db = "KEGG"
	elif pathway_database == "Reactome_2022":
		pathway_db = "Reactome"
	elif pathway_database == "WikiPathway_2023_Human":
		pathway_db = "Wiki"
	else: pathway_db = ""

	alpha_text = str(int(float(alpha) * 100))

	try:
		os.mkdir(output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/" % interactome_name)
	except FileExistsError: pass

	try:
		os.mkdir(output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/analysis/" % interactome_name)
	except FileExistsError: pass

	try:
		os.mkdir(output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/gene_lists/" % interactome_name)
	except FileExistsError: pass

	try:
		os.mkdir(output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/analysis/%s/" % (interactome_name, pathway_db))

	except FileExistsError: pass

	try:
		os.mkdir(output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/analysis/%s/%s/"
				 % (interactome_name, pathway_db, drug_name))

	except FileExistsError: pass

	pathway_path = output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/analysis/%s/%s/" % (interactome_name, pathway_db, drug_name)

	node_path = output_path + "network/modelling/PPR/%s/drugs/pathway_analysis/enrichr/gene_lists/" % interactome_name

	new_name = None
	if "%s_%s_module_gene_set_list.txt" % (drug_name, alpha_text) not in os.listdir(node_path):
		nodes = None
		if "ppr_network_%s_%s%s_weighted.csv" % (drug_name, alpha_text, random_text) in os.listdir(folder + "interactome/network/"):
			nodes = pandas.read_csv(folder + "interactome/network/ppr_network_%s_%s%s_weighted.csv"
									% (drug_name, alpha_text, random_text))[["From", "To"]]
			if nodes.empty:
				nodes = None

		if nodes is not None:
			gene_list1 = list(nodes["From"].unique())
			gene_list2 = list(nodes["To"].unique())
			gene_list = gene_list1 + gene_list2
			gene_list_df = pandas.DataFrame(gene_list)
			gene_list_df.columns = ["genes"]

			gene_list_df.to_csv(node_path + "%s_%s_module_gene_set_list.txt"
								% (drug_name, alpha_text), header=None, index=False)

	else:
		gene_list_df = pandas.read_csv(node_path + "%s_%s_module_gene_set_list.txt" % (drug_name, alpha_text), header=None)

	if len(gene_list_df.index) > 0:
		gene_list_path = node_path + "%s_%s_module_gene_set_list.txt" % (drug_name, alpha_text)
		new_name = "%s_%s_%s_pathways" % (drug_name, alpha_text, pathway_db)

	if new_name is not None:
		try:
			gseapy.enrichr(gene_list=gene_list_path, gene_sets=pathway_database, organism='Human',
						   outdir=pathway_path, cutoff=0.05)

			old_name = "%s.Human.enrichr.reports" % pathway_database

			try:
				os.rename("%s%s.txt" % (pathway_path, old_name),
						  "%s%s.txt" % (pathway_path, new_name))

				for item in os.listdir(pathway_path):
					if item.endswith(".log") or item.endswith(".pdf"):
						os.remove("%s%s" % (pathway_path, item))
			except FileNotFoundError: pass

		except ValueError: pass
	return 1



def main(args):
	run_enrichr(drug_name = args["DRUG"], interactome_name=args["INTERACTOME"], pathway_database=args["PATHWAY"],
				alpha=args["ALPHA"], random_tag=args["RANDOM"], seed=args["SEED"])
	return True



if __name__ == '__main__':

	args = take_input()

	print(args)

	_ = main(args=args)
	