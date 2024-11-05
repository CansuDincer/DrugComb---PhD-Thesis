"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 02 February 2023
Last Update : 28 February 2024
Input : Network Analysis
Output : Network Models
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, sys, time, numpy

if "/lustre/scratch125/casm/team215mg/cd7/CombDrug/" not in list(sys.path):
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/")
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/")

from CombDrug.module.path import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.drug import *

# ---------------------------------------------------------------------------#
#	      			D R U G  C O M B - Network Analysis
#	           			PhD Project of Cansu Dincer
#	            		 Wellcome Sanger Institute
# ---------------------------------------------------------------------------#

# ---------------------------------------------------------------------------#
# Uniprot
# ---------------------------------------------------------------------------#
"""
os.system("bsub -n 4 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' -M10000 -o /lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/network/uniprot/uniprot_swiss.o -e /lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/network/uniprot/uniprot_swiss.e -q long -J 'uni_sws' python3 CombDrug/module/data/dataset_info.py")
"""


# ---------------------------------------------------------------------------#
# Interactome
# ---------------------------------------------------------------------------#

def run_interactome(run_for):
	log_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/network/interactome"

	if run_for == "empiric_interactome":
		os.system("bsub -n 4 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' -M10000 "
				  "-o %s/interactome.o -e %s/interactome.e -q long -J 'int' "
				  "python3 CombDrug/module/network/interactome.py" % (log_path, log_path))

	elif run_for == "random_interactome":
		for seed in range(1000):
			# Run randomisation and random interactome normalisation
			print(seed)
			os.system("bsub -n 4 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' -M10000 "
					  "-o %s/random/norm_random_interactome_seed_%d.o -e %s/random/norm_random_interactome_seed_%d.e -q long "
					  "-J 'rand_int' python3 CombDrug/module/network/interactome.py -seed %d"
					  % (log_path, seed, log_path, seed, seed))
	return True


# ---------------------------------------------------------------------------#
# Drug Modulation
# ---------------------------------------------------------------------------#

def run_network_modelling(run_for, alphas, random, seed):
	if random:
		seed_text = "%d" % int(seed)
		random_title = "random"
		random_suffix = "_seed_%d" % int(seed)
		log_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/logs/network/modelling/ppr/random"
	else:
		seed_text = "None"
		random_title = "empiric"
		random_suffix = ""
		log_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/network/modelling/ppr"

	if run_for == "length":

		os.system("bsub -n 2 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' -M10000 "
				  "-o %s/lenghts/%s_drug_module_all_shortest_paths_nw%s.o -e %s/lenghts/%s_drug_module_all_shortest_paths_nw%s.e "
				  "-q normal -J 'shotp_uw_%s%s' python3 CombDrug/module/network/drug_modulation.py -run '%s' -seed '%s'"
				  % (log_path, random_title, random_suffix, log_path, random_title, random_suffix, random_title,
					 random_suffix, run_for, seed_text))

	elif run_for == "ppr":

		# Run PPR
		all_drugs = all_screened_compounds(project_list=None, integrated=True)
		c, t = 0, len(all_drugs)

		for alpha in alphas:
			alpha_text = int(alpha * 100)
			for drug in all_drugs:
				drug_text = "_".join(drug.split(" ")).replace("(", "").replace(")", "")
				print(drug + " - " + str(alpha_text))
				os.system("bsub -n 2 -G team215-grp -R'select[mem>2000] rusage[mem=2000]' -M2000 "
						  "-o %s/modules/%s_%s_%s%s.o -e %s/modules/%s_%s_%s%s.e -q normal -J 'ppr_%s_%s_%s%s' "
						  "python3 CombDrug/module/network/drug_modulation.py -drug '%s' -a '%s' -run '%s' -seed '%s'"
						  % (log_path, random_title, drug_text, alpha_text, random_suffix,
							 log_path, random_title, drug_text, alpha_text, random_suffix, drug_text, alpha_text,
							 random_title, random_suffix,
							 drug, str(alpha), run_for, seed_text))
				c += 1
				print(c * 100.0 / t)

	elif run_for == "module":

		# Get all modules into a dictionary
		for alpha in alphas:
			alpha_text = int(alpha * 100)
			os.system("bsub -n 2 -G team215-grp -R'select[mem>2000] rusage[mem=2000]' -M2000 "
					  "-o %s/%s_drug_modules_%s%s.o -e %s/%s_drug_modules_%s%s.e -q normal -J 'modules_%s_%s%s' "
					  "python3 CombDrug/module/network/drug_modulation.py -a '%s' -run '%s' -seed '%s'"
					  % (log_path, random_title, alpha_text, random_suffix, log_path, random_title, alpha_text,
						 random_suffix,
						 alpha_text, random_title, random_suffix, str(alpha), run_for, seed_text))

	elif run_for == "run_diameter":

		# Run Diameters
		# all_drugs = all_screened_compounds(project_list=None, integrated=True)
		for alpha in alphas:
			alpha_text = int(alpha * 100)
			for drug in ['Vismodegib']:  # all_drugs:
				drug_text = "_".join(drug.split(" ")).replace("(", "").replace(")", "")
				print(drug + " - " + str(alpha_text))
				os.system("bsub -n 2 -G team215-grp -R'select[mem>2000] rusage[mem=2000]' -M2000 "
						  "-o %s/diameters/%s_%s_%s%s.o -e %s/diameters/%s_%s_%s%s.e -q normal -J 'dia_%s_%s_%s%s' "
						  "python3 CombDrug/module/network/drug_modulation.py -drug '%s' -a '%s' -run '%s' -seed '%s'"
						  % (log_path, random_title, drug_text, alpha_text, random_suffix,
							 log_path, random_title, drug_text, alpha_text, random_suffix,
							 drug_text, alpha_text, random_title, random_suffix, drug, str(alpha), run_for, seed_text))

	elif run_for == "diameter":

		# Get all diameters into a dictionary
		for alpha in alphas:
			alpha_text = int(alpha * 100)
			os.system("bsub -n 2 -G team215-grp -R'select[mem>6000] rusage[mem=6000]' -M6000 "
					  "-o %s/%s_drug_modules_diameters_%s%s.o -e %s/%s_drug_modules_diameters_%s%s.e -q normal -J 'dia_%s_%s%s' "
					  "python3 CombDrug/module/network/drug_modulation.py -a '%s' -run '%s' -seed '%s'"
					  % (log_path, random_title, alpha_text, random_suffix, log_path, random_title, alpha_text,
						 random_suffix,
						 alpha_text, random_title, random_suffix, str(alpha), run_for, seed_text))

	elif run_for == "run_similarity":

		selected_alpha = [0.55]

		# Run similarity in parallel
		all_combinations = all_screened_combinations(project_list=None, integrated=True)
		i, total = 0, len(all_combinations)
		for drugs in all_combinations:
			drug_text = "_".join("_".join(drugs.split("/")).split(" "))
			drug_text = drug_text.replace("(", "")
			drug_text = drug_text.replace(")", "")
			for alpha in selected_alpha:
				alpha_text = int(alpha * 100)
				print((i * 100.0) / total)
				os.system("bsub -n 2 -G team215-grp -R'select[mem>20000] rusage[mem=20000]' "
						  "-M20000 -o '%s/similarity/%s_similarity_%s_%s%s.o' -e '%s/similarity/%s_similarity_%s_%s%s.e' -q normal "
						  "-J 'sim_%s_%s%s' python3 CombDrug/module/network/drug_modulation.py -d '%s' -a '%s' -run '%s' -seed '%s'"
						  % (log_path, random_title, drug_text, alpha_text, random_suffix,
							 log_path, random_title, drug_text, alpha_text, random_suffix,
							 drug_text, random_title, random_suffix, drugs, str(alpha), run_for, seed_text))
				i += 1

	elif run_for == "similarity":
		selected_alpha = 0.55
		alpha_text = int(selected_alpha * 100)
		os.system("bsub -n 2 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' "
				  "-M10000 -o '%s/%s_drug_modules_similarity_%s%s.o' -e '%s/%s_drug_modules_similarity_%s%s.e' -q normal "
				  "-J 'sim_%s_%s%s' python3 CombDrug/module/network/drug_modulation.py -a '%s' -run '%s' -seed '%s'"
				  % (
				  log_path, random_title, alpha_text, random_suffix, log_path, random_title, alpha_text, random_suffix,
				  alpha_text, random_title, random_suffix, str(alpha_text), run_for, seed_text))


	elif run_for == "random_distance":
		for dist_type in ["dd"]:  # , "db"]:
			os.system("bsub -n 2 -G team215-grp -R'select[mem>20000] rusage[mem=20000]' "
					  "-M20000 -o '%s/random_distance_files/%s_%s.o' -e '%s/random_distance_files/%s_%s.e' -q normal "
					  "-J 'random_%s_%s' python3 CombDrug/module/network/parallelise_distance.py -dist '%s' -seed '%s'"
					  % (
					  log_path, dist_type, seed_text, log_path, dist_type, seed_text, dist_type, seed_text, dist_type,
					  seed_text))

	return True


for i in range(50):
	_ = run_network_modelling(run_for="similarity", alphas=[0.55], random=True, seed=i)


# ---------------------------------------------------------------------------#
# Pathway Analysis
# ---------------------------------------------------------------------------#

def run_pathway_analysis(run_for, pathway_database, random, seed):
	if random:
		seed_text = "%d" % int(seed)
		random_title = "random"
		random_suffix = "_seed_%d" % int(seed)
	else:
		seed_text = "None"
		random_title = "empiric"
		random_suffix = ""

	log_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/network/modelling/ppr/pathway"

	if run_for == "pathway":

		os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
				  "-o %s/%s/%s%s.o -e %s/%s/%s%s.e -q normal -J 'ptw_%s%s' python3 CombDrug/module/network/run_enrichment.py -run '%s' -drug 'None' "
				  "-pathway_tool 'enrichr' -pathway_db '%s' "
				  % (
				  log_path, random_title, pathway_db, random_suffix, log_path, random_title, pathway_db, random_suffix,
				  pathway_db, random_suffix, run_for, pathway_db))

	elif run_for == "refine_pathway":

		all_drugs = os.listdir(
			output_path + "network/modelling/PPR/intact/drugs/pathway_analysis/enrichr/analysis/Reactome/")
		for drug in all_drugs:
			os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
					  "-o %s/%s/%s%s_refined.o -e %s/%s/%s%s_refined.e -q normal -J 'refined_%s%s' "
					  "python3 CombDrug/module/network/run_enrichment.py -run '%s' -drug '%s' -pathway_tool enrichr -pathway_db '%s'"
					  % (log_path, random_title, pathway_db, random_suffix, log_path, random_title, pathway_db,
						 random_suffix, pathway_db, random_suffix, run_for, drug, pathway_db))

	return True


# "WikiPathway_2023_Human", "Reactome_2022", "KEGG_2021_Human"
"""
for pathway_db in ["WikiPathway_2023_Human"]:
	_ = run_pathway_analysis(run_for="refine_pathway", pathway_database=pathway_db, random=False, seed=None)
"""
