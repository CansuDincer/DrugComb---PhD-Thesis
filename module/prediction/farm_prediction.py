"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 11 May 2024
Last Update : 16 May 2024
Input : Random Forest

Conda environment ==> "env_rf"
#------------------------------------------------------------------------#
"""
# !/usr/bin/python
# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, sys, time, argparse

if "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/" not in list(sys.path):
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/")
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/")


# ---------------------------------------------------------------------------#
#                        D R U G  C O M B - Prediction
#                         PhD Project of Cansu Dincer
#                          Wellcome Sanger Institute
# ---------------------------------------------------------------------------#


def run(run_for, seed):
	log_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs"

	if run_for == "omics_preparation":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' -M10000 "
				  "-o '%s/prediction/preparation/omics.o' -e '%s/prediction/preparation/omics.e' -q normal "
				  "-J 'omics' python3 CombDrug/module/prediction/random_forest.py -run_for '%s' -stage 'combo'"
				  % (log_path, log_path, run_for))

	elif run_for == "drug_preparation":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>10000] rusage[mem=10000]' -M10000 "
				  "-o '%s/prediction/preparation/drugs.o' -e '%s/prediction/preparation/drugs.e' -q normal "
				  "-J 'drugs' python3 CombDrug/module/prediction/random_forest.py -run_for '%s' -stage 'combo'"
				  % (log_path, log_path, run_for))

	elif run_for == "perturbation_preparation":
		os.system("bsub -n 8 -G team215-grp -R'select[mem>300000] rusage[mem=300000]' -M300000 "
				  "-o '%s/prediction/preparation/perturbation.o' -e '%s/prediction/preparation/perturbation.e' -q basement "
				  "-J 'perturbation' python3 CombDrug/module/prediction/random_forest.py -run_for '%s' -stage 'combo'"
				  % (log_path, log_path, run_for))

	elif run_for == "RF":

		# ----------------------------------------------------------------------------------------------------------#
		# HYP1 : What are the selected hyperparameters?
		"""
		# GEX
		title0= "combo_gex"
		drop_text = " "
		drop_title = ""
		os.system("bsub -n 20 -G team215-grp -R'select[mem>240000] rusage[mem=240000]' -M240000 "
				  "-o '%s/prediction/random_forest/rf_%s%s.o' -e '%s/prediction/random_forest/rf_%s%s.e' -q normal "
				  "-J 'rf_%s%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -param -tuning 'random' -gex -file '%s' "
				  %(log_path, title0, drop_title, log_path, title0, drop_title, title0, drop_title, '10', title0))

		# GEX + FP 
		title1= "combo_gex_fp"
		drop_text = " "
		drop_title = ""
		os.system("bsub -n 20 -G team215-grp -R'select[mem>120000] rusage[mem=120000]' -M120000 "
				  "-o '%s/prediction/random_forest/rf_%s%s.o' -e '%s/prediction/random_forest/rf_%s%s.e' -q basement "
				  "-J 'rf_%s%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -param -tuning 'random' -gex -fp -fpt 'morgan' -file '%s' "
				  %(log_path, title1, drop_title, log_path, title1, drop_title, title1, drop_title, '10', title1))

		# GEX + MODULE 
		title2= "combo_gex_module"
		drop_text = " "
		drop_title = ""
		os.system("bsub -n 20 -G team215-grp -R'select[mem>120000] rusage[mem=120000]' -M120000 "
				  "-o '%s/prediction/random_forest/rf_%s%s.o' -e '%s/prediction/random_forest/rf_%s%s.e' -q basement "
				  "-J 'rf_%s%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -param -tuning 'random' -gex -modprob -file '%s' "
				  %(log_path, title2, drop_title, log_path, title2, drop_title, title2, drop_title, '10', title2))
		"""

		# ----------------------------------------------------------------------------------------------------------#
		# SHUFFLE X NEW ONLY IN TRAINING - RANDOMISED
		"""
		# GEX + FP 
		title1= "combo_gex_fp"
		drop_text = " "
		drop_title = ""
		os.system("bsub -n 20 -G team215-grp -R'select[mem>40000] rusage[mem=40000]' -M40000 "
				  "-o '%s/prediction/random_forest/shuf_rf_%s%s_%d.o' -e '%s/prediction/random_forest/shuf_rf_%s%s_%d.e' -q normal "
				  "-J 'shuf_rf_%s%s_%d' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -fp -fpt 'morgan' -shuffle -file '%s' -seed '%d'"
				  %(log_path, title1, drop_title, seed, log_path, title1, drop_title, seed, title1, drop_title, seed, '10', title1, seed))

		# GEX + MODULE 
		title2= "combo_gex_module"
		drop_text = " "
		drop_title = ""
		os.system("bsub -n 20 -G team215-grp -R'select[mem>110000] rusage[mem=110000]' -M110000 "
				  "-o '%s/prediction/random_forest/shuf_rf_%s%s_%d.o' -e '%s/prediction/random_forest/shuf_rf_%s%s_%d.e' -q normal "
				  "-J 'shuf_rf_%s%s_%d' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -modprob -shuffle -file '%s' -seed '%d'"
				  %(log_path, title2, drop_title, seed, log_path, title2, drop_title, seed, title2, drop_title, seed, '10', title2, seed))
		"""
		# ----------------------------------------------------------------------------------------------------------#
		# HYP2 : If generalisation improve with network approach?

		drop_title = "_drop"

		# GEX + MODULE

		title2 = "combo_gex_module"
		os.system("bsub -n 20 -G team215-grp -R'select[mem>95000] rusage[mem=95000]' -M95000 "
				  "-o '%s/prediction/random_forest/rf_%s%s_%s.o' -e '%s/prediction/random_forest/rf_%s%s_%s.e' -q basement "
				  "-J 'rf_%s%s_%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -modprob -file '%s' -drop_dc -seed '%d'"
				  % (
				  log_path, title2, drop_title, str(seed), log_path, title2, drop_title, str(seed), title2, drop_title,
				  str(seed), '10', title2, seed))
		"""
		title2 = "combo_gex_module"
		os.system("bsub -n 20 -G team215-grp -R'select[mem>110000] rusage[mem=110000]' -M110000 "
				  "-o '%s/prediction/random_forest/rf_comp_%s%s_%s.o' -e '%s/prediction/random_forest/rf_comp_%s%s_%s.e' -q normal "
				  "-J 'rf_comp_%s%s_%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -modprob -file '%s' -drop_dc -drop_dc_comp -seed '%d'"
				  %(log_path, title2, drop_title, str(seed), log_path, title2, drop_title, str(seed), title2, drop_title, str(seed), 
					'10', title2, seed))


		title2 = "combo_gex_module"
		os.system("bsub -n 20 -G team215-grp -R'select[mem>110000] rusage[mem=110000]' -M110000 "
				  "-o '%s/prediction/random_forest/rf_%s%s_frandom.o' -e '%s/prediction/random_forest/rf_%s%s_frandom.e' -q long "
				  "-J 'rf_%s%s_frandom' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -modprob -file '%s' -drop_dc -frandom "
				  %(log_path, title2, drop_title, log_path, title2, drop_title, title2, drop_title, '10', title2))
		"""

		# HYP2  GEX + PF
		"""
		title3 = "combo_gex_fp"
		os.system("bsub -n 20 -G team215-grp -R'select[mem>40000] rusage[mem=40000]' -M40000 "
				  "-o '%s/prediction/random_forest/rf_comp_%s%s_%s.o' -e '%s/prediction/random_forest/rf_comp_%s%s_%s.e' -q basement "
				  "-J 'rf_%s%s_%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -fp -fpt 'morgan' -file '%s' -drop_dc -seed '%d'"
				  %(log_path, title3, drop_title, str(seed), log_path, title3, drop_title, str(seed), title3, drop_title, str(seed), '10', title3, seed))
		"""
		"""
		title3 = "combo_gex_fp"
		os.system("bsub -n 20 -G team215-grp -R'select[mem>40000] rusage[mem=40000]' -M40000 "
				  "-o '%s/prediction/random_forest/rf_%s%s_%s.o' -e '%s/prediction/random_forest/rf_%s%s_%s.e' -q normal "
				  "-J 'rf_comp_%s%s_%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -fp -fpt 'morgan' -file '%s' -drop_dc -drop_dc_comp -seed '%d'"
				  %(log_path, title3, drop_title, str(seed), log_path, title3, drop_title, str(seed), title3, drop_title, str(seed), '10', title3, seed))

		os.system("bsub -n 20 -G team215-grp -R'select[mem>40000] rusage[mem=40000]' -M40000 "
				  "-o '%s/prediction/random_forest/rf_%s%s_frandom.o' -e '%s/prediction/random_forest/rf_%s%s_frandom.e' -q long "
				  "-J 'rf_%s%s_frandom' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -fp -fpt 'morgan' -file '%s' -drop_dc -frandom "
				  %(log_path, title3, drop_title, log_path, title3, drop_title, title3, drop_title, '10', title3))

		# HYP2  GEX + TARGET
		title4 = "combo_gex_target"

		os.system("bsub -n 20 -G team215-grp -R'select[mem>110000] rusage[mem=110000]' -M110000 "
				  "-o '%s/prediction/random_forest/rf_%s%s_%s.o' -e '%s/prediction/random_forest/rf_%s%s_%s.e' -q normal "
				  "-J 'rf_%s%s_%s' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -target -file '%s' -drop_dc -seed '%d'"
				  %(log_path, title4, drop_title, str(seed), log_path, title4, drop_title, str(seed), title4, drop_title, str(seed), '10', title4, seed))

		os.system("bsub -n 20 -G team215-grp -R'select[mem>110000] rusage[mem=110000]' -M110000 "
				  "-o '%s/prediction/random_forest/rf_%s%s_frandom.o' -e '%s/prediction/random_forest/rf_%s%s_frandom.e' -q long "
				  "-J 'rf_%s%s_frandom' python3 CombDrug/module/prediction/random_forest.py -run_for 'RF' "
				  "-stage 'combo' -kf '%s' -njob '20' -niter '100' -gex -target -file '%s' -drop_dc -frandom "
				  %(log_path, title4, drop_title, log_path, title4, drop_title, title4, drop_title, '10', title4))
		"""
	return True


for i in range(5):
	run(run_for="RF", seed=i)
