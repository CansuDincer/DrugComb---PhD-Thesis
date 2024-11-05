"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 21 March 2024
Last Update : 9 May 2024
Input : Biomarker Analysis

-- Paralel analysis

No Conda environment ==> "base"
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

from CombDrug.module.path import *
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.responses import *
from CombDrug.module.data.omics import *


# ---------------------------------------------------------------------------#
#                D R U G  C O M B - Reproducibility Analysis
#                         PhD Project of Cansu Dincer
#                          Wellcome Sanger Institute
# ---------------------------------------------------------------------------#


def run(run_for):
	log_path = "/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs"

	if run_for == "response":
		for tissue in get_tissue_types():
			if pandas.isna(tissue) is False:
				tissue_text = "_".join(tissue.split(" "))
				for stage in ["mono", "combo", "delta"]:
					os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
							  "-o '%s/biomarker/preparation/response/response_%s_%s.o' "
							  "-e '%s/biomarker/preparation/response/response_%s_%s.e' -q normal "
							  "-J 'response_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
							  "-stage '%s' -run 'response'"
							  % (log_path, tissue_text, stage, log_path, tissue_text, stage, tissue_text, stage, tissue,
								 stage))

		for tissue in ["panliquid", "pansolid"]:
			for stage in ["mono", "combo", "delta"]:
				os.system("bsub -G team215-grp -R'select[mem>2000] rusage[mem=2000]' -M2000 "
						  "-o '%s/biomarker/preparation/response/response_%s_%s.o' "
						  "-e '%s/biomarker/preparation/response/response_%s_%s.e' -q normal "
						  "-J 'response_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
						  "-stage '%s' -run 'response'"
						  % (log_path, tissue, stage, log_path, tissue, stage, tissue, stage, tissue, stage))

	if run_for == "covariate":

		for tissue in get_tissue_types():
			if pandas.isna(tissue) is False:
				tissue_text = "_".join(tissue.split(" "))
				for stage in ["mono", "combo", "delta"]:
					os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
							  "-o '%s/biomarker/preparation/covariate/covariate_%s_%s.o' "
							  "-e '%s/biomarker/preparation/covariate/covariate_%s_%s.e' -q normal "
							  "-J 'covariate_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
							  "-stage '%s' -run 'covariate'"
							  % (log_path, tissue_text, stage, log_path, tissue_text, stage, tissue_text, stage, tissue,
								 stage))

		for tissue in ["panliquid", "pansolid"]:
			for stage in ["combo", "delta"]:
				os.system("bsub -n 6 -G team215-grp -R'select[mem>512000] rusage[mem=512000]' -M512000 "
						  "-o '%s/biomarker/preparation/covariate/covariate_%s_%s.o' "
						  "-e '%s/biomarker/preparation/covariate/covariate_%s_%s.e' -q normal "
						  "-J 'covariate_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' -stage '%s' -run 'covariate'"
						  % (log_path, tissue, stage, log_path, tissue, stage, tissue, stage, tissue, stage))

	if run_for == "feature":

		feature_dictionary = {"mutation": ["genes_driver_mut", "mutations_driver"],
							  "transcription": ["genes_cancer"], "proteomics": ["genes_cancer"],
							  "amplification": ["genes_cancer"], "deletion": ["genes_cancer"],
							  "hypermethylation": ["genes_cancer"],
							  "loss": ["genes_cancer"], "gain": ["genes_cancer"]}

		breast_special = {"clinicalsubtype": ["LumA", "Basal", 'Her2', 'LumA', 'LumB']}
		colo_special = {
			"clinicalsubtype": ['CRISA', 'CRISB', 'CRISC', 'CRISD', 'CRISE', 'CMS1', 'CMS2', 'CMS3', 'CMS4']}
		msi_special = {"msi": ["genes_cancer"]}

		for tissue in get_tissue_types():
			if pandas.isna(tissue) is False:
				tissue_text = "_".join(tissue.split(" "))
				for feature, level_list in feature_dictionary.items():
					for level in level_list:
						print(tissue, feature, level)

						os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
								  "-o '%s/biomarker/preparation/feature/%s_%s_%s.o' "
								  "-e '%s/biomarker/preparation/feature/%s_%s_%s.e' -q normal "
								  "-J 'feature_%s_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
								  "-feature '%s' -level '%s' -run 'feature'"
								  % (log_path, feature, level, tissue_text,
									 log_path, feature, level, tissue_text,
									 tissue_text, level, feature, tissue, feature, level))

				if tissue == "Breast":
					for level in breast_special["clinicalsubtype"]:
						os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
								  "-o '%s/biomarker/preparation/feature/%s_%s_%s.o' "
								  "-e '%s/biomarker/preparation/feature/%s_%s_%s.e' -q normal "
								  "-J 'feature_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
								  "-feature '%s' -level '%s' -run 'feature'"
								  % (log_path, "clinicalsubtype", level, tissue_text,
									 log_path, "clinicalsubtype", level, tissue_text,
									 tissue_text, level, tissue, "clinicalsubtype", level))

				if tissue == "Large Intestine":
					for level in colo_special["clinicalsubtype"]:
						os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
								  "-o '%s/biomarker/preparation/feature/%s_%s_%s.o' "
								  "-e '%s/biomarker/preparation/feature/%s_%s_%s.e' -q normal "
								  "-J 'feature_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
								  "-feature '%s' -level '%s' -run 'feature'"
								  % (log_path, "clinicalsubtype", level, tissue_text,
									 log_path, "clinicalsubtype", level, tissue_text,
									 tissue_text, level, tissue, "clinicalsubtype", level))

				if tissue in ["Large Intestine", "Stomach", "Endometrium", "Ovary"]:
					os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
							  "-o '%s/biomarker/preparation/feature/%s_%s_%s.o' "
							  "-e '%s/biomarker/preparation/feature/%s_%s_%s.e' -q normal "
							  "-J 'feature_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
							  "-feature '%s' -level '%s' -run 'feature'"
							  % (log_path, "msi", "genes_cancer", tissue_text,
								 log_path, "msi", "genes_cancer", tissue_text,
								 tissue_text, "msi", tissue, "msi", "genes_cancer"))

		for tissue in ["panliquid", "pansolid"]:
			for feature, level_list in feature_dictionary.items():
				for level in level_list:
					os.system("bsub -n 4 -G team215-grp -R'select[mem>2000] rusage[mem=2000]' -M2000 "
							  "-o '%s/biomarker/preparation/feature/%s_%s_%s.o' "
							  "-e '%s/biomarker/preparation/feature/%s_%s_%s.e' -q long "
							  "-J 'feature_%s_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
							  "-feature '%s' -level '%s' -run 'feature'"
							  % (log_path, feature, level, tissue, log_path, feature, level, tissue,
								 tissue, feature, tissue, feature, level))

	if run_for == "covariate_testing":

		for tissue in ["Central Nervous System"]:  # get_tissue_types():
			if pandas.isna(tissue) is False:
				tissue_text = "_".join(tissue.split(" "))

				os.system("bsub -n 8 -G team215-grp -R'select[mem>8000] rusage[mem=8000]' -M8000 "
						  "-o '%s/biomarker/covariates/%s.o' -e '%s/biomarker/covariates/%s.e' "
						  "-q basement -J 'cov_test_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
						  "-run 'covariate_testing'"
						  % (log_path, tissue_text, log_path, tissue_text, tissue_text, tissue))
		"""
		for tissue in ["Large Intestine", "Stomach", "Ovary", "Endometrium"]:
			tissue_text = "_".join(tissue.split(" "))

			os.system("bsub -n 8 -G team215-grp -R'select[mem>5000] rusage[mem=5000]' -M5000 "
					  "-o '%s/biomarker/covariates/%s_msi.o' -e '%s/biomarker/covariates/%s_msi.e' "
					  "-q basement -J 'cov_test_%s_msi' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
					  "-msi_cov -run 'covariate_testing'" 
					  %(log_path, tissue_text, log_path, tissue_text, tissue_text, tissue))

		for tissue in ["panliquid", "pansolid"]:

			os.system("bsub -n 12 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' -M12000 "
					  "-o '%s/biomarker/covariates/%s.o' -e '%s/biomarker/covariates/%s.e' "
					  "-q basement -J 'cov_test_%s' python3 CombDrug/module/model/prepare_LR.py -tissue '%s' "
					  "-run 'covariate_testing'" 
					  %(log_path, tissue, log_path, tissue, tissue, tissue))
		"""
	if run_for == "LR":

		feature_dictionary = {
			"mutation": {"levels": {"genes_driver_mut": {"selection_criteria": [5]},
									"mutations_driver": {"selection_criteria": [5]}}},
			"transcription": {"levels": {"genes_cancer": {"selection_criteria": ["variance"]}}},
			"proteomics": {"levels": {"genes_cancer": {"selection_criteria": ["variance"]}}},
			"amplification": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
			"deletion": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
			"hypermethylation": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
			"msi": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
			"loss": {"levels": {"genes_cancer": {"selection_criteria": [5]}}},
			"gain": {"levels": {"genes_cancer": {"selection_criteria": [5]}}}}

		estimate_list = ["XMID"]
		stage_list = ["mono", "combo", "delta"]
		msi_cov = False

		if msi_cov:
			msi_title = "_msi"
			msi = "-msi_cov "
		else:
			msi_title = ""
			msi = ""

		tissues = get_tissue_types() + ["pansolid", "panliquid"]

		for tissue in tissues:
			if tissue is None:
				tissue_name = "pancancer"
				tissue_text = tissue_name
			else:
				tissue_name = tissue
				tissue_text = "_".join(tissue_name.split())

			if feature_dictionary is not None:
				for feature, levels in feature_dictionary.items():
					for s in stage_list:
						for e in estimate_list:
							for l in levels["levels"].keys():
								for criteria in levels["levels"][l]["selection_criteria"]:
									if os.path.isdir("_".join(tissue_name.split())) == False:
										try:
											if not os.path.exists(os.path.dirname("%s/biomarker/regression/%s/" % (
											log_path, "_".join(tissue_name.split())))):
												os.system("mkdir %s/biomarker/regression/%s/" % (
												log_path, "_".join(tissue_name.split())))
										except OSError:
											continue

									os.system(
										"bsub -n 6 -G team215-grp -R'select[mem>20000] rusage[mem=20000]' -M20000 "
										"-o '%s/biomarker/regression/%s/%s_%s_%s_%s_%s%s_%s.o' "
										"-e '%s/biomarker/regression/%s/%s_%s_%s_%s_%s%s_%s.e' "
										"-q long -J 'lr_%s_%s_%s_%s_%s%s' "
										"python3 CombDrug/module/model/LR.py -feature '%s' -feature_level '%s' "
										"-selection_criteria '%s' -stage '%s' -estimate_lr '%s' -tissue '%s' "
										"-min_cl '15' %s-fdr '10'"

										% (
										log_path, tissue_text, tissue_text, s, e, feature, l, msi_title, str(criteria),
										log_path, tissue_text, tissue_text, s, e, feature, l, msi_title, str(criteria),
										tissue_text, s, e, l, feature, msi_title,
										feature, l, str(criteria), s, e, tissue, msi))

			if tissue == "Breast":
				e, criteria = "XMID", 5
				for s in ["mono", "combo", "delta"]:
					for level in ["LumA", 'Basal', 'Her2', 'LumA', 'LumB']:
						os.system("bsub -n 2 -G team215-grp -R'select[mem>6000] rusage[mem=6000]' -M6000 "
								  "-o '%s/biomarker/regression/%s/%s_%s_%s_%s_%s%s_%s.o' "
								  "-e '%s/biomarker/regression/%s/%s_%s_%s_%s_%s%s_%s.e' "
								  "-q normal -J 'lr_%s_%s_%s_%s_%s%s' "
								  "python3 CombDrug/module/model/LR.py -feature '%s' -feature_level '%s' "
								  "-selection_criteria '%s' -stage '%s' -estimate_lr '%s' -tissue '%s' "
								  "-min_cl '15' %s-fdr '10'"
								  % (log_path, tissue_text, tissue_text, s, e, "clinicalsubtype", level, msi_title,
									 str(criteria),
									 log_path, tissue_text, tissue_text, s, e, "clinicalsubtype", level, msi_title,
									 str(criteria),
									 tissue_text, s, e, level, "clinicalsubtype", msi_title,
									 "clinicalsubtype", level, str(criteria), s, e, tissue, msi))

			elif tissue == "Large Intestine":
				e, criteria = "XMID", 5
				for s in ["mono"]:  # , "combo", "delta"]:
					for level in [
						"CRISB"]:  # ['CRISA', 'CRISB', 'CRISC', 'CRISD', 'CRISE', 'CMS1', 'CMS2', 'CMS3','CMS4']:

						os.system("bsub -n 2 -G team215-grp -R'select[mem>6000] rusage[mem=6000]' -M6000 "
								  "-o '%s/biomarker/regression/%s/%s_%s_%s_%s_%s%s_%s.o' "
								  "-e '%s/biomarker/regression/%s/%s_%s_%s_%s_%s%s_%s.e' "
								  "-q normal -J 'lr_%s_%s_%s_%s_%s%s' "
								  "python3 CombDrug/module/model/LR.py -feature '%s' -feature_level '%s' "
								  "-selection_criteria '%s' -stage '%s' -estimate_lr '%s' -tissue '%s' "
								  "-min_cl '15' %s-fdr '10'"
								  % (log_path, tissue_text, tissue_text, s, e, "clinicalsubtype", level, msi_title,
									 str(criteria),
									 log_path, tissue_text, tissue_text, s, e, "clinicalsubtype", level, msi_title,
									 str(criteria),
									 tissue_text, s, e, level, "clinicalsubtype", msi_title,
									 "clinicalsubtype", level, str(criteria), s, e, tissue, msi))

	if run_for == "collection":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>50000] rusage[mem=50000]' -M50000 "
				  "-o %s/biomarker/regression/ALL/all_biomarkers.o -e %s/biomarker/regression/ALL/all_biomarkers.e "
				  "-q normal -J 'bm_collection' python3 CombDrug/module/model/LR_analysis.py -run_for '%s'"
				  % (log_path, log_path, run_for))

	if run_for == "robust":
		tissue_list = [i for i in get_tissue_types() if i is not None or pandas.isna(i) is False] + ["pansolid",
																									 "panliquid"]
		no_feature_project_list = ["Testis", "Vulva", "Uterus", "Small Intestine", "Prostate", "Biliary Tract",
								   "Adrenal Gland", "Thyroid", "Myeloma", "Endometrium", "Cervix", "Liver",
								   "Other liquid tumours", "Kidney"]
		tissue_list = [t for t in tissue_list if t not in no_feature_project_list]
		for tissue in tissue_list:
			tissue_text = "_".join(tissue.split())
			os.system("bsub -n 4 -G team215-grp -R'select[mem>40000] rusage[mem=40000]' -M40000 "
					  "-o %s/biomarker/regression/ROBUST/%s_biomarkers.o -e %s/biomarker/regression/ROBUST/%s_biomarkers.e "
					  "-q basement -J 'robust_%s' python3 CombDrug/module/model/LR_analysis.py -run_for '%s' -tissue '%s'"
					  % (log_path, tissue_text, log_path, tissue_text, tissue_text, run_for, tissue))

	return True


_ = run(run_for="robust")

