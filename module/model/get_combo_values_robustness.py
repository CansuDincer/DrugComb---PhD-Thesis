"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 16 August 2024
Last Update : 16 August 2024
Input : Drug Combination Final Analysis
Output : Drug Combination - Robust Biomarkers
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#

import os, sys

if "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/" not in list(sys.path):
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/")
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/")

import os, pandas, numpy, scipy, sys, matplotlib, networkx, re, argparse
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import r2_score
from statannot import add_stat_annotation
from datetime import datetime
from statannotations.Annotator import Annotator
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from CombDrug.module.path import output_path
from CombDrug.module.data.drug import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.omics import *
from CombDrug.module.data.responses import get_response_data
from CombDrug.module.model.LR_analysis import *


def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-ind", dest="IND")
	parser.add_argument("-gene", dest="GENE")
	parser.add_argument("-feature", dest="FEATURE")
	parser.add_argument("-level", dest="LEVEL")
	parser.add_argument("-tissue", dest="TISSUE")
	parser.add_argument("-dc", dest="DC")

	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                            Parallel analysis                               #
# ---------------------------------------------------------------------------#


def collect_combo_in_parallel(ind, gene, feature, tissue, drugcomb, level):
	if feature == "transcription":
		title = "log2_filtered_variance"
	elif feature == "proteomics":
		title = "averaged_variance"
	else:
		title = "5"

	dc_text = "_".join(drugcomb.split("/"))

	dim_df_combo = pandas.read_csv(output_path + "biomarker/LR/input_LR/%s/%s/%s/input_%s_%s_%s_%s_%s%s_%s%s%s.csv"
								   % ("combo", tissue, dc_text, "combo", feature, level, gene, dc_text,
									  "_" + tissue, title, "", ""), index_col=0)

	if feature in ["mutation", "amplification", "deletion", "hypermethylation", "gain", "loss", "clinicalsubtype"]:
		val = dim_df_combo[dim_df_combo[gene] == 1]["response"].values
	else:
		fet_df = dim_df_combo[[gene]]
		q4 = fet_df[gene].quantile([0.4, 0.6], interpolation="midpoint")
		selected_models = list(fet_df[fet_df[gene] > q4[0.6]].index)
		val = dim_df_combo[dim_df_combo.index.isin(selected_models)]["response"].values

	d = {"index": ind, "median_bm_combo": numpy.median(val), "mean_bm_combo": numpy.mean(val),
		 "std_bm_combo": numpy.std(val), "max_bm_combo": numpy.max(val)}

	pickle.dump(d, open(
		"/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/biomarker/LR/annotated_combination_effect/parallel/combo_%s_%s_%s_%s_%s.p"
		% (feature, level, gene, dc_text, tissue, title), "wb"))

	return True


args = take_input()
collect_combo_in_parallel(ind=args["IND"], gene=args["GENE"], feature=args["FEATURE"], tissue=args["TISSUE"],
						  drugcomb=args["DC"], level=args["LEVEL"])