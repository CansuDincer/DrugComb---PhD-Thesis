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

robust_biomarkers = collect_robust_biomarkers(tissue=None, estimate_lr="XMID", fdr_limit=10, merged=True,
											  msi_flag=False, random=False)[:10]
robust_biomarkers = robust_biomarkers[[i for i in robust_biomarkers.columns if i != "Unnamed: 0"]]
"""
robust_biomarkers["median_bm_combo"] = None
robust_biomarkers["mean_bm_combo"] = None
robust_biomarkers["std_bm_combo"] = None
robust_biomarkers["max_bm_combo"] = None
"""

log_path = ""

count, tot = 0, len(robust_biomarkers.index)
for ind, row in robust_biomarkers.iterrows():
	gene, feature, level, tissue, dc = row.Gene, row.Feature, row.Feature_level, row.tissue, row.DrugComb
	os.system("bsub -n 2 -G team215-grp -R'select[mem>1000] rusage[mem=1000]' -M1000 "
			  "-o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/biomarker/regression/ROBUST/parallel/%s_%s_%s_%s_%s.o' "
			  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/biomarker/regression/ROBUST/parallel/%s_%s_%s_%s_%s.e' -q normal "
			  "-J '%s_%s_%s_%s_%s' python3 CombDrug/module/model/get_combo_values_robustness.py -ind '%s' "
			  "-gene '%s' -feature '%s' -level '%s' -tissue '%s' -dc '%s'"
			  % (gene, feature, level, tissue, dc, gene, feature, level, tissue, dc,
				 gene, feature, level, tissue, dc, ind, gene, feature, level, tissue, dc))

	print((count * 100.0) / tot)
	count += 1

"""
	robust_biomarkers.loc[ind, "median_bm_combo"] = numpy.median(val)
	robust_biomarkers.loc[ind, "mean_bm_combo"] = numpy.mean(val)
	robust_biomarkers.loc[ind, "std_bm_combo"] = numpy.std(val)
	robust_biomarkers.loc[ind, "max_bm_combo"] = numpy.max(val)
	print((count * 100.0)/tot)
	count += 1

robust_biomarkers.to_csv(output_path + "biomarker/LR/annotated_combination_effect/combo_value_annotated_merged_LR_%s_FDR_%d.csv" % ("XMID", 10))
"""
