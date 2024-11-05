"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 2 May 2024
Last Update : 2 May 2024
Input : Data retrieval and curation

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


# ---------------------------------------------------------------------------#

def run(run_for, treatment, rep_type, tr_del, br_del, merge, status, combo_del, anchor, subproject):
	if run_for == "viability":
		for treatment in ["mono", "combination"]:
			os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
					  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/viability/%s_normalised_viability.o' "
					  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/viability/%s_normalised_viability.e' -q normal "
					  "-J '%s_via' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment '%s' -merge 'None' -trdel 'None' "
					  "-brdel 'None' -run '%s' -combo_del 'None'"
					  % (treatment, treatment, treatment, treatment, run_for))

	if run_for == "count":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>40000] rusage[mem=40000]' "
				  "-M40000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_count.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_count.e' -q basement "
				  "-J '%s_count' python3 CombDrug/module/data/reproducibility.py -rep_type '%s' -treatment '%s' -merge '%s' -trdel '%s' "
				  "-brdel '%s' -run '%s' -combo_del 'None'"
				  % (rep_type, rep_type, rep_type, rep_type, rep_type, rep_type, "combination", merge, tr_del, br_del,
					 run_for))

	if run_for == "regress":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_regress.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_regress.e' -q normal "
				  "-J '%s_regress' python3 CombDrug/module/data/reproducibility.py -rep_type '%s' -treatment '%s' -merge '%s' -trdel '%s' "
				  "-brdel '%s' -run '%s' -combo_del 'None'"
				  % (rep_type, rep_type, rep_type, rep_type, rep_type, rep_type, "combination", merge, tr_del, br_del,
					 run_for))

	if run_for == "statistics":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_stat.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_stat.e' -q normal "
				  "-J '%s_stat' python3 CombDrug/module/data/reproducibility.py -rep_type '%s' -treatment '%s' -merge 'None' -trdel 'None' "
				  "-brdel 'None' -run '%s' -combo_del 'None'"
				  % (rep_type, rep_type, rep_type, rep_type, rep_type, rep_type, "combination", run_for))

	if run_for == "bad_replicates":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_bad_rep.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_bad_rep.e' -q normal "
				  "-J '%s_badrep' python3 CombDrug/module/data/reproducibility.py -rep_type '%s' -treatment '%s' -merge 'None' -trdel 'None' "
				  "-brdel 'None' -run '%s' -combo_del 'None'"
				  % (rep_type, rep_type, rep_type, rep_type, rep_type, rep_type, "combination", run_for))

	if run_for == "worth":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_worth.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/%s_replicates/%s_rep_worth.e' -q normal "
				  "-J '%s_worth' python3 CombDrug/module/data/reproducibility.py -rep_type '%s' -treatment '%s' -merge 'None' -trdel 'None' -brdel 'None' -run '%s' -combo_del 'None'"
				  % (rep_type, rep_type, rep_type, rep_type, rep_type, rep_type, "combination", run_for))

	if run_for == "serialise":
		if tr_del:
			tr_del_text = "_delTR"
		else:
			tr_del_text = ""

		if br_del:
			br_del_text = "_delBR"
		else:
			br_del_text = ""

		if merge is not None:
			merge_text = "_%s_merge" % merge
		else:
			merge_text = ""

		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/objects/%s%s%s%s_%s.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/objects/%s%s%s%s_%s.e' -q normal "
				  "-J 'obj_%s%s%s%s' python3 CombDrug/module/data/viability.py -treatment '%s' -status '%s' -merge '%s' -trdel '%s' -brdel '%s'"
				  % (treatment, tr_del_text, br_del_text, merge_text, status, treatment, tr_del_text, br_del_text,
					 merge_text, status,
					 treatment, tr_del_text, br_del_text, merge_text, treatment, status, merge, tr_del, br_del))

	if run_for == "combi":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/all_combi.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/all_combi.e' -q normal "
				  "-J 'combi' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment 'None' -merge 'None' -trdel 'None' -brdel 'None' "
				  "-run '%s' -combo_del 'None'" % run_for)

	if run_for == "bad_fits":
		if anchor:
			anchor_text = "_anchor"
		else:
			anchor_text = ""

		if subproject:
			p_text = "_subproject"
		else:
			p_text = "_project"

		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/bad_fits%s%s.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/bad_fits%s%s.e' -q normal "
				  "-J 'combi_bad_fit%s%s' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment 'None' -merge '%s' -trdel 'None' -brdel 'None' "
				  "-run '%s' -combo_del '%s' -subproject '%s' -anchor '%s'"
				  % (
				  anchor_text, p_text, anchor_text, p_text, anchor_text, p_text, merge, run_for, combo_del, subproject,
				  anchor))

	if run_for == "select_fits":
		print("submitting")
		if anchor:
			anchor_text = "_anchor"
		else:
			anchor_text = ""

		if subproject:
			p_text = "_subproject"
		else:
			p_text = "_project"

		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/select_fits%s%s.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/select_fits%s%s.e' -q long "
				  "-J 'combi_select_fit%s%s' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment 'None' -merge '%s' -trdel 'None' -brdel 'None' "
				  "-run '%s' -combo_del '%s' -subproject '%s' -anchor '%s'"
				  % (
				  anchor_text, p_text, anchor_text, p_text, anchor_text, p_text, merge, run_for, combo_del, subproject,
				  anchor))

	if run_for == "regress_combi":
		if anchor:
			anchor_text = "_anchor"
		else:
			anchor_text = ""

		if subproject:
			p_text = "_subproject"
		else:
			p_text = "_project"

		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/regress_fits%s%s.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/regress_fits%s%s.e' -q normal "
				  "-J 'combi_regress_fit%s%s' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment 'None' -merge '%s' -trdel 'None' -brdel 'None' "
				  "-run '%s' -combo_del '%s' -subproject '%s' -anchor '%s'"
				  % (
				  anchor_text, p_text, anchor_text, p_text, anchor_text, p_text, merge, run_for, combo_del, subproject,
				  anchor))

	if run_for == "worth_combi":
		if anchor:
			anchor_text = "_anchor"
		else:
			anchor_text = ""

		if subproject:
			p_text = "_subproject"
		else:
			p_text = "_project"

		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' "
				  "-M12000 -o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/worth_combi%s%s.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/worth_combi%s%s.e' -q normal "
				  "-J 'combi_worth%s%s' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment 'None' -merge '%s' -trdel 'None' -brdel 'None' "
				  "-run '%s' -combo_del 'None' -subproject '%s' -anchor '%s'"
				  % (anchor_text, p_text, anchor_text, p_text, anchor_text, p_text, merge, run_for, subproject, anchor))

	if run_for == "combine":
		os.system("bsub -n 10 -G team215-grp -R'select[mem>12000] rusage[mem=12000]' -M12000 "
				  "-o '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/combine_%s.o' "
				  "-e '/lustre/scratch125/casm/team215mg/cd7/CombDrug/logs/reproducibility/combi/combine_%s.e' -q normal "
				  "-J 'combine_%s' python3 CombDrug/module/data/reproducibility.py -rep_type 'None' -treatment '%s' "
				  "-merge 'None' -trdel 'None' -brdel 'None' -run '%s' -combo_del 'None' -subproject 'None' -anchor 'None'"
				  % (treatment, treatment, treatment, treatment, run_for))

	return True

