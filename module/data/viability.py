# ---------------------------------------------------------------------------#
#       	  C o m b D r u g - S C R E E N   V I A B I L I T Y			     #
# ---------------------------------------------------------------------------#

"""
# ---------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 18 March 2024
Last Update : 19 March 2024
Output: Viability Objects
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

from CombDrug.module.paths import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.drug import get_drug_ids, drug_id2name
from CombDrug.module.data.cancer_model import *
from CombDrug.module.data.reproducibility import get_all_viabilities, bad_replicate


def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-treatment", dest="TREATMENT")
	parser.add_argument("-status", dest="STATUS")
	parser.add_argument("-merge", dest="MERGE")
	parser.add_argument("-trdel", dest="TR_DEL")
	parser.add_argument("-brdel", dest="BR_DEL")
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                             Viability Object                               #
# ---------------------------------------------------------------------------#

class Viability:

	def __init__(self, treatment):
		"""
		:param treatment: combination/library
		:return:
		"""
		self.treatment = treatment
		# Without any replicate analysis
		self.pre_viability = None
		# Without biological replicate analysis
		self.mid_viability = None
		# After both replicate analysis
		self.post_viability = None

	def call_pre_viability(self):
		df = get_all_viabilities(treatment=self.treatment)
		df["plate_num"] = df["plate_num"].astype(str)
		self.pre_viability = df.copy()
		return self.pre_viability

	def call_mid_viability(self, type_of_merge, del_rep):
		if del_rep:
			title = "del_rep"
		else:
			title = "all_rep"

		output_folder = output_path + "data/replicates/curated_viability/"
		print(os.listdir(output_folder))
		if "trep_%s_%s_curated_normalised_viability.csv" % (type_of_merge, title) not in os.listdir(output_folder):
			pre_trep_viability = self.pre_viability
			if del_rep:
				bad_reps = bad_replicate(replicate_type="technical", treatment=self.treatment,
										 type_of_merge=None, answer_t=None)
				if bad_reps is not None:
					for subproject, bad_replicate_dict_list in bad_reps.items():
						for bad_replicate_dict in bad_replicate_dict_list:
							removed_indices = list(
								pre_trep_viability[(pre_trep_viability.subproject == subproject) &
												   (pre_trep_viability.exp_date == bad_replicate_dict["exp_date"]) &
												   (pre_trep_viability.plate_num == str(
													   bad_replicate_dict["barcode"]))].index)
							pre_trep_viability = pre_trep_viability.drop(index=removed_indices)

			df = None
			if type_of_merge == "mean":
				pre_trep_viability["plate_num"] = pre_trep_viability["plate_num"].astype(str)
				df = pandas.DataFrame(pre_trep_viability.groupby(
					["SIDM", "project", "subproject", "D1", "D2", "Do1", "Do2", "exp_date"])[
										  ["viability"]].mean()).reset_index()

			elif type_of_merge == "median":
				pre_trep_viability["plate_num"] = pre_trep_viability["plate_num"].astype(str)
				df = pandas.DataFrame(pre_trep_viability.groupby(
					["SIDM", "project", "subproject", "D1", "D2", "Do1", "Do2", "exp_date"])[
										  ["viability"]].median()).reset_index()
			if df is not None:
				self.mid_viability = df.copy()

				df.to_csv(output_folder + "trep_%s_%s_curated_normalised_viability.csv" % (type_of_merge, title),
						  index=False)
		else:
			df = pandas.read_csv(output_folder + "trep_%s_%s_curated_normalised_viability.csv" % (type_of_merge, title))
			print(df)
			self.mid_viability = df.copy()
		return self.mid_viability

	def call_post_viability(self, type_of_merge, del_rep, answer_t):
		if del_rep:
			title = "del_rep"
		else:
			title = "all_rep"

		output_folder = output_path + "data/replicates/curated_viability/"

		if "brep_%s_%s_curated_normalised_viability.csv" % (type_of_merge, title) not in os.listdir(output_folder):
			pre_brep_viability = self.mid_viability
			if del_rep:
				bad_reps = bad_replicate(replicate_type="biological", treatment=self.treatment,
										 type_of_merge=type_of_merge, answer_t=answer_t)
				if bad_reps is not None:
					for subproject, bad_replicate_dict_list in bad_reps.items():
						for bad_replicate_dict in bad_replicate_dict_list:
							removed_indices = list(pre_brep_viability[(pre_brep_viability.subproject == subproject) &
																	  (pre_brep_viability.exp_date ==
																	   bad_replicate_dict["exp_date"])].index)
							pre_brep_viability = pre_brep_viability.drop(index=removed_indices)

			if type_of_merge == "mean":
				df = pandas.DataFrame(pre_brep_viability.groupby(
					["SIDM", "project", "subproject", "D1", "D2", "Do1", "Do2"])[
										  ["viability"]].mean()).reset_index()

			elif type_of_merge == "median":
				df = pandas.DataFrame(pre_brep_viability.groupby(
					["SIDM", "project", "subproject", "D1", "D2", "Do1", "Do2"])[
										  ["viability"]].median()).reset_index()

			self.post_viability = df.copy()
			df.to_csv(output_folder + "brep_%s_%s_curated_normalised_viability.csv" % (type_of_merge, title),
					  index=False)
		else:
			df = pandas.read_csv(output_folder + "brep_%s_%s_curated_normalised_viability.csv" % (type_of_merge, title))
			self.post_viability = df.copy()
		return self.post_viability


def serialise_viability_object(treatment, status, type_of_merge, del_rep_t, del_rep_b):
	"""
	Serialisation of the Viability object
	:param treatment: combination or mono
	:param status: pre / mid / post
	:param type_of_merge: median/mean merging replicates after analysis
	:param del_rep: Boolean - Deletion of bad replicates or not
	:return:
	"""

	if del_rep_t:
		del_title_t = "del_rep"
	else:
		del_title_t = "all_rep"

	if del_rep_b:
		del_title_b = "del_rep"
	else:
		del_title_b = "all_rep"

	if status != "pre":
		if status == "mid":
			title = "_" + type_of_merge + "_" + del_title_t + "_merged"
		elif status == "post":
			title = "_" + type_of_merge + "_" + del_title_b + "_merged"
	else:
		title = ""

	viability = dict()
	obj = Viability(treatment=treatment)
	obj.call_pre_viability()
	if status == "mid":
		obj.call_mid_viability(type_of_merge=type_of_merge, del_rep=del_rep_t)
	elif status == "post":
		obj.call_mid_viability(type_of_merge=type_of_merge, del_rep=del_rep_t)
		obj.call_post_viability(type_of_merge=type_of_merge, del_rep=del_rep_b, answer_t=del_rep_t)

	viability[treatment] = obj

	pickle.dump(viability, open(output_path + "data/replicates/%s_%s%s_viability_object.p"
								% (treatment, status, title), "wb"))
	return True


def deserialise_viability_object(treatment, status, type_of_merge, del_rep_t, del_rep_b):
	"""
	De-serialisation of the Viability object
	:param treatment: combination or mono
	:param status: pre / mid / post
	:param type_of_merge: median/mean merging replicates after analysis
	:param del_rep: Boolean - Deletion of bad replicates or not
	:return:
	"""
	if del_rep_t:
		del_title_t = "del_rep"
	else:
		del_title_t = "all_rep"

	if del_rep_b:
		del_title_b = "del_rep"
	else:
		del_title_b = "all_rep"

	if status != "pre":
		if status == "mid":
			title = "_" + type_of_merge + "_" + del_title_t + "_merged"
		elif status == "post":
			title = "_" + type_of_merge + "_" + del_title_b + "_merged"
	else:
		title = ""

	viability_objs = pickle.load(open(output_path + "data/replicates/%s_%s%s_viability_object.p"
									  % (treatment, status, title), "rb"))
	return viability_objs


if __name__ == '__main__':

	args = take_input()
	if args["TR_DEL"] in ["False", False, "None", None]:
		args["TR_DEL"] = False
	elif args["TR_DEL"] in ["True", True]:
		args["TR_DEL"] = True

	if args["BR_DEL"] in ["False", False, "None", None]:
		args["BR_DEL"] = False
	elif args["BR_DEL"] in ["True", True]:
		args["BR_DEL"] = True

	print(args)
	_ = serialise_viability_object(treatment=args["TREATMENT"], status=args["STATUS"], type_of_merge=args["MERGE"],
								   del_rep_t=args["TR_DEL"], del_rep_b=args["BR_DEL"])