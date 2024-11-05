"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 21 April 2024
Last Update : 21 April 2024
Input : Networks and biomarkers
Output : Module- and Distance-related statistics
#------------------------------------------------------------------------#
"""

import sys

if "/lustre/scratch125/casm/team215mg/cd7/CombDrug/" not in list(sys.path):
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/")
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/CombDrug/")

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import argparse
from scipy.stats import hypergeom

"""
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
"""

from drafts.pre_modelling_cells import *
from CombDrug.module.data.dataset_info import *
from CombDrug.module.data.omics import test_Z
from CombDrug.module.network.interactome import get_norm_interactome, get_norm_random_interactome
from drafts.pcsf_statistics import read_network_file, get_reconstructed_models, \
	get_selected_parameter_set

flierprops = dict(marker=".", markerfacecolor="darkgrey", markersize=1.7,
				  markeredgecolor="none")
medianprops = dict(linestyle="-", linewidth=1, color="red")
boxprops = dict(facecolor="white", edgecolor="darkgrey")
whiskerprops = dict(color="darkgrey")


# ---------------------------------------------------------------------------#
# INPUT

def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-drug", dest="DRUG", required=False)
	parser.add_argument("-a", dest="ALPHA", required=False)
	parser.add_argument("-run", dest="RUN_FOR", required=True)
	parser.add_argument("-seed", dest="SEED", required=False)
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict


# ---------------------------------------------------------------------------#
#                                  Functions                                 #
# ---------------------------------------------------------------------------#


# ---------------------------------------------------------------------------#
# GENERATING MODULES WITH PPR


def run_ppr_drug(graph, model_id, interactome, graph_type, interactome_name, drug_name, stage, alpha, random_tag, seed,
				 weight):
	"""
	Running Random walk
	:param drug_name: Drug Name
	:param graph: graph object
	:param model_id: Sanger cell model id
	:param interactome: interactome object
	:param graph_type: interactome/subnetwork
	:param interactome_name: Name of interactome
	:param stage: mono or combo
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:param weight: usage of weight or not
	:return: Probability distribution of nodes
	"""

	if weight:
		weight_text = "weighted"
		weight_opt = "Weight"
	else:
		weight_text = "unweighted"
		weight_opt = None

	alpha_text = "_" + str(int(alpha * 100))

	if stage == "combo":
		drug_text = drug_name.split("/")[0] + "_" + drug_name.split("/")[1]
		targets = list()
		both_targeted = True
		for drug in drug_name.split("/"):
			drug_target = Drug(drug).targets
			if drug_target is not None:
				for t in drug_target:
					if t not in targets:
						targets.append(t)
			else:
				both_targeted = False
		if both_targeted is False:
			targets = None
	else:
		drug_text = drug_name
		targets = Drug(drug_name).targets

	if random_tag:
		random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
		random_text = "_random_seed_%d" % seed
		random_col = "random"

	else:
		random_text = "_empiric"
		random_col = "empiric"

	if graph_type == "interactome":

		model_text = ""
		if random_tag:
			output_folder_initial = random_output_path + "network/modelling/PPR/%s/drugs/%s/modules/" % (
			interactome_name, random_col)
		else:
			output_folder_initial = output_path + "network/modelling/PPR/%s/drugs/%s/" % (interactome_name, random_col)

		output_folder = output_folder_initial + "%s/interactome/" % drug_text

		try:
			os.mkdir(output_folder_initial + "%s/" % drug_text)
			os.mkdir(output_folder)

		except FileExistsError:
			pass

		model_text = ""

	elif graph_type == "subnetwork":

		if random_tag:
			output_folder = random_output_path + "network/modelling/PPR/%s/drugs/%s/modules/%s/%s/" % (
			interactome_name, random_col, drug_text, model_id)
		else:

			output_folder = output_path + "network/modelling/PPR/%s/drugs/%s/%s/%s/" \
							% (interactome_name, random_col, drug_text, model_id)
		try:
			os.mkdir(output_path + "network/modelling/PPR/%s/drugs/%s/%s/" % (interactome_name, random_col, drug_text))
			os.mkdir(output_folder)
		except FileExistsError:
			pass

		model_text = "_%s" % model_id

	try:
		if not os.path.exists(output_folder + "probability/"):
			os.mkdir("%s" % output_folder + "probability/")
			os.mkdir("%s" % output_folder + "network/")

	except FileExistsError:
		pass

	"""
	if random_tag:
		try:
			if not os.path.exists(output_folder + "probability/%s" % drug_text):
				os.mkdir("%s" % output_folder + "probability/%s/" % drug_text)
				os.mkdir("%s" % output_folder + "network/%s/" % drug_text)
			probability_text, network_text = "probability/%s/" % drug_text, "network/%s/" % drug_text

		except FileExistsError:
			pass
	else:
	"""
	probability_text, network_text = "probability/", "network/"

	df_true = False
	if targets is not None:
		if "ppr_probability_%s%s%s%s_%s.csv" % (drug_text, model_text, alpha_text, random_text, weight_text) not in \
				os.listdir(output_folder + probability_text):

			# Check if the targets are in the graph

			if graph_type == "subnetwork":
				selected_targets = list(set(targets).intersection(set(graph.nodes)))
				if weight:
					for edge in graph.edges:
						graph[edge[0]][edge[1]][weight_opt] = interactome.get_edge_data(edge[0], edge[1])[weight_opt]
				G = graph

			elif graph_type == "interactome":
				selected_targets = list(set(targets).intersection(set(interactome.nodes)))
				G = interactome

			if len(selected_targets) > 0:
				df_true = True

				# PPR
				print("PPR is working...")
				ppr_probability = networkx.pagerank(G, weight=weight_opt, max_iter=1000, alpha=alpha,
													personalization={target: 1 for target in selected_targets})

				ppr_df = pandas.DataFrame.from_dict({i: [l] for i, l in ppr_probability.items()}).T
				ppr_df.columns = ["ppr_probability"]
				ppr_df["targets"] = ppr_df.apply(lambda x: True if x.name in selected_targets else False, axis=1)
				ppr_df.to_csv(output_folder + probability_text + "ppr_probability_%s%s%s%s_%s.csv"
							  % (drug_text, model_text, alpha_text, random_text, weight_text), index=True)

		else:
			try:
				ppr_df = pandas.read_csv(output_folder + probability_text + "ppr_probability_%s%s%s%s_%s.csv"
										 % (drug_text, model_text, alpha_text, random_text, weight_text), index_col=0)
				df_true = True
			except FileNotFoundError:
				df_true = False

		if df_true:
			if "ppr_node_attributes_%s%s%s%s_%s.csv" % (
			drug_text, model_text, alpha_text, random_text, weight_text) not in os.listdir(
					output_folder + network_text):
				# 1 sd away 0.68%
				threshold = test_Z(values=ppr_df[["ppr_probability"]], pvalue=0.32, selected_side="right")
				# threshold = ppr_df.ppr_probability.quantile([0.95]).loc[0.95]
				filtered_ppr_df = ppr_df[ppr_df.ppr_probability > threshold]

				# filtered_ppr_df = ppr_df[ppr_df.ppr_probability > ppr_df.ppr_probability.quantile(q=0.95)]

				# Take drug modulated part of the interactome
				drug_network = G.subgraph(list(filtered_ppr_df.index))

				# Add node attributes
				attributes = {node: {"target": filtered_ppr_df.loc[node, "targets"],
									 "ppr_probability": filtered_ppr_df.loc[node, "ppr_probability"]}
							  for node in drug_network.nodes}

				networkx.set_node_attributes(drug_network, attributes)

				# Create node and edge attribute files
				drug_network_df = pandas.DataFrame(None, columns=["From", "To", "Weight"])
				for edge in drug_network.edges:
					if weight:
						x = pandas.DataFrame({"From": [edge[0]], "To": [edge[1]],
											  "Weight": drug_network.get_edge_data(edge[0], edge[1])["Weight"]})
					else:
						x = pandas.DataFrame({"From": [edge[0]], "To": [edge[1]]})

					drug_network_df = pandas.concat([drug_network_df, x], ignore_index=True)

				drug_network_df.to_csv(output_folder + network_text + "ppr_network_%s%s%s%s_%s.csv"
									   % (drug_text, model_text, alpha_text, random_text, weight_text), index=False)

				drug_network_node_df = pandas.DataFrame(None, columns=["node", "target", "ppr_probability"])
				for node in drug_network.nodes:
					x = pandas.DataFrame({"node": [node], "target": [drug_network.nodes[node]["target"]],
										  "ppr_probability": [drug_network.nodes[node]["ppr_probability"]]})
					drug_network_node_df = pandas.concat([drug_network_node_df, x], ignore_index=True)

				drug_network_node_df.to_csv(output_folder + network_text + "ppr_node_attributes_%s%s%s%s_%s.csv"
											% (drug_text, model_text, alpha_text, random_text, weight_text),
											index=False)

			else:
				drug_network_df = pandas.read_csv(output_folder + network_text + "ppr_network_%s%s%s%s_%s.csv"
												  % (drug_text, model_text, alpha_text, random_text, weight_text))
				drug_network_node_df = pandas.read_csv(
					output_folder + network_text + "ppr_node_attributes_%s%s%s%s_%s.csv"
					% (drug_text, model_text, alpha_text, random_text, weight_text))

			return drug_network_df, drug_network_node_df
		else:
			return None, None
	else:
		return None, None


def get_drug_network(graph, model_id, interactome, graph_type, interactome_name, drug_name, stage, alpha, random_tag,
					 seed, weight):
	"""
	Get networkx object of drug/drug combination network
	:param drug_name: Drug Name
	:param graph: graph object
	:param model_id: Sanger cell model id
	:param interactome: interactome object
	:param graph_type: interactome/subnetwork
	:param interactome_name: Name of interactome
	:param stage: mono or combo
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:param weight: usage of weight or not
	:return: Networkx object of drug/drug combination
	"""
	network_df, _ = run_ppr_drug(graph=graph, model_id=model_id, graph_type=graph_type,
								 interactome_name=interactome_name, interactome=interactome,
								 weight=weight, drug_name=drug_name, stage=stage,
								 random_tag=random_tag, seed=seed, alpha=alpha)
	if network_df is not None:
		g = networkx.Graph()
		for ind, row in network_df.iterrows():
			g.add_edge(row.From, row.To, Weight=row.Weight)

		return g
	else:
		return None


def get_drug_networks(graph_type, interactome_name, stage, alpha, weight, network_name,
					  value_type, filter_mi, random_tag, seed):
	"""
	Collect all drug/drug combination objects
	:param interactome_name: Name of interactome
	:param graph_type: interactome/subnetwork
	:param stage: mono or combo
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param weight: usage of weight or not
	:param filter_mi: Filter MI score in interactome default =0.4
	:param value_type: bayesian, logfc
    :param network_name: interactome/curated_interactome
    :param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:return: Dictinary of networkx objects
	"""

	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
		mu, b, w, d = get_selected_parameter_set()["mu"], \
					  get_selected_parameter_set()["b"], \
					  get_selected_parameter_set()["w"], \
					  get_selected_parameter_set()["d"]
	else:
		network_text, value_text = "", ""

	alpha_text = "_" + str(int(alpha * 100))

	output_folder = output_path + "network/modelling/PPR/%s/drugs/statistics/" % interactome_name

	if random_tag:
		random_text = "random_seed_%d" % seed
	else:
		random_text = "empiric"

	if "%s_%s%s%s_drug_networks_%s%s_%s.p" % (
	stage, graph_type, network_text, value_text, weight_text, alpha_text, random_text) not in os.listdir(output_folder):

		if stage == "combo":
			all_drugs = [i for i in all_screened_combinations(project_list=None, integrated=True) if
						 len(i.split(" | ")) == 1]
		else:
			all_drugs = list(make_drug_df()["drug_name"].unique())

		interactome = get_norm_interactome(interactome_name=interactome_name, weight=weight, filter_mi=filter_mi)

		if graph_type == "interactome":
			# Get all drug network objects
			count, total = 0, len(all_drugs)
			network_dict = dict()
			for drug in all_drugs:
				x = get_drug_network(graph=None, model_id=None, interactome=interactome, graph_type="interactome",
									 interactome_name=interactome_name, weight=weight,
									 drug_name=drug, stage=stage, random_tag=random_tag, seed=seed, alpha=alpha)
				if x is not None and len(x.nodes) != 0:
					network_dict[drug] = x
				count += 1
				print(count * 100.0 / total)

		if graph_type == "subnetwork":
			models = get_reconstructed_models(interactome_name=interactome_name, network_name=network_name,
											  value_type=value_type)
			network_dict = dict()
			count, total = 0, len(all_drugs) * len(models)
			for model_id in models:
				graph = read_network_file(w=w, d=d, b=b, mu=mu, model_id=model_id, interactome_name=interactome_name,
										  compounds=None, value_type=value_type, network_name=network_name,
										  random_tag=random_tag, seed=seed)

				# Get all drug network objects for the model
				model_network_dict = dict()
				for drug in all_drugs:
					x = get_drug_network(graph=graph, model_id=model_id, interactome=interactome,
										 graph_type="subnetwork",
										 interactome_name=interactome_name, drug_name=drug, stage=stage, alpha=alpha,
										 random_tag=random_tag, seed=seed, weight=weight)
					if x is not None and len(x.nodes) != 0:
						model_network_dict[drug] = x

					count += 1
					print(count * 100.0 / total)

				network_dict[model_id] = model_network_dict

		pickle.dump(network_dict, open(output_folder + "%s_%s%s%s_drug_networks_%s%s_%s.p"
									   % (stage, graph_type, network_text, value_text, weight_text, alpha_text,
										  random_text), "wb"))

	else:
		network_dict = pickle.load(open(output_folder + "%s_%s%s%s_drug_networks_%s%s_%s.p"
										% (stage, graph_type, network_text, value_text, weight_text, alpha_text,
										   random_text), "rb"))
	return network_dict


def compare_empiric_alpha(graph_type, interactome_name, weight, network_name, value_type, filter_mi,
						  random_tag, seed):
	"""
	Compare number of drug networks which can be created with PPR via different alpha values
	:param graph_type: interactome/subnetwork
	:param interactome_name: Name of interactome
	:param weight: usage of weight or not
	:param filter_mi: Filter MI score in interactome default =0.4
	:param value_type: bayesian, logfc
    :param network_name: interactome/curated_interactome
    :param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:return: Dictinary of networkx objects
	"""
	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
	else:
		network_text, value_text = "", ""

	output_folder = output_path + "network/modelling/PPR/%s/drugs/statistics/" % interactome_name

	all_drugs = [i for i in list(make_drug_df()["drug_name"].unique()) if len(i.split(" | ")) == 1]
	drug_numb = {"Total": len(all_drugs),
				 "Total with targets": len([i for i in all_drugs if Drug(i).targets is not None])}

	alphas = numpy.arange(0.05, 1.00, 0.05)

	for i in alphas:
		x = get_drug_networks(graph_type=graph_type, interactome_name=interactome_name, stage="mono", alpha=i,
							  weight=weight, network_name=network_name, value_type=value_type, filter_mi=filter_mi,
							  random_tag=random_tag, seed=seed)

		if graph_type == "interactome":
			drug_numb["alpha %.2f" % i] = len(x.keys())

		elif graph_type == "subnetwork":
			drug_numb["alpha %.2f" % i] = numpy.max([len(x[m].keys()) for m in x.keys()])

	fig, ax = plt.subplots(figsize=(5, 8))

	ax.bar(range(len(drug_numb)), list(drug_numb.values()), align='center', color="navy")
	if graph_type == "interactome":
		ax.set_ylabel("Number of drugs")
	elif graph_type == "subnetwork":
		ax.set_ylabel("Max of drugs across all reconstructed models")
	ax.set_title("Drug Networks with Personalised PageRank\nNumber of drug networks with alpha values")
	plt.xticks(range(len(drug_numb)), list(drug_numb.keys()), rotation=90)
	plt.tight_layout()
	plt.savefig(output_folder + "../figures/mono_%s%s%s_drug_network_drug_number_comparison_%s.png"
				% (graph_type, network_text, value_text, weight_text), dpi=300)
	plt.savefig(output_folder + "../figures/mono_%s%s%s_drug_network_drug_number_comparison_%s.jpg"
				% (graph_type, network_text, value_text, weight_text), dpi=300)
	plt.savefig(output_folder + "../figures/mono_%s%s%s_drug_network_drug_number_comparison_%s.pdf"
				% (graph_type, network_text, value_text, weight_text), dpi=300)
	plt.show()
	plt.close()

	return True


# ---------------------------------------------------------------------------#
# SELECTION OF ALPHA


def collect_neighbour_probabilities(graph_type, interactome_name, stage, weight, network_name,
									value_type, filter_mi, random_tag, seed):
	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	if random_tag:
		random_text = "_random_seed_%d" % seed
		random_col = "random"
	else:
		random_text = "_empiric"
		random_col = "empiric"

	fig_output_folder = output_path + "network/modelling/PPR/%s/drugs/statistics/" % interactome_name

	if "alpha_module_neig_probabilities.npy" not in os.listdir(
			output_path + "network/modelling/PPR/%s/drugs/statistics/" % interactome_name):
		# Probabilities of the path ending at a neighbouring node
		alpha_module_probabilities = dict()
		for alpha in numpy.arange(0.05, 1.00, 0.05):
			print(alpha)
			drug_network = get_drug_networks(graph_type=graph_type, interactome_name=interactome_name, stage=stage,
											 alpha=alpha, weight=weight, network_name=network_name,
											 value_type=value_type,
											 filter_mi=filter_mi, random_tag=False, seed=None)

			module_probabilities = dict()
			for drug_name, drug_g in drug_network.items():

				alpha_text = "_" + str(int(alpha * 100))

				if stage == "combo":
					drug_text = drug_name.split("/")[0] + "_" + drug_name.split("/")[1]
					targets = list()
					for drug in drug_name.split("/"):
						drug_target = Drug(drug).targets
						if drug_target is not None:
							for t in drug_target:
								if t not in targets:
									targets.append(t)
				else:
					drug_text = drug_name
					targets = Drug(drug_name).targets

				model_text = ""
				output_folder = output_path + "network/modelling/PPR/%s/drugs/%s/%s/interactome/probability/" \
								% (interactome_name, random_col, drug_text)

				ppr_df = pandas.read_csv(output_folder + "ppr_probability_%s%s%s%s_%s.csv"
										 % (drug_text, model_text, alpha_text, random_text, weight_text), index_col=0)

				filtered_ppr_df = ppr_df.loc[list(drug_g.nodes)]

				neighbour_probabilities = list()
				if len(drug_g.nodes) > len(targets):
					for t in targets:
						if t in drug_g.nodes:
							for n in drug_g.neighbors(t):
								neighbour_probabilities.append(filtered_ppr_df.loc[n, "ppr_probability"])
				else:
					neighbour_probabilities = [0]

				module_probabilities[drug_name] = neighbour_probabilities

			alpha_module_probabilities[alpha] = module_probabilities
		numpy.save(
			output_path + "network/modelling/PPR/%s/drugs/statistics/alpha_module_neig_probabilities.npy" % interactome_name,
			alpha_module_probabilities)
	else:
		alpha_module_probabilities = numpy.load(
			output_path + "network/modelling/PPR/%s/drugs/statistics/alpha_module_neig_probabilities.npy"
			% interactome_name, allow_pickle=True)

	# Plot all probabilities
	probability_list = list()
	for alpha in numpy.arange(0.05, 1.00, 0.05):
		for drug, prob_list in alpha_module_probabilities[alpha].items():
			for p in prob_list:
				row = {"alpha": alpha, "drug_name": drug, "probability": p}
				probability_list.append(pandas.DataFrame.from_dict(row, orient="index").T)

	probability_df = pandas.concat(probability_list)
	probability_df = probability_df.reset_index()[["alpha", "drug_name", "probability"]]

	drugs = dict()
	for a, a_df in probability_df.groupby(["alpha"]):
		drugs[a] = set(list(a_df.drug_name.unique()))

	drug_list = [v for k, v in drugs.items()]
	common_drugs = set.intersection(*map(set, drug_list))

	common_probability_df = probability_df[probability_df.drug_name.isin(common_drugs)]

	fig, axis = plt.subplots(1, 1, squeeze=False)

	plt.suptitle("Neighbour node probability distribution\nacross different Damping Factors")

	axis[0, 0].set_ylabel("Neighbour Node Probability")
	axis[0, 0].set_xlabel("Damping Factors")

	sns.lineplot(data=common_probability_df, x="alpha", y="probability", color="navy",
				 markers=False, dashes=False, legend=False, ax=axis[0, 0])
	plt.xticks([round(float(i), 2) for i in numpy.arange(0.05, 1.00, 0.05)], fontsize=8, rotation=45)
	plt.axvline(x=0.1, color='grey', linestyle=':')
	plt.axvline(x=0.2, color='grey', linestyle=':')
	plt.axvline(x=0.55, color='grey', linestyle=':')
	plt.axvline(x=0.35, color='grey', linestyle=':')
	plt.tight_layout()
	plt.savefig(fig_output_folder + "../figures/mono_%s_drug_network_neighbour_probability_comparison_%s.png"
				% (graph_type, weight_text), dpi=300)
	plt.savefig(fig_output_folder + "../figures/mono_%s_drug_network_neighbour_probability_comparison_%s.jpg"
				% (graph_type, weight_text), dpi=300)
	plt.savefig(fig_output_folder + "../figures/mono_%s_drug_network_neighbour_probability_comparison_%s.pdf"
				% (graph_type, weight_text), dpi=300)
	plt.show()
	plt.close()

	return True


# ---------------------------------------------------------------------------#
# MODULE DIAMETER


def get_network_diameter(network):
	# graph, network, weight
	"""
	ds : shortest distance
	For each of the Nd disease proteins, we determine the distance ds to the next-closest protein
	associated with the same disease.
	The average 〈ds〉 can be interpreted as the diameter of a disease on the interactome.
	:return:
	"""

	if networkx.is_connected(G=network):
		d = networkx.diameter(G=network)
	else:
		d = None
	return d


def get_single_drug_network_diameter(graph_type, drug_networks, interactome_name, stage, drug, weight, alpha,
									 network_name, value_type,
									 random_tag, seed):
	"""
	:param graph_type: interactome/subnetwork
	:param drug_networks: drug network dictionary from get_drug_networks
	:param interactome_name: Name of interactome
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param weight: usage of weight or not
	:param stage: mono or combo
	:param drug: Name of the drug or drug combination
	:param value_type: bayesian, logfc
    :param network_name: interactome/curated_interactome
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:return:
	"""

	if stage == "combo":
		drug_text = drug.split("/")[0] + "_" + drug.split("/")[1]
	else:
		drug_text = drug

	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	alpha_text = "_" + str(int(alpha * 100))

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
	else:
		network_text, value_text = "", ""

	if random_tag:
		random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
		random_text = "_random_seed_%d" % seed
		random_col = "random"

		output_folder = random_output_path + "network/modelling/PPR/%s/drugs/%s/diameter/" % (
		interactome_name, random_col)

	else:
		random_text = "_empiric"
		random_col = "/empiric/"

		output_folder = output_path + "network/modelling/PPR/%s/drugs/diameter%s" % (interactome_name, random_col)

	if "%s_%s_%s%s%s_%s_network_diameter_nx_%s%s%s.p" % (
	stage, interactome_name, graph_type, network_text, value_text, drug_text, weight_text, alpha_text,
	random_text) not in os.listdir(output_folder):

		if graph_type == "interactome":
			single_diameter_dict = dict()
			if drug in drug_networks.keys():
				network = drug_networks[drug]
				diameter = get_network_diameter(network=network)

				if diameter is not None:
					single_diameter_dict[drug] = diameter
					pickle.dump(single_diameter_dict, open(
						output_folder + "%s_%s_%s%s%s_%s_network_diameter_nx_%s%s%s.p"
						% (stage, interactome_name, graph_type, network_text, value_text,
						   drug_text, weight_text, alpha_text, random_text), "wb"))
				else:
					single_diameter_dict = None
		else:
			model_diameter_dict, single_diameter_dict = dict(), dict()
			for model_id, model_networks in drug_networks.items():
				if drug in model_networks.keys():
					network = model_networks[drug]
					diameter = get_network_diameter(network=network)
				else:
					diameter = None

				if diameter is not None:
					model_diameter_dict[model_id] = diameter

			if len(model_diameter_dict.keys()) != 0:
				single_diameter_dict[drug] = model_diameter_dict
				pickle.dump(single_diameter_dict, open(
					output_folder + "%s_%s_%s%s%s_%s_network_diameter_nx_%s%s%s.p"
					% (stage, interactome_name, graph_type, network_text, value_text,
					   drug_text, weight_text, alpha_text, random_text), "wb"))
			else:
				single_diameter_dict = None

	else:
		single_diameter_dict = pickle.load(open(
			output_folder + "%s_%s_%s%s%s_%s_network_diameter_nx_%s%s%s.p"
			% (stage, interactome_name, graph_type, network_text, value_text,
			   drug_text, weight_text, alpha_text, random_text), "rb"))

	return single_diameter_dict


def collect_drug_network_diameter(graph_type, interactome_name, stage, weight, alpha, network_name, value_type,
								  random_tag, seed):
	"""
	Collect all random drug/drug combination objects
	:param graph_type: interactome/subnetwork
	:param interactome_name: Name of interactome
	:param stage: mono or combo
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param weight: usage of weight or not
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:param value_type: bayesian, logfc
    :param network_name: interactome/curated_interactome
	:return: Dictinary of networkx objects for random
	"""

	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	alpha_text = "_" + str(int(alpha * 100))

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
	else:
		network_text, value_text = "", ""

	if random_tag:
		random_text = "_random_seed_%d" % seed
	else:
		random_text = "_empiric"

	output_folder = output_path + "network/modelling/PPR/%s/drugs/statistics/" % interactome_name

	if "%s_%s_%s%s%s_drug_network_diameter_nx_%s%s%s.p" % (
			stage, interactome_name, graph_type, network_text, value_text, weight_text, alpha_text,
			random_text) not in os.listdir(output_folder):

		drug_networks = get_drug_networks(graph_type=graph_type, interactome_name=interactome_name, stage=stage,
										  alpha=alpha, weight=True, network_name=network_name, value_type=value_type,
										  filter_mi=0.4, random_tag=random_tag, seed=seed)

		whole_diameter_dict = dict()

		if graph_type == "interactome":
			for drug in drug_networks.keys():
				d = get_single_drug_network_diameter(graph_type=graph_type, drug_networks=drug_networks,
													 interactome_name=interactome_name, stage=stage, drug=drug,
													 weight=weight, alpha=alpha, network_name=network_name,
													 value_type=value_type, random_tag=random_tag, seed=seed)
				if d is not None:
					# If there is more than one connected graph --> diameter cannot be computed --> None
					whole_diameter_dict[drug] = d[drug]

		elif graph_type == "subnetwork":
			all_drugs = [drug for model, drugs in drug_networks.items() for drug in drugs]
			all_drugs = list(set(all_drugs))
			for drug in all_drugs:
				d = get_single_drug_network_diameter(graph_type=graph_type, drug_networks=drug_networks,
													 interactome_name=interactome_name, stage=stage, drug=drug,
													 weight=weight, alpha=alpha, network_name=network_name,
													 value_type=value_type, random_tag=random_tag, seed=seed)
				whole_diameter_dict[drug] = d[drug]

		pickle.dump(whole_diameter_dict, open(output_folder + "%s_%s_%s%s%s_drug_network_diameter_nx_%s%s%s.p"
											  % (stage, interactome_name, graph_type, network_text, value_text,
												 weight_text, alpha_text, random_text), "wb"))
	else:
		whole_diameter_dict = pickle.load(open(output_folder + "%s_%s_%s%s%s_drug_network_diameter_nx_%s%s%s.p"
											   % (stage, interactome_name, graph_type, network_text, value_text,
												  weight_text, alpha_text, random_text), "rb"))
	return whole_diameter_dict


# ---------------------------------------------------------------------------#
# SHORTEST PATH


def get_graph_shortest_length(graph_type, model_id, interactome_name, weight, network_name, value_type, filter_mi,
							  random_tag, seed):
	if weight:
		weight_text = "weighted"
		weight_opt = "Weight"
	else:
		weight_text = "unweighted"
		weight_opt = None

	if random_tag:
		random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
		random_text = "_random_seed_%d" % seed
		random_col = "random"
		output_folder = random_output_path + "network/modelling/PPR/%s/drugs/%s/length/" % (
		interactome_name, random_col)

	else:
		random_text = "_empiric"
		random_col = "empiric/"
		output_folder = output_path + "network/modelling/PPR/%s/drugs/length/%s" % (interactome_name, random_col)

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
		model_text = "_%s" % model_id
		mu, b, w, d = get_selected_parameter_set()["mu"], \
					  get_selected_parameter_set()["b"], \
					  get_selected_parameter_set()["w"], \
					  get_selected_parameter_set()["d"]
	else:
		network_text, value_text, model_text = "", "", ""

	if "%s_%s%s%s%s_shortest_path_dict_%s%s.p" % (
	interactome_name, graph_type, model_text, network_text, value_text, weight_text, random_text) not in os.listdir(
			output_folder):

		if graph_type == "interactome":
			if random_tag is False:
				graph = get_norm_interactome(interactome_name=interactome_name, weight=weight, filter_mi=filter_mi)
			else:
				graph = get_norm_random_interactome(interactome_name=interactome_name, weight=weight,
													filter_mi=filter_mi, seed=seed)
		elif graph_type == "subnetwork":
			graph = read_network_file(w=w, d=d, b=b, mu=mu, model_id=model_id, interactome_name=interactome_name,
									  compounds=None, value_type=value_type, network_name=network_name,
									  random_tag=random_tag, seed=seed)

		length = dict(networkx.shortest_path_length(graph, weight=weight_opt))

		pickle.dump(length, open(output_folder + "%s_%s%s%s%s_shortest_path_dict_%s%s.p"
								 % (interactome_name, graph_type, model_text, network_text,
									value_text, weight_text, random_text), "wb"))
	else:
		length = pickle.load(open(output_folder + "%s_%s%s%s%s_shortest_path_dict_%s%s.p"
								  % (interactome_name, graph_type, model_text, network_text,
									 value_text, weight_text, random_text), "rb"))

	return length


# ---------------------------------------------------------------------------#
# MODULE SIMILARITY


def get_network_overlaps(node_set1, node_set2, all_nodes, overlap_method):
	"""
	:param node_set1: Nodes of network1
	:param node_set2: Nodes of network2
	:param all_nodes: Nodes from interactome
	:param overlap_method: hypergeometric or jaccard
	"""

	# Get numbers of nodes
	n1, n2 = len(node_set1), len(node_set2)
	n_inter, n_all = len(list(set(node_set1).intersection(set(node_set2)))), len(all_nodes)

	if overlap_method == "hypergeometric":
		x = sum(hypergeom.pmf(range(n_inter, min(n1, n2) + 1), n_all, n2, n1))

	elif overlap_method == "jaccard":
		if (len(set(node_set1).union(set(node_set2)))) != 0:
			x = (len(set(node_set1).intersection(set(node_set2)))) / (len(set(node_set1).union(set(node_set2))))
		else:
			x = None
	elif overlap_method == "coefficient":
		if (len(set(node_set1).union(set(node_set2)))) != 0:
			x = (len(set(node_set1).intersection(set(node_set2)))) / min(len(node_set1), len(node_set2))
	return x


def get_similarity(graph, length, nodes1, nodes2, weight):
	"""
	ds : shortest distance
	For each of the Nd disease proteins, we determine the distance ds to the next-closest protein
	associated with the same disease.
	The average 〈ds〉 can be interpreted as the diameter of a disease on the interactome.
	:return:
	"""

	if weight:
		weight_text = "Weight"
	else:
		weight_text = None

	"""
	ds = list()
	for n1 in nodes1:
		if n1 in graph.nodes():
			node_ds = list()
			for n2 in nodes2:
				if n2 in graph.nodes():
					if n1 == n2:
						ds.append(0)
					else:
						if networkx.has_path(graph, source=n1, target=n2):
							x = networkx.shortest_path_length(graph, source=n1, target=n2, weight=weight_text)
							node_ds.append(x)
			if node_ds:
				ds.extend(node_ds)
	"""

	ds = list()
	for n1 in nodes1:
		if n1 in graph.nodes():
			node_ds = list()
			for n2 in nodes2:
				if n2 in graph.nodes():
					if n1 != n2:
						node_ds.append(length[n1][n2])
			if node_ds:
				ds.extend(node_ds)

	"""
	ds = list()
	for n1 in network1.nodes:
		node_ds = list()
		for n2 in network2.nodes:
			if n1 == n2:
				ds.append(0)
			else:
				if networkx.has_path(interactome, source=n1, target=n2):
					x = networkx.shortest_path_length(interactome, source=n1, target=n2, weight=weight_text)
					node_ds.append(x)
		if node_ds:
			#d = min(node_ds)
			#ds.append(d)
			ds.extend(node_ds)
	"""

	if ds:
		average_ds = numpy.mean(ds)
		return average_ds
	else:
		return None


def get_similarity_score(graph, length, drug1, drug2, diameter1, diameter2, weight):
	# graph, g1, g2, drug1, drug2, diameter1, diameter2, weight
	"""
	Retrieved from Menche (2015)
	ds : shortest distance
	For each of the Nd disease proteins, we determine the distance ds to the next-closest protein
	associated with the same disease.
	The average 〈ds〉 can be interpreted as the diameter of a disease on the interactome.
	The network-based overlap between two diseases A and B is measured by comparing the
	diameters〈dAA〉 and 〈dBB〉 of the respective diseases to the mean shortest distance〈dAB〉
	between their proteins: sAB = 〈dAB〉 – (〈dAA〉 + 〈dBB〉)/2.

	Positive sAB indicates that the two disease modules are separated on the interactome,
	whereas negative values correspond to overlapping modules
	:return:
	"""

	if len(drug1.split("/")) == 1:
		nodes1 = Drug(drug1).targets
	else:
		nodes1 = list()
		for drug in drug1.split("/"):
			drug_target = Drug(drug).targets
			if drug_target is not None:
				for t in drug_target:
					if t not in nodes1:
						nodes1.append(t)
	if nodes1 is []:
		nodes1 = None

	if len(drug2.split("/")) == 1:
		nodes2 = Drug(drug2).targets
	else:
		nodes2 = list()
		for drug in drug2.split("/"):
			drug_target = Drug(drug).targets
			if drug_target is not None:
				for t in drug_target:
					if t not in nodes2:
						nodes2.append(t)
	if nodes2 is []:
		nodes2 = None

	# Get shortest paths

	if nodes1 is not None and nodes2 is not None:
		# Use only the drug targets
		average_g12_ds = get_similarity(graph=graph, length=length, nodes1=nodes1, nodes2=nodes2, weight=weight)

		"""
		g1_ds = list()
		for node_pair in list(itertools.permutations(g1.nodes, 2)):
			if networkx.has_path(interactome, source=node_pair[0], target=node_pair[1]):
				x = networkx.shortest_path_length(interactome, source=node_pair[0], target=node_pair[1])
				g1_ds.append(x)

		g2_ds = list()
		for node_pair in list(itertools.permutations(g2.nodes, 2)):
			if networkx.has_path(interactome, source=node_pair[0], target=node_pair[1]):
				x = networkx.shortest_path_length(interactome, source=node_pair[0], target=node_pair[1])
				g2_ds.append(x)

		g12_ds = list()
		for n1 in g1.nodes:
			node_ds = list()
			for n2 in g2.nodes:
				if n1 == n2:
					g12_ds.append(0)
				else:
					if networkx.has_path(interactome_obj, source=n1, target=n2):
						x = networkx.shortest_path_length(interactome_obj, source=n1, target=n2)
						node_ds.append(x)
			d = min(node_ds)
			g12_ds.append(d)

		# Average is diameter of the drug network on interactome
		average_g1_ds, average_g2_ds, average_g12_ds = numpy.mean(g1_ds), numpy.mean(g2_ds), numpy.mean(g12_ds)
		"""

		"""
		sAB = <dAB> - (<dAA> + <dBB>) / 2
		"""

		if diameter1 is not None and diameter2 is not None and average_g12_ds is not None:
			sab = average_g12_ds - ((diameter1 + diameter2) / 2)
		else:
			sab = None
	else:
		sab = None

	return sab


def get_single_combination_network_similarity(graph_type, interactome_name, drug_networks, diameters, drug_comb,
											  weight, alpha, network_name, value_type,
											  random_tag, seed):
	"""
	Drug network similarity calculation
	:param graph_type: interactome/subnetwork
	:param interactome_name: Name of interactome
	:param drug_networks: drug network dictionary from get_drug_networks
	:param drug_comb: drug combination
	:param diameters: diameter dictionary from collect_drug_network_diameter
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param weight: usage of weight or not
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:param value_type: bayesian, logfc
    :param network_name: interactome/curated_interactome
	:return:
	"""

	drug_text = drug_comb.split("/")[0] + "_" + drug_comb.split("/")[1]
	drug1, drug2 = drug_comb.split("/")[0], drug_comb.split("/")[1]

	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	alpha_text = "_" + str(int(alpha * 100))

	if random_tag:
		random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
		random_col = "random"
		random_text = "_random_seed_%d" % seed
		output_folder = random_output_path + "network/modelling/PPR/%s/drugs/%s/similarity/" % (
		interactome_name, random_col)
	else:
		random_text = "_empiric"
		random_col = "/empiric/"
		output_folder = output_path + "network/modelling/PPR/%s/drugs/similarity%s" % (interactome_name, random_col)

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
	else:
		network_text, value_text = "", ""

	if "%s_%s%s%s_%s_network_similarity_nx_%s%s%s.csv" % (
	interactome_name, graph_type, network_text, value_text, drug_text, weight_text, alpha_text,
	random_text) not in os.listdir(output_folder):

		# Get interacrome object
		interactome = get_norm_interactome(interactome_name=interactome_name, weight=weight, filter_mi=0.4)

		if graph_type == "interactome":
			df = pandas.DataFrame(columns=["jaccard_index", "overlap_coefficient"], index=[drug_comb])
		else:
			df = pandas.DataFrame(columns=["drug_combination", "model_id", "jaccard_index", "overlap_coefficient"])

		if graph_type == "interactome":

			length = get_graph_shortest_length(graph_type=graph_type, model_id=None, interactome_name=interactome_name,
											   weight=False,
											   network_name=network_name, value_type=value_type, filter_mi=0.4,
											   random_tag=random_tag, seed=seed)

			if drug1 in drug_networks.keys() and drug2 in drug_networks.keys():
				network1, network2 = drug_networks[drug1], drug_networks[drug2]
				nodes1, nodes2 = network1.nodes, network2.nodes
				all_nodes = list(interactome.nodes)

				# D1 - D2
				jaccard_index = get_network_overlaps(node_set1=nodes1, node_set2=nodes2,
													 all_nodes=all_nodes, overlap_method="jaccard")

				overlap_coefficient = get_network_overlaps(node_set1=nodes1, node_set2=nodes2,
														   all_nodes=all_nodes, overlap_method="coefficient")

				df.loc[drug_comb, "jaccard_index"] = jaccard_index
				df.loc[drug_comb, "overlap_coefficient"] = overlap_coefficient


		elif graph_type == "subnetwork":
			drug_df = pandas.DataFrame(columns=["jaccard_index", "overlap_coefficient"],
									   index=list(drug_networks.keys()))
			for model_id, model_networks in drug_networks.items():

				length = get_graph_shortest_length(graph_type=graph_type, model_id=model_id,
												   interactome_name=interactome_name, weight=weight,
												   network_name=network_name, value_type=value_type, filter_mi=0.4,
												   random_tag=random_tag, seed=seed)

				graph = read_network_file(w=w, d=d, b=b, mu=mu, model_id=model_id,
										  interactome_name=interactome_name,
										  compounds=None, value_type=value_type, network_name=network_name,
										  random_tag=random_tag, seed=seed)
				if weight:
					for edge in graph.edges:
						graph[edge[0]][edge[1]][weight_opt] = interactome.get_edge_data(edge[0], edge[1])[
							weight_opt]

				all_nodes = list(graph.nodes())

				if drug1 in model_networks.keys() and drug2 in model_networks.keys():
					network1, network2 = model_networks[drug1], model_networks[drug2]
					nodes1, nodes2 = network1.nodes, network2.nodes

					# D1 - D2
					jaccard_index = get_network_overlaps(node_set1=nodes1, node_set2=nodes2,
														 all_nodes=all_nodes, overlap_method="jaccard")

					overlap_coefficient = get_network_overlaps(node_set1=nodes1, node_set2=nodes2,
															   all_nodes=all_nodes, overlap_method="coefficient")
					"""
					similarity_score = get_similarity_score(graph=graph, length=length,
															drug1=drug1, drug2=drug2,
															diameter1=diameters[model_id][drug1],
															diameter2=diameters[model_id][drug2],
															weight=weight)
					"""
					drug_df.loc[model_id, "jaccard_index"] = jaccard_index
					drug_df.loc[model_id, "overlap_coefficient"] = overlap_coefficient
			# drug_df.loc[model_id, "similarity_score"] = similarity_score

			drug_df = drug_df.reset_index()
			drug_df.columns = ["model_id", "jaccard_index", "overlap_coefficient"]
			drug_df["drug_combination"] = drugs
			df = pandas.concat([df, drug_df], axis=1)

		df.to_csv(output_folder + "%s_%s%s%s_%s_network_similarity_nx_%s%s%s.csv"
				  % (interactome_name, graph_type, network_text, value_text, drug_text, weight_text, alpha_text,
					 random_text),
				  index=True)

	else:
		df = pandas.read_csv(output_folder + "%s_%s%s%s_%s_network_similarity_nx_%s%s%s.csv"
							 % (
							 interactome_name, graph_type, network_text, value_text, drug_text, weight_text, alpha_text,
							 random_text), index_col=0)

	return df


def collect_drug_network_similarity(graph_type, interactome_name, weight, alpha, network_name, value_type,
									random_tag, seed):
	"""
	Collect all random drug/drug combination objects
	:param graph_type: interactome/subnetwork
	:param interactome_name: Name of interactome
	:param alpha: Damping Factor (probability at any step that the walk will continue following edges
	:param weight: usage of weight or not
	:param random_tag: If the interactome random or not
	:param seed: Randomness constant
	:param value_type: bayesian, logfc
    :param network_name: interactome/curated_interactome
	:return: Dictinary of networkx objects for random
	"""

	if weight:
		weight_text = "weighted"
	else:
		weight_text = "unweighted"

	alpha_text = "_" + str(int(alpha * 100))

	if graph_type == "subnetwork":
		network_text = "_%s" % network_name
		value_text = "_%s" % value_type
	else:
		network_text, value_text = "", ""

	if random_tag:
		random_text = "_random_%d" % seed
	else:
		random_text = "_empiric"

	output_folder = output_path + "network/modelling/PPR/%s/drugs/statistics/" % interactome_name

	if "%s_%s%s%s_drug_network_similarity_%s%s%s.csv" % (
	interactome_name, graph_type, network_text, value_text, weight_text, alpha_text, random_text) not in os.listdir(
			output_folder):

		all_combinations = [i for i in all_screened_combinations(project_list=None, integrated=True) if
							len(i.split(" | ")) == 1]

		drug_networks = get_drug_networks(graph_type=graph_type, interactome_name=interactome_name, stage="mono",
										  alpha=alpha, weight=weight, network_name=network_name, value_type=value_type,
										  filter_mi=0.4, random_tag=random_tag, seed=seed)

		diameters = collect_drug_network_diameter(graph_type=graph_type, interactome_name=interactome_name,
												  stage="mono",
												  weight=weight, alpha=alpha, network_name=network_name,
												  value_type=value_type,
												  random_tag=random_tag, seed=seed)

		dfs = list()
		for drug_comb in all_combinations:
			df = get_single_combination_network_similarity(graph_type=graph_type, interactome_name=interactome_name,
														   drug_networks=drug_networks, diameters=diameters,
														   drug_comb=drug_comb, weight=weight, alpha=alpha,
														   network_name=network_name, value_type=value_type,
														   random_tag=random_tag, seed=seed)
			dfs.append(df)

		whole_similarity_df = pandas.concat(dfs)

		whole_similarity_df.to_csv(output_folder + "%s_%s%s%s_drug_network_similarity_%s%s%s.csv"
								   % (interactome_name, graph_type, network_text, value_text, weight_text,
									  alpha_text, random_text), index=True)

	else:
		whole_similarity_df = pandas.read_csv(output_folder + "%s_%s%s%s_drug_network_similarity_%s%s%s.csv"
											  % (interactome_name, graph_type, network_text, value_text, weight_text,
												 alpha_text, random_text), index_col=0)
	return whole_similarity_df


def main(args):
	if "SEED" in args.keys() and args["SEED"] is not None and args["SEED"] != "None":
		random = True
		seed = int(args["SEED"])
	else:
		random, seed = False, None

	if args["RUN_FOR"] == "ppr":
		interactome = get_norm_interactome(interactome_name="intact", weight=True, filter_mi=0.4)
		_ = run_ppr_drug(graph=None, model_id=None, interactome=interactome, graph_type="interactome",
						 interactome_name="intact", drug_name=args["DRUG"],
						 stage="mono", alpha=args["ALPHA"], random_tag=random, seed=seed, weight=True)

	if args["RUN_FOR"] == "module":
		drug_networks = get_drug_networks(graph_type="interactome", interactome_name="intact", stage="mono",
										  alpha=args["ALPHA"], weight=True,
										  network_name="interactome", value_type="bayesian", filter_mi=0.4,
										  random_tag=random, seed=seed)

	if args["RUN_FOR"] == "random_module":
		_ = get_random_drug_networks(interactome_name="intact", stage="mono", weight=True, alpha=0.85, seed=seed,
									 filter_mi=0.4)

	if args["RUN_FOR"] == "run_diameter":
		drug_networks = get_drug_networks(graph_type="interactome", interactome_name="intact", stage="mono",
										  alpha=args["ALPHA"], weight=True,
										  network_name="interactome", value_type="bayesian", filter_mi=0.4,
										  random_tag=random, seed=seed)

		_ = get_single_drug_network_diameter(graph_type="interactome", drug_networks=drug_networks,
											 interactome_name="intact", drug=args["DRUG"],
											 weight=True, stage="mono", alpha=args["ALPHA"], network_name="interactome",
											 value_type=None, random_tag=random, seed=seed)

	if args["RUN_FOR"] == "diameter":
		diameters = collect_drug_network_diameter(graph_type="interactome", interactome_name="intact", stage="mono",
												  weight=True, alpha=args["ALPHA"],
												  network_name="interactome", value_type=None, random_tag=random,
												  seed=seed)

	if args["RUN_FOR"] == "length":
		length = get_graph_shortest_length(graph_type="interactome", model_id=None, interactome_name="intact",
										   weight=False, network_name="interactome",
										   value_type=None, filter_mi=0.4, random_tag=random, seed=seed)

	if args["RUN_FOR"] == "run_similarity":
		drug_networks = get_drug_networks(graph_type="interactome", interactome_name="intact", stage="mono",
										  alpha=args["ALPHA"], weight=True,
										  network_name="interactome", value_type="bayesian", filter_mi=0.4,
										  random_tag=random, seed=seed)

		diameters = collect_drug_network_diameter(graph_type="interactome", interactome_name="intact", stage="mono",
												  weight=True, alpha=args["ALPHA"],
												  network_name="interactome", value_type=None, random_tag=random,
												  seed=seed)

		_ = get_single_combination_network_similarity(graph_type="interactome", interactome_name="intact",
													  drug_networks=drug_networks, diameters=diameters,
													  drug_comb=args["DRUG"], weight=True, alpha=args["ALPHA"],
													  network_name="interactome", value_type=None,
													  random_tag=random, seed=seed)

	if args["RUN_FOR"] == "similarity":
		similarities = collect_drug_network_similarity(graph_type="interactome", interactome_name="intact", weight=True,
													   alpha=args["ALPHA"],
													   network_name="interactome", value_type=None, random_tag=random,
													   seed=seed)

	return True


if __name__ == '__main__':

	args = take_input()

	print(args)

	if "ALPHA" in args.keys():
		if args["ALPHA"] is not None and args["ALPHA"] != "None":
			args["ALPHA"] = float(args["ALPHA"])

	if "SEED" in args.keys():
		if args["SEED"] is not None and args["SEED"] != "None":
			args["SEED"] = int(args["SEED"])

	print(args)

	_ = main(args=args)




