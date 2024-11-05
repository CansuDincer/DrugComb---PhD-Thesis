"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 29 November 2023
Last Update : 16 May 2024
Input : Interactome obj
Output : Randomisation and normalisation of interactome
#------------------------------------------------------------------------#
"""

import sys

import matplotlib.pyplot as plt

if "/lustre/scratch125/casm/team215mg/cd7/CombDrug/" not in sys.path:
	sys.path.insert(0, "/lustre/scratch125/casm/team215mg/cd7/CombDrug/")

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
import os, pandas, numpy, scipy, sklearn, networkx, pickle,  re, requests, argparse
import seaborn as sns
import matplotlib.pyplot as plt
#from statannot import add_stat_annotation

from CombDrug.module.path import *
from CombDrug.module.data.dataset_info import *


def take_input():
	parser = argparse.ArgumentParser()

	# Feature
	parser.add_argument("-ppi", dest="PPI", default="intact")
	parser.add_argument("-w", dest="WEIGHT", default=True)
	parser.add_argument("-filter", dest="filter", default=0.4)
	parser.add_argument("-seed", dest="SEED", required=False)
	parsed_input = parser.parse_args()
	input_dict = vars(parsed_input)

	return input_dict



# ---------------------------------------------------------------------------#
#                                  Functions                                 #
# ---------------------------------------------------------------------------#

def extract_intact_gene_name(alias_interactor):
	for i in alias_interactor.split("|"):
		if i.split("(")[1] == "gene name)":
			return i.split(":")[1].split("(")[0]



def get_interactome(interactome_name, filter_mi):
	"""
	Get the whole interactome
	:param interactome_name:
	:param filter_mi:
	:return:
	"""
	random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
	if interactome_name == "omnipath":
		if "omnipath_23.csv" not in os.listdir(random_output_path + "network/interactome/omnipath/empiric/"):
			interactome_dict = dict()
			omnipath_request = requests.get("http://omnipathdb.org/interactions/?genesymbols=1")
			# op_annotation_request =/ requests.get("https://omnipathdb.org/annotations")
			count = 0
			for row in omnipath_request.text.split("\n")[1:]:
				l = row.split("\t")
				if l != [""]:
					interactome_dict[count] = {"source": l[2], "target": l[3], "is_directed": l[4],
											   "is_stimulation": l[5], "is_inhibition": l[6],
											   "source_uniprot": l[0], "target_uniprot": l[1]}
					count += 1

			interaction_df = pandas.DataFrame.from_dict(interactome_dict, orient="index")

			G = networkx.from_pandas_edgelist(interaction_df, 'source', 'target', True, networkx.MultiDiGraph())

			removed_nodes = list()
			for node in G.nodes:
				if len(node.split("_")) > 1:
					removed_nodes.append(node)
			G.remove_nodes_from(removed_nodes)
			G.remove_edges_from(networkx.selfloop_edges(G))

			interactome = pandas.DataFrame(None, columns = ["From", "To", "is_directed", "is_stimulation",
															"is_inhibition"])
			for edge in G.edges:
				x = pandas.DataFrame({"From": [edge[0]], "To": [edge[1]],
									  "is_directed": G.get_edge_data(edge[0], edge[1])[0]["is_directed"],
									  "is_stimulation": G.get_edge_data(edge[0], edge[1])[0]["is_stimulation"],
									  "is_inhibition": G.get_edge_data(edge[0], edge[1])[0]["is_inhibition"]})
				interactome = pandas.concat([interactome, x], ignore_index=True)

			interactome.to_csv(random_output_path + "network/interactome/omnipath/empiric/omnipath_23.csv", index=False)
			#networkx.write_weighted_edgelist(G, output_path + "network/interactome/omnipath/omnipath23_removed.csv")

		else:
			interactome = pandas.read_csv(random_output_path + "network/interactome/omnipath/empiric/omnipath_23.csv")
			#G = networkx.read_weighted_edgelist(output_path + "network/interactome/omnipath/omnipath23_removed.csv")
			#G = networkx.from_pandas_edgelist(interactome, 'source', 'target', True, networkx.MultiDiGraph())

			G = networkx.MultiDiGraph()
			for i,row in interactome.iterrows():
				G.add_edge(row["From"], row["To"], is_directed= row["is_directed"],
						   is_stimulation= row["is_stimulation"], is_inhibition =  row["is_inhibition"])
		return G

	elif interactome_name == "intact":
		if "intact.csv" not in os.listdir(random_output_path + "network/interactome/intact/empiric/"):
			df = pandas.read_csv(input_path + "intact/intact.txt", sep="\t")
			df = df[['#ID(s) interactor A', 'ID(s) interactor B', 'Alias(es) interactor A',
					 'Alias(es) interactor B', 'Interaction type(s)', 'Confidence value(s)',
					 'Taxid interactor A', 'Taxid interactor B']]

			# Select only the interaction happening in Human
			df["selected_host"] = df.apply(
				lambda x: True if (re.search(r'9606', x["Taxid interactor A"])) and
								  (re.search(r'9606', x["Taxid interactor B"])) else False, axis=1)
			df = df[df.selected_host]

			# Annotate interaction type
			df["interaction_type"] = df.apply(lambda x: x["Interaction type(s)"].split("(")[1].split(")")[0], axis=1)

			# Extract Uniprot names
			df["uniprot_1"] = df.apply(
				lambda x: x['#ID(s) interactor A'].split(":")[1].split("-")[0] if x['#ID(s) interactor A'].split(":")[0] =="uniprotkb"
				else None, axis=1)
			df["uniprot_2"] = df.apply(
				lambda x: x['ID(s) interactor B'].split(":")[1].split("-")[0] if x['ID(s) interactor B'].split(":")[0] =="uniprotkb"
				else None, axis=1)

			# Get only the SwissProt entries
			uniprot = get_uniprot_conversion().reset_index()
			df["swissprot"] = df.apply(
				lambda x: True if (x.uniprot_1 in uniprot["index"].unique()) and
								  (x.uniprot_2 in uniprot["index"].unique()) else False, axis=1)
			df = df[df.swissprot]

			# Select the row having gene name rather than domains etc.
			df["selected"] = df.apply(
				lambda x: True if (re.search(r'gene name', x['Alias(es) interactor A'])) and
								  (re.search(r'gene name', x['Alias(es) interactor B'])) else False, axis=1)
			df = df[df.selected]

			# Annotate gene names
			df["gene_1"] = df.apply(lambda x: extract_intact_gene_name(x["Alias(es) interactor A"]), axis=1)
			df["gene_2"] = df.apply(lambda x: extract_intact_gene_name(x["Alias(es) interactor B"]), axis=1)

			# Remove self loops
			df["different_genes"] = df.apply(lambda x: True if x.gene_1 != x.gene_2 else False, axis=1)
			df["different_uniprots"] = df.apply(lambda x: True if x.uniprot_1 != x.uniprot_2 else False, axis=1)
			df = df[df.different_genes]
			#df = df[df.different_uniprots]

			# Annotate MI scores
			df["MI_score"] = df.apply(lambda x: float(x["Confidence value(s)"].split("intact-miscore:")[1]), axis=1)
			df = df[["gene_1", "gene_2", "uniprot_1", "uniprot_2", "interaction_type", "MI_score"]]

			# Remove duplication in the dataframe
			df = df.dropna().drop_duplicates()

			# Get specific interactions including direct and physical interactions as well as phosphorylation
			df = df[df.interaction_type.isin(["direct interaction", "physical association", "phosphorylation reaction"])]

			# Take the highest MI score interactions from the same set of Gene1 and Gene2
			# Since there are repetitions
			repetitive_rows = list()
			for g, g_df in df.groupby(["gene_1", "gene_2"]):
				if len(g_df.index) > 1:
					if len(g_df.uniprot_1.unique()) == 1 and len(g_df.uniprot_2.unique()) == 1:
						if len(g_df.MI_score.unique()) > 1:
							ind = list(g_df[g_df.MI_score == max(g_df.MI_score)].index)
							deleted_ind = [i for i in g_df.index if i != ind]
							repetitive_rows.extend(deleted_ind)

			df2 = df.drop(index=repetitive_rows)

			# Check each gene has only one uniprot -- 16 of them
			"""
			uniprot_gene_intact = dict()
			for gene, gene_df in df2.groupby(["gene_1"]):
				if gene not in uniprot_gene_intact.keys():
					uniprot_gene_intact[gene] = list(gene_df.uniprot_1.unique())
				else:
					t = uniprot_gene_intact[gene]
					for u in list(gene_df.uniprot_1.unique()):
						if u not in t:
							t.append(u)
					uniprot_gene_intact[gene] = t

			for gene, gene_df in df2.groupby(["gene_2"]):
				if gene not in uniprot_gene_intact.keys():
					uniprot_gene_intact[gene] = list(gene_df.uniprot_2.unique())
				else:
					t = uniprot_gene_intact[gene]
					for u in list(gene_df.uniprot_2.unique()):
						if u not in t:
							t.append(u)
					uniprot_gene_intact[gene] = t

			for gene, uniprot_list in uniprot_gene_intact.items():
				if len(uniprot_list) > 1:
					print(gene)
			"""

			# Dump interactome into a file
			G = networkx.from_pandas_edgelist(df, 'gene_1', 'gene_2', edge_attr=True, create_using=networkx.Graph())
			G.remove_edges_from(networkx.selfloop_edges(G))
			interactome = pandas.DataFrame(None, columns = ["From", "To", "Weight"])
			for edge in G.edges:
				x = pandas.DataFrame({"From": [edge[0]], "To": [edge[1]],
									  "Weight": G.get_edge_data(edge[0], edge[1])["MI_score"]})
				interactome = pandas.concat([interactome, x], ignore_index=True)

			interactome.to_csv(random_output_path + "network/interactome/intact/empiric/intact.csv", index=False)

			# FIlter out less confident edges/interactions
			filtered_G = G.copy()
			filtered_G.remove_edges_from([edge for edge in filtered_G.edges if filtered_G.get_edge_data(edge[0], edge[1])["MI_score"] < filter_mi])

			filtered_interactome = pandas.DataFrame(None, columns = ["From", "To", "Weight"])
			for edge in filtered_G.edges:
				x = pandas.DataFrame({"From": [edge[0]], "To": [edge[1]],
									  "Weight": filtered_G.get_edge_data(edge[0], edge[1])["MI_score"]})
				filtered_interactome = pandas.concat([filtered_interactome, x], ignore_index=True)

			filtered_interactome.to_csv(random_output_path + "network/interactome/intact/empiric/intact_filtered_%d.csv"
										% int(filter_mi * 10), index=False)

		else:
			#interactome = networkx.read_weighted_edgelist(output_path + "network/interactome/intact/intact.csv")
			filtered_interactome = pandas.read_csv(random_output_path + "network/interactome/intact/empiric/intact_filtered_%d.csv"
												   % int(filter_mi * 10))
			filtered_G = networkx.from_pandas_edgelist(filtered_interactome, 'From', 'To', True, networkx.Graph())

		return filtered_G




def shuffle_network(interactome_obj, network_name, interactome_name, seed):
	"""
	Shuffling the edges of the nodes
	:param interactome_obj: Networkx object of PPI graph
	:param network_name: interactome, curated_interactome, modelled
	:param interactome_name: Name of the interactome
	:param seed: The random seed number
	:return: Randomised network
	"""

	random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
	if "random_%s_%s_seed_%d.csv" % (network_name, interactome_name, seed) not in os.listdir(
			random_output_path + "network/interactome/%s/random/" % interactome_name):

		# Make it undirected
		g = interactome_obj.to_undirected()
		# Remove the additional node added for modelling
		if "a_node" in g.nodes:
			g.remove_node("a_node")

		# Node degree preserving network randomisation
		nodes, degree_list = {}, []
		k = 0
		for i in networkx.degree(g):
			nodes[k] = i[0]
			degree_list.append(i[1])
			k += 1

		weight_list = []
		for edge in g.edges():
			weight_list.append(g.get_edge_data(edge[0], edge[1])["Weight"])

		shuffled_g = networkx.configuration_model(deg_sequence=degree_list, create_using=networkx.Graph,
												  seed = seed)
		shuffled_g = networkx.relabel_nodes(shuffled_g, nodes)

		# Configuration model can create self loops so we need to remove them
		shuffled_g.remove_edges_from(networkx.selfloop_edges(shuffled_g))

		# Shuffle edges
		weight_list = numpy.array(weight_list)
		numpy.array(numpy.random.shuffle(weight_list))
		shuffled_weights_ind = numpy.random.choice(len(weight_list), size=len(shuffled_g.edges), replace=False)
		shuffled_weights = weight_list[shuffled_weights_ind]

		shuffled_edge_df = pandas.DataFrame(columns = ["From", "To"], index = list(range(len(shuffled_g.edges))))
		i = 0
		for edge in shuffled_g.edges:
			shuffled_edge_df.loc[i, "From"] = edge[0]
			shuffled_edge_df.loc[i, "To"] = edge[1]
			shuffled_edge_df.loc[i, "Weight"] = shuffled_weights[i]
			i += 1

		shuffled_edge_df.to_csv(random_output_path + "network/interactome/%s/random/random_%s_%s_seed_%d.csv"
								% (interactome_name, network_name, interactome_name, seed), index=False)

	else:
		shuffled_edge_df = pandas.read_csv(random_output_path + "network/interactome/%s/random/random_%s_%s_seed_%d.csv"
										   % (interactome_name, network_name, interactome_name, seed))

		shuffled_g = networkx.Graph()
		for ind, row in shuffled_edge_df.iterrows():
			shuffled_g.add_edge(row.From, row.To, weight=row.Weight)

	return shuffled_g


def normalised_interactome(interactome, interactome_name, weight, random_tag, seed):
	"""
	Laplacian normalisation of network edges
	:return:
	"""

	if interactome_name == "intact":
		if weight:
			weight_text = "weighted"
			weight_op = "Weight"
		else:
			weight_text = "unweighted"
			weight_op = None
	else:
		weight_text = "unweighted"
		weight_op = None

	if random_tag:
		random_text = "_random_seed_%d" % seed
		random_col = "random/"
	else:
		random_text = ""
		random_col = "empiric/"

	random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"
	if "normalised_%s_%s%s.csv" % (interactome_name, weight_text, random_text) not in os.listdir(
			random_output_path + "network/interactome/intact/%s" % random_col):

		interactome.remove_edges_from(networkx.selfloop_edges(interactome))

		# Alphabetically sort the nodes
		new_node_list = sorted(list(interactome.nodes), key=str.casefold)
		node_map = {i: new_node_list[i] for i in range(len(new_node_list))}

		# Laplacian normalisation
		norm_laplacian_matrix = networkx.normalized_laplacian_matrix(
			interactome, nodelist=new_node_list, weight=weight_op).toarray()

		# Due to approximation there are -16th decimal differences from 1, so I fixed it
		numpy.fill_diagonal(norm_laplacian_matrix, 1)

		# Identity matrix
		identity_matrix = numpy.identity(len(new_node_list))

		# Get normalised adjencecy matrix
		norm_adjacency_matrix = numpy.array(identity_matrix - norm_laplacian_matrix)

		# Normalised interactome
		norm_interactome = networkx.from_numpy_matrix(norm_adjacency_matrix, create_using=networkx.Graph())
		norm_interactome = networkx.relabel_nodes(norm_interactome, node_map)

		graph_df = pandas.DataFrame(columns=["From", "To", "Weight"], index = list(range(len(norm_interactome.edges))))
		i = 0
		for edge in norm_interactome.edges:
			graph_df.loc[i, "From"] = edge[0]
			graph_df.loc[i, "To"] = edge[1]
			graph_df.loc[i, "Weight"] = norm_interactome.get_edge_data(edge[0], edge[1])["weight"]
			i += 1

		graph_df.to_csv(random_output_path + "network/interactome/intact/%snormalised_%s_%s%s.csv"
						% (random_col, interactome_name, weight_text, random_text), index=False)

	else:
		graph_df = pandas.read_csv(random_output_path + "network/interactome/intact/%snormalised_%s_%s%s.csv"
								   % (random_col, interactome_name, weight_text, random_text))
		norm_interactome = networkx.from_pandas_edgelist(graph_df, 'From', 'To', True, networkx.Graph())

	return norm_interactome

"""
def plot_random_network_degree(interactome_obj, network_name, interactome_name, weight, numb, filter_mi):

	if weight: weight_text = "weighted"
	else: weight_text = "unweighted"

	# Check the degree distibution
	degree_df = pandas.DataFrame(columns=["Empirical", "Averaged_Randomised"], index=list(interactome_obj.nodes))
	for node in list(interactome_obj.nodes):
		degree_e = networkx.degree(interactome_obj)[node]
		degree_r_list = list()
		for random_g_seed in range(numb):
			random_g = get_norm_random_interactome(interactome_name=interactome_name, weight=weight, seed=random_g_seed, filter_mi=filter_mi)
			if node in random_g.nodes:
				x = networkx.degree(random_g)[node]
				degree_r_list.append(x)
		degree_r = numpy.mean(degree_r_list)
		degree_df.loc[node, "Empirical"] = degree_e
		degree_df.loc[node, "Averaged_Randomised"] = degree_r

	slope, intercept, r_value, p_value, _ = \
		scipy.stats.linregress([list(degree_df["Empirical"]), list(degree_df["Averaged_Randomised"])])

	sns.scatterplot(data=degree_df, x="Empirical", y="Averaged_Randomised", color="navy", alpha=0.4, label="Node Degree")
	plt.plot(degree_df["Empirical"], intercept + slope * degree_df["Empirical"], "r", label="Fitted Line",
			 alpha=0.6)
	# onetoone = numpy.linspace(min(degree_df["Empirical"]), max(degree_df["Empirical"]), 10)
	# plt.plot(onetoone, onetoone, label="1:1 Fit", linestyle=":", color="grey")
	plt.suptitle("Degree Correlation")
	plt.xlabel("Empirical Degree")
	plt.ylabel("Randomised Averaged  Degree")
	plt.text(x=max(degree_df["Averaged_Randomised"]) * 0.00001, y=max(degree_df["Averaged_Randomised"]) * 0.95,
			 s=r"y = %.2f + %.2f x" % (intercept, slope))
	plt.text(x=max(degree_df["Averaged_Randomised"]) * 0.00001, y=max(degree_df["Averaged_Randomised"]) * 0.89,
			 s=r"R square: %.3f" % r_value ** 2)
	plt.legend(loc='upper left')
	plt.tight_layout()
	plt.savefig(output_path + "network/figures/random_network_degrees/random_%s_%s_%s_%d.pdf"
				% (network_name, interactome_name, weight_text, numb), dpi=300)
	plt.savefig(output_path + "network/figures/random_network_degrees/random_%s_%s_%s_%d.jpg"
				% (network_name, interactome_name, weight_text, numb), dpi=300)
	plt.savefig(output_path + "network/figures/random_network_degrees/random_%s_%s_%s_%d.png"
				% (network_name, interactome_name, weight_text, numb), dpi=300)

	return True
"""

def plot_random_network_degree(interactome_obj, network_name, interactome_name, weight, numb, filter_mi):
	if weight: weight_text = "weighted"
	else: weight_text = "unweighted"

	# Check the degree distibution
	degree_df = pandas.DataFrame(columns=["Node", "Type", "Degree"])
	for node in list(interactome_obj.nodes):
		degree_e = networkx.degree(interactome_obj)[node]
		x = {"Node": [node], "Type": ["empiric"], "Degree": [degree_e]}
		x_df = pandas.DataFrame.from_dict(x)
		degree_df = pandas.concat([degree_df, x_df], axis=0, ignore_index=True)

	for random_g_seed in range(numb):
		random_g = get_norm_random_interactome(interactome_name=interactome_name, weight=weight,
											   seed=random_g_seed, filter_mi=filter_mi)
		for node in list(interactome_obj.nodes):
			if node in list(random_g.nodes):
				degree_r = networkx.degree(random_g)[node]
				x = {"Node": [node], "Type": ["random_%d" % random_g_seed], "Degree": [degree_r]}
				x_df = pandas.DataFrame.from_dict(x)
				degree_df = pandas.concat([degree_df, x_df], axis=0, ignore_index=True)

	degree_df.to_csv(output_path + "network/interactome/%s/statistics/%s_%s_degree_distribution.csv"
					 % (interactome_name, weight_text, interactome_name), index=False)
	return True


def get_norm_interactome(interactome_name, weight, filter_mi):
	"""
	Laplacian normalised empiric interactome
	"""

	# Get interactome itself

	interactome = get_interactome(interactome_name, filter_mi=filter_mi)
	norm_interactome = normalised_interactome(interactome=interactome, interactome_name=interactome_name,
											  weight=weight, random_tag=False, seed=None)

	return norm_interactome


def get_norm_random_interactome(interactome_name, weight, seed, filter_mi):
	"""
	Prepare or call random interactomes
	:return:
	"""

	if weight:
		weight_text = "weighted"
		weight_op = "Weight"
	else:
		weight_text = "unweighted"
		weight_op = None

	random_output_path = "/lustre/scratch127/casm/team215mg/cd7/CombDrug/output/"

	if "normalised_%s_%s_random_seed_%d.csv" % (interactome_name, weight_text, seed) not in os.listdir(
			random_output_path + "network/interactome/intact/random/"):

		# Get interacrome object
		interactome = get_interactome(interactome_name=interactome_name, filter_mi=filter_mi)

		shuffled_interactome = shuffle_network(interactome_obj=interactome, network_name="interactome",
											   interactome_name=interactome_name, seed=seed)

		norm_shuffled_interactome = normalised_interactome(interactome=shuffled_interactome, weight=weight_op,
														   random_tag=True, seed=seed, interactome_name=interactome_name)

	else:
		graph_df = pandas.read_csv(random_output_path + "network/interactome/intact/random/normalised_%s_%s_random_seed_%d.csv"
								   % (interactome_name, weight_text, seed))
		norm_shuffled_interactome = networkx.from_pandas_edgelist(graph_df, 'From', 'To', True, networkx.Graph())

	return norm_shuffled_interactome



def main(args):

	if args["RUN_FOR"] == "empiric_interactome":
		_ = get_norm_interactome(interactome_name=args["PPI"], weight=args["WEIGHT"], filter_mi=args["FILTER"])

	if args["RUN_FOR"] == "random_interactome":
		_ = get_norm_random_interactome(interactome_name=args["PPI"], weight=args["WEIGHT"], seed=args["SEED"], filter_mi=args["FILTER"])

	return True



if __name__ == '__main__':

	args = take_input()

	if "SEED" in args.keys():
		args["SEED"] = int(args["SEED"])

	if "FILTER" in args.keys():
		args["FILTER"] = float(args["SEED"])

	if "WEIGHT" in args.keys():
		if args["WEIGHT"] in [True, "True"]:
			args["WEIGHT"] = True
		elif args["WEIGHT"] in [False, None, "False", "None"]:
			args["WEIGHT"] = False

	_ = main(args=args)







