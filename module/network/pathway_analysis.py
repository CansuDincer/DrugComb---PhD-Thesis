"""
# ------------------------------------------------------------------------#
Author : Cansu Dincer
Date : 21 February 2023
Last Update : 15 February 2024
Input : Pathway Analysis after network modelling
Output : Biological comparison of the modelling
#------------------------------------------------------------------------#
"""

# ---------------------------------------------------------------------------#
#                                   Import                                   #
# ---------------------------------------------------------------------------#
from drafts.pcsf_statistics import *


# ---------------------------------------------------------------------------#
#                                Take Pathways                               #
# ---------------------------------------------------------------------------#

def get_pathways(model_id, drug_name, prize, interactome_name, pathway_db,
                 modelling_approach, random_tag, alpha):
    """
    Retrieving the pathway table
    :param model_id: Sanger cell model id
    :param drug_name: Name of the single agent drug or drug combinations
    :param prize: Boolean - if the pathways of the prize nodes
    :param interactome_name : Name of the used interactome
    :param pathway_db: KEGG/Wiki/Reactome
    :param modelling_approach: PPR or PCSF
    :return:
    """

    if model_id is not None and drug_name is None:
        col_text = "cell_models"
        sample_col_text = model_id
    else:
        col_text = "drugs"
        if len(drug_name.split("/")) > 1:
            sample_col_text = "_".join(drug_name.split("/"))
        else:
            sample_col_text = drug_name

    if random_tag:
        random_col = "/random/"
    else:
        random_col = "/empiric/"

    if prize: path_col = "nodes"
    else: path_col = "networks"

    if alpha is not None and alpha != 'None': alpha_text = str(int(float(alpha) * 100))

    refined_path = output_path + "network/modelling/%s/%s/%s/pathway_analysis/enrichr%sanalysis/refined/%s/%s/" \
                   % (modelling_approach, interactome_name, col_text, random_col, path_col, sample_col_text)

    file_name = "Refined_%s_%s_pathways" % (sample_col_text, pathway_db)

    pathway_df = pandas.read_csv(refined_path + file_name + ".txt", sep ="\t")

    return pathway_df


def plot_pathway(model_id, drug_name, prize, interactome_name, pathway_db,
                 modelling_approach, random_tag, alpha):
    """
    Retrieving the pathway table
    :param model_id: Sanger cell model id
    :param drug_name: Name of the single agent drug or drug combinations
    :param prize: Boolean - if the pathways of the prize nodes
    :param interactome_name : Name of the used interactome
    :param pathway_db: KEGG/Wiki/Reactome
    :param modelling_approach: PPR or PCSF
    :return:
    """

    if model_id is not None and drug_name is None:
        col_text = "cell_models"
        sample_col_text = model_id
    else:
        col_text = "drugs"
        if len(drug_name.split("/")) > 1:
            sample_col_text = "_".join(drug_name.split("/"))
        else:
            sample_col_text = drug_name

    if interactome_name == "omnipath":
        interactome_title = "Omnipath PPI"
    if interactome_name == "intact":
        interactome_title = "IntAct PPI"

    if prize:
        if model_id is not None:
            sub_title = "Context-Essential Gene Pathways"
        else:
            sub_title = "Drug Target Pathways"
    else:
        if model_id is not None:
            sub_title = "%s - Omics Integrator Sub-Networks" % interactome_title
        else:
            sub_title = "%s - PPR Drug Modules" % interactome_title

    save_col = output_path + "network/modelling/%s/%s/%s/figures/" % (modelling_approach, interactome_name, col_text)
    file_name = "%s_%s_%s_%s_pathway_plot" % (sample_col_text, modelling_approach, interactome_name, pathway_db)

    pathway_df = get_pathways(model_id=model_id, drug_name=drug_name, prize=prize, alpha=alpha,
                              interactome_name=interactome_name, pathway_db=pathway_db,
                              modelling_approach=modelling_approach, random_tag=random_tag)

    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5, "axes.facecolor": "0.7"})
    plt.figure(figsize=(30, 50))
    sns.scatterplot(data=pathway_df, x="ratio",
                    y="pathways", hue="adjpvalue", alpha=.7,
                    sizes=(20, 200), palette=sns.color_palette('dark:r_r', as_cmap = True))

    plt.ylabel("%s Pathways" % pathway_db, fontsize=10)
    plt.xlabel("Enrichment Ratio", fontsize=10)
    """
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=2)
    """
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5), ncol=1)
    plt.title(sub_title)
    plt.tight_layout()
    plt.savefig(save_col + file_name + ".pdf",  dpi=300)
    plt.savefig(save_col + file_name + ".jpg",  dpi=300)
    plt.savefig(save_col + file_name + ".png",  dpi=300)
    plt.close()

    return 1


def plot_benchmarking_pathways(cancer_model_list, cancer_model_text, interactome_name, pathway_db, modelling_type,
                               modelling_approach, alpha, random_tag, fdr_limit):

    if modelling_approach == "PCSF": col_text = "cell_models"
    else: col_text = "drugs"

    save_col = output_path + "network/modelling/%s/%s/%s/figures/" % (modelling_approach, interactome_name, col_text)
    file_name = "%s_%s_%s_%s_pathway_plot_benchmarking" % (modelling_approach, modelling_type, interactome_name, pathway_db)

    network_pathway_dfs = list()
    for model_id in cancer_model_list:
        network_pathways = dict()
        pathway_df = get_pathways(model_id=model_id, drug_name=drug_name, prize=prize, interactome_name=interactome_name,
                                  pathway_db=pathway_db, modelling_approach=modelling_approach,
                                  random_tag=random_tag, alpha=alpha)

        pathways = pathway_df["pathways"].unique()
        for path in pathways:
            if path not in network_pathways.keys():
                network_pathways[path] = \
                    {"Enrichment Ratio": pathway_df[pathway_df.pathways == path]["ratio"].values[0],
                     "Adjusted p-value": pathway_df[pathway_df.pathways == path]["adjpvalue"].values[0]}

        df = pandas.DataFrame.from_dict(network_pathways, orient="index")
        df["Sanger Model"] = model_id
        network_pathway_dfs.append(df)

    network_pathway_df = pandas.concat(network_pathway_dfs)
    network_pathway_df = network_pathway_df.reset_index()
    network_pathway_df["-log(Adj p-val)"] = network_pathway_df.apply(lambda x: numpy.log(x["Adjusted p-value"]) * -1, axis=1)
    network_pathway_df = network_pathway_df[network_pathway_df["Adjusted p-value"] < fdr_limit/100.0]

    network_pathway_df2 = network_pathway_df[["Sanger Model", "index", "Enrichment Ratio"]]
    network_pathway_df3 = network_pathway_df2.pivot_table(index="index", columns="Sanger Model", values="Enrichment Ratio")
    #network_pathway_df4 = network_pathway_df3.loc[network_pathway_df3.count().sort_values(ascending=False).index]
    network_pathway_df4 = network_pathway_df3.replace(numpy.nan, 0)

    """
    network_pathway_df5 = network_pathway_df4[network_pathway_df4.index.isin([
        "AMPK signaling pathway", "B cell receptor singaling pathway", "Base excision repair",
        "C-type lectin receptor signaling pathway", "ErbB signaling pathway", "Estrogen signaling pathway",
        "FoxO signaling pathway", "GnRH signaling pathway", "HIF-1 signaling pathway", "Hedgehog signaling pathway",
        "Hippo signaling pathway", "IL-17 signaling pathway", "JAK-STAT signaling pathway", "MAPK signaling pathway",
        "NF-kappa B signaling pathway", "Neurotrophin signaling pathway", "Notch signaling pathway", "Oxytocin signaling pathway",
        "PI3K-Akt signaling pathway", "PPAR signaling pathway", "Prolactin signaling pathway", "RIG-I-like receptor signaling pathway",
        "Rap1 signaling pathway", "Ras signaling pathway", "Relaxin signaling pathway", "T cell receptor signaling pathway",
        "TGF-beta signaling pathway", "TNF signaling pathway", "Thyroid hormone signaling pathway", "Toll-like receptor signaling pathway",
        "VEGF signaling pathway", "Wnt signaling pathway", "cAMP signaling pathway", "mTOR signaling pathway", "p53 signaling pathway"])]


    network_pathway_df5 = network_pathway_df4[network_pathway_df4.index.isin([
        "Base excision repair", "DNA replication", "HIF-1 signaling pathway", "Homologous recombination",
        "IL-17 signaling pathway", "Cell cycle", "TNF signaling pathway", "NF-kappa B signaling pathway", "PI3K-Akt signaling pathway",
        "MAPK signaling pathway", "T cell receptor signaling pathway"])]
    """
    # Plotting
    #mask = network_pathway_df4.isnull()
    mask = network_pathway_df4 ==0
    fig = plt.figure(figsize=(8, 40))
    ax = fig.subplots()
    ax = sns.heatmap(data=network_pathway_df4, linewidths=.2, linecolor="lightgrey",
                     cmap = 'Reds', xticklabels = True, mask=mask, cbar=False,
                     yticklabels=True, square=True)
    #flare
    ax = sns.clustermap(data=network_pathway_df4, linewidths=.2, linecolor="lightgrey",
                        cmap = 'Reds', xticklabels = True, mask=mask, col_cluster=False,
                        yticklabels=True, square=True)
    ax.ax_row_dendrogram.set_visible(False)
    #ax.set_aspect("equal")
    #ax.invert_yaxis()
    plt.xlabel("Cancer Cell Lines", fontsize=12)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel("KEGG Pathways", fontsize=12)
    plt.tight_layout()
    plt.title("Pathway Comparison of %s" % cancer_model_text, fontsize =12)
    plt.savefig("%s%s.pdf" % (save_col, file_name), dpi=300)
    plt.savefig("%s%s.png" % (save_col, file_name), dpi=300)
    plt.savefig("%s%s.jpg" % (save_col, file_name), dpi=300)
    return True



def create_cytockape_node_attributes(prize, network, model_id, parameter_combination,
                                     interactome_name, cancer_genes, network_type):
    if cancer_genes:
        title = "bagel_cancer_filtered"
    else:
        title = "bagel_not_filtered"

    if prize:
        sub_title = "terminal"
    if network:
        sub_title = "network_%s" % network_type

    if "%s_%s_%s_node_attributes.txt" % (model_id, sub_title, parameter_combination) not in os.listdir(
            output_path + "network/cytoscape/%s/node_attributes/%s/" %(interactome_name, title)):
        nodes = get_nodes(model_id=model_id, prize=prize, network=network,
                          parameter_combination=parameter_combination,
                          interactome_name=interactome_name,
                          cancer_genes=cancer_genes, network_type=network_type)

        # Add Node attributes
        node_attributes = pandas.read_csv(output_path + "network/interactome/omnipath/node_attributes.csv")
        node_attributes = node_attributes[["gene_symbol", "role"]].drop_duplicates()
        node_attributes = node_attributes.replace(numpy.nan, "protein")
        node_attributes = node_attributes[node_attributes.gene_symbol != "a_node"]
        node_attributes = node_attributes[node_attributes.gene_symbol.isin(nodes)]

        if interactome_name == "intact":
            not_annot_nodes = list(set(nodes).difference(set(node_attributes["gene_symbol"].unique())))
            if len(not_annot_nodes) >0:
                for n in not_annot_nodes:
                    node_attributes.append({n: "not_annotated"}, ignore_index=True)

        if prize:
            node_attributes["type"] = "terminal"

        if network:
            terminal_nodes = get_nodes(model_id=model_id, prize=True, network=False,
                              parameter_combination=None,
                              interactome_name=interactome_name,
                              cancer_genes=cancer_genes, network_type=network_type)
            node_attributes["type"] = node_attributes.apply(
                lambda x: "terminal" if x["gene_symbol"] in terminal_nodes else "steiner", axis=1)

        node_attributes.to_csv(output_path + "network/cytoscape/%s/node_attributes/%s/%s_%s_%s_node_attributes.txt"
                               %(interactome_name, title, model_id, sub_title, parameter_combination), sep="\t", index=False)

    else:
        node_attributes = pandas.read_csv(output_path + "network/cytoscape/%s/node_attributes/%s/%s_%s_%s_node_attributes.txt"
                                          %(interactome_name, title, model_id, sub_title, parameter_combination), sep="\t")
    return node_attributes


def create_cytockape_edge_attributes(model_id, parameter_combination,
                                     interactome_name, cancer_genes, network_type):
    if cancer_genes:
        title = "bagel_cancer_filtered"
    else:
        title = "bagel_not_filtered"

    sub_title = "network_%s" % network_type

    if "%s_%s_%s_edge_attributes.txt" % (model_id, sub_title, parameter_combination) not in os.listdir(
            output_path + "network/cytoscape/%s/edge_attributes/%s/" % (interactome_name, title)):

        # Add Node attributes
        edge_attributes = pandas.read_csv(output_path + "network/interactome/omnipath/edge_attributes.csv")
        edge_attributes = edge_attributes[edge_attributes.origin == "omnipath"]
        edge_attributes["int_role"] = edge_attributes.apply(
            lambda x: "inhibition" if x.is_inhibition == 1 and x.is_stimulation == 0 else (
                "stimulation" if x.is_inhibition == 0 and x.is_stimulation == 1 else (
                    "")), axis=1)
        edge_attributes = edge_attributes[["gene_from", "gene_to", "int_role"]]

        cl_interactome = pandas.read_csv(output_path + "network/interactome/%s/cell_line_specific/interactome_%s_%s.txt"
                                         % (interactome_name, interactome_name, model_id), sep="\t")
        cl_interactome = cl_interactome[(cl_interactome.From != "a_node") | (cl_interactome.To != "a_node")]
        cl_interactome.columns = ["gene_from", "gene_to", "weight"]

        edge_attributes = cl_interactome.merge(edge_attributes, how="left", on=["gene_from", "gene_to"]).drop_duplicates()
        edge_attributes["int_type"] = None

        w, d, b, mu = parameter_combination.split("_")[1], parameter_combination.split("_")[3],\
                      parameter_combination.split("_")[5], parameter_combination.split("_")[7]
        g = read_network_file(w, d, b, mu, model_id, interactome_name,
                              cancer_genes, network_type, False)
        for edge in g.edges:
            indx = edge_attributes[(edge_attributes.gene_from == edge[0]) &
                                   (edge_attributes.gene_to == edge[1])].index
            edge_attributes.loc[indx, "int_type"] = "steiner"

        edge_attributes = edge_attributes[edge_attributes.int_type == "steiner"]
        edge_attributes["pp"] = edge_attributes.apply(
            lambda x: "%s (pp) %s" % (x.gene_from, x.gene_to), axis=1)
        edge_attributes.to_csv(output_path + "network/cytoscape/%s/edge_attributes/%s/%s_%s_%s_edge_attributes.txt"
                               % (interactome_name, title, model_id, sub_title, parameter_combination), sep="\t", index=False)

    else:
        edge_attributes = pandas.read_csv(
            output_path + "network/cytoscape/%s/edge_attributes/%s/%s_%s_%s_edge_attributes.txt"
            % (interactome_name, title, model_id, sub_title, parameter_combination), sep="\t")
    return edge_attributes

def run_pathway_plot(model_id):

    for mu in [0.0, 0.005, 0.01]:
        for b in [2, 6, 10, 16]:
            for w in [10]:
                for d in [10]:
                    plot_pathway(prize=False, network=True, model_id=model_id,
                                 parameter_combination = "w_%s_D_%s_b_%s_mu_%s" % (str(w), str(d), str(b), str(mu)),
                                 interactome_name = "omnipath", cancer_genes=True,
                                 network_type="optimal", pathway_database="Reactome")
    plot_pathway(prize=True, network=False, model_id=model_id,
                 parameter_combination=None, interactome_name="omnipath", cancer_genes=True,
                 network_type="optimal", pathway_database="Reactome")

    return 1




def plot_whole_pathways(model_id, interactome_name, cancer_genes, network_type, pathway_database):
    """

    :param model_id:
    :param interactome_name:
    :param cancer_genes:
    :param network_type:
    :param pathway_database:
    :return:
    """

    if interactome_name == "omnipath":
        interactome_title = "Omnipath PPI"

    if cancer_genes:
        save_title = "bagel_cancer_filtered"
    else:
        save_title = "bagel_not_filtered"

    if network:
        save_col = "modelled/%s/%s" % (interactome_name, save_title)

    network_pathway_dfs = list()
    for mu in [0.0, 0.005, 0.01]:
        for b in [2, 6, 10, 16]:
            for w in [0.5, 2, 6, 10]:
                for d in [10]:
                    network_pathways = dict()
                    pathway_df = get_pathways(
                        model_id=model_id, prize=False, network=True,
                        parameter_combination="w_%s_D_%s_b_%s_mu_%s" % (str(w), str(d), str(b), str(mu)),
                        interactome_name=interactome_name, cancer_genes=cancer_genes,
                        pathway_database=pathway_database)

                    pathways = pathway_df["description"].unique()
                    for path in pathways:
                        if path not in network_pathways.keys():
                            network_pathways[path] = \
                                {"input": "w_%s_D_%s_b_%s_mu_%s" % (str(w), str(d), str(b), str(mu)),
                                 "input_type": "network",
                                 "enrichment": pathway_df[pathway_df.description == path]["enrichmentRatio"].values[0],
                                 "FDR": pathway_df[pathway_df.description == path]["FDR"].values[0]}

                    df = pandas.DataFrame.from_dict(network_pathways, orient="index")
                    network_pathway_dfs.append(df)

    network_pathway_df = pandas.concat(network_pathway_dfs)
    network_pathway_df2 = network_pathway_df.reset_index()
    """
    network_pathway_df2["-log(adj p)"] = network_pathway_df2.apply(lambda x: numpy.log(x.FDR) * -1, axis=1)
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5, "axes.facecolor": "0.5"})
    plt.figure(figsize=(40, 50))
    sns.scatterplot(data=network_pathway_df2, x="input",
                    y="index", size="-log(adj p)", hue="enrichment", alpha=.7,
                    sizes=(20, 200), palette=sns.color_palette('dark:r', as_cmap = True))
    plt.ylabel("%s Pathways" % pathway_database, fontsize=14)
    plt.xticks(rotation=90)
    plt.xlabel("Parameter Sets", fontsize=14)
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5), ncol=1)
    plt.title("%s - %s Pathways" % (model_id, pathway_database))
    plt.tight_layout()
    plt.savefig(output_path + "network/figures/pathway_analysis/webgestaltR/%s/%s_oi_parameter_comparison_optimal.pdf"
                % (save_col, model_id), dpi=150)
    plt.savefig(output_path + "network/figures/pathway_analysis/webgestaltR/%s/%s_oi_parameter_comparison_optimal.jpg"
                % (save_col, model_id), dpi=150)
    plt.savefig(output_path + "network/figures/pathway_analysis/webgestaltR/%s/%s_oi_parameter_comparison_optimal.png"
                % (save_col, model_id), dpi=150)
    plt.close()
    """
    network_pathway_df3 = network_pathway_df2[["index", "input", "enrichment"]]
    network_pathway_df4=network_pathway_df3.pivot_table(index="index", columns="input", values="enrichment")
    network_pathway_df5 = network_pathway_df5.loc[network_pathway_df4.count().sort_values(ascending=False).index]


    # Plotting
    mask = network_pathway_df5.isnull()
    fig = plt.figure(figsize=(60, 30))
    ax = fig.subplots()
    ax = sns.heatmap(data=network_pathway_df5, linewidths=.3, linecolor="lightgrey",
                     cmap = 'flare', xticklabels = True, mask=mask, cbar=False)
    """
    ax = sns.clustermap(data=network_pathway_df5, linewidths=.3, linecolor="lightgrey",
                        cmap = 'flare', xticklabels = True, mask=mask, col_cluster=False)
    ax.ax_row_dendrogram.set_visible(False)
    """
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.xlabel("Reactome Pathways", fontsize=14)
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    plt.ylabel("Parameter Sets", fontsize=14)
    plt.tight_layout()
    plt.title("SIDM00136 - Reactome Pathways")
    plt.savefig(output_path + "network/figures/pathway_analysis/webgestaltR/%s/%s_oi_%s_parameter_comparison_optimal.pdf"
                % ("modelled/omnipath/bagel_cancer_filtered", "SIDM00136", "Reactome"), dpi=150)
    plt.savefig(output_path + "network/figures/pathway_analysis/webgestaltR/%s/%s_oi_%s_parameter_comparison_optimal.jpg"
                % ("modelled/omnipath/bagel_cancer_filtered", "SIDM00136", "Reactome"), dpi=150)
    plt.savefig(output_path + "network/figures/pathway_analysis/webgestaltR/%s/%s_oi_%s_parameter_comparison_optimal.png"
                % ("modelled/omnipath/bagel_cancer_filtered", "SIDM00136", "Reactome"), dpi=150)



def extract_added_pathways():
    prize_pathways = list(get_pathways(model_id="SIDM00136", prize=True, network=False,
                                       parameter_combination=None, interactome_name="omnipath",
                                       cancer_genes=True, pathway_database="Reactome").description.unique())
    all_modelled_pathways = list(network_pathway_df.index)

    added =list(set(all_modelled_pathways).difference(set(prize_pathways)))



networkx.from_edgelist()
def cluster_graph():
    node_w = get_node_prizes(model_id="SIDM00136", cancer_genes=True)
    G = read_network_file(w, d, b, mu, model_id, interactome_name,
                          cancer_genes, network_type, directionality)
    g = igraph.Graph.from_networkx(G)
    comms = g.community_leiden(objective_function = "modularity")

    modules = dict()
    for name, membership in zip(g.vs["_nx_name"], comms.membership):
        if membership not in modules.keys():
            modules[membership] = [name]
        else:
            modules[membership].append(name)


    for module, gene_list in modules.items():
        if len(gene_list) > 10:
            module_pathways = gseapy.enrichr(
                gene_list=gene_list,
                gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human', 'Reactome_2022',
                           'TRRUST_Transcription_Factors_2019', 'Transcription_Factor_PPIs'],
                organism='Human', description='module %s' %module,
                outdir='test/enr_DEGs_Reactome_up',
                                 cutoff=0.5
                                 )




module_pathways = gseapy.enrichr(gene_list=list(G.nodes),gene_sets=["Transcription_Factor_PPIs"], organism="Human",
                                 cutoff=0.5,
                                 outdir="/Volumes/team215/Cansu/CombDrug/output/network/pathway_analysis/test/tf_ppi_ht29")

sidms = ["SIDM00136", "SIDM00795", "SIDM00083", "SIDM00272", "SIDM00903",
		 "SIDM00313", "SIDM00049", "SIDM00150", "SIDM00146"]

paths = dict()
all_paths= list()
path_df = pandas.DataFrame(columns=["sidm", "description", "enrichmentRatio", "FDR"])
for c in sidms:
    p = get_pathways(model_id=c, prize=False, network=True, parameter_combination="w_10_D_10_b_6_mu_0.005",
                     interactome_name="intact", cancer_genes=True, pathway_database="Reactome")
    if p is not None:
        p["sidm"] = c
        p = p[p.FDR < 0.05]
        p = p[["sidm", "description", "enrichmentRatio", "FDR"]]
        path_df = pandas.concat([path_df, p])
        paths[c] = list(p["description"].unique())
        for pp in list(p["description"].unique()):
            if pp not in all_paths:
                all_paths.append(pp)


hm = path_df[["sidm", "description", "enrichmentRatio"]]
hm_m = pandas.pivot_table(hm, index="sidm", values="enrichmentRatio", columns="description")
hm_m = hm_m.replace(numpy.nan, 0)

plt.figure(figsize=(40, 40))
a = sns.clustermap(hm_m, cmap="Reds", mask=(hm_m==0),
                   cbar_kws={"label": "Enrichment Ratio"})
a.ax_row_dendrogram.set_visible(False)
plt.xticks(fontsize=4)

plt.savefig(output_path + "network/pathway_analysis/webgestaltR/modelled/selected_cell_lines.png", dpi=300)


receptors = list(pandas.read_csv(output_path + "network/interactome/intact/receptors.txt", sep="\t", header=None)[0].values)
nodes = dict()
all_nodes= list()
for c in sidms:
    p = get_nodes(model_id=c, prize=False, network=True, parameter_combination="w_10_D_10_b_6_mu_0.005",
                     interactome_name="intact", cancer_genes=True, network_type="optimal")

    if p is not None:
        re = list()
        for n in p:
            if n in receptors:
                re.append(n)
        nodes[c] = re
        for n in re:
            if n not in all_nodes:
                all_nodes.append(n)


hm_r = pandas.DataFrame(0, columns = all_nodes, index=sidms)
for i, row in hm_r.iterrows():
    for c in hm_r.columns:
        if i in nodes.keys():
            if c in nodes[i]:
                hm_r.loc[i, c] = 1



a = sns.clustermap(hm_r, cmap="Reds", mask=(hm_r==0), linewidths=0.2, linecolor='darkgray')


plt.savefig(output_path + "network/pathway_analysis/webgestaltR/modelled/selected_cell_lines_receptor.png", dpi=300)




