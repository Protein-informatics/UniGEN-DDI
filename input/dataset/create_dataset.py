import pandas as pd

# comment out edges you don't want to use
edges = [
    "./edges/drug_path.txt",
    "./edges/drug_enzyme.txt",
    "./edges/drug_target.txt",
    # "./edges/drug_prop_similarity.txt",
    "./edges/drug_substructure.txt",
]


nodes: pd.DataFrame = pd.read_csv(
    "./nodes/node_id.txt", sep=" ", names=["node_id", "node_name"]
)

# join all the files together
for i, file_name in enumerate(edges):
    if i == 0:
        edges = pd.read_csv(file_name, sep=" ", names=["node_i", "node_j"])
        edges.drop_duplicates(inplace=True)
        labels = pd.DataFrame({"label": [i] * edges.shape[0]})
    else:
        temp = pd.read_csv(file_name, sep=" ", names=["node_i", "node_j"])
        temp.drop_duplicates(inplace=True)
        edges = pd.concat([edges, temp])
        labels = pd.concat([labels, pd.DataFrame({"label": [i] * temp.shape[0]})])

num_nodes = pd.DataFrame({"data": [nodes.shape[0]]})
num_edges = pd.DataFrame({"data": [edges.shape[0]]})
edges.to_csv(
    "../ogbl_dkp/raw/edge.csv.gz", index=False, header=False, compression="gzip"
)
labels.to_csv(
    "../ogbl_dkp/raw/edge_label.csv.gz",
    index=False,
    header=False,
    compression="gzip",
)
num_nodes.to_csv(
    "../ogbl_dkp/raw/num-node-list.csv.gz",
    index=False,
    header=False,
    compression="gzip",
)
num_edges.to_csv(
    "../ogbl_dkp/raw/num-edge-list.csv.gz",
    index=False,
    header=False,
    compression="gzip",
)

################ Drug Drug Interaction #############

drug_edges = ["./edges/drug_drug.txt"]

nodes: pd.DataFrame = pd.read_csv(
    "./nodes/drug_id.txt", sep=" ", names=["node_id", "node_name"]
)

# join all the files together
for i, file_name in enumerate(drug_edges):
    if i == 0:
        edges = pd.read_csv(file_name, sep=" ", names=["node_i", "node_j"])
    else:
        edges = pd.concat(
            [edges, pd.read_csv(file_name, sep=" ", names=["node_i", "node_j"])]
        )

num_nodes = pd.DataFrame({"data": [nodes.shape[0]]})
num_edges = pd.DataFrame({"data": [edges.shape[0]]})
edges.to_csv(
    "../ogbl_dkp/ddi/edge.csv.gz", index=False, header=False, compression="gzip"
)
num_nodes.to_csv(
    "../ogbl_dkp/ddi/num-node-list.csv.gz",
    index=False,
    header=False,
    compression="gzip",
)
num_edges.to_csv(
    "../ogbl_dkp/ddi/num-edge-list.csv.gz",
    index=False,
    header=False,
    compression="gzip",
)
