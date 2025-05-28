import json
import pandas as pd
import numpy as np

with open("../input/graph.json", "r") as f:
    graph = json.load(f)

# 'interacts_with': 0,
# 'has_targets': 1,
# 'is_metabolized_by_the_enzymes': 2,
# 'has_the_pathways': 3

drugs = set()
targets = set()
enzymes = set()
pathways = set()
for g in graph:
    drug = list(g.keys())[0]
    drugs.add(drug)
    for k, val in g[drug].items():
        if k == "interacts_with":
            drugs.update(val)
        elif k == "has_the_pathways":
            pathways.update(val)
        elif k == "has_targets":
            targets.update(val)
        elif k == "is_metabolized_by_the_enzymes":
            enzymes.update(val)
        else:
            print(f"{k} is an unexpected relation")
print(*[len(t) for t in [drugs, targets, enzymes, pathways]], sep="\n")
proteins = targets.union(enzymes)
print(len(proteins))

drug_mapping = {d: idx for idx, d in enumerate(drugs)}
protein_mapping = {d: idx for idx, d in enumerate(proteins)}
pathway_mapping = {d: idx for idx, d in enumerate(pathways)}
drug_mapping_df = pd.DataFrame(
    list(drug_mapping.items()), columns=["ent idx", "ent name"]
)
drug_mapping_df.to_csv(
    "../input/ogbl-cddi/mapping/drug_entidx2name.csv.gz", index=False
)

protein_mapping_df = pd.DataFrame(
    list(protein_mapping.items()), columns=["ent idx", "ent name"]
)
protein_mapping_df.to_csv(
    "../input/ogbl-cddi/mapping/protein_entidx2name.csv.gz", index=False
)

pathway_mapping_df = pd.DataFrame(
    list(pathway_mapping.items()), columns=["ent idx", "ent name"]
)
pathway_mapping_df.to_csv(
    "../input/ogbl-cddi/mapping/pathway_entidx2name.csv.gz", index=False
)
data = []
edge_mapping = {
    "interacts_with": 0,
    "has_targets": 1,
    "is_metabolized_by_the_enzymes": 2,
    "has_the_pathways": 3,
}

for g in graph:
    drug = list(g.keys())[0]
    for k, val in g[drug].items():
        for v in val:
            if k == "interacts_with":
                dst = drug_mapping[v]
            elif k == "has_the_pathways":
                dst = pathway_mapping[v]
            elif k == "has_targets":
                dst = protein_mapping[v]
            elif k == "is_metabolized_by_the_enzymes":
                dst = protein_mapping[v]
            else:
                print(f"{k} is an unexpected relation")
                exit(-1)
            data.append(
                {"src": drug_mapping[drug], "dst": dst, "label": edge_mapping[k]}
            )
edge_df = pd.DataFrame(data)
for l, df in edge_df.groupby("label"):
    # Sort the values in each row to make sure interchangeables are considered duplicates
    if l == 0:
        df[["src", "dst"]] = np.sort(df[["src", "dst"]], axis=1)
        df.drop_duplicates(inplace=True)
        df[["src", "dst"]].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___interacts_with___drug/edge.csv.gz",
            index=False,
            header=False,
        )
        df["label"].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___interacts_with___drug/edge_reltype.csv.gz",
            index=False,
            header=False,
        )
        sz = pd.DataFrame([df.shape[0]], columns=["size"])
        sz.to_csv(
            "../input/ogbl-cddi/raw/relations/drug___interacts_with___drug/num-edge-list.csv.gz",
            index=False,
            header=False,
        )
    elif l == 1:
        df[["src", "dst"]].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___has_target___target/edge.csv.gz",
            index=False,
            header=False,
        )
        df["label"].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___has_target___target/edge_reltype.csv.gz",
            index=False,
            header=False,
        )
        sz = pd.DataFrame([df.shape[0]], columns=["size"])
        sz.to_csv(
            "../input/ogbl-cddi/raw/relations/drug___has_target___target/num-edge-list.csv.gz",
            index=False,
            header=False,
        )
    elif l == 2:
        df[["src", "dst"]].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___is_metabolized_by_enzyme___enzyme/edge.csv.gz",
            index=False,
            header=False,
        )
        df["label"].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___is_metabolized_by_enzyme___enzyme/edge_reltype.csv.gz",
            index=False,
            header=False,
        )
        sz = pd.DataFrame([df.shape[0]], columns=["size"])
        sz.to_csv(
            "../input/ogbl-cddi/raw/relations/drug___is_metabolized_by_enzyme___enzyme/num-edge-list.csv.gz",
            index=False,
            header=False,
        )
    elif l == 3:
        df[["src", "dst"]].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___has_pathway___pathway/edge.csv.gz",
            index=False,
            header=False,
        )
        df["label"].to_csv(
            "../input/ogbl-cddi/raw/relations/drug___has_pathway___pathway/edge_reltype.csv.gz",
            index=False,
            header=False,
        )
        sz = pd.DataFrame([df.shape[0]], columns=["size"])
        sz.to_csv(
            "../input/ogbl-cddi/raw/relations/drug___has_pathway___pathway/num-edge-list.csv.gz",
            index=False,
            header=False,
        )
    else:
        print(f"{l} is not an encoding of edge")
        exit(-1)
