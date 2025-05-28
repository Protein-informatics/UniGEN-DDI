import torch
import pickle

SPLIT = "./split/fold2"
for split in ["train", "valid", "test"]:
    edges = torch.load(f"{SPLIT}/{split}.pt")
    edges["edge"] = edges["edge"].numpy()
    edges["edge_neg"] = edges["edge_neg"].numpy()
    print(edges)
    pickle.dump(edges, open(f"{SPLIT}/{split}.pkl", "wb"))
