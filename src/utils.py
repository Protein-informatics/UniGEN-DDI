import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import pandas as pd
from torch_geometric.utils import to_undirected
from ogb.io.read_graph_pyg import read_graph_pyg
from tqdm import tqdm


class DNN(nn.Module):
    def __init__(self, inDim, hiddenDim, outDim, num_layers, dropout):
        super(DNN, self).__init__()
        self.dropout = dropout

        # attentation
        self.A1 = nn.Linear(
            inDim * 2,
            inDim,
        )
        nn.init.xavier_uniform_(self.A1.weight)

        self.A2 = nn.Linear(inDim, inDim)
        nn.init.uniform_(self.A2.weight)

        self.lins = nn.ModuleList()
        # fcc

        # id = inDim
        # od = 4096
        # for _ in range(num_layers - 1):
        #     self.lins.append(nn.Linear(id, od))
        #     self.lins.append(nn.BatchNorm1d(od))
        #     id = od
        #     od //= 2
        # self.lins.append(nn.Linear(od, outDim))

        self.lins.append(nn.Linear(inDim, 4096))
        self.lins.append(nn.BatchNorm1d(4096))

        self.lins.append(nn.Linear(4096, 2048))
        self.lins.append(nn.BatchNorm1d(2048))

        self.lins.append(nn.Linear(2048, 1024))
        self.lins.append(nn.BatchNorm1d(1024))

        self.lins.append(nn.Linear(1024, 512))
        self.lins.append(nn.BatchNorm1d(512))

        self.lins.append(nn.Linear(512, 256))
        self.lins.append(nn.BatchNorm1d(256))

        self.lins.append(nn.Linear(256, 128))
        self.lins.append(nn.BatchNorm1d(128))

        self.lins.append(nn.Linear(128, 64))
        self.lins.append(nn.BatchNorm1d(64))
        self.lins.append(nn.Linear(64, outDim))

    def reset_parameters(self):
        self.A1.reset_parameters()
        self.A2.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, s, d):
        h_cat = torch.cat([s, d], 1)

        att_proba = F.relu(self.A1(h_cat))
        att = F.softmax(self.A2(att_proba), 1)

        h = s * d * att
        for idx, lin in enumerate(self.lins):
            if idx % 2 == 0:
                h = F.relu(lin(h))
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = lin(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        # self.A1 = nn.Linear(
        #     inDim * 2,
        #     inDim,
        # )
        # nn.init.xavier_uniform_(self.A1.weight)
        #
        # self.A2 = nn.Linear(inDim, inDim)
        # nn.init.uniform_(self.A2.weight)

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # x_concat = torch.cat([x_i, x_j], dim=1)
        # att_proba = F.relu(self.A1(x_concat))
        # att = F.softmax(self.A2(att_proba), 1)

        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
        # return torch.sigmoid(x)


def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size, loader=None):
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(x.device)
    neg_train_edge = split_edge["train"]["edge_neg"].to(x.device)

    total_loss = total_examples = 0
    loss_func = torch.nn.BCEWithLogitsLoss()

    for perm in tqdm(
        DataLoader(range(pos_train_edge.size(1)), batch_size, shuffle=True)
    ):
        if loader == None:
            h = model(x, adj_t)
        else:
            h = torch.zeros(x.size())
            for batch in loader:
                adj_t = batch.edge_index.to(x.device)
                edge_type = batch.edge_types.to(x.device)
                bh = model(x[batch.n_id], adj_t, edge_type)
                h[batch.input_id] = bh[: batch.batch_size].detach().cpu()
            h = torch.tensor(h, requires_grad=True)

        h = h.to(x.device)
        edge = pos_train_edge[:, perm]  # [0, 2, 3] [31, 56, 78]
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_ones = torch.stack([torch.tensor([0.0, 1.0])] * edge.size(1)).to(x.device)
        pos_loss = loss_func(pos_out, pos_ones)

        edge = neg_train_edge[:, perm]
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_ones = torch.stack([torch.tensor([1.0, 0.0])] * edge.size(1)).to(x.device)
        neg_loss = loss_func(neg_out, neg_ones)

        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(x, 1.0)
        clip_grad_norm_(model.parameters(), 1.0)
        clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples


@torch.no_grad()
def test(
    model, predictor, x, edge_index, split_edge, evaluator, batch_size, loader=None
):
    model.eval()
    predictor.eval()

    if loader == None:
        h = model(x, edge_index)
    else:
        h = torch.zeros(x.size())
        for batch in loader:
            adj_t = batch.edge_index.to(x.device)
            edge_type = batch.edge_types.flatten().to(x.device)
            bh = model(x[batch.n_id], adj_t, edge_type)
            h[batch.input_id] = bh[: batch.batch_size].detach().cpu()
        h = h.to(x.device)

    pos_valid_edge = split_edge["valid"]["edge"].to(
        x.device
    )  # [0,0,0,0,....] [31, 34, 56, ....]
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)
    pos_test_edge = split_edge["test"]["edge"].to(x.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

    pos_valid_preds = []
    pos_valid_trues = []
    for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size * 64):
        edge = pos_valid_edge[:, perm]  # pos_valid_edge [:, [3, 5, 6]]
        preds = (
            torch.argmax(torch.sigmoid(predictor(h[edge[0]], h[edge[1]])), dim=1)
            .squeeze()
            .cpu()
        )
        pos_valid_preds += [preds]
        pos_valid_trues += [torch.ones_like(preds)]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    pos_valid_true = torch.cat(pos_valid_trues, dim=0)

    neg_valid_preds = []
    neg_valid_trues = []
    for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size * 64):
        edge = neg_valid_edge[:, perm]
        preds = (
            torch.argmax(torch.sigmoid(predictor(h[edge[0]], h[edge[1]])), dim=1)
            .squeeze()
            .cpu()
        )
        neg_valid_preds += [preds]
        neg_valid_trues += [torch.zeros_like(preds)]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    neg_valid_true = torch.cat(neg_valid_trues, dim=0)

    pos_test_preds = []
    pos_test_trues = []
    for perm in DataLoader(range(pos_test_edge.size(1)), batch_size * 64):
        edge = pos_test_edge[:, perm]
        preds = (
            torch.argmax(torch.sigmoid(predictor(h[edge[0]], h[edge[1]])), dim=1)
            .squeeze()
            .cpu()
        )
        pos_test_preds += [preds]
        pos_test_trues += [torch.ones_like(preds)]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    pos_test_true = torch.cat(pos_test_trues, dim=0)

    neg_test_preds = []
    neg_test_trues = []
    for perm in DataLoader(range(neg_test_edge.size(1)), batch_size * 64):
        edge = neg_test_edge[:, perm]
        preds = (
            torch.argmax(torch.sigmoid(predictor(h[edge[0]], h[edge[1]])), dim=1)
            .squeeze()
            .cpu()
        )
        neg_test_preds += [preds]
        neg_test_trues += [torch.zeros_like(preds)]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    neg_test_true = torch.cat(neg_test_trues, dim=0)
    # print(f"{torch.cat([pos_test_pred, neg_test_pred])}")

    results = {}
    for K in ["Precision", "Recall", "F1", "Accuracy"]:
        valid_metric = evaluator.eval(
            {
                "y_true": torch.cat([pos_valid_true, neg_valid_true], dim=0),
                "y_pred": torch.cat([pos_valid_pred, neg_valid_pred], dim=0),
            }
        )[K]
        test_metric = evaluator.eval(
            {
                "y_true": torch.cat([pos_test_true, neg_test_true], dim=0),
                "y_pred": torch.cat([pos_test_pred, neg_test_pred], dim=0),
            }
        )[K]

        results[K] = (valid_metric, test_metric)
    return results


def get_dataset():
    meta_data = {
        "eval metric": "hits@20",  # "classification"
        "task type": "mrr",
        "download_name": "ogbl_dkp",
        "version": "1",
        "url": "https://github.com/deba-iitbh/datasets/raw/main/ogbl_dkps.zip",
        "add_inverse_edge": "False",
        "has_node_attr": "False",
        "has_edge_attr": "False",
        "split": "body",
        "additional node files": "None",
        "additional edge files": "edge_types",
        "is hetero": "False",
        "binary": "False",
        "dir_path": "input/ogbl_dkp",
    }
    dataset = PygLinkPropPredDataset(name="ogbl-dkp", root="input", meta_dict=meta_data)
    return dataset


def train_test_split_edges(data, sr=1):
    assert data.num_nodes is not None
    assert data.edge_index is not None

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    # Positive edges.
    zero_indices = (edge_attr == 0).nonzero()
    one_indices = (edge_attr == 1).nonzero()

    train_tensor = zero_indices.flatten()
    test_tensor = one_indices.flatten()

    r, c = row[test_tensor], col[test_tensor]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[train_tensor], col[train_tensor]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[train_tensor])
        data.train_pos_edge_index, _ = out

    # Negative edges.
    n_t = test_tensor.size(0) * sr
    n_tr = train_tensor.size(0) * sr

    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0) * sr)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    # assigning neg edges
    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
    row, col = neg_row[n_t : n_t + n_tr], neg_col[n_t : n_t + n_tr]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)
    return data


def get_train_test_data(fold):
    edge_type = pd.read_csv("./input/ogbl-dkp/mapping/edge_with_type.txt")
    edge_type["label"] = 0
    test_idx = edge_type.query(f"(label1 == {fold}) | (label2 == {fold})").index
    edge_type.loc[test_idx, "label"] = 1
    edge_type["label"].to_csv(
        "./input/ogbl_dkp/ddi/edge-feat.csv.gz",
        header=None,
        compression="gzip",
        index=False,
    )
    data = read_graph_pyg(
        "../input/ogbl_dkb/ddi/",
        add_inverse_edge=True,
        additional_node_files=[],
        additional_edge_files=[],
        binary=False,
    )[0]
    data = train_test_split_edges(data)
    split = {
        "train": {
            "edge": data["train_pos_edge_index"],
            "edge_neg": data["train_neg_edge_index"],
        },
        "test": {
            "edge": data["test_pos_edge_index"],
            "edge_neg": data["test_neg_edge_index"],
        },
    }
    return split
