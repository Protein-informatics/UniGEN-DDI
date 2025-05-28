import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from eval import Evaluator
from logger import Logger
from utils import train, test, get_dataset, LinkPredictor


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def main():
    parser = argparse.ArgumentParser(description="OGBL-DKP (GNN)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--use_sage", action="store_true")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="output/gnn")
    args = parser.parse_args()
    print(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = get_dataset()
    data = dataset[0]
    adj_t = data["edge_index"].to(device)
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)

    if args.use_sage:
        model = SAGE(
            args.hidden_channels,
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        ).to(device)
    else:
        model = GCN(
            args.hidden_channels,
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        ).to(device)

    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 2, args.num_layers, args.dropout
    ).to(device)

    evaluator = Evaluator("classification")
    loggers = {
        "Precision": Logger(args.runs, args),
        "Recall": Logger(args.runs, args),
        "F1": Logger(args.runs, args),
        "Accuracy": Logger(args.runs, args),
    }

    for run in range(args.runs):
        best_f1 = 0
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(emb.parameters())
            + list(predictor.parameters()),
            lr=args.lr,
        )

        for epoch in range(1, 1 + args.epochs):
            loss = train(
                model,
                predictor,
                emb.weight,
                adj_t,
                split_edge,
                optimizer,
                args.batch_size,
            )

            if epoch % args.eval_steps == 0:
                results = test(
                    model,
                    predictor,
                    emb.weight,
                    adj_t,
                    split_edge,
                    evaluator,
                    args.batch_size,
                )
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                print(loggers["F1"].results[run])
                if loggers["F1"].results[run][0][0] > best_f1:
                    best_f1 = loggers["F1"].results[run][0][0]
                    if args.use_sage:
                        torch.save(
                            model.state_dict(), f"{args.model_dir}/sage_{run}.pth"
                        )
                        torch.save(
                            predictor.state_dict(),
                            f"{args.model_dir}/sage_predictor_{run}.pth",
                        )
                    else:
                        torch.save(
                            model.state_dict(), f"{args.model_dir}/gnn_{run}.pth"
                        )
                        torch.save(
                            predictor.state_dict(),
                            f"{args.model_dir}/gnn_predictor_{run}.pth",
                        )

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Valid: {100 * valid_hits:.2f}%, "
                            f"Test: {100 * test_hits:.2f}%"
                        )
                    print("---")

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
