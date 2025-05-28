import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Valid: {result[:, 0].max():.2f}")
            print(f"   Final Test: {result[argmax, 1]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Valid: {r.mean():.4f} ± {r.std():.4f}")
            r = best_result[:, 1]
            print(f"   Final Test: {r.mean():.4f} ± {r.std():.4f}")


if __name__ == "__main__":
    loggers = {
        "Precision": Logger(1),
        "Recall": Logger(1),
        "F1": Logger(1),
    }
    results = {
        "Precision": (0.1, 0.2),
        "Recall": (0.3, 0.4),
        "F1": (0.5, 0.6),
    }

    for key, result in results.items():
        loggers[key].add_result(0, result)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(0)
