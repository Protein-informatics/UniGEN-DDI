from typing import Dict, Any
import numpy as np
from sklearn.metrics import confusion_matrix

try:
    import torch
except ImportError:
    torch = None


### Evaluator for graph classification
class Evaluator:
    """
    Evaluation metrics grouped together.
    """

    def __init__(self, name: str):
        self.eval_metric = name

    def _parse_and_check_input(self, input_dict: Dict[str, Any]):
        """
        y_true: numpy ndarray or torch tensor of shape (num_nodes)
        y_pred: numpy ndarray or torch tensor of shape (num_nodes)
        """
        if self.eval_metric == "classification":
            if not "y_true" in input_dict:
                raise RuntimeError("Missing key of y_true")
            if not "y_pred" in input_dict:
                raise RuntimeError("Missing key of y_pred")

            y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError(
                    "Arguments to Evaluator need to be either numpy ndarray or torch tensor"
                )

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same")

            if not y_true.ndim == 1:
                raise RuntimeError(
                    "y_true and y_pred mush to 1-dim arrray, {}-dim array given".format(
                        y_true.ndim
                    )
                )
            return y_true, y_pred

        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    def eval(self, input_dict):
        if self.eval_metric == "classification":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_F1(y_true, y_pred)
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(
            self.eval_metric
        )
        if self.eval_metric == "classification":
            desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_nodes)\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_nodes)\n"
            desc += "where y_pred stores predicted class label (integer).\n"
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    def _eval_F1(self, y_true: np.ndarray, y_pred: np.ndarray):
        precision = 0
        recall = 0
        f1 = 0
        acc = 0

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            y_true, y_pred
        ).ravel()

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        if (true_positive + false_negative + true_negative + false_positive) > 0:
            acc = (true_positive + true_negative) / (
                true_positive + false_negative + true_negative + false_positive
            )
        else:
            acc = 0

        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Accuracy": round(acc, 4),
        }


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1])
    eval = Evaluator("classification")
    res = eval.eval({"y_true": y_true, "y_pred": y_pred})
    print(res)
