import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class Analysis:
    def __init__(self, configs: dict):
        self.configs = configs

    def run(self, y_true, y_pred, labels: np.ndarray = None, sample_weight: np.ndarray = None) -> list:
        ans = []
        for submodule in self.configs["submodules"]:
            submodule_type = submodule["type"]
            submodule_configs = submodule.get("configs", {})
            if submodule_type == "confusion_matrix":
                ans.append(
                    confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight, **submodule_configs)
                )
            elif submodule_type == "accuracy_score":
                ans.append(
                    accuracy_score(y_true, y_pred, sample_weight=sample_weight, **submodule_configs)
                )
            elif submodule_type == "precision_score":
                ans.append(
                    precision_score(y_true, y_pred, labels=labels, sample_weight=sample_weight, **submodule_configs)
                )
            elif submodule_type == "recall_score":
                ans.append(
                    recall_score(y_true, y_pred, labels=labels, sample_weight=sample_weight, **submodule_configs)
                )
            elif submodule_type == "f1_score":
                ans.append(
                    f1_score(y_true, y_pred, labels=labels, sample_weight=sample_weight, **submodule_configs)
                )
            else:
                raise NameError('Unsupported submodule type in Multi-Class Analysis Module !')
        return ans