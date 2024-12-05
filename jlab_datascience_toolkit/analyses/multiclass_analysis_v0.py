import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, configs: dict):
        self.configs = configs

    def run(
        self,
        y_true,
        y_pred,
        labels: np.ndarray = None,
        target_names: np.ndarray = None,
        sample_weight: np.ndarray = None,
        logdir: str = None,
    ) -> list:
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
        ans = []
        for submodule in self.configs["submodules"]:
            submodule_type = submodule["type"]
            submodule_configs = submodule.get("configs", {})
            if submodule_type == "confusion_matrix":
                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=labels,
                    sample_weight=sample_weight,
                    **submodule_configs,
                )
                ans.append(cm)
                if logdir is not None:
                    np.save(os.path.join(logdir, "confusion_matrix.npy"), cm)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
                    disp.plot()
                    plt.tight_layout()
                    plt.savefig(os.path.join(logdir, "confusion_matrix.jpg"), transparent=True, dpi=300)
                    plt.close()
            elif submodule_type == "accuracy_score":
                acc = accuracy_score(
                    y_true, y_pred, sample_weight=sample_weight, **submodule_configs
                )
                ans.append(acc)
                if logdir is not None:
                    np.save(os.path.join(logdir, "accuracy_score.npy"), acc)
            elif submodule_type == "classification_report":
                cr = classification_report(
                    y_true,
                    y_pred,
                    labels=labels,
                    target_names=target_names,
                    sample_weight=sample_weight,
                    **submodule_configs,
                )
                ans.append(cr)
                if (logdir is not None) and isinstance(cr, dict):
                    for metric in ["precision", "recall", "f1-score"]:
                        metric_list = []
                        for k, v in cr.items():
                            if isinstance(v, dict) and (metric in v.keys()):
                                metric_list.append((k, v[metric]))
                        # save metric_list as a bar chart
                        fig, ax = plt.subplots()
                        ax.bar(
                            [tup[0] for tup in metric_list],
                            [tup[1] for tup in metric_list],
                        )
                        ax.set_title(metric)
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(logdir, f"{metric}.jpg"),
                            transparent=True,
                            dpi=300,
                        )
                        plt.close(fig=fig)
            else:
                raise NameError(
                    "Unsupported submodule type in Multi-Class Analysis Module !"
                )
        return ans
