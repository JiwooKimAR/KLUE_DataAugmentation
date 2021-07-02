from scipy.stats import pearsonr
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from seqeval.metrics import classification_report

import datasets


class KLUE(datasets.Metric):
    def _info(self):
        if self.config_name not in [
            "ynat", "sts", "nli", "ner", "re", "dp", "mrc", "wos",
        ]:
            raise KeyError("No such dataset in KLUE")
        return datasets.MetricInfo(
            description="KLUE metric",
            citation="citation",
            inputs_description="predictions, references",
            features=datasets.Features({
                # https://github.com/huggingface/datasets/blob/master/metrics/glue/glue.py
                "predictions": datasets.Value("int64") if self.config_name != "ner" else datasets.Sequence(datasets.Value("string")),
                "references": datasets.Value("int64") if self.config_name != "ner" else datasets.Sequence(datasets.Value("string")),
            }),
        )

    def _compute(self, predictions, references):
        if self.config_name == "ynat":
            return {
                "f1": f1_score(y_true=references, y_pred=predictions, average="macro")
            }
        elif self.config_name == "sts":
            return {
                "pearson": pearsonr(predictions, references)[0],
                "f1": f1_score(y_true=references, y_pred=predictions, average="macro") # TODO: binary
            }
        elif self.config_name == "nli":
            return {
                "accuracy": accuracy_score(y_true=references, y_pred=predictions)
            }
        elif self.config_name == "ner":
            classes = ["DT", "IT", "LC", "LS", "OG", "PS", "QT", "TI"]
            report = classification_report(y_pred=predictions, y_true=references, output_dict=True, mode=None)

            report.pop("macro avg")
            report.pop("weighted avg")
            overall_score = report.pop("micro avg")
            scores = {
                type_name: {
                    "entity_f1": score["f1-score"],
                    "number": score["support"]
                }
                for type_name, score in report.items()
            }
            
            entity_f1 = 0.0
            for type_name, score in report.items():
                entity_f1 += score["f1-score"]
            entity_f1 /= len(scores)
            scores["overall_f1"] = overall_score["f1-score"]

            return {
                "entity_f1": entity_f1
            }
        elif self.config_name == "re":
            return {
                "f1": f1_score(y_true=references, y_pred=predictions, average="micro"),
                #"AUPRC": average_precision_score(y_true=references, y_score=predictions, average="micro")
            }
        elif self.config_name == "dp":
            return {
                "TODO": 1
            }
        elif self.config_name == "mrc":
            return {
                "TODO": 1
            }
        elif self.config_name== "wos":
            return {
                "TODO": 1
            }
        else:
            raise KeyError("No such dataset in KLUE")
