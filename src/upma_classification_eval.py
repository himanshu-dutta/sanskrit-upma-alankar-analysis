import json
import argparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def main(json_path: str, results_path: str):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    predicted = [itm["label"] for itm in data]
    actual = [itm["human_label"] for itm in data]
    is_reasoning_correct = [int(itm["is_reasoning_correct"]) for itm in data]

    acc = accuracy_score(actual, predicted)
    prec = precision_score(actual, predicted, average="weighted")
    rec = recall_score(actual, predicted, average="weighted")
    f1 = f1_score(actual, predicted, average="weighted")

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_corr_reasoning": sum(is_reasoning_correct) / len(data),
    }

    print(classification_report(actual, predicted))

    with open(results_path, "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-r", "--results", type=str, required=True)
    args = parser.parse_args()

    main(args.file, args.results)
