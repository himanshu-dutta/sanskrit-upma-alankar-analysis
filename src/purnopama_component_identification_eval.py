import json
import argparse
from collections import defaultdict


def compare_components(a_components, b_components):
    matches = {
        key: a_components.get(key) == b_components.get(key) for key in b_components
    }
    exact_matches = sum(matches.values())
    total_keys = len(matches)
    return matches, exact_matches, total_keys


def calculate_metrics(a_file, b_file, output_file):
    with open(a_file, "r", encoding="utf-8") as file_a, open(
        b_file, "r", encoding="utf-8"
    ) as file_b:
        a_data = json.load(file_a)
        b_data = json.load(file_b)

    a_dict = {item["sentence"]: item for item in a_data}

    results = {
        "total_sentences": 0,
        "exact_match_count": 0,
        "component_matches": defaultdict(int),
        "overall_match_count": 0,
    }

    for b_item in b_data:
        sentence = b_item["sentence"]

        b_components = b_item.get("components", {})

        if sentence in a_dict:
            if a_dict[sentence]["human_label"] != "pūrṇopamā":
                continue
            a_components = a_dict[sentence].get("component_corr", {})
            matches, exact_matches, total_keys = compare_components(
                a_components, b_components
            )
            results["total_sentences"] += 1

            results["exact_match_count"] += exact_matches == total_keys
            results["overall_match_count"] += all(matches.values())

            for key, matched in matches.items():
                results["component_matches"][key] += matched

    # Calculate percentages
    results["exact_match_percentage"] = (
        results["exact_match_count"] / results["total_sentences"] * 100
    )
    results["overall_match_percentage"] = (
        results["overall_match_count"] / results["total_sentences"] * 100
    )
    for key in results["component_matches"]:
        results["component_matches"][key] = {
            "count": results["component_matches"][key],
            "percentage": results["component_matches"][key]
            / results["total_sentences"]
            * 100,
        }

    # Save results to JSON file
    with open(output_file, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare components in JSON files and generate metrics."
    )
    parser.add_argument("-a_file", type=str, help="Path to A.json")
    parser.add_argument("-b_file", type=str, help="Path to B.json")
    parser.add_argument(
        "-output_file", type=str, help="Path to save the results JSON file"
    )

    args = parser.parse_args()

    calculate_metrics(args.a_file, args.b_file, args.output_file)


