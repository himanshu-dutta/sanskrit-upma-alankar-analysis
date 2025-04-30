import json
import argparse
import random
import os


def load_json_file(filepath):
    """Load a JSON file containing a list of objects."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def sample_subsets(data, m, n):
    """Generate m subsets of size n from the data without replacement."""
    if len(data) < m * n:
        raise ValueError(
            "Not enough data to create m subsets of size n without replacement."
        )

    sampled_subsets = []
    for _ in range(m):
        subset = random.sample(data, n)
        sampled_subsets.append(subset)
        data = [item for item in data if item not in subset]

    return sampled_subsets


def save_subsets_to_files(subsets, output_dir):
    """Save each subset to a separate JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    for i, subset in enumerate(subsets):
        output_path = os.path.join(output_dir, f"subset_{i + 1}.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(subset, file, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Sample subsets from combined JSON files."
    )
    parser.add_argument("-file1", type=str, help="Path to the first JSON file.")
    parser.add_argument("-file2", type=str, help="Path to the second JSON file.")
    parser.add_argument("-m", type=int, help="Number of subsets to generate.")
    parser.add_argument("-n", type=int, help="Size of each subset.")
    parser.add_argument(
        "-output_dir", type=str, help="Directory to save the output JSON files."
    )

    args = parser.parse_args()

    # Load data from the JSON files
    data1 = load_json_file(args.file1)
    data2 = load_json_file(args.file2)

    # Combine the data from both files
    combined_data = data1 + data2

    # Sample subsets
    try:
        subsets = sample_subsets(combined_data, args.m, args.n)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Save the subsets to files
    save_subsets_to_files(subsets, args.output_dir)
    print(f"Successfully saved {args.m} subsets of size {args.n} to {args.output_dir}.")


if __name__ == "__main__":
    main()
