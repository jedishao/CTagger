import argparse
from pathlib import Path


def merge_files(input_dir1, input_dir2, output_dir, selected_patterns):
    pattern_indices = {"KW": 0, "PH": 1, "SE": 2, "BR": 3}

    for file_path in input_dir1.iterdir():
        if file_path.is_file():
            file_name = file_path.name
            input_file2 = input_dir2 / file_name
            temp = []

            if input_file2.exists():
                with input_file2.open(encoding='utf-8') as file:
                    lines = file.readlines()
                    for pattern in selected_patterns:
                        if pattern in pattern_indices and pattern_indices[pattern] < len(lines):
                            temp.append(lines[pattern_indices[pattern]])
                    if len(lines) > 4:
                        temp.append(lines[4])

            temp.append("[Bug Report:]\n")
            temp.append(file_path.read_text())

            output_file = output_dir / file_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file.write_text(''.join(temp), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Merge linguistic pattern and bug report files by pattern combination.")

    parser.add_argument("--br_dir", type=str, required=True, help="Path to the bug report directory (e.g., ./br).")
    parser.add_argument("--patterns_dir", type=str, required=True, help="Path to the extracted linguistic patterns directory (e.g., ./patterns).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save merged output files (e.g., ./lps_results).")
    parser.add_argument("--pattern", type=str, default=None)

    args = parser.parse_args()

    pattern_names = [
        "KW", "PH", "SE", "BR",
        "KW+PH", "KW+SE", "KW+BR", "PH+SE", "PH+BR", "SE+BR",
        "KW+PH+SE", "KW+PH+BR", "KW+SE+BR", "PH+SE+BR", "ALL"
    ]

    selected_patterns_list = [args.pattern] if args.pattern else pattern_names

    print(f"patterns = {', '.join(selected_patterns_list)}")
    print("=====================================")

    for pattern in selected_patterns_list:
        if pattern == "ALL":
            selected_patterns = "KW+PH+SE+BR".split("+")
        else:
            selected_patterns = pattern.split("+")

        for subdir in ['con', 'non']:
            input_dir1 = Path(f'{args.br_dir}/{subdir}')
            input_dir2 = Path(f'{args.patterns_dir}/{subdir}')
            output_dir = Path(f'{args.output_dir}/{pattern}/{subdir}')

            merge_files(input_dir1, input_dir2, output_dir, selected_patterns)

    print("\n✅ Finished processing all pattern combinations!\n")


if __name__ == "__main__":
    main()
