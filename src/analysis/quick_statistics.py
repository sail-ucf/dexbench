import json
import numpy as np
import os
import re
import argparse

def count_all_lines(filepath: str) -> int:
    """Count ALL lines in a file exactly"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        return len(content.split('\n'))
    except:
        return 0

def get_cruxeval_filepath(program_num: str, cruxeval_dir: str) -> str:
    """Get the filepath for a CRUXEval program"""
    return os.path.join(
        cruxeval_dir,
        f"sample_{program_num}.py"
    )

def get_humaneval_filepath(program_num: str, runner_programs_dir: str) -> str:
    """Get the filepath for a HumanEval program"""
    return os.path.join(
        runner_programs_dir,
        f"HumanEval_{program_num}",
        "solution.py"
    )

def get_pythonsaga_filepath(program_num: str, runner_programs_dir: str) -> str:
    """Get the filepath for a PythonSaga program"""
    return os.path.join(
        runner_programs_dir,
        f"PythonSaga_{program_num}",
        "solution.py"
    )
    
def calculate_accurate_statistics(json_file_path: str,cruxeval_dir: str,runner_programs_dir: str):
    """Calculate accurate statistics by reading actual source files"""

    print("=" * 60)
    print("ACCURATE STATISTICS FROM SOURCE FILES")
    print("=" * 60)

    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found!")
        return

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        total_programs = len(data)
        print(f"\nTotal programs in JSON: {total_programs}")


        dataset_counts = {}
        for item in data:
            dataset = item.get('dataset', 'Unknown')
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        print("\nDataset distribution:")
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} programs")


        loc_values = []
        cc_values = []


        min_loc = float('inf')
        max_loc = 0
        min_loc_id = ""
        max_loc_id = ""

        min_cc = float('inf')
        max_cc = 0
        min_cc_id = ""
        max_cc_id = ""


        missing_files = []
        found_files = 0

        print(f"\nReading actual source files...")

        for idx, item in enumerate(data):
            dataset = item.get('dataset', 'Unknown')
            task_id = item.get('task_id', f'item_{idx}')
            provided_cc = item.get('complexity', 0)

            filepath = None
            loc = 0

            try:
                if dataset == "CRUXEval":
                    match = re.search(r'CRUXEval/(\d+)', task_id)
                    if match:
                        program_num = match.group(1)
                        filepath = get_cruxeval_filepath(program_num, cruxeval_dir)

                elif dataset == "HumanEval":
                    match = re.search(r'HumanEval/(\d+)', task_id)
                    if match:
                        program_num = match.group(1)
                        filepath = get_humaneval_filepath(program_num, runner_programs_dir)

                elif dataset == "PythonSaga":
                    match = re.search(r'PythonSaga/(\d+)', task_id)
                    if match:
                        program_num = match.group(1)
                        filepath = get_pythonsaga_filepath(program_num, runner_programs_dir)


                if filepath and os.path.exists(filepath):
                    loc = count_all_lines(filepath)
                    loc_values.append(loc)
                    found_files += 1


                    if loc < min_loc:
                        min_loc = loc
                        min_loc_id = task_id
                    if loc > max_loc:
                        max_loc = loc
                        max_loc_id = task_id
                else:
                    missing_files.append((task_id, filepath))
                    continue


                if isinstance(provided_cc, (int, float)):
                    cc_values.append(provided_cc)


                    if provided_cc < min_cc:
                        min_cc = provided_cc
                        min_cc_id = task_id
                    if provided_cc > max_cc:
                        max_cc = provided_cc
                        max_cc_id = task_id

            except Exception as e:
                missing_files.append((task_id, str(e)))
                continue


        print(f"\nFound {found_files}/{total_programs} source files")
        if missing_files and len(missing_files) <= 10:
            print(f"\nMissing files (first {len(missing_files)}):")
            for task_id, filepath in missing_files[:10]:
                print(f"  {task_id}: {filepath}")
        elif missing_files:
            print(f"\nMissing {len(missing_files)} files")

        if not loc_values:
            print("\nNo source files could be read!")
            return


        avg_loc = np.mean(loc_values)
        median_loc = np.median(loc_values)
        std_loc = np.std(loc_values, ddof=1) if len(loc_values) >= 2 else 0

        avg_cc = np.mean(cc_values)
        median_cc = np.median(cc_values)
        std_cc = np.std(cc_values, ddof=1) if len(cc_values) >= 2 else 0


        loc_sorted = sorted(loc_values)
        q1_loc = loc_sorted[len(loc_sorted) // 4] if len(loc_sorted) >= 4 else median_loc
        q3_loc = loc_sorted[3 * len(loc_sorted) // 4] if len(loc_sorted) >= 4 else median_loc

        cc_sorted = sorted(cc_values)
        q1_cc = cc_sorted[len(cc_sorted) // 4] if len(cc_sorted) >= 4 else median_cc
        q3_cc = cc_sorted[3 * len(cc_sorted) // 4] if len(cc_sorted) >= 4 else median_cc

        print(f"\n{'='*60}")
        print("ACCURATE RESULTS (FROM SOURCE FILES)")
        print("=" * 60)

        print(f"\nLines of Code (LOC) Statistics:")
        print(f"  Average LOC: {avg_loc:.2f}")
        print(f"  Standard Deviation: {std_loc:.2f}")
        print(f"  Minimum LOC: {min_loc} (program: {min_loc_id})")
        print(f"  Maximum LOC: {max_loc} (program: {max_loc_id})")
        print(f"  Range: {max_loc - min_loc} lines")
        print(f"  Median: {median_loc} lines")
        print(f"  25th Percentile (Q1): {q1_loc} lines")
        print(f"  75th Percentile (Q3): {q3_loc} lines")
        print(f"  Interquartile Range: {q3_loc - q1_loc} lines")

        print(f"\nCyclomatic Complexity (CC) Statistics:")
        print(f"  Average CC: {avg_cc:.2f}")
        print(f"  Standard Deviation: {std_cc:.2f}")
        print(f"  Minimum CC: {min_cc} (program: {min_cc_id})")
        print(f"  Maximum CC: {max_cc} (program: {max_cc_id})")
        print(f"  Range: {max_cc - min_cc}")
        print(f"  Median: {median_cc}")
        print(f"  25th Percentile (Q1): {q1_cc}")
        print(f"  75th Percentile (Q3): {q3_cc}")
        print(f"  Interquartile Range: {q3_cc - q1_cc}")


        print(f"\n{'='*60}")
        print("VERIFICATION - HumanEval/83:")
        print("=" * 60)


        match = re.search(r'HumanEval/(\d+)', "HumanEval/83")
        if match:
            program_num = match.group(1)
            filepath = get_humaneval_filepath(program_num, runner_programs_dir)

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                lines = content.split('\n')

                print(f"\nFile: {filepath}")
                print(f"Total lines: {len(lines)}")
                print("\nFirst 15 lines:")
                for i, line in enumerate(lines[:15], 1):
                    print(f"  {i:2d}: {repr(line)}")


                total_count = len(lines)
                print(f"\nOK HumanEval/83 has {total_count} lines total")


        print(f"\n{'='*60}")
        print("FOR YOUR PAPER:")
        print("=" * 60)
        print(f"\nThe {found_files} subject programs have:")
        print(f"  - Average size of {avg_loc:.2f} lines (ranging from {min_loc} to {max_loc} lines)")
        print(f"  - Mean cyclomatic complexity of {avg_cc:.2f}+/-{std_cc:.2f} (ranging between {min_cc} and {max_cc})")

        print(f"\nFormatted sentence:")
        print(f"  and a mean cyclomatic complexity of {avg_cc:.2f}+/-{std_cc:.2f} (ranging between {min_cc} and {max_cc}),")


        total_loc = sum(loc_values)
        total_cc = sum(cc_values)
        print(f"\n{'='*60}")
        print("TOTALS")
        print("=" * 60)
        print(f"\nTotal lines of code: {total_loc:,} lines")
        print(f"Total cyclomatic complexity: {total_cc:,}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate program statistics from source files."
    )

    parser.add_argument(
        "--json-file",
        default="artifacts/programs/runner_programs_with_coverage.json",
        help="Path to the program metadata JSON file."
    )

    parser.add_argument(
        "--cruxeval-dir",
        default="data/CRUXEval/formatted_cruxeval_programs",
        help="Directory containing formatted CRUXEval programs."
    )

    parser.add_argument(
        "--runner-programs-dir",
        default="artifacts/programs/runner_programs",
        help="Directory containing HumanEval and PythonSaga runner programs."
    )

    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()

    print("Calculating ACCURATE statistics from source files...")
    print(f"Reading from: {args.json_file}")
    print("-" * 50)

    calculate_accurate_statistics(
        args.json_file,
        args.cruxeval_dir,
        args.runner_programs_dir
    )
if __name__ == "__main__":
    main()