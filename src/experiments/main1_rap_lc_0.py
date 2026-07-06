import pandas as pd
import re
import argparse
from pathlib import Path

def load_csv(file_path):
    """Load CSV file, handling potential parsing issues"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_task_id(task_id):
    """Normalize task ID to a common format, being agnostic to dataset names"""
    if pd.isna(task_id):
        return None

    task_id = str(task_id)










    if '/' in task_id:
        parts = task_id.split('/')
        if len(parts) > 1:

            last_part = parts[-1]

            if last_part.isdigit():
                dataset_part = parts[-2] if len(parts) > 1 else parts[0]

                if 'API_Model_Outputs' in dataset_part:

                    match = re.search(r'(HumanEval|CRUXEval|PythonSaga)', dataset_part)
                    if match:
                        dataset_name = match.group(0)
                    else:
                        dataset_name = dataset_part
                else:
                    dataset_name = dataset_part
                return f"{dataset_name}_{last_part}"
            else:

                return last_part


    if '_' in task_id:

        return task_id


    if task_id.isdigit():
        return task_id


    return task_id

def find_zero_backward_tasks(main_results_csv: Path, rap_results_csv: Path, least_coverage_results_csv: Path):

    main_experiment = load_csv(main_results_csv)
    rap_experiment = load_csv(rap_results_csv)
    least_coverage_experiment = load_csv(least_coverage_results_csv)

    if main_experiment is None or rap_experiment is None or least_coverage_experiment is None:
        print("Error loading one or more files. Please check the file paths.")
        return


    print("DEBUG: Task ID formats in each file:")
    print("Main experiment sample task_ids:", main_experiment['task_id'].head(5).tolist())
    print("RAP experiment sample task_ids:", rap_experiment['task_id'].head(5).tolist())
    print("Least Coverage sample task_ids:", least_coverage_experiment['task_id'].head(5).tolist())
    print("-" * 80)


    grok_main = main_experiment[
        (main_experiment['model'] == 'grok-4-fast-reasoning') &
        (main_experiment['pass1_strict_backward'] == 1.0)
    ].copy()


    grok_main['normalized_task_id'] = grok_main['task_id'].apply(normalize_task_id)
    print(f"Main experiment normalized samples: {grok_main['normalized_task_id'].head(5).tolist()}")


    rap_zero_backward = rap_experiment[
        rap_experiment['pass1_strict_backward'] == 0.0
    ].copy()
    rap_zero_backward['normalized_task_id'] = rap_zero_backward['task_id'].apply(normalize_task_id)
    print(f"RAP normalized samples: {rap_zero_backward['normalized_task_id'].head(5).tolist()}")


    lc_zero_backward = least_coverage_experiment[
        least_coverage_experiment['pass1_strict_backward'] == 0.0
    ].copy()
    lc_zero_backward['normalized_task_id'] = lc_zero_backward['task_id'].apply(normalize_task_id)
    print(f"Least Coverage normalized samples: {lc_zero_backward['normalized_task_id'].head(5).tolist()}")
    print("-" * 80)

    print(f"Found {len(grok_main)} tasks where grok-4-fast-reasoning has pass1_strict_backward = 1.0 in main experiment")
    print(f"Found {len(rap_zero_backward)} tasks with pass1_strict_backward = 0.0 in RAP experiment")
    print(f"Found {len(lc_zero_backward)} tasks with pass1_strict_backward = 0.0 in Least Coverage experiment")
    print("-" * 80)


    main_set = set(grok_main['normalized_task_id'])
    rap_zero_set = set(rap_zero_backward['normalized_task_id'])
    lc_zero_set = set(lc_zero_backward['normalized_task_id'])


    tasks_in_main_with_zero = main_set.intersection(rap_zero_set.union(lc_zero_set))

    print(f"\nTasks with 1.0 in main AND 0.0 in either RAP or Least Coverage: {len(tasks_in_main_with_zero)}")


    main_id_map = dict(zip(grok_main['normalized_task_id'], grok_main['task_id']))
    rap_id_map = dict(zip(rap_zero_backward['normalized_task_id'], rap_zero_backward['task_id']))
    lc_id_map = dict(zip(lc_zero_backward['normalized_task_id'], lc_zero_backward['task_id']))


    lc_approach_map = {}
    lc_priority_map = {}
    for idx, row in lc_zero_backward.iterrows():
        norm_id = row['normalized_task_id']
        lc_approach_map[norm_id] = row.get('approach', 'N/A')
        lc_priority_map[norm_id] = row.get('priority_line', 'N/A')


    results = []
    for norm_id in sorted(tasks_in_main_with_zero):
        in_rap = norm_id in rap_zero_set
        in_lc = norm_id in lc_zero_set

        result = {
            'normalized_id': norm_id,
            'main_id': main_id_map.get(norm_id, 'N/A'),
            'in_rap': in_rap,
            'in_lc': in_lc,
            'rap_id': rap_id_map.get(norm_id, 'N/A') if in_rap else None,
            'lc_id': lc_id_map.get(norm_id, 'N/A') if in_lc else None,
        }

        if in_lc:
            result['approach'] = lc_approach_map.get(norm_id, 'N/A')
            result['priority_line'] = lc_priority_map.get(norm_id, 'N/A')

        results.append(result)


    print("\n" + "=" * 80)
    print("DETAILED RESULTS: Tasks with 1.0 in Main but 0.0 in Either RAP or Least Coverage")
    print("=" * 80)

    if results:
        for i, result in enumerate(results, 1):
            print(f"\nTask #{i}:")
            print(f"  Normalized ID: {result['normalized_id']}")
            print(f"  Main Experiment ID: {result['main_id']}")

            if result['in_rap']:
                print(f"  OK Has 0.0 in RAP: {result['rap_id']}")
            else:
                print(f"  NO NOT in RAP (or has 1.0 in RAP)")

            if result['in_lc']:
                print(f"  OK Has 0.0 in Least Coverage: {result['lc_id']}")
                print(f"    Approach: {result.get('approach', 'N/A')}")
                print(f"    Priority Line: {result.get('priority_line', 'N/A')}")
            else:
                print(f"  NO NOT in Least Coverage (or has 1.0 in LC)")
    else:
        print("No tasks found matching the criteria.")


    rap_only = [r for r in results if r['in_rap'] and not r['in_lc']]
    lc_only = [r for r in results if r['in_lc'] and not r['in_rap']]
    both = [r for r in results if r['in_rap'] and r['in_lc']]


    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY:")
    print("=" * 80)
    print(f"\n1. Zero only in RAP: {len(rap_only)} tasks")
    for task in rap_only:
        print(f"   - {task['normalized_id']}")

    print(f"\n2. Zero only in Least Coverage: {len(lc_only)} tasks")
    for task in lc_only:
        print(f"   - {task['normalized_id']} (Approach: {task.get('approach', 'N/A')}, Priority: {task.get('priority_line', 'N/A')})")

    print(f"\n3. Zero in BOTH RAP and Least Coverage: {len(both)} tasks")
    for task in both:
        print(f"   - {task['normalized_id']}")

    print("\n" + "=" * 80)
    print("FINAL STATISTICS:")
    print("=" * 80)
    print(f"Total tasks with 1.0 backward in main experiment: {len(grok_main)}")
    print(f"Tasks with 0.0 backward in RAP experiment: {len(rap_zero_backward)}")
    print(f"Tasks with 0.0 backward in Least Coverage experiment: {len(lc_zero_backward)}")
    print(f"\nTasks with 1.0 in main but 0.0 in EITHER experiment: {len(results)}")
    print(f"  - RAP only: {len(rap_only)}")
    print(f"  - Least Coverage only: {len(lc_only)}")
    print(f"  - Both: {len(both)}")

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find tasks where Grok succeeds in the main experiment "
            "but fails in RAP or least-coverage experiments."
        )
    )

    parser.add_argument(
        "--main-results-csv",
        type=Path,
        default=Path("evaluation_reports_main/detailed_results.csv"),
        help="Path to the main experiment detailed results CSV."
    )

    parser.add_argument(
        "--rap-results-csv",
        type=Path,
        default=Path(
            "rap_evaluation_reports_grok/"
            "rap_detailed_results.csv"
        ),
        help="Path to the RAP detailed results CSV."
    )

    parser.add_argument(
        "--least-coverage-results-csv",
        type=Path,
        default=Path(
            "least_coverage_evaluation_reports_grok-4-fast-reasoning/"
            "grok-4-fast-reasoning_least_coverage_detailed_results.csv"
        ),
        help="Path to the least-coverage detailed results CSV."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = find_zero_backward_tasks(
        args.main_results_csv,
        args.rap_results_csv,
        args.least_coverage_results_csv
    )