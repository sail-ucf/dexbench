import json
import re
import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm



MODELS = {
    "gpt-5-mini": {
        "CRISPE": "API_Model_Outputs_CRISPE",
        "CRISPE_no_focc": "API_Model_Outputs_CRISPE_no_focc",
        "CRISPE_no_examplar": "API_Model_Outputs_CRISPE_no_examplar"
    },
    "grok-4-fast-reasoning": {
        "CRISPE": "API_Model_Outputs_CRISPE_grok",
        "CRISPE_no_focc": "API_Model_Outputs_CRISPE_grok_no_focc",
        "CRISPE_no_examplar": "API_Model_Outputs_CRISPE_grok_no_examplar"
    }
}
EXPERIMENTS = ["CRISPE", "CRISPE_no_focc", "CRISPE_no_examplar"]


ANSWER_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)


DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LLM coverage answers against ground-truth FOCCs."
    )

    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=Path("src/evaluation/data/programs/focc/all_programs_foccs.json"),
        help="Path to the ground-truth FOCC JSON file."
    )

    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("src/evaluation"),
        help="Parent directory containing the model-output directories."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("focc_analysis_results"),
        help="Directory where analysis CSV files will be saved."
    )

    return parser.parse_args()

def debug_print(msg):
    """Print debug messages if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {msg}")

def normalize_program_id(program_id):
    """Normalize program ID for comparison (convert / to _)."""
    return program_id.replace("/", "_")

def extract_executed_lines_from_response(response_text):
    """Extract executed lines from LLM response text."""
    if not response_text:
        return None

    try:

        match = ANSWER_RE.search(response_text)
        if match:
            json_str = match.group(1).strip()


            json_str = re.sub(r'^[^{[]*', '', json_str)
            json_str = re.sub(r'[^}\]]*$', '', json_str)

            try:
                data = json.loads(json_str)
                if "executed_lines" in data and isinstance(data["executed_lines"], list):
                    return tuple(sorted(data["executed_lines"]))
                elif "coverage" in data and isinstance(data["coverage"], list):
                    return tuple(sorted(data["coverage"]))
            except json.JSONDecodeError:

                json_str = json_str.replace("'", '"')
                try:
                    data = json.loads(json_str)
                    if "executed_lines" in data and isinstance(data["executed_lines"], list):
                        return tuple(sorted(data["executed_lines"]))
                    elif "coverage" in data and isinstance(data["coverage"], list):
                        return tuple(sorted(data["coverage"]))
                except:
                    pass



        list_pattern = r'\[([\d,\s]+)\]'
        list_match = re.search(list_pattern, response_text)
        if list_match:
            numbers = [int(n.strip()) for n in list_match.group(1).split(',') if n.strip().isdigit()]
            if numbers:
                return tuple(sorted(numbers))


        lines_pattern = r'Lines?\s+([\d,\s]+)'
        lines_match = re.search(lines_pattern, response_text, re.IGNORECASE)
        if lines_match:
            numbers = [int(n.strip()) for n in lines_match.group(1).split(',') if n.strip().isdigit()]
            if numbers:
                return tuple(sorted(numbers))


        all_numbers = re.findall(r'\b\d+\b', response_text)
        if all_numbers:
            numbers = [int(n) for n in all_numbers if int(n) <= 100]
            if numbers:
                return tuple(sorted(set(numbers)))

        return None
    except Exception as e:
        debug_print(f"Error extracting lines: {e}")
        return None

def parse_foccs(foccs_list):
    """Parse FOCCs from ground truth data into tuples for comparison."""
    parsed_foccs = []
    if not foccs_list:
        return parsed_foccs

    for focc in foccs_list:
        if isinstance(focc, str):
            if focc.lower().startswith("error"):
                continue
            numbers = re.findall(r'\b\d+\b', focc)
            if numbers:
                parsed_foccs.append(tuple(sorted([int(n) for n in numbers])))
        elif isinstance(focc, list):
            if focc and isinstance(focc[0], int):
                parsed_foccs.append(tuple(sorted(focc)))

    return parsed_foccs

def load_ground_truth(ground_truth_file: Path):
    """Load ground truth FOCCs from JSON file and normalize IDs."""
    try:
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}

    ground_truth = {}
    for item in data:
        program_id = item["program_id"]
        normalized_id = normalize_program_id(program_id)

        foccs_raw = item.get("foccs", [])
        foccs = parse_foccs(foccs_raw)

        if foccs:
            ground_truth[normalized_id] = {
                "foccs": foccs,
                "test_case": item.get("test_case", ""),
                "dataset": item.get("dataset", ""),
                "original_id": program_id
            }

    print(f"Loaded {len(ground_truth)} programs with valid FOCCs from ground truth")
    return ground_truth

def analyze_experiment(model, experiment, ground_truth, outputs_root: Path):
    """Analyze one experiment for a specific model."""
    print(f"\n{'='*60}")
    print(f"STARTING ANALYSIS: {model}/{experiment}")
    print(f"{'='*60}")

    results = {
        "pass1_mismatch": [],
        "pass5_mismatch": [],
        "pass1_exact_match": [],
        "pass5_any_match": [],
        "pass1_mismatch_details": [],
        "pass5_mismatch_details": [],
        "total_programs": 0,
        "no_response_count": 0,
        "invalid_response_count": 0,
        "by_dataset": {}
    }


    if model in MODELS and experiment in MODELS[model]:
        base_dir_name = MODELS[model][experiment]
        base_dir = outputs_root / base_dir_name / model
        print(f"Looking in directory: {base_dir}")
    else:
        print(f"Invalid model/experiment combination: {model}/{experiment}")
        return results

    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return results


    try:
        program_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        results["total_programs"] = len(program_dirs)
        print(f"Found {results['total_programs']} program directories")
    except Exception as e:
        print(f"Error reading directory {base_dir}: {e}")
        return results


    datasets = set()

    for prog_dir in program_dirs:
        program_id = prog_dir.name
        print(f"\n{'-'*50}")
        print(f"Analyzing program: {program_id}")
        print(f"Experiment type: {experiment}")


        if program_id not in ground_truth:
            print(f"  WARNING: {program_id} not found in ground truth (normalized)")

            found = False
            for gt_id in ground_truth:
                if normalize_program_id(gt_id) == program_id:
                    program_id = gt_id
                    found = True
                    break
            if not found:
                print(f"  Skipping {program_id} - not in ground truth")
                continue

        program_data = ground_truth[program_id]
        ground_foccs = program_data["foccs"]
        dataset = program_data["dataset"]


        if dataset not in results["by_dataset"]:
            results["by_dataset"][dataset] = {
                "pass1_mismatch": [],
                "pass5_mismatch": [],
                "pass1_exact_match": [],
                "pass5_any_match": [],
                "total": 0
            }

        results["by_dataset"][dataset]["total"] += 1

        print(f"  Dataset: {dataset}")
        print(f"  Ground truth FOCCs ({len(ground_foccs)} sets):")
        for i, focc in enumerate(ground_foccs[:3]):
            print(f"    Set {i+1}: {focc}")
        if len(ground_foccs) > 3:
            print(f"    ... and {len(ground_foccs) - 3} more sets")


        pass1_answer = None
        pass5_answers = []

        for i in range(1, 6):
            output_dir = prog_dir / f"output_{i}"
            response_file = output_dir / "crispe_coverage_response.txt"

            if not response_file.exists():
                continue

            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    response_text = f.read()

                executed_lines = extract_executed_lines_from_response(response_text)

                if executed_lines:
                    pass5_answers.append(executed_lines)
                    if i == 1:
                        pass1_answer = executed_lines
            except Exception as e:
                results["invalid_response_count"] += 1

        if not pass5_answers:
            print(f"  No valid responses found for {program_id}")
            results["no_response_count"] += 1
            continue


        if pass1_answer:
            if pass1_answer in ground_foccs:
                results["pass1_exact_match"].append(program_id)
                results["by_dataset"][dataset]["pass1_exact_match"].append(program_id)
                print(f"  OK PASS@1 MATCH: {pass1_answer}")
            else:
                results["pass1_mismatch"].append(program_id)
                results["by_dataset"][dataset]["pass1_mismatch"].append(program_id)
                results["pass1_mismatch_details"].append({
                    "program_id": program_id,
                    "dataset": dataset,
                    "llm_answer": pass1_answer,
                    "ground_foccs": ground_foccs
                })
                print(f"  NO PASS@1 MISMATCH")
                print(f"    LLM Answer: {pass1_answer}")
                print(f"    Not in any FOCC set")


        any_match = False
        for answer in pass5_answers:
            if answer in ground_foccs:
                any_match = True
                break

        if any_match:
            results["pass5_any_match"].append(program_id)
            results["by_dataset"][dataset]["pass5_any_match"].append(program_id)
            print(f"  OK PASS@5 MATCH (at least one output matches)")
        else:
            results["pass5_mismatch"].append(program_id)
            results["by_dataset"][dataset]["pass5_mismatch"].append(program_id)
            results["pass5_mismatch_details"].append({
                "program_id": program_id,
                "dataset": dataset,
                "llm_answers": pass5_answers,
                "ground_foccs": ground_foccs
            })
            print(f"  NO PASS@5 MISMATCH (no outputs match)")
            print(f"    LLM Answers: {pass5_answers}")

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE: {model}/{experiment}")
    print(f"Total programs: {results['total_programs']}")
    print(f"Pass@1 matches: {len(results['pass1_exact_match'])}")
    print(f"Pass@1 mismatches: {len(results['pass1_mismatch'])}")
    print(f"Pass@5 matches: {len(results['pass5_any_match'])}")
    print(f"Pass@5 mismatches: {len(results['pass5_mismatch'])}")
    print(f"No responses: {results['no_response_count']}")


    print(f"\nDataset Breakdown:")
    for dataset, data in results["by_dataset"].items():
        if data["total"] > 0:
            p1_rate = len(data["pass1_exact_match"]) / data["total"] * 100 if data["total"] > 0 else 0
            p5_rate = len(data["pass5_any_match"]) / data["total"] * 100 if data["total"] > 0 else 0
            print(f"  {dataset}: {data['total']} programs")
            print(f"    Pass@1: {p1_rate:.1f}% ({len(data['pass1_exact_match'])}/{data['total']})")
            print(f"    Pass@5: {p5_rate:.1f}% ({len(data['pass5_any_match'])}/{data['total']})")

    print(f"{'='*60}")

    return results

def create_summary_table(all_results):
    """Create a comprehensive summary table with dataset breakdown."""
    summary_data = []

    for model in MODELS:
        for experiment in EXPERIMENTS:
            key = f"{model}_{experiment}"
            if key in all_results:
                results = all_results[key]
                total = results["total_programs"]

                if total > 0:

                    overall_p1_rate = len(results["pass1_exact_match"]) / total * 100
                    overall_p5_rate = len(results["pass5_any_match"]) / total * 100


                    summary_data.append({
                        "Model": model,
                        "Experiment": experiment,
                        "Dataset": "OVERALL",
                        "Total_Programs": total,
                        "Pass@1_Match_Rate": f"{overall_p1_rate:.2f}%",
                        "Pass@5_Match_Rate": f"{overall_p5_rate:.2f}%",
                        "Pass@1_Match_Count": len(results["pass1_exact_match"]),
                        "Pass@5_Match_Count": len(results["pass5_any_match"]),
                        "Pass@1_Mismatch_Count": len(results["pass1_mismatch"]),
                        "Pass@5_Mismatch_Count": len(results["pass5_mismatch"]),
                        "No_Response": results["no_response_count"]
                    })


                    for dataset, data in results.get("by_dataset", {}).items():
                        if data["total"] > 0:
                            p1_rate = len(data["pass1_exact_match"]) / data["total"] * 100 if data["total"] > 0 else 0
                            p5_rate = len(data["pass5_any_match"]) / data["total"] * 100 if data["total"] > 0 else 0

                            summary_data.append({
                                "Model": model,
                                "Experiment": experiment,
                                "Dataset": dataset,
                                "Total_Programs": data["total"],
                                "Pass@1_Match_Rate": f"{p1_rate:.2f}%",
                                "Pass@5_Match_Rate": f"{p5_rate:.2f}%",
                                "Pass@1_Match_Count": len(data["pass1_exact_match"]),
                                "Pass@5_Match_Count": len(data["pass5_any_match"]),
                                "Pass@1_Mismatch_Count": len(data["pass1_mismatch"]),
                                "Pass@5_Mismatch_Count": len(data["pass5_mismatch"]),
                                "No_Response": "N/A"
                            })

    return pd.DataFrame(summary_data)

def save_detailed_results(all_results, ground_truth, output_dir: Path):
    """Save detailed mismatch results to files."""
    output_dir.mkdir(exist_ok=True)


    summary_df = create_summary_table(all_results)
    summary_df.to_csv(output_dir / "summary_with_datasets.csv", index=False)


    for key, results in all_results.items():
        model, experiment = key.split("_", 1)


        if results["pass1_mismatch_details"]:
            df_details = pd.DataFrame(results["pass1_mismatch_details"])
            df_details.to_csv(output_dir / f"{model}_{experiment}_pass1_mismatch_details.csv", index=False)


            df_ids = pd.DataFrame({
                "program_id": results["pass1_mismatch"],
                "dataset": [ground_truth.get(pid, {}).get("dataset", "unknown") for pid in results["pass1_mismatch"]]
            })
            df_ids.to_csv(output_dir / f"{model}_{experiment}_pass1_mismatch_programs.csv", index=False)


        if results["pass5_mismatch_details"]:
            df_details = pd.DataFrame(results["pass5_mismatch_details"])
            df_details.to_csv(output_dir / f"{model}_{experiment}_pass5_mismatch_details.csv", index=False)


            df_ids = pd.DataFrame({
                "program_id": results["pass5_mismatch"],
                "dataset": [ground_truth.get(pid, {}).get("dataset", "unknown") for pid in results["pass5_mismatch"]]
            })
            df_ids.to_csv(output_dir / f"{model}_{experiment}_pass5_mismatch_programs.csv", index=False)

def main():
    args = parse_args()
    print("=" * 80)
    print("LLM Answer vs FOCCs Analysis - COUNT MISMATCHES BY DATASET")
    print("=" * 80)

    print("\nLoading ground truth FOCCs...")
    ground_truth = load_ground_truth(args.ground_truth_file)
    if not ground_truth:
        print("Error: No ground truth data loaded")
        return

    all_results = {}


    for model in MODELS:
        for experiment in EXPERIMENTS:
            results = analyze_experiment(model, experiment, ground_truth, args.outputs_root)
            all_results[f"{model}_{experiment}"] = results


    print("\n" + "="*80)
    print("FINAL SUMMARY - MISMATCH COUNTS BY DATASET")
    print("="*80)

    summary_rows = []

    for model in MODELS:
        for experiment in EXPERIMENTS:
            key = f"{model}_{experiment}"
            if key in all_results:
                results = all_results[key]


                dataset_mismatch_p1 = {}
                dataset_mismatch_p5 = {}
                dataset_total = {}


                for prog_id in results["pass1_mismatch"]:
                    if prog_id in ground_truth:
                        dataset = ground_truth[prog_id].get("dataset", "unknown")
                        dataset_mismatch_p1[dataset] = dataset_mismatch_p1.get(dataset, 0) + 1
                        dataset_total[dataset] = dataset_total.get(dataset, 0) + 1

                for prog_id in results["pass5_mismatch"]:
                    if prog_id in ground_truth:
                        dataset = ground_truth[prog_id].get("dataset", "unknown")
                        dataset_mismatch_p5[dataset] = dataset_mismatch_p5.get(dataset, 0) + 1
                        dataset_total[dataset] = dataset_total.get(dataset, 0) + 1


                summary_rows.append({
                    "Model": model,
                    "Experiment": experiment,
                    "Dataset": "TOTAL",
                    "Total_Programs": results["total_programs"],
                    "Pass@1_Mismatch_Count": len(results["pass1_mismatch"]),
                    "Pass@5_Mismatch_Count": len(results["pass5_mismatch"]),
                    "Pass@1_Match_Count": len(results["pass1_exact_match"]),
                    "Pass@5_Match_Count": len(results["pass5_any_match"])
                })


                for dataset in sorted(dataset_total.keys()):
                    summary_rows.append({
                        "Model": model,
                        "Experiment": experiment,
                        "Dataset": dataset,
                        "Total_Programs": dataset_total[dataset],
                        "Pass@1_Mismatch_Count": dataset_mismatch_p1.get(dataset, 0),
                        "Pass@5_Mismatch_Count": dataset_mismatch_p5.get(dataset, 0),
                        "Pass@1_Match_Count": "N/A",
                        "Pass@5_Match_Count": "N/A"
                    })


    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))


    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    summary_df.to_csv(output_dir / "mismatch_counts_by_dataset.csv", index=False)

    print(f"\nSummary saved to: {output_dir}/mismatch_counts_by_dataset.csv")


if __name__ == "__main__":
    main()
