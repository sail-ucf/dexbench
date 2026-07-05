import ast
import json
import os
import re
import runpy
import shutil
import tempfile
import signal
from pathlib import Path
from coverage import Coverage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


INPUT_PLACEHOLDER = "??"
MODEL = "grok-4-fast-reasoning"


ANSWER_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def run_script_with_timeout(code_str: str, timeout=1):
    """Run code with 1 second timeout to prevent hanging - IGNORE ASSERTION ERRORS"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as t:
        t.write(code_str)
        tmp_path = t.name

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    executed = set()
    try:
        cov = Coverage(data_file=None)
        cov.start()

        try:
            runpy.run_path(tmp_path, run_name="__main__")
        except AssertionError:
            print("    Assertion failed (IGNORED - this is expected for backward reasoning)")
        except Exception as e:
            print(f"    Other execution error (but continuing): {e}")

        cov.stop()
        cov.save()

        try:
            lookup_path = os.path.realpath(tmp_path)
            lines = cov.get_data().lines(lookup_path)
            if lines:
                executed = set(lines)
                print(f"    Coverage collected: {len(executed)} lines executed")
        except Exception as e:
            print(f"    Coverage collection error: {e}")

    except TimeoutError:
        print("    TIMEOUT")
    except Exception as e:
        print(f"    Fatal error: {e}")
    finally:
        signal.alarm(0)
        try:
            os.remove(tmp_path)
        except:
            pass

    return executed

def extract_answer_text(answer_file: Path):
    """Extract text between [ANSWER] tags"""
    if not answer_file.exists():
        return None
    try:
        txt = answer_file.read_text(encoding="utf-8")
        m = ANSWER_RE.search(txt)
        if not m:
            return None
        return m.group(1).strip()
    except:
        return None

def find_function_call_and_replace(script_source: str, new_args_text: str):
    """Replace the ablation input placeholder with predicted arguments."""
    lines = script_source.splitlines()

    for i, line in enumerate(lines):
        if INPUT_PLACEHOLDER in line:
            print(f"    Found function call to replace: {line}")

            new_args_text = new_args_text.strip()
            new_line = line.replace(INPUT_PLACEHOLDER, new_args_text)
            lines[i] = new_line

            print(f"    Replaced: {line} -> {new_line}")
            return '\n'.join(lines)

    print("    No ablation input placeholder found")
    return script_source

def get_total_lines_of_code(script_source: str):
    """Get total number of lines in the script"""
    return len(script_source.splitlines())

def create_one_hot_encoding(executed_lines, total_lines):
    """Create one-hot encoding for coverage data"""
    one_hot = [0] * total_lines
    for line_num in executed_lines:
        if 1 <= line_num <= total_lines:
            one_hot[line_num - 1] = 1
    return one_hot

def calculate_jaccard_similarity(true_one_hot, pred_one_hot):
    """Calculate Jaccard similarity between two one-hot encoded vectors"""
    true_set = set(i for i, val in enumerate(true_one_hot) if val == 1)
    pred_set = set(i for i, val in enumerate(pred_one_hot) if val == 1)

    if not true_set and not pred_set:
        return 0.0

    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)

    return intersection / union if union > 0 else 0.0

def parse_predicted_coverage(path: Path, total_lines: int):
    """Parse model's predicted coverage - SIMPLE AND ROBUST"""

    response_files = [
        path / "ask_predict_coverage_response.txt",
        path / "crispe_coverage_response.txt",
    ]

    response_file = None
    for file_path in response_files:
        if file_path.exists():
            response_file = file_path
            break

    if not response_file:
        return None

    try:
        content = response_file.read_text(encoding="utf-8").strip()
    except:
        return None

    if not content:
        return None


    answer_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', content, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()


    numbers = re.findall(r'\b\d+\b', content)
    if numbers:
        pred_lines = set()
        for num in numbers:
            try:
                line_num = int(num)
                if 1 <= line_num <= total_lines:
                    pred_lines.add(line_num)
            except:
                continue
        if pred_lines:
            return create_one_hot_encoding(pred_lines, total_lines)

    return None

def get_line_type(script_source: str, line_number: int):
    """Determine if a line is a return statement, conditional, or loop"""
    try:
        tree = ast.parse(script_source)
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_number:
                if isinstance(node, ast.Return):
                    return "return"
                elif isinstance(node, (ast.If, ast.While, ast.For)):
                    return "control"
    except:
        pass
    return "other"

def get_control_body_lines(script_source: str, line_number: int):
    """Get the line numbers in the body of a control structure"""
    try:
        tree = ast.parse(script_source)
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_number:
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    body_lines = set()

                    for stmt in node.body:

                        if hasattr(stmt, 'lineno'):
                            body_lines.add(stmt.lineno)

                        for subnode in ast.walk(stmt):
                            if hasattr(subnode, 'lineno') and subnode.lineno != line_number:
                                body_lines.add(subnode.lineno)
                    return body_lines
    except Exception as e:
        print(f"    Error parsing control body: {e}")
    return set()

def evaluate_backward_reasoning(program_obj, output_dir: Path, approach: str):
    """Evaluate backward reasoning (input generation)"""
    runnable_src = program_obj["runnable_script"]


    priority_line = int(program_obj["coverage_metadata"]["advanced_priority_line"])

    print(f"    Evaluating {approach.upper()} backward reasoning for line {priority_line}")

    input_response_file = output_dir / "ablation_input_response.txt"
    ans_text = extract_answer_text(input_response_file)
    if not ans_text:
        print(f"    No input prediction found")
        return 0.0

    print(f"    Raw answer: {repr(ans_text)}")


    ans_text = ans_text.strip()

    mutated_src = find_function_call_and_replace(runnable_src, ans_text)
    if mutated_src == runnable_src:
        print("    Failed to replace input placeholder")
        return 0.0


    print(f"    Executing program with generated input...")
    executed_lines = run_script_with_timeout(mutated_src, timeout=1)
    print(f"    Executed lines: {sorted(executed_lines)}")


    line_type = get_line_type(runnable_src, priority_line)
    print(f"    Line {priority_line} type: {line_type}")

    success = False
    if line_type == "return":
        success = priority_line in executed_lines
        print(f"    Return line check: {priority_line} in {executed_lines} = {success}")
    elif line_type == "control":
        body_lines = get_control_body_lines(runnable_src, priority_line)
        print(f"    Control body lines: {body_lines}")

        body_executed = body_lines & executed_lines
        success = len(body_executed) > 0
        print(f"    Control body check: {body_executed} executed = {success}")
    else:
        success = priority_line in executed_lines
        print(f"    Other line check: {priority_line} in {executed_lines} = {success}")

    print(f"    {approach.upper()} backward reasoning result: {success}")
    return 1.0 if success else 0.0

def evaluate_forward_reasoning(program_obj, output_dir: Path, approach: str):
    """Evaluate forward reasoning (coverage prediction) - using copied results"""
    runnable_src = program_obj["runnable_script"]
    true_covered = set(program_obj["coverage_metadata"]["covered_lines"])
    total_lines = get_total_lines_of_code(runnable_src)


    true_one_hot = create_one_hot_encoding(true_covered, total_lines)


    pred_one_hot = parse_predicted_coverage(output_dir, total_lines)
    if pred_one_hot is None:
        return 0.0


    similarity = calculate_jaccard_similarity(true_one_hot, pred_one_hot)

    return similarity

def normalize_task_id(dir_name):
    """Convert directory name to standardized task ID format"""
    if dir_name.startswith("CRUXEval_"):
        number = dir_name.replace("CRUXEval_", "")
        return f"CRUXEval/{number}"
    elif dir_name.startswith("HumanEval_"):
        number = dir_name.replace("HumanEval_", "")
        return f"HumanEval/{number}"
    elif dir_name.startswith("PythonSaga_"):
        number = dir_name.replace("PythonSaga_", "")
        return f"PythonSaga/{number}"

    return dir_name

def copy_forward_reasoning_results(original_outputs_dir: Path, ablation_outputs_dir: Path):
    """Copy forward reasoning results from original experiment to grok ablation directory"""
    print("Copying forward reasoning results from original experiment to grok directory...")

    copied_count = 0


    for original_program_dir in (original_outputs_dir / MODEL).iterdir():
        if not original_program_dir.is_dir():
            continue

        task_id = normalize_task_id(original_program_dir.name)
        ablation_program_dir = ablation_outputs_dir / MODEL / original_program_dir.name

        if not ablation_program_dir.exists():
            print(f"    Grok ablation directory not found for {task_id}, skipping")
            continue


        for output_num in range(1, 6):
            original_output_dir = original_program_dir / f"output_{output_num}"
            ablation_output_dir = ablation_program_dir / f"output_{output_num}"

            if not original_output_dir.exists() or not ablation_output_dir.exists():
                continue


            coverage_files = [
                "ask_predict_coverage_response.txt",
                "ask_predict_coverage_usage.json"
            ]

            for file_name in coverage_files:
                original_file = original_output_dir / file_name
                ablation_file = ablation_output_dir / file_name

                if original_file.exists():
                    shutil.copy2(original_file, ablation_file)
                    copied_count += 1

    print(f"Copied {copied_count} forward reasoning files to grok directory")
    return copied_count

def evaluate_pass_at_k(program_obj, output_base_dir: Path, k: int, approach: str):
    """Evaluate pass@k metrics for a program"""
    results = []

    for i in range(1, k + 1):
        output_dir = output_base_dir / f"output_{i}"
        if not output_dir.exists():
            continue

        print(f"    Evaluating {approach.upper()} output {i}...")

        try:
            forward_score = evaluate_forward_reasoning(program_obj, output_dir, approach)
            backward_score = evaluate_backward_reasoning(program_obj, output_dir, approach)


            strict_forward = 1.0 if forward_score >= 0.999 else 0.0

            strict_backward = backward_score

            results.append({
                "output_id": i,
                "forward_score": forward_score,
                "backward_score": backward_score,
                "strict_forward": strict_forward,
                "strict_backward": strict_backward
            })

            print(f"      {approach.upper()} Output {i}: F{forward_score:.3f} B{backward_score:.3f} SF{strict_forward} SB{strict_backward}")

        except Exception as e:
            print(f"    Error evaluating {approach.upper()} output {i}: {e}")
            continue

    return results

def calculate_pass_at_k_metrics(output_results, k: int):
    """Calculate pass@k metrics from output results following the formal evaluation metrics"""
    if not output_results:
        return {
            "strict_forward": 0.0,
            "strict_backward": 0.0,
            "strict_overall": 0.0,
            "relaxed_forward": 0.0,
            "relaxed_backward": 0.0,
            "relaxed_overall": 0.0
        }


    if k == 1:
        result = output_results[0]
        strict_forward = result["strict_forward"]
        strict_backward = result["strict_backward"]
        strict_overall = strict_forward * strict_backward

        relaxed_forward = result["forward_score"]
        relaxed_backward = result["backward_score"]
        relaxed_overall = relaxed_forward * relaxed_backward


    else:

        strict_success = any(
            result["strict_forward"] == 1.0 and result["strict_backward"] == 1.0
            for result in output_results
        )
        strict_overall = 1.0 if strict_success else 0.0


        strict_forward = 1.0 if any(result["strict_forward"] == 1.0 for result in output_results) else 0.0
        strict_backward = 1.0 if any(result["strict_backward"] == 1.0 for result in output_results) else 0.0


        max_forward = max(result["forward_score"] for result in output_results)
        max_backward = max(result["backward_score"] for result in output_results)
        relaxed_forward = max_forward
        relaxed_backward = max_backward
        relaxed_overall = max_forward * max_backward

    return {
        "strict_forward": strict_forward,
        "strict_backward": strict_backward,
        "strict_overall": strict_overall,
        "relaxed_forward": relaxed_forward,
        "relaxed_backward": relaxed_backward,
        "relaxed_overall": relaxed_overall
    }

def get_ablation_programs(ablation_coverage_json: Path):
    """Get all ablation study programs"""
    if not ablation_coverage_json.exists():
        print(f"Missing {ablation_coverage_json}")
        return []

    try:
        with open(ablation_coverage_json, "r", encoding="utf-8") as f:
            ablation_programs = json.load(f)
        print(f"Loaded {len(ablation_programs)} ablation programs")
        return ablation_programs
    except Exception as e:
        print(f"Error loading ablation JSON: {e}")
        return []

def process_ablation_experiments(ablation_coverage_json: Path, ablation_outputs_dir: Path):
    """Process ablation study experiments for all programs with grok"""


    ablation_programs = get_ablation_programs(ablation_coverage_json)
    if not ablation_programs:
        return []

    results = []
    processed_count = 0


    for ablation_program in ablation_programs:
        task_id = ablation_program["task_id"]
        dir_name = task_id.replace("/", "_")

        print(f"Inspecting Processing program: {task_id}")


        ablation_program_dir = ablation_outputs_dir / MODEL / dir_name

        if not ablation_program_dir.exists():
            print(f"   Grok ablation directory not found: {ablation_program_dir}")
            continue

        print(f"   Found grok ablation directory")


        ablation_pass1_results = evaluate_pass_at_k(ablation_program, ablation_program_dir, 1, "grok_ablation")
        ablation_pass5_results = evaluate_pass_at_k(ablation_program, ablation_program_dir, 5, "grok_ablation")


        ablation_pass1_metrics = calculate_pass_at_k_metrics(ablation_pass1_results, 1) if ablation_pass1_results else create_empty_metrics()
        ablation_pass5_metrics = calculate_pass_at_k_metrics(ablation_pass5_results, 5) if ablation_pass5_results else create_empty_metrics()


        results.append({
            "task_id": task_id,
            "dataset": task_id.split("/")[0],


            "grok_ablation_pass1_strict_forward": ablation_pass1_metrics["strict_forward"],
            "grok_ablation_pass1_strict_backward": ablation_pass1_metrics["strict_backward"],
            "grok_ablation_pass1_strict_overall": ablation_pass1_metrics["strict_overall"],
            "grok_ablation_pass1_relaxed_forward": ablation_pass1_metrics["relaxed_forward"],
            "grok_ablation_pass1_relaxed_backward": ablation_pass1_metrics["relaxed_backward"],
            "grok_ablation_pass1_relaxed_overall": ablation_pass1_metrics["relaxed_overall"],

            "grok_ablation_pass5_strict_forward": ablation_pass5_metrics["strict_forward"],
            "grok_ablation_pass5_strict_backward": ablation_pass5_metrics["strict_backward"],
            "grok_ablation_pass5_strict_overall": ablation_pass5_metrics["strict_overall"],
            "grok_ablation_pass5_relaxed_forward": ablation_pass5_metrics["relaxed_forward"],
            "grok_ablation_pass5_relaxed_backward": ablation_pass5_metrics["relaxed_backward"],
            "grok_ablation_pass5_relaxed_overall": ablation_pass5_metrics["relaxed_overall"],
        })

        processed_count += 1
        print(f"   Completed grok evaluation for {task_id} ({processed_count}/{len(ablation_programs)})")

    print(f"Total grok evaluations: {len(results)}")
    return results

def create_empty_metrics():
    """Create empty metrics dictionary"""
    return {
        "strict_forward": 0.0,
        "strict_backward": 0.0,
        "strict_overall": 0.0,
        "relaxed_forward": 0.0,
        "relaxed_backward": 0.0,
        "relaxed_overall": 0.0
    }

def create_visualization_tables(df, output_dir):
    """Create visualization tables for grok ablation study results"""


    pass1_data = []
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]

        row = [dataset]
        row.extend([
            f"{dataset_data['grok_ablation_pass1_strict_forward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass1_strict_backward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass1_strict_overall'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass1_relaxed_forward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass1_relaxed_backward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass1_relaxed_overall'].mean():.3f}",
            f"{len(dataset_data):d}"
        ])
        pass1_data.append(row)


    overall_row = ['Overall']
    overall_row.extend([
        f"{df['grok_ablation_pass1_strict_forward'].mean():.3f}",
        f"{df['grok_ablation_pass1_strict_backward'].mean():.3f}",
        f"{df['grok_ablation_pass1_strict_overall'].mean():.3f}",
        f"{df['grok_ablation_pass1_relaxed_forward'].mean():.3f}",
        f"{df['grok_ablation_pass1_relaxed_backward'].mean():.3f}",
        f"{df['grok_ablation_pass1_relaxed_overall'].mean():.3f}",
        f"{len(df):d}"
    ])
    pass1_data.append(overall_row)

    pass1_df = pd.DataFrame(pass1_data,
                           columns=['Dataset',
                                   'Pass1_Strict_Forward', 'Pass1_Strict_Backward', 'Pass1_Strict_Overall',
                                   'Pass1_Relaxed_Forward', 'Pass1_Relaxed_Backward', 'Pass1_Relaxed_Overall',
                                   'N'])
    pass1_df.to_csv(output_dir / 'grok_ablation_pass1_metrics.csv', index=False)


    pass5_data = []
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]

        row = [dataset]
        row.extend([
            f"{dataset_data['grok_ablation_pass5_strict_forward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass5_strict_backward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass5_strict_overall'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass5_relaxed_forward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass5_relaxed_backward'].mean():.3f}",
            f"{dataset_data['grok_ablation_pass5_relaxed_overall'].mean():.3f}",
            f"{len(dataset_data):d}"
        ])
        pass5_data.append(row)


    overall_row = ['Overall']
    overall_row.extend([
        f"{df['grok_ablation_pass5_strict_forward'].mean():.3f}",
        f"{df['grok_ablation_pass5_strict_backward'].mean():.3f}",
        f"{df['grok_ablation_pass5_strict_overall'].mean():.3f}",
        f"{df['grok_ablation_pass5_relaxed_forward'].mean():.3f}",
        f"{df['grok_ablation_pass5_relaxed_backward'].mean():.3f}",
        f"{df['grok_ablation_pass5_relaxed_overall'].mean():.3f}",
        f"{len(df):d}"
    ])
    pass5_data.append(overall_row)

    pass5_df = pd.DataFrame(pass5_data,
                           columns=['Dataset',
                                   'Pass5_Strict_Forward', 'Pass5_Strict_Backward', 'Pass5_Strict_Overall',
                                   'Pass5_Relaxed_Forward', 'Pass5_Relaxed_Backward', 'Pass5_Relaxed_Overall',
                                   'N'])
    pass5_df.to_csv(output_dir / 'grok_ablation_pass5_metrics.csv', index=False)


    create_table_visualizations(pass1_df, pass5_df, output_dir)

    return pass1_df, pass5_df

def create_table_visualizations(pass1_df, pass5_df, output_dir):
    """Create visualization tables for the report"""


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))


    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=pass1_df.values,
                      colLabels=pass1_df.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8'] * len(pass1_df.columns))

    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.8)


    for i in range(len(pass1_df.columns)):
        table1[(0, i)].set_facecolor('#2E86AB')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    ax1.set_title('Grok-4-Fast-Reasoning Ablation Study - Pass@1 Evaluation Metrics',
                  fontsize=14, fontweight='bold', pad=20)


    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=pass5_df.values,
                      colLabels=pass5_df.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8'] * len(pass5_df.columns))

    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.8)


    for i in range(len(pass5_df.columns)):
        table2[(0, i)].set_facecolor('#A23B72')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Grok-4-Fast-Reasoning Ablation Study - Pass@5 Evaluation Metrics',
                  fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'grok_ablation_metrics_tables.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_ablation_report(results, report_dir: Path):
    """Generate grok ablation study report"""

    if not results:
        print("No results!")
        return None

    df = pd.DataFrame(results)

    report_dir.mkdir(parents=True, exist_ok=True)


    df.to_json(report_dir / "grok_ablation_detailed_results.json", indent=2, orient="records")
    df.to_csv(report_dir / "grok_ablation_detailed_results.csv", index=False)


    pass1_df, pass5_df = create_visualization_tables(df, report_dir)


    print("=" * 80)
    print("GROK-4-FAST-REASONING ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"Total programs evaluated: {len(df)}")

    print("\nGrok Pass@1 Averages:")
    print(pass1_df.to_string(index=False))

    print("\nGrok Pass@5 Averages:")
    print(pass5_df.to_string(index=False))

    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Grok ablation-study model outputs."
    )

    parser.add_argument(
        "--ablation-coverage-json",
        type=Path,
        default=Path(
            "src/evaluation/data/programs/"
            "ablation_study_programs_with_coverage.json"
        ),
        help="Path to the ablation-study coverage JSON file."
    )

    parser.add_argument(
        "--original-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs"),
        help="Directory containing the original Grok outputs."
    )

    parser.add_argument(
        "--ablation-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs_Ablation_grok"),
        help="Directory containing the Grok ablation outputs."
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("grok_ablation_evaluation_reports"),
        help="Directory where ablation evaluation reports will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()
    print("Starting Grok-4-Fast-Reasoning Ablation Study Evaluation...")


    copy_forward_reasoning_results(args.original_outputs_dir, args.ablation_outputs_dir)


    results = process_ablation_experiments(args.ablation_coverage_json, args.ablation_outputs_dir)

    if not results:
        print("No results found!")
        return

    df = generate_ablation_report(results, args.report_dir)

    print("\n" + "=" * 80)
    print("GROK-4-FAST-REASONING ABLATION STUDY EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.report_dir}")
    print(f"Total evaluations: {len(results)}")
    print("\nGenerated files:")
    print("  - grok_ablation_pass1_metrics.csv (Grok Pass@1 metrics table)")
    print("  - grok_ablation_pass5_metrics.csv (Grok Pass@5 metrics table)")
    print("  - grok_ablation_metrics_tables.png (visualization of both tables)")
    print("  - grok_ablation_detailed_results.json (raw evaluation data)")
    print("  - grok_ablation_detailed_results.csv (raw evaluation data in CSV)")

if __name__ == "__main__":
    main()
