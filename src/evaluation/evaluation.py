import ast
import json
import os
import re
import runpy
import tempfile
import signal
from pathlib import Path
from coverage import Coverage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse




MODELS = ["gpt-5-mini", "gemini-2.5-flash", "grok-4-fast-reasoning", "claude-sonnet-4-sonnet", "AI21-Jamba-Reasoning-3B", "Llama-3.1-Nemotron-Nano-8B-v1"]



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

def find_assert_call_node_and_source(script_source: str):
    """Find the first unittest assertion call in the code."""
    try:
        tree = ast.parse(script_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Attribute):
                    func_name = func.attr
                elif isinstance(func, ast.Name):
                    func_name = func.id
                if func_name and func_name.startswith("assert"):
                    if not node.args:
                        return (node, None, None, None, None)
                    arg0 = node.args[0]
                    try:
                        arg0_src = ast.get_source_segment(script_source, arg0)
                    except:
                        arg0_src = None
                    func_text = None
                    if isinstance(arg0, ast.Call):
                        try:
                            func_text = ast.get_source_segment(script_source, arg0.func)
                        except:
                            func_text = None
                    try:
                        call_src = ast.get_source_segment(script_source, node)
                    except:
                        call_src = None

                    print(f"    Found assertion: {call_src}")
                    print(f"    First argument: {arg0_src}")
                    print(f"    Function in arg: {func_text}")

                    return (node, arg0, func_text, arg0_src, call_src)
    except Exception as e:
        print(f"    AST parsing error: {e}")
    return None

def replace_first_arg_with_new_args(script_source: str, node_info, new_args_text: str):
    """Replace the first argument of the assertion with predicted args."""
    node, arg0_node, func_text, arg0_src, call_src = node_info
    if arg0_node is None:
        return script_source


    new_args_text = new_args_text.strip()

    print(f"    Replacing: {arg0_src} with {new_args_text}")

    if func_text and arg0_src:

        new_call_text = f"{func_text}({new_args_text})"
        print(f"    Replacing function call: {arg0_src} -> {new_call_text}")
        return script_source.replace(arg0_src, new_call_text, 1)
    elif arg0_src:

        print(f"    Replacing argument: {arg0_src} -> {new_args_text}")
        return script_source.replace(arg0_src, new_args_text, 1)

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
    response_file = path / "ask_predict_coverage_response.txt"
    if not response_file.exists():
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

def evaluate_backward_reasoning(program_obj, output_dir: Path):
    """Evaluate backward reasoning (input prediction)"""
    runnable_src = program_obj["runnable_script"]
    priority_line = int(program_obj["coverage_metadata"]["advanced_priority_line"])

    print(f"    Evaluating backward reasoning for line {priority_line}")


    node_info = find_assert_call_node_and_source(runnable_src)
    if not node_info:
        print(f"    No assertion found")
        return 0.0


    input_response_file = output_dir / "ask_predict_input_response.txt"
    ans_text = extract_answer_text(input_response_file)
    if not ans_text:
        print(f"    No input prediction found")
        return 0.0

    print(f"    Raw answer: {repr(ans_text)}")


    ans_text = ans_text.strip()


    mutated_src = replace_first_arg_with_new_args(runnable_src, node_info, ans_text)


    print(f"    Executing mutated program...")
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

    print(f"    Backward reasoning result: {success}")
    return 1.0 if success else 0.0

def evaluate_forward_reasoning(program_obj, output_dir: Path):
    """Evaluate forward reasoning (coverage prediction)"""
    runnable_src = program_obj["runnable_script"]
    true_covered = set(program_obj["coverage_metadata"]["covered_lines"])
    total_lines = get_total_lines_of_code(runnable_src)


    true_one_hot = create_one_hot_encoding(true_covered, total_lines)


    pred_one_hot = parse_predicted_coverage(output_dir, total_lines)
    if pred_one_hot is None:
        return 0.0


    similarity = calculate_jaccard_similarity(true_one_hot, pred_one_hot)

    return similarity

def normalize_task_id(dataset, dir_name):
    """Convert directory name to standardized task ID format"""
    if dataset.lower() == "cruxeval":
        if dir_name.startswith("CRUXEval_"):
            number = dir_name.replace("CRUXEval_", "")
            return f"CRUXEval/{number}"
    elif dataset.lower() == "humaneval":
        if dir_name.startswith("HumanEval_"):
            number = dir_name.replace("HumanEval_", "")
            return f"HumanEval/{number}"
    elif dataset.lower() == "pythonsaga":
        if dir_name.startswith("PythonSaga_"):
            number = dir_name.replace("PythonSaga_", "")
            return f"PythonSaga/{number}"

    return f"{dataset}/{dir_name}"

def evaluate_pass_at_k(program_obj, output_base_dir: Path, k: int):
    """Evaluate pass@k metrics for a program"""
    results = []

    for i in range(1, k + 1):
        output_dir = output_base_dir / f"output_{i}"
        if not output_dir.exists():
            continue

        print(f"    Evaluating output {i}...")

        try:
            forward_score = evaluate_forward_reasoning(program_obj, output_dir)
            backward_score = evaluate_backward_reasoning(program_obj, output_dir)


            strict_forward = 1.0 if forward_score >= 0.999 else 0.0

            strict_backward = backward_score

            results.append({
                "output_id": i,
                "forward_score": forward_score,
                "backward_score": backward_score,
                "strict_forward": strict_forward,
                "strict_backward": strict_backward
            })

            print(f"      Output {i}: F{forward_score:.3f} B{backward_score:.3f} SF{strict_forward} SB{strict_backward}")

        except Exception as e:
            print(f"    Error evaluating output {i}: {e}")
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

def process_all_experiments(coverage_json: Path, model_outputs_dir: Path):
    """Process all backup experiments and calculate pass@1 and pass@5 metrics"""

    if not coverage_json.exists():
        print(f"Missing {coverage_json}")
        return []

    try:
        with open(coverage_json, "r", encoding="utf-8") as f:
            programs_data = json.load(f)
    except:
        print("Error loading coverage JSON")
        return []


    coverage_lookup = {}
    for prog in programs_data:
        coverage_lookup[prog["task_id"]] = prog

    results = []


    for backup_path in [model_outputs_dir]:
        backup_dir = backup_path.name
        if not backup_path.exists():
            continue

        dataset = backup_dir.replace("successful_", "").replace("_4llms_1out", "").replace("passat5_", "")
        print(f"Processing: {dataset}")


        for model in MODELS:
            model_path = backup_path / model
            if not model_path.exists():
                continue

            print(f"  Model: {model}")


            program_dirs = list(model_path.iterdir())
            for program_dir in program_dirs:
                if not program_dir.is_dir():
                    continue


                dir_name = program_dir.name
                normalized_task_id = normalize_task_id(dataset, dir_name)


                program_obj = coverage_lookup.get(normalized_task_id)
                if not program_obj:

                    for task_id, prog in coverage_lookup.items():
                        if task_id.replace("/", "_") == dir_name:
                            program_obj = prog
                            break
                    if not program_obj:
                        continue

                print(f"    Program: {normalized_task_id}")


                pass1_results = evaluate_pass_at_k(program_obj, program_dir, 1)


                pass5_results = evaluate_pass_at_k(program_obj, program_dir, 5)

                if pass1_results:
                    pass1_metrics = calculate_pass_at_k_metrics(pass1_results, 1)
                else:
                    pass1_metrics = {
                        "strict_forward": 0.0, "strict_backward": 0.0, "strict_overall": 0.0,
                        "relaxed_forward": 0.0, "relaxed_backward": 0.0, "relaxed_overall": 0.0
                    }

                if pass5_results:
                    pass5_metrics = calculate_pass_at_k_metrics(pass5_results, 5)
                else:
                    pass5_metrics = {
                        "strict_forward": 0.0, "strict_backward": 0.0, "strict_overall": 0.0,
                        "relaxed_forward": 0.0, "relaxed_backward": 0.0, "relaxed_overall": 0.0
                    }

                results.append({
                    "dataset": dataset,
                    "model": model,
                    "task_id": normalized_task_id,


                    "pass1_strict_forward": pass1_metrics["strict_forward"],
                    "pass1_strict_backward": pass1_metrics["strict_backward"],
                    "pass1_strict_overall": pass1_metrics["strict_overall"],
                    "pass1_relaxed_forward": pass1_metrics["relaxed_forward"],
                    "pass1_relaxed_backward": pass1_metrics["relaxed_backward"],
                    "pass1_relaxed_overall": pass1_metrics["relaxed_overall"],


                    "pass5_strict_forward": pass5_metrics["strict_forward"],
                    "pass5_strict_backward": pass5_metrics["strict_backward"],
                    "pass5_strict_overall": pass5_metrics["strict_overall"],
                    "pass5_relaxed_forward": pass5_metrics["relaxed_forward"],
                    "pass5_relaxed_backward": pass5_metrics["relaxed_backward"],
                    "pass5_relaxed_overall": pass5_metrics["relaxed_overall"]
                })

                print(f"      Pass@1: SF{pass1_metrics['strict_forward']} SB{pass1_metrics['strict_backward']} SO{pass1_metrics['strict_overall']:.3f} RF{pass1_metrics['relaxed_forward']:.3f} RB{pass1_metrics['relaxed_backward']} RO{pass1_metrics['relaxed_overall']:.3f}")
                print(f"      Pass@5: SF{pass5_metrics['strict_forward']} SB{pass5_metrics['strict_backward']} SO{pass5_metrics['strict_overall']:.3f} RF{pass5_metrics['relaxed_forward']:.3f} RB{pass5_metrics['relaxed_backward']} RO{pass5_metrics['relaxed_overall']:.3f}")

    print(f"Total evaluations: {len(results)}")
    return results

def create_separate_tables(df, output_dir):
    """Create two separate tables for Pass@1 and Pass@5 metrics"""


    pass1_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        row = [model]
        row.extend([
            f"{model_data['pass1_strict_forward'].mean():.3f}",
            f"{model_data['pass1_strict_backward'].mean():.3f}",
            f"{model_data['pass1_strict_overall'].mean():.3f}",
            f"{model_data['pass1_relaxed_forward'].mean():.3f}",
            f"{model_data['pass1_relaxed_backward'].mean():.3f}",
            f"{model_data['pass1_relaxed_overall'].mean():.3f}",
            f"{len(model_data):d}"
        ])
        pass1_data.append(row)

    pass1_df = pd.DataFrame(pass1_data,
                           columns=['Model',
                                   'Pass1_Strict_Forward', 'Pass1_Strict_Backward', 'Pass1_Strict_Overall',
                                   'Pass1_Relaxed_Forward', 'Pass1_Relaxed_Backward', 'Pass1_Relaxed_Overall',
                                   'N'])
    pass1_df.to_csv(output_dir / 'pass1_metrics.csv', index=False)


    pass5_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        row = [model]
        row.extend([
            f"{model_data['pass5_strict_forward'].mean():.3f}",
            f"{model_data['pass5_strict_backward'].mean():.3f}",
            f"{model_data['pass5_strict_overall'].mean():.3f}",
            f"{model_data['pass5_relaxed_forward'].mean():.3f}",
            f"{model_data['pass5_relaxed_backward'].mean():.3f}",
            f"{model_data['pass5_relaxed_overall'].mean():.3f}",
            f"{len(model_data):d}"
        ])
        pass5_data.append(row)

    pass5_df = pd.DataFrame(pass5_data,
                           columns=['Model',
                                   'Pass5_Strict_Forward', 'Pass5_Strict_Backward', 'Pass5_Strict_Overall',
                                   'Pass5_Relaxed_Forward', 'Pass5_Relaxed_Backward', 'Pass5_Relaxed_Overall',
                                   'N'])
    pass5_df.to_csv(output_dir / 'pass5_metrics.csv', index=False)


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

    ax1.set_title('Pass@1 Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)


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

    ax2.set_title('Pass@5 Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'separate_metrics_tables.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results, report_dir: Path):
    """Generate evaluation report with all six metrics"""

    if not results:
        print("No results!")
        return None

    df = pd.DataFrame(results)


    report_dir.mkdir(parents=True, exist_ok=True)


    df.to_json(report_dir / "detailed_results.json", indent=2, orient="records")
    df.to_csv(report_dir / "detailed_results.csv", index=False)


    pass1_df, pass5_df = create_separate_tables(df, report_dir)


    print("=" * 80)
    print("COMPREHENSIVE PASS@K EVALUATION RESULTS")
    print("=" * 80)

    print("\nPASS@1 METRICS (Averages):")
    pass1_summary = df.groupby('model')[[
        'pass1_strict_forward', 'pass1_strict_backward', 'pass1_strict_overall',
        'pass1_relaxed_forward', 'pass1_relaxed_backward', 'pass1_relaxed_overall'
    ]].mean()
    print(pass1_summary.round(3))

    print("\nPASS@5 METRICS (Averages):")
    pass5_summary = df.groupby('model')[[
        'pass5_strict_forward', 'pass5_strict_backward', 'pass5_strict_overall',
        'pass5_relaxed_forward', 'pass5_relaxed_backward', 'pass5_relaxed_overall'
    ]].mean()
    print(pass5_summary.round(3))

    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved model outputs and generate pass@k reports."
    )

    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path(
            "artifacts/programs/runner_programs_with_coverage.json"
        ),
        help="Path to the program coverage JSON file."
    )

    parser.add_argument(
        "--model-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs"),
        help="Directory containing saved model outputs."
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("evaluation_reports"),
        help="Directory where evaluation reports will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()
    print("Starting comprehensive pass@k evaluation...")

    results = process_all_experiments(args.coverage_json, args.model_outputs_dir)

    if not results:
        print("No results found!")
        return

    df = generate_report(results, args.report_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.report_dir}")
    print(f"Total evaluations: {len(results)}")
    print("\nGenerated files:")
    print("  - pass1_metrics.csv (Pass@1 metrics table)")
    print("  - pass5_metrics.csv (Pass@5 metrics table)")
    print("  - separate_metrics_tables.png (visualization of both tables)")
    print("  - detailed_results.json (raw evaluation data)")
    print("  - detailed_results.csv (raw evaluation data in CSV)")

if __name__ == "__main__":
    main()
