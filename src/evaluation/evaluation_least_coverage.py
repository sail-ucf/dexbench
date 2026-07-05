import ast
import json
import re
import runpy
import tempfile
import signal
from pathlib import Path
from coverage import Coverage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

MODEL = "gpt-5-mini"


ANSWER_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def run_script_with_timeout(code_str: str, timeout=1):
    """Run code with 1 second timeout"""
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

            pass
        except Exception:
            pass

        cov.stop()
        cov.save()

        try:
            lookup_path = os.path.realpath(tmp_path)
            lines = cov.get_data().lines(lookup_path)
            if lines:
                executed = set(lines)
        except Exception:
            pass

    except TimeoutError:
        pass
    except Exception:
        pass
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
                    return (node, arg0, func_text, arg0_src, call_src)
    except Exception:
        pass
    return None

def replace_first_arg_with_new_args(script_source: str, node_info, new_args_text: str):
    """Replace the first argument of the assertion with predicted args."""
    node, arg0_node, func_text, arg0_src, call_src = node_info
    if arg0_node is None:
        return script_source

    new_args_text = new_args_text.strip()

    if func_text and arg0_src:
        new_call_text = f"{func_text}({new_args_text})"
        return script_source.replace(arg0_src, new_call_text, 1)
    elif arg0_src:
        return script_source.replace(arg0_src, new_args_text, 1)

    return script_source

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
    except Exception:
        pass
    return set()

def parse_predicted_coverage_simple(content: str, total_lines: int):
    """Parse model's predicted coverage - SIMPLE EXTRACTION"""
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
            return pred_lines

    return None

def evaluate_forward_reasoning_single(output_dir: Path, program_obj):
    """Evaluate forward reasoning for a single output"""
    runnable_src = program_obj["runnable_script"]
    true_covered = set(program_obj["coverage_metadata"]["covered_lines"])


    coverage_file = output_dir / "ask_predict_coverage_response.txt"
    if not coverage_file.exists():
        return 0.0

    try:
        content = coverage_file.read_text(encoding="utf-8").strip()
    except:
        return 0.0

    total_lines = len(runnable_src.splitlines())


    pred_lines = parse_predicted_coverage_simple(content, total_lines)
    if pred_lines is None:
        return 0.0


    true_set = set(true_covered)
    intersection = len(true_set & pred_lines)
    union = len(true_set | pred_lines)

    if union == 0:
        return 0.0

    return intersection / union

def evaluate_backward_reasoning_single(output_dir: Path, program_obj, priority_line: int, approach: str):
    """Evaluate backward reasoning for a single output"""
    runnable_src = program_obj["runnable_script"]


    node_info = find_assert_call_node_and_source(runnable_src)
    if not node_info:
        return 0.0


    if approach == "least_coverage":
        input_file = output_dir / "least_coverage_response.txt"
    else:
        input_file = output_dir / "ask_predict_input_response.txt"

    ans_text = extract_answer_text(input_file)
    if not ans_text:
        return 0.0


    mutated_src = replace_first_arg_with_new_args(runnable_src, node_info, ans_text.strip())


    executed_lines = run_script_with_timeout(mutated_src, timeout=1)


    line_type = get_line_type(runnable_src, priority_line)

    if line_type == "return":

        success = priority_line in executed_lines
    elif line_type == "control":

        body_lines = get_control_body_lines(runnable_src, priority_line)
        body_executed = body_lines & executed_lines
        success = len(body_executed) > 0
    else:

        success = priority_line in executed_lines

    return 1.0 if success else 0.0

def load_and_merge_programs(least_coverage_json: Path, original_coverage_json: Path):
    """Load both JSON files and merge metadata"""

    if not least_coverage_json.exists():
        print(f"Missing Least Coverage file: {least_coverage_json}")
        return []

    if not original_coverage_json.exists():
        print(f"Missing original coverage file: {original_coverage_json}")
        return []

    try:

        with open(least_coverage_json, "r", encoding="utf-8") as f:
            lc_programs = json.load(f)
        print(f"Loaded {len(lc_programs)} Least Coverage programs")


        with open(original_coverage_json, "r", encoding="utf-8") as f:
            original_programs = json.load(f)
        print(f"Loaded {len(original_programs)} original programs")


        original_lookup = {prog["task_id"]: prog for prog in original_programs}


        merged_programs = []
        for lc_prog in lc_programs:
            task_id = lc_prog["task_id"]
            if task_id in original_lookup:
                original_prog = original_lookup[task_id]


                merged_prog = lc_prog.copy()
                merged_prog["coverage_metadata"] = lc_prog["coverage_metadata"].copy()


                if "advanced_priority_line" in original_prog["coverage_metadata"]:
                    merged_prog["coverage_metadata"]["advanced_priority_line"] = original_prog["coverage_metadata"]["advanced_priority_line"]
                else:

                    merged_prog["coverage_metadata"]["advanced_priority_line"] = lc_prog["coverage_metadata"]["priority_line"]

                merged_programs.append(merged_prog)
            else:
                print(f"Program {task_id} not found in original data, skipping")

        print(f"Merged {len(merged_programs)} programs with complete metadata")
        return merged_programs

    except Exception as e:
        print(f"Error loading/merging JSON files: {e}")
        return []

def evaluate_single_program(program, program_dir: Path, priority_line: int, approach: str, original_output_dir: Path = None):
    """Evaluate a single program for both Pass@1 and Pass@5"""


    pass1_forward_scores = []
    pass1_backward_scores = []

    output_1_dir = program_dir / "output_1"
    if output_1_dir.exists():

        if approach == "least_coverage" and original_output_dir:

            original_forward_dir = original_output_dir / "output_1"
            forward_score = evaluate_forward_reasoning_single(original_forward_dir, program)
        else:

            forward_score = evaluate_forward_reasoning_single(output_1_dir, program)

        backward_score = evaluate_backward_reasoning_single(output_1_dir, program, priority_line, approach)
        pass1_forward_scores.append(forward_score)
        pass1_backward_scores.append(backward_score)


    pass1_strict_forward = 1.0 if pass1_forward_scores and max(pass1_forward_scores) >= 0.999 else 0.0
    pass1_strict_backward = 1.0 if pass1_backward_scores and max(pass1_backward_scores) == 1.0 else 0.0
    pass1_strict_overall = pass1_strict_forward * pass1_strict_backward
    pass1_relaxed_forward = max(pass1_forward_scores) if pass1_forward_scores else 0.0
    pass1_relaxed_backward = max(pass1_backward_scores) if pass1_backward_scores else 0.0
    pass1_relaxed_overall = pass1_relaxed_forward * pass1_relaxed_backward


    pass5_forward_scores = []
    pass5_backward_scores = []

    for i in range(1, 6):
        output_dir = program_dir / f"output_{i}"
        if output_dir.exists():

            if approach == "least_coverage" and original_output_dir:

                original_forward_dir = original_output_dir / f"output_{i}"
                forward_score = evaluate_forward_reasoning_single(original_forward_dir, program)
            else:

                forward_score = evaluate_forward_reasoning_single(output_dir, program)

            backward_score = evaluate_backward_reasoning_single(output_dir, program, priority_line, approach)
            pass5_forward_scores.append(forward_score)
            pass5_backward_scores.append(backward_score)


    pass5_strict_forward = 1.0 if any(f >= 0.999 for f in pass5_forward_scores) else 0.0
    pass5_strict_backward = 1.0 if any(b == 1.0 for b in pass5_backward_scores) else 0.0
    pass5_strict_overall = 1.0 if (pass5_strict_forward == 1.0 and pass5_strict_backward == 1.0) else 0.0
    pass5_relaxed_forward = max(pass5_forward_scores) if pass5_forward_scores else 0.0
    pass5_relaxed_backward = max(pass5_backward_scores) if pass5_backward_scores else 0.0
    pass5_relaxed_overall = pass5_relaxed_forward * pass5_relaxed_backward

    return {
        "pass1_strict_forward": pass1_strict_forward,
        "pass1_strict_backward": pass1_strict_backward,
        "pass1_strict_overall": pass1_strict_overall,
        "pass1_relaxed_forward": pass1_relaxed_forward,
        "pass1_relaxed_backward": pass1_relaxed_backward,
        "pass1_relaxed_overall": pass1_relaxed_overall,
        "pass5_strict_forward": pass5_strict_forward,
        "pass5_strict_backward": pass5_strict_backward,
        "pass5_strict_overall": pass5_strict_overall,
        "pass5_relaxed_forward": pass5_relaxed_forward,
        "pass5_relaxed_backward": pass5_relaxed_backward,
        "pass5_relaxed_overall": pass5_relaxed_overall,
    }

def evaluate_all_programs(least_coverage_json: Path, original_coverage_json: Path, least_coverage_outputs_dir: Path, original_outputs_dir: Path):
    """Evaluate all programs for both Least Coverage and Original approaches"""


    merged_programs = load_and_merge_programs(least_coverage_json, original_coverage_json)
    if not merged_programs:
        return None, None

    lc_results = []
    original_results = []

    for program in merged_programs:
        task_id = program["task_id"]
        dataset = program["dataset"]
        dir_name = task_id.replace("/", "_")


        lc_priority_line = int(program["coverage_metadata"]["priority_line"])
        original_priority_line = int(program["coverage_metadata"]["advanced_priority_line"])


        lc_program_dir = least_coverage_outputs_dir / MODEL / dir_name
        original_program_dir = original_outputs_dir / MODEL / dir_name

        if not lc_program_dir.exists():
            print(f"Least Coverage directory not found for {task_id}")
            continue

        if not original_program_dir.exists():
            print(f"Original directory not found for {task_id}")
            continue

        print(f"Evaluating: {task_id} ({dataset})")



        lc_metrics = evaluate_single_program(
            program, lc_program_dir, lc_priority_line, "least_coverage", original_program_dir
        )
        lc_results.append({
            "dataset": dataset,
            "model": MODEL,
            "task_id": task_id,
            "approach": "least_coverage",
            "priority_line": lc_priority_line,
            **lc_metrics
        })


        original_metrics = evaluate_single_program(
            program, original_program_dir, original_priority_line, "original", None
        )
        original_results.append({
            "dataset": dataset,
            "model": MODEL,
            "task_id": task_id,
            "approach": "original",
            "priority_line": original_priority_line,
            **original_metrics
        })

    return pd.DataFrame(lc_results), pd.DataFrame(original_results)

def create_table_visualization(table_df, title, output_path):
    """Create a single table visualization - MATCHES RAP SCRIPT FORMAT"""
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis('tight')
    ax.axis('off')


    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)


    table_data = table_df.values
    col_labels = table_df.columns.tolist()


    formatted_data = []
    for row in table_data:
        formatted_row = []
        for i, val in enumerate(row):
            if isinstance(val, (int, np.integer)):
                formatted_row.append(str(int(val)))
            elif isinstance(val, float):
                formatted_row.append(f"{val:.3f}")
            else:
                formatted_row.append(str(val))
        formatted_data.append(formatted_row)


    col_widths = [0.15] * len(col_labels)
    col_widths[0] = 0.18
    col_widths[1] = 0.15


    table = ax.table(cellText=formatted_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)


    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)


    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4F81BD')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')


    for i in range(1, len(formatted_data) + 1):
        for j in range(len(col_labels)):
            table[(i, j)].set_edgecolor('lightgray')


            if formatted_data[i-1][1] == 'Overall':
                table[(i, j)].set_facecolor('#E6B8B7')

            elif formatted_data[i-1][1] == 'CRUXEval':
                table[(i, j)].set_facecolor('#FFF2CC')

            elif formatted_data[i-1][1] == 'HumanEval':
                table[(i, j)].set_facecolor('#DDEBF7')

            elif formatted_data[i-1][1] == 'PythonSaga':
                table[(i, j)].set_facecolor('#E2EFDA')

            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_tables(lc_df, original_df, output_dir):
    """Create summary tables for both approaches - MATCHES RAP SCRIPT FORMAT"""


    pass1_data = []


    lc_pass1_row = {
        'LLM': MODEL,
        'Dataset': 'Overall',
        'Approach': 'Least Coverage',
        'Strict Forward': lc_df['pass1_strict_forward'].mean(),
        'Strict Backward': lc_df['pass1_strict_backward'].mean(),
        'Strict Overall': lc_df['pass1_strict_overall'].mean(),
        'Relaxed Forward': lc_df['pass1_relaxed_forward'].mean(),
        'Relaxed Backward': lc_df['pass1_relaxed_backward'].mean(),
        'Relaxed Overall': lc_df['pass1_relaxed_overall'].mean(),
        'N': len(lc_df)
    }
    pass1_data.append(lc_pass1_row)


    original_pass1_row = {
        'LLM': MODEL,
        'Dataset': 'Overall',
        'Approach': 'Original',
        'Strict Forward': original_df['pass1_strict_forward'].mean(),
        'Strict Backward': original_df['pass1_strict_backward'].mean(),
        'Strict Overall': original_df['pass1_strict_overall'].mean(),
        'Relaxed Forward': original_df['pass1_relaxed_forward'].mean(),
        'Relaxed Backward': original_df['pass1_relaxed_backward'].mean(),
        'Relaxed Overall': original_df['pass1_relaxed_overall'].mean(),
        'N': len(original_df)
    }
    pass1_data.append(original_pass1_row)


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_data = lc_df[lc_df['dataset'] == dataset]
        if len(dataset_data) > 0:
            pass1_data.append({
                'LLM': MODEL,
                'Dataset': dataset,
                'Approach': 'Least Coverage',
                'Strict Forward': dataset_data['pass1_strict_forward'].mean(),
                'Strict Backward': dataset_data['pass1_strict_backward'].mean(),
                'Strict Overall': dataset_data['pass1_strict_overall'].mean(),
                'Relaxed Forward': dataset_data['pass1_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_data['pass1_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_data['pass1_relaxed_overall'].mean(),
                'N': len(dataset_data)
            })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_data = original_df[original_df['dataset'] == dataset]
        if len(dataset_data) > 0:
            pass1_data.append({
                'LLM': MODEL,
                'Dataset': dataset,
                'Approach': 'Original',
                'Strict Forward': dataset_data['pass1_strict_forward'].mean(),
                'Strict Backward': dataset_data['pass1_strict_backward'].mean(),
                'Strict Overall': dataset_data['pass1_strict_overall'].mean(),
                'Relaxed Forward': dataset_data['pass1_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_data['pass1_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_data['pass1_relaxed_overall'].mean(),
                'N': len(dataset_data)
            })

    pass1_df = pd.DataFrame(pass1_data)


    pass5_data = []


    lc_pass5_row = {
        'LLM': MODEL,
        'Dataset': 'Overall',
        'Approach': 'Least Coverage',
        'Strict Forward': lc_df['pass5_strict_forward'].mean(),
        'Strict Backward': lc_df['pass5_strict_backward'].mean(),
        'Strict Overall': lc_df['pass5_strict_overall'].mean(),
        'Relaxed Forward': lc_df['pass5_relaxed_forward'].mean(),
        'Relaxed Backward': lc_df['pass5_relaxed_backward'].mean(),
        'Relaxed Overall': lc_df['pass5_relaxed_overall'].mean(),
        'N': len(lc_df)
    }
    pass5_data.append(lc_pass5_row)


    original_pass5_row = {
        'LLM': MODEL,
        'Dataset': 'Overall',
        'Approach': 'Original',
        'Strict Forward': original_df['pass5_strict_forward'].mean(),
        'Strict Backward': original_df['pass5_strict_backward'].mean(),
        'Strict Overall': original_df['pass5_strict_overall'].mean(),
        'Relaxed Forward': original_df['pass5_relaxed_forward'].mean(),
        'Relaxed Backward': original_df['pass5_relaxed_backward'].mean(),
        'Relaxed Overall': original_df['pass5_relaxed_overall'].mean(),
        'N': len(original_df)
    }
    pass5_data.append(original_pass5_row)


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_data = lc_df[lc_df['dataset'] == dataset]
        if len(dataset_data) > 0:
            pass5_data.append({
                'LLM': MODEL,
                'Dataset': dataset,
                'Approach': 'Least Coverage',
                'Strict Forward': dataset_data['pass5_strict_forward'].mean(),
                'Strict Backward': dataset_data['pass5_strict_backward'].mean(),
                'Strict Overall': dataset_data['pass5_strict_overall'].mean(),
                'Relaxed Forward': dataset_data['pass5_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_data['pass5_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_data['pass5_relaxed_overall'].mean(),
                'N': len(dataset_data)
            })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_data = original_df[original_df['dataset'] == dataset]
        if len(dataset_data) > 0:
            pass5_data.append({
                'LLM': MODEL,
                'Dataset': dataset,
                'Approach': 'Original',
                'Strict Forward': dataset_data['pass5_strict_forward'].mean(),
                'Strict Backward': dataset_data['pass5_strict_backward'].mean(),
                'Strict Overall': dataset_data['pass5_strict_overall'].mean(),
                'Relaxed Forward': dataset_data['pass5_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_data['pass5_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_data['pass5_relaxed_overall'].mean(),
                'N': len(dataset_data)
            })

    pass5_df = pd.DataFrame(pass5_data)


    pass1_df.to_csv(output_dir / 'least_coverage_comparison_pass1_summary.csv', index=False)
    pass5_df.to_csv(output_dir / 'least_coverage_comparison_pass5_summary.csv', index=False)


    create_table_visualization(
        pass1_df,
        "LEAST COVERAGE vs ORIGINAL COMPARISON: Pass@1 Results\nGPT-5-mini with Different Priority Lines\n(Forward Reasoning from Original Experiment)",
        output_dir / 'least_coverage_comparison_pass1_table.png'
    )

    create_table_visualization(
        pass5_df,
        "LEAST COVERAGE vs ORIGINAL COMPARISON: Pass@5 Results\nGPT-5-mini with Different Priority Lines\n(Forward Reasoning from Original Experiment)",
        output_dir / 'least_coverage_comparison_pass5_table.png'
    )

    return pass1_df, pass5_df

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Least Coverage evaluation results against "
            "the original evaluation results."
        )
    )

    parser.add_argument(
        "--least-coverage-json",
        type=Path,
        default=Path(
            "artifacts/programs/runner_programs_with_least_coverage.json"
        ),
        help="Path to the Least Coverage program JSON file."
    )

    parser.add_argument(
        "--original-coverage-json",
        type=Path,
        default=Path(
            "artifacts/programs/runner_programs_with_coverage.json"
        ),
        help="Path to the original program coverage JSON file."
    )

    parser.add_argument(
        "--least-coverage-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs_Least_Coverage"),
        help="Directory containing Least Coverage model outputs."
    )

    parser.add_argument(
        "--original-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs"),
        help="Directory containing original model outputs."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("least_coverage_evaluation_reports"),
        help="Directory where evaluation reports will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()
        
    print("=" * 80)
    print("COMPREHENSIVE LEAST COVERAGE vs ORIGINAL EVALUATION")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Least Coverage Directory: "f"{args.least_coverage_outputs_dir}")
    print(f"Original Directory: {args.original_outputs_dir}")
    print("\nNOTE: Using forward reasoning from Original experiment")
    print("   for Least Coverage approach (same as RAP experiment)")
    print("=" * 80)


    lc_df, original_df = evaluate_all_programs(args.least_coverage_json, args.original_coverage_json, args.least_coverage_outputs_dir, args.original_outputs_dir)

    if lc_df is None or original_df is None:
        print("No results to process!")
        return

    print(f"\n Evaluated {len(lc_df)} programs for Least Coverage approach")
    print(f"Evaluated {len(original_df)} programs for Original approach")


    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)


    lc_df.to_csv(output_dir / "least_coverage_detailed_results.csv", index=False)
    original_df.to_csv(output_dir / "original_detailed_results.csv", index=False)


    pass1_summary, pass5_summary = create_summary_tables(lc_df, original_df, output_dir)


    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)


    print(f"\n OVERALL COMPARISON (All {len(lc_df)} programs):")
    print(f"   Least Coverage Approach (Hardest Paths):")
    print(f"     Pass@1 Strict Overall: {lc_df['pass1_strict_overall'].mean():.3f}")
    print(f"     Pass@1 Strict Forward: {lc_df['pass1_strict_forward'].mean():.3f} (from Original)")
    print(f"     Pass@1 Strict Backward: {lc_df['pass1_strict_backward'].mean():.3f}")
    print(f"     Pass@5 Strict Overall: {lc_df['pass5_strict_overall'].mean():.3f}")

    print(f"\n   Original Approach (Priority Paths):")
    print(f"     Pass@1 Strict Overall: {original_df['pass1_strict_overall'].mean():.3f}")
    print(f"     Pass@1 Strict Forward: {original_df['pass1_strict_forward'].mean():.3f}")
    print(f"     Pass@1 Strict Backward: {original_df['pass1_strict_backward'].mean():.3f}")
    print(f"     Pass@5 Strict Overall: {original_df['pass5_strict_overall'].mean():.3f}")

    print(f"\n   DIFFERENCE (Least Coverage - Original):")
    print(f"     Pass@1 Strict Overall: {lc_df['pass1_strict_overall'].mean() - original_df['pass1_strict_overall'].mean():+.3f}")
    print(f"     Pass@1 Strict Backward: {lc_df['pass1_strict_backward'].mean() - original_df['pass1_strict_backward'].mean():+.3f}")
    print(f"     Pass@5 Strict Overall: {lc_df['pass5_strict_overall'].mean() - original_df['pass5_strict_overall'].mean():+.3f}")


    print(f"\n DATASET BREAKDOWN:")
    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        lc_dataset = lc_df[lc_df['dataset'] == dataset]
        orig_dataset = original_df[original_df['dataset'] == dataset]

        if len(lc_dataset) > 0 and len(orig_dataset) > 0:
            print(f"\n   {dataset}: {len(lc_dataset)} programs")
            print(f"     Least Coverage Pass@1 Strict Backward: {lc_dataset['pass1_strict_backward'].mean():.3f}")
            print(f"     Original Pass@1 Strict Backward: {orig_dataset['pass1_strict_backward'].mean():.3f}")
            print(f"     Difference: {lc_dataset['pass1_strict_backward'].mean() - orig_dataset['pass1_strict_backward'].mean():+.3f}")


    print("\n" + "=" * 80)
    print("DUALITY ANALYSIS (Forward vs Backward Correlation)")
    print("=" * 80)
    print("For Least Coverage (using Original's forward reasoning):")
    print(f"  Strict Forward Score: {lc_df['pass1_strict_forward'].mean():.3f}")
    print(f"  Strict Backward Score: {lc_df['pass1_strict_backward'].mean():.3f}")
    print(f"  Duality Gap: {lc_df['pass1_strict_backward'].mean() - lc_df['pass1_strict_overall'].mean():.3f}")

    print("\nFor Original:")
    print(f"  Strict Forward Score: {original_df['pass1_strict_forward'].mean():.3f}")
    print(f"  Strict Backward Score: {original_df['pass1_strict_backward'].mean():.3f}")
    print(f"  Duality Gap: {original_df['pass1_strict_backward'].mean() - original_df['pass1_strict_overall'].mean():.3f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n Results saved to: {output_dir}/")
    print("\n Generated files:")
    print("  Detailed Results:")
    print("    - least_coverage_detailed_results.csv (Least Coverage approach)")
    print("    - original_detailed_results.csv (Original approach)")
    print("\n  Summary Tables (CSV):")
    print("    - least_coverage_comparison_pass1_summary.csv (Pass@1 summary)")
    print("    - least_coverage_comparison_pass5_summary.csv (Pass@5 summary)")
    print("\n  Visualizations (PNG):")
    print("    - least_coverage_comparison_pass1_table.png (Pass@1 table)")
    print("    - least_coverage_comparison_pass5_table.png (Pass@5 table)")

if __name__ == "__main__":
    main()