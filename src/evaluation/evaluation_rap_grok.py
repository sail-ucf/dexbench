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

                    print(f"    Inspecting Found assertion: {call_src}")
                    print(f"    Inspecting First argument: {arg0_src}")
                    print(f"    Inspecting Function in arg: {func_text}")

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

    print(f"    Replacing Replacing: {arg0_src} with {new_args_text}")

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
    """Evaluate backward reasoning (input prediction)"""
    runnable_src = program_obj["runnable_script"]


    if approach == "rap":
        priority_line = int(program_obj["coverage_metadata"]["priority_line"])
    else:
        priority_line = int(program_obj["coverage_metadata"]["advanced_priority_line"])

    print(f"    Evaluating {approach.upper()} backward reasoning for line {priority_line}")


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

    print(f"    {approach.upper()} backward reasoning result: {success}")
    return 1.0 if success else 0.0

def evaluate_forward_reasoning(program_obj, output_dir: Path, approach: str):
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

def get_rap_programs(rap_coverage_json: Path):
    """Get the list of 87 programs that have RAP support"""
    if not rap_coverage_json.exists():
        print(f"Missing {rap_coverage_json}")
        return []

    try:
        with open(rap_coverage_json, "r", encoding="utf-8") as f:
            rap_programs = json.load(f)
        print(f"Loaded {len(rap_programs)} RAP programs")
        return rap_programs
    except Exception as e:
        print(f"Error loading RAP JSON: {e}")
        return []

def get_original_coverage_lookup(original_coverage_json: Path):
    """Create lookup dictionary for original coverage data"""
    if not original_coverage_json.exists():
        print(f"Missing {original_coverage_json}")
        return {}

    try:
        with open(original_coverage_json, "r", encoding="utf-8") as f:
            original_programs = json.load(f)
        coverage_lookup = {prog["task_id"]: prog for prog in original_programs}
        print(f"Loaded {len(coverage_lookup)} original programs")
        return coverage_lookup
    except Exception as e:
        print(f"Error loading original JSON: {e}")
        return {}

def process_comparison_experiments(rap_coverage_json: Path, original_coverage_json: Path, rap_outputs_dir: Path, original_outputs_dir: Path):
    """Process both RAP and Original experiments for the same 87 programs"""


    rap_programs = get_rap_programs(rap_coverage_json)
    if not rap_programs:
        return []


    original_lookup = get_original_coverage_lookup(original_coverage_json)
    if not original_lookup:
        return []

    results = []


    for rap_program in rap_programs:
        task_id = rap_program["task_id"]
        dir_name = task_id.replace("/", "_")

        print(f"Inspecting Processing program: {task_id}")


        original_program = original_lookup.get(task_id)
        if not original_program:
            print(f"   Program {task_id} not found in original data, skipping")
            continue


        rap_program_dir = rap_outputs_dir / MODEL / dir_name
        original_program_dir = original_outputs_dir / MODEL / dir_name

        if not rap_program_dir.exists():
            print(f"   RAP directory not found: {rap_program_dir}")
            continue
        if not original_program_dir.exists():
            print(f"   Original directory not found: {original_program_dir}")
            continue

        print(f"   Found both RAP and Original directories")


        rap_pass1_results = evaluate_pass_at_k(rap_program, rap_program_dir, 1, "rap")
        rap_pass5_results = evaluate_pass_at_k(rap_program, rap_program_dir, 5, "rap")


        original_pass1_results = evaluate_pass_at_k(original_program, original_program_dir, 1, "original")
        original_pass5_results = evaluate_pass_at_k(original_program, original_program_dir, 5, "original")


        rap_pass1_metrics = calculate_pass_at_k_metrics(rap_pass1_results, 1) if rap_pass1_results else create_empty_metrics()
        rap_pass5_metrics = calculate_pass_at_k_metrics(rap_pass5_results, 5) if rap_pass5_results else create_empty_metrics()

        original_pass1_metrics = calculate_pass_at_k_metrics(original_pass1_results, 1) if original_pass1_results else create_empty_metrics()
        original_pass5_metrics = calculate_pass_at_k_metrics(original_pass5_results, 5) if original_pass5_results else create_empty_metrics()


        results.append({
            "task_id": task_id,
            "dataset": task_id.split("/")[0],


            "rap_pass1_strict_forward": rap_pass1_metrics["strict_forward"],
            "rap_pass1_strict_backward": rap_pass1_metrics["strict_backward"],
            "rap_pass1_strict_overall": rap_pass1_metrics["strict_overall"],
            "rap_pass1_relaxed_forward": rap_pass1_metrics["relaxed_forward"],
            "rap_pass1_relaxed_backward": rap_pass1_metrics["relaxed_backward"],
            "rap_pass1_relaxed_overall": rap_pass1_metrics["relaxed_overall"],

            "rap_pass5_strict_forward": rap_pass5_metrics["strict_forward"],
            "rap_pass5_strict_backward": rap_pass5_metrics["strict_backward"],
            "rap_pass5_strict_overall": rap_pass5_metrics["strict_overall"],
            "rap_pass5_relaxed_forward": rap_pass5_metrics["relaxed_forward"],
            "rap_pass5_relaxed_backward": rap_pass5_metrics["relaxed_backward"],
            "rap_pass5_relaxed_overall": rap_pass5_metrics["relaxed_overall"],


            "original_pass1_strict_forward": original_pass1_metrics["strict_forward"],
            "original_pass1_strict_backward": original_pass1_metrics["strict_backward"],
            "original_pass1_strict_overall": original_pass1_metrics["strict_overall"],
            "original_pass1_relaxed_forward": original_pass1_metrics["relaxed_forward"],
            "original_pass1_relaxed_backward": original_pass1_metrics["relaxed_backward"],
            "original_pass1_relaxed_overall": original_pass1_metrics["relaxed_overall"],

            "original_pass5_strict_forward": original_pass5_metrics["strict_forward"],
            "original_pass5_strict_backward": original_pass5_metrics["strict_backward"],
            "original_pass5_strict_overall": original_pass5_metrics["strict_overall"],
            "original_pass5_relaxed_forward": original_pass5_metrics["relaxed_forward"],
            "original_pass5_relaxed_backward": original_pass5_metrics["relaxed_backward"],
            "original_pass5_relaxed_overall": original_pass5_metrics["relaxed_overall"],
        })

        print(f"   Completed comparison for {task_id}")

    print(f"Total comparisons: {len(results)}")
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

def create_dataset_separated_tables(df, output_dir):
    """Create tables separated by dataset for both RAP and Original approaches"""


    datasets = sorted(df['dataset'].unique())




    rap_pass1_data = []
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]

        row = [dataset]
        row.extend([
            f"{dataset_data['rap_pass1_strict_forward'].mean():.3f}",
            f"{dataset_data['rap_pass1_strict_backward'].mean():.3f}",
            f"{dataset_data['rap_pass1_strict_overall'].mean():.3f}",
            f"{dataset_data['rap_pass1_relaxed_forward'].mean():.3f}",
            f"{dataset_data['rap_pass1_relaxed_backward'].mean():.3f}",
            f"{dataset_data['rap_pass1_relaxed_overall'].mean():.3f}",
            f"{len(dataset_data):d}"
        ])
        rap_pass1_data.append(row)


    overall_rap_pass1_row = ['Overall']
    overall_rap_pass1_row.extend([
        f"{df['rap_pass1_strict_forward'].mean():.3f}",
        f"{df['rap_pass1_strict_backward'].mean():.3f}",
        f"{df['rap_pass1_strict_overall'].mean():.3f}",
        f"{df['rap_pass1_relaxed_forward'].mean():.3f}",
        f"{df['rap_pass1_relaxed_backward'].mean():.3f}",
        f"{df['rap_pass1_relaxed_overall'].mean():.3f}",
        f"{len(df):d}"
    ])
    rap_pass1_data.append(overall_rap_pass1_row)

    rap_pass1_df = pd.DataFrame(rap_pass1_data,
                               columns=['Dataset',
                                       'RAP_Pass1_Strict_Forward', 'RAP_Pass1_Strict_Backward', 'RAP_Pass1_Strict_Overall',
                                       'RAP_Pass1_Relaxed_Forward', 'RAP_Pass1_Relaxed_Backward', 'RAP_Pass1_Relaxed_Overall',
                                       'N'])
    rap_pass1_df.to_csv(output_dir / 'grok_rap_pass1_metrics_by_dataset.csv', index=False)


    original_pass1_data = []
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]

        row = [dataset]
        row.extend([
            f"{dataset_data['original_pass1_strict_forward'].mean():.3f}",
            f"{dataset_data['original_pass1_strict_backward'].mean():.3f}",
            f"{dataset_data['original_pass1_strict_overall'].mean():.3f}",
            f"{dataset_data['original_pass1_relaxed_forward'].mean():.3f}",
            f"{dataset_data['original_pass1_relaxed_backward'].mean():.3f}",
            f"{dataset_data['original_pass1_relaxed_overall'].mean():.3f}",
            f"{len(dataset_data):d}"
        ])
        original_pass1_data.append(row)


    overall_original_pass1_row = ['Overall']
    overall_original_pass1_row.extend([
        f"{df['original_pass1_strict_forward'].mean():.3f}",
        f"{df['original_pass1_strict_backward'].mean():.3f}",
        f"{df['original_pass1_strict_overall'].mean():.3f}",
        f"{df['original_pass1_relaxed_forward'].mean():.3f}",
        f"{df['original_pass1_relaxed_backward'].mean():.3f}",
        f"{df['original_pass1_relaxed_overall'].mean():.3f}",
        f"{len(df):d}"
    ])
    original_pass1_data.append(overall_original_pass1_row)

    original_pass1_df = pd.DataFrame(original_pass1_data,
                                    columns=['Dataset',
                                            'Original_Pass1_Strict_Forward', 'Original_Pass1_Strict_Backward', 'Original_Pass1_Strict_Overall',
                                            'Original_Pass1_Relaxed_Forward', 'Original_Pass1_Relaxed_Backward', 'Original_Pass1_Relaxed_Overall',
                                            'N'])
    original_pass1_df.to_csv(output_dir / 'grok_original_pass1_metrics_by_dataset.csv', index=False)




    rap_pass5_data = []
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]

        row = [dataset]
        row.extend([
            f"{dataset_data['rap_pass5_strict_forward'].mean():.3f}",
            f"{dataset_data['rap_pass5_strict_backward'].mean():.3f}",
            f"{dataset_data['rap_pass5_strict_overall'].mean():.3f}",
            f"{dataset_data['rap_pass5_relaxed_forward'].mean():.3f}",
            f"{dataset_data['rap_pass5_relaxed_backward'].mean():.3f}",
            f"{dataset_data['rap_pass5_relaxed_overall'].mean():.3f}",
            f"{len(dataset_data):d}"
        ])
        rap_pass5_data.append(row)


    overall_rap_pass5_row = ['Overall']
    overall_rap_pass5_row.extend([
        f"{df['rap_pass5_strict_forward'].mean():.3f}",
        f"{df['rap_pass5_strict_backward'].mean():.3f}",
        f"{df['rap_pass5_strict_overall'].mean():.3f}",
        f"{df['rap_pass5_relaxed_forward'].mean():.3f}",
        f"{df['rap_pass5_relaxed_backward'].mean():.3f}",
        f"{df['rap_pass5_relaxed_overall'].mean():.3f}",
        f"{len(df):d}"
    ])
    rap_pass5_data.append(overall_rap_pass5_row)

    rap_pass5_df = pd.DataFrame(rap_pass5_data,
                               columns=['Dataset',
                                       'RAP_Pass5_Strict_Forward', 'RAP_Pass5_Strict_Backward', 'RAP_Pass5_Strict_Overall',
                                       'RAP_Pass5_Relaxed_Forward', 'RAP_Pass5_Relaxed_Backward', 'RAP_Pass5_Relaxed_Overall',
                                       'N'])
    rap_pass5_df.to_csv(output_dir / 'grok_rap_pass5_metrics_by_dataset.csv', index=False)


    original_pass5_data = []
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]

        row = [dataset]
        row.extend([
            f"{dataset_data['original_pass5_strict_forward'].mean():.3f}",
            f"{dataset_data['original_pass5_strict_backward'].mean():.3f}",
            f"{dataset_data['original_pass5_strict_overall'].mean():.3f}",
            f"{dataset_data['original_pass5_relaxed_forward'].mean():.3f}",
            f"{dataset_data['original_pass5_relaxed_backward'].mean():.3f}",
            f"{dataset_data['original_pass5_relaxed_overall'].mean():.3f}",
            f"{len(dataset_data):d}"
        ])
        original_pass5_data.append(row)


    overall_original_pass5_row = ['Overall']
    overall_original_pass5_row.extend([
        f"{df['original_pass5_strict_forward'].mean():.3f}",
        f"{df['original_pass5_strict_backward'].mean():.3f}",
        f"{df['original_pass5_strict_overall'].mean():.3f}",
        f"{df['original_pass5_relaxed_forward'].mean():.3f}",
        f"{df['original_pass5_relaxed_backward'].mean():.3f}",
        f"{df['original_pass5_relaxed_overall'].mean():.3f}",
        f"{len(df):d}"
    ])
    original_pass5_data.append(overall_original_pass5_row)

    original_pass5_df = pd.DataFrame(original_pass5_data,
                                    columns=['Dataset',
                                            'Original_Pass5_Strict_Forward', 'Original_Pass5_Strict_Backward', 'Original_Pass5_Strict_Overall',
                                            'Original_Pass5_Relaxed_Forward', 'Original_Pass5_Relaxed_Backward', 'Original_Pass5_Relaxed_Overall',
                                            'N'])
    original_pass5_df.to_csv(output_dir / 'grok_original_pass5_metrics_by_dataset.csv', index=False)


    create_table_visualizations(rap_pass1_df, rap_pass5_df, original_pass1_df, original_pass5_df, output_dir)

    return rap_pass1_df, rap_pass5_df, original_pass1_df, original_pass5_df

def create_table_visualizations(rap_pass1_df, rap_pass5_df, original_pass1_df, original_pass5_df, output_dir):
    """Create visualization tables for the report"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))


    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=rap_pass1_df.values,
                      colLabels=rap_pass1_df.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8'] * len(rap_pass1_df.columns))

    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1, 1.8)

    for i in range(len(rap_pass1_df.columns)):
        table1[(0, i)].set_facecolor('#2E86AB')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    ax1.set_title('Grok: RAP Pass@1 Metrics by Dataset', fontsize=12, fontweight='bold', pad=20)


    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=rap_pass5_df.values,
                      colLabels=rap_pass5_df.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8'] * len(rap_pass5_df.columns))

    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1, 1.8)

    for i in range(len(rap_pass5_df.columns)):
        table2[(0, i)].set_facecolor('#A23B72')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Grok: RAP Pass@5 Metrics by Dataset', fontsize=12, fontweight='bold', pad=20)


    ax3.axis('tight')
    ax3.axis('off')
    table3 = ax3.table(cellText=original_pass1_df.values,
                      colLabels=original_pass1_df.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8'] * len(original_pass1_df.columns))

    table3.auto_set_font_size(False)
    table3.set_fontsize(8)
    table3.scale(1, 1.8)

    for i in range(len(original_pass1_df.columns)):
        table3[(0, i)].set_facecolor('#2E86AB')
        table3[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title('Grok: Original Pass@1 Metrics by Dataset', fontsize=12, fontweight='bold', pad=20)


    ax4.axis('tight')
    ax4.axis('off')
    table4 = ax4.table(cellText=original_pass5_df.values,
                      colLabels=original_pass5_df.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8'] * len(original_pass5_df.columns))

    table4.auto_set_font_size(False)
    table4.set_fontsize(8)
    table4.scale(1, 1.8)

    for i in range(len(original_pass5_df.columns)):
        table4[(0, i)].set_facecolor('#A23B72')
        table4[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Grok: Original Pass@5 Metrics by Dataset', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'grok_rap_original_metrics_tables.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(results, output_dir: Path):
    """Generate comparison report with dataset-separated tables"""

    if not results:
        print("No results!")
        return None

    df = pd.DataFrame(results)


    report_dir = output_dir
    report_dir.mkdir(parents=True, exist_ok=True)


    df.to_json(report_dir / "grok_comparison_detailed_results.json", indent=2, orient="records")
    df.to_csv(report_dir / "grok_comparison_detailed_results.csv", index=False)


    rap_pass1_df, rap_pass5_df, original_pass1_df, original_pass5_df = create_dataset_separated_tables(df, report_dir)


    print("=" * 80)
    print("GROK-4-FAST-REASONING: RAP vs ORIGINAL COMPARISON RESULTS")
    print("=" * 80)
    print(f"Total programs compared: {len(df)}")

    print("\n" + "=" * 80)
    print("RAP PASS@1 RESULTS (by dataset)")
    print("=" * 80)
    print(rap_pass1_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("RAP PASS@5 RESULTS (by dataset)")
    print("=" * 80)
    print(rap_pass5_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ORIGINAL PASS@1 RESULTS (by dataset)")
    print("=" * 80)
    print(original_pass1_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ORIGINAL PASS@5 RESULTS (by dataset)")
    print("=" * 80)
    print(original_pass5_df.to_string(index=False))

    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Grok RAP evaluation results against "
            "the original Grok evaluation results."
        )
    )

    parser.add_argument(
        "--rap-coverage-json",
        type=Path,
        default=Path(
            "artifacts/programs/runner_programs_with_coverage_rap.json"
        ),
        help="Path to the RAP program coverage JSON file."
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
        "--rap-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs_RAP_grok"),
        help="Directory containing Grok RAP outputs."
    )

    parser.add_argument(
        "--original-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs"),
        help="Directory containing original Grok outputs."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grok_rap_comparison_reports"),
        help="Directory where Grok RAP comparison reports will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()

    print("Starting Grok-4-Fast-Reasoning RAP vs Original comparison evaluation...")

    results = process_comparison_experiments(args.rap_coverage_json, args.original_coverage_json, args.rap_outputs_dir, args.original_outputs_dir)

    if not results:
        print("No results found!")
        return

    df = generate_comparison_report(results, args.output_dir)

    print("\n" + "=" * 80)
    print("GROK RAP COMPARISON EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}/")
    print(f"Total comparisons: {len(results)}")
    print("\nGenerated files:")
    print("  - grok_rap_pass1_metrics_by_dataset.csv (RAP Pass@1 by dataset)")
    print("  - grok_rap_pass5_metrics_by_dataset.csv (RAP Pass@5 by dataset)")
    print("  - grok_original_pass1_metrics_by_dataset.csv (Original Pass@1 by dataset)")
    print("  - grok_original_pass5_metrics_by_dataset.csv (Original Pass@5 by dataset)")
    print("  - grok_rap_original_metrics_tables.png (visualization of all tables)")
    print("  - grok_comparison_detailed_results.json (raw comparison data)")
    print("  - grok_comparison_detailed_results.csv (raw comparison data in CSV)")

if __name__ == "__main__":
    main()