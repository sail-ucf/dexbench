import ast
import json
import os
import re
import runpy
import shutil
import tempfile
import signal
import warnings
from pathlib import Path
from coverage import Coverage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

MODEL = "grok-4-fast-reasoning"


warnings.filterwarnings("ignore", message="No data was collected")


ANSWER_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def run_script_with_timeout(code_str: str, timeout=2):
    """Run code with timeout to prevent hanging"""
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
        except Exception as e:
            pass

        cov.stop()

        try:
            cov_data = cov.get_data()
            if cov_data and cov_data.lines(tmp_path):
                executed = set(cov_data.lines(tmp_path))
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
        return m.group(1).strip() if m else txt.strip()
    except:
        return None

def parse_crispe_coverage(path: Path):
    """Parse CRISPE coverage response to get predicted lines"""
    response_file = path / "crispe_coverage_response.txt"
    if not response_file.exists():
        return None

    try:
        content = response_file.read_text(encoding="utf-8")


        patterns = [

            r'\[ANSWER\](.*?)\[/ANSWER\]',

            r'\{.*?\"executed_lines\".*?:.*?\[.*?\]\}',
            r'\{.*?\"coverage\".*?:.*?\[.*?\]\}',

            r'\[[\d\s,\n]+\]',

            r'Lines?[:\s]*\[?([\d\s,]+)\]?',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:

                match_text = matches[0] if isinstance(matches[0], str) else matches[0][0]
                match_text = match_text.strip()


                try:
                    if match_text.startswith('{'):
                        data = json.loads(match_text)
                        if "executed_lines" in data and isinstance(data["executed_lines"], list):
                            return set(data["executed_lines"])
                        elif "coverage" in data and isinstance(data["coverage"], list):
                            return set(data["coverage"])
                    elif match_text.startswith('['):

                        import ast
                        parsed = ast.literal_eval(match_text)
                        if isinstance(parsed, list):
                            return set(parsed)
                except (json.JSONDecodeError, SyntaxError, ValueError):

                    numbers = re.findall(r'\b\d+\b', match_text)
                    if numbers:
                        return set([int(n) for n in numbers])


        numbers = re.findall(r'\b\d+\b', content)
        if numbers:
            return set([int(n) for n in numbers])

    except Exception as e:
        print(f"Error parsing coverage: {e}")

    return None

def calculate_jaccard(true_set, pred_set):
    """Calculate Jaccard similarity between two sets"""
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)
    return intersection / union if union > 0 else 0.0



def find_assert_call_node_and_source(script_source: str):
    """Find the first unittest assertion call in the code"""
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
    except Exception as e:

        pass
    return None

def replace_first_arg_with_new_args(script_source: str, node_info, new_args_text: str):
    """Replace the first argument of the assertion with predicted args"""
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

def evaluate_backward_reasoning(runnable_src: str, pred_input: str, priority_line: int):
    """Evaluate backward reasoning with 1s timeout"""

    node_info = find_assert_call_node_and_source(runnable_src)
    if not node_info or not pred_input:
        return 0.0


    mutated_src = replace_first_arg_with_new_args(runnable_src, node_info, pred_input)


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


def copy_backward_results(original_outputs_dir: Path, crispe_outputs_dir: Path):
    """Copy backward reasoning results from original grok study to CRISPE directories"""
    print("Syncing Backward Reasoning results from grok...")
    source_base = original_outputs_dir
    dest_base = crispe_outputs_dir / MODEL

    if not source_base.exists():
        print(f"Source directory not found: {source_base}")
        return 0

    if not dest_base.exists():
        print(f"Destination directory not found: {dest_base}")
        return 0

    copied = 0
    missing = 0


    for prog_dir in dest_base.iterdir():
        if not prog_dir.is_dir():
            continue

        dir_name = prog_dir.name
        src_prog_dir = source_base / dir_name

        if not src_prog_dir.exists():
            missing += 1
            continue


        for i in range(1, 6):
            src_out = src_prog_dir / f"output_{i}"
            dest_out = prog_dir / f"output_{i}"

            if src_out.exists() and dest_out.exists():

                target_file = "ask_predict_input_response.txt"
                src_file = src_out / target_file

                if src_file.exists():
                    shutil.copy2(src_file, dest_out / target_file)
                    copied += 1

    print(f"Copied {copied} backward results, {missing} programs missing in source")
    return copied

def normalize_program_id(task_id: str, dataset: str):
    """Normalize program ID for consistent lookup"""
    if dataset == "PythonSaga":
        try:
            num = int(task_id.split("/")[-1])
            return f"PythonSaga_{num}"
        except:
            return task_id.replace("/", "_")
    else:
        return task_id.replace("/", "_")

def load_program_data(coverage_json: Path):
    """Load program data and create lookup dictionary with normalized IDs"""
    if not coverage_json.exists():
        print(f"Coverage JSON not found: {coverage_json}")
        return {}

    try:
        with open(coverage_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

    lookup = {}
    for p in data:
        task_id = p.get("task_id", "")
        dataset = p.get("dataset", "")
        normalized_id = normalize_program_id(task_id, dataset)
        lookup[normalized_id] = p

    print(f"Loaded {len(lookup)} programs from coverage data")
    return lookup

def get_datasets_in_crispe(model_dir: Path):
    """Get list of datasets present in CRISPE directory"""
    datasets = set()
    for prog_dir in model_dir.iterdir():
        if not prog_dir.is_dir():
            continue

        dir_name = prog_dir.name
        if dir_name.startswith("CRUXEval_"):
            datasets.add("CRUXEval")
        elif dir_name.startswith("HumanEval_"):
            datasets.add("HumanEval")
        elif dir_name.startswith("PythonSaga_"):
            datasets.add("PythonSaga")

    return sorted(list(datasets))


def create_sensitivity_analysis_table(df_p1, df_p5, output_file, model_name="grok-4-fast-reasoning"):
    """Generate sensitivity analysis visualization with two tables"""


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
    fig.patch.set_visible(False)
    fig.suptitle(f"Sensitivity Analysis: CRISPE Forward + Original Backward\nModel: {model_name}",
                 fontsize=18, fontweight='bold', y=0.98)


    def prepare_table_data(stats_df, title_ax, title_text):
        title_ax.axis('off')


        datasets_order = ["Overall", "CRUXEval", "HumanEval", "PythonSaga"]
        rows = []

        for ds in datasets_order:
            if ds not in stats_df.index:
                continue

            row_data = stats_df.loc[ds]
            rows.append([
                ds,
                f"{row_data['Strict Fwd']:.3f}",
                f"{row_data['Strict Bwd']:.3f}",
                f"{row_data['Strict All']:.3f}",
                f"{row_data['Relax Fwd']:.3f}",
                f"{row_data['Relax Bwd']:.3f}",
                f"{row_data['Relax All']:.3f}",
                int(row_data['N'])
            ])

        cols = ["Dataset", "Strict Forward", "Strict Backward", "Strict Overall",
                "Relaxed Forward", "Relaxed Backward", "Relaxed Overall", "N"]


        table = title_ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.0)


        header_color = '#404040'
        row_colors = {
            'Overall': '#F0F0F0',
            'CRUXEval': '#E6F3FF',
            'HumanEval': '#FFF5E6',
            'PythonSaga': '#E6FFE6'
        }


        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(header_color)
                cell.set_edgecolor('#2E2E2E')
                cell.set_fontsize(13)
            else:
                ds_name = rows[row-1][0]
                color = row_colors.get(ds_name, 'white')
                cell.set_facecolor(color)
                cell.set_edgecolor('#D3D3D3')

                if col in [1, 2, 3, 4, 5, 6] and row > 0:
                    try:
                        score = float(rows[row-1][col])
                        if score > 0.8:
                            cell.set_text_props(weight='bold')
                        if score < 0.4:
                            cell.set_text_props(style='italic', color='#666666')
                    except:
                        pass

        title_ax.set_title(title_text, fontweight="bold", fontsize=16, pad=20)


    prepare_table_data(df_p5, ax1, "Pass@5 Results (Best of 5 attempts)")


    prepare_table_data(df_p1, ax2, "Pass@1 Results (First attempt only)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Grok CRISPE forward reasoning with original "
            "backward reasoning."
        )
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
        "--crispe-outputs-dir",
        type=Path,
        default=Path("API_Model_Outputs_CRISPE_grok_no_focc"),
        help="Directory containing the Grok CRISPE outputs."
    )

    parser.add_argument(
        "--original-outputs-dir",
        type=Path,
        default=Path(
            "API_Model_Outputs/grok-4-fast-reasoning"
        ),
        help="Directory containing the original Grok backward-reasoning outputs."
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("tables_grok_no_focc"),
        help="Directory where evaluation tables and reports will be saved."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("SENSITIVITY ANALYSIS: CRISPE Forward + Original Backward")
    print(f"MODEL: {MODEL}")
    print("=" * 70)


    tables_dir = args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)


    print("\nStep 1: Copying backward reasoning results...")
    copied = copy_backward_results(args.original_outputs_dir, args.crispe_outputs_dir)
    if copied == 0:
        print("No backward results copied. Check if source directory exists.")
        print(f"   Source: {args.original_outputs_dir}")
        print(f"   Destination: {args.crispe_outputs_dir / MODEL}")


    print("\n Step 2: Loading program data...")
    prog_lookup = load_program_data(args.coverage_json)
    if not prog_lookup:
        print("No program data loaded. Exiting.")
        return


    model_dir = args.crispe_outputs_dir / MODEL
    if not model_dir.exists():
        print(f"CRISPE directory not found: {model_dir}")
        print(f"   Expected path: {model_dir}")
        print(f"   Please ensure CRISPE experiment was run first.")
        return


    datasets_present = get_datasets_in_crispe(model_dir)
    print(f"Datasets found in CRISPE directory: {datasets_present}")


    all_rows = []
    dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

    print(f"\nEvaluating {len(dirs)} programs...")

    for prog_dir in tqdm(dirs, desc="Processing programs"):
        dir_name = prog_dir.name


        if dir_name.startswith("CRUXEval_"):
            dataset = "CRUXEval"
        elif dir_name.startswith("HumanEval_"):
            dataset = "HumanEval"
        elif dir_name.startswith("PythonSaga_"):
            dataset = "PythonSaga"
        else:

            continue


        program_data = prog_lookup.get(dir_name)
        if not program_data:

            if dataset == "PythonSaga":
                try:
                    num = int(dir_name.split("_")[-1])
                    alt_id = f"PythonSaga/{num}"
                    program_data = prog_lookup.get(normalize_program_id(alt_id, dataset))
                except:
                    program_data = None

            if not program_data:

                continue


        runnable_src = program_data.get("runnable_script", "")
        if not runnable_src:
            continue


        coverage_meta = program_data.get("coverage_metadata", {})
        priority_line = coverage_meta.get("advanced_priority_line", 0)
        if not priority_line:
            priority_line = coverage_meta.get("priority_line", 0)

        if not priority_line or priority_line == 0:
            continue


        true_covered = set(coverage_meta.get("covered_lines", []))


        results = []
        for i in range(1, 6):
            out_dir = prog_dir / f"output_{i}"


            pred_covered = parse_crispe_coverage(out_dir)
            if pred_covered is None:
                fwd_jacc = 0.0
            else:
                fwd_jacc = calculate_jaccard(true_covered, pred_covered)


            bwd_succ = 0.0
            pred_input = extract_answer_text(out_dir / "ask_predict_input_response.txt")

            if pred_input:
                bwd_succ = evaluate_backward_reasoning(runnable_src, pred_input, priority_line)

            results.append({
                "sf": 1.0 if fwd_jacc > 0.99 else 0.0,
                "sb": bwd_succ,
                "rf": fwd_jacc,
                "rb": bwd_succ
            })


        p1 = results[0]


        p5_sf = 1.0 if any(r["sf"] == 1 for r in results) else 0.0
        p5_sb = 1.0 if any(r["sb"] == 1 for r in results) else 0.0
        p5_rf = max(r["rf"] for r in results)
        p5_rb = max(r["rb"] for r in results)
        p5_so = 1.0 if any(r["sf"] == 1 and r["sb"] == 1 for r in results) else 0.0
        p5_ro = p5_rf * p5_rb


        all_rows.append({
            "Dataset": dataset,
            "Program_ID": dir_name,

            "P1_SF": p1["sf"], "P1_SB": p1["sb"], "P1_SO": p1["sf"] * p1["sb"],
            "P1_RF": p1["rf"], "P1_RB": p1["rb"], "P1_RO": p1["rf"] * p1["rb"],

            "P5_SF": p5_sf, "P5_SB": p5_sb, "P5_SO": p5_so,
            "P5_RF": p5_rf, "P5_RB": p5_rb, "P5_RO": p5_ro
        })


    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No results to analyze. Exiting.")
        return

    print(f"\nResults summary:")
    print(f"   Total programs evaluated: {len(df)}")
    print(f"   Dataset distribution:")
    for dataset in ["CRUXEval", "HumanEval", "PythonSaga"]:
        count = len(df[df["Dataset"] == dataset])
        if count > 0:
            print(f"     - {dataset}: {count} programs")


    summary = df.groupby("Dataset").mean(numeric_only=True)
    counts = df["Dataset"].value_counts()


    overall = df.mean(numeric_only=True)
    overall.name = "Overall"
    summary = pd.concat([summary, overall.to_frame().T])
    counts["Overall"] = len(df)


    for dataset in ["CRUXEval", "HumanEval", "PythonSaga"]:
        if dataset not in summary.index:

            empty_row = pd.Series({col: 0.0 for col in summary.columns}, name=dataset)
            summary = pd.concat([summary, empty_row.to_frame().T])



    p1_summary = summary[["P1_SF", "P1_SB", "P1_SO", "P1_RF", "P1_RB", "P1_RO"]].copy()
    p1_summary.columns = ["Strict Fwd", "Strict Bwd", "Strict All", "Relax Fwd", "Relax Bwd", "Relax All"]
    p1_summary["N"] = counts


    p5_summary = summary[["P5_SF", "P5_SB", "P5_SO", "P5_RF", "P5_RB", "P5_RO"]].copy()
    p5_summary.columns = ["Strict Fwd", "Strict Bwd", "Strict All", "Relax Fwd", "Relax Bwd", "Relax All"]
    p5_summary["N"] = counts


    row_order = ["Overall", "CRUXEval", "HumanEval", "PythonSaga"]
    p1_summary = p1_summary.reindex(row_order)
    p5_summary = p5_summary.reindex(row_order)


    output_file = tables_dir / f"sensitivity_analysis_{MODEL}.png"
    create_sensitivity_analysis_table(p1_summary, p5_summary, output_file, MODEL)


    csv_file = tables_dir / f"sensitivity_analysis_{MODEL}_details.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to {csv_file}")


    p1_csv = tables_dir / f"{MODEL}_pass1_summary.csv"
    p5_csv = tables_dir / f"{MODEL}_pass5_summary.csv"
    p1_summary.to_csv(p1_csv)
    p5_summary.to_csv(p5_csv)


    print(f"\n{MODEL} RESULTS SUMMARY:")
    print("=" * 85)
    print("Pass@1 (First attempt):")
    print(p1_summary.round(3).to_string())
    print("\nPass@5 (Best of 5 attempts):")
    print(p5_summary.round(3).to_string())


    print("\n" + "=" * 85)
    print("KEY INSIGHTS:")


    fwd_avg = p5_summary.loc["Overall", "Strict Fwd"]
    bwd_avg = p5_summary.loc["Overall", "Strict Bwd"]
    duality_gap = bwd_avg - fwd_avg

    print(f"   Overall Strict Forward:  {fwd_avg:.3f}")
    print(f"   Overall Strict Backward: {bwd_avg:.3f}")
    print(f"   Duality Gap (Bwd - Fwd): {duality_gap:.3f}")

    if duality_gap > 0.1:
        print(f"   Significant duality gap: Models better at control than simulation")
    elif duality_gap < -0.1:
        print(f"   Inverse duality gap: Models better at simulation than control")
    else:
        print(f"   Balanced performance: Similar simulation and control capabilities")


    print(f"\n   Dataset Performance (Strict Forward, Pass@5):")
    for dataset in ["CRUXEval", "HumanEval", "PythonSaga"]:
        if dataset in p5_summary.index:
            score = p5_summary.loc[dataset, "Strict Fwd"]
            count = p5_summary.loc[dataset, "N"]
            print(f"     - {dataset}: {score:.3f} (N={count})")


    if "PythonSaga" in p5_summary.index:
        python_saga_score = p5_summary.loc["PythonSaga", "Strict Fwd"]
        if python_saga_score < 0.5:
            print(f"   PythonSaga shows significantly lower forward reasoning ({python_saga_score:.3f})")
            print(f"     Possible reasons: More complex control flow or dataset characteristics")

    print("\n" + "=" * 70)
    print(f"SENSITIVITY ANALYSIS FOR {MODEL} COMPLETED SUCCESSFULLY!")
    print("=" * 70)


    print(f"\n Output files:")
    print(f"   Tables directory: {tables_dir}")
    print(f"   Visualization: {output_file}")
    print(f"   Detailed results: {csv_file}")
    print(f"   Pass@1 summary: {p1_csv}")
    print(f"   Pass@5 summary: {p5_csv}")

if __name__ == "__main__":
    main()
