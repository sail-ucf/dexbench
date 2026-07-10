"""
coverage_collector.py - Updated for multiple datasets

Handles:
- HumanEval: artifacts/programs/runner_programs/HumanEval_X/solution.py
- PythonSaga: artifacts/programs/runner_programs/PythonSaga_X/solution.py
- CRUXEval: data/CRUXEval/formatted_cruxeval_programs/sample_XXX.py
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SLIPCOVER_TIMEOUT_SECONDS = 60

def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj

def natural_sort_key(item: Dict[str, Any]) -> List[Any]:
    tid = item.get("task_id", "")
    parts = re.split(r"([0-9]+)", tid)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

def path_to_task_id(script_path: Path, dataset: str) -> str:
    """
    Convert script path to dataset/task-id form.
    """
    if dataset == "HumanEval":

        dir_name = script_path.parent.name
        return dir_name.replace("HumanEval_", "HumanEval/")
    elif dataset == "PythonSaga":

        dir_name = script_path.parent.name
        return dir_name.replace("PythonSaga_", "PythonSaga/")
    elif dataset == "CRUXEval":

        stem = script_path.stem
        number = stem.replace("sample_", "")
        return f"CRUXEval/{number}"
    else:
        return str(script_path)

def get_dataset_from_path(script_path: Path) -> str:
    """
    Determine which dataset a script belongs to based on its path.
    """
    path_str = str(script_path)
    if "HumanEval_" in path_str and script_path.parent.name.startswith("HumanEval_"):
        return "HumanEval"
    elif "PythonSaga_" in path_str and script_path.parent.name.startswith("PythonSaga_"):
        return "PythonSaga"
    elif "formatted_cruxeval_programs" in path_str:
        return "CRUXEval"
    else:
        return "Unknown"



def find_all_branches(tree: ast.AST) -> List[ast.AST]:
    branches: List[ast.AST] = []

    class Finder(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            branches.append(node)
            self.generic_visit(node)
        def visit_For(self, node: ast.For) -> None:
            branches.append(node)
            self.generic_visit(node)
        def visit_While(self, node: ast.While) -> None:
            branches.append(node)
            self.generic_visit(node)
        def visit_Try(self, node: ast.Try) -> None:
            branches.append(node)
            self.generic_visit(node)
        def visit_With(self, node: ast.With) -> None:
            branches.append(node)
            self.generic_visit(node)
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.generic_visit(node)

    Finder().visit(tree)
    return branches

def lines_in_branch_body(branch_node: ast.AST) -> Set[int]:
    lines: Set[int] = set()
    def collect(node: ast.AST) -> None:
        if hasattr(node, "lineno"):
            try:
                lines.add(int(node.lineno))
            except Exception:
                pass
        for child in ast.iter_child_nodes(node):
            collect(child)

    if hasattr(branch_node, "body"):
        for stmt in getattr(branch_node, "body", []):
            collect(stmt)

    if isinstance(branch_node, ast.If) and getattr(branch_node, "orelse", None):
        for stmt in branch_node.orelse:
            collect(stmt)

    if isinstance(branch_node, ast.Try):
        for handler in getattr(branch_node, "handlers", []):
            for stmt in getattr(handler, "body", []):
                collect(stmt)
        for stmt in getattr(branch_node, "orelse", []):
            collect(stmt)
        for stmt in getattr(branch_node, "finalbody", []):
            collect(stmt)

    return lines

def get_branch_header_line(branch_node: ast.AST) -> Optional[int]:
    if hasattr(branch_node, "lineno"):
        try:
            return int(getattr(branch_node, "lineno"))
        except Exception:
            pass
    if hasattr(branch_node, "body") and branch_node.body:
        first = branch_node.body[0]
        if hasattr(first, "lineno"):
            try:
                return int(getattr(first, "lineno"))
            except Exception:
                pass
    return None



def find_priority_line_advanced(code_string: str, uncovered_lines: Set[int]) -> Optional[int]:
    """
    Original algorithm: selects the largest fully uncovered branch.
    """
    if not uncovered_lines:
        return None

    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return min(uncovered_lines)

    all_branches = find_all_branches(tree)

    candidate_branches: List[Tuple[ast.AST, Set[int]]] = []
    for b in all_branches:
        body_lines = lines_in_branch_body(b)
        if not body_lines:
            continue
        if body_lines.issubset(uncovered_lines):
            candidate_branches.append((b, body_lines))

    if candidate_branches:
        candidate_branches.sort(key=lambda x: len(x[1]), reverse=True)
        top_branch = candidate_branches[0][0]
        header = get_branch_header_line(top_branch)
        if header is not None:
            return header

    all_branch_lines: Set[int] = set()
    for b in all_branches:
        all_branch_lines |= lines_in_branch_body(b)

    standalone = sorted(list(uncovered_lines - all_branch_lines))
    if standalone:
        return standalone[0]

    return min(uncovered_lines)

def random_alternative_path_selection_algorithm(code_string: str, uncovered_lines: Set[int]) -> Optional[int]:
    """
    Alternative algorithm: randomly selects from fully uncovered branches
    (excluding the largest one that the original algorithm would pick).
    Returns None if no alternative branches exist (so we can filter these out).
    """
    if not uncovered_lines:
        return None

    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return min(uncovered_lines)

    all_branches = find_all_branches(tree)

    candidate_branches: List[Tuple[ast.AST, Set[int]]] = []
    for b in all_branches:
        body_lines = lines_in_branch_body(b)
        if not body_lines:
            continue
        if body_lines.issubset(uncovered_lines):
            candidate_branches.append((b, body_lines))


    if len(candidate_branches) > 1:

        candidate_branches.sort(key=lambda x: len(x[1]), reverse=True)
        alternative_branches = candidate_branches[1:]

        if alternative_branches:

            random_branch = random.choice(alternative_branches)
            header = get_branch_header_line(random_branch[0])
            if header is not None:
                return header


    return None

def find_priority_line_least_coverage(code_string: str, uncovered_lines: Set[int]) -> Optional[int]:
    """
    New algorithm: selects the smallest fully uncovered branch.
    Only activates when there are multiple candidate branches (otherwise same as original).
    Returns None if only one candidate branch exists.
    """
    if not uncovered_lines:
        return None

    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return min(uncovered_lines)

    all_branches = find_all_branches(tree)

    candidate_branches: List[Tuple[ast.AST, Set[int]]] = []
    for b in all_branches:
        body_lines = lines_in_branch_body(b)
        if not body_lines:
            continue
        if body_lines.issubset(uncovered_lines):
            candidate_branches.append((b, body_lines))


    if len(candidate_branches) > 1:

        candidate_branches.sort(key=lambda x: len(x[1]))
        smallest_branch = candidate_branches[0][0]
        header = get_branch_header_line(smallest_branch)
        if header is not None:
            return header


    return None



def parse_slipcover_missing_lines(missing_str: str) -> Set[int]:
    missing: Set[int] = set()
    if not missing_str:
        return missing

    tokens = re.split(r'[,\s]+', missing_str.strip())
    for token in tokens:
        if not token:
            continue
        if re.fullmatch(r'\d+', token):
            try:
                missing.add(int(token))
            except ValueError:
                continue
        elif re.fullmatch(r'\d+-\d+', token):
            try:
                a_str, b_str = token.split('-', 1)
                a, b = int(a_str), int(b_str)
                if b >= a:
                    missing.update(range(a, b + 1))
            except Exception:
                continue
    return missing

def parse_slipcover_text_output(text_output: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {'files': []}
    if not text_output or not text_output.strip():
        return result

    lines = [ln for ln in text_output.splitlines() if ln.strip()]
    for line in lines:
        text = line.strip()
        if text.startswith("File") or set(text) <= set("- "):
            continue
        parts = re.split(r'\s+', text)
        if not parts:
            continue

        idx_total = None
        for i in range(1, len(parts)):
            if re.fullmatch(r'\d+', parts[i]):
                idx_total = i
                break
        if idx_total is None:
            continue

        if idx_total + 2 >= len(parts):
            continue

        filename = ' '.join(parts[:idx_total])
        try:
            total_lines = int(parts[idx_total])
            missing_count = int(parts[idx_total + 1])
            percent_raw = parts[idx_total + 2].rstrip('%')
            percent = float(percent_raw)
            missing_token = ''
            if idx_total + 3 < len(parts):
                missing_token = ''.join(parts[idx_total + 3:])
            missing_token = missing_token.strip()
        except Exception:
            continue

        missing_set = parse_slipcover_missing_lines(missing_token)
        file_entry = {
            'filename': filename,
            'summary': {
                'lines_total': total_lines,
                'lines_missing': missing_count,
                'percent_covered': percent
            },
            'missing_lines_raw': missing_token,
            'missing_lines_set': sorted(list(missing_set))
        }
        result['files'].append(file_entry)
    return result

def parse_slipcover_output(raw_output: str) -> Dict[str, Any]:
    if not raw_output:
        return {}

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict) and 'files' in parsed:
            files = []
            for f in parsed.get('files', []):
                fn = f.get('filename', '')
                summary = f.get('summary', {})
                missing_raw = f.get('missing_lines') or f.get('missing') or ''
                percent = summary.get('percent_covered') or summary.get('percent') or summary.get('coverage', 0.0)
                total_lines = summary.get('lines_total') or summary.get('lines') or 0
                missing_count = summary.get('lines_missing') or summary.get('missing') or 0
                missing_set = parse_slipcover_missing_lines(str(missing_raw))
                files.append({
                    'filename': fn,
                    'summary': {
                        'lines_total': total_lines,
                        'lines_missing': missing_count,
                        'percent_covered': float(percent)
                    },
                    'missing_lines_raw': str(missing_raw),
                    'missing_lines_set': sorted(list(missing_set))
                })
            return {'files': files}
    except json.JSONDecodeError:
        pass
    except Exception:
        logging.debug("SlipCover JSON parsing failed:\n" + traceback.format_exc())

    return parse_slipcover_text_output(raw_output)



def get_all_executable_lines(code_string: str) -> Set[int]:
    exec_lines: Set[int] = set()
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return exec_lines
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        try:
            exec_lines.add(int(node.lineno))
        except Exception:
            continue
    return exec_lines



def find_all_scripts(runner_programs_dir: Path, cruxeval_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find all scripts across all datasets.
    Returns list of (script_path, dataset_name) tuples.
    """
    scripts = []

    if runner_programs_dir.exists():
        for dataset_dir in runner_programs_dir.iterdir():
            if dataset_dir.is_dir():
                if dataset_dir.name.startswith("HumanEval_"):
                    solution_path = dataset_dir / "solution.py"
                    if solution_path.exists():
                        scripts.append((solution_path, "HumanEval"))
                elif dataset_dir.name.startswith("PythonSaga_"):
                    solution_path = dataset_dir / "solution.py"
                    if solution_path.exists():
                        scripts.append((solution_path, "PythonSaga"))


    if cruxeval_dir.exists():
        for script_path in cruxeval_dir.glob("sample_*.py"):
            scripts.append((script_path, "CRUXEval"))

    return scripts

def create_runner_info(script_path: Path, dataset: str, script_content: str) -> Dict[str, Any]:
    """
    Create a runner info entry for scripts that don't have one in artifacts/programs/runner_programs.json
    (mainly for CRUXEval).
    """
    task_id = path_to_task_id(script_path, dataset)


    if dataset == "CRUXEval":
        return {
            "dataset": dataset,
            "task_id": task_id,
            "solution_code": script_content,
            "runnable_script": script_content,
            "complexity": 1
        }
    else:

        return {
            "dataset": dataset,
            "task_id": task_id,
            "solution_code": script_content,
            "runnable_script": script_content,
            "complexity": 1
        }



def extract_coverage_from_output(script_path: Path, script_content: str, slipcover_output: str) -> Dict[str, Any]:
    coverage_metadata: Dict[str, Any] = {
        "covered_lines": [],
        "uncovered_lines": [],
        "coverage_percentage": 0.0,
        "advanced_priority_line": None,
        "rap_priority_line": None,
        "least_coverage_priority_line": None
    }

    parsed = parse_slipcover_output(slipcover_output)
    if not parsed or 'files' not in parsed or not parsed['files']:
        return coverage_metadata

    chosen = None
    for file_entry in parsed['files']:
        fname = file_entry.get('filename', '')
        if not fname:
            continue
        if fname == str(script_path) or fname.endswith(script_path.name) or script_path.name in fname:
            chosen = file_entry
            break
    if chosen is None:
        chosen = parsed['files'][0]

    summary = chosen.get('summary', {})
    total_lines = int(summary.get('lines_total') or 0)
    percent_covered = float(summary.get('percent_covered') or 0.0)

    missing_set: Set[int] = set(chosen.get('missing_lines_set', []))
    if not missing_set and chosen.get('missing_lines_raw'):
        missing_set = parse_slipcover_missing_lines(chosen.get('missing_lines_raw', ''))

    exec_lines = get_all_executable_lines(script_content)

    if exec_lines:
        covered = sorted(list(exec_lines - missing_set))
        uncovered = sorted(list(missing_set & exec_lines))
    else:
        covered = []
        uncovered = sorted(list(missing_set))

    coverage_metadata['covered_lines'] = covered
    coverage_metadata['uncovered_lines'] = uncovered
    coverage_metadata['coverage_percentage'] = round(percent_covered, 2)

    try:
        if uncovered:

            adv = find_priority_line_advanced(script_content, set(uncovered))
            if adv is None:
                adv = min(uncovered)
            coverage_metadata['advanced_priority_line'] = adv


            rap = random_alternative_path_selection_algorithm(script_content, set(uncovered))
            coverage_metadata['rap_priority_line'] = rap


            lc = find_priority_line_least_coverage(script_content, set(uncovered))
            coverage_metadata['least_coverage_priority_line'] = lc
        else:
            coverage_metadata['advanced_priority_line'] = None
            coverage_metadata['rap_priority_line'] = None
            coverage_metadata['least_coverage_priority_line'] = None
    except Exception as e:
        logging.debug("Failed computing priority lines:\n" + traceback.format_exc())
        coverage_metadata['advanced_priority_line'] = None
        coverage_metadata['rap_priority_line'] = None
        coverage_metadata['least_coverage_priority_line'] = None

    return coverage_metadata



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SlipCover on runner programs and collect coverage metadata."
    )

    parser.add_argument(
        "--runner-json",
        type=Path,
        default=Path("artifacts/programs/runner_programs.json"),
        help="Path to runner_programs.json."
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/programs/runner_programs_with_coverage.json"),
        help="Path to output JSON with coverage metadata."
    )

    parser.add_argument(
        "--rap-output-json",
        type=Path,
        default=Path("artifacts/programs/runner_programs_with_coverage_rap.json"),
        help="Path to output JSON with RAP coverage metadata."
    )

    parser.add_argument(
        "--least-coverage-output-json",
        type=Path,
        default=Path("artifacts/programs/runner_programs_with_least_coverage.json"),
        help="Path to output JSON with Least Coverage metadata."
    )

    parser.add_argument(
        "--runner-programs-dir",
        type=Path,
        default=Path("artifacts/programs/runner_programs"),
        help="Directory containing HumanEval and PythonSaga runner programs."
    )

    parser.add_argument(
        "--cruxeval-dir",
        type=Path,
        default=Path("data/CRUXEval/formatted_cruxeval_programs"),
        help="Directory containing formatted CRUXEval programs."
    )

    parser.add_argument(
        "--venv-path",
        type=Path,
        default=None,
        help="Optional virtualenv path containing python and slipcover."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=SLIPCOVER_TIMEOUT_SECONDS,
        help="Timeout seconds per script run."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    runner_json_path = args.runner_json
    output_json_path = args.output_json
    rap_output_json_path = args.rap_output_json
    least_coverage_output_json_path = args.least_coverage_output_json
    venv_path = args.venv_path


    runner_map: Dict[str, Dict[str, Any]] = {}
    if runner_json_path.exists():
        try:
            with open(runner_json_path, "r", encoding="utf-8") as f:
                runner_data = json.load(f)
                runner_map = {item.get("task_id"): item for item in runner_data if isinstance(item, dict) and item.get("task_id")}
            logging.info("Loaded %d entries from runner JSON", len(runner_map))
        except Exception as e:
            logging.error("Failed to read runner JSON: %s", e)
    else:
        logging.warning("Runner JSON not found: %s. Will create entries for all scripts.", runner_json_path)


    all_scripts = find_all_scripts(args.runner_programs_dir, args.cruxeval_dir)
    if not all_scripts:
        logging.error("No scripts found in any dataset directory!")
        return

    logging.info("Found %d scripts across all datasets", len(all_scripts))

    updated_results_original: List[Dict[str, Any]] = []
    updated_results_rap: List[Dict[str, Any]] = []
    updated_results_least_coverage: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0
    full_coverage_count = 0
    rap_eligible_count = 0
    lc_eligible_count = 0

    for idx, (script_path, dataset) in enumerate(all_scripts, start=1):
        task_id = path_to_task_id(script_path, dataset)
        logging.info("[%d/%d] Processing %s -> %s", idx, len(all_scripts), script_path.name, task_id)


        try:
            script_content = script_path.read_text(encoding="utf-8")
        except Exception as e:
            logging.error(" -> Could not read script file %s: %s", script_path, e)
            error_count += 1
            continue


        runner_info = runner_map.get(task_id)
        if runner_info is None:
            logging.info(" -> Creating new entry for %s", task_id)
            runner_info = create_runner_info(script_path, dataset, script_content)


        cmd = ["python3", "-m", "slipcover", str(script_path)]
        if venv_path:
            venv_python = Path(venv_path) / "bin" / "python"
            cmd = [str(venv_python), "-m", "slipcover", str(script_path)]

        logging.debug(" -> Running: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            raw_output = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                status = "success"
            else:
                status = "test_failure"
                logging.warning(" -> slipcover returned non-zero exit code %d", proc.returncode)
        except subprocess.TimeoutExpired as te:
            raw_output = (te.stdout or "") + (te.stderr or "") if hasattr(te, "stdout") else ""
            status = "timeout"
            logging.error(" -> slipcover timed out after %ds", args.timeout)
        except Exception as e:
            raw_output = ""
            status = "error"
            logging.error(" -> Unexpected error: %s", str(e))


        try:
            extracted = extract_coverage_from_output(script_path, script_content, raw_output)
        except Exception as e:
            logging.error(" -> Failed to extract coverage: %s", e)
            extracted = {
                "covered_lines": [],
                "uncovered_lines": [],
                "coverage_percentage": 0.0,
                "advanced_priority_line": None,
                "rap_priority_line": None,
                "least_coverage_priority_line": None,
                "error_message": f"extraction_error: {e}"
            }


        coverage_metadata = {
            "status": status,
            **extracted
        }
        if status != "success":
            stderr_text = (proc.stderr or "") if 'proc' in locals() else ""
            coverage_metadata.setdefault("error_message", "")
            coverage_metadata["error_message"] = (coverage_metadata["error_message"] or "") + ("\n" + stderr_text if stderr_text else "")

        coverage_metadata = make_json_safe(coverage_metadata)


        if status == "success":
            if coverage_metadata["coverage_percentage"] < 100.0 and coverage_metadata["uncovered_lines"]:

                entry_original = runner_info.copy()
                entry_original["coverage_metadata"] = coverage_metadata.copy()

                for field in ["rap_priority_line", "least_coverage_priority_line"]:
                    if field in entry_original["coverage_metadata"]:
                        del entry_original["coverage_metadata"][field]
                updated_results_original.append(entry_original)


                if coverage_metadata.get("rap_priority_line") is not None:
                    entry_rap = runner_info.copy()
                    entry_rap["coverage_metadata"] = coverage_metadata.copy()

                    for field in ["advanced_priority_line", "least_coverage_priority_line"]:
                        if field in entry_rap["coverage_metadata"]:
                            del entry_rap["coverage_metadata"][field]

                    if "rap_priority_line" in entry_rap["coverage_metadata"]:
                        entry_rap["coverage_metadata"]["priority_line"] = entry_rap["coverage_metadata"].pop("rap_priority_line")

                    updated_results_rap.append(entry_rap)
                    rap_eligible_count += 1


                if coverage_metadata.get("least_coverage_priority_line") is not None:
                    entry_lc = runner_info.copy()
                    entry_lc["coverage_metadata"] = coverage_metadata.copy()

                    for field in ["advanced_priority_line", "rap_priority_line"]:
                        if field in entry_lc["coverage_metadata"]:
                            del entry_lc["coverage_metadata"][field]

                    if "least_coverage_priority_line" in entry_lc["coverage_metadata"]:
                        entry_lc["coverage_metadata"]["priority_line"] = entry_lc["coverage_metadata"].pop("least_coverage_priority_line")

                    updated_results_least_coverage.append(entry_lc)
                    lc_eligible_count += 1

                success_count += 1


                eligibility_msg = "Added to "
                parts = []
                parts.append("original")
                if coverage_metadata.get("rap_priority_line") is not None:
                    parts.append("RAP")
                if coverage_metadata.get("least_coverage_priority_line") is not None:
                    parts.append("LeastCoverage")
                logging.info(" -> SUCCESS: Added to %s (coverage: %.1f%%)", ", ".join(parts), coverage_metadata["coverage_percentage"])
            else:
                full_coverage_count += 1
                logging.info(" -> SKIPPED: Full coverage (100%%) or no uncovered lines")
        else:
            error_count += 1
            logging.info(" -> SKIPPED: Error status (%s)", status)


    updated_results_original.sort(key=natural_sort_key)
    updated_results_rap.sort(key=natural_sort_key)
    updated_results_least_coverage.sort(key=natural_sort_key)

    try:

        with open(output_json_path, "w", encoding="utf-8") as out_f:
            json.dump(updated_results_original, out_f, indent=4)


        with open(rap_output_json_path, "w", encoding="utf-8") as out_f:
            json.dump(updated_results_rap, out_f, indent=4)


        with open(least_coverage_output_json_path, "w", encoding="utf-8") as out_f:
            json.dump(updated_results_least_coverage, out_f, indent=4)


        dataset_counts_original = {}
        dataset_counts_rap = {}
        dataset_counts_lc = {}

        for result in updated_results_original:
            dataset = result.get("dataset", "Unknown")
            dataset_counts_original[dataset] = dataset_counts_original.get(dataset, 0) + 1

        for result in updated_results_rap:
            dataset = result.get("dataset", "Unknown")
            dataset_counts_rap[dataset] = dataset_counts_rap.get(dataset, 0) + 1

        for result in updated_results_least_coverage:
            dataset = result.get("dataset", "Unknown")
            dataset_counts_lc[dataset] = dataset_counts_lc.get(dataset, 0) + 1

        logging.info("=" * 60)
        logging.info("COVERAGE COLLECTION SUMMARY:")
        logging.info("  Total scripts processed: %d", len(all_scripts))
        logging.info("  Successfully added to results: %d", success_count)
        logging.info("  Full coverage (skipped): %d", full_coverage_count)
        logging.info("  Errors/failures (skipped): %d", error_count)
        logging.info("  RAP eligible programs: %d", rap_eligible_count)
        logging.info("  Least Coverage eligible programs: %d", lc_eligible_count)
        logging.info("  Final results saved:")
        logging.info("    - Original algorithm: %d programs to %s", len(updated_results_original), output_json_path)
        logging.info("    - RAP algorithm: %d programs to %s", len(updated_results_rap), rap_output_json_path)
        logging.info("    - Least Coverage algorithm: %d programs to %s", len(updated_results_least_coverage), least_coverage_output_json_path)
        logging.info("  Breakdown by dataset (Original/RAP/LC):")
        all_datasets = set(dataset_counts_original.keys()) | set(dataset_counts_rap.keys()) | set(dataset_counts_lc.keys())
        for dataset in sorted(all_datasets):
            count_orig = dataset_counts_original.get(dataset, 0)
            count_rap = dataset_counts_rap.get(dataset, 0)
            count_lc = dataset_counts_lc.get(dataset, 0)
            logging.info("    - %s: %d / %d / %d", dataset, count_orig, count_rap, count_lc)
    except Exception as e:
        logging.error("Failed to write output JSON: %s", e)

if __name__ == "__main__":
    main()