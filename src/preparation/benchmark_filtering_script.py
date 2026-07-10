import json
import os
import re
import argparse
import random
import ast
import textwrap
import logging
from pathlib import Path

def remove_docstrings(source: str) -> str:
    """
    Remove all docstrings (triple-quoted strings) from Python source code.
    """

    source = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)

    source = re.sub(r"'''.*?'''", '', source, flags=re.DOTALL)

    source = re.sub(r'\n\s*\n', '\n\n', source)
    return source.strip()

def preserve_original_indentation(prompt: str, solution: str, entry_point: str) -> str:
    """
    Combine prompt and solution while preserving the original indentation exactly.
    Handles the case where solution might already contain the function definition.
    """
    if f"def {entry_point}" in solution:
        imports_section = ""
        import_matches = re.findall(r'^(import\s+.*|from\s+.*\s+import\s+.*)', prompt, re.MULTILINE)
        if import_matches:
            imports_section = '\n'.join(import_matches) + '\n\n'
        return imports_section + solution
    else:
        prompt_lines = prompt.split('\n')
        function_def_line = None
        for line in prompt_lines:
            if line.strip().startswith('def '):
                function_def_line = line
                break

        if function_def_line:
            indent_level = len(function_def_line) - len(function_def_line.lstrip())
            indent = ' ' * (indent_level + 4)

            solution_lines = solution.split('\n')
            indented_solution = []
            for line in solution_lines:
                if line.strip():
                    indented_solution.append(indent + line)
                else:
                    indented_solution.append('')

            return prompt + '\n'.join(indented_solution)
        else:
            return prompt + solution

def count_complexity(code_string: str) -> int:
    """
    Parses a Python code string and counts ALL control flow statements.
    """
    count = 0
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                count += 1
                if isinstance(node, ast.If):
                    count += len(node.orelse) if node.orelse and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)) else 0
            elif isinstance(node, ast.Try):
                count += len(node.handlers)
                if node.finalbody:
                    count += 1
            elif isinstance(node, ast.With):
                count += 1
            elif isinstance(node, ast.BoolOp):
                count += len(node.values) - 1
            elif isinstance(node, ast.IfExp):
                count += 1
    except SyntaxError:
        return 0
    return count

def extract_first_single_line_assertion(test_code: str) -> str:
    """
    Extract the first assertion and convert it to a single line.
    """

    start_pos = test_code.find('assert candidate(')
    if start_pos == -1:
        return None


    end_pos = test_code.find('\nassert candidate(', start_pos + 1)
    if end_pos == -1:
        end_pos = test_code.find('\n\n', start_pos + 1)
    if end_pos == -1:
        end_pos = len(test_code)


    assertion_block = test_code[start_pos:end_pos].strip()


    lines = assertion_block.split('\n')
    single_line_assertion = lines[0].strip()


    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('assert '):

            if line.startswith('[') or line.startswith('{') or line.endswith(',') or line.endswith('\\'):
                single_line_assertion += line
            else:
                single_line_assertion += ' ' + line
        else:
            break


    single_line_assertion = re.sub(r'\s+', ' ', single_line_assertion)
    single_line_assertion = single_line_assertion.replace(' [ ', ' [').replace(' ]', ']')
    single_line_assertion = single_line_assertion.replace(' , ', ', ')


    single_line_assertion = re.sub(r',\s*\"[^\"]*\"\s*$', '', single_line_assertion)

    return single_line_assertion

def process_human_eval_item(item):
    """
    Processes a single item from the HumanEval dataset.
    """
    prompt = item.get("prompt", "")
    solution = item.get("canonical_solution", "")
    entry_point = item.get("entry_point")
    test_code = item.get("test", "")
    task_id = item.get("task_id")

    if not all([prompt, solution, entry_point, test_code]):
        return None

    full_solution_code = prompt + solution
    complexity_score = count_complexity(full_solution_code)

    if complexity_score == 0:
        logging.info(f"  [SIMPLE] {task_id}: Complexity is 0. Archiving original code.")
        return {"status": "simple", "original_code": full_solution_code}


    assertion_line = extract_first_single_line_assertion(test_code)
    if not assertion_line:
        logging.warning(f"  [SKIP] {task_id}: No assertion found in test code.")
        return None

    assertion_line = assertion_line.replace("candidate", entry_point)

    expression = assertion_line[len("assert"):].strip()


    expression = re.sub(r'\bcandidate\s*\(([^)]*)\)', r'candidate("dummy_input")', expression)
    expression = re.sub(r'==\s*(str|int|float|bool|list|dict|tuple|set)\b', r'== "\1"', expression)

    if "==" in expression:
        parts = expression.split("==", 1)
        runnable_assertion = f"unittest.TestCase().assertEqual({parts[0].strip()}, {parts[1].strip()})"
    elif "!=" in expression:
        parts = expression.split("!=", 1)
        runnable_assertion = f"unittest.TestCase().assertNotEqual({parts[0].strip()}, {parts[1].strip()})"
    elif expression.startswith("not "):
        runnable_assertion = f"unittest.TestCase().assertFalse({expression[4:].strip()})"
    else:
        runnable_assertion = f"unittest.TestCase().assertTrue({expression})"


    imports = "import unittest\nimport sys\nimport io\nimport datetime\nimport math\n"

    cleaned_code = remove_docstrings(full_solution_code)
    final_script = f"{imports}\n{cleaned_code}\n\n{runnable_assertion}"

    return {"status": "complex", "runnable_script": final_script, "complexity": complexity_score}

def process_pythonsaga_item(item):
    """
    Processes a single item from the PythonSaga dataset.
    """
    prompt = item.get("prompt", "")
    solution = item.get("canonical_solution", "") or item.get("solution", "")
    entry_point = item.get("entry_point")
    test_code = item.get("test", "")
    task_id = item.get("task_id")

    if not all([prompt, solution, entry_point, test_code]):
        return None

    full_solution_code = preserve_original_indentation(prompt, solution, entry_point)
    complexity_score = count_complexity(full_solution_code)

    if complexity_score == 0:
        logging.info(f"  [SIMPLE] {task_id}: Complexity is 0. Archiving original code.")
        return {"status": "simple", "original_code": full_solution_code}

    try:
        ast.parse(full_solution_code)
        processed_code = full_solution_code
    except SyntaxError:
        logging.info(f"  [EXTRACT] {task_id}: Attempting to extract main function due to syntax error")
        if f"def {entry_point}" in solution:
            processed_code = solution
            import_matches = re.findall(r'^(import\s+.*|from\s+.*\s+import\s+.*)', prompt, re.MULTILINE)
            if import_matches:
                processed_code = '\n'.join(import_matches) + '\n\n' + processed_code
        else:
            function_start = f"def {entry_point}"
            if function_start in full_solution_code:
                parts = full_solution_code.split("def ")
                main_function_part = None

                for part in parts:
                    if part.strip().startswith(entry_point + "("):
                        main_function_part = "def " + part
                        if "def " in main_function_part:
                            main_function_part = main_function_part.split("def ")[0]
                        break

                if main_function_part:
                    imports_section = ""
                    import_matches = re.findall(r'^(import\s+.*|from\s+.*\s+import\s+.*)', full_solution_code, re.MULTILINE)
                    if import_matches:
                        imports_section = '\n'.join(import_matches) + '\n\n'
                    processed_code = imports_section + main_function_part.strip()
                else:
                    logging.warning(f"  [SKIP] {task_id}: Could not extract main function from syntax error")
                    return None
            else:
                logging.warning(f"  [SKIP] {task_id}: Could not find main function definition")
                return None


    assertion_line = extract_first_single_line_assertion(test_code)
    if not assertion_line:
        logging.warning(f"  [SKIP] {task_id}: No assertion found in test code.")
        return None

    assertion_line = assertion_line.replace("candidate", entry_point)

    expression = assertion_line[len("assert"):].strip()


    expression = re.sub(r'\bcandidate\s*\(([^)]*)\)', r'candidate("dummy_input")', expression)
    expression = re.sub(r'==\s*(str|int|float|bool|list|dict|tuple|set)\b', r'== "\1"', expression)

    if "==" in expression:
        parts = expression.split("==", 1)
        runnable_assertion = f"unittest.TestCase().assertEqual({parts[0].strip()}, {parts[1].strip()})"
    elif "!=" in expression:
        parts = expression.split("!=", 1)
        runnable_assertion = f"unittest.TestCase().assertNotEqual({parts[0].strip()}, {parts[1].strip()})"
    elif expression.startswith("not "):
        runnable_assertion = f"unittest.TestCase().assertFalse({expression[4:].strip()})"
    else:
        runnable_assertion = f"unittest.TestCase().assertTrue({expression})"


    imports = "import unittest\nimport sys\nimport io\nimport datetime\nimport math\nimport heapq\n"

    cleaned_code = remove_docstrings(processed_code)
    final_script = f"{imports}\n{cleaned_code}\n\n{runnable_assertion}"

    return {"status": "complex", "runnable_script": final_script, "complexity": complexity_score}

import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate runnable scripts from HumanEval and PythonSaga datasets."
    )

    parser.add_argument(
        "--human-eval",
        type=Path,
        required=True,
        help="Path to HumanEval.jsonl."
    )

    parser.add_argument(
        "--pythonsaga",
        type=Path,
        required=True,
        help="Path to PythonSaga JSONL file."
    )

    parser.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=Path("artifacts/programs/runner_programs.json"),
        help="Output JSON file name for complex programs."
    )

    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        default=Path("artifacts/programs"),
        help="Main output directory for all program files."
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    main_output_dir = args.output_dir
    runner_dir = main_output_dir / "runner_programs"
    simple_dir = main_output_dir / "programs_lacking_conditional_or_loop"

    runner_dir.mkdir(parents=True, exist_ok=True)
    simple_dir.mkdir(parents=True, exist_ok=True)

    all_complex_scripts = []
    processed_ids = set()
    simple_count = 0
    failed_programs = []

    def save_solution(task_id, result, test_code, full_solution_code, dataset_name):
        task_dir = runner_dir / task_id.replace("/", "_")
        task_dir.mkdir(parents=True, exist_ok=True)

        solution_path = task_dir / "solution.py"
        test_path = task_dir / "test.py"

        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(result["runnable_script"])


        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code.strip())


        try:
            subprocess.run(
                ["python3", str(solution_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=True
            )
        except Exception as e:
            failed_programs.append(task_id)
            shutil.rmtree(task_dir)
            logging.warning(f"  [FAIL] {task_id} crashed during execution. Skipping.")
            return False


        script_info = {
            "dataset": dataset_name,
            "task_id": task_id,
            "solution_code": full_solution_code,
            "runnable_script": result["runnable_script"],
            "complexity": result["complexity"]
        }
        all_complex_scripts.append(script_info)
        return True


    logging.info(f"Processing {args.human_eval}...")
    try:
        with open(args.human_eval, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                task_id = item.get("task_id")
                if not task_id or task_id in processed_ids:
                    continue

                result = process_human_eval_item(item)
                if not result:
                    continue

                processed_ids.add(task_id)
                if result["status"] == "complex":
                    full_solution_code = item.get("prompt", "") + item.get("canonical_solution", "")
                    success = save_solution(task_id, result, item.get("test", ""), full_solution_code, "HumanEval")
                    if not success:
                        continue
                else:
                    simple_count += 1
                    cleaned_simple_code = remove_docstrings(result["original_code"])
                    filename = task_id.replace("/", "_") + ".py"
                    with open(simple_dir / filename, 'w', encoding='utf-8') as sf:
                        sf.write(cleaned_simple_code)

    except Exception as e:
        logging.error(f"Error processing HumanEval file: {e}", exc_info=True)


    logging.info(f"Processing {args.pythonsaga}...")
    try:
        with open(args.pythonsaga, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                task_id = item.get("task_id")
                if not task_id or task_id in processed_ids:
                    continue

                result = process_pythonsaga_item(item)
                if not result:
                    continue

                processed_ids.add(task_id)
                if result["status"] == "complex":
                    full_solution_code = item.get("prompt", "") + (item.get("canonical_solution", "") or item.get("solution", ""))
                    success = save_solution(task_id, result, item.get("test", ""), full_solution_code, "PythonSaga")
                    if not success:
                        continue
                else:
                    simple_count += 1
                    cleaned_simple_code = remove_docstrings(result["original_code"])
                    filename = task_id.replace("/", "_") + ".py"
                    with open(simple_dir / filename, 'w', encoding='utf-8') as sf:
                        sf.write(cleaned_simple_code)

    except Exception as e:
        logging.error(f"Error processing PythonSaga file: {e}", exc_info=True)


    all_complex_scripts.sort(key=lambda x: x.get('complexity', 0), reverse=True)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_complex_scripts, f, indent=4)

    logging.info("Done.")
    logging.info(f"Valid programs saved: {len(all_complex_scripts)}")
    logging.info(f"Programs that failed execution: {len(failed_programs)}")
    if failed_programs:
        logging.info("Failed task_ids:")
        for fail_id in failed_programs:
            logging.info(f" - {fail_id}")


if __name__ == "__main__":
    main()
