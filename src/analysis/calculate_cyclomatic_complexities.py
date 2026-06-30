import json
import ast
import os
import re
from typing import Dict, List, Any
import argparse

def calculate_cyclomatic_complexity(code: str) -> int:
    """
    Calculate McCabe's cyclomatic complexity for Python code.
    Returns the complexity (minimum 1 for any function).
    """
    try:
        tree = ast.parse(code)


        has_function = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          for node in ast.walk(tree))

        if not has_function:
            return 0


        complexity = 1

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1

            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_Try(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_BoolOp(self, node):

                self.complexity += len(node.values) - 1
                self.generic_visit(node)

            def visit_IfExp(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_Match(self, node):

                self.complexity += max(1, len(node.cases))
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        return visitor.complexity

    except SyntaxError:
        return 0
    except Exception:
        return 0

def read_cruxeval_program(filepath: str) -> str:
    """Read a CRUXEval program from file"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception:
        return ""

def read_humaneval_program(directory: str) -> str:
    """Read a HumanEval program from solution.py"""
    filepath = os.path.join(directory, "solution.py")
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception:
        return ""

def read_pythonsaga_program(directory: str) -> str:
    """Read a PythonSaga program from solution.py"""
    filepath = os.path.join(directory, "solution.py")
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception:
        return ""

def update_json_complexity(json_file: str, cruxeval_dir: str, runner_programs_dir: str,  backup: bool = True):
    """Update cyclomatic complexity values in the program metadata JSON."""

    print("=" * 60)
    print("UPDATING CYCLOMATIC COMPLEXITY VALUES")
    print("=" * 60)


    if backup:
        backup_file = json_file + ".backup"
        import shutil
        shutil.copy2(json_file, backup_file)
        print(f"Created backup: {backup_file}")


    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    print(f"Loaded {len(data)} programs from JSON")


    updated_count = 0
    failed_count = 0
    complexity_stats = {"CRUXEval": [], "HumanEval": [], "PythonSaga": []}


    for idx, item in enumerate(data):
        dataset = item.get('dataset', '')
        task_id = item.get('task_id', '')
        original_complexity = item.get('complexity', 0)

        if not dataset or not task_id:
            continue

        try:

            program_code = ""

            if dataset == "CRUXEval":

                match = re.search(r'CRUXEval/(\d+)', task_id)
                if match:
                    program_num = match.group(1)
                    filepath = os.path.join(cruxeval_dir,f"sample_{program_num}.py")
                    program_code = read_cruxeval_program(filepath)

            elif dataset == "HumanEval":

                match = re.search(r'HumanEval/(\d+)', task_id)
                if match:
                    program_num = match.group(1)
                    directory = os.path.join(runner_programs_dir,f"HumanEval_{program_num}")
                    program_code = read_humaneval_program(directory)

            elif dataset == "PythonSaga":

                match = re.search(r'PythonSaga/(\d+)', task_id)
                if match:
                    program_num = match.group(1)
                    directory = os.path.join(runner_programs_dir,f"PythonSaga_{program_num}")
                    program_code = read_pythonsaga_program(directory)


            if program_code:
                new_complexity = calculate_cyclomatic_complexity(program_code)


                if new_complexity != original_complexity:
                    item['complexity'] = new_complexity
                    updated_count += 1


                    if updated_count <= 5:
                        print(f"  Updated {task_id}: {original_complexity} -> {new_complexity}")


                complexity_stats[dataset].append(new_complexity)
            else:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  Could not find code for {task_id}")

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"  Error processing {task_id}: {e}")


        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(data)} programs...")


    try:
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"\nOK Successfully saved updated JSON to {json_file}")
        print(f"  Updated {updated_count} programs")
        print(f"  Failed to process {failed_count} programs")


        print(f"\n{'='*60}")
        print("COMPLEXITY STATISTICS BY DATASET")
        print("=" * 60)

        for dataset in ["CRUXEval", "HumanEval", "PythonSaga"]:
            values = complexity_stats[dataset]
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"\n{dataset}:")
                print(f"  Programs: {len(values)}")
                print(f"  Average CC: {avg:.2f}")
                print(f"  Min CC: {min_val}")
                print(f"  Max CC: {max_val}")
                print(f"  Range: {max_val - min_val}")


        print(f"\n{'='*60}")
        print("SAMPLE UPDATED VALUES")
        print("=" * 60)

        sample_count = 0
        for item in data[:20]:
            dataset = item.get('dataset', '')
            task_id = item.get('task_id', '')
            complexity = item.get('complexity', 0)

            if dataset == "CRUXEval" and complexity > 1:
                print(f"  {task_id}: CC = {complexity}")
                sample_count += 1
                if sample_count >= 5:
                    break

    except Exception as e:
        print(f"Error saving JSON file: {e}")

def verify_updates(json_file: str):
    """Verify that complexity values were updated"""

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print("=" * 60)

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)


        cc_greater_than_1 = {"CRUXEval": 0, "HumanEval": 0, "PythonSaga": 0}
        total_by_dataset = {"CRUXEval": 0, "HumanEval": 0, "PythonSaga": 0}

        for item in data:
            dataset = item.get('dataset', '')
            complexity = item.get('complexity', 0)

            if dataset in cc_greater_than_1:
                total_by_dataset[dataset] += 1
                if complexity > 1:
                    cc_greater_than_1[dataset] += 1

        print("\nPrograms with CC > 1:")
        for dataset in ["CRUXEval", "HumanEval", "PythonSaga"]:
            total = total_by_dataset[dataset]
            greater = cc_greater_than_1[dataset]
            percentage = (greater / total * 100) if total > 0 else 0
            print(f"  {dataset}: {greater}/{total} ({percentage:.1f}%)")

    except Exception as e:
        print(f"Error during verification: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Update cyclomatic complexity values in the program metadata JSON."
    )

    parser.add_argument(
        "--json-file",
        default="artifacts/programs/runner_programs_with_coverage.json",
        help="Path to the JSON file that will be updated."
    )

    parser.add_argument(
        "--cruxeval-dir",
        default="data/CRUXEval/formatted_cruxeval_programs",
        help="Directory containing formatted CRUXEval Python programs."
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
    
    print("Starting cyclomatic complexity update...")
    print(f"Input JSON file: {args.json_file}")


    response = input("\nThis will update complexity values in the JSON file. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return


    update_json_complexity(args.json_file, args.cruxeval_dir, args.runner_programs_dir, backup=True)

    verify_updates(args.json_file)

    print(f"\n{'='*60}")
    print("UPDATE COMPLETE!")
    print("=" * 60)
    print(f"\nNote: Original JSON was backed up as '{args.json_file}.backup'")
if __name__ == "__main__":
    main()
