"""
ablation_study_preparer.py

Creates ablation study programs by modifying the original programs:
- Replaces assertion with a function call that uses an input placeholder.
- Creates new directory structure and JSON file
- All programs in flat directory with consistent naming
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

INPUT_PLACEHOLDER = "DEXBENCH_INPUT_PLACEHOLDER"

def convert_to_ablation_format(script_source: str) -> Optional[str]:
    """Convert a script to ablation study format."""
    lines = script_source.splitlines()


    assertion_idx = None
    assertion_line = None
    for i, line in enumerate(lines):
        if 'unittest.TestCase().assert' in line:
            assertion_idx = i
            assertion_line = line
            break

    if assertion_idx is None:
        print("    No assertion found, skipping")
        return None

    print(f"    Found assertion: {assertion_line}")


    function_name = None


    parts = assertion_line.split('assert', 1)
    if len(parts) >= 2:
        after_assert = parts[1]
        start_idx = after_assert.find('(')
        if start_idx != -1:
            content = after_assert[start_idx + 1:]
            func_match = re.search(r'(\w+)\(', content)
            if func_match:
                function_name = func_match.group(1)
                print(f"    Function name from assertion: {function_name}")


    if not function_name:

        func_def_match = re.findall(r'^def\s+(\w+)\s*\(', script_source, re.MULTILINE)
        if func_def_match:
            function_name = func_def_match[-1]
            print(f"    Function name from definition: {function_name}")

    if not function_name:
        print(f"    No function name found")
        return None

    new_function_call = f"{function_name}({INPUT_PLACEHOLDER})"
    lines[assertion_idx] = new_function_call

    return '\n'.join(lines)

def get_program_filename(task_id: str) -> str:
    """Convert task_id to appropriate filename"""
    if task_id.startswith("CRUXEval/"):
        number = task_id.replace("CRUXEval/", "")
        return f"sample_{number}.py"
    elif task_id.startswith("HumanEval/"):
        number = task_id.replace("HumanEval/", "")
        return f"HumanEval_{number}.py"
    elif task_id.startswith("PythonSaga/"):
        number = task_id.replace("PythonSaga/", "")
        return f"PythonSaga_{number}.py"
    else:

        return f"{task_id.replace('/', '_')}.py"

def create_ablation_program_files(programs_data: List[Dict[str, Any]], ablation_dir: Path):
    """Create ablation study program files in flat directory"""
    print("Creating ablation study program files...")


    if ablation_dir.exists():
        shutil.rmtree(ablation_dir)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    created_count = 0
    skipped_count = 0

    for program in programs_data:
        task_id = program["task_id"]
        original_script = program["runnable_script"]

        print(f"  Processing: {task_id}")


        ablation_script = convert_to_ablation_format(original_script)
        if ablation_script is None:
            skipped_count += 1
            continue


        filename = get_program_filename(task_id)
        program_file = ablation_dir / filename


        program_file.write_text(ablation_script, encoding="utf-8")

        created_count += 1
        print(f"    Created: {filename}")

    print(f"Created {created_count} ablation study program files")
    print(f"Skipped {skipped_count} programs")
    return created_count

def create_ablation_json(programs_data: List[Dict[str, Any]], ablation_json: Path):
    """Create ablation study JSON with updated runnable scripts"""
    print("Creating ablation study JSON...")

    ablation_data = []
    skipped_count = 0

    for program in programs_data:
        task_id = program["task_id"]
        original_script = program["runnable_script"]


        ablation_script = convert_to_ablation_format(original_script)
        if ablation_script is None:
            skipped_count += 1
            continue


        ablation_program = program.copy()
        ablation_program["runnable_script"] = ablation_script
        ablation_program["solution_code"] = ablation_script


        ablation_program["ablation_study"] = True


        ablation_program["ablation_filename"] = get_program_filename(task_id)

        ablation_data.append(ablation_program)


    ablation_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ablation_json, "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, indent=2)

    print(f"Created ablation JSON with {len(ablation_data)} programs")
    print(f"Skipped {skipped_count} programs")
    return ablation_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ablation-study programs and metadata JSON."
    )

    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path(
            "data/programs/"
            "runner_programs_with_coverage.json"
        ),
        help="Path to the original coverage JSON file."
    )

    parser.add_argument(
        "--ablation-json",
        type=Path,
        default=Path(
            "data/programs/"
            "ablation_study_programs_with_coverage.json"
        ),
        help="Path where the ablation-study JSON file will be saved."
    )

    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path(
            "data/programs/"
            "ablation_study_programs"
        ),
        help="Directory where ablation-study program files will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()

    print("Preparing Ablation Study Programs")
    print("=" * 50)


    if not args.coverage_json.exists():
        print(f"Original coverage JSON not found: {args.coverage_json}")
        return

    try:
        with open(args.coverage_json, "r", encoding="utf-8") as f:
            programs_data = json.load(f)
        print(f"Loaded {len(programs_data)} programs from original data")
    except Exception as e:
        print(f"Error loading original JSON: {e}")
        return


    file_count = create_ablation_program_files(programs_data, args.ablation_dir)


    ablation_data = create_ablation_json(programs_data, args.ablation_json)


    print("\n" + "=" * 50)
    print("ABLATION STUDY PREPARATION COMPLETE")
    print("=" * 50)
    print(f"Created files: {file_count}")
    print(f"Created JSON entries: {len(ablation_data)}")
    print(f"Output directory: {args.ablation_dir}")
    print(f"Output JSON: {args.ablation_json}")


    print(f"\nDirectory structure:")
    sample_files = list(args.ablation_dir.glob("*.py"))[:5]
    for file in sample_files:
        print(f"  {file.name}")
    total_files = len(list(args.ablation_dir.glob("*.py")))
    if total_files > 5:
        print(f"  ... and {total_files - 5} more files")


    if ablation_data:
        sample = ablation_data[0]
        print(f"\nSample converted program ({sample['task_id']}):")
        print("-" * 40)
        script_lines = sample["runnable_script"].splitlines()[-3:]
        for line in script_lines:
            print(f"  {line}")
        print("-" * 40)

if __name__ == "__main__":
    main()
