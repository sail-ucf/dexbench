import subprocess
import re
import shutil
import json
from pathlib import Path
import argparse

TIMEOUT = 30

def run_slipcover(program_path: Path) -> tuple[float | None, bool, str]:
    try:
        result = subprocess.run(
            ["python3", "-m", "slipcover", str(program_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        raw_output = result.stdout + result.stderr
        print(f"Inspecting SlipCover raw output for {program_path.name}:")
        print("=" * 50)
        print(raw_output)
        print("=" * 50)
    except subprocess.TimeoutExpired as e:
        raw_output = getattr(e, 'stdout', '') + getattr(e, 'stderr', '') or "Timeout expired"
        print(f"Timeout for {program_path.name}")
        return None, False, raw_output
    except Exception as e:
        print(f"Exception running {program_path.name}: {e}")
        return None, False, f"Exception: {e}"

    if result.returncode != 0:
        print(f"Non-zero exit code for {program_path.name}: {result.returncode}")
        return None, False, result.stdout + result.stderr

    return parse_slipcover_output(result.stdout), True, result.stdout + result.stderr

def parse_slipcover_output(output: str) -> float | None:
    """Parse SlipCover output to extract coverage percentage"""
    print(f"Parsing SlipCover output...")

    lines = output.splitlines()
    print(f"Raw output lines: {len(lines)}")

    for line in lines:
        line = line.strip()

        if not line or line.startswith("File") or set(line) <= set("- "):
            continue

        print(f"Analyzing line: '{line}'")



        parts = re.split(r'\s+', line)
        print(f"Line parts: {parts}")


        if len(parts) >= 4:
            coverage_str = parts[3]
            try:
                coverage = float(coverage_str)
                print(f"SUCCESS: Parsed coverage = {coverage}%")
                return coverage
            except ValueError:
                print(f"Could not parse '{coverage_str}' as float")
                continue

    print("Could not parse coverage from output")
    return None

def extract_assertions(test_path: Path) -> list[str]:
    """Returns list of all assert lines in test file"""
    try:
        content = test_path.read_text()
        lines = content.splitlines()
        assertions = [line.strip() for line in lines if line.strip().startswith("assert ")]
        print(f"Found {len(assertions)} assertions in test file")
        for i, assertion in enumerate(assertions):
            print(f"  Assertion {i+1}: {assertion}")
        return assertions
    except Exception as e:
        print(f"Failed to read test file {test_path}: {e}")
        return []

def update_solution_with_assertion(solution_path: Path, func_name: str, assertion_line: str) -> None:
    try:
        original_code = solution_path.read_text()
    except Exception as e:
        print(f"Failed to read {solution_path}: {e}")
        return

    print(f"Updating solution with assertion: {assertion_line}")


    assertion_expr = assertion_line[len("assert "):].strip()
    new_assertion = assertion_expr.replace("candidate", func_name)

    print(f"Converted assertion: {new_assertion}")


    if "==" in new_assertion:
        lhs, rhs = new_assertion.split("==", 1)
        test = f'unittest.TestCase().assertEqual({lhs.strip()}, {rhs.strip()})'
    elif "!=" in new_assertion:
        lhs, rhs = new_assertion.split("!=", 1)
        test = f'unittest.TestCase().assertNotEqual({lhs.strip()}, {rhs.strip()})'
    elif new_assertion.startswith("not "):
        test = f'unittest.TestCase().assertFalse({new_assertion[4:].strip()})'
    else:
        test = f'unittest.TestCase().assertTrue({new_assertion})'

    print(f"Final test code: {test}")


    code_lines = original_code.strip().splitlines()
    while code_lines and code_lines[-1].strip().startswith("unittest.TestCase().assert"):
        code_lines.pop()
    code_without_test = "\n".join(code_lines).rstrip()


    updated_code = f"{code_without_test}\n\n{test}\n"

    try:
        solution_path.write_text(updated_code)
        print(f"Successfully updated solution.py")
        return updated_code
    except Exception as e:
        print(f"Failed to write updated solution: {e}")
        return None

def get_entry_point_name(code: str) -> str | None:
    match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code)
    if match:
        func_name = match.group(1)
        print(f"Found entry point: {func_name}")
        return func_name
    else:
        print("Could not find entry point function")
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SlipCover on runner programs and remove or update "
            "programs that still have 100% coverage."
        )
    )

    parser.add_argument(
        "--runner-dir",
        type=Path,
        default=Path("artifacts/programs/runner_programs"),
        help="Directory containing runner program folders."
    )

    parser.add_argument(
        "--runner-json",
        type=Path,
        default=Path("artifacts/programs/runner_programs.json"),
        help="Path to the runner programs JSON file."
    )

    return parser.parse_args()

def main():

    args = parse_args()

    if not args.runner_json.exists():
        print(f"Runner programs JSON not found: {args.runner_json}")
        return

    try:
        with open(args.runner_json, 'r', encoding='utf-8') as f:
            runner_programs = json.load(f)
        print(f"Loaded {len(runner_programs)} programs from {args.runner_json}")
    except Exception as e:
        print(f"Failed to load {args.runner_json}: {e}")
        return


    program_map = {prog['task_id']: prog for prog in runner_programs}
    json_updated = False

    runner_dirs = sorted(p for p in args.runner_dir.iterdir() if p.is_dir())
    print(f"Found {len(runner_dirs)} runner programs. Processing...")

    kept = 0
    removed = 0
    updated = 0

    for task_dir in runner_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {task_dir.name}")
        print(f"{'='*60}")

        solution_path = task_dir / "solution.py"
        test_path = task_dir / "test.py"

        if not solution_path.exists():
            print(f"Missing solution.py in {task_dir}")
            continue
        if not test_path.exists():
            print(f"Missing test.py in {task_dir}")
            continue


        print(f"Running initial SlipCover check...")
        coverage, success, _ = run_slipcover(solution_path)

        if not success:
            print(f"Could not run SlipCover on {task_dir.name}")
            removed += 1
            shutil.rmtree(task_dir)
            continue

        if coverage is None:
            print(f"Could not parse coverage for {task_dir.name}")
            removed += 1
            shutil.rmtree(task_dir)
            continue

        print(f"Initial coverage: {coverage}%")

        if coverage < 100:
            print(f"Already <100% coverage - keeping.")
            kept += 1
            continue

        print(f"100% coverage - trying alternate tests...")
        assertions = extract_assertions(test_path)
        if not assertions:
            print(f"No assertions found in {test_path}")
            shutil.rmtree(task_dir)
            removed += 1
            continue

        original_code = solution_path.read_text()
        entry_point = get_entry_point_name(original_code)
        if not entry_point:
            print("Couldn't find entry point function.")
            shutil.rmtree(task_dir)
            removed += 1
            continue

        downgraded = False
        for i, assertion in enumerate(assertions):
            print(f"\n Trying assertion {i+1}/{len(assertions)}...")
            updated_code = update_solution_with_assertion(solution_path, entry_point, assertion)

            new_cov, success, _ = run_slipcover(solution_path)
            if not success:
                print(f"Failed to run with new assertion")
                continue
            if new_cov is None:
                print(f"Could not parse coverage with new assertion")
                continue

            print(f"New coverage: {new_cov}%")

            if new_cov < 100:
                print(f"SUCCESS: Coverage downgraded to {new_cov}% with new test")
                updated += 1
                downgraded = True


                task_id = task_dir.name.replace("_", "/")
                if task_id in program_map and updated_code:
                    program_map[task_id]['runnable_script'] = updated_code
                    json_updated = True
                    print(f"Updated {args.runner_json} entry for {task_id}")
                break
            else:
                print(f"Still 100% coverage with this assertion")

        if not downgraded:
            print("Still 100% after trying all assertions - removing.")
            shutil.rmtree(task_dir)
            removed += 1


    if json_updated:
        try:
            with open(args.runner_json, 'w', encoding='utf-8') as f:
                json.dump(runner_programs, f, indent=4)
            print(f"\n Successfully updated {args.runner_json} with {updated} modified programs")
        except Exception as e:
            print(f"Failed to update {args.runner_json}: {e}")
    else:
        print(f"\n No updates needed for {args.runner_json}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY:")
    print(f"{'='*60}")
    print(f"Kept as-is:     {kept}")
    print(f"Updated tests:  {updated}")
    print(f"Removed:        {removed}")
    print(f"Final count:     {kept + updated}")

if __name__ == "__main__":
    main()
