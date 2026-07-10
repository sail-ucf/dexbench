"""
FOCC Generator with Ground Truth Validation
This script generates Feasible Options of Code Coverage (FOCCs) and validates
them against ground truth coverage data. Missing ground truth coverage sets
are automatically added to ensure comprehensiveness.
"""

import json
import ast
import re
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
import argparse


class CFGBuilder:
    """Build Control Flow Graph and generate execution paths."""

    def __init__(self, function_registry: Dict[str, ast.FunctionDef]):
        self.graph = nx.DiGraph()
        self.counter = 0
        self.block_lines = {}
        self.registry = function_registry
        self.processed_funcs = set()
        self.current_call_stack = []

    def new_block(self) -> int:
        """Create a new basic block in the CFG."""
        bid = self.counter
        self.graph.add_node(bid)
        self.block_lines[bid] = []
        self.counter += 1
        return bid

    def add_line(self, block_id: int, lineno: int):
        """Add a line number to a basic block."""
        if lineno is not None and lineno > 0:
            if lineno not in self.block_lines[block_id]:
                self.block_lines[block_id].append(lineno)

    def add_lines_from_node(self, block_id: int, node: ast.AST):
        """Recursively add line numbers from an AST node."""
        if hasattr(node, 'lineno'):
            self.add_line(block_id, node.lineno)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.add_lines_from_node(block_id, child)

    def build(self, entry_func_name: str) -> Tuple[nx.DiGraph, Dict[int, List[int]], int, int]:
        """Build CFG for the entry function."""
        if entry_func_name not in self.registry:
            return self.graph, self.block_lines, 0, 0

        func_node = self.registry[entry_func_name]
        entry_id = self.new_block()
        exit_id = self.new_block()
        first_block = self.new_block()

        self.graph.add_edge(entry_id, first_block)
        self.processed_funcs.add(entry_func_name)
        self.current_call_stack.append(entry_func_name)

        final_blocks = self._process_stmts(func_node.body, first_block, exit_id, loop_ctx=None)

        for bid in final_blocks:
            self.graph.add_edge(bid, exit_id)

        self.current_call_stack.pop()
        return self.graph, self.block_lines, entry_id, exit_id

    def _find_nodes_of_type(self, root: ast.AST, node_types: tuple) -> List[ast.AST]:
        """Recursively find specific node types."""
        found = []
        for node in ast.walk(root):
            if isinstance(node, node_types):
                found.append(node)
        return found

    def _get_calls(self, node: ast.AST) -> List[str]:
        """Extract function/method calls from AST node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
                elif isinstance(child.func, ast.Lambda):
                    for subchild in ast.walk(child.func):
                        if isinstance(subchild, ast.Call) and isinstance(subchild.func, ast.Name):
                            calls.append(subchild.func.id)
            elif isinstance(child, ast.BinOp):
                op_to_method = {
                    ast.Add: '__add__', ast.Sub: '__sub__', ast.Mult: '__mul__',
                    ast.Div: '__truediv__', ast.FloorDiv: '__floordiv__',
                    ast.Mod: '__mod__', ast.Pow: '__pow__',
                }
                method_name = op_to_method.get(type(child.op))
                if method_name:
                    calls.append(method_name)
            elif isinstance(child, ast.Compare):
                op_to_method = {
                    ast.Gt: '__gt__', ast.Lt: '__lt__', ast.GtE: '__ge__',
                    ast.LtE: '__le__', ast.Eq: '__eq__', ast.NotEq: '__ne__',
                }
                for op in child.ops:
                    method_name = op_to_method.get(type(op))
                    if method_name:
                        calls.append(method_name)
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for subchild in ast.walk(child):
                    if isinstance(subchild, ast.Call) and isinstance(subchild.func, ast.Name):
                        calls.append(subchild.func.id)
        return calls

    def _find_nested_functions(self, node: ast.AST) -> List[ast.FunctionDef]:
        """Find nested function definitions."""
        nested_funcs = []
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.Lambda)):
                nested_funcs.append(child)
        return nested_funcs

    def _process_stmts(self, stmts: List[ast.stmt], current_block: int, function_exit_block: int, loop_ctx=None) -> List[int]:
        """Process a list of statements and update CFG."""
        active_blocks = [current_block]

        for stmt in stmts:
            if not active_blocks: break

            next_active = []
            called_funcs = self._get_calls(stmt)
            all_calls = called_funcs.copy()


            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                        class_name = stmt.value.func.id
                        for func_key in self.registry:
                            if func_key.startswith(f"{class_name}."):
                                method_name = func_key.split('.')[1]
                                if f"{class_name}.{method_name}" not in self.current_call_stack:
                                    all_calls.append(method_name)


            nested_funcs = self._find_nested_functions(stmt)
            for nested_func in nested_funcs:
                if isinstance(nested_func, ast.FunctionDef):
                    for subchild in ast.walk(stmt):
                        if isinstance(subchild, ast.Call) and isinstance(subchild.func, ast.Name):
                            if subchild.func.id == nested_func.name:
                                all_calls.append(nested_func.name)


            sibling_targets = []
            for call_name in all_calls:
                if call_name in self.registry and call_name not in self.processed_funcs:
                    sibling_targets.append(call_name)
                else:
                    for func_key in self.registry:
                        if func_key.endswith(f".{call_name}"):
                            if func_key not in self.current_call_stack:
                                sibling_targets.append(func_key)

            comps = self._find_nodes_of_type(stmt, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
            ternaries = self._find_nodes_of_type(stmt, (ast.IfExp,))

            for block_id in active_blocks:
                if hasattr(stmt, 'lineno'):
                    self.add_line(block_id, stmt.lineno)


                for target_name in sibling_targets:
                    if target_name in self.current_call_stack:
                        continue

                    target_node = self.registry[target_name]
                    sib_entry = self.new_block()
                    sib_exit = self.new_block()

                    self.graph.add_edge(block_id, sib_entry)
                    self.processed_funcs.add(target_name)
                    self.current_call_stack.append(target_name)

                    sib_finals = self._process_stmts(target_node.body, sib_entry, sib_exit, None)

                    self.processed_funcs.remove(target_name)
                    self.current_call_stack.pop()

                    for s_fin in sib_finals:
                        self.graph.add_edge(s_fin, sib_exit)
                    block_id = sib_exit


                if ternaries:
                    ternary_entry = self.new_block()
                    ternary_exit = self.new_block()
                    self.graph.add_edge(block_id, ternary_entry)

                    t_block = self.new_block()
                    self.graph.add_edge(ternary_entry, t_block)
                    self.graph.add_edge(t_block, ternary_exit)

                    f_block = self.new_block()
                    self.graph.add_edge(ternary_entry, f_block)
                    self.graph.add_edge(f_block, ternary_exit)

                    block_id = ternary_exit


                if comps:
                    comp_exits = self._handle_comprehension_logic(block_id, comps, None)
                    if comp_exits:
                        block_id = comp_exits[0]



                if isinstance(stmt, ast.Try):
                    try_start = self.new_block()
                    self.graph.add_edge(block_id, try_start)

                    except_start = self.new_block()
                    self.graph.add_edge(block_id, except_start)

                    try_exits = self._process_stmts(stmt.body, try_start, function_exit_block, loop_ctx)

                    handler_exits = []
                    for handler in stmt.handlers:
                        if hasattr(handler, 'lineno'):
                            self.add_line(except_start, handler.lineno)
                        h_exits = self._process_stmts(handler.body, except_start, function_exit_block, loop_ctx)
                        handler_exits.extend(h_exits)

                    next_active.extend(try_exits)
                    next_active.extend(handler_exits)

                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = stmt.name
                    is_called = False
                    for sub_stmt in stmts[stmts.index(stmt)+1:]:
                        for node in ast.walk(sub_stmt):
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                                if node.func.id == func_name:
                                    is_called = True
                                    break
                        if is_called:
                            break

                    if is_called:
                        nested_entry = self.new_block()
                        nested_exit = self.new_block()
                        next_active.append(block_id)
                        self.graph.add_edge(block_id, nested_entry)
                        nest_finals = self._process_stmts(stmt.body, nested_entry, nested_exit, None)
                        for nf in nest_finals:
                            self.graph.add_edge(nf, nested_exit)
                        next_active.append(nested_exit)
                    else:
                        next_active.append(block_id)

                elif isinstance(stmt, ast.Return):
                    self.graph.add_edge(block_id, function_exit_block)

                elif isinstance(stmt, ast.Break):
                    if loop_ctx:
                        self.graph.add_edge(block_id, loop_ctx['end'])

                elif isinstance(stmt, ast.Continue):
                    if loop_ctx:
                        self.graph.add_edge(block_id, loop_ctx['start'], type='back_edge')
                        self.graph.add_edge(block_id, loop_ctx['end'], type='focc_bypass')

                elif isinstance(stmt, ast.If):
                    true_start = self.new_block()
                    self.graph.add_edge(block_id, true_start)
                    true_exits = self._process_stmts(stmt.body, true_start, function_exit_block, loop_ctx)

                    false_start = self.new_block()
                    self.graph.add_edge(block_id, false_start)
                    if stmt.orelse:
                        false_exits = self._process_stmts(stmt.orelse, false_start, function_exit_block, loop_ctx)
                    else:
                        false_exits = [false_start]

                    next_active.extend(true_exits)
                    next_active.extend(false_exits)

                elif isinstance(stmt, (ast.For, ast.While)):
                    loop_start = self.new_block()
                    loop_body_start = self.new_block()
                    loop_end = self.new_block()

                    self.graph.add_edge(block_id, loop_start)
                    self.graph.add_edge(loop_start, loop_body_start)
                    self.graph.add_edge(loop_start, loop_end)

                    ctx = {'start': loop_start, 'end': loop_end}
                    body_exits = self._process_stmts(stmt.body, loop_body_start, function_exit_block, ctx)

                    for end_b in body_exits:
                        self.graph.add_edge(end_b, loop_start, type='back_edge')
                        self.graph.add_edge(end_b, loop_end, type='focc_bypass')

                    next_active.append(loop_end)

                else:
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Lambda):
                            return_node = ast.Return(value=node.body)
                            if hasattr(node, 'lineno'):
                                return_node.lineno = node.lineno
                            lambda_body = [return_node]
                            lambda_exits = self._process_stmts(lambda_body, block_id, function_exit_block, loop_ctx)
                            next_active.extend(lambda_exits)

                    next_active.append(block_id)

            active_blocks = list(set(next_active))

        return active_blocks

    def _handle_comprehension_logic(self, start_block: int, comps: List[ast.AST], explicit_exit: int = None) -> List[int]:
        """Create loop structure for comprehensions."""
        loop_header = self.new_block()
        loop_body = self.new_block()
        loop_exit = self.new_block()

        self.graph.add_edge(start_block, loop_header)
        self.graph.add_edge(loop_header, loop_body)
        self.graph.add_edge(loop_header, loop_exit)

        self.graph.add_edge(loop_body, loop_header, type='back_edge')
        self.graph.add_edge(loop_body, loop_exit, type='focc_bypass')

        for comp in comps:
            if hasattr(comp, 'elt'):
                self.add_lines_from_node(loop_body, comp.elt)
            if hasattr(comp, 'key'):
                self.add_lines_from_node(loop_body, comp.key)
            if hasattr(comp, 'value'):
                self.add_lines_from_node(loop_body, comp.value)
            for gen in comp.generators:
                for if_node in gen.ifs:
                    self.add_lines_from_node(loop_body, if_node)

            for node in ast.walk(comp):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.registry and func_name not in self.processed_funcs:
                        target_node = self.registry[func_name]
                        call_entry = self.new_block()
                        call_exit = self.new_block()

                        self.graph.add_edge(loop_body, call_entry)
                        self.processed_funcs.add(func_name)
                        self.current_call_stack.append(func_name)

                        call_finals = self._process_stmts(target_node.body, call_entry, call_exit, None)

                        self.processed_funcs.remove(func_name)
                        self.current_call_stack.pop()

                        for cf in call_finals:
                            self.graph.add_edge(cf, call_exit)
                        self.graph.add_edge(call_exit, loop_body)

        if explicit_exit is not None:
            self.graph.add_edge(loop_exit, explicit_exit)
            return []
        else:
            return [loop_exit]


class FOCCGenerator:
    """Generate and validate FOCCs with ground truth integration."""

    def __init__(self, max_paths: int = 50):
        self.programs_data = []
        self.max_paths = max_paths
        self.coverage_data = None

    def load_ground_truth_coverage(self) -> Dict[str, Dict]:
        """Load ground truth coverage data."""
        if not COVERAGE_FILE.exists():
            print(f"Coverage file not found: {COVERAGE_FILE}")
            return {}

        try:
            with open(COVERAGE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            coverage_by_id = {}
            for item in data:
                task_id = item.get('task_id', '')
                if task_id:

                    if '_' in task_id:
                        parts = task_id.split('_', 1)
                        if len(parts) == 2:
                            normalized_id = f"{parts[0]}/{parts[1]}"
                        else:
                            normalized_id = task_id
                    else:
                        normalized_id = task_id
                    coverage_by_id[normalized_id] = item

            print(f"Loaded {len(coverage_by_id)} ground truth coverage records")
            return coverage_by_id
        except Exception as e:
            print(f"Error loading coverage data: {e}")
            return {}

    def get_static_lines(self, code: str) -> List[int]:
        """Get all lines that should always be considered covered."""
        lines = []
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if hasattr(node, 'lineno'):
                        lines.append(node.lineno)

                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'lineno'):
                        lines.append(node.lineno)

                    for decorator in node.decorator_list:
                        if hasattr(decorator, 'lineno'):
                            lines.append(decorator.lineno)

                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for arg in node.args.args + node.args.kwonlyargs:
                            if hasattr(arg, 'lineno') and arg.lineno:
                                lines.append(arg.lineno)

                        if node.returns and hasattr(node.returns, 'lineno'):
                            lines.append(node.returns.lineno)

            for node in tree.body:
                if hasattr(node, 'lineno'):
                    lines.append(node.lineno)

            return sorted(list(set(lines)))
        except Exception as e:
            print(f"Static lines error: {e}")
            return []

    def find_dag_paths(self, graph, start, end) -> List[List[int]]:
        """Find paths in the CFG DAG."""

        dag_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') != 'back_edge']
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes())
        dag.add_edges_from(dag_edges)

        try:

            path_gen = nx.all_simple_paths(dag, start, end)
            paths = []
            for i, path in enumerate(path_gen):
                if i >= self.max_paths:
                    break
                paths.append(path)
            return paths
        except:
            return []

    def determine_entry_point(self, code: str) -> Tuple[str, Dict[str, ast.FunctionDef]]:
        """Determine the entry function for CFG construction."""
        tree = ast.parse(code)
        func_map = {}
        first_func = None


        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node


        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_class = None
                parent = getattr(node, 'parent', None)
                while parent:
                    if isinstance(parent, ast.ClassDef):
                        parent_class = parent.name
                        break
                    parent = getattr(parent, 'parent', None)

                if parent_class:
                    func_map[f"{parent_class}.{node.name}"] = node
                else:
                    func_map[node.name] = node

                if first_func is None and not node.name.startswith('test_') and node.name != 'main':
                    if parent_class:
                        first_func = f"{parent_class}.{node.name}"
                    else:
                        first_func = node.name

        if not func_map:
            return None, {}


        test_call_str = self.extract_test_case_ast(code)
        target_func = first_func

        if test_call_str:
            for fname in func_map:
                simple_name = fname.split('.')[-1]
                if re.search(r'\b' + re.escape(simple_name) + r'\b', test_call_str):
                    if fname in test_call_str or simple_name in test_call_str:
                        target_func = fname
                        break

            try:
                test_tree = ast.parse(test_call_str + "()" if test_call_str else "")
                for node in ast.walk(test_tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in func_map:
                            target_func = func_name
                            break
            except:
                pass

        return target_func, func_map

    def extract_test_case_ast(self, code: str) -> str:
        """Extract the test case from the code."""
        try:
            tree = ast.parse(code)
            test_calls = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and
                        node.func.attr in ['assertEqual', 'assertTrue', 'assertFalse', 'assertRaises']):
                        if node.args:
                            try:
                                return ast.unparse(node.args[0])
                            except:
                                import astunparse
                                return astunparse.unparse(node.args[0])

                    if isinstance(node.func, ast.Name):
                        test_calls.append(ast.unparse(node) if hasattr(ast, 'unparse') else str(node))

            if test_calls:
                return test_calls[0]

            return ""
        except:

            lines = code.split('\n')
            for line in reversed(lines):
                if 'assertEqual' in line or 'assertTrue' in line or 'assertFalse' in line:
                    match = re.search(r'assert(?:Equal|True|False)\(([^,]+)', line)
                    if match:
                        return match.group(1).strip()
            return ""

    def generate_foccs(self, code: str) -> Tuple[List[List[int]], str]:
        """Generate FOCCs for the given code."""
        try:
            entry_name, func_map = self.determine_entry_point(code)
            if not entry_name:
                return [], "unknown"

            builder = CFGBuilder(func_map)
            graph, block_lines, start, end = builder.build(entry_name)

            raw_paths = self.find_dag_paths(graph, start, end)

            static_lines = self.get_static_lines(code)
            foccs = []

            for path in raw_paths:
                coverage = set(static_lines)
                for block_id in path:
                    lines = block_lines.get(block_id, [])
                    coverage.update(lines)
                if coverage:
                    foccs.append(sorted(list(coverage)))


            unique_foccs = []
            seen = set()
            for f in foccs:
                t = tuple(f)
                if t not in seen:
                    seen.add(t)
                    unique_foccs.append(f)


            simple_name = entry_name.split('.')[-1] if '.' in entry_name else entry_name
            return unique_foccs, simple_name

        except Exception as e:
            print(f"FOCC Gen Error: {e}")
            import traceback
            traceback.print_exc()
            return [], "error"

    def normalize_coverage(self, coverage_set: List[int]) -> Tuple[int, ...]:
        """Normalize coverage set for comparison."""
        return tuple(sorted(set(coverage_set)))

    def find_matching_foccs(self, ground_truth: List[int], foccs: List[List[int]]) -> bool:
        """Check if ground truth coverage exists in FOCCs."""
        normalized_gt = self.normalize_coverage(ground_truth)

        for focc in foccs:
            if self.normalize_coverage(focc) == normalized_gt:
                return True
        return False

    def validate_and_augment_foccs(self, program_id: str, foccs: List[List[int]]) -> Tuple[List[List[int]], bool]:
        """Validate FOCCs against ground truth and augment if missing."""
        if not self.coverage_data:
            return foccs, False

        if program_id not in self.coverage_data:
            return foccs, False

        coverage_item = self.coverage_data[program_id]
        ground_truth = coverage_item.get('coverage_metadata', {}).get('covered_lines', [])

        if not ground_truth:
            return foccs, False


        if self.find_matching_foccs(ground_truth, foccs):
            return foccs, False


        augmented_foccs = foccs.copy()
        augmented_foccs.append(sorted(list(set(ground_truth))))


        unique_foccs = []
        seen = set()
        for f in augmented_foccs:
            t = tuple(sorted(set(f)))
            if t not in seen:
                seen.add(t)
                unique_foccs.append(sorted(list(t)))

        return unique_foccs, True

    def process_program_file(self, filepath: Path, dataset: str, program_id: str) -> Dict:
        """Process a single program file."""
        print(f"Processing {program_id}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            foccs, func_name = self.generate_foccs(code)
            test_case = self.extract_test_case_ast(code)


            foccs, was_augmented = self.validate_and_augment_foccs(program_id, foccs)

            if was_augmented:
                print(f"  -> Augmented with ground truth coverage")

            if not foccs:
                return None


            formatted_foccs = ["Set 1: Error"]
            for i, focc in enumerate(foccs, 2):
                line_str = ", ".join(map(str, focc))
                formatted_foccs.append(f"{i}. Lines {line_str}")


            lines = code.split('\n')
            serialized = "\n".join([f"Line {i+1}: {l}" for i, l in enumerate(lines)])
            serialized += f"\n\nGIVEN TEST CASE - \n{test_case}"
            serialized += f"\n\nGIVEN POSSIBLE SETS OF CODE COVERAGE - \n" + "\n".join(formatted_foccs)

            return {
                "program_id": program_id,
                "dataset": dataset,
                "filepath": str(filepath),
                "function_name": func_name,
                "test_case": test_case,
                "foccs": foccs,
                "augmented": was_augmented,
                "serialized_code": serialized
            }
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None

    def process_wrapper(self, path: Path, dataset: str, pid: str):
        """Wrapper to process a program file."""
        if path.exists():
            res = self.process_program_file(path, dataset, pid)
            if res:
                self.programs_data.append(res)
        else:
            print(f"File not found: {path}")

    def collect_and_save(self):
        """Main method to collect, validate, and save FOCCs."""
        print("Inspecting Loading ground truth coverage data...")
        self.coverage_data = self.load_ground_truth_coverage()

        print("\nInspecting Scanning HumanEval...")
        for p_dir in HUMANEVAL_BASE.glob("HumanEval_*"):
            if p_dir.is_dir():
                dir_name = p_dir.name
                program_id = dir_name.replace('_', '/')
                self.process_wrapper(p_dir / "solution.py", "HumanEval", program_id)

        print("\nInspecting Scanning PythonSaga...")
        for p_dir in PYTHONSAGA_BASE.glob("PythonSaga_*"):
            if p_dir.is_dir():
                dir_name = p_dir.name
                program_id = dir_name.replace('_', '/')
                self.process_wrapper(p_dir / "solution.py", "PythonSaga", program_id)

        print("\nInspecting Scanning CRUXEval...")
        for p_file in CRUXEVAL_BASE.glob("sample_*.py"):
            stem = p_file.stem
            match = re.search(r'sample_(\d+)', stem)
            if match:
                number = match.group(1)
                program_id = f"CRUXEval/{number}"
            else:
                program_id = f"CRUXEval/{stem}"

            self.process_wrapper(p_file, "CRUXEval", program_id)


        self.save_results()


        self.generate_crispe_files()

    def save_results(self):
        """Save FOCCs data to JSON file."""
        outfile = FOCC_OUTPUT_DIR / "all_programs_foccs.json"


        total_programs = len(self.programs_data)
        augmented_count = sum(1 for p in self.programs_data if p.get('augmented', False))


        json_str = json.dumps(self.programs_data, indent=2)
        json_str = re.sub(r'\[\s*([\d,\s]+?)\s*\]',
                         lambda m: '[' + re.sub(r'\s+', ' ', m.group(1).replace('\n', '')).strip() + ']',
                         json_str)

        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(json_str)

        print(f"\n Saved {total_programs} programs to {outfile}")
        if augmented_count > 0:
            print(f"   - {augmented_count} programs were augmented with ground truth coverage")


        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate a summary report of the FOCC generation process."""
        if not self.programs_data:
            print("No programs processed!")
            return


        dataset_stats = {}
        augmentation_stats = {}

        for prog in self.programs_data:
            dataset = prog['dataset']
            dataset_stats[dataset] = dataset_stats.get(dataset, 0) + 1
            if prog.get('augmented', False):
                augmentation_stats[dataset] = augmentation_stats.get(dataset, 0) + 1

        print("\n" + "="*80)
        print("FOCC GENERATION SUMMARY REPORT")
        print("="*80)

        print(f"\n Dataset Statistics:")
        for dataset, count in sorted(dataset_stats.items()):
            augmented = augmentation_stats.get(dataset, 0)
            print(f"   {dataset}: {count} programs ({augmented} augmented)")


        focc_sizes = []
        for prog in self.programs_data:
            for focc in prog['foccs']:
                focc_sizes.append(len(focc))

        if focc_sizes:
            avg_size = sum(focc_sizes) / len(focc_sizes)
            print(f"\n FOCC Size Statistics:")
            print(f"   Average FOCC size: {avg_size:.1f} lines")
            print(f"   Min FOCC size: {min(focc_sizes)} lines")
            print(f"   Max FOCC size: {max(focc_sizes)} lines")


        if self.coverage_data:
            coverage_ids = set(self.coverage_data.keys())
            focc_ids = {p['program_id'] for p in self.programs_data}
            missing_in_foccs = coverage_ids - focc_ids
            missing_in_coverage = focc_ids - coverage_ids

            if missing_in_foccs:
                print(f"\n Programs in coverage data but not in FOCCs ({len(missing_in_foccs)}):")
                for pid in sorted(missing_in_foccs)[:5]:
                    print(f"   - {pid}")
                if len(missing_in_foccs) > 5:
                    print(f"   ... and {len(missing_in_foccs) - 5} more")

            if missing_in_coverage:
                print(f"\n Programs in FOCCs but not in coverage data ({len(missing_in_coverage)}):")
                for pid in sorted(missing_in_coverage)[:5]:
                    print(f"   - {pid}")
                if len(missing_in_coverage) > 5:
                    print(f"   ... and {len(missing_in_coverage) - 5} more")

    def generate_crispe_files(self):
        """Generate CRISPE-ready text files."""
        crispe_dir = FOCC_OUTPUT_DIR / "crispe_ready"
        crispe_dir.mkdir(exist_ok=True)


        template_path = CRISPE_TEMPLATE
        if template_path.exists():
            template = template_path.read_text(encoding='utf-8')
        else:
            template = "{program_code}"

        generated_count = 0
        for prog in self.programs_data:
            filename = f"{prog['program_id'].replace('/', '_')}_crispe.txt"
            filepath = crispe_dir / filename

            content = template.replace("{program_code}", prog['serialized_code'])
            filepath.write_text(content, encoding='utf-8')
            generated_count += 1

        print(f"\n Generated {generated_count} CRISPE-ready files in {crispe_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and validate FOCCs."
    )

    parser.add_argument(
        "--max-paths",
        type=int,
        default=50,
        help="Maximum number of paths to explore in CFG."
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ground truth validation."
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/programs/focc"),
        help="Directory where FOCC outputs will be saved."
    )

    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path(
            "artifacts/programs/"
            "runner_programs_with_coverage.json"
        ),
        help="Path to the coverage JSON file used for validation."
    )

    parser.add_argument(
        "--crispe-template",
        type=Path,
        default=Path(
            "data/prompts/reasoning/"
            "crispe_predict_coverage_original.txt"
        ),
        help="Path to the CRISPE prompt template."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    global HUMANEVAL_BASE
    global PYTHONSAGA_BASE
    global CRUXEVAL_BASE
    global FOCC_OUTPUT_DIR
    global COVERAGE_FILE
    global CRISPE_TEMPLATE

    HUMANEVAL_BASE = args.runner_programs_dir
    PYTHONSAGA_BASE = args.runner_programs_dir
    CRUXEVAL_BASE = args.cruxeval_dir
    FOCC_OUTPUT_DIR = args.output_dir
    COVERAGE_FILE = args.coverage_file
    CRISPE_TEMPLATE = args.crispe_template

    FOCC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = FOCCGenerator(max_paths=args.max_paths)

    if args.no_validate:
        generator.coverage_data = {}
        print("Ground truth validation disabled")

    generator.collect_and_save()


if __name__ == "__main__":
    main()