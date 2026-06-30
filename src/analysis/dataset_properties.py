import json
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from typing import Dict, List, Any


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CodeAnalyzer:
    """Analyze Python code for various complexity metrics"""

    def __init__(self, print_metrics=False):
        self.ast_cache = {}
        self.print_metrics = print_metrics

    def parse_code(self, code: str) -> ast.AST:
        """Parse Python code into AST"""
        if code in self.ast_cache:
            return self.ast_cache[code]

        try:
            tree = ast.parse(code)
            self.ast_cache[code] = tree
            return tree
        except:
            tree = ast.Module(body=[], type_ignores=[])
            self.ast_cache[code] = tree
            return tree

    def cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate McCabe's cyclomatic complexity"""

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

        visitor = ComplexityVisitor()
        visitor.visit(tree)


        has_function = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          for node in ast.walk(tree))


        if not has_function:
            return 0

        return visitor.complexity

    def lines_of_code(self, code: str) -> int:
        """Count non-empty lines of code"""
        lines = code.strip().split('\n')
        loc = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                loc += 1
        return loc

    def intra_class_dependencies(self, tree: ast.AST) -> int:
        """Count intra-class dependencies - returns INTEGER count (0, 1, 2, ...)"""
        dependencies = 0

        class ClassDependencyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_class = False
                self.dependencies = 0
                self.class_methods = set()

            def visit_ClassDef(self, node):
                self.in_class = True
                self.class_methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                self.generic_visit(node)
                self.in_class = False

            def visit_Call(self, node):
                if self.in_class and isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.class_methods:
                        self.dependencies += 1
                self.generic_visit(node)

        visitor = ClassDependencyVisitor()
        visitor.visit(tree)
        return visitor.dependencies

    def nested_constructs(self, tree: ast.AST) -> int:
        """Count maximum nesting depth of constructs"""
        max_depth = 0

        class NestedVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0

            def visit_If(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_For(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_While(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

        visitor = NestedVisitor()
        visitor.visit(tree)
        return visitor.max_depth

    def loop_length(self, tree: ast.AST) -> float:
        """Calculate average loop length"""
        loop_lengths = []

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_lengths = []

            def _count_body_statements(self, node):
                """Count statements in a loop body"""
                count = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
                                        ast.Return, ast.Delete, ast.Pass, ast.Break,
                                        ast.Continue, ast.Assert)):
                        count += 1
                return count

            def visit_For(self, node):
                body_statements = self._count_body_statements(node)
                if body_statements > 0:
                    self.loop_lengths.append(body_statements)
                self.generic_visit(node)

            def visit_While(self, node):
                body_statements = self._count_body_statements(node)
                if body_statements > 0:
                    self.loop_lengths.append(body_statements)
                self.generic_visit(node)

        visitor = LoopVisitor()
        visitor.visit(tree)

        if visitor.loop_lengths:
            return float(np.mean(visitor.loop_lengths))
        return 0.0

    def analyze_code(self, code: str, sample_id: str = "") -> Dict[str, float]:
        """Analyze code and return all metrics"""
        try:
            tree = self.parse_code(code)


            cc = self.cyclomatic_complexity(tree)
            loc = self.lines_of_code(code)
            dep = self.intra_class_dependencies(tree)
            nc = self.nested_constructs(tree)
            ll = self.loop_length(tree)


            if self.print_metrics:
                print(f"  {sample_id}: CC={cc}, LOC={loc}, DEP={dep}, NC={nc}, LL={ll:.2f}")

            return {
                'CC': float(cc),
                'LOC': float(loc),
                'DEP': float(dep),
                'NC': float(nc),
                'LL': float(ll)
            }

        except Exception:
            if self.print_metrics:
                print(f"  {sample_id}: ERROR in analysis")
            return {
                'CC': 0.0,
                'LOC': 0.0,
                'DEP': 0.0,
                'NC': 0.0,
                'LL': 0.0
            }

def load_dataset(filepath: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Load dataset from jsonl file"""
    data = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return data

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                code = ""

                if dataset_name == "CRUXEval":
                    code = item.get('code', '')
                    task_id = item.get('id', 'unknown')
                elif dataset_name == "HumanEval":
                    code = item.get('prompt', '') + item.get('canonical_solution', '')
                    task_id = item.get('task_id', 'unknown')
                elif dataset_name == "PythonSaga":
                    code = item.get('prompt', '') + item.get('canonical_solution', '')
                    task_id = item.get('task_id', 'unknown')

                if code:
                    data.append({
                        'id': task_id,
                        'code': code,
                        'dataset': dataset_name
                    })

            except json.JSONDecodeError:
                continue

    return data

def create_violin_plots(df: pd.DataFrame, output_dir: str):
    """Create violin plots like the reference image"""


    os.makedirs(output_dir, exist_ok=True)

    metrics = ["CC", "LOC", "DEP", "NC", "LL"]
    metric_labels = ["CC", "LoC", "DEP", "NC", "LL"]


    valid_df = df[(df['CC'] > 0) & (df['LOC'] > 0)].copy()

    print(f"\nSamples with classes (DEP > 0):")
    for dataset in valid_df['Dataset'].unique():
        has_classes = len(valid_df[(valid_df['Dataset'] == dataset) & (valid_df['DEP'] > 0)])
        total = len(valid_df[valid_df['Dataset'] == dataset])
        if total > 0:
            print(f"  {dataset}: {has_classes}/{total} ({has_classes/total*100:.1f}%)")


    fig, axes = plt.subplots(1, 5, figsize=(20, 5))


    palette = {"CRUXEval": "#d9e39d", "HumanEval": "#f7b0d1", "PythonSaga": "#a6d8a8"}

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]


        plot_data = []
        for dataset in valid_df['Dataset'].unique():
            values = valid_df[valid_df['Dataset'] == dataset][metric]


            if len(values) > 10:
                q99 = values.quantile(0.99)
                filtered = values[values <= q99]
            else:
                filtered = values


            if metric == 'DEP' and len(filtered) > 0:
                non_zero = filtered[filtered > 0]
                if len(non_zero) > 0:
                    print(f"  {dataset} DEP: {len(non_zero)}/{len(filtered)} non-zero, "
                          f"mean={non_zero.mean():.1f}, max={non_zero.max():.0f}")

            for val in filtered:
                plot_data.append({'Dataset': dataset, metric: val})

        if plot_data:
            temp_df = pd.DataFrame(plot_data)


            sns.violinplot(
                data=temp_df,
                x="Dataset",
                y=metric,
                ax=ax,
                inner="quart",
                palette=palette,
                cut=0
            )


            medians = temp_df.groupby('Dataset')[metric].median()
            for j, dataset in enumerate(['CRUXEval', 'HumanEval', 'PythonSaga']):
                if dataset in medians.index:
                    ax.hlines(medians[dataset], j-0.3, j+0.3,
                             color='red', linestyle='--', linewidth=1, alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)


        ax.set_title(f"({chr(97+i)}) {label}", fontsize=14, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=12)


        ax.tick_params(axis='both', labelsize=10)


        ax.grid(True, alpha=0.3, linestyle='--')


        if metric == 'CC':
            ax.set_ylim(bottom=0.5)
        elif metric in ['LOC', 'DEP', 'NC', 'LL']:
            ax.set_ylim(bottom=0)


        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.suptitle("Distribution of Subject Programs per Different Complexity Metrics",
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()


    output_path = os.path.join(output_dir, "violin_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nOK Violin plots saved as {output_path}")


    stats_file = os.path.join(output_dir, "detailed_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("DETAILED METRICS STATISTICS\n")
        f.write("=" * 50 + "\n\n")

        for metric in metrics:
            f.write(f"{metric}:\n")
            f.write("-" * 40 + "\n")
            for dataset in valid_df['Dataset'].unique():
                values = valid_df[valid_df['Dataset'] == dataset][metric]
                non_zero = values[values > 0] if metric in ['DEP', 'NC', 'LL'] else values

                if len(non_zero) > 0:
                    f.write(f"  {dataset:12s} | Mean: {non_zero.mean():7.2f} | "
                           f"Median: {non_zero.median():7.2f} | Min: {non_zero.min():7.2f} | "
                           f"Max: {non_zero.max():7.2f} | N: {len(non_zero)}\n")
                else:
                    f.write(f"  {dataset:12s} | No non-zero values\n")
            f.write("\n")

    print(f"OK Detailed statistics saved as {stats_file}")

    return output_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze dataset complexity and create visualizations."
    )

    parser.add_argument(
        "--cruxeval-file",
        default="data/CRUXEval/cruxeval.jsonl",
        help="Path to the CRUXEval JSONL file."
    )

    parser.add_argument(
        "--humaneval-file",
        default="data/HumanEval/HumanEval.jsonl",
        help="Path to the HumanEval JSONL file."
    )

    parser.add_argument(
        "--pythonsaga-file",
        default="data/PythonSaga/basic185.jsonl",
        help="Path to the PythonSaga JSONL file."
    )

    parser.add_argument(
        "--output-dir",
        default="dataset_properties",
        help="Directory where the plot and statistics file will be saved."
    )

    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()

    print("=" * 60)
    print("DATASET COMPLEXITY ANALYSIS")
    print("=" * 60)


    print_choice = input("\nPrint metrics for each program? (y/n): ").strip().lower()
    print_metrics = print_choice == 'y'

    print("\nLoading and analyzing datasets...")

    datasets = [
        ("CRUXEval", args.cruxeval_file),
        ("HumanEval", args.humaneval_file),
        ("PythonSaga", args.pythonsaga_file)
    ]

    analyzer = CodeAnalyzer(print_metrics=print_metrics)
    all_results = []

    for dataset_name, filepath in datasets:
        print(f"\nProcessing {dataset_name}...")

        data = load_dataset(filepath, dataset_name)
        if not data:
            print(f"  No data loaded from {filepath}")
            continue

        print(f"  Loaded {len(data)} samples")

        if print_metrics:
            print(f"  Metrics for each program:")


        analyzed_count = 0
        sample_deps = []

        for idx, item in enumerate(data):
            try:
                metrics = analyzer.analyze_code(item['code'], item['id'])

                result = {
                    'Dataset': item['dataset'],
                    'CC': metrics['CC'],
                    'LOC': metrics['LOC'],
                    'DEP': metrics['DEP'],
                    'NC': metrics['NC'],
                    'LL': metrics['LL']
                }
                all_results.append(result)
                analyzed_count += 1

                if metrics['DEP'] > 0:
                    sample_deps.append((item['id'], metrics['DEP']))

            except Exception:
                continue

        print(f"  Analyzed {analyzed_count} samples")


        if sample_deps and not print_metrics:
            print(f"  Sample programs with DEP > 0 (first 5):")
            for sample_id, dep_value in sample_deps[:5]:
                print(f"    {sample_id}: DEP = {dep_value}")

    if not all_results:
        print("\nNO Error: No data was successfully analyzed!")
        return


    df = pd.DataFrame(all_results)

    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples analyzed: {len(df)}")
    print("\nDataset distribution:")
    print(df['Dataset'].value_counts())


    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    create_violin_plots(df, args.output_dir)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
