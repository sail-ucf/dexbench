import pandas as pd
import os
import argparse

def analyze_closed_source_models(csv_path):
    """Analyze closed-source models from the main CSV file"""

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)


    target_models = ['gpt-5-mini', 'grok-4-fast-reasoning', 'claude-sonnet-4-sonnet', 'gemini-2.5-flash']
    df_filtered = df[df['model'].isin(target_models)].copy()

    results = {}

    for model in target_models:
        model_df = df_filtered[df_filtered['model'] == model]


        backward_only = len(model_df[
            (model_df['pass1_strict_backward'] == 1.0) &
            (model_df['pass1_strict_forward'] == 0.0)
        ])

        forward_only = len(model_df[
            (model_df['pass1_strict_forward'] == 1.0) &
            (model_df['pass1_strict_backward'] == 0.0)
        ])

        both_correct = len(model_df[
            (model_df['pass1_strict_forward'] == 1.0) &
            (model_df['pass1_strict_backward'] == 1.0)
        ])

        both_wrong = len(model_df[
            (model_df['pass1_strict_forward'] == 0.0) &
            (model_df['pass1_strict_backward'] == 0.0)
        ])

        total_tasks = len(model_df)


        calculated_total = backward_only + forward_only + both_correct + both_wrong
        if calculated_total != total_tasks:
            print(f"Warning: Count mismatch for {model}. Total: {total_tasks}, Sum: {calculated_total}")

        results[model] = {
            'backward_only': backward_only,
            'forward_only': forward_only,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'total_tasks': total_tasks
        }

    return results

def analyze_open_source_models(csv_path):
    """Analyze open-source models from the open-source CSV file"""

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    results = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]


        backward_only = len(model_df[
            (model_df['pass1_strict_backward'] == 1.0) &
            (model_df['pass1_strict_forward'] == 0.0)
        ])

        forward_only = len(model_df[
            (model_df['pass1_strict_forward'] == 1.0) &
            (model_df['pass1_strict_backward'] == 0.0)
        ])

        both_correct = len(model_df[
            (model_df['pass1_strict_forward'] == 1.0) &
            (model_df['pass1_strict_backward'] == 1.0)
        ])

        both_wrong = len(model_df[
            (model_df['pass1_strict_forward'] == 0.0) &
            (model_df['pass1_strict_backward'] == 0.0)
        ])

        total_tasks = len(model_df)


        calculated_total = backward_only + forward_only + both_correct + both_wrong
        if calculated_total != total_tasks:
            print(f"Warning: Count mismatch for {model}. Total: {total_tasks}, Sum: {calculated_total}")

        results[model] = {
            'backward_only': backward_only,
            'forward_only': forward_only,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'total_tasks': total_tasks
        }

    return results

def print_results(closed_source_results, open_source_results):
    """Print comprehensive results"""

    print("\n" + "=" * 90)
    print("CLOSED-SOURCE MODELS")
    print("=" * 90)
    print(f"{'Model':<25} {'Back Only':<12} {'Fwd Only':<12} {'Both OK':<12} {'Both NO':<12} {'Total':<12}")
    print("-" * 90)

    if closed_source_results:
        for model, stats in closed_source_results.items():
            print(f"{model:<25} "
                  f"{stats['backward_only']:<12} "
                  f"{stats['forward_only']:<12} "
                  f"{stats['both_correct']:<12} "
                  f"{stats['both_wrong']:<12} "
                  f"{stats['total_tasks']:<12}")

    print("\n" + "=" * 90)
    print("OPEN-SOURCE MODELS")
    print("=" * 90)
    print(f"{'Model':<40} {'Back Only':<12} {'Fwd Only':<12} {'Both OK':<12} {'Both NO':<12} {'Total':<12}")
    print("-" * 90)

    if open_source_results:
        for model, stats in open_source_results.items():

            if model.startswith('models_non-reasoning_'):
                display_name = model.replace('models_non-reasoning_', '')
            elif model.startswith('models_reasoning_'):
                display_name = model.replace('models_reasoning_', '')
            else:
                display_name = model

            print(f"{display_name:<40} "
                  f"{stats['backward_only']:<12} "
                  f"{stats['forward_only']:<12} "
                  f"{stats['both_correct']:<12} "
                  f"{stats['both_wrong']:<12} "
                  f"{stats['total_tasks']:<12}")

    print("\n" + "=" * 90)
    print("SUMMARY TOTALS")
    print("=" * 90)


    all_models = {}
    if closed_source_results:
        all_models.update(closed_source_results)
    if open_source_results:
        all_models.update(open_source_results)


    total_backward_only = sum(stats['backward_only'] for stats in all_models.values())
    total_forward_only = sum(stats['forward_only'] for stats in all_models.values())
    total_both_correct = sum(stats['both_correct'] for stats in all_models.values())
    total_both_wrong = sum(stats['both_wrong'] for stats in all_models.values())
    total_tasks_all = sum(stats['total_tasks'] for stats in all_models.values())

    print(f"Total models analyzed: {len(all_models)}")
    print(f"Total tasks analyzed: {total_tasks_all}")
    print(f"\nCategory breakdown:")
    print(f"  Backward only (backward=1, forward=0): {total_backward_only}")
    print(f"  Forward only (forward=1, backward=0): {total_forward_only}")
    print(f"  Both correct (both=1): {total_both_correct}")
    print(f"  Both wrong (both=0): {total_both_wrong}")


    if total_tasks_all > 0:
        print(f"\nPercentage breakdown (of total tasks):")
        print(f"  Backward only: {total_backward_only/total_tasks_all*100:.1f}%")
        print(f"  Forward only: {total_forward_only/total_tasks_all*100:.1f}%")
        print(f"  Both correct: {total_both_correct/total_tasks_all*100:.1f}%")
        print(f"  Both wrong: {total_both_wrong/total_tasks_all*100:.1f}%")


    print("\n" + "=" * 90)
    print("TOP PERFORMERS BY CATEGORY")
    print("=" * 90)


    print("\nTOP 3 MODELS - BACKWARD ONLY:")
    sorted_backward = sorted(all_models.items(), key=lambda x: x[1]['backward_only'], reverse=True)[:3]
    for model, stats in sorted_backward:
        display_name = clean_model_name_for_display(model)
        print(f"  {display_name}: {stats['backward_only']} tasks")


    print("\nTOP 3 MODELS - FORWARD ONLY:")
    sorted_forward = sorted(all_models.items(), key=lambda x: x[1]['forward_only'], reverse=True)[:3]
    for model, stats in sorted_forward:
        display_name = clean_model_name_for_display(model)
        print(f"  {display_name}: {stats['forward_only']} tasks")


    print("\nTOP 3 MODELS - BOTH CORRECT:")
    sorted_both_correct = sorted(all_models.items(), key=lambda x: x[1]['both_correct'], reverse=True)[:3]
    for model, stats in sorted_both_correct:
        display_name = clean_model_name_for_display(model)
        print(f"  {display_name}: {stats['both_correct']} tasks")


    print("\nTOP 3 MODELS - BOTH WRONG:")
    sorted_both_wrong = sorted(all_models.items(), key=lambda x: x[1]['both_wrong'], reverse=True)[:3]
    for model, stats in sorted_both_wrong:
        display_name = clean_model_name_for_display(model)
        print(f"  {display_name}: {stats['both_wrong']} tasks")


    print("\nTOP 3 MODELS - HIGHEST AGREEMENT (Both Correct + Both Wrong):")
    sorted_agreement = sorted(all_models.items(),
                             key=lambda x: x[1]['both_correct'] + x[1]['both_wrong'],
                             reverse=True)[:3]
    for model, stats in sorted_agreement:
        display_name = clean_model_name_for_display(model)
        agreement_total = stats['both_correct'] + stats['both_wrong']
        print(f"  {display_name}: {agreement_total} tasks ({agreement_total/stats['total_tasks']*100:.1f}%)")

def clean_model_name_for_display(model_name):
    """Clean model name for display in summary"""
    if model_name.startswith('models_non-reasoning_'):
        return model_name.replace('models_non-reasoning_', '')
    elif model_name.startswith('models_reasoning_'):
        return model_name.replace('models_reasoning_', '')
    else:
        return model_name

def save_to_csv(closed_source_results, open_source_results, output_csv):
    """Save results to CSV file for further analysis"""

    all_data = []


    if closed_source_results:
        for model, stats in closed_source_results.items():
            all_data.append({
                'model': model,
                'type': 'closed_source',
                'backward_only': stats['backward_only'],
                'forward_only': stats['forward_only'],
                'both_correct': stats['both_correct'],
                'both_wrong': stats['both_wrong'],
                'total_tasks': stats['total_tasks']
            })


    if open_source_results:
        for model, stats in open_source_results.items():
            all_data.append({
                'model': model,
                'type': 'open_source',
                'backward_only': stats['backward_only'],
                'forward_only': stats['forward_only'],
                'both_correct': stats['both_correct'],
                'both_wrong': stats['both_wrong'],
                'total_tasks': stats['total_tasks']
            })

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")

def parse_args():
    parser = argparse.ArgumentParser(
        description= "Analyze forward and backward model performance discrepancies."
    )

    parser.add_argument(
        "--closed-source-csv",
        default = "evaluation_reports/detailed_results.csv",
        help="Path to the closed-source evaluation results CSV."
    )

    parser.add_argument(
        "--open-source-csv",
        default="evaluation_results_open_source/fixed/evaluation_results_debug/debug_results.csv",
        help="Path to the open-source evaluation results CSV."
    )

    parser.add_argument(
        "--output-csv",
        default="model_discrepancy_analysis.csv",
        help="Path where the discrepancy analysis CSV will be saved."
    )

    return parser.parse_args()


def main():

    args = parse_args()
    
    print("Analyzing model performance discrepancies...")
    print("=" * 90)
    print("Analyzing four categories for each model:")
    print("1. Backward Only (backward=1, forward=0)")
    print("2. Forward Only (forward=1, backward=0)")
    print("3. Both Correct (both=1)")
    print("4. Both Wrong (both=0)")
    print("=" * 90)


    print("\nAnalyzing closed-source models...")
    closed_source_results = analyze_closed_source_models(args.closed_source_csv)


    print("Analyzing open-source models...")
    open_source_results = analyze_open_source_models(args.open_source_csv)


    print_results(closed_source_results, open_source_results)


    save_to_csv(closed_source_results, open_source_results, args.output_csv)

    print("\n" + "=" * 90)
    print("Analysis complete!")
    print("=" * 90)

if __name__ == "__main__":
    main()