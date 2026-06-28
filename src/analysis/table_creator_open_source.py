import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


plt.style.use('default')

def create_table_visualization(table_df, title, output_path):
    """Create a single table visualization"""
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis('tight')
    ax.axis('off')


    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)


    table_data = table_df.values
    col_labels = table_df.columns.tolist()


    formatted_data = []
    for row in table_data:
        formatted_row = []
        for i, val in enumerate(row):
            if isinstance(val, (int, np.integer)):
                formatted_row.append(str(int(val)))
            elif isinstance(val, float):
                formatted_row.append(f"{val:.3f}")
            else:
                formatted_row.append(str(val))
        formatted_data.append(formatted_row)


    col_widths = [0.15] * len(col_labels)
    col_widths[0] = 0.22
    col_widths[1] = 0.15


    table = ax.table(cellText=formatted_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)


    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)


    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4F81BD')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')


    for i in range(1, len(formatted_data) + 1):
        for j in range(len(col_labels)):
            table[(i, j)].set_edgecolor('lightgray')


            if formatted_data[i-1][1] == 'Overall':
                table[(i, j)].set_facecolor('#E6B8B7')

            elif formatted_data[i-1][1] == 'CRUXEval':
                table[(i, j)].set_facecolor('#FFF2CC')

            elif formatted_data[i-1][1] == 'HumanEval':
                table[(i, j)].set_facecolor('#DDEBF7')

            elif formatted_data[i-1][1] == 'PythonSaga':
                table[(i, j)].set_facecolor('#E2EFDA')

            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"OK Saved: {output_path}")

def clean_model_name(model_name):
    """Clean model name for display"""
    if model_name.startswith('models_non-reasoning_'):
        return model_name.replace('models_non-reasoning_', '')
    elif model_name.startswith('models_reasoning_'):
        return model_name.replace('models_reasoning_', '')
    else:
        return model_name

def create_open_source_tables(input_csv, output_dir):
    """Create tables for all open-source models"""
    print("Creating Open Source models tables...")

    detailed_df = pd.read_csv(input_csv)

    print(f"Total rows in open-source CSV: {len(detailed_df)}")
    print(f"Unique models: {len(detailed_df['model'].unique())}")


    models = detailed_df['model'].unique()


    output_dir.mkdir(parents=True, exist_ok=True)


    for model in models:
        print(f"\nProcessing model: {model}")


        model_df = detailed_df[detailed_df['model'] == model]


        pass1_data = []


        overall_strict_forward = model_df['pass1_strict_forward'].mean()
        overall_strict_backward = model_df['pass1_strict_backward'].mean()
        overall_strict_overall = model_df['pass1_strict_overall'].mean()
        overall_n = len(model_df)


        pass1_data.append({
            'Model': clean_model_name(model),
            'Dataset': 'Overall',
            'Strict Forward': overall_strict_forward,
            'Strict Backward': overall_strict_backward,
            'Strict Overall': overall_strict_overall,
            'N': overall_n
        })


        for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:

            if 'task_id' in model_df.columns:
                dataset_df = model_df[model_df['task_id'].str.contains(dataset)]
            elif 'dataset' in model_df.columns:
                dataset_df = model_df[model_df['dataset'] == dataset]
            else:

                dataset_df = model_df[model_df['task_id'].str.contains(dataset)]

            if len(dataset_df) > 0:
                pass1_data.append({
                    'Model': clean_model_name(model),
                    'Dataset': dataset,
                    'Strict Forward': dataset_df['pass1_strict_forward'].mean(),
                    'Strict Backward': dataset_df['pass1_strict_backward'].mean(),
                    'Strict Overall': dataset_df['pass1_strict_overall'].mean(),
                    'N': len(dataset_df)
                })

        pass1_df = pd.DataFrame(pass1_data)


        pass5_data = []


        overall_strict_forward_5 = model_df['pass5_strict_forward'].mean()
        overall_strict_backward_5 = model_df['pass5_strict_backward'].mean()
        overall_strict_overall_5 = model_df['pass5_strict_overall'].mean()


        pass5_data.append({
            'Model': clean_model_name(model),
            'Dataset': 'Overall',
            'Strict Forward': overall_strict_forward_5,
            'Strict Backward': overall_strict_backward_5,
            'Strict Overall': overall_strict_overall_5,
            'N': overall_n
        })


        for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:

            if 'task_id' in model_df.columns:
                dataset_df = model_df[model_df['task_id'].str.contains(dataset)]
            elif 'dataset' in model_df.columns:
                dataset_df = model_df[model_df['dataset'] == dataset]
            else:

                dataset_df = model_df[model_df['task_id'].str.contains(dataset)]

            if len(dataset_df) > 0:
                pass5_data.append({
                    'Model': clean_model_name(model),
                    'Dataset': dataset,
                    'Strict Forward': dataset_df['pass5_strict_forward'].mean(),
                    'Strict Backward': dataset_df['pass5_strict_backward'].mean(),
                    'Strict Overall': dataset_df['pass5_strict_overall'].mean(),
                    'N': len(dataset_df)
                })

        pass5_df = pd.DataFrame(pass5_data)


        safe_model_name = clean_model_name(model).replace(' ', '_').replace('/', '_').replace('.', '_')


        create_table_visualization(
            pass1_df,
            f"Open Source: {clean_model_name(model)}\nPass@1 Results\nModel -> Dataset -> Metrics -> Count",
            output_dir / f"{safe_model_name}_pass1_table.png"
        )


        create_table_visualization(
            pass5_df,
            f"Open Source: {clean_model_name(model)}\nPass@5 Results\nModel -> Dataset -> Metrics -> Count",
            output_dir / f"{safe_model_name}_pass5_table.png"
        )


    print("\nCreating combined summary table for all models...")

    summary_pass1_data = []
    summary_pass5_data = []

    for model in models:
        model_df = detailed_df[detailed_df['model'] == model]


        summary_pass1_data.append({
            'Model': clean_model_name(model),
            'Total Tasks': len(model_df),
            'Pass1 Strict Forward': model_df['pass1_strict_forward'].mean(),
            'Pass1 Strict Backward': model_df['pass1_strict_backward'].mean(),
            'Pass1 Strict Overall': model_df['pass1_strict_overall'].mean(),
            'Pass1 Forward': model_df['pass1_forward'].mean(),
            'Pass1 Backward': model_df['pass1_backward'].mean(),
        })


        summary_pass5_data.append({
            'Model': clean_model_name(model),
            'Total Tasks': len(model_df),
            'Pass5 Strict Forward': model_df['pass5_strict_forward'].mean(),
            'Pass5 Strict Backward': model_df['pass5_strict_backward'].mean(),
            'Pass5 Strict Overall': model_df['pass5_strict_overall'].mean(),
            'Pass5 Forward': model_df['pass5_forward'].mean(),
            'Pass5 Backward': model_df['pass5_backward'].mean(),
        })

    summary_pass1_df = pd.DataFrame(summary_pass1_data)
    summary_pass5_df = pd.DataFrame(summary_pass5_data)


    create_table_visualization(
        summary_pass1_df,
        "Open Source Models - All Models Summary\nPass@1 Results\nSorted by Model",
        output_dir / "all_models_pass1_summary.png"
    )

    create_table_visualization(
        summary_pass5_df,
        "Open Source Models - All Models Summary\nPass@5 Results\nSorted by Model",
        output_dir / "all_models_pass5_summary.png"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create table visualizations for open-source model results."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "evaluation_results_open_source/fixed/"
            "evaluation_results_debug/debug_results.csv"
        ),
        help="Path to the open-source evaluation results CSV."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visuals_open_source"),
        help="Directory where the generated PNG tables will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution function"""

    args = parse_args()

    print("="*80)
    print("CREATING TABLE VISUALIZATIONS FOR OPEN SOURCE MODELS")
    print("="*80)


    create_open_source_tables(args.input_csv, args.output_dir)

    print("\n" + "="*80)
    print("ALL OPEN SOURCE VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)

    output_dir = args.output_dir
    print(f"\n Tables saved in: {output_dir}/")


    png_files = list(output_dir.glob("*.png"))
    print(f"\nGenerated {len(png_files)} files:")


    models_dict = {}
    for file in png_files:
        model_name = file.stem.replace('_pass1_table', '').replace('_pass5_table', '').replace('_summary', '')
        if model_name not in models_dict:
            models_dict[model_name] = []
        models_dict[model_name].append(file.name)


    for model_name, files in sorted(models_dict.items()):
        print(f"\n{model_name}:")
        for file in sorted(files):
            print(f"  - {file}")

    print("\n Table format for ALL files:")
    print("  Column 1: Model name")
    print("  Column 2: Dataset (Overall, CRUXEval, HumanEval, PythonSaga)")
    print("  Columns 3-5: Strict Forward/Backward/Overall metrics")
    print("  Column 6: N (count of programs in that dataset)")

if __name__ == "__main__":
    main()