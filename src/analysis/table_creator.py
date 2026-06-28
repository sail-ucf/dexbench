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
    col_widths[0] = 0.18
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

def create_main_study_tables(input_csv, output_dir):
    """Create tables for Main Study (4 LLMs)"""
    print("Creating Main Study tables...")


    detailed_df = pd.read_csv(input_csv)


    detailed_df = detailed_df[~detailed_df['model'].isin(['AI21-Jamba-Reasoning-3B', 'Llama-3.1-Nemotron-Nano-8B-v1'])]


    models = detailed_df['model'].unique()


    pass1_data = []
    for model in models:
        model_df = detailed_df[detailed_df['model'] == model]


        overall_strict_forward = model_df['pass1_strict_forward'].mean()
        overall_strict_backward = model_df['pass1_strict_backward'].mean()
        overall_strict_overall = model_df['pass1_strict_overall'].mean()
        overall_relaxed_forward = model_df['pass1_relaxed_forward'].mean()
        overall_relaxed_backward = model_df['pass1_relaxed_backward'].mean()
        overall_relaxed_overall = model_df['pass1_relaxed_overall'].mean()
        overall_n = len(model_df)


        pass1_data.append({
            'LLM': model,
            'Dataset': 'Overall',
            'Strict Forward': overall_strict_forward,
            'Strict Backward': overall_strict_backward,
            'Strict Overall': overall_strict_overall,
            'Relaxed Forward': overall_relaxed_forward,
            'Relaxed Backward': overall_relaxed_backward,
            'Relaxed Overall': overall_relaxed_overall,
            'N': overall_n
        })


        for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:

            dataset_df = model_df[model_df['task_id'].str.contains(dataset)]

            if len(dataset_df) > 0:
                pass1_data.append({
                    'LLM': model,
                    'Dataset': dataset,
                    'Strict Forward': dataset_df['pass1_strict_forward'].mean(),
                    'Strict Backward': dataset_df['pass1_strict_backward'].mean(),
                    'Strict Overall': dataset_df['pass1_strict_overall'].mean(),
                    'Relaxed Forward': dataset_df['pass1_relaxed_forward'].mean(),
                    'Relaxed Backward': dataset_df['pass1_relaxed_backward'].mean(),
                    'Relaxed Overall': dataset_df['pass1_relaxed_overall'].mean(),
                    'N': len(dataset_df)
                })

    pass1_df = pd.DataFrame(pass1_data)


    pass5_data = []
    for model in models:
        model_df = detailed_df[detailed_df['model'] == model]


        overall_strict_forward = model_df['pass5_strict_forward'].mean()
        overall_strict_backward = model_df['pass5_strict_backward'].mean()
        overall_strict_overall = model_df['pass5_strict_overall'].mean()
        overall_relaxed_forward = model_df['pass5_relaxed_forward'].mean()
        overall_relaxed_backward = model_df['pass5_relaxed_backward'].mean()
        overall_relaxed_overall = model_df['pass5_relaxed_overall'].mean()
        overall_n = len(model_df)


        pass5_data.append({
            'LLM': model,
            'Dataset': 'Overall',
            'Strict Forward': overall_strict_forward,
            'Strict Backward': overall_strict_backward,
            'Strict Overall': overall_strict_overall,
            'Relaxed Forward': overall_relaxed_forward,
            'Relaxed Backward': overall_relaxed_backward,
            'Relaxed Overall': overall_relaxed_overall,
            'N': overall_n
        })


        for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
            dataset_df = model_df[model_df['task_id'].str.contains(dataset)]

            if len(dataset_df) > 0:
                pass5_data.append({
                    'LLM': model,
                    'Dataset': dataset,
                    'Strict Forward': dataset_df['pass5_strict_forward'].mean(),
                    'Strict Backward': dataset_df['pass5_strict_backward'].mean(),
                    'Strict Overall': dataset_df['pass5_strict_overall'].mean(),
                    'Relaxed Forward': dataset_df['pass5_relaxed_forward'].mean(),
                    'Relaxed Backward': dataset_df['pass5_relaxed_backward'].mean(),
                    'Relaxed Overall': dataset_df['pass5_relaxed_overall'].mean(),
                    'N': len(dataset_df)
                })

    pass5_df = pd.DataFrame(pass5_data)

    output_dir.mkdir(parents=True, exist_ok=True)


    create_table_visualization(
        pass1_df,
        "MAIN STUDY: Pass@1 Results\n4 LLMs with Input Mutation\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "main_study_pass1_table.png"
    )


    create_table_visualization(
        pass5_df,
        "MAIN STUDY: Pass@5 Results\n4 LLMs with Input Mutation\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "main_study_pass5_table.png"
    )

def create_ablation_study_tables(input_csv, output_dir):
    """Create tables for Ablation Study (GPT-5-mini only)"""
    print("Creating Ablation Study tables...")


    ablation_df = pd.read_csv(input_csv)


    pass1_data = []


    overall_n = len(ablation_df)
    pass1_data.append({
        'LLM': 'gpt-5-mini',
        'Dataset': 'Overall',
        'Strict Forward': ablation_df['ablation_pass1_strict_forward'].mean(),
        'Strict Backward': ablation_df['ablation_pass1_strict_backward'].mean(),
        'Strict Overall': ablation_df['ablation_pass1_strict_overall'].mean(),
        'Relaxed Forward': ablation_df['ablation_pass1_relaxed_forward'].mean(),
        'Relaxed Backward': ablation_df['ablation_pass1_relaxed_backward'].mean(),
        'Relaxed Overall': ablation_df['ablation_pass1_relaxed_overall'].mean(),
        'N': overall_n
    })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_df = ablation_df[ablation_df['dataset'] == dataset]

        if len(dataset_df) > 0:
            pass1_data.append({
                'LLM': 'gpt-5-mini',
                'Dataset': dataset,
                'Strict Forward': dataset_df['ablation_pass1_strict_forward'].mean(),
                'Strict Backward': dataset_df['ablation_pass1_strict_backward'].mean(),
                'Strict Overall': dataset_df['ablation_pass1_strict_overall'].mean(),
                'Relaxed Forward': dataset_df['ablation_pass1_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_df['ablation_pass1_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_df['ablation_pass1_relaxed_overall'].mean(),
                'N': len(dataset_df)
            })

    pass1_df = pd.DataFrame(pass1_data)


    pass5_data = []


    pass5_data.append({
        'LLM': 'gpt-5-mini',
        'Dataset': 'Overall',
        'Strict Forward': ablation_df['ablation_pass5_strict_forward'].mean(),
        'Strict Backward': ablation_df['ablation_pass5_strict_backward'].mean(),
        'Strict Overall': ablation_df['ablation_pass5_strict_overall'].mean(),
        'Relaxed Forward': ablation_df['ablation_pass5_relaxed_forward'].mean(),
        'Relaxed Backward': ablation_df['ablation_pass5_relaxed_backward'].mean(),
        'Relaxed Overall': ablation_df['ablation_pass5_relaxed_overall'].mean(),
        'N': overall_n
    })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_df = ablation_df[ablation_df['dataset'] == dataset]

        if len(dataset_df) > 0:
            pass5_data.append({
                'LLM': 'gpt-5-mini',
                'Dataset': dataset,
                'Strict Forward': dataset_df['ablation_pass5_strict_forward'].mean(),
                'Strict Backward': dataset_df['ablation_pass5_strict_backward'].mean(),
                'Strict Overall': dataset_df['ablation_pass5_strict_overall'].mean(),
                'Relaxed Forward': dataset_df['ablation_pass5_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_df['ablation_pass5_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_df['ablation_pass5_relaxed_overall'].mean(),
                'N': len(dataset_df)
            })

    pass5_df = pd.DataFrame(pass5_data)

    output_dir.mkdir(parents=True, exist_ok=True)


    create_table_visualization(
        pass1_df,
        "ABLATION STUDY: Pass@1 Results\nGPT-5-mini with Input Generation\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "ablation_study_pass1_table.png"
    )


    create_table_visualization(
        pass5_df,
        "ABLATION STUDY: Pass@5 Results\nGPT-5-mini with Input Generation\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "ablation_study_pass5_table.png"
    )

def create_rap_study_tables(input_csv, output_dir):
    """Create tables for RAP Study (GPT-5-mini only)"""
    print("Creating RAP Study tables...")


    rap_df = pd.read_csv(input_csv)


    pass1_data = []


    overall_n = len(rap_df)
    pass1_data.append({
        'LLM': 'gpt-5-mini',
        'Dataset': 'Overall',
        'Strict Forward': rap_df['pass1_strict_forward'].mean(),
        'Strict Backward': rap_df['pass1_strict_backward'].mean(),
        'Strict Overall': rap_df['pass1_strict_overall'].mean(),
        'Relaxed Forward': rap_df['pass1_relaxed_forward'].mean(),
        'Relaxed Backward': rap_df['pass1_relaxed_backward'].mean(),
        'Relaxed Overall': rap_df['pass1_relaxed_overall'].mean(),
        'N': overall_n
    })


    datasets_count = {'CRUXEval': 0, 'HumanEval': 0, 'PythonSaga': 0}
    datasets_data = {'CRUXEval': [], 'HumanEval': [], 'PythonSaga': []}

    for _, row in rap_df.iterrows():
        task_id = row['task_id']
        if 'CRUXEval' in task_id:
            datasets_count['CRUXEval'] += 1
            datasets_data['CRUXEval'].append(row)
        elif 'HumanEval' in task_id:
            datasets_count['HumanEval'] += 1
            datasets_data['HumanEval'].append(row)
        elif 'PythonSaga' in task_id:
            datasets_count['PythonSaga'] += 1
            datasets_data['PythonSaga'].append(row)

    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        if datasets_count[dataset] > 0:
            dataset_rows = datasets_data[dataset]
            df_temp = pd.DataFrame(dataset_rows)

            pass1_data.append({
                'LLM': 'gpt-5-mini',
                'Dataset': dataset,
                'Strict Forward': df_temp['pass1_strict_forward'].mean(),
                'Strict Backward': df_temp['pass1_strict_backward'].mean(),
                'Strict Overall': df_temp['pass1_strict_overall'].mean(),
                'Relaxed Forward': df_temp['pass1_relaxed_forward'].mean(),
                'Relaxed Backward': df_temp['pass1_relaxed_backward'].mean(),
                'Relaxed Overall': df_temp['pass1_relaxed_overall'].mean(),
                'N': datasets_count[dataset]
            })

    pass1_df = pd.DataFrame(pass1_data)


    pass5_data = []


    pass5_data.append({
        'LLM': 'gpt-5-mini',
        'Dataset': 'Overall',
        'Strict Forward': rap_df['pass5_strict_forward'].mean(),
        'Strict Backward': rap_df['pass5_strict_backward'].mean(),
        'Strict Overall': rap_df['pass5_strict_overall'].mean(),
        'Relaxed Forward': rap_df['pass5_relaxed_forward'].mean(),
        'Relaxed Backward': rap_df['pass5_relaxed_backward'].mean(),
        'Relaxed Overall': rap_df['pass5_relaxed_overall'].mean(),
        'N': overall_n
    })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        if datasets_count[dataset] > 0:
            dataset_rows = datasets_data[dataset]
            df_temp = pd.DataFrame(dataset_rows)

            pass5_data.append({
                'LLM': 'gpt-5-mini',
                'Dataset': dataset,
                'Strict Forward': df_temp['pass5_strict_forward'].mean(),
                'Strict Backward': df_temp['pass5_strict_backward'].mean(),
                'Strict Overall': df_temp['pass5_strict_overall'].mean(),
                'Relaxed Forward': df_temp['pass5_relaxed_forward'].mean(),
                'Relaxed Backward': df_temp['pass5_relaxed_backward'].mean(),
                'Relaxed Overall': df_temp['pass5_relaxed_overall'].mean(),
                'N': datasets_count[dataset]
            })

    pass5_df = pd.DataFrame(pass5_data)

    output_dir.mkdir(parents=True, exist_ok=True)

    create_table_visualization(
        pass1_df,
        "RAP STUDY: Pass@1 Results\nGPT-5-mini with Random Alternative Paths\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "rap_study_pass1_table.png"
    )


    create_table_visualization(
        pass5_df,
        "RAP STUDY: Pass@5 Results\nGPT-5-mini with Random Alternative Paths\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "rap_study_pass5_table.png"
    )

def create_least_coverage_study_tables(input_csv, output_dir):
    """Create tables for Least Coverage Study (GPT-5-mini only)"""
    print("Creating Least Coverage Study tables...")


    lc_df = pd.read_csv(input_csv)


    pass1_lc_data = []


    overall_n = len(lc_df)
    pass1_lc_data.append({
        'LLM': 'gpt-5-mini',
        'Dataset': 'Overall',
        'Strict Forward': lc_df['lc_pass1_strict_forward'].mean(),
        'Strict Backward': lc_df['lc_pass1_strict_backward'].mean(),
        'Strict Overall': lc_df['lc_pass1_strict_overall'].mean(),
        'Relaxed Forward': lc_df['lc_pass1_relaxed_forward'].mean(),
        'Relaxed Backward': lc_df['lc_pass1_relaxed_backward'].mean(),
        'Relaxed Overall': lc_df['lc_pass1_relaxed_overall'].mean(),
        'N': overall_n
    })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_df = lc_df[lc_df['dataset'] == dataset]

        if len(dataset_df) > 0:
            pass1_lc_data.append({
                'LLM': 'gpt-5-mini',
                'Dataset': dataset,
                'Strict Forward': dataset_df['lc_pass1_strict_forward'].mean(),
                'Strict Backward': dataset_df['lc_pass1_strict_backward'].mean(),
                'Strict Overall': dataset_df['lc_pass1_strict_overall'].mean(),
                'Relaxed Forward': dataset_df['lc_pass1_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_df['lc_pass1_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_df['lc_pass1_relaxed_overall'].mean(),
                'N': len(dataset_df)
            })

    pass1_lc_df = pd.DataFrame(pass1_lc_data)


    pass5_lc_data = []


    pass5_lc_data.append({
        'LLM': 'gpt-5-mini',
        'Dataset': 'Overall',
        'Strict Forward': lc_df['lc_pass5_strict_forward'].mean(),
        'Strict Backward': lc_df['lc_pass5_strict_backward'].mean(),
        'Strict Overall': lc_df['lc_pass5_strict_overall'].mean(),
        'Relaxed Forward': lc_df['lc_pass5_relaxed_forward'].mean(),
        'Relaxed Backward': lc_df['lc_pass5_relaxed_backward'].mean(),
        'Relaxed Overall': lc_df['lc_pass5_relaxed_overall'].mean(),
        'N': overall_n
    })


    for dataset in ['CRUXEval', 'HumanEval', 'PythonSaga']:
        dataset_df = lc_df[lc_df['dataset'] == dataset]

        if len(dataset_df) > 0:
            pass5_lc_data.append({
                'LLM': 'gpt-5-mini',
                'Dataset': dataset,
                'Strict Forward': dataset_df['lc_pass5_strict_forward'].mean(),
                'Strict Backward': dataset_df['lc_pass5_strict_backward'].mean(),
                'Strict Overall': dataset_df['lc_pass5_strict_overall'].mean(),
                'Relaxed Forward': dataset_df['lc_pass5_relaxed_forward'].mean(),
                'Relaxed Backward': dataset_df['lc_pass5_relaxed_backward'].mean(),
                'Relaxed Overall': dataset_df['lc_pass5_relaxed_overall'].mean(),
                'N': len(dataset_df)
            })

    pass5_lc_df = pd.DataFrame(pass5_lc_data)

    output_dir.mkdir(parents=True, exist_ok=True)


    create_table_visualization(
        pass1_lc_df,
        "LEAST COVERAGE STUDY: Pass@1 Results\nGPT-5-mini with LeastCoverage Priority Lines\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "least_coverage_study_pass1_table.png"
    )


    create_table_visualization(
        pass5_lc_df,
        "LEAST COVERAGE STUDY: Pass@5 Results\nGPT-5-mini with LeastCoverage Priority Lines\nLLM -> Dataset -> Metrics -> Count",
        output_dir / "least_coverage_study_pass5_table.png"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create table visualizations for all DEXBENCH studies."
    )

    parser.add_argument(
        "--main-results-csv",
        type=Path,
        default=Path("evaluation_reports/detailed_results.csv"),
        help="Path to the main-study detailed results CSV."
    )

    parser.add_argument(
        "--ablation-results-csv",
        type=Path,
        default=Path(
            "ablation_evaluation_reports/ablation_detailed_results.csv"
        ),
        help="Path to the ablation-study detailed results CSV."
    )

    parser.add_argument(
        "--rap-results-csv",
        type=Path,
        default=Path("rap_evaluation_reports/rap_detailed_results.csv"),
        help="Path to the RAP-study detailed results CSV."
    )

    parser.add_argument(
        "--least-coverage-results-csv",
        type=Path,
        default=Path(
            "evaluation_reports_least_coverage/"
            "least_coverage_detailed_results.csv"
        ),
        help="Path to the least-coverage detailed results CSV."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Directory where generated PNG tables will be saved."
    )

    return parser.parse_args()

def main():
    """Main execution function"""
    
    args = parse_args()
    
    print("="*80)
    print("CREATING TABLE VISUALIZATIONS FOR ALL 4 STUDIES")
    print("="*80)


    tables_dir = args.output_dir
    tables_dir.mkdir(parents=True, exist_ok=True)


    create_main_study_tables(
        args.main_results_csv,
        args.output_dir
    )

    create_ablation_study_tables(
        args.ablation_results_csv,
        args.output_dir
    )

    create_rap_study_tables(
        args.rap_results_csv,
        args.output_dir
    )

    create_least_coverage_study_tables(
        args.least_coverage_results_csv,
        args.output_dir
    )
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n Tables saved in: {tables_dir}/")
    print("\nGenerated files:")
    print("MAIN STUDY:")
    print("  - main_study_pass1_table.png - Pass@1 results for 4 LLMs")
    print("  - main_study_pass5_table.png - Pass@5 results for 4 LLMs")
    print("\nABLATION STUDY:")
    print("  - ablation_study_pass1_table.png - Pass@1 results for GPT-5-mini")
    print("  - ablation_study_pass5_table.png - Pass@5 results for GPT-5-mini")
    print("\n RAP STUDY:")
    print("  - rap_study_pass1_table.png - Pass@1 results for GPT-5-mini")
    print("  - rap_study_pass5_table.png - Pass@5 results for GPT-5-mini")
    print("\nLEAST COVERAGE STUDY:")
    print("  - least_coverage_study_pass1_table.png - Pass@1 results for GPT-5-mini")
    print("  - least_coverage_study_pass5_table.png - Pass@5 results for GPT-5-mini")
    print("\n Table format for ALL files:")
    print("  Column 1: LLM (gpt-5-mini, etc.)")
    print("  Column 2: Dataset (Overall, CRUXEval, HumanEval, PythonSaga)")
    print("  Columns 3-8: Strict/Relaxed Forward/Backward/Overall metrics")
    print("  Column 9: N (count of programs in that dataset)")

if __name__ == "__main__":
    main()