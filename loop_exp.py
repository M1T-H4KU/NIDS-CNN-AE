import subprocess
import argparse
import os
import json
import pandas as pd
from datetime import datetime

def analyze_results(output_dir, num_runs, classifier_mode):
    """
    Loads all run*.json files from the output directory, analyzes them,
    and prints the final summary.
    """
    print("\n--- Starting Analysis of All Runs ---")
    results = []
    
    for i in range(num_runs):
        json_filename = os.path.join(output_dir, f"run_{i:03d}.json")
        if not os.path.exists(json_filename):
            print(f"Warning: Result file not found: {json_filename}")
            continue
        try:
            with open(json_filename, 'r') as f:
                data = json.load(f)
                
                # 'Flatten' the JSON data for easier pandas import
                run_metrics = data.get('metrics', {})
                flat_data = {'run_id': data.get('run_id', i)}
                
                # Add overall metrics
                flat_data['accuracy'] = run_metrics.get('accuracy', 0)
                
                # Add macro/weighted averages
                for avg_type in ['macro avg', 'weighted avg']:
                    for metric in ['precision', 'recall', 'f1-score']:
                        key_name = f"{avg_type}_{metric}"
                        flat_data[key_name] = run_metrics.get(avg_type, {}).get(metric, 0)
                
                # Add per-class metrics
                if classifier_mode == 'binary':
                    class_names = ['Normal (Class 0)', 'Abnormal (Class 1)']
                else:
                    # Dynamically get class names from the report (excluding averages)
                    class_names = [k for k in run_metrics.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

                for class_name in class_names:
                    for metric in ['precision', 'recall', 'f1-score']:
                        key_name = f"{class_name}_{metric}"
                        flat_data[key_name] = run_metrics.get(class_name, {}).get(metric, 0)
                        
                results.append(flat_data)
        except Exception as e:
            print(f"Error reading {json_filename}: {e}")

    if not results:
        print("No valid result files found. Analysis cannot be performed.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)
    df.set_index('run_id', inplace=True)
    
    # 1. FIND THE BEST RUN (as per the paper's methodology)
    # We define "best" as the one with the highest 'macro avg_f1-score'
    # You can change this to 'accuracy' if you prefer.
    best_run_metric = 'macro avg_f1-score' if classifier_mode == 'multiclass' else 'accuracy'
    best_run = df.loc[df[best_run_metric].idxmax()]
    
    print("\n--- Best Run (Selected by max '{best_run_metric}') ---")
    print(best_run.to_string())

    # 2. SHOW OVERALL TREND (as you requested)
    print("\n--- Overall Trend (Statistics over all runs) ---")
    # Show only the most important metrics in the summary
    summary_cols = ['accuracy', 'macro avg_precision', 'macro avg_recall', 'macro avg_f1-score']
    # Filter out columns that don't exist (e.g., macro avg for binary mode if not present)
    summary_cols = [col for col in summary_cols if col in df.columns]
    
    print(df[summary_cols].describe().round(4))

def main():
    parser = argparse.ArgumentParser(
        description="Run N-Trial experiments for NSL-KDD.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Experiment control
    parser.add_argument('--num_runs', type=int, default=10, help="Number of independent runs to perform. (Default: 10, paper uses 100)")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save all experiment outputs (JSONs, graphs, etc.).")
    
    # Arguments to pass-through to main.py
    parser.add_argument('--use_outlier_removal', action='store_true', help="Enable MAD-based outlier removal.")
    parser.add_argument('--use_gan', action='store_true', help="Enable BEGAN for data augmentation.")
    parser.add_argument('--classifier_mode', type=str, required=True, choices=['binary', 'multiclass'], help="Set the classifier mode (REQUIRED).")
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'dnn', 'lstm', 'cnnae', 'dnnae', 'lstmae'], help="Select the model pipeline to use (REQUIRED).")

    args = parser.parse_args()

    # Create a unique output directory for this batch of experiments
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{args.model}_{args.classifier_mode}_{'gan' if args.use_gan else 'orig'}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting {args.num_runs} independent runs...")
    print(f"Results will be saved in: {output_dir}")

    # Build the base command for main.py
    base_command = [
        'python', 'main.py',
        '--classifier_mode', args.classifier_mode,
        '--model', args.model
    ]
    if args.use_outlier_removal:
        base_command.append('--use_outlier_removal')
    if args.use_gan:
        base_command.append('--use_gan')

    # --- Run the 100-run loop ---
    for i in range(args.num_runs):
        print(f"\n--- Starting Run {i+1}/{args.num_runs} ---")
        
        # Add the unique run_id and output_dir for this specific run
        run_command = base_command + [
            '--run_id', str(i),
            '--output_dir', output_dir
        ]
        
        # Call main.py as a subprocess
        # We use 'uv run' to ensure it uses the correct environment
        subprocess_command = ['uv', 'run'] + run_command
        
        try:
            # We don't capture output, letting main.py print its minimal log
            subprocess.run(subprocess_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Run {i} failed with exit code {e.returncode}. Skipping. !!!")
        except KeyboardInterrupt:
            print("--- Experiment loop interrupted by user. Proceeding to analysis... ---")
            break

    # --- Analyze the results ---
    analyze_results(output_dir, args.num_runs, args.classifier_mode)

if __name__ == '__main__':
    main()