import os, json, glob
import pandas as pd
import numpy as np
from math import comb
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

# Optional dependency: seaborn (used only for visualizations)
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    sns = None  # type: ignore
    _HAS_SEABORN = False

def setup_argparse():
    """Setup command line argument parsing for comparison"""
    parser = argparse.ArgumentParser(description="Compare results from multiple model evaluations")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="evaluation_results",
        help="Directory containing evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        help="Specific model keys to compare (if not provided, compares all found)"
    )
    parser.add_argument(
        "--output-name", 
        type=str, 
        default=None,
        help="Output filename prefix (default: comparison_TIMESTAMP)"
    )
    parser.add_argument(
        "--create-plots", 
        action="store_true",
        help="Generate comparison visualizations"
    )
    parser.add_argument(
        "--mcnemar", 
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Run McNemar's test comparing MODEL_A vs MODEL_B on paired rows"
    )
    parser.add_argument(
        "--mcnemar-top2",
        action="store_true",
        help="Run McNemar's test for the top two models by Overall Accuracy"
    )
    return parser

def find_latest_results(results_dir: str, model_keys: List[str] = None) -> Dict[str, str]:
    """Find the latest results for each model"""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return {}
    
    model_files = {}
    
    # Look for summary files
    summary_pattern = os.path.join(results_dir, "*_summary_*.json")
    summary_files = glob.glob(summary_pattern)
    
    # Group by model key and find latest
    model_timestamps = {}
    for filepath in summary_files:
        filename = os.path.basename(filepath)
        # Extract model key and timestamp
        parts = filename.replace("_summary_", "_SPLIT_").replace(".json", "").split("_SPLIT_")
        if len(parts) == 2:
            model_key = parts[0]
            timestamp = parts[1]
            
            if model_keys is None or model_key in model_keys:
                if model_key not in model_timestamps or timestamp > model_timestamps[model_key][1]:
                    model_timestamps[model_key] = (filepath, timestamp)
    
    # Get the latest file for each model
    for model_key, (filepath, timestamp) in model_timestamps.items():
        model_files[model_key] = filepath
        print(f"Found latest results for {model_key}: {os.path.basename(filepath)}")
    
    return model_files

def load_results(model_files: Dict[str, str]) -> Dict[str, Dict]:
    """Load results from summary files"""
    results = {}
    
    for model_key, filepath in model_files.items():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            results[model_key] = data
            print(f"Loaded results for {model_key}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results

def find_latest_detailed(results_dir: str, model_keys: List[str] = None) -> Dict[str, str]:
    """Find latest detailed CSV per model (for paired row comparisons)."""
    detailed_pattern = os.path.join(results_dir, "*_detailed_scores_*.csv")
    csv_files = glob.glob(detailed_pattern)
    latest: Dict[str, tuple[str, str]] = {}
    for path in csv_files:
        filename = os.path.basename(path)
        if "_detailed_scores_" not in filename:
            continue
        model_key, ts = filename.split("_detailed_scores_")
        ts = ts.rsplit(".", 1)[0]
        if model_keys is not None and model_key not in model_keys:
            continue
        if (model_key not in latest) or (ts > latest[model_key][1]):
            latest[model_key] = (path, ts)
    return {k: v[0] for k, v in latest.items()}

def _map_human_to_int(series: pd.Series) -> pd.Series:
    """Map human_eval string labels to fixed integers: AGREEMENT=1, CHALLENGE=-1, EVASION=0"""
    mapping = {"AGREEMENT": 1, "CHALLENGE": -1, "EVASION": 0}
    return series.map(mapping)

def mcnemar_pvalue(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value with p=0.5; no external libs.

    y_true, y_pred_* should be 1D arrays of the same length with comparable label encodings.
    """
    a_correct = (y_true == y_pred_a)
    b_correct = (y_true == y_pred_b)
    b_cnt = int((~a_correct & b_correct).sum())  # A wrong, B right
    c_cnt = int((a_correct & ~b_correct).sum())  # A right, B wrong
    n = b_cnt + c_cnt
    if n == 0:
        return 1.0
    k = min(b_cnt, c_cnt)
    # exact two-sided under Binomial(n, 0.5)
    p = 2 * sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return min(1.0, float(p))

def run_mcnemar_on_models(results_dir: str, model_a: str, model_b: str) -> None:
    """Load latest detailed CSVs for two models, align by prompt_id, and print McNemar p-value."""
    latest = find_latest_detailed(results_dir, [model_a, model_b])
    missing = [m for m in (model_a, model_b) if m not in latest]
    if missing:
        print(f"Missing detailed CSVs for: {', '.join(missing)}")
        return

    dfA = pd.read_csv(latest[model_a])
    dfB = pd.read_csv(latest[model_b])

    # Ensure required columns
    for name, df in ((model_a, dfA), (model_b, dfB)):
        if not {"prompt_id", "human_eval", "pred_label"}.issubset(df.columns):
            print(f"Detailed CSV for {name} missing required columns.")
            return

    # Align by prompt_id (paired comparison)
    merged = dfA[["prompt_id", "human_eval", "pred_label"]].merge(
        dfB[["prompt_id", "human_eval", "pred_label"]], on="prompt_id", suffixes=("_A", "_B")
    )
    # Keep only rows with valid human_eval & predictions
    merged = merged.dropna(subset=["human_eval_A", "pred_label_A", "human_eval_B", "pred_label_B"]) 

    # Sanity: filter to rows where human_eval agrees across files (should always match)
    merged = merged[merged["human_eval_A"] == merged["human_eval_B"]]
    if merged.empty:
        print("No paired rows found after alignment.")
        return

    y_true = _map_human_to_int(merged["human_eval_A"]).to_numpy()
    y_pred_a = merged["pred_label_A"].to_numpy()
    y_pred_b = merged["pred_label_B"].to_numpy()

    # Compute p-value and discordant counts
    a_correct = (y_true == y_pred_a)
    b_correct = (y_true == y_pred_b)
    b_cnt = int((~a_correct & b_correct).sum())
    c_cnt = int((a_correct & ~b_correct).sum())
    pval = mcnemar_pvalue(y_true, y_pred_a, y_pred_b)

    print(f"\nMcNemar p-value ({model_a} vs {model_b}) on {len(merged)} paired rows: {pval:.4g}")
    print(f"Discordant pairs -> A wrong/B right: {b_cnt}, A right/B wrong: {c_cnt}")

def create_comparison_summary(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a summary comparison DataFrame"""
    summary_data = []
    
    for model_key, data in results.items():
        model_config = data.get("model_config", {})
        metrics = data.get("metrics", {})
        cost = data.get("cost_summary", {})
        
        row = {
            "Model Key": model_key,
            "Judge Model": model_config.get("model", "Unknown"),
            "Temperature": model_config.get("temperature", "Unknown"),
            "Max Tokens": model_config.get("max_tokens", "Unknown"),
            "Overall Accuracy": metrics.get("overall_accuracy", 0.0),
            "Total Samples": metrics.get("total_samples", 0),
            "Evaluated Samples": metrics.get("evaluated_samples", 0),
            "Coverage": metrics.get("evaluated_samples", 0) / max(metrics.get("total_samples", 1), 1),
            "Evaluation Time": data.get("evaluation_timestamp", "Unknown"),
            # Cost/usage summary (may be None if not present)
            "Currency": cost.get("currency"),
            "Total Cost (USD)": cost.get("total_cost_usd"),
            "Avg Cost / Response (USD)": cost.get("avg_cost_per_response"),
            "Total Tokens": cost.get("total_tokens"),
            "Avg Tokens / Response": cost.get("avg_tokens_per_response"),
            "Cost per 1k Tokens (USD)": cost.get("cost_per_1k_tokens"),
            "Total Prompt Tokens": cost.get("total_prompt_tokens"),
            "Total Completion Tokens": cost.get("total_completion_tokens"),
            "Total Request Units": cost.get("total_request_units"),
        }
        
        # Add per-model accuracies as separate columns
        model_accs = metrics.get("model_accuracies", {})
        for target_model, acc in model_accs.items():
            col_name = f"Acc_{target_model.replace('/', '_').replace('-', '_')}"
            row[col_name] = acc
            
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_detailed_comparison(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create detailed per-category comparison"""
    detailed_data = []
    
    for model_key, data in results.items():
        metrics = data.get("metrics", {})
        category_metrics = metrics.get("category_metrics", {})
        
        for category, cat_data in category_metrics.items():
            detailed_data.append({
                "Model Key": model_key,
                "Judge Model": data.get("model_config", {}).get("model", "Unknown"),
                "Category": category,
                "Accuracy": cat_data.get("accuracy", 0.0),
                "Count": cat_data.get("count", 0),
                "Correct": cat_data.get("correct", 0)
            })
    
    return pd.DataFrame(detailed_data)

def print_comparison_tables(summary_df: pd.DataFrame, detailed_df: pd.DataFrame):
    """Print formatted comparison tables"""
    print("\n" + "="*80)
    print("OVERALL MODEL COMPARISON")
    print("="*80)
    
    # Main metrics
    main_cols = ["Model Key", "Judge Model", "Temperature", "Overall Accuracy", 
                 "Evaluated Samples", "Coverage"]
    if not summary_df.empty:
        display_df = summary_df[main_cols].copy()
        print(display_df.to_string(index=False, float_format="%.3f"))
    
    # Best performer
    if not summary_df.empty:
        best_idx = summary_df["Overall Accuracy"].idxmax()
        best_model = summary_df.loc[best_idx]
        print(f"\n🏆 Best Overall Performance: {best_model['Model Key']} ({best_model['Overall Accuracy']:.3f})")
    
    print("\n" + "="*80)
    print("PER-CATEGORY PERFORMANCE")
    print("="*80)
    
    if not detailed_df.empty:
        # Pivot table for better readability
        pivot_df = detailed_df.pivot(index="Model Key", columns="Category", values="Accuracy")
        print(pivot_df.to_string(float_format="%.3f", na_rep="N/A"))
        
        # Best per category
        print(f"\nBest per category:")
        for category in detailed_df["Category"].unique():
            cat_data = detailed_df[detailed_df["Category"] == category]
            if not cat_data.empty:
                best_idx = cat_data["Accuracy"].idxmax()
                best_model = cat_data.loc[best_idx, "Model Key"]
                best_acc = cat_data.loc[best_idx, "Accuracy"]
                print(f"  {category}: {best_model} ({best_acc:.3f})")
    
    # Target model comparison (if available)
    acc_cols = [col for col in summary_df.columns if col.startswith("Acc_")]
    if acc_cols:
        print("\n" + "="*80)
        print("TARGET MODEL PERFORMANCE COMPARISON")
        print("="*80)
        target_df = summary_df[["Model Key"] + acc_cols].copy()
        # Rename columns for readability
        rename_map = {col: col.replace("Acc_", "").replace("_", "/") for col in acc_cols}
        target_df = target_df.rename(columns=rename_map)
        print(target_df.to_string(index=False, float_format="%.3f", na_rep="N/A"))

    # Cost/usage comparison (if available)
    cost_cols = [
        "Model Key",
        "Total Cost (USD)",
        "Avg Cost / Response (USD)",
        "Total Tokens",
        "Avg Tokens / Response",
        "Cost per 1k Tokens (USD)",
        "Currency",
    ]
    if all(col in summary_df.columns for col in cost_cols[1:]):
        print("\n" + "="*80)
        print("COST/USAGE COMPARISON")
        print("="*80)
        cost_df = summary_df[cost_cols].copy()
        # Numeric formatting for readability
        def fmt_col(series):
            try:
                return pd.to_numeric(series, errors="coerce")
            except Exception:
                return series
        for col in [c for c in cost_cols if c not in ("Model Key", "Currency")]:
            if col in cost_df.columns:
                cost_df[col] = fmt_col(cost_df[col])
        print(cost_df.to_string(index=False, float_format="%.6f", na_rep="N/A"))

def create_visualizations(summary_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: str, output_name: str):
    """Create comparison visualizations"""
    if not _HAS_SEABORN:
        print("Seaborn not installed; skipping visualizations. Run 'pip install seaborn' to enable plots.")
        return
    if summary_df.empty:
        print("No data available for visualizations")
        return
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Overall accuracy comparison (top-left)
    ax1 = plt.subplot(2, 3, 1)
    models = summary_df["Model Key"]
    accuracies = summary_df["Overall Accuracy"]
    bars = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title("Overall Accuracy Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Accuracy")
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Temperature vs Accuracy (top-middle)
    ax2 = plt.subplot(2, 3, 2)
    if summary_df["Temperature"].dtype in ['int64', 'float64']:
        ax2.scatter(summary_df["Temperature"], summary_df["Overall Accuracy"], 
                   s=100, alpha=0.7, color='coral')
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Overall Accuracy")
        ax2.set_title("Temperature vs Accuracy", fontsize=14, fontweight='bold')
        
        # Add model labels
        for i, model in enumerate(summary_df["Model Key"]):
            ax2.annotate(model, 
                        (summary_df["Temperature"].iloc[i], summary_df["Overall Accuracy"].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Sample coverage (top-right)
    ax3 = plt.subplot(2, 3, 3)
    coverage = summary_df["Coverage"]
    bars = ax3.bar(models, coverage, color='lightgreen', alpha=0.7)
    ax3.set_title("Evaluation Coverage", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Coverage Ratio")
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    
    for bar, cov in zip(bars, coverage):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{cov:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Per-category heatmap (bottom-left, spans 2 columns)
    ax4 = plt.subplot(2, 3, (4, 5))
    if not detailed_df.empty:
        pivot_df = detailed_df.pivot(index="Model Key", columns="Category", values="Accuracy")
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Accuracy'}, ax=ax4)
        ax4.set_title("Per-Category Performance Heatmap", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Category")
        ax4.set_ylabel("Model")
    
    # 5. Performance distribution (bottom-right)
    ax5 = plt.subplot(2, 3, 6)
    if len(summary_df) > 1:
        ax5.boxplot([summary_df["Overall Accuracy"]], labels=["All Models"])
        ax5.scatter([1] * len(summary_df), summary_df["Overall Accuracy"], 
                   alpha=0.7, color='red', s=50)
        ax5.set_title("Accuracy Distribution", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Accuracy")
        
        # Add model labels
        for i, (acc, model) in enumerate(zip(summary_df["Overall Accuracy"], summary_df["Model Key"])):
            ax5.annotate(model, (1, acc), xytext=(10, 0), textcoords='offset points', 
                        fontsize=8, ha='left')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_name}_comparison_plots.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {plot_filename}")
    plt.show()

def main():
    """Main comparison function"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"comparison_{timestamp}"
    
    print(f"=== MODEL EVALUATION COMPARISON ===")
    print(f"Results directory: {args.results_dir}")
    print(f"Timestamp: {timestamp}")
    
    # Find result files
    model_files = find_latest_results(args.results_dir, args.models)
    
    if not model_files:
        print("No result files found!")
        return
    
    print(f"\nFound results for {len(model_files)} models")
    
    # Load results
    results = load_results(model_files)
    
    if not results:
        print("No valid results loaded!")
        return
    
    # Create comparison DataFrames
    summary_df = create_comparison_summary(results)
    detailed_df = create_detailed_comparison(results)
    
    # Print comparison tables
    print_comparison_tables(summary_df, detailed_df)
    
    # Save comparison results
    os.makedirs(args.results_dir, exist_ok=True)
    
    summary_filename = f"{output_name}_summary.csv"
    summary_path = os.path.join(args.results_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary comparison: {summary_filename}")
    
    if not detailed_df.empty:
        detailed_filename = f"{output_name}_detailed.csv"
        detailed_path = os.path.join(args.results_dir, detailed_filename)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Saved detailed comparison: {detailed_filename}")
    
    # Optional: McNemar's test
    if args.mcnemar:
        A, B = args.mcnemar
        run_mcnemar_on_models(args.results_dir, A, B)
    elif args.mcnemar_top2 and not summary_df.empty:
        # Choose top two by Overall Accuracy
        ordered = summary_df.sort_values("Overall Accuracy", ascending=False)["Model Key"].tolist()
        if len(ordered) >= 2:
            A, B = ordered[:2]
            run_mcnemar_on_models(args.results_dir, A, B)

    # Create visualizations if requested
    if args.create_plots:
        print(f"\nCreating visualizations...")
        create_visualizations(summary_df, detailed_df, args.results_dir, output_name)
    
    print(f"\n=== COMPARISON COMPLETE ===")

if __name__ == "__main__":
    main()
