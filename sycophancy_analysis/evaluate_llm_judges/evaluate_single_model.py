import os, json, sys, time
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# Calculate project root (go up 2 levels from evaluate_llm_judges)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sycophancy_analysis.api import SCORING_CONFIG
from sycophancy_analysis.scoring import (
    PromptMeta,
    PromptScores,
    score_response_llm,
    score_response,
    score_response_llm_with_meta,
)

# Model configurations with optimized settings
MODEL_CONFIGS = {
    "hermes-4-405b": {
        "model": "nousresearch/hermes-4-405b",
        "temperature": 0.4,
        "min_p": 0.01,
        "top_p": 0.9,
        "max_tokens": 1000,
        "provider_prefs": {}
    },
    "kimi-k2": {
        "model": "moonshotai/kimi-k2",
        "temperature": 0.67,
        "min_p": 0.01,
        "top_p": 0.9,
        "max_tokens": 1000,
        "provider_prefs": {"moonshotai": 1.0}
    },
    "deepseek-v3.1": {
        "model": "deepseek/deepseek-chat-v3.1",
        "temperature": 0.6,
        "min_p": 0.01,
        "top_p": 0.9,
        "max_tokens": 1000,
        "provider_prefs": {"fireworks": 1.0}
    },
    "Qwen3-235B-A22B-Instruct-2507": {
        "model": "qwen/qwen3-235b-a22b-2507",
        "temperature": 0.6,
        "min_p": 0.05,
        "top_p": 0.9,
        "max_tokens": 1000,
        "provider_prefs": {"only": ["cerebras"]}
    },
    "gpt-5-mini": {
        "model": "openai/gpt-5-mini",
        "temperature": 0.5,
        "min_p": 0.01,
        "top_p": 0.9,
        "max_tokens": 1000,
        "provider_prefs": {"openai": 1.0}
    },
}

def setup_argparse():
    """Setup command line argument parsing"""
    parser = argparse.ArgumentParser(description="Evaluate sycophancy detection with specific model configurations")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to use"
    )
    parser.add_argument(
        "--sample-n", 
        type=int, 
        default=140,
        help="Number of samples to evaluate (default: 140)"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for results (default: project_root/evaluation_results)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser

def load_and_prepare_data(sample_n: int, random_seed: int, verbose: bool = False):
    """Load and prepare the evaluation dataset"""
    DATASET_DIR = os.path.join(project_root, "dataset")
    PROMPTS_JSON = os.path.join(DATASET_DIR, "prompt_battery.json")
    GROUND_TRUTH_CSV = os.path.join(DATASET_DIR, "ground_truth.csv")

    # Load prompts
    with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_df = pd.DataFrame(prompts)
    prompts_df["prompt_id"] = prompts_df["prompt_id"].astype(str)

    # Load ground truth
    all_gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    all_gt_df["prompt_id"] = all_gt_df["prompt_id"].astype(str)

    # Sample data if needed
    if sample_n and sample_n > 0 and sample_n < len(all_gt_df):
        gt_df = all_gt_df.sample(n=sample_n, random_state=random_seed).reset_index(drop=True)
    else:
        gt_df = all_gt_df.copy().reset_index(drop=True)

    if verbose:
        print(f"Loaded prompts: {len(prompts_df)} | GT rows (sampled): {len(gt_df)}")

    # Build prompt metadata
    pmeta: Dict[str, PromptMeta] = {}
    for r in prompts_df.itertuples(index=False):
        s_val = int(getattr(r, "stance", 0))
        topic_harm_flag = bool(getattr(r, "is_harmful", False))
        pmeta[r.prompt_id] = PromptMeta(
            stance=s_val,
            strength=float(getattr(r, "strength", 0.0) or 0.0),
            is_harmful=(topic_harm_flag and (s_val == +1)),
            ask_devil=bool(getattr(r, "ask_devil", False)),
            topic=str(getattr(r, "topic", "misc")),
            persona=str(getattr(r, "persona", "neutral")),
            prompt_text=str(getattr(r, "text", "")),
        )

    return gt_df, pmeta

def configure_model(model_key: str, verbose: bool = False):
    """Configure the scoring system for a specific model"""
    config = MODEL_CONFIGS[model_key]
    
    # Update scoring configuration
    SCORING_CONFIG["USE_LLM_JUDGE"] = True
    SCORING_CONFIG["LLM_JUDGE_MODEL"] = config["model"]
    SCORING_CONFIG["LLM_JUDGE_TEMPERATURE"] = config["temperature"]
    SCORING_CONFIG["LLM_JUDGE_MAX_TOKENS"] = config["max_tokens"]
    SCORING_CONFIG["LLM_JUDGE_PROVIDER_PREFS"] = config["provider_prefs"]
    
    if verbose:
        print(f"Configured model: {config['model']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  Max tokens: {config['max_tokens']}")
        print(f"  Provider preferences: {config['provider_prefs']}")
    
    return config

def evaluate_responses(gt_df: pd.DataFrame, pmeta: Dict[str, PromptMeta], 
                      model_config: Dict, api_key: str, verbose: bool = False):
    """Evaluate responses using the configured model with progress tracking"""
    if not api_key:
        raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or add it to .env file")
    
    rows: List[Dict] = []
    total_rows = len(gt_df)
    
    # Setup progress tracking
    start_time = time.time()
    progress_bar = tqdm(total=total_rows, desc="Evaluating responses", unit="response")
    
    # Track processing statistics
    processed_count = 0
    error_count = 0
    
    for i, r in enumerate(gt_df.itertuples(index=False)):
        pid = getattr(r, "prompt_id")
        meta = pmeta.get(pid)
        if meta is None:
            progress_bar.update(1)
            continue
            
        resp = str(getattr(r, "response", ""))

        try:
            s, call_meta = score_response_llm_with_meta(resp=resp, meta=meta, api_key=api_key)
        except Exception as e:
            error_count += 1
            progress_bar.update(1)
            if verbose:
                logging.error(f"Error scoring row {i+1}: {e}")
            continue

        # Ensure pred_label is present
        if not hasattr(s, 'pred_label') or s.pred_label is None:
            logging.warning(f"Missing pred_label for prompt {pid}, skipping")
            progress_bar.update(1)
            continue

        row = {
            "model": getattr(r, "model"),
            "prompt_id": pid,
            "human_eval": getattr(r, "human_eval"),
            "judge_model": model_config["model"],
            "judge_temperature": model_config["temperature"],
            "judge_max_tokens": model_config["max_tokens"],
            "pred_label": s.pred_label,
            "endorse_stance": s.endorse_stance,
            "harm_validation": s.harm_validation,
            **{k: v for k, v in s.__dict__.items() if k not in ["pred_label", "endorse_stance", "harm_validation"]},
        }
        # Append API usage/cost metadata
        row.update({
            "latency_ms": getattr(call_meta, "latency_ms", None),
            "retry_count": getattr(call_meta, "retry_count", None),
            "http_status": getattr(call_meta, "http_status", None),
            "provider": getattr(call_meta, "provider", None),
            "prompt_tokens": getattr(call_meta, "prompt_tokens", None),
            "completion_tokens": getattr(call_meta, "completion_tokens", None),
            "total_tokens": getattr(call_meta, "total_tokens", None),
            "total_cost_usd": getattr(call_meta, "total_cost_usd", None),
            "request_units": getattr(call_meta, "request_units", None),
            "currency": getattr(call_meta, "currency", None),
            "stop_reason": getattr(call_meta, "stop_reason", None),
            "request_id": getattr(call_meta, "request_id", None),
            "response_id": getattr(call_meta, "response_id", None),
            "reasoning_applied": getattr(call_meta, "reasoning_applied", None),
            "structured_output_applied": getattr(call_meta, "structured_output_applied", None),
            "provider_forced": getattr(call_meta, "provider_forced", None),
        })
        rows.append(row)
        
        processed_count += 1
        
        # Update progress bar with ETA
        elapsed = time.time() - start_time
        if processed_count > 0:
            rate = processed_count / elapsed
            eta = (total_rows - processed_count) / rate
            progress_bar.set_postfix({
                'Rate': f'{rate:.1f} resp/s',
                'ETA': f'{eta:.0f}s',
                'Errors': error_count
            })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    total_time = time.time() - start_time
    if verbose:
        print(f"Evaluation completed in {total_time:.1f}s")
        print(f"Processed: {processed_count}, Errors: {error_count}")
        print(f"Average rate: {processed_count/total_time:.1f} responses/second")
    
    if processed_count == 0:
        raise RuntimeError("No responses were successfully processed")
    
    return pd.DataFrame(rows)

def calculate_metrics(scored_df: pd.DataFrame, verbose: bool = False):
    """Calculate evaluation metrics"""
    # Filter to evaluable rows
    eval_subset = scored_df[scored_df["human_eval"].isin(["EVASION", "CHALLENGE", "AGREEMENT"])].copy()
    
    if len(eval_subset) == 0:
        return {
            "overall_accuracy": 0.0,
            "model_accuracies": {},
            "confusion_matrix": pd.DataFrame(),
            "total_samples": len(scored_df),
            "evaluated_samples": 0
        }
    
    # Map integer pred_label to category strings for comparison
    pred_map = {1: "AGREEMENT", -1: "CHALLENGE", 0: "EVASION"}
    pred_cat = eval_subset["pred_label"].map(pred_map)
    
    # Overall accuracy
    overall_acc = float((eval_subset["human_eval"] == pred_cat).mean())
    
    # Per-model accuracy
    model_accuracies = {}
    for model_name in eval_subset["model"].unique():
        model_subset = eval_subset[eval_subset["model"] == model_name]
        if len(model_subset) > 0:
            model_acc = (model_subset["human_eval"] == pred_cat[model_subset.index]).mean()
            model_accuracies[model_name] = float(model_acc)
    
    # Confusion matrix
    cm_df = pd.DataFrame({
        "human_eval": eval_subset["human_eval"],
        "pred_category": pred_cat,
    })
    confusion_matrix = cm_df.groupby(["human_eval", "pred_category"]).size().unstack(fill_value=0)
    
    # Per-category metrics
    category_metrics = {}
    for category in ["EVASION", "CHALLENGE", "AGREEMENT"]:
        cat_subset = eval_subset[eval_subset["human_eval"] == category]
        if len(cat_subset) > 0:
            correct = (cat_subset["human_eval"] == pred_cat[cat_subset.index]).sum()
            total = len(cat_subset)
            category_metrics[category] = {
                "accuracy": float(correct / total),
                "count": total,
                "correct": int(correct)
            }
    
    if verbose:
        print(f"Overall accuracy: {overall_acc:.3f}")
        print(f"Evaluated samples: {len(eval_subset)}/{len(scored_df)}")
        print("\nPer-model accuracies:")
        for model, acc in model_accuracies.items():
            print(f"  {model}: {acc:.3f}")
    
    return {
        "overall_accuracy": overall_acc,
        "model_accuracies": model_accuracies,
        "confusion_matrix": confusion_matrix,
        "category_metrics": category_metrics,
        "total_samples": len(scored_df),
        "evaluated_samples": len(eval_subset)
    }

def save_results(scored_df: pd.DataFrame, metrics: Dict, model_key: str, 
                model_config: Dict, output_dir: str, timestamp: str):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed scores
    detailed_filename = f"{model_key}_detailed_scores_{timestamp}.csv"
    scored_df.to_csv(os.path.join(output_dir, detailed_filename), index=False)
    
    # Save summary metrics
    summary_filename = f"{model_key}_summary_{timestamp}.json"
    
    # Build cost/usage summary from scored_df
    def safe_sum(series: pd.Series):
        try:
            return pd.to_numeric(series, errors="coerce").sum(min_count=1)
        except Exception:
            return None
    n = len(scored_df)
    total_cost = safe_sum(scored_df["total_cost_usd"]) if "total_cost_usd" in scored_df.columns else None
    total_req_units = safe_sum(scored_df["request_units"]) if "request_units" in scored_df.columns else None
    total_prompt_tokens = safe_sum(scored_df["prompt_tokens"]) if "prompt_tokens" in scored_df.columns else None
    total_completion_tokens = safe_sum(scored_df["completion_tokens"]) if "completion_tokens" in scored_df.columns else None
    total_tokens = safe_sum(scored_df["total_tokens"]) if "total_tokens" in scored_df.columns else None
    currency = None
    if "currency" in scored_df.columns:
        nn = scored_df["currency"].dropna()
        if len(nn) > 0:
            try:
                currency = str(nn.mode().iloc[0])
            except Exception:
                currency = str(nn.iloc[0])
    avg_cost_per_response = (float(total_cost) / n) if (total_cost is not None and n) else None
    avg_tokens_per_response = (float(total_tokens) / n) if (total_tokens is not None and n) else None
    cost_per_1k_tokens = (float(total_cost) / (float(total_tokens) / 1000.0)) if (total_cost and total_tokens and float(total_tokens) > 0) else None

    summary_data = {
        "model_config": model_config,
        "evaluation_timestamp": timestamp,
        "metrics": {
            "overall_accuracy": metrics["overall_accuracy"],
            "total_samples": metrics["total_samples"],
            "evaluated_samples": metrics["evaluated_samples"],
            "model_accuracies": metrics["model_accuracies"],
            "category_metrics": metrics["category_metrics"]
        },
        "cost_summary": {
            "currency": currency,
            "total_cost_usd": float(total_cost) if total_cost is not None else None,
            "total_request_units": float(total_req_units) if total_req_units is not None else None,
            "total_prompt_tokens": float(total_prompt_tokens) if total_prompt_tokens is not None else None,
            "total_completion_tokens": float(total_completion_tokens) if total_completion_tokens is not None else None,
            "total_tokens": float(total_tokens) if total_tokens is not None else None,
            "avg_cost_per_response": avg_cost_per_response,
            "avg_tokens_per_response": avg_tokens_per_response,
            "cost_per_1k_tokens": cost_per_1k_tokens,
        },
        "confusion_matrix": metrics["confusion_matrix"].to_dict() if not metrics["confusion_matrix"].empty else {}
    }
    
    with open(os.path.join(output_dir, summary_filename), "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    return detailed_filename, summary_filename

def main():
    """Main execution function with comprehensive progress tracking"""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(project_root, "evaluation_results")
    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
    
    if not api_key:
        print("❌ Error: OpenRouter API key not found")
        print("Please set OPENROUTER_API_KEY environment variable or add it to .env file")
        print("Example: OPENROUTER_API_KEY=your_key_here")
        return
    
    # Progress tracking setup
    print("🚀 Starting LLM Judge Evaluation")
    print("=" * 50)
    
    overall_start_time = time.time()
    
    print(f"📊 Model: {args.model}")
    print(f"🕐 Timestamp: {timestamp}")
    print(f"📈 Sample size: {args.sample_n}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔑 API key loaded from environment")
    print()
    
    # Load data with progress
    print("📂 Loading dataset...")
    load_start = time.time()
    gt_df, pmeta = load_and_prepare_data(args.sample_n, args.random_seed, args.verbose)
    load_time = time.time() - load_start
    print(f"✅ Dataset loaded in {load_time:.1f}s - {len(gt_df)} samples")
    print()
    
    # Configure model
    print("⚙️  Configuring model...")
    config_start = time.time()
    model_config = configure_model(args.model, args.verbose)
    config_time = time.time() - config_start
    print(f"✅ Model configured in {config_time:.1f}s")
    print()
    
    # Run evaluation with progress bar
    print("🎯 Running evaluation...")
    eval_start = time.time()
    scored_df = evaluate_responses(gt_df, pmeta, model_config, api_key, args.verbose)
    eval_time = time.time() - eval_start
    print(f"✅ Evaluation completed in {eval_time:.1f}s")
    print(f"📊 Scored {len(scored_df)} responses")
    print()
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics_start = time.time()
    metrics = calculate_metrics(scored_df, args.verbose)
    metrics_time = time.time() - metrics_start
    print(f"✅ Metrics calculated in {metrics_time:.1f}s")
    print()
    
    # Print comprehensive summary
    total_time = time.time() - overall_start_time
    print("🎉 === EVALUATION RESULTS ===")
    print("-" * 30)
    print(f"🤖 Model: {model_config['model']}")
    print(f"🎯 Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"📊 Samples Evaluated: {metrics['evaluated_samples']}/{metrics['total_samples']}")
    print(f"⏱️  Total Runtime: {total_time:.1f}s")
    
    if metrics['model_accuracies']:
        print("\n📊 Per-Target-Model Accuracy:")
        for model, acc in metrics['model_accuracies'].items():
            print(f"  🏷️  {model}: {acc:.3f}")
    
    if metrics['category_metrics']:
        print("\n📊 Per-Category Performance:")
        for category, cat_metrics in metrics['category_metrics'].items():
            print(f"  🏷️  {category}: {cat_metrics['accuracy']:.3f} ({cat_metrics['correct']}/{cat_metrics['count']})")
    
    if not metrics['confusion_matrix'].empty:
        print("\n📊 Confusion Matrix (rows=Ground Truth, cols=Predicted):")
        print(metrics['confusion_matrix'])
    
    # Save results with progress
    print(f"\n💾 Saving results to {output_dir}...")
    save_start = time.time()
    detailed_file, summary_file = save_results(
        scored_df, metrics, args.model, model_config, output_dir, timestamp
    )
    save_time = time.time() - save_start
    print(f"✅ Results saved in {save_time:.1f}s")
    print(f"  📄 Detailed scores: {detailed_file}")
    print(f"  📄 Summary metrics: {summary_file}")
    
    print("\n🎊 === EVALUATION COMPLETE ===")
    print(f"📊 Total processing time: {total_time:.1f}s")
    print(f"📈 Average rate: {len(scored_df)/total_time:.1f} responses/second")

if __name__ == "__main__":
    main()