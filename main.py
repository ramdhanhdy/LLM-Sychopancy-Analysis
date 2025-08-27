# main.py
import argparse
import os
from dotenv import load_dotenv
from sycophancy_analysis.pipeline import run_sycophancy_pipeline, collect_responses
from sycophancy_analysis.scoring import run_scoring
from sycophancy_analysis.visualization import run_visualization
from sycophancy_analysis.api import MODEL_CONFIGS, SCORING_CONFIG
from sycophancy_analysis.data import build_sycophancy_battery

# Load environment variables early so downstream modules can read them
load_dotenv()

def ask_bool(prompt: str, default: bool) -> bool:
    options = ["Yes", "No"]
    default_idx = 1 if default else 2
    while True:
        print(f"\n{prompt}")
        for i, opt in enumerate(options, 1):
            marker = " (default)" if i == default_idx else ""
            print(f"  {i}) {opt}{marker}")
        s = input("> ").strip()
        if s == "":
            return default
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(options):
                return idx == 1


def ask_choice(prompt: str, choices, default: str) -> str:
    default_idx = (choices.index(default) + 1) if default in choices else 1
    while True:
        print(f"\n{prompt}")
        for i, opt in enumerate(choices, 1):
            marker = " (default)" if i == default_idx else ""
            print(f"  {i}) {opt}{marker}")
        s = input("> ").strip()
        if s == "":
            return choices[default_idx - 1]
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]


def ask_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _parse_multi_select(s: str, max_idx: int) -> set[int]:
    out: set[int] = set()
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        if tok.isdigit():
            i = int(tok)
            if 1 <= i <= max_idx:
                out.add(i)
    return out


def configure_scoring_menu():
    cfg = SCORING_CONFIG
    print("\n=== Scoring Options (select all that apply) ===")
    print("Enter numbers separated by commas. Press Enter to keep current settings.")
    # Build current state
    cur_regex = bool(cfg.get("USE_REGEX", False))
    cur_stem = bool(cfg.get("USE_STEMMING", False))
    cur_judge = bool(cfg.get("USE_LLM_JUDGE", True))
    print(f"  1) Regex matching [{'ON' if cur_regex else 'OFF'}]")
    print(f"  2) Stemming [{'ON' if cur_stem else 'OFF'}]")
    print(f"  3) LLM Judge [{'ON' if cur_judge else 'OFF'}]")
    s = input("> ").strip()
    if s:
        sel = _parse_multi_select(s, 3)
        cfg["USE_REGEX"] = 1 in sel
        cfg["USE_STEMMING"] = 2 in sel
        cfg["USE_LLM_JUDGE"] = 3 in sel
    # Precedence rule: stemming overrides regex in implementation
    if cfg.get("USE_STEMMING", False) and cfg.get("USE_REGEX", False):
        print("[note] Stemming takes precedence over Regex; disabling Regex.")
        cfg["USE_REGEX"] = False

    # If LLM judge is enabled, it takes precedence over all others
    if cfg.get("USE_LLM_JUDGE", False):
        if cur_regex or cur_stem:
            print("[note] LLM judge overrides heuristic scoring; disabling Regex/Stemming.")
        cfg["USE_REGEX"] = False
        cfg["USE_STEMMING"] = False

        # Configure judge model and params
        if not os.getenv("OPENROUTER_API_KEY"):
            print("[warn] OPENROUTER_API_KEY not set; judge calls will fail.")
        judge_default = cfg.get("LLM_JUDGE_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
        presets = [
            judge_default,
            "qwen/qwen2.5-72b-instruct",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4.1-mini",
            "Custom...",
        ]
        default_choice = judge_default if judge_default in presets else presets[0]
        pick = ask_choice("LLM judge model (OpenRouter slug)", presets, default_choice)
        if pick == "Custom...":
            custom = input("Enter model slug: ").strip()
            if custom:
                cfg["LLM_JUDGE_MODEL"] = custom
        else:
            cfg["LLM_JUDGE_MODEL"] = pick

        temp_default = float(cfg.get("LLM_JUDGE_TEMPERATURE", 0.0))
        cfg["LLM_JUDGE_TEMPERATURE"] = ask_float("Judge temperature", temp_default)
        max_toks_default = int(cfg.get("LLM_JUDGE_MAX_TOKENS", 512))
        try:
            mt_in = input(f"Judge max tokens [{max_toks_default}]: ").strip()
            cfg["LLM_JUDGE_MAX_TOKENS"] = int(mt_in) if mt_in else max_toks_default
        except Exception:
            cfg["LLM_JUDGE_MAX_TOKENS"] = max_toks_default

    # Embedding configuration removed

    print("\nSelected scoring config:")
    print({
        k: SCORING_CONFIG[k]
        for k in [
            "USE_REGEX",
            "USE_STEMMING", 
            "USE_LLM_JUDGE",
            "LLM_JUDGE_MODEL",
            "LLM_JUDGE_TEMPERATURE",
            "LLM_JUDGE_MAX_TOKENS",
        ]
        if k in SCORING_CONFIG
    })


def main():
    parser = argparse.ArgumentParser(description="Run the Sycophancy Analysis Pipeline")
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        default=None,
        help="OpenRouter API Key (or set OPENROUTER_API_KEY in your environment/.env)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "collect", "score", "viz"],
        default="all",
        help="Which stage to run: all, collect, score, viz",
    )
    parser.add_argument("--samples_per_prompt", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--knn_k", type=int, default=8, help="Number of neighbors for k-NN graph")
    parser.add_argument("--leiden_resolution", type=float, default=1.0, help="Resolution parameter for Leiden clustering")
    parser.add_argument("--bridge_threshold", type=float, default=0.5, help="Participation coefficient threshold for bridges")
    parser.add_argument("--save_prefix", type=str, default="sycophancy_run", help="Prefix for saving output files")
    parser.add_argument("--interactive", action="store_true", help="Interactively choose scoring options")
    parser.add_argument("--export_prompts", type=str, default=None, help="Export prompt battery JSON to this path and exit (e.g., dataset/prompt_battery.json)")
    parser.add_argument(
        "--include_slugs",
        type=str,
        default=None,
        help="Comma-separated list of model slugs to include (e.g., 'openai/gpt-4.1,anthropic/claude-3.5-haiku'). If set, only these slugs will run.",
    )
    parser.add_argument(
        "--exclude_slugs",
        type=str,
        default=None,
        help="Comma-separated list of model slugs to exclude (e.g., 'openai/gpt-5-chat').",
    )
    parser.add_argument(
        "--include_names",
        type=str,
        default=None,
        help="Comma-separated list of human-readable model names to include (matches config 'name' field).",
    )
    parser.add_argument(
        "--exclude_names",
        type=str,
        default=None,
        help="Comma-separated list of human-readable model names to exclude (matches config 'name' field).",
    )

    args = parser.parse_args()

    # Allow exporting prompts without requiring an API key
    if args.export_prompts:
        out_path = args.export_prompts
        df = build_sycophancy_battery()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_json(out_path, orient="records", indent=2)
        print(f"Exported {len(df)} prompts to {out_path}")
        return

    # Resolve API key: CLI arg takes precedence, else from env
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    # Only require API key for stages that invoke LLMs
    if args.stage in ("all", "collect", "score") and not api_key:
        raise SystemExit("Missing OpenRouter API key. Pass --api_key or set OPENROUTER_API_KEY in .env/environment.")
    if args.stage == "viz" and not api_key:
        print("[note] No API key provided; proceeding with visualization (reasoning support snapshot will be skipped).")

    if args.interactive:
        configure_scoring_menu()

    # Build filtered model config list based on CLI selection
    def _parse_csv(s: str | None) -> set[str]:
        if not s:
            return set()
        return {tok.strip() for tok in s.split(",") if tok.strip()}

    include_slugs = _parse_csv(args.include_slugs)
    exclude_slugs = _parse_csv(args.exclude_slugs)
    include_names = _parse_csv(args.include_names)
    exclude_names = _parse_csv(args.exclude_names)

    selected_configs = MODEL_CONFIGS
    if include_slugs or include_names:
        selected_configs = [
            c for c in selected_configs
            if ((not include_slugs or c.get("slug") in include_slugs) or (not include_names or c.get("name") in include_names))
        ]
    if exclude_slugs or exclude_names:
        selected_configs = [
            c for c in selected_configs
            if (c.get("slug") not in exclude_slugs) and (c.get("name") not in exclude_names)
        ]

    if not selected_configs:
        raise SystemExit("No models selected after applying filters. Adjust include/exclude flags.")

    # Ensure the output directory exists if saving files
    if args.save_prefix:
        output_dir = os.path.dirname(args.save_prefix) or "."
        os.makedirs(output_dir, exist_ok=True)

    if args.stage == "all":
        run_sycophancy_pipeline(
            api_key=api_key,
            model_configs=selected_configs,
            samples_per_prompt=args.samples_per_prompt,
            temperature=args.temperature,
            system_prompt=args.system_prompt,
            knn_k=args.knn_k,
            leiden_resolution=args.leiden_resolution,
            bridge_threshold=args.bridge_threshold,
            save_prefix=args.save_prefix,
        )
        print("Pipeline (all stages) completed successfully.")
    elif args.stage == "collect":
        prompts_df = build_sycophancy_battery()
        collect_responses(
            model_configs=selected_configs,
            prompts_df=prompts_df,
            api_key=api_key,
            samples_per_prompt=args.samples_per_prompt,
            base_temperature=args.temperature,
            system_prompt=args.system_prompt,
            save_prefix=args.save_prefix,
        )
        print("Collection stage completed successfully.")
    elif args.stage == "score":
        prompts_df = build_sycophancy_battery()
        run_scoring(
            api_key=api_key,
            save_prefix=args.save_prefix,
            model_configs=selected_configs,
            prompts_df=prompts_df,
        )
        print("Scoring stage completed successfully.")
    elif args.stage == "viz":
        run_visualization(
            save_prefix=args.save_prefix,
            knn_k=args.knn_k,
            leiden_resolution=args.leiden_resolution,
            bridge_threshold=args.bridge_threshold,
            model_configs=selected_configs,
            api_key=api_key,
        )
        print("Visualization stage completed successfully.")

    print(f"Results saved with prefix: {args.save_prefix}")


if __name__ == "__main__":
    main()