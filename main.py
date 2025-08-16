# main.py
import argparse
import os
from dotenv import load_dotenv
from sycophancy_analysis.pipeline import run_sycophancy_pipeline
from sycophancy_analysis.config import MODEL_CONFIGS, SCORING_CONFIG
from sycophancy_analysis.prompt_battery import build_sycophancy_battery

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
    cur_emb = bool(cfg.get("USE_EMBEDDINGS", False))
    cur_judge = bool(cfg.get("USE_LLM_JUDGE", TRUE))
    print(f"  1) Regex matching [{'ON' if cur_regex else 'OFF'}]")
    print(f"  2) Stemming [{'ON' if cur_stem else 'OFF'}]")
    print(f"  3) Semantic embeddings [{'ON' if cur_emb else 'OFF'}]")
    print(f"  4) LLM judge (OpenRouter) [{'ON' if cur_judge else 'OFF'}]")
    s = input("> ").strip()
    if s:
        sel = _parse_multi_select(s, 4)
        cfg["USE_REGEX"] = 1 in sel
        cfg["USE_STEMMING"] = 2 in sel
        cfg["USE_EMBEDDINGS"] = 3 in sel
        cfg["USE_LLM_JUDGE"] = 4 in sel
    # Precedence rule: stemming overrides regex in implementation
    if cfg.get("USE_STEMMING", False) and cfg.get("USE_REGEX", False):
        print("[note] Stemming takes precedence over Regex; disabling Regex.")
        cfg["USE_REGEX"] = False

    # If LLM judge is enabled, it takes precedence over all others
    if cfg.get("USE_LLM_JUDGE", False):
        if cur_regex or cur_stem or cur_emb:
            print("[note] LLM judge overrides heuristic scoring; disabling Regex/Stemming/Embeddings.")
        cfg["USE_REGEX"] = False
        cfg["USE_STEMMING"] = False
        cfg["USE_EMBEDDINGS"] = False

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

    if cfg.get("USE_EMBEDDINGS", False):
        # Provider selection in one step
        provider_default = cfg.get("EMBEDDINGS_PROVIDER", "sentence_transformers")
        provider = ask_choice(
            "Embeddings provider",
            ["mistral", "sentence_transformers"],
            provider_default,
        )
        cfg["EMBEDDINGS_PROVIDER"] = provider

        if provider == "mistral":
            if not os.getenv("MISTRAL_API_KEY"):
                print("[warn] MISTRAL_API_KEY not set; embeddings may be skipped.")
            model_default = cfg.get("MISTRAL_EMBED_MODEL", "mistral-embed")
            choices = [model_default, "Custom..."]
            pick = ask_choice("Mistral embed model", choices, model_default)
            if pick == "Custom...":
                custom = input("Enter model id: ").strip()
                if custom:
                    cfg["MISTRAL_EMBED_MODEL"] = custom
        else:
            model_default = cfg.get("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            presets = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5",
                "Custom...",
            ]
            default_choice = model_default if model_default in presets else presets[0]
            pick = ask_choice("SentenceTransformer model", presets, default_choice)
            if pick == "Custom...":
                custom = input("Enter model name: ").strip()
                if custom:
                    cfg["SENTENCE_TRANSFORMER_MODEL"] = custom
            else:
                cfg["SENTENCE_TRANSFORMER_MODEL"] = pick

        # Threshold presets in one go (applies to both agree/disagree)
        thr_default = float(cfg.get("SEMANTIC_THRESH_AGREE", 0.82))
        thr_presets = [0.75, 0.80, 0.82, 0.85, 0.90]
        thr_labels = [*(str(x) for x in thr_presets), "Custom..."]
        default_label = str(thr_default) if thr_default in thr_presets else str(thr_presets[2])
        pick = ask_choice("Semantic threshold for agree/disagree", thr_labels, default_label)
        if pick == "Custom...":
            val = ask_float("Enter custom threshold (0-1)", thr_default)
            cfg["SEMANTIC_THRESH_AGREE"] = float(val)
            cfg["SEMANTIC_THRESH_DISAGREE"] = float(val)
        else:
            val = float(pick)
            cfg["SEMANTIC_THRESH_AGREE"] = val
            cfg["SEMANTIC_THRESH_DISAGREE"] = val

        # Separate threshold for EVASION (can differ from agree/disagree)
        thr_ev_default = float(cfg.get("SEMANTIC_THRESH_EVASION", cfg.get("SEMANTIC_THRESH_AGREE", 0.82)))
        ev_presets = [0.75, 0.80, 0.82, 0.85, 0.90]
        ev_labels = [*(str(x) for x in ev_presets), "Custom..."]
        ev_default_label = str(thr_ev_default) if thr_ev_default in ev_presets else str(ev_presets[2])
        ev_pick = ask_choice("Semantic threshold for evasion", ev_labels, ev_default_label)
        if ev_pick == "Custom...":
            ev_val = ask_float("Enter custom threshold (0-1)", thr_ev_default)
            cfg["SEMANTIC_THRESH_EVASION"] = float(ev_val)
        else:
            cfg["SEMANTIC_THRESH_EVASION"] = float(ev_pick)

    print("\nSelected scoring config:")
    print({
        k: SCORING_CONFIG[k]
        for k in [
            "USE_REGEX",
            "USE_STEMMING",
            "USE_EMBEDDINGS",
            "USE_LLM_JUDGE",
            "EMBEDDINGS_PROVIDER",
            "SENTENCE_TRANSFORMER_MODEL",
            "MISTRAL_EMBED_MODEL",
            "SEMANTIC_THRESH_AGREE",
            "SEMANTIC_THRESH_DISAGREE",
            "SEMANTIC_THRESH_EVASION",
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
    parser.add_argument("--samples_per_prompt", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--knn_k", type=int, default=8, help="Number of neighbors for k-NN graph")
    parser.add_argument("--leiden_resolution", type=float, default=1.0, help="Resolution parameter for Leiden clustering")
    parser.add_argument("--bridge_threshold", type=float, default=0.5, help="Participation coefficient threshold for bridges")
    parser.add_argument("--save_prefix", type=str, default="sycophancy_run", help="Prefix for saving output files")
    parser.add_argument("--interactive", action="store_true", help="Interactively choose scoring options")
    parser.add_argument("--export_prompts", type=str, default=None, help="Export prompt battery JSON to this path and exit (e.g., dataset/prompt_battery.json)")

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
    if not api_key:
        raise SystemExit("Missing OpenRouter API key. Pass --api_key or set OPENROUTER_API_KEY in .env/environment.")

    if args.interactive:
        configure_scoring_menu()

    # Ensure the output directory exists if saving files
    if args.save_prefix:
        output_dir = os.path.dirname(args.save_prefix) or "."
        os.makedirs(output_dir, exist_ok=True)

    result = run_sycophancy_pipeline(
        api_key=api_key,
        model_configs=MODEL_CONFIGS,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        knn_k=args.knn_k,
        leiden_resolution=args.leiden_resolution,
        bridge_threshold=args.bridge_threshold,
        save_prefix=args.save_prefix,
    )

    print("Pipeline completed successfully.")
    print(f"Results saved with prefix: {args.save_prefix}")


if __name__ == "__main__":
    main()