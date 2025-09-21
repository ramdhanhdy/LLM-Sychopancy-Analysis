"""Command-line interface for the sycophancy analysis pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

import typer
from dotenv import load_dotenv

from sycophancy_analysis.api import MODEL_CONFIGS, SCORING_CONFIG
from sycophancy_analysis.data import build_sycophancy_battery, collect_responses
from sycophancy_analysis.pipeline import run_sycophancy_pipeline
from sycophancy_analysis.scoring import run_scoring
from sycophancy_analysis.visualization import run_visualization

load_dotenv()

app = typer.Typer(help="Run stages of the sycophancy analysis workflow.")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Simple terminal chooser used for interactive scoring config."""
    default_idx = (choices.index(default) + 1) if default in choices else 1
    while True:
        typer.echo(f"\n{prompt}")
        for i, opt in enumerate(choices, 1):
            marker = " (default)" if i == default_idx else ""
            typer.echo(f"  {i}) {opt}{marker}")
        s = input("> ").strip()
        if s == "":
            return choices[default_idx - 1]
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]


def ask_float(prompt: str, default: float) -> float:
    val = input(f"{prompt} [{default}]: ").strip()
    if val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _parse_multi_select(raw: str, max_idx: int) -> Set[int]:
    out: Set[int] = set()
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= max_idx:
                out.add(idx)
    return out


def configure_scoring_menu() -> None:
    """Interactive helper that mutates SCORING_CONFIG in-place."""
    cfg = SCORING_CONFIG
    typer.echo("\n=== Scoring Options (select all that apply) ===")
    typer.echo("Enter numbers separated by commas. Press Enter to keep current settings.")
    cur_regex = bool(cfg.get("USE_REGEX", False))
    cur_stem = bool(cfg.get("USE_STEMMING", False))
    cur_judge = bool(cfg.get("USE_LLM_JUDGE", True))
    typer.echo(f"  1) Regex matching [{'ON' if cur_regex else 'OFF'}]")
    typer.echo(f"  2) Stemming [{'ON' if cur_stem else 'OFF'}]")
    typer.echo(f"  3) LLM Judge [{'ON' if cur_judge else 'OFF'}]")
    selection = input("> ").strip()
    if selection:
        sel = _parse_multi_select(selection, 3)
        cfg["USE_REGEX"] = 1 in sel
        cfg["USE_STEMMING"] = 2 in sel
        cfg["USE_LLM_JUDGE"] = 3 in sel
    if cfg.get("USE_STEMMING", False) and cfg.get("USE_REGEX", False):
        typer.echo("[note] Stemming takes precedence over Regex; disabling Regex.")
        cfg["USE_REGEX"] = False
    if cfg.get("USE_LLM_JUDGE", False):
        if cur_regex or cur_stem:
            typer.echo("[note] LLM judge overrides heuristic scoring; disabling Regex/Stemming.")
        cfg["USE_REGEX"] = False
        cfg["USE_STEMMING"] = False
        if not os.getenv("OPENROUTER_API_KEY"):
            typer.echo("[warn] OPENROUTER_API_KEY not set; judge calls will fail.")
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
            max_tok_in = input(f"Judge max tokens [{max_toks_default}]: ").strip()
            cfg["LLM_JUDGE_MAX_TOKENS"] = int(max_tok_in) if max_tok_in else max_toks_default
        except ValueError:
            cfg["LLM_JUDGE_MAX_TOKENS"] = max_toks_default


def _parse_csv_option(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {tok.strip() for tok in value.split(",") if tok.strip()}


@app.command()
def pipeline(
    stage: str = typer.Option(
        "all",
        "--stage",
        "-s",
        help="Which stage to run: all, collect, score, or viz.",
        show_default=True,
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key", "--api_key",
        help="OpenRouter API key (falls back to OPENROUTER_API_KEY).",
        envvar="OPENROUTER_API_KEY",
    ),
    samples_per_prompt: int = typer.Option(1, "--samples-per-prompt", "--samples_per_prompt", help="Number of samples per prompt when collecting."),
    temperature: float = typer.Option(0.2, help="Base sampling temperature."),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", "--system_prompt", help="Optional system prompt."),
    knn_k: int = typer.Option(8, "--knn-k", "--knn_k", help="k for kNN graph."),
    leiden_resolution: float = typer.Option(1.0, "--leiden-resolution", "--leiden_resolution", help="Leiden resolution for clustering."),
    bridge_threshold: float = typer.Option(0.5, "--bridge-threshold", "--bridge_threshold", help="Bridge edge threshold when plotting network."),
    save_prefix: Optional[str] = typer.Option(None, "--save-prefix", "--save_prefix", help="Prefix for saving outputs."),
    interactive: bool = typer.Option(False, "--interactive", help="Launch scoring configuration wizard."),
    include_slugs: Optional[str] = typer.Option(None, "--include-slugs", "--include_slugs", help="Comma-separated slugs to include."),
    exclude_slugs: Optional[str] = typer.Option(None, "--exclude-slugs", "--exclude_slugs", help="Comma-separated slugs to exclude."),
    include_names: Optional[str] = typer.Option(None, "--include-names", "--include_names", help="Comma-separated model names to include."),
    exclude_names: Optional[str] = typer.Option(None, "--exclude-names", "--exclude_names", help="Comma-separated model names to exclude."),
    export_prompts: Optional[Path] = typer.Option(None, "--export-prompts", "--export_prompts", help="Write prompt battery JSON then exit."),
) -> None:
    """Run one or more stages of the pipeline."""
    stage = stage.lower()
    allowed = {"all", "collect", "score", "viz"}
    if stage not in allowed:
        raise typer.BadParameter("Stage must be one of: all, collect, score, viz.")

    if export_prompts:
        df = build_sycophancy_battery()
        export_prompts.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(export_prompts, orient="records", indent=2)
        typer.echo(f"Exported {len(df)} prompts to {export_prompts}")
        raise typer.Exit()

    if interactive:
        configure_scoring_menu()

    if stage in {"all", "collect", "score"} and not api_key:
        raise typer.BadParameter("Missing OpenRouter API key for stages that call LLMs.")
    if stage == "viz" and not api_key:
        typer.echo("[note] Running visualization without API key; reasoning snapshots will be skipped.")

    if stage in {"score", "viz"} and not save_prefix:
        raise typer.BadParameter("Missing --save-prefix: score and viz require persisted artifacts to run.")

    include_slug_set = _parse_csv_option(include_slugs)
    exclude_slug_set = _parse_csv_option(exclude_slugs)
    include_name_set = _parse_csv_option(include_names)
    exclude_name_set = _parse_csv_option(exclude_names)

    selected = [
        cfg for cfg in MODEL_CONFIGS
        if (not include_slug_set and not include_name_set or cfg.get("slug") in include_slug_set or cfg.get("name") in include_name_set)
    ] if (include_slug_set or include_name_set) else list(MODEL_CONFIGS)

    if exclude_slug_set or exclude_name_set:
        selected = [
            cfg for cfg in selected
            if (cfg.get("slug") not in exclude_slug_set and cfg.get("name") not in exclude_name_set)
        ]

    if not selected:
        raise typer.BadParameter("No models remain after include/exclude filters.")

    if save_prefix:
        Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)

    if stage == "all":
        run_sycophancy_pipeline(
            api_key=api_key or "",
            model_configs=selected,
            samples_per_prompt=samples_per_prompt,
            temperature=temperature,
            system_prompt=system_prompt,
            knn_k=knn_k,
            leiden_resolution=leiden_resolution,
            bridge_threshold=bridge_threshold,
            save_prefix=save_prefix,
        )
        typer.echo("Pipeline (all stages) completed successfully.")
    elif stage == "collect":
        prompts_df = build_sycophancy_battery()
        collect_responses(
            model_configs=selected,
            prompts_df=prompts_df,
            api_key=api_key or "",
            samples_per_prompt=samples_per_prompt,
            base_temperature=temperature,
            system_prompt=system_prompt,
            save_prefix=save_prefix,
        )
        typer.echo("Collection stage completed successfully.")
    elif stage == "score":
        prompts_df = build_sycophancy_battery()
        run_scoring(
            api_key=api_key or "",
            save_prefix=save_prefix,
            model_configs=selected,
            prompts_df=prompts_df,
        )
        typer.echo("Scoring stage completed successfully.")
    elif stage == "viz":
        run_visualization(
            save_prefix=save_prefix,
            knn_k=knn_k,
            leiden_resolution=leiden_resolution,
            bridge_threshold=bridge_threshold,
            model_configs=selected,
            api_key=api_key,
        )
        typer.echo("Visualization stage completed successfully.")

    if save_prefix:
        typer.echo(f"Results saved with prefix: {save_prefix}")


@app.command("export-prompts")
def export_prompts_cmd(destination: Path) -> None:
    """Export the prompt battery to a JSON file."""
    df = build_sycophancy_battery()
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(destination, orient="records", indent=2)
    typer.echo(f"Exported {len(df)} prompts to {destination}")


@app.command("dashboard")
def dashboard_cmd(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Serve the dashboard locally using the Flask dev server."""
    from sycophancy_analysis.dashboard import create_app

    flask_app = create_app()
    typer.echo(f"Serving dashboard on http://{host}:{port}")
    flask_app.run(debug=debug, host=host, port=port)






