"""Local dev runner for the dashboard Flask application."""
from __future__ import annotations

import os

from sycophancy_analysis.dashboard import create_app
from sycophancy_analysis.dashboard.app import DEFAULT_FINAL_RESULTS_DIR, data_loader

app = create_app()


def _str_to_bool(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def main() -> None:
    print("Starting LLM Judge Dashboard Server...")
    print(f"Loaded {len(data_loader.prompt_battery)} prompts")
    print(f"Loaded {len(data_loader.evaluation_results)} model evaluations")
    print("Available models:", data_loader.get_model_list())
    print("Available topics:", data_loader.get_topics())
    print("Final results path (default):", DEFAULT_FINAL_RESULTS_DIR)

    env_mode = (os.getenv("ENV") or os.getenv("FLASK_ENV") or "production").strip().lower()
    debug_env = os.getenv("DEBUG") or os.getenv("FLASK_DEBUG")
    host_env = os.getenv("HOST")
    port_env = os.getenv("PORT")

    if debug_env is not None:
        debug = _str_to_bool(debug_env)
    else:
        debug = env_mode in {"development", "dev"}

    host = host_env or ("0.0.0.0" if env_mode in {"development", "dev"} else "127.0.0.1")
    try:
        port = int(port_env) if port_env else 5000
    except ValueError:
        port = 5000

    print(f"\nENV mode: {env_mode}")
    print(f"Debug: {debug}")
    print(f"Binding: http://{host}:{port}")

    if env_mode not in {"development", "dev"} and host == "0.0.0.0":
        print("[warning] Binding to 0.0.0.0 outside development is unsafe; switching to 127.0.0.1")
        host = "127.0.0.1"

    if env_mode not in {"development", "dev"}:
        print("\n[notice] Production mode detected. For production, prefer a WSGI server (e.g., gunicorn):")
        print("         gunicorn -w 4 'dashboard_server:app'")

    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    main()
