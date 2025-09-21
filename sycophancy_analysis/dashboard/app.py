"""
Dashboard server for LLM Judge Evaluation Results
Serves the HTML dashboard and provides API endpoints for evaluation data
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from flask_cors import CORS
import glob
from pathlib import Path
from typing import Optional
import logging

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / 'templates'
BASE_DIR = PACKAGE_DIR.parents[1]
DATASET_DIR = BASE_DIR / 'dataset'
EVALUATION_DIR = BASE_DIR / 'evaluation_results'
RESULTS_DIR = BASE_DIR / 'results'
PROMPT_BATTERY_FILE = DATASET_DIR / 'prompt_battery.json'
DEFAULT_FINAL_RESULTS_DIR = Path(os.environ.get('FINAL_RESULTS_DIR') or (RESULTS_DIR / 'combined_run_0c_1_1b'))

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
CORS(app)

__all__ = ['app', 'create_app']

def create_app() -> Flask:
    '''Return the configured Flask application instance.'''
    return app

# Custom JSON encoder to handle NaN and infinity values
class SafeJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, dict):
            obj = {k: self._safe_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            obj = [self._safe_value(v) for v in obj]
        else:
            obj = self._safe_value(obj)
        return super().encode(obj)

    def _safe_value(self, value):
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
        elif isinstance(value, dict):
            return {k: self._safe_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._safe_value(v) for v in value]
        return value

# Flask 2.3+ JSON provider approach (preferred)
try:
    # Try modern Flask JSON provider approach first
    from flask.json.provider import DefaultJSONProvider
    class SafeJSONProvider(DefaultJSONProvider):
        def default(self, obj):
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
            elif isinstance(obj, dict):
                return {k: self._safe_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._safe_value(v) for v in obj]
            return super().default(obj)

        def _safe_value(self, value):
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    return None
            elif isinstance(value, dict):
                return {k: self._safe_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [self._safe_value(v) for v in value]
            return value

    app.json = SafeJSONProvider(app)
except ImportError:
    # Fallback to legacy json_encoder for older Flask versions
    app.json_encoder = SafeJSONEncoder

# Force server reload - updated CSV loading logic

class DashboardDataLoader:
    def __init__(self):
        self.prompt_battery = self.load_prompt_battery()
        self.evaluation_results = self.load_evaluation_results()
    
    def load_prompt_battery(self):
        """Load the prompt battery JSON file"""
        try:
            with open(PROMPT_BATTERY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {PROMPT_BATTERY_FILE} not found; falling back to programmatic battery build")
            try:
                # Build in-memory prompt battery from the code if file is missing
                from sycophancy_analysis.data.prompts import build_sycophancy_battery
                df = build_sycophancy_battery()
                items = df.to_dict('records')
                # Best-effort: persist for the dashboard to serve next time
                os.makedirs(os.path.dirname(PROMPT_BATTERY_FILE), exist_ok=True)
                with open(PROMPT_BATTERY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                print(f"Rebuilt and saved prompt battery to {PROMPT_BATTERY_FILE}")
                return items
            except Exception as e:
                print(f"Error rebuilding prompt battery: {e}")
                return []
    
    def load_evaluation_results(self):
        """Load all evaluation results from CSV and JSON files"""
        results = {}
        
        # Find all detailed CSV files
        csv_pattern = os.path.join(EVALUATION_DIR, "*_detailed_scores_*.csv")
        csv_files = glob.glob(csv_pattern)
        print(f"[Loader] Searching for CSVs with pattern: {csv_pattern}")
        print(f"[Loader] Found {len(csv_files)} CSV(s)")
        
        for csv_file in csv_files:
            # Extract model name and timestamp from filename
            filename = os.path.basename(csv_file)
            model_name = filename.split('_detailed_scores_')[0]
            detailed_ts = filename.split('_detailed_scores_')[1].rsplit('.', 1)[0]
            
            try:
                # Load detailed scores with proper handling of empty fields
                df = pd.read_csv(csv_file, keep_default_na=False, na_values=[''])
                # Replace NaN/NaT with None so JSON serialization stays standards-compliant
                df = df.where(pd.notna(df), None)
                # Also handle any infinity values
                df = df.replace([float('inf'), float('-inf')], None)
                # Convert any remaining NaN values to None
                df = df.replace({pd.NA: None, 'nan': None, 'NaN': None})
                
                # Load corresponding summary file (match timestamp first, otherwise take latest)
                summary_pattern = os.path.join(EVALUATION_DIR, f"{model_name}_summary_*.json")
                summary_files = glob.glob(summary_pattern)

                selected_summary = None
                if summary_files:
                    # Prefer exact timestamp match
                    for sf in summary_files:
                        if detailed_ts in os.path.basename(sf):
                            selected_summary = sf
                            break
                    # Otherwise pick the latest by timestamp suffix after `_summary_`
                    if not selected_summary:
                        try:
                            selected_summary = sorted(
                                summary_files,
                                key=lambda p: os.path.basename(p).split('_summary_')[1]
                            )[-1]
                        except (IndexError, ValueError):
                            # Handle empty list or malformed filenames
                            if summary_files:
                                selected_summary = summary_files[-1]
                                print(f"Warning: Could not sort summary files by timestamp, using most recent: {selected_summary}")
                            else:
                                selected_summary = None
                                print("Warning: No summary files found")
                        except Exception as e:
                            # Log and re-raise unexpected exceptions
                            print(f"Error selecting summary file: {e}")
                            raise

                summary_data = {}
                if selected_summary:
                    # Convert NaN/Infinity to None during JSON load to avoid invalid JSON downstream
                    with open(selected_summary, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f, parse_constant=lambda _const: None)
                print(f"[Loader] Loaded model: {model_name} (rows: {len(df)})")
                
                # Store results (prefer the latest timestamp per model)
                existing = results.get(model_name)
                if existing and existing.get('timestamp') and existing['timestamp'] >= detailed_ts:
                    # Existing is newer or equal; skip older run
                    continue
                results[model_name] = {
                    'detailed': df.to_dict('records'),
                    'summary': summary_data,
                    'csv_file': csv_file,
                    'summary_file': selected_summary,
                    'timestamp': detailed_ts,
                }
                
            except FileNotFoundError as e:
                print(f"[Loader] File not found: {csv_file} - {e}")
            except PermissionError as e:
                print(f"[Loader] Permission denied accessing: {csv_file} - {e}")
            except UnicodeDecodeError as e:
                print(f"[Loader] Encoding error reading: {csv_file} - {e}")
            except pd.errors.EmptyDataError as e:
                print(f"[Loader] Empty CSV file: {csv_file} - {e}")
            except pd.errors.ParserError as e:
                print(f"[Loader] CSV parsing error in: {csv_file} - {e}")
            except Exception as e:
                import traceback
                print(f"[Loader] Unexpected error loading {csv_file}: {e}")
                traceback.print_exc()
                raise
        
        return results
    
    def get_model_list(self):
        """Get list of available models"""
        return list(self.evaluation_results.keys())
    
    def get_topics(self):
        """Get unique topics from prompt battery"""
        return list(set(prompt['topic'] for prompt in self.prompt_battery))
    
    def get_prompt_with_scores(self, prompt_id, model_name=None):
        """Get prompt data with evaluation scores"""
        # Find prompt in battery
        prompt_data = next((p for p in self.prompt_battery if p['prompt_id'] == prompt_id), None)
        if not prompt_data:
            return None
        
        # Add scores from evaluation results
        scores = {}
        if model_name and model_name in self.evaluation_results:
            detailed_results = self.evaluation_results[model_name]['detailed']
            prompt_scores = next((r for r in detailed_results if r['prompt_id'] == prompt_id), None)
            if prompt_scores:
                scores = {
                    'endorse_stance': prompt_scores.get('endorse_stance'),
                    'harm_validation': prompt_scores.get('harm_validation'),
                    'challenge': prompt_scores.get('challenge'),
                    'flattery_density': prompt_scores.get('flattery_density'),
                    'pred_label': prompt_scores.get('pred_label'),
                    'human_eval': prompt_scores.get('human_eval'),
                }
        
        return {**prompt_data, 'scores': scores}

# Initialize data loader
data_loader = DashboardDataLoader()


class FinalResultsLoader:
    """Load and expose final (non-judge) results for the dashboard."""
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def _load_csv(self, filename: str):
        path = self.base_dir / filename
        if not path.exists():
            return []
        try:
            df = pd.read_csv(path)
            df = df.where(pd.notna(df), None)
            return df.to_dict('records')
        except FileNotFoundError:
            # File doesn't exist - handled by path.exists() check above, but include for completeness
            return []
        except PermissionError as e:
            import logging
            logging.getLogger(__name__).error(f"Permission denied reading {path}: {e}")
            return []
        except UnicodeDecodeError as e:
            import logging
            logging.getLogger(__name__).error(f"Encoding error reading {path}: {e}")
            return []
        except pd.errors.EmptyDataError as e:
            import logging
            logging.getLogger(__name__).warning(f"Empty CSV file {path}: {e}")
            return []
        except pd.errors.ParserError as e:
            import logging
            logging.getLogger(__name__).error(f"CSV parsing error in {path}: {e}")
            return []
        except json.JSONDecodeError as e:
            import logging
            logging.getLogger(__name__).error(f"JSON decode error in {path}: {e}")
            return []
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Unexpected error reading {path}: {e}")
            raise

    def sycophancy_scores(self):
        return self._load_csv("sycophancy_scores.csv")

    def sss_scores(self):
        return self._load_csv("sss_scores.csv")

    def scored_rows(self, limit: Optional[int] = None):
        rows = self._load_csv("scored_rows.csv")
        if limit is not None:
            try:
                return rows[: int(limit)]
            except (ValueError, TypeError) as e:
                logging.getLogger(__name__).error(f"Invalid limit parameter: {limit}, error: {e}")
                return rows
            except Exception as e:
                logging.getLogger(__name__).exception(f"Unexpected error processing limit in scored_rows: {limit}")
                return rows
        return rows

    def stance_elasticity_metrics(self, min_n_per_topic: int = 8, min_topics: int = 6):
        """Compute per-model stance elasticity variability and topic dispersion from scored_rows.csv.

        Returns a dict with:
          - items: list of {model, topics_used, elasticity_var, topic_dispersion_wMAD}
          - summary: medians across models for convenience (x_median, y_median)
        """
        path = self.base_dir / "scored_rows.csv"
        if not path.exists():
            return {"items": [], "summary": {}}
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logging.getLogger(__name__).exception(f"Failed reading {path}")
            return {"items": [], "summary": {}}

        # Resolve stance strength column (accept common aliases)
        strength_col = None
        if "user_stance_strength" in df.columns:
            strength_col = "user_stance_strength"
        elif "strength" in df.columns:
            strength_col = "strength"
            logging.getLogger(__name__).warning("Using 'strength' as alias for 'user_stance_strength' while computing stance elasticity")
        elif "stance_strength" in df.columns:
            strength_col = "stance_strength"
            logging.getLogger(__name__).warning("Using 'stance_strength' as alias for 'user_stance_strength' while computing stance elasticity")

        # Validate required columns
        req = {"model", "topic", "endorse_stance"}
        missing = req - set(df.columns)
        if missing or not strength_col:
            need = sorted(list(req | {"user_stance_strength"}))
            logging.getLogger(__name__).warning(
                f"Missing required columns for stance elasticity. Have: {sorted(df.columns.tolist())[:20]}..., need at least: {need} (strength_col resolved as: {strength_col})"
            )
            return {"items": [], "summary": {}}

        gdf = df.dropna(subset=["endorse_stance", strength_col]).copy()
        if gdf.empty:
            return {"items": [], "summary": {}}

        def _has_variation(x, min_unique=3):
            x = pd.Series(x).dropna()
            # Numeric stability guard
            try:
                return x.nunique() >= min_unique and float(np.nanstd(x)) > 1e-8
            except Exception:
                return False

        rows = []
        # Compute an elasticity value for each (model, topic)
        for (m, t), g in gdf.groupby(["model", "topic"], sort=False):
            n = len(g)
            if n < 5:
                continue
            x = g[strength_col].to_numpy()
            y = g["endorse_stance"].to_numpy()

            # Prefer slope when sufficient coverage & variation
            if n >= int(min_n_per_topic) and _has_variation(x, 3):
                try:
                    slope = float(np.polyfit(x, y, 1)[0])
                    rows.append({"model": m, "topic": t, "elasticity": slope, "n": int(n)})
                    continue
                except Exception:
                    pass

            # Fallback: Spearman correlation on ranks
            try:
                rx = pd.Series(x).rank(method="average")
                ry = pd.Series(y).rank(method="average")
                if _has_variation(rx, 2) and _has_variation(ry, 2):
                    r = float(np.corrcoef(rx, ry)[0, 1])
                    rows.append({"model": m, "topic": t, "elasticity": r, "n": int(n)})
            except Exception:
                continue

        elasticity_df = pd.DataFrame(rows)
        if elasticity_df.empty:
            return {"items": [], "summary": {}}

        def weighted_std(x, w):
            x, w = np.asarray(x, float), np.asarray(w, float)
            mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
            x, w = x[mask], w[mask]
            if len(x) == 0:
                return np.nan
            mu = float(np.average(x, weights=w))
            return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))

        # Per-model: number of topics contributing and weighted std of elasticity by n
        model_elasticity = (
            elasticity_df.groupby("model")
            .apply(lambda g: pd.Series({
                "topics_used": g["topic"].nunique(),
                "elasticity_var": weighted_std(g["elasticity"], g["n"]),
            }))
            .reset_index()
        )

        # Topic dispersion (wMAD) across topic means per model
        topic_means = (
            gdf.groupby(["model", "topic"])  # mean endorse per model-topic
            ["endorse_stance"].mean().reset_index()
        )

        def weighted_mad(x, weights=None):
            x = np.asarray(x, float)
            if weights is None:
                weights = np.ones_like(x)
            else:
                weights = np.asarray(weights, float)
            mask = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
            x, weights = x[mask], weights[mask]
            if len(x) == 0:
                return np.nan
            median_x = float(np.median(x))
            mad = float(np.average(np.abs(x - median_x), weights=weights))
            return mad

        model_dispersion = (
            topic_means.groupby("model")
            .apply(lambda g: pd.Series({
                "topic_count": int(len(g)),
                "topic_dispersion_wMAD": weighted_mad(g["endorse_stance"]),
            }))
            .reset_index()
        )
        
        # Mirror notebook: require sufficient topic coverage for dispersion as well
        model_dispersion = model_dispersion.query("topic_count >= @min_topics").dropna()

        # Merge and filter
        df_out = (
            model_elasticity.merge(
                model_dispersion[["model", "topic_dispersion_wMAD"]], on="model", how="inner"
            )
            .dropna()
        )
        df_out = df_out.query("topics_used >= @min_topics")
        if df_out.empty:
            return {"items": [], "summary": {}}

        # Compute medians for guideline lines
        try:
            x_med = float(np.median(df_out["topic_dispersion_wMAD"]))
        except Exception:
            x_med = None
        try:
            y_med = float(np.median(df_out["elasticity_var"]))
        except Exception:
            y_med = None

        # Round for compact JSON and consistency
        def _round(v):
            try:
                return None if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else round(float(v), 6)
            except Exception:
                return None

        items = []
        for _, row in df_out.iterrows():
            items.append({
                "model": row["model"],
                "topics_used": int(row["topics_used"]) if pd.notna(row["topics_used"]) else None,
                "elasticity_var": _round(row["elasticity_var"]),
                "topic_dispersion_wMAD": _round(row["topic_dispersion_wMAD"]),
            })

        return {"items": items, "summary": {"x_median": _round(x_med), "y_median": _round(y_med)}}

    def network_data(self, k: int = 6, random_state: int = 42):
        """Build a kNN+MST network from similarity/distance matrix with UMAP layout.

        Returns a dict with nodes (id, label, x, y, community) and links (source, target, weight, similarity, kind).
        """
        import numpy as np
        import networkx as nx
        import umap
        try:
            import igraph as ig
            import leidenalg as la
            have_leiden = True
        except Exception:
            have_leiden = False

        sim_path = self.base_dir / "similarity_matrix.npy"
        dist_path = self.base_dir / "distance_matrix.npy"
        names_path = self.base_dir / "model_names.json"
        if not names_path.exists() or (not sim_path.exists() and not dist_path.exists()):
            return {"nodes": [], "links": []}

        try:
            if dist_path.exists():
                D = np.load(dist_path)
                # Robustness: clip negatives and ensure symmetry
                D = np.maximum(D, 0.0)
                D = (D + D.T) / 2.0
                S = 1.0 - (D / (np.max(D) + 1e-9))
            else:
                S = np.load(sim_path)
                S = (S + S.T) / 2.0
                D = 1.0 - S
            with open(names_path, 'r', encoding='utf-8') as f:
                names = json.load(f)
        except Exception as e:
            logging.getLogger(__name__).exception(f"Error loading network data")
            return {"nodes": [], "links": []}

        n = len(names)
        if S.shape[0] != n or S.shape[1] != n:
            print("[FinalResultsLoader] similarity/distance matrix shape mismatch")
            return {"nodes": [], "links": []}

        # kNN edges (union, undirected)
        k = max(1, int(k))
        knn_edges = set()
        for i in range(n):
            sims = [(j, float(S[i, j])) for j in range(n) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            for j, w in sims[:k]:
                a, b = (i, j) if i < j else (j, i)
                knn_edges.add((a, b, w))

        # MST edges on distances (ensure connectivity)
        mst_edges = set()
        try:
            G_full = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    G_full.add_edge(i, j, weight=float(D[i, j]))
            T = nx.minimum_spanning_tree(G_full, weight='weight')
            for u, v, data in T.edges(data=True):
                w = float(S[u, v])
                a, b = (u, v) if u < v else (v, u)
                mst_edges.add((a, b, w))
        except Exception as e:
            logging.getLogger(__name__).exception(f"MST failed")

        # Combine edges, track kind
        edge_map = {}
        for a, b, w in knn_edges:
            edge_map[(a, b)] = {"source": a, "target": b, "weight": w, "similarity": w, "kind": "knn"}
        for a, b, w in mst_edges:
            key = (a, b)
            if key in edge_map:
                edge_map[key]["kind"] = "knn+mst"
            else:
                edge_map[key] = {"source": a, "target": b, "weight": w, "similarity": w, "kind": "mst"}

        # UMAP layout from distances (precomputed)
        try:
            reducer = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=min(10, n-1), min_dist=0.2, random_state=random_state)
            emb = reducer.fit_transform(D)
            # Normalize to [0,1]
            min_xy = emb.min(axis=0)
            max_xy = emb.max(axis=0)
            span = np.where((max_xy - min_xy) == 0, 1.0, (max_xy - min_xy))
            norm = (emb - min_xy) / span
        except Exception as e:
            logging.getLogger(__name__).exception(f"UMAP failed, falling back to spring_layout")
            # Build proper NetworkX graph for spring layout fallback
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for (a, b) in edge_map.keys():
                G.add_edge(a, b, weight=float(edge_map[(a, b)]['similarity']))
            pos = nx.spring_layout(G, seed=random_state)
            # pos might be empty if graph construction failed; fallback to grid
            norm = np.zeros((n, 2), dtype=float)
            for i in range(n):
                p = pos.get(i, (i % 10, i // 10))
                norm[i, 0] = float(p[0])
                norm[i, 1] = float(p[1])
            # Normalize
            min_xy = norm.min(axis=0)
            max_xy = norm.max(axis=0)
            span = np.where((max_xy - min_xy) == 0, 1.0, (max_xy - min_xy))
            norm = (norm - min_xy) / span

        # Community detection using Leiden (if available), otherwise degree-based buckets
        membership = [0] * n
        if have_leiden:
            try:
                # Build igraph graph with weights from similarity
                edges_list = [(int(a), int(b)) for (a, b) in edge_map.keys()]
                weights = [float(edge_map[(a, b)]["similarity"]) for (a, b) in edge_map.keys()]
                g = ig.Graph(n=n, edges=edges_list, directed=False)
                g.es['weight'] = weights
                part = la.find_partition(g, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=1.0)
                membership = part.membership
            except Exception:
                logging.getLogger(__name__).exception(f"Leiden failed")
        else:
            # Simple heuristic: assign all to one community
            membership = [0] * n

        nodes = [{"id": i, "label": names[i], "x": float(norm[i, 0]), "y": float(norm[i, 1]), "community": int(membership[i])} for i in range(n)]
        links = list(edge_map.values())
        return {"nodes": nodes, "links": links, "layout": "umap", "params": {"k": k}}

    def network_png(self, knn_k: int = 6, layout: str = "umap", leiden_resolution: float = 1.0,
                    umap_neighbors: int = 25, umap_min_dist: float = 0.08, seed: int = 42,
                    bridge_threshold: float = 0.5):
        """Render network using the project's Matplotlib codepath for visual parity with notebooks."""
        import numpy as np
        import matplotlib.pyplot as plt
        from io import BytesIO
        try:
            from sycophancy_analysis.visualization import analyze_network, plot_network, _symmetrize_clip
        except Exception as e:
            logging.getLogger(__name__).exception(f"Import error for visualization pipeline")
            return None

        sim_path = self.base_dir / "similarity_matrix.npy"
        names_path = self.base_dir / "model_names.json"
        if not names_path.exists() or not sim_path.exists():
            return None

        try:
            S = np.load(sim_path)
            S = _symmetrize_clip(S)
            with open(names_path, 'r', encoding='utf-8') as f:
                names = json.load(f)
        except Exception as e:
            print(f"[FinalResultsLoader] Error loading for PNG render: {e}")
            return None

        res = analyze_network(
            names,
            S,
            knn_k=int(knn_k),
            leiden_resolution=float(leiden_resolution),
            layout_method=str(layout),
            umap_neighbors=int(umap_neighbors),
            umap_min_dist=float(umap_min_dist),
            seed=int(seed),
        )
        fig = plot_network(
            names,
            S,
            pos=res['layout'],
            G_backbone=res['graph'],
            node_to_comm=res['communities'],
            Q=float(res.get('modularity', 0.0)),
            conductance=res.get('conductance', {}),
            participation=res.get('participation', {}),
            title='Model Similarity Network (Final Results)',
            bridge_threshold=float(bridge_threshold),
        )
        buf = BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        finally:
            plt.close(fig)
        buf.seek(0)
        return buf


final_loader = FinalResultsLoader(DEFAULT_FINAL_RESULTS_DIR)


def _get_final_loader():
    """Return a FinalResultsLoader based on optional ?prefix= path; fall back to the default loader.

    Allows dashboard consumers to point at any results subfolder without changing server code.
    """
    try:
        prefix = request.args.get('prefix', default=None, type=str)
    except Exception:
        prefix = None
    if prefix:
        # Accept both absolute and relative paths; restrict to existing dir within RESULTS_DIR subtree
        try:
            p = Path(prefix)
            # If relative, resolve against BASE_DIR
            if not p.is_absolute():
                p = (BASE_DIR / p).resolve()
            # Security: ensure the path is within RESULTS_DIR subtree to prevent path traversal
            results_dir = RESULTS_DIR.resolve()
            if p.exists() and p.is_dir() and results_dir in p.parents or p == results_dir:
                return FinalResultsLoader(str(p))
        except Exception:
            pass
    return final_loader

@app.route('/')
def dashboard():
    """Serve the main dashboard HTML"""
    return render_template('llm_judge_dashboard.html')

@app.route('/judge')
def judge_dashboard():
    """Serve the judge-only dashboard HTML"""
    return render_template('llm_judge_dashboard.html')

@app.route('/final')
def final_results_dashboard():
    """Serve the final results dashboard HTML."""
    return render_template('final_results_dashboard.html')

@app.route('/prompts')
def prompt_explorer():
    """Serve the prompt explorer page."""
    return render_template('prompt_explorer.html')

@app.route('/api/prompt_battery')
def get_prompt_battery():
    """API endpoint to get prompt battery data"""
    return jsonify(data_loader.prompt_battery)

@app.route('/api/models')
def get_models():
    """API endpoint to get available models"""
    return jsonify(data_loader.get_model_list())

@app.route('/api/topics')
def get_topics():
    """API endpoint to get available topics"""
    return jsonify(data_loader.get_topics())

@app.route('/api/evaluation_results')
def get_evaluation_results():
    """API endpoint to get all evaluation results"""
    # Return summary data only to avoid large payloads
    summary_results = {}
    for model_name, data in data_loader.evaluation_results.items():
        summary_results[model_name] = {
            'summary': data['summary'],
            'prompt_count': len(data['detailed'])
        }
    return jsonify(summary_results)

@app.route('/api/evaluation_results/<model_name>')
def get_model_results(model_name):
    """API endpoint to get detailed results for a specific model"""
    if model_name not in data_loader.evaluation_results:
        return jsonify({'error': 'Model not found'}), 404
    
    # Clean the data to ensure valid JSON serialization
    result = data_loader.evaluation_results[model_name].copy()
    if 'detailed' in result:
        # Convert any NaN values to None in detailed results
        cleaned_detailed = []
        for row in result['detailed']:
            cleaned_row = {}
            for k, v in row.items():
                if pd.isna(v) or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    cleaned_row[k] = None
                else:
                    cleaned_row[k] = v
            cleaned_detailed.append(cleaned_row)
        result['detailed'] = cleaned_detailed
    
    return jsonify(result)

@app.route('/api/prompt_scores/<prompt_id>')
def get_prompt_scores(prompt_id):
    """API endpoint to get scores for a specific prompt across all models"""
    results = {}
    for model_name in data_loader.evaluation_results:
        prompt_data = data_loader.get_prompt_with_scores(prompt_id, model_name)
        if prompt_data:
            results[model_name] = prompt_data['scores']
    
    return jsonify(results)

@app.route('/api/prompt_scores/<prompt_id>/<model_name>')
def get_prompt_model_scores(prompt_id, model_name):
    """API endpoint to get scores for a specific prompt and model"""
    prompt_data = data_loader.get_prompt_with_scores(prompt_id, model_name)
    if not prompt_data:
        return jsonify({'error': 'Prompt or model not found'}), 404
    
    return jsonify(prompt_data)

@app.route('/api/stats')
def get_dashboard_stats():
    """API endpoint to get overall dashboard statistics"""
    stats = {
        'total_prompts': len(data_loader.prompt_battery),
        'total_models': len(data_loader.evaluation_results),
        'topics': data_loader.get_topics(),
        'model_accuracies': {}
    }
    
    # Calculate model accuracies
    for model_name, data in data_loader.evaluation_results.items():
        summary = data.get('summary', {})
        metrics = summary.get('metrics', {})
        stats['model_accuracies'][model_name] = metrics.get('overall_accuracy', 0)
    
    return jsonify(stats)

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    """Serve files from dataset directory"""
    return send_from_directory(str(DATASET_DIR), filename)

@app.route('/evaluation_results/<path:filename>')
def serve_evaluation_results(filename):
    """Serve files from evaluation_results directory"""
    return send_from_directory(str(EVALUATION_DIR), filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files from results directory"""
    return send_from_directory(str(RESULTS_DIR), filename)

@app.route('/api/final_results/sycophancy_scores')
def api_final_sycophancy_scores():
    """Return per-model sycophancy metrics from the combined run folder."""
    ldr = _get_final_loader()
    return jsonify({'run_path': str(ldr.base_dir), 'items': ldr.sycophancy_scores()})

@app.route('/api/final_results/sss_scores')
def api_final_sss_scores():
    """Return SSS scores table if present."""
    ldr = _get_final_loader()
    return jsonify({'run_path': str(ldr.base_dir), 'items': ldr.sss_scores()})

@app.route('/api/final_results/scored_rows')
def api_final_scored_rows():
    """Return scored rows with an optional limit parameter."""
    ldr = _get_final_loader()
    limit = request.args.get('limit', default=None, type=int)
    return jsonify({'run_path': str(ldr.base_dir), 'items': ldr.scored_rows(limit=limit)})

@app.route('/api/final_results/network')
def api_final_network():
    """Return kNN+MST network graph from similarity matrix for visualization."""
    try:
        k = request.args.get('k', default=6, type=int)
    except Exception:
        k = 6
    ldr = _get_final_loader()
    data = ldr.network_data(k=k)
    return jsonify(data)

@app.route('/api/final_results/network_png')
def api_final_network_png():
    """Return a PNG network plot rendered via Matplotlib pipeline (same as notebooks)."""
    knn_k = request.args.get('k', default=6, type=int)
    layout = request.args.get('layout', default='umap', type=str)
    leiden_res = request.args.get('resolution', default=1.0, type=float)
    umap_neighbors = request.args.get('umap_neighbors', default=25, type=int)
    umap_min_dist = request.args.get('umap_min_dist', default=0.08, type=float)
    seed = request.args.get('seed', default=42, type=int)
    bridge_threshold = request.args.get('bridge_threshold', default=0.5, type=float)
    ldr = _get_final_loader()
    buf = ldr.network_png(
        knn_k=knn_k,
        layout=layout,
        leiden_resolution=leiden_res,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        seed=seed,
        bridge_threshold=bridge_threshold,
    )
    if buf is None:
        return jsonify({"error": "Network image could not be generated."}), 500
    return send_file(buf, mimetype='image/png')

@app.route('/api/final_results/stance_elasticity')
def api_final_stance_elasticity():
    """Return per-model stance elasticity and topic dispersion metrics suitable for dashboard charting."""
    loader = _get_final_loader()
    min_n = int(request.args.get('min_n', 8))
    min_topics = int(request.args.get('min_topics', 6))
    result = loader.stance_elasticity_metrics(min_n_per_topic=min_n, min_topics=min_topics)
    return jsonify(result)

@app.route('/api/responses_combined')
def api_responses_combined():
    """Return combined responses data with scores for prompt explorer."""
    loader = _get_final_loader()
    responses_path = loader.base_dir / "responses_combined.json"
    scores_path = loader.base_dir / "scored_rows.csv"
    
    if not responses_path.exists():
        return jsonify({"error": "responses_combined.json not found"}), 404
    
    try:
        # Load responses
        with open(responses_path, 'r', encoding='utf-8') as f:
            responses_data = json.load(f)
        
        # Load scores if available
        scores_dict = {}
        if scores_path.exists():
            import pandas as pd
            scores_df = pd.read_csv(scores_path)
            # Create lookup dictionary: (model, prompt_id) -> scores
            for _, row in scores_df.iterrows():
                key = (row['model'], row['prompt_id'])
                scores_dict[key] = {
                    'endorse_stance': float(row['endorse_stance']) if pd.notna(row['endorse_stance']) else None,
                    'challenge': float(row['challenge']) if pd.notna(row['challenge']) else None,
                    'harm_validation': float(row['harm_validation']) if pd.notna(row['harm_validation']) else None,
                    'devil_advocate': float(row['devil_advocate']) if pd.notna(row['devil_advocate']) else None,
                    'flattery_density': float(row['flattery_density']) if pd.notna(row['flattery_density']) else None,
                    'intens_density': float(row['intens_density']) if pd.notna(row['intens_density']) else None,
                    'hedge_density': float(row['hedge_density']) if pd.notna(row['hedge_density']) else None,
                    'refusal_markers': float(row['refusal_markers']) if pd.notna(row['refusal_markers']) else None,
                    'safe_alt_markers': float(row['safe_alt_markers']) if pd.notna(row['safe_alt_markers']) else None,
                    'evasion_markers': float(row['evasion_markers']) if pd.notna(row['evasion_markers']) else None,
                    'caveat_in_open': float(row['caveat_in_open']) if pd.notna(row['caveat_in_open']) else None,
                    'pred_label': str(row['pred_label']) if pd.notna(row['pred_label']) else None
                }
        
        # Merge scores with responses
        for response in responses_data:
            key = (response['model'], response['prompt_id'])
            if key in scores_dict:
                response['scores'] = scores_dict[key]
            else:
                response['scores'] = None
        
        return jsonify(responses_data)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Failed to load responses with scores")
        return jsonify({"error": str(e)}), 500


