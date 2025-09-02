#!/usr/bin/env python3
"""
Dashboard server for LLM Judge Evaluation Results
Serves the HTML dashboard and provides API endpoints for evaluation data
"""

import os
import json
import pandas as pd
from flask import Flask, jsonify, send_from_directory, render_template_string, request, send_file
from flask_cors import CORS
import glob
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration (resolve paths relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = str(BASE_DIR / "dataset")
EVALUATION_DIR = str(BASE_DIR / "evaluation_results")
RESULTS_DIR = str(BASE_DIR / "results")
PROMPT_BATTERY_FILE = str(BASE_DIR / "dataset" / "prompt_battery.json")
FINAL_RESULTS_DIR = str(BASE_DIR / "results" / "combined_run_0c_1_1b")

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
            print(f"Warning: {PROMPT_BATTERY_FILE} not found")
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
                # Load detailed scores
                df = pd.read_csv(csv_file)
                # Replace NaN/NaT with None so JSON serialization stays standards-compliant
                df = df.where(pd.notna(df), None)
                
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
                        except Exception:
                            selected_summary = summary_files[-1]

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
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
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
                    'endorse_stance': prompt_scores.get('endorse_stance', 0),
                    'harm_validation': prompt_scores.get('harm_validation', 0),
                    'challenge': prompt_scores.get('challenge', 0),
                    'flattery_density': prompt_scores.get('flattery_density', 0),
                    'pred_label': prompt_scores.get('pred_label', 0),
                    'human_eval': prompt_scores.get('human_eval', ''),
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
        except Exception as e:
            print(f"[FinalResultsLoader] Error reading {path}: {e}")
            return []

    def sycophancy_scores(self):
        return self._load_csv("sycophancy_scores.csv")

    def sss_scores(self):
        return self._load_csv("sss_scores.csv")

    def scored_rows(self, limit: int | None = None):
        rows = self._load_csv("scored_rows.csv")
        if limit is not None:
            try:
                return rows[: int(limit)]
            except Exception:
                return rows
        return rows

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
            print(f"[FinalResultsLoader] Error loading network data: {e}")
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
            print(f"[FinalResultsLoader] MST failed: {e}")

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
            print(f"[FinalResultsLoader] UMAP failed, falling back to spring_layout: {e}")
            pos = nx.spring_layout(nx.Graph(list(edge_map.values())), seed=random_state)
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
            except Exception as e:
                print(f"[FinalResultsLoader] Leiden failed: {e}")
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
            print(f"[FinalResultsLoader] Import error for visualization pipeline: {e}")
            return None

        sim_path = self.base_dir / "similarity_matrix.npy"
        names_path = self.base_dir / "model_names.json"
        if not sim_path.exists() or not names_path.exists():
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


final_loader = FinalResultsLoader(FINAL_RESULTS_DIR)

@app.route('/')
def dashboard():
    """Serve the main dashboard HTML"""
    return send_from_directory(str(BASE_DIR), 'llm_judge_dashboard.html')

@app.route('/judge')
def judge_dashboard():
    """Serve the judge-only dashboard HTML"""
    return send_from_directory(str(BASE_DIR), 'llm_judge_dashboard.html')

@app.route('/final')
def final_results_dashboard():
    """Serve the final results + prompt explorer dashboard HTML"""
    return send_from_directory(str(BASE_DIR), 'final_results_dashboard.html')

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
    
    return jsonify(data_loader.evaluation_results[model_name])

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
    return send_from_directory(DATASET_DIR, filename)

@app.route('/evaluation_results/<path:filename>')
def serve_evaluation_results(filename):
    """Serve files from evaluation_results directory"""
    return send_from_directory(EVALUATION_DIR, filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files from results directory"""
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/api/final_results/sycophancy_scores')
def api_final_sycophancy_scores():
    """Return per-model sycophancy metrics from the combined run folder."""
    return jsonify({
        'run_path': str(FINAL_RESULTS_DIR),
        'items': final_loader.sycophancy_scores(),
    })

@app.route('/api/final_results/sss_scores')
def api_final_sss_scores():
    """Return SSS scores table if present."""
    return jsonify({
        'run_path': str(FINAL_RESULTS_DIR),
        'items': final_loader.sss_scores(),
    })

@app.route('/api/final_results/scored_rows')
def api_final_scored_rows():
    """Return scored rows with an optional limit parameter."""
    limit = request.args.get('limit', default=None, type=int)
    return jsonify({
        'run_path': str(FINAL_RESULTS_DIR),
        'items': final_loader.scored_rows(limit=limit),
    })

@app.route('/api/final_results/network')
def api_final_network():
    """Return kNN+MST network graph from similarity matrix for visualization."""
    try:
        k = request.args.get('k', default=6, type=int)
    except Exception:
        k = 6
    data = final_loader.network_data(k=k)
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
    buf = final_loader.network_png(
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

if __name__ == '__main__':
    print("Starting LLM Judge Dashboard Server...")
    print(f"Loaded {len(data_loader.prompt_battery)} prompts")
    print(f"Loaded {len(data_loader.evaluation_results)} model evaluations")
    print("Available models:", data_loader.get_model_list())
    print("Available topics:", data_loader.get_topics())
    print("Final results path:", FINAL_RESULTS_DIR)
    print("\nDashboard will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
