#!/usr/bin/env python3
"""
Dashboard server for LLM Judge Evaluation Results
Serves the HTML dashboard and provides API endpoints for evaluation data
"""

import os
import json
import pandas as pd
from flask import Flask, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import glob
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration (resolve paths relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = str(BASE_DIR / "dataset")
EVALUATION_DIR = str(BASE_DIR / "evaluation_results")
PROMPT_BATTERY_FILE = str(BASE_DIR / "dataset" / "prompt_battery.json")

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

@app.route('/')
def dashboard():
    """Serve the main dashboard HTML"""
    return send_from_directory(str(BASE_DIR), 'llm_judge_dashboard.html')

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

if __name__ == '__main__':
    print("Starting LLM Judge Dashboard Server...")
    print(f"Loaded {len(data_loader.prompt_battery)} prompts")
    print(f"Loaded {len(data_loader.evaluation_results)} model evaluations")
    print("Available models:", data_loader.get_model_list())
    print("Available topics:", data_loader.get_topics())
    print("\nDashboard will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
