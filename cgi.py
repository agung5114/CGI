import os
import json
import pickle
import random
import base64
import io
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from flask import Flask, render_template_string, jsonify, request, send_file
from PIL import Image, ImageOps

# --- Machine Learning Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# --- Optional: Advanced NLP & Vision ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

# ==============================================================================
# 1. BACKEND LOGIC (MODELS & DATA)
# ==============================================================================

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "disease_model.pkl"
CLIP_LOCAL_PATH = "local_clip_model"
GLOBAL_DATA = {}  # Store graph and synthetic data in memory

# --- Helper: Load/Train Models ---
def get_semantic_model():
    if HAS_BERT:
        if os.path.exists(CLIP_LOCAL_PATH):
            try:
                return SentenceTransformer(CLIP_LOCAL_PATH)
            except:
                pass
        try:
            # Download and save if not exists
            model = SentenceTransformer('clip-ViT-B-32')
            model.save(CLIP_LOCAL_PATH)
            return model
        except:
            return None
    return None

def get_disease_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

# Initialize Global Models
semantic_model = get_semantic_model()
disease_model_data = get_disease_model()

# --- Helper: Data Simulation ---
def generate_synthetic_data(n_claims=1500, fraud_ratio=0.05):
    np.random.seed(42)
    n_providers = 50
    n_patients = 200
    
    providers = [f"PRV-{i:03d}" for i in range(n_providers)]
    patients = [f"PAT-{i:03d}" for i in range(n_patients)]
    specialties = ['Cardiology', 'General Practice', 'Orthopedics', 'Dermatology', 'Neurology']
    prov_specialty = {p: np.random.choice(specialties) for p in providers}
    
    # Create Fraud Rings
    bad_providers = np.random.choice(providers, size=int(n_providers * 0.15), replace=False)
    bad_patients = np.random.choice(patients, size=int(n_patients * 0.15), replace=False)
    
    claims = []
    for i in range(n_claims):
        is_fraud = False
        if np.random.random() < fraud_ratio:
            prov = np.random.choice(bad_providers)
            pat = np.random.choice(bad_patients)
            amount = np.random.normal(loc=5000, scale=1500)
            is_fraud = True
        else:
            prov = np.random.choice(providers)
            pat = np.random.choice(patients)
            amount = np.random.normal(loc=800, scale=300)
        
        claims.append({
            'id': f"CLM-{i:04d}",
            'source': prov,
            'target': pat,
            'amount': round(max(50, amount), 2),
            'is_fraud': is_fraud,
            'specialty': prov_specialty[prov],
            'date': pd.Timestamp('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365), unit='D')
        })
    
    df = pd.DataFrame(claims)
    
    # Build Graph
    G = nx.Graph()
    for _, row in df.iterrows():
        # Add nodes
        G.add_node(row['source'], group='provider', fraud=row['source'] in bad_providers)
        G.add_node(row['target'], group='patient', fraud=row['target'] in bad_patients)
        # Add edge
        if G.has_edge(row['source'], row['target']):
            G[row['source']][row['target']]['weight'] += 1
        else:
            G.add_edge(row['source'], row['target'], weight=1)
            
    return df, G, list(bad_providers)

# Initialize Data
df_claims, G_network, bad_actors = generate_synthetic_data()

# ==============================================================================
# 2. API ROUTES
# ==============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def api_stats():
    total_claims = len(df_claims)
    fraud_claims = df_claims[df_claims['is_fraud'] == True]
    fraud_amt = fraud_claims['amount'].sum()
    return jsonify({
        'total_claims': total_claims,
        'fraud_claims': len(fraud_claims),
        'fraud_amount': f"${fraud_amt:,.2f}",
        'network_density': f"{nx.density(G_network):.4f}"
    })

@app.route('/api/graph_data')
def api_graph():
    fraud_reasons = [
        "High volume of high-value claims",
        "Abnormal patient sharing ratio",
        "Billing code frequency anomaly (Z-12)",
        "Geographic mismatch in claim origin",
        "Excessive daily throughput",
        "Cyclical claim patterns detected"
    ]

    degrees = sorted([d for n, d in G_network.degree()], reverse=True)
    threshold = degrees[min(len(degrees)-1, 150)] 
    nodes_to_keep = [n for n, d in G_network.degree() if d >= threshold]
    G_sub = G_network.subgraph(nodes_to_keep)
    
    nodes = []
    for n in G_sub.nodes():
        data = G_sub.nodes[n]
        is_fraud = data.get('fraud')
        
        if is_fraud:
            risk_score = 0.85 + (hash(n) % 15)/100
            reason = fraud_reasons[hash(n) % len(fraud_reasons)]
            status = "Suspicious"
            color = "#ef4444"
            size = 25
        else:
            risk_score = 0.05 + (hash(n) % 10)/100
            reason = "N/A"
            status = "Verified"
            color = "#3b82f6" if data['group'] == 'provider' else "#10b981"
            size = 15
            
        shape = "dot" if data['group'] == 'patient' else "diamond"
        
        nodes.append({
            'id': n, 
            'label': n, 
            'color': color, 
            'shape': shape, 
            'size': size, 
            'group': data['group'],
            'risk': f"{risk_score:.2f}",
            'reason': reason,
            'status': status,
            'connections': G_network.degree[n]
        })
        
    edges = []
    for u, v, d in G_sub.edges(data=True):
        edges.append({'from': u, 'to': v, 'value': d['weight']})
        
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/api/entity_details/<entity_id>')
def api_entity_details(entity_id):
    is_fraud = entity_id in bad_actors
    fraud_reasons = [
        "High volume of high-value claims",
        "Abnormal patient sharing ratio",
        "Billing code frequency anomaly (Z-12)",
        "Geographic mismatch in claim origin",
        "Excessive daily throughput",
        "Cyclical claim patterns detected"
    ]
    
    if is_fraud:
        risk_score = 0.85 + (hash(entity_id) % 15)/100
        reason = fraud_reasons[hash(entity_id) % len(fraud_reasons)]
        status = "Suspicious"
    else:
        risk_score = 0.05 + (hash(entity_id) % 10)/100
        reason = "Normal activity patterns detected."
        status = "Verified"

    if 'PRV' in entity_id:
        role = 'Healthcare Provider'
        related_claims = df_claims[df_claims['source'] == entity_id].copy()
    else:
        role = 'Beneficiary (Patient)'
        related_claims = df_claims[df_claims['target'] == entity_id].copy()

    related_claims['date'] = related_claims['date'].dt.strftime('%Y-%m-%d')
    claims_list = related_claims.head(50).to_dict(orient='records')

    return jsonify({
        'id': entity_id,
        'role': role,
        'status': status,
        'risk_score': f"{risk_score:.2f}",
        'reason': reason,
        'total_amount': f"${related_claims['amount'].sum():,.2f}",
        'claim_count': len(related_claims),
        'claims': claims_list
    })

@app.route('/api/alerts')
def api_alerts():
    alerts = [
        {"id": 1, "type": "fraud", "message": "High-risk collusion detected in Orthopedics network", "time": "2 mins ago", "severity": "critical"},
        {"id": 2, "type": "system", "message": "Model retraining completed successfully", "time": "1 hour ago", "severity": "info"},
        {"id": 3, "type": "audit", "message": "User 'Analyst_01' flagged Provider PRV-042", "time": "3 hours ago", "severity": "medium"},
        {"id": 4, "type": "fraud", "message": "Duplicate claim batch detected from API endpoint", "time": "5 hours ago", "severity": "high"},
    ]
    return jsonify(alerts)

@app.route('/api/predict_disease', methods=['POST'])
def api_predict_disease():
    data = request.json
    symptoms = data.get('symptoms', '')
    if not disease_model_data: return jsonify({'error': 'Model not loaded'}), 400
    model = disease_model_data['model']
    m_type = disease_model_data.get('type', 'tfidf')
    try:
        if m_type == 'bert' and HAS_BERT and semantic_model:
            vec = semantic_model.encode([symptoms])
            pred = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
        else:
            pred = model.predict([symptoms])[0]
            probs = model.predict_proba([symptoms])[0]
        class_idx = np.where(model.classes_ == pred)[0][0]
        return jsonify({'prediction': pred, 'confidence': float(probs[class_idx])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_fraud', methods=['POST'])
def api_check_fraud():
    data = request.json
    symptoms = data.get('symptoms')
    claimed = data.get('claimed_disease')
    if not disease_model_data: return jsonify({'error': 'Disease model missing'}), 400
    model = disease_model_data['model']
    m_type = disease_model_data.get('type', 'tfidf')
    if m_type == 'bert' and HAS_BERT and semantic_model:
        vec = semantic_model.encode([symptoms])
        pred = model.predict(vec)[0]
    else:
        pred = model.predict([symptoms])[0]
    if HAS_BERT and semantic_model:
        vec_sym = semantic_model.encode([symptoms])
        vec_clm = semantic_model.encode([claimed])
        vec_pred = semantic_model.encode([pred])
        sim_sym = float(util.cos_sim(vec_sym, vec_clm)[0][0])
        sim_pred = float(util.cos_sim(vec_pred, vec_clm)[0][0])
    else:
        sim_sym = 0.5; sim_pred = 0.5
    risk = 1.0 - ((sim_sym + sim_pred) / 2)
    return jsonify({'ai_diagnosis': pred, 'similarity_symptoms': sim_sym, 'similarity_prediction': sim_pred, 'risk_score': risk})

@app.route('/api/verify_signature', methods=['POST'])
def api_verify_sig():
    if 'ref' not in request.files or 'que' not in request.files: return jsonify({'error': 'Missing files'}), 400
    if not HAS_BERT or not semantic_model: return jsonify({'error': 'Semantic model unavailable'}), 500
    try:
        ref_img = Image.open(request.files['ref'])
        que_img = Image.open(request.files['que'])
        emb1 = semantic_model.encode(ref_img)
        emb2 = semantic_model.encode(que_img)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# 3. FRONTEND TEMPLATE
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CGI | Claim Guard Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>tailwind.config = { darkMode: 'class' }</script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
    
    <style>
        body { font-family: 'Inter', sans-serif; }
        .sidebar-link { transition: all 0.3s; cursor: pointer; }
        .sidebar-link:hover, .active-link { background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; }
        .dark .sidebar-link:hover, .dark .active-link { background-color: #1e293b; }
        #network-graph { height: 500px; border-radius: 0.5rem; border: 1px solid; }
        .custom-scrollbar::-webkit-scrollbar { width: 8px; height: 8px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #64748b; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 12px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid; cursor: pointer; user-select: none; }
        th:hover { background-color: rgba(255,255,255,0.05); }
        td { padding: 12px; border-bottom: 1px solid; font-size: 0.875rem; }
        tr:last-child td { border-bottom: none; }
        .selected-row { background-color: rgba(59, 130, 246, 0.2) !important; border-left: 4px solid #3b82f6; }
        /* Toggle Switch */
        .toggle-checkbox:checked { right: 0; border-color: #3b82f6; }
        .toggle-checkbox:checked + .toggle-label { background-color: #3b82f6; }
    </style>
</head>
<body class="flex h-screen overflow-hidden bg-slate-100 dark:bg-slate-950 text-slate-800 dark:text-slate-200 transition-colors duration-300">

    <!-- Sidebar -->
    <div class="w-64 bg-white dark:bg-slate-900 flex flex-col border-r border-slate-200 dark:border-slate-800 flex-shrink-0 transition-colors duration-300">
        <div class="p-6 flex items-center gap-3">
            <div class="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/30">🛡️</div>
            <div>
                <h1 class="font-bold text-lg tracking-wide text-slate-900 dark:text-white">CGI</h1>
                <p class="text-xs text-slate-500 dark:text-slate-400">Claim Guard Intelligence</p>
            </div>
        </div>
        
        <nav class="flex-1 mt-6 overflow-y-auto custom-scrollbar">
            <div class="px-4 mb-2 text-xs font-semibold text-slate-500 uppercase">Analytics</div>
            <a onclick="showPage('dashboard')" id="nav-dashboard" class="sidebar-link active-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                <i class="fas fa-chart-pie w-5"></i> Dashboard
            </a>
            <a onclick="showPage('graph')" id="nav-graph" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                <i class="fas fa-project-diagram w-5"></i> Graph Intelligence
            </a>
            
            <div class="px-4 mt-6 mb-2 text-xs font-semibold text-slate-500 uppercase">AI Tools</div>
            <a onclick="showPage('auditor')" id="nav-auditor" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                <i class="fas fa-brain w-5"></i> Semantic Matching
            </a>
            <a onclick="showPage('signature')" id="nav-signature" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                <i class="fas fa-file-signature w-5"></i> CoVision Verification
            </a>

            <!-- Admin Section -->
            <div id="admin-section" class="hidden">
                <div class="px-4 mt-6 mb-2 text-xs font-semibold text-slate-500 uppercase">Administration</div>
                <a onclick="showPage('rules')" id="nav-rules" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                    <i class="fas fa-gavel w-5"></i> Rule Engine
                </a>
                <a onclick="showPage('reports')" id="nav-reports" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                    <i class="fas fa-file-alt w-5"></i> Case Reports
                </a>
                <a onclick="showPage('audit')" id="nav-audit" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                    <i class="fas fa-history w-5"></i> System Audit Log
                </a>
                <a onclick="showPage('settings')" id="nav-settings" class="sidebar-link flex items-center gap-3 px-6 py-3 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-white">
                    <i class="fas fa-cogs w-5"></i> System Settings
                </a>
            </div>
        </nav>
        
        <div class="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
            <div class="flex items-center gap-3">
                <div id="user-avatar" class="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white ring-2 ring-white dark:ring-slate-800">AD</div>
                <div class="flex-1 min-w-0">
                    <p id="user-name" class="font-medium text-sm text-slate-900 dark:text-white truncate">Admin User</p>
                    <p id="user-role" class="text-xs text-slate-500 dark:text-slate-400 truncate">Senior Investigator</p>
                </div>
                <span id="user-role-badge" class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">ADMIN</span>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 overflow-auto bg-slate-50 dark:bg-slate-950 relative custom-scrollbar transition-colors duration-300">
        <header class="bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 sticky top-0 z-20 px-8 py-4 flex justify-between items-center">
            <div>
                <h2 id="page-title" class="text-xl font-bold text-slate-800 dark:text-white">Executive Overview</h2>
                <p class="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Last updated: <span id="last-update">Just now</span></p>
            </div>
            <div class="flex gap-4 items-center">
                <button onclick="toggleTheme()" class="p-2 text-slate-500 hover:text-blue-600 dark:text-slate-400 dark:hover:text-white transition-colors rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800">
                    <i class="fas fa-sun hidden dark:inline"></i><i class="fas fa-moon inline dark:hidden"></i>
                </button>
                <button onclick="showPage('alerts')" class="relative p-2 text-slate-500 hover:text-blue-600 dark:text-slate-400 dark:hover:text-white transition-colors rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800">
                    <i class="fas fa-bell"></i>
                    <span class="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full animate-pulse"></span>
                </button>
            </div>
        </header>

        <!-- RULE ENGINE PAGE (NEW) -->
        <main id="page-rules" class="p-8 hidden space-y-6 pb-20">
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                <div class="flex justify-between items-center mb-6">
                    <h3 class="text-lg font-bold text-slate-900 dark:text-white">Active Fraud Detection Rules</h3>
                    <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium"><i class="fas fa-plus mr-2"></i> Add New Rule</button>
                </div>
                
                <table class="w-full text-left text-sm">
                    <thead class="bg-slate-50 dark:bg-slate-800/50 text-xs uppercase text-slate-500 dark:text-slate-400">
                        <tr>
                            <th class="px-4 py-3">Rule Name</th>
                            <th class="px-4 py-3">Logic</th>
                            <th class="px-4 py-3">Severity</th>
                            <th class="px-4 py-3">Status</th>
                            <th class="px-4 py-3">Actions</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-200 dark:divide-slate-800 text-slate-700 dark:text-slate-300">
                        <tr>
                            <td class="px-4 py-4 font-medium">High Value Claim Anomaly</td>
                            <td class="px-4 py-4">Amount > $50,000 AND Frequency > 3/day</td>
                            <td class="px-4 py-4"><span class="px-2 py-1 rounded text-xs font-bold bg-rose-100 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400">CRITICAL</span></td>
                            <td class="px-4 py-4 text-emerald-500 font-bold">Active</td>
                            <td class="px-4 py-4"><button class="text-slate-400 hover:text-blue-500"><i class="fas fa-edit"></i></button></td>
                        </tr>
                        <tr>
                            <td class="px-4 py-4 font-medium">Network Cyclical Pattern</td>
                            <td class="px-4 py-4">Graph Cycle Length < 4 AND Weight > 10</td>
                            <td class="px-4 py-4"><span class="px-2 py-1 rounded text-xs font-bold bg-orange-100 text-orange-600 dark:bg-orange-900/30 dark:text-orange-400">HIGH</span></td>
                            <td class="px-4 py-4 text-emerald-500 font-bold">Active</td>
                            <td class="px-4 py-4"><button class="text-slate-400 hover:text-blue-500"><i class="fas fa-edit"></i></button></td>
                        </tr>
                        <tr>
                            <td class="px-4 py-4 font-medium">Geographic Mismatch</td>
                            <td class="px-4 py-4">Provider Loc != Patient Loc (>500 miles)</td>
                            <td class="px-4 py-4"><span class="px-2 py-1 rounded text-xs font-bold bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">MEDIUM</span></td>
                            <td class="px-4 py-4 text-slate-400">Disabled</td>
                            <td class="px-4 py-4"><button class="text-slate-400 hover:text-blue-500"><i class="fas fa-edit"></i></button></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </main>

        <!-- SETTINGS PAGE (NEW) -->
        <main id="page-settings" class="p-8 hidden space-y-6 pb-20">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <!-- General Config -->
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <h3 class="text-lg font-bold text-slate-900 dark:text-white mb-4">System Configuration</h3>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-xs font-bold text-slate-500 uppercase mb-1">Fraud Threshold Sensitivity</label>
                            <input type="range" min="1" max="100" value="75" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer dark:bg-slate-700">
                            <div class="flex justify-between text-xs text-slate-500 mt-1"><span>Low</span><span>High (Aggressive)</span></div>
                        </div>
                        <div class="flex items-center justify-between py-2 border-b border-slate-200 dark:border-slate-800">
                            <span class="text-sm text-slate-700 dark:text-slate-300">Enable Auto-Retraining</span>
                            <div class="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                                <input type="checkbox" name="toggle" id="toggle1" class="toggle-checkbox absolute block w-5 h-5 rounded-full bg-white border-4 appearance-none cursor-pointer" checked/>
                                <label for="toggle1" class="toggle-label block overflow-hidden h-5 rounded-full bg-blue-500 cursor-pointer"></label>
                            </div>
                        </div>
                        <div class="flex items-center justify-between py-2">
                            <span class="text-sm text-slate-700 dark:text-slate-300">Dark Mode Default</span>
                            <div class="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                                <input type="checkbox" name="toggle" id="toggle2" class="toggle-checkbox absolute block w-5 h-5 rounded-full bg-white border-4 appearance-none cursor-pointer" checked/>
                                <label for="toggle2" class="toggle-label block overflow-hidden h-5 rounded-full bg-blue-500 cursor-pointer"></label>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- API Keys -->
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <h3 class="text-lg font-bold text-slate-900 dark:text-white mb-4">Integration Keys</h3>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-xs font-bold text-slate-500 uppercase mb-1">Hugging Face API Token</label>
                            <input type="password" value="hf_xxxxxxxxxxxxxxxxx" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg p-2 text-sm text-slate-900 dark:text-white" readonly>
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-slate-500 uppercase mb-1">Database Connection String</label>
                            <input type="password" value="postgresql://admin:******@aws-rds-01" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg p-2 text-sm text-slate-900 dark:text-white" readonly>
                        </div>
                        <div class="pt-4">
                            <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium">Update Keys</button>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- EXISTING MODULES (DASHBOARD, GRAPH, REPORTS, AUDIT, ALERT, AUDITOR, SIGNATURE, DETAIL) -->
        <!-- DASHBOARD PAGE -->
        <main id="page-dashboard" class="p-8 space-y-6 pb-20">
            <!-- Stats Row -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <div class="text-slate-500 dark:text-slate-400 text-sm font-medium mb-1">Total Claims</div>
                    <div class="text-3xl font-bold text-slate-900 dark:text-white" id="stat-total">...</div>
                    <div class="text-emerald-500 dark:text-emerald-400 text-sm mt-2"><i class="fas fa-arrow-up"></i> 12% vs last week</div>
                </div>
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <div class="text-slate-500 dark:text-slate-400 text-sm font-medium mb-1">Fraud Detected</div>
                    <div class="text-3xl font-bold text-rose-500" id="stat-fraud">...</div>
                </div>
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <div class="text-slate-500 dark:text-slate-400 text-sm font-medium mb-1">Prevented Loss</div>
                    <div class="text-3xl font-bold text-emerald-500" id="stat-amount">...</div>
                </div>
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <div class="text-slate-500 dark:text-slate-400 text-sm font-medium mb-1">Network Density</div>
                    <div class="text-3xl font-bold text-blue-500" id="stat-density">...</div>
                </div>
            </div>
            <!-- Charts -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 col-span-2 relative h-96 shadow-sm flex flex-col">
                    <h3 class="text-lg font-medium mb-4 text-slate-900 dark:text-white">Fraud Trends</h3>
                    <div class="relative flex-1 w-full overflow-hidden"><canvas id="trendChart"></canvas></div>
                </div>
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 relative h-96 shadow-sm flex flex-col">
                    <h3 class="text-lg font-medium mb-4 text-slate-900 dark:text-white">Risk by Specialty</h3>
                    <div class="relative flex-1 w-full overflow-hidden"><canvas id="pieChart"></canvas></div>
                </div>
            </div>
        </main>

        <!-- GRAPH PAGE -->
        <main id="page-graph" class="p-8 hidden pb-20">
            <div class="flex justify-between items-center mb-4">
                <div class="flex gap-2">
                    <span class="px-3 py-1 rounded-full bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 text-xs font-bold border border-rose-200 dark:border-rose-500/30">Fraud Ring</span>
                    <span class="px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-bold border border-blue-200 dark:border-blue-500/30">Provider</span>
                    <span class="px-3 py-1 rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-200 dark:border-emerald-500/30">Patient</span>
                </div>
                <button onclick="loadGraph()" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"><i class="fas fa-sync-alt mr-2"></i> Refresh</button>
            </div>
            <div id="network-graph" class="w-full shadow-inner relative bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800"></div>
            
            <!-- ENTITY TABLE -->
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-sm mt-6">
                <div class="p-6 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
                    <h3 class="text-lg font-bold text-slate-900 dark:text-white">Network Entity Intelligence</h3>
                    <div class="flex gap-4">
                         <!-- Search Input -->
                        <div class="relative">
                            <input type="text" id="table-search" onkeyup="filterTable()" placeholder="Filter ID or Status..." class="bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg pl-8 pr-4 py-1 text-sm focus:outline-none focus:border-blue-500 text-slate-900 dark:text-white">
                            <i class="fas fa-search absolute left-2.5 top-2 text-slate-400 text-xs"></i>
                        </div>
                        <!-- Action Button -->
                        <button id="btn-view-detail" onclick="viewEntityDetail()" class="bg-slate-200 dark:bg-slate-700 text-slate-400 px-4 py-1 rounded-lg text-sm font-medium cursor-not-allowed" disabled>
                            View Intelligence Dossier <i class="fas fa-arrow-right ml-1"></i>
                        </button>
                    </div>
                </div>
                <div class="overflow-x-auto max-h-96 custom-scrollbar">
                    <table id="entity-table" class="text-slate-700 dark:text-slate-300 w-full">
                        <thead class="sticky top-0 z-10 bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                            <tr>
                                <th onclick="sortTable(0)">ID <i class="fas fa-sort ml-1"></i></th>
                                <th onclick="sortTable(1)">Type <i class="fas fa-sort ml-1"></i></th>
                                <th onclick="sortTable(2)">Status <i class="fas fa-sort ml-1"></i></th>
                                <th onclick="sortTable(3)">Risk Score <i class="fas fa-sort ml-1"></i></th>
                                <th onclick="sortTable(4)">Connections <i class="fas fa-sort ml-1"></i></th>
                                <th>Flag Reason</th>
                            </tr>
                        </thead>
                        <tbody id="entity-table-body" class="divide-y divide-slate-200 dark:divide-slate-800"></tbody>
                    </table>
                </div>
            </div>
        </main>

        <!-- DETAIL PAGE -->
        <main id="page-detail" class="p-8 hidden space-y-6 pb-20">
            <button onclick="showPage('graph')" class="text-slate-500 hover:text-blue-500 mb-4 flex items-center gap-2"><i class="fas fa-arrow-left"></i> Back to Network Graph</button>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm col-span-1">
                     <div class="flex items-center gap-4 mb-4">
                        <div id="detail-icon" class="w-16 h-16 rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center text-3xl"></div>
                        <div><h2 id="detail-id" class="text-2xl font-bold text-slate-900 dark:text-white">...</h2><p id="detail-role" class="text-slate-500 text-sm">...</p></div>
                     </div>
                     <div id="detail-status-badge" class="inline-block px-3 py-1 rounded-full text-xs font-bold mb-4"></div>
                     <div class="grid grid-cols-2 gap-4 border-t border-slate-200 dark:border-slate-700 pt-4">
                        <div><p class="text-xs text-slate-500">Total Claims</p><p id="detail-claims-count" class="font-bold text-slate-900 dark:text-white text-lg">...</p></div>
                         <div><p class="text-xs text-slate-500">Total Volume</p><p id="detail-claims-amt" class="font-bold text-slate-900 dark:text-white text-lg">...</p></div>
                     </div>
                </div>
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm col-span-2 relative overflow-hidden">
                    <div class="absolute top-0 right-0 p-4 opacity-10"><i class="fas fa-exclamation-triangle text-9xl text-rose-500"></i></div>
                    <h3 class="text-lg font-bold text-slate-900 dark:text-white mb-4">AI Risk Analysis</h3>
                    <div class="flex items-end gap-2 mb-2"><span class="text-4xl font-black text-rose-500" id="detail-risk">0.00</span><span class="text-sm text-slate-500 mb-1">/ 1.00 Risk Score</span></div>
                    <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mb-6"><div id="detail-risk-bar" class="bg-rose-500 h-2 rounded-full" style="width: 0%"></div></div>
                    <div><p class="text-xs text-slate-500 uppercase font-bold">Primary Flag Reason</p><p id="detail-reason" class="text-slate-700 dark:text-slate-300 mt-1 bg-slate-100 dark:bg-slate-800 p-3 rounded-lg border-l-4 border-rose-500">...</p></div>
                </div>
            </div>
             <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-sm">
                <div class="p-6 border-b border-slate-200 dark:border-slate-800"><h3 class="text-lg font-bold text-slate-900 dark:text-white">Transaction History</h3></div>
                <div class="overflow-x-auto">
                    <table class="text-slate-700 dark:text-slate-300 w-full text-sm">
                        <thead class="bg-slate-50 dark:bg-slate-900 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                            <tr><th class="px-6 py-3">Claim ID</th><th class="px-6 py-3">Date</th><th class="px-6 py-3">Counterparty</th><th class="px-6 py-3">Specialty</th><th class="px-6 py-3 text-right">Amount</th><th class="px-6 py-3 text-center">Flag</th></tr>
                        </thead>
                        <tbody id="detail-claims-body" class="divide-y divide-slate-200 dark:divide-slate-800"></tbody>
                    </table>
                </div>
             </div>
        </main>

        <!-- AI AUDITOR PAGE -->
        <main id="page-auditor" class="p-8 hidden space-y-6 pb-20">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <h3 class="text-lg font-bold text-blue-500 dark:text-blue-400 mb-4"><i class="fas fa-stethoscope mr-2"></i> Symptom and Disease Matching</h3>
                    <div><label class="text-xs text-slate-500 uppercase">We use LLM Model: "clip-ViT-B-3" and symptom_to_disease datasets (kaggle) for machine-learning training</label></div>
                    <textarea id="symptom-input" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg p-4 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 h-32 transition-colors" placeholder="Patient complains of severe chest pain radiating to left arm..."></textarea>
                    <div class="flex justify-end mt-4"><button onclick="predictDisease()" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">Analyze Symptom</button></div>
                    <div id="prediction-result" class="mt-6 hidden p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
                        <div class="text-xs text-slate-500 uppercase mb-1">Predicted Condition</div><div class="text-xl font-bold text-slate-900 dark:text-white" id="pred-text">...</div><div class="mt-2 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2"><div id="pred-bar" class="bg-blue-500 h-2 rounded-full" style="width: 0%"></div></div><div class="text-right text-xs text-blue-500 dark:text-blue-400 mt-1" id="pred-conf">0%</div>
                    </div>
                </div>
                <!-- FRAUD RESULT SECTION -->
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm">
                    <h3 class="text-lg font-bold text-rose-500 dark:text-rose-400 mb-4"><i class="fas fa-shield-alt mr-2"></i> Semantic Fraud Check</h3>
                    <div class="space-y-4">
                        <div><label class="text-xs text-slate-500 uppercase">Provider Claimed Disease</label><input type="text" id="claim-input" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg p-3 text-sm text-slate-900 dark:text-white mt-1 transition-colors" placeholder="e.g. Heart Attack"></div>
                        <button onclick="checkFraud()" class="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-bold transition-colors">Calculate Risk Score</button>
                    </div>
                    <div id="fraud-result" class="mt-6 hidden">
                        <div class="grid grid-cols-2 gap-4 mb-4">
                            <div class="flex flex-col items-center p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
                                <div class="relative w-16 h-16 mb-2"><svg class="w-full h-full transform -rotate-90"><circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="4" fill="transparent" class="text-slate-200 dark:text-slate-700" /><circle id="circle-sim-sym" cx="32" cy="32" r="28" stroke="currentColor" stroke-width="4" fill="transparent" stroke-dasharray="175.9" stroke-dashoffset="175.9" class="text-blue-500" /></svg><div class="absolute top-0 left-0 w-full h-full flex items-center justify-center text-xs font-bold text-slate-700 dark:text-slate-200" id="text-sim-sym">0%</div></div>
                                <div class="text-[10px] uppercase text-slate-500 text-center font-bold">Symptom Match</div>
                            </div>
                            <div class="flex flex-col items-center p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
                                <div class="relative w-16 h-16 mb-2"><svg class="w-full h-full transform -rotate-90"><circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="4" fill="transparent" class="text-slate-200 dark:text-slate-700" /><circle id="circle-sim-pred" cx="32" cy="32" r="28" stroke="currentColor" stroke-width="4" fill="transparent" stroke-dasharray="175.9" stroke-dashoffset="175.9" class="text-purple-500" /></svg><div class="absolute top-0 left-0 w-full h-full flex items-center justify-center text-xs font-bold text-slate-700 dark:text-slate-200" id="text-sim-pred">0%</div></div>
                                <div class="text-[10px] uppercase text-slate-500 text-center font-bold">Prediction Match</div>
                            </div>
                        </div>
                        <div class="text-center p-4 bg-slate-100 dark:bg-slate-800 rounded-lg border border-rose-500/30 mb-4"><div class="text-xs text-rose-500 dark:text-rose-400 uppercase font-bold mb-1">Composite Risk Score</div><div class="text-3xl font-black text-rose-600 dark:text-rose-500" id="score-risk">0.00</div></div>
                        <div id="risk-badge" class="w-full py-2 text-center font-bold rounded-lg"></div>
                    </div>
                </div>
            </div>
        </main>

        <!-- SIGNATURE PAGE -->
        <main id="page-signature" class="p-8 hidden pb-20">
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl max-w-4xl mx-auto p-8 shadow-sm">
                <h3 class="text-xl font-bold text-purple-500 dark:text-purple-400 mb-2"><i class="fas fa-fingerprint mr-2"></i> Documents and Biometric Verification</h3>
                <p class="text-slate-500 dark:text-slate-400 mb-8">Compare documents or images using AI.</p>
                <div><label class="text-xs text-slate-500 uppercase">We use LLM Model: "clip-ViT-B-3" for image recognition and similarity scoring</label></div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-6 text-center hover:border-purple-500 transition cursor-pointer relative bg-slate-50 dark:bg-slate-800/50">
                        <input type="file" id="file-ref" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" onchange="previewImage(this, 'img-ref')">
                        <div id="preview-ref-box"><i class="fas fa-id-card text-4xl text-slate-400 dark:text-slate-600 mb-2"></i><p class="text-sm text-slate-500 dark:text-slate-400">Upload Reference</p></div>
                        <img id="img-ref" class="hidden max-h-40 mx-auto mt-2 rounded shadow">
                    </div>
                    <div>
                        <div class="flex mb-4 bg-slate-200 dark:bg-slate-800 rounded-lg p-1">
                            <button onclick="setCamMode('file')" id="tab-file" class="flex-1 py-1 text-xs font-bold rounded bg-white dark:bg-slate-700 shadow text-slate-800 dark:text-white">Upload File</button>
                            <button onclick="setCamMode('cam')" id="tab-cam" class="flex-1 py-1 text-xs font-bold rounded text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white">Use Camera</button>
                        </div>
                        <div id="mode-file" class="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-6 text-center hover:border-purple-500 transition cursor-pointer relative bg-slate-50 dark:bg-slate-800/50 h-48 flex flex-col justify-center">
                            <input type="file" id="file-que" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" onchange="previewImage(this, 'img-que')">
                            <div id="preview-que-box"><i class="fas fa-file-signature text-4xl text-slate-400 dark:text-slate-600 mb-2"></i><p class="text-sm text-slate-500 dark:text-slate-400">Upload Questioned</p></div>
                            <img id="img-que" class="hidden max-h-40 mx-auto mt-2 rounded shadow z-10 pointer-events-none">
                        </div>
                        <div id="mode-cam" class="hidden relative rounded-xl overflow-hidden bg-black h-48 border-2 border-slate-600">
                            <video id="camera-stream" autoplay playsinline class="w-full h-full object-cover"></video>
                            <canvas id="camera-canvas" class="hidden"></canvas>
                            <div class="absolute bottom-2 w-full text-center"><button onclick="capturePhoto()" class="bg-red-500 hover:bg-red-600 text-white rounded-full p-3 shadow-lg"><i class="fas fa-camera"></i></button></div>
                            <img id="captured-img" class="hidden absolute inset-0 w-full h-full object-cover z-20">
                        </div>
                        <button id="retake-btn" onclick="startCamera()" class="hidden mt-2 text-xs text-blue-500 underline w-full text-center">Retake Photo</button>
                    </div>
                </div>
                <div class="mt-8 text-center">
                    <button onclick="verifySignature()" class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-full font-bold text-lg shadow-lg shadow-blue-500/30 transition-colors">Run AI Verification</button>
                </div>
                <div id="sig-result" class="mt-8 hidden text-center">
                    <div class="text-4xl font-black mb-2 text-slate-900 dark:text-white" id="sig-score">98%</div>
                    <div class="text-sm uppercase tracking-widest text-slate-500 mb-4">Match Probability</div>
                    <div id="sig-verdict" class="inline-block px-6 py-2 rounded-full font-bold">GENUINE</div>
                </div>
            </div>
        </main>
        
        <!-- ALERTS PAGE -->
        <main id="page-alerts" class="p-8 hidden space-y-6 pb-20">
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-sm">
                <div class="p-6 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center"><h3 class="text-lg font-bold text-slate-900 dark:text-white">System Notifications</h3><button class="text-sm text-blue-500 hover:underline">Mark all as read</button></div>
                <div id="alerts-container" class="divide-y divide-slate-200 dark:divide-slate-800"></div>
            </div>
        </main>

        <!-- AUDIT LOG -->
        <main id="page-audit" class="p-8 hidden space-y-6 pb-20">
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-sm">
                <div class="p-6 border-b border-slate-200 dark:border-slate-800"><h3 class="text-lg font-bold text-slate-900 dark:text-white">System Audit Log</h3><p class="text-sm text-slate-500">Immutable record of system activities.</p></div>
                <table class="w-full text-left text-sm text-slate-600 dark:text-slate-400">
                    <thead class="bg-slate-50 dark:bg-slate-800/50 text-xs uppercase"><tr><th class="px-6 py-3">Timestamp</th><th class="px-6 py-3">User</th><th class="px-6 py-3">Action</th><th class="px-6 py-3">Module</th><th class="px-6 py-3">Status</th></tr></thead>
                    <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                        <tr><td class="px-6 py-4">2024-03-15 10:42:12</td><td class="px-6 py-4 font-medium text-slate-900 dark:text-white">Admin User</td><td class="px-6 py-4">Retrained Disease Model</td><td class="px-6 py-4">AI Core</td><td class="px-6 py-4 text-emerald-500">Success</td></tr>
                        <tr><td class="px-6 py-4">2024-03-15 09:15:00</td><td class="px-6 py-4 font-medium text-slate-900 dark:text-white">System</td><td class="px-6 py-4">Daily Data Ingestion</td><td class="px-6 py-4">ETL Pipeline</td><td class="px-6 py-4 text-emerald-500">Success</td></tr>
                        <tr><td class="px-6 py-4">2024-03-14 16:20:33</td><td class="px-6 py-4 font-medium text-slate-900 dark:text-white">Analyst_04</td><td class="px-6 py-4">Flagged Claim #99281</td><td class="px-6 py-4">Graph Intel</td><td class="px-6 py-4 text-amber-500">Pending Review</td></tr>
                    </tbody>
                </table>
            </div>
        </main>

        <!-- REPORTS PAGE -->
        <main id="page-reports" class="p-8 hidden space-y-6 pb-20">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm hover:border-blue-500 transition cursor-pointer group">
                    <div class="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center text-blue-600 dark:text-blue-400 mb-4 group-hover:scale-110 transition"><i class="fas fa-file-pdf text-xl"></i></div>
                    <h3 class="font-bold text-lg text-slate-900 dark:text-white">Daily Fraud Summary</h3>
                    <p class="text-sm text-slate-500 mt-2">Automated PDF report of all high-risk flags from the last 24 hours.</p><button class="mt-4 text-sm text-blue-600 dark:text-blue-400 font-medium hover:underline">Download PDF</button>
                </div>
                <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm hover:border-purple-500 transition cursor-pointer group">
                    <div class="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center text-purple-600 dark:text-purple-400 mb-4 group-hover:scale-110 transition"><i class="fas fa-network-wired text-xl"></i></div>
                    <h3 class="font-bold text-lg text-slate-900 dark:text-white">Network Topology Export</h3>
                    <p class="text-sm text-slate-500 mt-2">Export current graph state (nodes/edges) as JSON/GEXF for external auditing.</p><button class="mt-4 text-sm text-purple-600 dark:text-purple-400 font-medium hover:underline">Export Data</button>
                </div>
            </div>
        </main>
    </div>

<script>
    // --- State Management ---
    const APP_STATE = { currentUser: { name: 'Administrator', role: 'Administrator', id: 'AD' }, alerts: [] };

    function toggleTheme() { document.documentElement.classList.toggle('dark'); }
    
    function showPage(pageId) {
        document.querySelectorAll('main').forEach(el => el.classList.add('hidden'));
        document.getElementById('page-' + pageId).classList.remove('hidden');
        document.querySelectorAll('.sidebar-link').forEach(el => el.classList.remove('active-link'));
        const navLink = document.getElementById('nav-' + (pageId=='detail' ? 'graph' : pageId));
        if(navLink) navLink.classList.add('active-link');
        const titles = {
            'dashboard': 'Executive Overview', 'graph': 'Graph Intelligence', 'detail': 'Entity Intelligence Dossier', 
            'auditor': 'AI Semantic Matching', 'signature': 'Computer Vision Verification', 'alerts': 'Alert Center', 'reports': 'Report Generator',
            'audit': 'System Audit Log', 'rules': 'Rule Engine', 'settings': 'System Settings'
        };
        document.getElementById('page-title').innerText = titles[pageId] || 'CGI PRO';
        if(pageId === 'graph') loadGraph();
        if(pageId === 'alerts') renderAlerts();
    }

    async function renderAlerts() {
        const container = document.getElementById('alerts-container');
        container.innerHTML = '';
        if(APP_STATE.alerts.length === 0) { const res = await fetch('/api/alerts'); APP_STATE.alerts = await res.json(); }
        APP_STATE.alerts.forEach(alert => {
            const color = alert.severity === 'critical' ? 'bg-rose-500' : alert.severity === 'high' ? 'bg-orange-500' : 'bg-blue-500';
            const item = document.createElement('div');
            item.className = "p-4 flex items-start gap-4 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition";
            item.innerHTML = `<div class="w-2 h-2 mt-2 rounded-full ${color}"></div><div class="flex-1"><p class="text-sm font-medium text-slate-900 dark:text-white">${alert.message}</p><div class="flex items-center gap-2 mt-1 text-xs text-slate-500"><span class="uppercase font-bold tracking-wider">${alert.type}</span><span>•</span><span>${alert.time}</span></div></div>`;
            container.appendChild(item);
        });
    }

    function initCharts() {
        const chartGridColor = 'rgba(148, 163, 184, 0.1)';
        try {
            new Chart(document.getElementById('trendChart').getContext('2d'), { type: 'line', data: { labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], datasets: [{ label: 'Fraud Attempts', data: [12, 19, 3, 5, 2, 3, 15], borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.1)', fill: true, tension: 0.4 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#94a3b8' } }, y: { grid: { color: chartGridColor }, ticks: { color: '#94a3b8' } } } } });
            new Chart(document.getElementById('pieChart').getContext('2d'), { type: 'doughnut', data: { labels: ['Cardiology', 'Gen. Practice', 'Orthopedics', 'Dermatology'], datasets: [{ data: [30, 50, 15, 5], backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'], borderWidth: 0 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#94a3b8' } } } } });
        } catch(e) {}
    }

    async function loadStats() {
        initCharts();
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('stat-total').innerText = data.total_claims;
            document.getElementById('stat-fraud').innerText = data.fraud_claims;
            document.getElementById('stat-amount').innerText = data.fraud_amount;
            document.getElementById('stat-density').innerText = data.network_density;
        } catch(e) {}
    }

    let network = null; let tableData = []; let selectedEntityId = null;
    async function loadGraph() {
        const container = document.getElementById('network-graph');
        if(network) return; 
        const res = await fetch('/api/graph_data');
        const data = await res.json();
        tableData = data.nodes; 
        const options = { nodes: { font: { color: "#64748b" }, borderWidth: 2 }, physics: { stabilization: { enabled: true, iterations: 200 }, barnesHut: { gravitationalConstant: -3000 } }, interaction: { hover: true, tooltipDelay: 200 }, layout: { improvedLayout: false } };
        network = new vis.Network(container, data, options);
        network.on("stabilizationIterationsDone", function () { network.setOptions( { physics: false } ); });
        setTimeout(() => { network.setOptions( { physics: false } ); }, 2000);
        renderTable(tableData); 
        network.on("click", function (params) { if(params.nodes.length > 0) selectEntity(params.nodes[0]); });
    }

    function renderTable(data) {
        const tableBody = document.getElementById('entity-table-body');
        tableBody.innerHTML = "";
        data.forEach(node => {
            const row = document.createElement('tr'); row.id = `row-${node.id}`; row.onclick = () => selectEntity(node.id); row.className = "cursor-pointer transition-colors";
            const riskColor = node.status === 'Suspicious' ? 'text-rose-500 font-bold' : 'text-emerald-500';
            const typeIcon = node.group === 'provider' ? '<i class="fas fa-user-md text-blue-400"></i>' : '<i class="fas fa-user text-emerald-400"></i>';
            row.innerHTML = `<td class="font-mono text-slate-600 dark:text-slate-300 px-4">${typeIcon} <span class="ml-2">${node.id}</span></td><td class="text-slate-600 dark:text-slate-300">${node.group}</td><td class="${riskColor}">${node.status}</td><td><div class="flex items-center gap-2"><span class="text-slate-600 dark:text-slate-400 w-8">${node.risk}</span><div class="w-16 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5"><div class="bg-blue-500 h-1.5 rounded-full" style="width: ${node.risk * 100}%"></div></div></div></td><td class="text-slate-600 dark:text-slate-300 text-center">${node.connections}</td><td class="text-slate-500 italic truncate max-w-xs" title="${node.reason}">${node.reason}</td>`;
            tableBody.appendChild(row);
        });
    }

    function selectEntity(id) {
        selectedEntityId = id;
        document.querySelectorAll('#entity-table-body tr').forEach(r => r.classList.remove('selected-row'));
        const row = document.getElementById(`row-${id}`); if(row) row.classList.add('selected-row');
        const btn = document.getElementById('btn-view-detail'); btn.disabled = false; btn.classList.remove('bg-slate-200', 'dark:bg-slate-700', 'text-slate-400', 'cursor-not-allowed'); btn.classList.add('bg-blue-600', 'hover:bg-blue-700', 'text-white', 'cursor-pointer');
        network.focus(id, { scale: 1.2, animation: true });
    }

    function filterTable() {
        const term = document.getElementById('table-search').value.toLowerCase();
        const filtered = tableData.filter(n => n.id.toLowerCase().includes(term) || n.status.toLowerCase().includes(term) || n.reason.toLowerCase().includes(term));
        renderTable(filtered);
    }

    let sortDirection = 1;
    function sortTable(colIndex) {
        const keys = ['id', 'group', 'status', 'risk', 'connections'];
        const key = keys[colIndex]; sortDirection *= -1;
        tableData.sort((a, b) => { let valA = key === 'risk' || key === 'connections' ? parseFloat(a[key]) : a[key]; let valB = key === 'risk' || key === 'connections' ? parseFloat(b[key]) : b[key]; return (valA > valB ? 1 : -1) * sortDirection; });
        renderTable(tableData);
    }

    async function viewEntityDetail() {
        if(!selectedEntityId) return;
        Swal.fire({title: 'Loading Intelligence...', didOpen: () => Swal.showLoading()});
        try {
            const res = await fetch(`/api/entity_details/${selectedEntityId}`); const data = await res.json(); Swal.close();
            document.getElementById('detail-id').innerText = data.id; document.getElementById('detail-role').innerText = data.role; document.getElementById('detail-icon').innerHTML = data.role.includes('Provider') ? '<i class="fas fa-user-md text-blue-500"></i>' : '<i class="fas fa-user text-emerald-500"></i>';
            const badge = document.getElementById('detail-status-badge');
            if(data.status === 'Suspicious') { badge.className = "inline-block px-3 py-1 rounded-full text-xs font-bold mb-4 bg-rose-500/20 text-rose-500 border border-rose-500/50"; badge.innerText = "HIGH PRIORITY INVESTIGATION"; document.getElementById('detail-reason').className = "text-slate-700 dark:text-slate-300 mt-1 bg-slate-100 dark:bg-slate-800 p-3 rounded-lg border-l-4 border-rose-500"; } 
            else { badge.className = "inline-block px-3 py-1 rounded-full text-xs font-bold mb-4 bg-emerald-500/20 text-emerald-500 border border-emerald-500/50"; badge.innerText = "VERIFIED ENTITY"; document.getElementById('detail-reason').className = "text-slate-700 dark:text-slate-300 mt-1 bg-slate-100 dark:bg-slate-800 p-3 rounded-lg border-l-4 border-emerald-500"; }
            document.getElementById('detail-claims-count').innerText = data.claim_count; document.getElementById('detail-claims-amt').innerText = data.total_amount; document.getElementById('detail-risk').innerText = data.risk_score; document.getElementById('detail-risk-bar').style.width = (parseFloat(data.risk_score) * 100) + '%'; document.getElementById('detail-reason').innerText = data.reason;
            const tbody = document.getElementById('detail-claims-body'); tbody.innerHTML = "";
            data.claims.forEach(c => { const r = document.createElement('tr'); const flag = c.is_fraud ? '<span class="text-rose-500 font-bold">⚠️ FRAUD</span>' : '<span class="text-emerald-500">OK</span>'; const target = data.role.includes('Provider') ? c.target : c.source; r.innerHTML = `<td class="px-6 py-3 font-mono">${c.id}</td><td class="px-6 py-3">${c.date}</td><td class="px-6 py-3 font-mono">${target}</td><td class="px-6 py-3">${c.specialty}</td><td class="px-6 py-3 text-right font-mono">$${c.amount}</td><td class="px-6 py-3 text-center text-xs">${flag}</td>`; tbody.appendChild(r); });
            showPage('detail');
        } catch(e) { Swal.fire('Error', 'Failed to load details', 'error'); }
    }

    // --- Camera & AI Logic ---
    let stream = null; let capturedBlob = null;
    function setCamMode(mode) { const fileBox=document.getElementById('mode-file'), camBox=document.getElementById('mode-cam'); const tabFile=document.getElementById('tab-file'), tabCam=document.getElementById('tab-cam'); if(mode==='cam'){ fileBox.classList.add('hidden'); camBox.classList.remove('hidden'); tabCam.classList.add('bg-white', 'dark:bg-slate-700', 'shadow', 'text-slate-800', 'dark:text-white'); tabFile.classList.remove('bg-white', 'dark:bg-slate-700', 'shadow', 'text-slate-800', 'dark:text-white'); startCamera(); } else { camBox.classList.add('hidden'); fileBox.classList.remove('hidden'); tabFile.classList.add('bg-white', 'dark:bg-slate-700', 'shadow', 'text-slate-800', 'dark:text-white'); tabCam.classList.remove('bg-white', 'dark:bg-slate-700', 'shadow', 'text-slate-800', 'dark:text-white'); stopCamera(); } }
    async function startCamera() { try { stream=await navigator.mediaDevices.getUserMedia({video:true}); document.getElementById('camera-stream').srcObject=stream; } catch(e){Swal.fire('Error','Cam access denied','error');} }
    function stopCamera() { if(stream) stream.getTracks().forEach(t=>t.stop()); }
    function capturePhoto() { const v=document.getElementById('camera-stream'), c=document.getElementById('camera-canvas'); c.width=v.videoWidth; c.height=v.videoHeight; c.getContext('2d').drawImage(v,0,0); c.toBlob(b=>{ capturedBlob=b; document.getElementById('captured-img').src=URL.createObjectURL(b); document.getElementById('captured-img').classList.remove('hidden'); document.getElementById('retake-btn').classList.remove('hidden'); stopCamera(); },'image/jpeg'); }
    function previewImage(i,id){ if(i.files[0]){ const r=new FileReader(); r.onload=e=>{document.getElementById(id).src=e.target.result;document.getElementById(id).classList.remove('hidden');}; r.readAsDataURL(i.files[0]); }}
    
    async function verifySignature() {
        const f1=document.getElementById('file-ref').files[0], f2=capturedBlob||document.getElementById('file-que').files[0];
        if(!f1||!f2) return Swal.fire('Info','Upload both signatures','info');
        const fd=new FormData(); fd.append('ref',f1); fd.append('que',f2,'c.jpg');
        Swal.showLoading();
        const res=await fetch('/api/verify_signature',{method:'POST',body:fd});
        const d=await res.json(); Swal.close();
        document.getElementById('sig-score').innerText=(d.score*100).toFixed(1)+'%';
        document.getElementById('sig-result').classList.remove('hidden');
        
        // FIX: Update verdict logic
        const verdict = document.getElementById('sig-verdict');
        if(d.score > 0.95) {
            verdict.innerText = "GENUINE";
            verdict.className = "inline-block px-6 py-2 rounded-full bg-emerald-500 text-white font-bold shadow-lg shadow-emerald-500/50";
        } else if(d.score > 0.80) {
             verdict.innerText = "INCONCLUSIVE";
             verdict.className = "inline-block px-6 py-2 rounded-full bg-orange-500 text-white font-bold shadow-lg shadow-orange-500/50";
        } else {
            verdict.innerText = "FORGERY";
            verdict.className = "inline-block px-6 py-2 rounded-full bg-rose-600 text-white font-bold shadow-lg shadow-rose-600/50";
        }
    }
    async function predictDisease() { const t=document.getElementById('symptom-input').value; if(!t) return; const res=await fetch('/api/predict_disease',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symptoms:t})}); const d=await res.json(); document.getElementById('pred-text').innerText=d.prediction; document.getElementById('prediction-result').classList.remove('hidden'); }
    async function checkFraud() { 
        const s=document.getElementById('symptom-input').value, c=document.getElementById('claim-input').value; 
        const res=await fetch('/api/check_fraud',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symptoms:s,claimed_disease:c})}); 
        const d=await res.json(); 
        document.getElementById('text-sim-sym').innerText = (d.similarity_symptoms * 100).toFixed(0) + '%';
        document.getElementById('circle-sim-sym').style.strokeDashoffset = 175.9 - (175.9 * d.similarity_symptoms);
        document.getElementById('text-sim-pred').innerText = (d.similarity_prediction * 100).toFixed(0) + '%';
        document.getElementById('circle-sim-pred').style.strokeDashoffset = 175.9 - (175.9 * d.similarity_prediction);
        document.getElementById('score-risk').innerText = d.risk_score.toFixed(2); 
        document.getElementById('fraud-result').classList.remove('hidden');
        const badge = document.getElementById('risk-badge');
        if(d.risk_score < 0.1) { badge.innerText = "LOW RISK"; badge.className = "w-full py-2 text-center font-bold rounded-lg bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/50"; } else if(d.risk_score < 0.3) { badge.innerText = "MEDIUM RISK"; badge.className = "w-full py-2 text-center font-bold rounded-lg bg-orange-100 dark:bg-orange-500/20 text-orange-600 dark:text-orange-400 border border-orange-200 dark:border-orange-500/50"; } else { badge.innerText = "HIGH FRAUD RISK"; badge.className = "w-full py-2 text-center font-bold rounded-lg bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-500/50 animate-pulse"; }
    }

    function initRoleBasedUI() {
        const user = APP_STATE.currentUser;
        document.getElementById('user-name').innerText = user.name;
        document.getElementById('user-role').innerText = user.role;
        document.getElementById('user-avatar').innerText = user.id;
        document.getElementById('user-role-badge').innerText = user.role.toUpperCase();
        if (user.role === 'Administrator') {
            document.getElementById('admin-section').classList.remove('hidden');
        }
    }
    
    initRoleBasedUI();
    loadStats();

    setInterval(() => {
        if(Math.random() > 0.7) {
            const newAlert = { type: 'system', message: 'New fraud pattern detected in claims batch', time: 'just now', severity: Math.random() > 0.5 ? 'critical' : 'high' };
            APP_STATE.alerts.unshift(newAlert);
            if (document.getElementById('page-alerts').classList.contains('hidden') === false) { renderAlerts(); }
            const dot = document.querySelector('.fa-bell + span');
            if(dot) dot.style.display = 'block';
        }
    }, 15000);

</script>
</body>
</html>
"""

if __name__ == '__main__':
    # Try to train model on startup if CSV exists and PKL doesn't
    if not os.path.exists(MODEL_PATH) and os.path.exists("final_symptoms_to_disease.csv"):
        print("Initializing model...")
        try:
            df = pd.read_csv("final_symptoms_to_disease.csv")
            df.dropna(subset=['symptom_text', 'diseases'], inplace=True)
            pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
            pipeline.fit(df['symptom_text'], df['diseases'])
            disease_model_data = {'model': pipeline, 'type': 'tfidf', 'classes': sorted(list(set(df['diseases'])))}
            print("Model trained successfully.")
        except Exception as e:
            print(f"Model training skipped: {e}")

    app.run(debug=True, port=5000)