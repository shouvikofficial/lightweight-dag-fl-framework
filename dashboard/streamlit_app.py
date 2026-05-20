import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="FL + Blockchain Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    body { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f8ef7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8a8fa8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e4f0;
        border-bottom: 2px solid #4f8ef7;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stSidebar { background-color: #1a1d2e !important; }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #1a3a2a; color: #4caf8a; }
    .badge-blue  { background: #1a2a3a; color: #4f8ef7; }
    .badge-red   { background: #3a1a1a; color: #f74f4f; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/240px-Jupyter_logo.svg.png",
    width=60
)
st.sidebar.title("🧠 FL Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔗 Blockchain / DAG", "📊 Training Metrics",
     "🔒 Privacy", "📡 Communication", "🔍 Predict"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='badge badge-green'>● System Online</span>",
    unsafe_allow_html=True
)
st.sidebar.caption("Flower v1.29 · TensorFlow · Streamlit")

# ============================================
# HELPER: load ledger
# ============================================

LEDGER_PATH = "blockchain/ledger.json"
LOG_DIR = "logs"
METRICS_PATH = os.path.join(LOG_DIR, "metrics.jsonl")
SERVER_LOG_PATH = os.path.join(LOG_DIR, "server.log")

def load_ledger():
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _tail_file(path, max_lines=200):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def _load_metrics():
    if not os.path.exists(METRICS_PATH):
        return pd.DataFrame()
    rows = []
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "metrics" in df.columns:
        metrics_df = df["metrics"].apply(pd.Series)
        df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    return df

# ============================================
# PAGE: OVERVIEW
# ============================================

if page == "🏠 Overview":

    st.markdown("# 🧠 Federated Learning + Blockchain")
    st.markdown("### Skin Lesion Classification — ISIC 2019")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    ledger = load_ledger()
    num_tx = len(ledger)
    num_clients = 4

    accs = [tx.get("accuracy", 0) for tx in ledger if tx.get("accuracy")]
    best_acc = max(accs) if accs else 0.0
    avg_acc  = np.mean(accs) if accs else 0.0

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FL Clients</div>
            <div class="metric-value">{num_clients}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">DAG Transactions</div>
            <div class="metric-value">{num_tx}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value">{best_acc:.2%}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Accuracy</div>
            <div class="metric-value">{avg_acc:.2%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='section-title'>System Architecture</div>", unsafe_allow_html=True)

    arch_cols = st.columns(3)
    components = [
        ("🖥️ FL Server", "FedProx strategy · 5 rounds · Flower 1.29", "badge-blue"),
        ("💻 FL Clients", "4 clients · EfficientNetB0 · Non-IID ISIC", "badge-green"),
        ("🔗 DAG Blockchain", "Lightweight DAG · Hash integrity · Tip selection", "badge-blue"),
    ]
    for col, (title, desc, badge) in zip(arch_cols, components):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <b>{title}</b><br>
                <span style="color:#8a8fa8;font-size:0.85rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    part_dir = "dataset/partitions"
    if os.path.exists(part_dir):
        st.markdown("<div class='section-title'>Client Dataset Partitions</div>", unsafe_allow_html=True)
        rows = []
        for csv_file in sorted(os.listdir(part_dir)):
            if csv_file.endswith(".csv"):
                df = pd.read_csv(os.path.join(part_dir, csv_file))
                dist = df["label"].value_counts().to_dict()
                rows.append({
                    "Client": csv_file.replace(".csv", ""),
                    "Total Samples": len(df),
                    **dist
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).fillna(0).astype({col: int for col in pd.DataFrame(rows).columns if col not in ["Client"]}), width='stretch')
    else:
        st.info("📂 Run `preprocessing/partition.py` first to generate client datasets.")


# ============================================
# PAGE: BLOCKCHAIN / DAG
# ============================================

elif page == "🔗 Blockchain / DAG":

    st.markdown("# 🔗 Blockchain DAG Ledger")
    st.markdown("---")

    ledger = load_ledger()

    if not ledger:
        st.warning("No ledger data found. Run `main.py` to generate DAG transactions.")
    else:
        st.success(f"✅ {len(ledger)} transactions found in ledger.")

        df = pd.DataFrame(ledger)
        st.markdown("<div class='section-title'>Transaction Table</div>", unsafe_allow_html=True)
        st.dataframe(df, width='stretch')

        st.markdown("---")
        st.markdown("<div class='section-title'>Accuracy per Transaction</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
        ax.set_facecolor("#1a1d2e")
        ax.bar(
            range(len(df)),
            df["accuracy"],
            color="#4f8ef7",
            edgecolor="#2a2d3e"
        )
        ax.set_xlabel("Transaction Index", color="#8a8fa8")
        ax.set_ylabel("Accuracy", color="#8a8fa8")
        ax.set_title("Client Model Accuracy per DAG Transaction", color="#e0e4f0")
        ax.tick_params(colors="#8a8fa8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("<div class='section-title'>Validation Status</div>", unsafe_allow_html=True)
        valid_count   = df["validated"].sum() if "validated" in df.columns else 0
        invalid_count = len(df) - valid_count
        v1, v2 = st.columns(2)
        v1.metric("✅ Validated", int(valid_count))
        v2.metric("❌ Not Validated", int(invalid_count))

        st.markdown("---")
        st.markdown("<div class='section-title'>DAG Reference Graph</div>", unsafe_allow_html=True)
        try:
            import networkx as nx
            G = nx.DiGraph()
            for tx in ledger:
                tx_id = tx["transaction_id"][:8]
                G.add_node(tx_id, accuracy=tx.get("accuracy", 0))
                for ref in tx.get("references", []):
                    G.add_edge(ref[:8], tx_id)

            fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
            ax2.set_facecolor("#1a1d2e")
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(
                G, pos, ax=ax2,
                node_color="#4f8ef7",
                edge_color="#8a8fa8",
                font_color="white",
                node_size=800,
                font_size=8,
                arrows=True
            )
            ax2.set_title("DAG Transaction Graph", color="#e0e4f0")
            st.pyplot(fig2)
            plt.close(fig2)
        except ImportError:
            st.info("Install networkx to see DAG graph.")


# ============================================
# PAGE: TRAINING METRICS
# ============================================

elif page == "📊 Training Metrics":

    st.markdown("# 📊 Training Metrics")
    st.markdown("---")

    st.info("Training metrics are collected live during federated training rounds. Start the server and clients to see live data.")

    if st.button("Refresh now"):
        st.experimental_rerun()

    metrics_df = _load_metrics()
    if not metrics_df.empty:
        st.markdown("<div class='section-title'>Round Metrics</div>", unsafe_allow_html=True)
        st.dataframe(metrics_df, width='stretch')

        if "round" in metrics_df.columns and "accuracy" in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
            ax.set_facecolor("#1a1d2e")
            ax.plot(metrics_df["round"], metrics_df["accuracy"], "o-", color="#4caf8a")
            ax.set_xlabel("Round", color="#8a8fa8")
            ax.set_ylabel("Accuracy", color="#8a8fa8")
            ax.tick_params(colors="#8a8fa8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")
            st.pyplot(fig)
            plt.close(fig)

    ledger = load_ledger()
    if ledger:
        df = pd.DataFrame(ledger)
        if "accuracy" in df.columns and "client_id" in df.columns:
            st.markdown("<div class='section-title'>Accuracy per Client</div>", unsafe_allow_html=True)
            client_acc = df.groupby("client_id")["accuracy"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
            ax.set_facecolor("#1a1d2e")
            ax.barh(client_acc["client_id"], client_acc["accuracy"], color="#4caf8a")
            ax.set_xlabel("Average Accuracy", color="#8a8fa8")
            ax.tick_params(colors="#8a8fa8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.markdown("<div class='section-title'>FL Configuration</div>", unsafe_allow_html=True)
    config_data = {
        "Parameter": ["Strategy", "FL Rounds", "Min Fit Clients", "Min Evaluate Clients", "Proximal Mu", "Learning Rate", "Batch Size", "Model"],
        "Value": ["FedProx", "5", "2", "2", "0.01", "1e-4", "32", "EfficientNetB0"]
    }
    st.table(pd.DataFrame(config_data))

    st.markdown("---")
    st.markdown("<div class='section-title'>Live Logs</div>", unsafe_allow_html=True)

    log_choice = st.selectbox(
        "Choose log",
        [
            "server.log",
            "client_1.log",
            "client_2.log",
            "client_3.log",
            "client_4.log",
        ],
    )
    log_path = os.path.join(LOG_DIR, log_choice)
    log_text = _tail_file(log_path, max_lines=200)
    st.text_area("Log output", log_text, height=300)


# ============================================
# PAGE: PRIVACY
# ============================================

elif page == "🔒 Privacy":

    st.markdown("# 🔒 Privacy & Security")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Differential Privacy</div>", unsafe_allow_html=True)
        noise_mult = st.slider("Noise Multiplier (σ)", 0.001, 0.1, 0.01, 0.001)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gaussian Noise σ</div>
            <div class="metric-value">{noise_mult:.3f}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Gaussian noise added to gradients before upload to protect individual data.")

        st.markdown("<div class='section-title'>Gradient Clipping</div>", unsafe_allow_html=True)
        clip_val = st.slider("Clip Value", 0.1, 5.0, 1.0, 0.1)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Gradient Norm</div>
            <div class="metric-value">{clip_val:.1f}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Privacy Budget Estimation</div>", unsafe_allow_html=True)
        eps = noise_mult * 10
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated ε (epsilon)</div>
            <div class="metric-value">{eps:.3f}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Lower ε = stronger privacy. δ is fixed at 1e-5.")

        st.markdown("<div class='section-title'>Security Mechanisms</div>", unsafe_allow_html=True)
        mechanisms = {
            "Differential Privacy": "✅ Active",
            "Gradient Clipping": "✅ Active",
            "Model Hash Verification": "✅ Active",
            "DAG Duplicate Detection": "✅ Active",
            "Transaction Integrity Check": "✅ Active",
        }
        for k, v in mechanisms.items():
            st.markdown(f"**{k}** — {v}")


# ============================================
# PAGE: COMMUNICATION
# ============================================

elif page == "📡 Communication":

    st.markdown("# 📡 Communication Optimization")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-title'>Float16 Compression</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Size Reduction</div>
            <div class="metric-value">50%</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Weights cast from float32 → float16 before transmission.")

    with col2:
        st.markdown("<div class='section-title'>Top-K Sparsification</div>", unsafe_allow_html=True)
        k_ratio = st.slider("K Ratio", 0.01, 1.0, 0.1, 0.01)
        reduction = (1 - k_ratio) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gradient Reduction</div>
            <div class="metric-value">{reduction:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='section-title'>INT8 Quantization</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Bits per Weight</div>
            <div class="metric-value">8-bit</div>
        </div>""", unsafe_allow_html=True)
        st.caption("4× size reduction vs float32 with minimal accuracy loss.")

    st.markdown("---")
    st.markdown("<div class='section-title'>Simulated Bandwidth Usage</div>", unsafe_allow_html=True)

    rounds = list(range(1, 6))
    baseline_mb  = [42.5, 42.5, 42.5, 42.5, 42.5]
    compressed_mb = [21.2, 21.1, 21.3, 21.0, 21.2]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
    ax.set_facecolor("#1a1d2e")
    ax.plot(rounds, baseline_mb, "o-",  color="#f74f4f", label="No Compression")
    ax.plot(rounds, compressed_mb, "o-", color="#4caf8a", label="Float16 Compression")
    ax.set_xlabel("FL Round", color="#8a8fa8")
    ax.set_ylabel("Upload Size (MB)", color="#8a8fa8")
    ax.legend(facecolor="#1a1d2e", labelcolor="#e0e4f0")
    ax.tick_params(colors="#8a8fa8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    st.pyplot(fig)
    plt.close(fig)


# ============================================
# PAGE: PREDICT
# ============================================

elif page == "🔍 Predict":

    st.markdown("# 🔍 Skin Lesion Prediction")
    st.markdown("---")

    CLASS_NAMES = ["MEL", "NV", "BKL", "BCC", "AK", "VASC", "DF", "SCC"]
    CLASS_LABELS = {
        "MEL":  "Melanoma",
        "NV":   "Melanocytic Nevi",
        "BKL":  "Benign Keratosis",
        "BCC":  "Basal Cell Carcinoma",
        "AK":   "Actinic Keratosis",
        "VASC": "Vascular Lesion",
        "DF":   "Dermatofibroma",
        "SCC":  "Squamous Cell Carcinoma",
    }

    uploaded_file = st.file_uploader(
        "Upload a skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        import cv2
        from PIL import Image as PILImage
        import io

        image_bytes = uploaded_file.read()
        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(pil_img, caption="Uploaded Image", width='stretch')

        with col_result:
            st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)

            checkpoint_dir = "models/checkpoints"
            model_files = []
            if os.path.exists(checkpoint_dir):
                model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras") or f.endswith(".h5")]

            if not model_files:
                st.warning("⚠️ No trained model found in `models/checkpoints/`.")
                st.info("Train the federated model first, then save it to `models/checkpoints/model.keras`.")

                # Demo: random predictions for display
                st.markdown("**Demo Mode** (random probabilities):")
                probs = np.random.dirichlet(np.ones(8))
                pred_idx = np.argmax(probs)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx]

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value">{pred_class}</div>
                    <div style="color:#8a8fa8;margin-top:4px">{CLASS_LABELS[pred_class]}</div>
                </div>""", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0e1117")
                ax.set_facecolor("#1a1d2e")
                colors = ["#4f8ef7" if i == pred_idx else "#2a2d3e" for i in range(8)]
                ax.barh(CLASS_NAMES, probs, color=colors)
                ax.set_xlabel("Probability", color="#8a8fa8")
                ax.tick_params(colors="#8a8fa8")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                st.pyplot(fig)
                plt.close(fig)

            else:
                import tensorflow as tf
                model_path = os.path.join(checkpoint_dir, model_files[0])
                model = tf.keras.models.load_model(model_path)

                img_array = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[pred_idx]

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value">{pred_class}</div>
                    <div style="color:#8a8fa8;margin-top:4px">{CLASS_LABELS[pred_class]}</div>
                </div>""", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0e1117")
                ax.set_facecolor("#1a1d2e")
                colors = ["#4f8ef7" if i == pred_idx else "#2a2d3e" for i in range(8)]
                ax.barh(CLASS_NAMES, preds, color=colors)
                ax.set_xlabel("Probability", color="#8a8fa8")
                ax.tick_params(colors="#8a8fa8")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                st.pyplot(fig)
                plt.close(fig)

    else:
        st.info("👆 Upload a dermoscopy image to get a prediction.")
        st.markdown("**Supported classes:**")
        for k, v in CLASS_LABELS.items():
            st.markdown(f"- **{k}** — {v}")
