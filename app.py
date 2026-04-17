import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Bias Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG   = "#0F1117"
CARD_BG   = "#1A1D2E"
ACCENT    = "#7C3AED"   # violet
ACCENT2   = "#06B6D4"   # cyan
SUCCESS   = "#10B981"   # emerald
DANGER    = "#EF4444"   # red
TEXT      = "#E2E8F0"
MUTED     = "#94A3B8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT};
}}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #12152A 0%, #0F1117 100%);
    border-right: 1px solid rgba(124,58,237,0.3);
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}

/* ---------- Main header ---------- */
.hero-banner {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1a1d2e 100%);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.hero-banner::before {{
    content: '';
    position: absolute;
    top: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}}
.hero-sub {{
    font-size: 1.05rem;
    color: {MUTED};
    margin: 0;
}}

/* ---------- Metric cards ---------- */
.metric-row {{
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}}
.metric-card {{
    flex: 1;
    min-width: 160px;
    background: {CARD_BG};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}}
.metric-card:hover {{ transform: translateY(-3px); }}
.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    margin: 6px 0 2px;
}}
.metric-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {MUTED};
}}

/* ---------- Section headings ---------- */
.section-title {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {TEXT};
    margin: 28px 0 14px;
    padding-left: 12px;
    border-left: 4px solid {ACCENT};
}}

/* ---------- Alert banners ---------- */
.bias-alert {{
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 16px 0;
}}
.bias-ok {{
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.5);
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 16px 0;
}}
.alert-icon {{ font-size: 1.8rem; }}
.alert-text {{ font-size: 1rem; font-weight: 600; }}
.alert-sub  {{ font-size: 0.85rem; color: {MUTED}; margin-top: 2px; }}

/* ---------- Progress bar ---------- */
.bar-wrap {{
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    margin-top: 4px;
}}
.bar-fill-violet {{
    height: 100%;
    background: linear-gradient(90deg, #7C3AED, #a78bfa);
    border-radius: 8px;
    transition: width 0.8s ease;
}}
.bar-fill-cyan {{
    height: 100%;
    background: linear-gradient(90deg, #0891B2, #67e8f9);
    border-radius: 8px;
    transition: width 0.8s ease;
}}
.bar-fill-red {{
    height: 100%;
    background: linear-gradient(90deg, #DC2626, #FCA5A5);
    border-radius: 8px;
}}

/* ---------- Info card ---------- */
.info-card {{
    background: {CARD_BG};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 24px;
    height: 100%;
}}

/* ---------- Step badge ---------- */
.step {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 30px;
    padding: 6px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #a78bfa;
    margin: 4px 4px 4px 0;
}}

/* ---------- Dataframe overrides ---------- */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}

/* Hide default Streamlit footer */
footer {{ visibility: hidden; }}

/* ---------- Button overrides ---------- */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, #5b21b6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.85; }}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ AI Bias Detector")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random State", min_value=0, max_value=999, value=42, step=1)
    bias_threshold = st.slider("Bias Alert Threshold", 0.05, 0.50, 0.10, 0.05,
                               help="Bias scores above this trigger a warning (U.S. '80% rule' suggests 0.10).")

    st.markdown("---")
    st.markdown("### 📌 Pipeline Steps")
    steps = ["Load CSV", "Drop NAs", "Label Encode", "Train / Test Split",
             "Train Logistic Regression", "Predict", "Bias Audit"]
    for i, s in enumerate(steps, 1):
        st.markdown(f'<span class="step">#{i} {s}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Demonstrates an end-to-end ML fairness audit. "
        "The model predicts loan approvals and we measure "
        "**statistical parity** across gender groups.",
        unsafe_allow_html=False,
    )


# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">⚖️ AI Bias Detector</p>
  <p class="hero-sub">
    Loan Approval Prediction · Fairness Audit · Statistical Parity Analysis
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD & VALIDATE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("./loan_data.csv")

raw_df = load_data()

required = {"Gender", "Married", "ApplicantIncome", "LoanAmount", "Loan_Status"}
if not required.issubset(raw_df.columns):
    st.error(f"❌ Missing columns. Expected: {required}. Found: {set(raw_df.columns)}")
    st.stop()

# ─────────────────────────────────────────────
#  DATASET PREVIEW
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">📋 Dataset Preview</p>', unsafe_allow_html=True)

col_data, col_info = st.columns([3, 2], gap="large")

with col_data:
    st.dataframe(
        raw_df.style.map(
            lambda v: "background-color: rgba(16,185,129,0.15); color: #6EE7B7;" if v == 1 else
                      "background-color: rgba(239,68,68,0.12); color: #FCA5A5;" if v == 0 else "",
            subset=["Loan_Status"]
        ),
        use_container_width=True,
        height=360,
    )

with col_info:
    total = len(raw_df)
    approved = int(raw_df["Loan_Status"].sum())
    rejected = total - approved
    males    = int((raw_df["Gender"] == "Male").sum())
    females  = int((raw_df["Gender"] == "Female").sum())

    st.markdown(f"""
    <div class="info-card">
      <p class="metric-label">Dataset Summary</p>
      <br/>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <span style="color:{MUTED};font-size:.9rem;">Total Records</span>
        <span style="font-weight:700;font-size:1.1rem;">{total}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <span style="color:{MUTED};font-size:.9rem;">✅ Approved</span>
        <span style="color:#6EE7B7;font-weight:700;">{approved}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <span style="color:{MUTED};font-size:.9rem;">❌ Rejected</span>
        <span style="color:#FCA5A5;font-weight:700;">{rejected}</span>
      </div>
      <hr style="border-color:rgba(255,255,255,0.08);margin:14px 0;"/>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <span style="color:{MUTED};font-size:.9rem;">👨 Male</span>
        <span style="font-weight:700;">{males}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <span style="color:{MUTED};font-size:.9rem;">👩 Female</span>
        <span style="font-weight:700;">{females}</span>
      </div>
      <hr style="border-color:rgba(255,255,255,0.08);margin:14px 0;"/>
      <p style="font-size:.8rem;color:{MUTED};">
        ⚠️ Toy dataset — all males approved, all females rejected.
        Designed to illustrate maximum disparity.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREPROCESSING & MODEL TRAINING
# ─────────────────────────────────────────────
df = raw_df.dropna().copy()

# Label encode — fit per column so we can recover mappings
encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

model = LogisticRegression(max_iter=1000, random_state=int(random_state))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ─────────────────────────────────────────────
#  ACCURACY METRICS
# ─────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(cm[0][0]))

train_size = len(X_train)
test_size_n = len(X_test)

st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Accuracy</div>
    <div class="metric-value" style="color:#a78bfa;">{accuracy*100:.1f}%</div>
    <div class="metric-label">on test set</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Train Samples</div>
    <div class="metric-value" style="color:{ACCENT2};">{train_size}</div>
    <div class="metric-label">records</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Test Samples</div>
    <div class="metric-value" style="color:{ACCENT2};">{test_size_n}</div>
    <div class="metric-label">records</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">True Positives</div>
    <div class="metric-value" style="color:{SUCCESS};">{tp}</div>
    <div class="metric-label">approved correctly</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">True Negatives</div>
    <div class="metric-value" style="color:{SUCCESS};">{tn}</div>
    <div class="metric-label">rejected correctly</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">False Positives</div>
    <div class="metric-value" style="color:{DANGER};">{fp}</div>
    <div class="metric-label">wrongly approved</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">False Negatives</div>
    <div class="metric-value" style="color:{DANGER};">{fn}</div>
    <div class="metric-label">wrongly rejected</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONFUSION MATRIX CHART
# ─────────────────────────────────────────────
col_cm, col_coef = st.columns(2, gap="large")

with col_cm:
    fig_cm, ax = plt.subplots(figsize=(4, 3.2))
    fig_cm.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    cmap_data = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cmap_data, cmap="RdYlGn", vmin=0, vmax=max(cmap_data.max(), 1))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: 0", "Pred: 1"], color=TEXT)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: 0", "True: 1"], color=TEXT)
    ax.tick_params(colors=TEXT)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cmap_data[i, j], ha="center", va="center",
                    fontsize=18, fontweight="bold", color=DARK_BG)
    ax.set_title("Confusion Matrix", color=TEXT, fontsize=11, pad=10)
    fig_cm.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color=TEXT)
    plt.tight_layout()
    st.pyplot(fig_cm)

with col_coef:
    feature_names = X.columns.tolist()
    coefs = model.coef_[0]
    colors = [SUCCESS if c >= 0 else DANGER for c in coefs]

    fig_coef, ax2 = plt.subplots(figsize=(4, 3.2))
    fig_coef.patch.set_facecolor(CARD_BG)
    ax2.set_facecolor(CARD_BG)

    bars = ax2.barh(feature_names, coefs, color=colors, edgecolor="none", height=0.55)
    ax2.axvline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    ax2.set_title("Feature Coefficients", color=TEXT, fontsize=11)
    ax2.tick_params(colors=TEXT)
    for sp in ax2.spines.values(): sp.set_visible(False)
    for bar, v in zip(bars, coefs):
        ax2.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", ha="left" if v >= 0 else "right",
                 color=TEXT, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_coef)


# ─────────────────────────────────────────────
#  BIAS DETECTION
# ─────────────────────────────────────────────
def calculate_bias(df_encoded, test_indices, predictions, sensitive_col="Gender"):
    df_test = df_encoded.iloc[test_indices].copy()
    df_test["Pred"] = predictions

    group0 = df_test[df_test[sensitive_col] == 0]["Pred"]
    group1 = df_test[df_test[sensitive_col] == 1]["Pred"]

    g0_rate = float(group0.mean()) if len(group0) > 0 else 0.0
    g1_rate = float(group1.mean()) if len(group1) > 0 else 0.0
    bias_score = abs(g0_rate - g1_rate)

    return g0_rate, g1_rate, bias_score, len(group0), len(group1)

g0_rate, g1_rate, bias_score, n0, n1 = calculate_bias(
    df, list(X_test.index), y_pred, "Gender"
)

# Recover human-readable group labels
gender_le = encoders.get("Gender")
label0 = gender_le.inverse_transform([0])[0] if gender_le else "Group 0"
label1 = gender_le.inverse_transform([1])[0] if gender_le else "Group 1"

st.markdown('<p class="section-title">🔍 Fairness Audit — Statistical Parity</p>', unsafe_allow_html=True)

# Bias score alert
severity = bias_score / 1.0  # 0→1 scale
if bias_score > bias_threshold:
    color_hex = DANGER
    st.markdown(f"""
    <div class="bias-alert">
      <span class="alert-icon">🚨</span>
      <div>
        <div class="alert-text" style="color:{DANGER};">
          Significant bias detected — Bias Score: {bias_score:.2f} ({bias_score*100:.0f}%)
        </div>
        <div class="alert-sub">
          Exceeds your threshold of {bias_threshold:.2f} ({bias_threshold*100:.0f}%).
          The model favors <strong>{label1}</strong> applicants over <strong>{label0}</strong>.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    color_hex = SUCCESS
    st.markdown(f"""
    <div class="bias-ok">
      <span class="alert-icon">✅</span>
      <div>
        <div class="alert-text" style="color:{SUCCESS};">
          Fairness within threshold — Bias Score: {bias_score:.2f} ({bias_score*100:.0f}%)
        </div>
        <div class="alert-sub">
          Below your threshold of {bias_threshold:.2f} ({bias_threshold*100:.0f}%).
          Approval rates are approximately equal across groups.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Approval rate bars
col_g0, col_g1, col_bias = st.columns(3, gap="large")

with col_g0:
    st.markdown(f"""
    <div class="info-card">
      <p class="metric-label">👩 {label0} Approval Rate</p>
      <p style="font-size:2rem;font-weight:700;color:{ACCENT};margin:8px 0 4px;">
        {g0_rate*100:.1f}%
      </p>
      <p style="color:{MUTED};font-size:.85rem;margin-bottom:10px;">
        n = {n0} test sample{'s' if n0!=1 else ''}
      </p>
      <div class="bar-wrap">
        <div class="bar-fill-violet" style="width:{g0_rate*100:.1f}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_g1:
    st.markdown(f"""
    <div class="info-card">
      <p class="metric-label">👨 {label1} Approval Rate</p>
      <p style="font-size:2rem;font-weight:700;color:{ACCENT2};margin:8px 0 4px;">
        {g1_rate*100:.1f}%
      </p>
      <p style="color:{MUTED};font-size:.85rem;margin-bottom:10px;">
        n = {n1} test sample{'s' if n1!=1 else ''}
      </p>
      <div class="bar-wrap">
        <div class="bar-fill-cyan" style="width:{g1_rate*100:.1f}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_bias:
    bias_color = DANGER if bias_score > bias_threshold else SUCCESS
    st.markdown(f"""
    <div class="info-card">
      <p class="metric-label">📐 Bias Score (|Δ|)</p>
      <p style="font-size:2rem;font-weight:700;color:{bias_color};margin:8px 0 4px;">
        {bias_score*100:.1f}%
      </p>
      <p style="color:{MUTED};font-size:.85rem;margin-bottom:10px;">
        |{g0_rate*100:.1f}% − {g1_rate*100:.1f}%|
      </p>
      <div class="bar-wrap">
        <div class="bar-fill-red" style="width:{bias_score*100:.1f}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  GROUP APPROVAL CHART
# ─────────────────────────────────────────────
st.markdown("")

col_chart, col_explain = st.columns([3, 2], gap="large")

with col_chart:
    fig_g, ax_g = plt.subplots(figsize=(5, 3.5))
    fig_g.patch.set_facecolor(CARD_BG)
    ax_g.set_facecolor(CARD_BG)

    groups = [label0, label1]
    rates  = [g0_rate * 100, g1_rate * 100]
    bar_colors = ["#7C3AED", "#06B6D4"]

    bars2 = ax_g.bar(groups, rates, color=bar_colors, width=0.45,
                     edgecolor="none", zorder=3)
    ax_g.axhline(100, color=(1, 1, 1, 0.1), linewidth=0.5)
    ax_g.set_ylim(0, 115)
    ax_g.set_ylabel("Approval Rate (%)", color=TEXT)
    ax_g.set_title("Group Approval Rates", color=TEXT, fontsize=11, pad=10)
    ax_g.tick_params(colors=TEXT)
    for sp in ax_g.spines.values(): sp.set_visible(False)
    ax_g.yaxis.grid(True, color=(1, 1, 1, 0.06), zorder=0)

    for bar, rate in zip(bars2, rates):
        ax_g.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  f"{rate:.1f}%", ha="center", va="bottom",
                  color=TEXT, fontsize=13, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig_g)

with col_explain:
    st.markdown(f"""
    <div class="info-card">
      <p style="font-size:1rem;font-weight:700;color:{TEXT};margin-bottom:12px;">
        🧮 How is Bias Measured?
      </p>
      <p style="color:{MUTED};font-size:.88rem;line-height:1.65;">
        We use <strong>Statistical Parity Difference</strong> — a standard group fairness
        metric that compares the fraction of approved predictions across demographic groups.
      </p>
      <br/>
      <code style="background:rgba(124,58,237,0.2);padding:6px 10px;border-radius:6px;
                   font-size:.82rem;color:#c4b5fd;display:block;margin-bottom:12px;">
        bias = |approval_rate(Female) − approval_rate(Male)|
      </code>
      <ul style="color:{MUTED};font-size:.85rem;line-height:1.8;padding-left:18px;">
        <li><strong>0.00</strong> → Perfect parity</li>
        <li><strong>&lt; {bias_threshold:.2f}</strong> → Within acceptable range</li>
        <li><strong>&gt; {bias_threshold:.2f}</strong> → Significant disparity ⚠️</li>
        <li><strong>1.00</strong> → Maximum disparity (100%)</li>
      </ul>
      <p style="color:{MUTED};font-size:.82rem;margin-top:12px;">
        In our dataset, all males are approved and all females rejected, 
        yielding a bias score of <strong style="color:{DANGER};">1.00 (100%)</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MITIGATION STRATEGIES
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">🛠️ Bias Mitigation Strategies</p>', unsafe_allow_html=True)

strategies = [
    ("⚖️", "Reweighting", ACCENT,
     "Assign higher weights to under-represented group outcomes during training. "
     "Use scikit-learn's `sample_weight` or `class_weight` parameters."),
    ("🔄", "Resampling", ACCENT2,
     "Oversample minority outcomes (e.g. SMOTE for female-approved) or undersample "
     "the dominant group to balance the training set."),
    ("🤖", "Fairness-Aware Algorithms", "#F59E0B",
     "Use IBM's AI Fairness 360 or Microsoft Fairlearn for transformations like "
     "Adversarial Debiasing or Prejudice Remover."),
    ("🎯", "Threshold Adjustment", "#EC4899",
     "Set different decision thresholds per group. Lower the bar for the disadvantaged "
     "group to equalize approval rates (the 'reject option' approach)."),
]

cols = st.columns(4, gap="medium")
for col, (icon, title, color, desc) in zip(cols, strategies):
    with col:
        st.markdown(f"""
        <div class="info-card" style="border-top: 3px solid {color};">
          <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
          <p style="font-weight:700;font-size:.95rem;color:{TEXT};margin-bottom:8px;">
            {title}
          </p>
          <p style="color:{MUTED};font-size:.82rem;line-height:1.6;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;padding:20px;color:{MUTED};font-size:.8rem;
            border-top:1px solid rgba(255,255,255,0.07);margin-top:20px;">
  ⚖️ <strong>AI Bias Detector</strong> · Built with Streamlit, scikit-learn & Pandas ·
  Fairness metric: Statistical Parity Difference
</div>
""", unsafe_allow_html=True)