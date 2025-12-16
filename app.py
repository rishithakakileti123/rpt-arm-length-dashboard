import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Driven RPT & Arm’s Length Risk",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL CSS — ONLY FONT CHANGES
# --------------------------------------------------
st.markdown("""
<style>
/* Base */
html, body {
    font-size: 16px !important;
}


/* Header spacing */
.block-container {
    padding-top: 1.4rem !important;
}

/* Header box */
.header-box {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 30px;
}

/* Title & objective */
.header-box h1 { font-size: 48px !important; }
.header-box h3 { font-size: 34px !important; }
.header-box p  { font-size: 24px !important; }

/* Company snapshot metrics — BIGGER */
[data-testid="stMetricValue"] {
    font-size: 44px !important;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    font-size: 26px !important;
}

/* Tabs font size */
button[data-baseweb="tab"] > div {
    font-size: 26px !important;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] * {
    font-size: 22px !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("AI_RPT_ArmLength_MultiIndustry_Dataset.csv")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🔍 Filters")

industry = st.sidebar.selectbox(
    "Select Industry",
    sorted(df["Industry"].unique())
)

company = st.sidebar.selectbox(
    "Select Company",
    sorted(df[df["Industry"] == industry]["Company"].unique())
)

year = st.sidebar.selectbox(
    "Select Financial Year",
    sorted(df["Year"].unique(), reverse=True)
)

risk_threshold = st.sidebar.slider(
    "Risk Sensitivity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

filtered_df = df[
    (df["Industry"] == industry) &
    (df["Company"] == company) &
    (df["Year"] == year)
]

peer_df = df[
    (df["Industry"] == industry) &
    (df["Year"] == year)
]

# --------------------------------------------------
# HEADER + OBJECTIVE
# --------------------------------------------------
st.markdown("""
<div class="header-box">
<h1>AI-Driven Detection of Suspicious Related-Party Transactions & Arm’s Length Pricing Deviations</h1>

<h3>🎯 Objective</h3>
<p>
Identify potentially non–arm’s length related-party transactions using
<b>financial-statement ratios, peer benchmarking, and AI-based anomaly detection.</b>
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# COMPANY SNAPSHOT
# --------------------------------------------------
st.subheader("📊 Company Financial Snapshot")

c1, c2, c3 = st.columns(3)
c1.metric("Revenue (₹ Cr)", f"{filtered_df['Revenue'].values[0]:,.0f}")
c2.metric("EBITDA (₹ Cr)", f"{filtered_df['EBITDA'].values[0]:,.0f}")
c3.metric("Total Assets (₹ Cr)", f"{filtered_df['Total_Assets'].values[0]:,.0f}")

# --------------------------------------------------
# AI RISK SCORE (UNCHANGED)
# --------------------------------------------------
rpt_features = [
    "RPT_Sales_Ratio",
    "RPT_Purchase_Ratio",
    "RPT_Loan_Ratio",
    "RPT_Expense_to_EBITDA"
]

model = IsolationForest(
    n_estimators=200,
    contamination=0.15,
    random_state=42
)

model.fit(peer_df[rpt_features])

peer_scores = -model.decision_function(peer_df[rpt_features])
company_score = -model.decision_function(filtered_df[rpt_features])[0]

ai_score = (company_score - peer_scores.min()) / (peer_scores.max() - peer_scores.min())
ai_score = round(float(ai_score), 3)

st.subheader("🚨 AI Risk Assessment")

if ai_score >= risk_threshold:
    risk_state = "high"
    st.error(f"AI Risk Score: {ai_score}")
elif ai_score >= risk_threshold * 0.7:
    risk_state = "medium"
    st.warning(f"AI Risk Score: {ai_score}")
else:
    risk_state = "low"
    st.success(f"AI Risk Score: {ai_score}")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2 = st.tabs([
    "🔴 Suspicious Related-Party Transactions",
    "⚖️ Arm’s Length Pricing Deviation"
])

# ==================================================
# TAB 1
# ==================================================
with tab1:
    st.subheader("🔎 Related-Party Transaction Exposure")

    rpt_long = filtered_df[rpt_features].melt(
        var_name="Metric",
        value_name="Company Value"
    )

    fig1 = px.bar(
        rpt_long,
        x="Metric",
        y="Company Value",
        title="Company-Level RPT Ratios",
        color="Metric"
    )
    fig1.update_layout(
        title_font_size=34,
        xaxis_title_font_size=26,
        yaxis_title_font_size=26,
        xaxis_tickfont_size=22,
        yaxis_tickfont_size=22,
        legend_font_size=22
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig_box = px.box(
        peer_df,
        y="RPT_Purchase_Ratio",
        title="Peer Distribution: RPT Purchase Ratio"
    )
    fig_box.update_layout(
        title_font_size=30,
        yaxis_title_font_size=26,
        yaxis_tickfont_size=22
    )
    st.plotly_chart(fig_box, use_container_width=True)

    fig_iso = px.scatter(
        peer_df,
        y=peer_scores,
        title="Anomaly Score Distribution (Isolation Forest)"
    )
    fig_iso.update_layout(
        title_font_size=30,
        xaxis_tickfont_size=22,   # ✅ FIXED
        yaxis_title_font_size=26,
        yaxis_tickfont_size=22
    )
    st.plotly_chart(fig_iso, use_container_width=True)

    st.subheader("🧠 AI Interpretation")

    if risk_state == "high":
        st.error("• High related-party purchases vs peers")
    elif risk_state == "medium":
        st.warning("• Elevated loans/advances to related parties")
    else:
        st.success("No major related-party transaction anomalies detected.")

# ==================================================
# TAB 2
# ==================================================
with tab2:
    st.subheader("⚖️ Arm’s Length Pricing Benchmark")

    pricing_metrics = [
        "EBITDA_Margin",
        "RPT_Expense_to_EBITDA"
    ]

    pricing_long = filtered_df[pricing_metrics].melt(
        var_name="Metric",
        value_name="Company Value"
    )

    peer_pricing = peer_df[pricing_metrics].median().reset_index()
    peer_pricing.columns = ["Metric", "Industry Median"]

    pricing_compare = pd.merge(
        pricing_long,
        peer_pricing,
        on="Metric"
    )

    fig2 = px.bar(
        pricing_compare,
        x="Metric",
        y=["Company Value", "Industry Median"],
        barmode="group",
        title="Company vs Industry Arm’s Length Benchmark"
    )
    fig2.update_layout(
        title_font_size=34,
        xaxis_title_font_size=26,
        yaxis_title_font_size=26,
        xaxis_tickfont_size=22,
        yaxis_tickfont_size=22,
        legend_font_size=22
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig_heat = px.imshow(
        pricing_compare.set_index("Metric"),
        title="Pricing Deviation Heatmap"
    )
    fig_heat.update_layout(
        title_font_size=30,
        xaxis_title_font_size=26,  # ✅ FIXED
        yaxis_title_font_size=26,  # ✅ FIXED
        xaxis_tickfont_size=22,    # ✅ FIXED
        yaxis_tickfont_size=22     # ✅ FIXED
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    fig_scatter = px.scatter(
        peer_df,
        x="RPT_Expense_to_EBITDA",
        y="EBITDA_Margin",
        title="EBITDA vs RPT Expense Intuition"
    )
    fig_scatter.update_layout(
        title_font_size=30,
        xaxis_title_font_size=26,
        yaxis_title_font_size=26,
        xaxis_tickfont_size=22,
        yaxis_tickfont_size=22
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("🧠 Pricing Deviation Insight")

    if risk_state == "high":
        st.error("Below-peer margins combined with RPT exposure may indicate non–arm’s length pricing.")
    else:
        st.success("Pricing margins appear broadly aligned with industry benchmarks.")

# --------------------------------------------------
# RAW DATA
# --------------------------------------------------
with st.expander("📄 View Raw Financial Data"):
    st.dataframe(filtered_df)
