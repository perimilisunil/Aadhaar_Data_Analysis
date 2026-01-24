import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os
import sys
import warnings
from datetime import datetime

# --- 1. INITIALIZATION & PATHS ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Aadhaar Integrity Analytics | National Forensic Suite", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Path Resolution
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
audit_path = os.path.join(project_root, "output", "final_audit_report.parquet")
master_path = os.path.join(project_root, "datasets", "pincode_master_clean.csv")

# Import Forensic PDF Module
try:
    from project_pdf import generate_forensic_dossier
except ImportError:
    def generate_forensic_dossier(*args, **kwargs): return None

# --- 2. DUCKDB ENGINE (CRASH-PROOF DATA LOADING) ---
@st.cache_resource
def get_connection():
    conn = duckdb.connect(database=':memory:')
    if os.path.exists(audit_path):
        # Create a virtual view of the Parquet file. 
        # We calculate integrity_risk_pct and pincode_str inside SQL to save RAM.
        conn.execute(f"""
            CREATE OR REPLACE VIEW audit_view AS 
            SELECT *, 
            CAST(pincode AS VARCHAR) as pincode_str,
            (integrity_score * 10) as integrity_risk_pct,
            CASE 
                WHEN primary_risk_driver = 'age_18_greater' THEN 'Adult Entry Spikes'
                WHEN primary_risk_driver = 'service_delivery_rate' THEN 'Child Biometric Lags'
                WHEN primary_risk_driver = 'demo_age_17_' THEN 'Activity Bursts'
                WHEN primary_risk_driver = 'security_anomaly_score' THEN 'Suspicious Creation'
                ELSE 'Systemic Risk'
            END as risk_diagnosis
            FROM read_parquet('{audit_path}')
        """)
    return conn

db = get_connection()

# --- 3. CSS STYLING (STRETCHED & PROFESSIONAL) ---
st.markdown("""
<style>
    .main-title { font-size: 2.8rem !important; font-weight: 900 !important; color: #1e3a8a; margin-bottom: 0.5rem; letter-spacing: -2px; line-height: 1.1; }
    section[data-testid="stSidebar"] { width: 280px !important; }
    .section-header { font-size: 1.6rem; font-weight: 700; color: #1e3a8a; border-left: 10px solid #ef4444; padding-left: 15px; margin: 25px 0; background: #f1f5f9; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800 !important; color: #1e3a8a !important; }
</style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR WIDGETS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=100)
    st.markdown("---")
    
    # State selection via DuckDB
    state_list = db.execute("SELECT DISTINCT UPPER(state) FROM audit_view WHERE state IS NOT NULL ORDER BY 1").df().iloc[:,0].tolist()
    sel_state = st.selectbox("Select State", ["INDIA"] + state_list)
    
    st.markdown("---")
    st.markdown("### Risk Profiles")
    f1 = st.checkbox("Adult Entry Spikes", value=True)
    f2 = st.checkbox("Child Biometric Lags", value=True)
    f3 = st.checkbox("Unusual Activity Bursts", value=True)
    f4 = st.checkbox("Suspicious Profile Creation", value=True)
    
    active_drivers = []
    if f1: active_drivers.append('age_18_greater')
    if f2: active_drivers.append('service_delivery_rate')
    if f3: active_drivers.append('demo_age_17_')
    if f4: active_drivers.append('security_anomaly_score')
    
    st.markdown("---")
    # Debounced Pincode Enquiry
    with st.form("pin_search"):
        pin_input = st.text_input("Pincode Enquiry:", placeholder="Enter 6-digit PIN")
        pin_submitted = st.form_submit_button("Search PIN")
    
    st.markdown("---")
    if st.button("Download Final Report"):
        with st.spinner("Compiling Dossier..."):
            # Sample for PDF logic
            sample_df = db.execute("SELECT * FROM audit_view LIMIT 5000").df()
            pdf_bytes = generate_forensic_dossier(sample_df, sel_state, project_root, pin_input, "UIDAI_11060")
            if pdf_bytes:
                st.download_button("📥 Download Submission PDF", pdf_bytes, "Forensic_Dossier.pdf", "application/pdf")

# --- 5. SQL QUERY BUILDER ---
where_clause = "WHERE 1=1"
if sel_state != "INDIA":
    where_clause += f" AND UPPER(state) = '{sel_state}'"
if active_drivers:
    driver_str = "', '".join(active_drivers)
    where_clause += f" AND primary_risk_driver IN ('{driver_str}')"

# --- 6. 6-KPI COMMAND ROW ---
kpi_query = f"""
    SELECT 
        COUNT(DISTINCT pincode) as unique_pins,
        SUM(CASE WHEN integrity_risk_pct > 75 THEN 1 ELSE 0 END) as high_risk_sites,
        AVG(integrity_risk_pct) as mean_risk,
        AVG(CASE WHEN age_5_17 > 0 THEN ((bio_age_5_17 + demo_age_5_17) / CAST(age_5_17 AS FLOAT)) * 100 ELSE 0 END) as child_upd,
        COUNT(*) as records
    FROM audit_view {where_clause}
"""
kpi_res = db.execute(kpi_query).df()

st.markdown('<p class="main-title">Aadhaar National Integrity Dashboard</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Audit Scope", sel_state)
k2.metric("Unique Pincodes", f"{int(kpi_res['unique_pins'][0]):,}")
k3.metric("High Risk Sites", f"{int(kpi_res['high_risk_sites'][0]):,}")
k4.metric("Integrity", f"{100 - kpi_res['mean_risk'][0]:.1f}%")
k5.metric("Child Updates", f"{kpi_res['child_upd'][0]:.1f}%")
k6.metric("Records Analyzed", f"{int(kpi_res['records'][0]):,}")

st.markdown("---")

# --- 7. TABS ---
t1, t2, t3, t4, t5 = st.tabs(["Executive Overview", "Behavioral DNA", "Strategic Action", "Risk Drives", "Pincode Drilldown"])

# TAB 1: EXECUTIVE OVERVIEW
with t1:
    st.markdown('<div class="section-header">National Service Demand vs. Forensic Risk</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        # Lifecycle Bar
        life_query = f"SELECT state, SUM(age_0_5) as Infants, SUM(age_5_17) as Children, SUM(age_18_greater) as Adults FROM audit_view {where_clause} GROUP BY state LIMIT 10"
        life_df = db.execute(life_query).df()
        fig_life = px.bar(life_df, x='state', y=['Infants', 'Children', 'Adults'], barmode='group', title="Lifecycle Service Demand", template="plotly_white")
        st.plotly_chart(fig_life, use_container_width=True)
    with col2:
        # Risk Pie
        pie_query = f"SELECT risk_diagnosis, COUNT(*) as count FROM audit_view {where_clause} GROUP BY 1"
        pie_df = db.execute(pie_query).df()
        fig_pie = px.pie(pie_df, names='risk_diagnosis', values='count', hole=0.4, title="Risk Profile Composition")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Regional Integrity Hierarchy</div>', unsafe_allow_html=True)
    tree_query = f"SELECT state, district, AVG(integrity_risk_pct) as risk, COUNT(*) as vol FROM audit_view {where_clause} GROUP BY 1, 2"
    tree_df = db.execute(tree_query).df()
    fig_tree = px.treemap(tree_df, path=['state', 'district'], values='vol', color='risk', color_continuous_scale='RdYlGn_r', height=600)
    st.plotly_chart(fig_tree, use_container_width=True)

# TAB 2: BEHAVIORAL DNA
with t2:
    st.markdown('<div class="section-header">Automated Risk Profiling: Behavioral DNA</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        img2 = os.path.join(project_root, "output", "charts", "02_risk_leaderboard.png")
        if os.path.exists(img2): st.image(img2, use_container_width=True, caption="Top 25 Priority Districts")
    with r2:
        img5 = os.path.join(project_root, "output", "ML_Anomaly_charts", "05_ml_threat_radar.png")
        if os.path.exists(img5): st.image(img5, use_container_width=True, caption="Behavioral Threat Radar")
    
    # Live Matrix
    heat_query = f"SELECT district, AVG(integrity_risk_pct) as Risk, AVG(age_18_greater) as Spikes, AVG(security_anomaly_score) as Fraud FROM audit_view {where_clause} GROUP BY 1 LIMIT 20"
    heat_df = db.execute(heat_query).df().set_index('district')
    fig_heat = px.imshow(heat_df, color_continuous_scale='YlOrRd', title="Forensic DNA Scorecard")
    st.plotly_chart(fig_heat, use_container_width=True)

# TAB 3: STRATEGIC ACTION
with t3:
    st.markdown('<div class="section-header">Root-Cause Modelling & Audit Directives</div>', unsafe_allow_html=True)
    img11 = os.path.join(project_root, "output", "Deep_Analysis", "11_strategic_portfolio.png")
    if os.path.exists(img11): st.image(img11, use_container_width=True)
    
    st.markdown("### High-Priority Forensic Audit List")
    audit_query = f"SELECT district, pincode_str, integrity_risk_pct, risk_diagnosis FROM audit_view {where_clause} ORDER BY integrity_risk_pct DESC LIMIT 45"
    audit_df = db.execute(audit_query).df()
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

# TAB 4: RISK DRIVES (Operational Friction)
with t4:
    st.markdown('<div class="section-header">The Administrative Pressure Index</div>', unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        img9 = os.path.join(project_root, "output", "Deep_Analysis", "09_state_anomaly_concentration.png")
        if os.path.exists(img9): st.image(img9, use_container_width=True)
    with f_col2:
        img8 = os.path.join(project_root, "output", "Deep_Analysis", "08_global_feature_importance.png")
        if os.path.exists(img8): st.image(img8, use_container_width=True)

    # Live Friction Bar
    friction_query = f"SELECT district, AVG(demo_age_17_) as workload, AVG(integrity_risk_pct) as risk FROM audit_view {where_clause} GROUP BY 1 ORDER BY risk DESC LIMIT 15"
    friction_df = db.execute(friction_query).df()
    fig_friction = go.Figure()
    fig_friction.add_trace(go.Bar(x=friction_df['district'], y=friction_df['workload'], name="Maintenance Stress", marker_color='#197ADB'))
    fig_friction.add_trace(go.Bar(x=friction_df['district'], y=friction_df['risk'], name="Forensic Threat", marker_color='#E74C3C'))
    fig_friction.update_layout(title="Operational Friction: Contextual Analysis", barmode='group', template="plotly_white")
    st.plotly_chart(fig_friction, use_container_width=True)

# TAB 5: PINCODE DRILLDOWN
with t5:
    if pin_input:
        st.markdown(f'<div class="section-header">Forensic Investigation: PIN {pin_input}</div>', unsafe_allow_html=True)
        pin_data = db.execute(f"SELECT *, risk_diagnosis FROM audit_view WHERE pincode_str = '{pin_input}'").df()
        
        if not pin_data.empty:
            row = pin_data.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Local Risk Score", f"{row['integrity_risk_pct']:.1f}%")
            m2.metric("Primary Diagnosis", row['risk_diagnosis'])
            m3.metric("District Context", row['district'])
            
            # Peer analysis inside the district
            dist = row['district']
            peer_query = f"SELECT pincode_str, integrity_risk_pct FROM audit_view WHERE district = '{dist}' ORDER BY integrity_risk_pct DESC"
            peer_df = db.execute(peer_query).df()
            fig_peer = px.bar(peer_df, x='pincode_str', y='integrity_risk_pct', title=f"Risk Hierarchy: {dist} Cluster", color='integrity_risk_pct', color_continuous_scale='Reds')
            st.plotly_chart(fig_peer, use_container_width=True)
        else:
            st.error("Pincode not found in audit database.")
    else:
        st.info("Please enter a 6-digit Pincode in the sidebar and click Search.")

# Footer
st.markdown("---")
st.caption(f"Aadhaar National Integrity Analytics |Sync: {datetime.now().strftime('%Y-%m-%d')}")
