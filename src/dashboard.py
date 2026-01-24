import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from project_pdf import generate_forensic_dossier
import os
import warnings
import duckdb
import sys

# --- 1. INITIALIZATION ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Aadhaar Integrity Analytics | National Forensic Suite", layout="wide")

# --- 2. SESSION STATE MANAGEMENT (BEFORE DATA LOAD) ---
if 'pincode_val' not in st.session_state:
    st.session_state.pincode_val = ""
if 'pincode_query' not in st.session_state:
    st.session_state.pincode_query = ""

def clear_pincode():
    st.session_state.pincode_val = ""

def reset_search():
    st.session_state.pincode_query = ""

# --- 3. CSS---
st.markdown("""
<style>
    .main-title { 
        font-size: 2.8rem !important; 
        font-weight: 900 !important; 
        color: #1e3a8a; 
        margin-bottom: 0.5rem; 
        letter-spacing: -2px;
        line-height: 1.1;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 1.2rem !important; 
        font-weight: 800 !important;   
        margin-bottom: 10px !important;
    }
    .section-header { 
        font-size: 1.6rem; 
        font-weight: 700; 
        color: #1e3a8a; 
        border-left: 10px solid #ef4444; 
        padding-left: 15px; 
        margin: 25px 0; 
        background: #f1f5f9; 
    }
    [data-testid="stMetric"] { 
        background: white; 
        border: 1px solid #e2e8f0; 
        border-radius: 12px; 
        padding: 15px; 
    }
    [data-testid="stMetricValue"] { 
        font-size: 1.5rem !important;
        white-space: nowrap; 
        font-weight: 800 !important; 
        color: #1e3a8a !important; 
    }
</style>
""", unsafe_allow_html=True)

# mapping for internal keys
label_fix = {
    'age_18_greater': 'Adult Entry Spikes', 
    'service_delivery_rate': 'Child Biometric Lags',
    'demo_age_17_': 'Activity Bursts', 
    'security_anomaly_score': 'Suspicious Creation'
}

# --- 4. DEFINE ROOT PATH GLOBALLY ---
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 5. OPTIMIZED DATA LOADING ---
@st.cache_data(ttl=1800, show_spinner="Loading audit data...")
def load_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audit_path = os.path.join(project_root, "output", "final_audit_report.parquet")
    master_path = os.path.join(project_root, "datasets", "pincode_master_clean.csv")
    
    if not os.path.exists(audit_path): 
        return None
    
    # CRITICAL: Use context manager to auto-close DuckDB connection
    with duckdb.connect(database=':memory:') as con:
        # OPTIMIZED: Sample 50% instead of 99.9% to reduce memory
        query = f"""
            SELECT * FROM read_parquet('{audit_path}')
            WHERE integrity_score > 5
            UNION ALL
            SELECT * FROM read_parquet('{audit_path}')
            WHERE integrity_score <= 5
            USING SAMPLE 48% (bernoulli)
        """
        df = con.execute(query).df()
    
    # MEMORY OPTIMIZATION: Convert to smaller dtypes early
    df['integrity_score'] = df['integrity_score'].astype('float32')
    df['pincode_str'] = df['pincode'].astype(str).str.split('.').str[0].str.zfill(6)
    
    # GEOGRAPHIC RESCUE (Removing Unknowns)
    if os.path.exists(master_path):
        # Only load required columns
        m_df = pd.read_csv(master_path, usecols=['pincode', 'statename'])
        m_df['pincode_str'] = m_df['pincode'].astype(str).str.split('.').str[0].str.zfill(6)
        state_lookup = m_df.set_index('pincode_str')['statename'].to_dict()
        
        # Replace UNKNOWN states with data from Master Lookup
        invalid = ['UNKNOWN', 'NAN', 'NONE', '0', 'OTHER/UNCATEGORIZED', 'UNCATEGORIZED']
        df['state'] = df['state'].astype(str).str.upper().str.strip()
        df['state'] = np.where(df['state'].isin(invalid), df['pincode_str'].map(state_lookup), df['state'])
        
        # Free memory
        del m_df, state_lookup

    # Final Filter: Remove any records that are still Unknown
    df = df[~df['state'].isna() & (df['state'] != 'NAN') & (df['state'] != 'UNKNOWN')]
    
    # DASHBOARD METRICS
    df['integrity_risk_pct'] = (df['integrity_score'] * 10).clip(0, 100).round(2).astype('float32')
    label_map = {
        'age_18_greater': 'Adult Entry Spikes', 
        'service_delivery_rate': 'Child Biometric Lags', 
        'demo_age_17_': 'Activity Bursts', 
        'security_anomaly_score': 'Suspicious Creation'
    }
    df['risk_diagnosis'] = df['primary_risk_driver'].map(label_map).fillna("Systemic Risk")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df

# --- 6. LOAD DATA ---
df = load_data()

# STOP if data fails to load
if df is None:
    st.error("⚠️ Dataset not found. Check your file paths.")
    st.stop()

# Optional: Show memory usage in sidebar for debugging
df_size_mb = sys.getsizeof(df) / 1_000_000

# --- 7. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=120)
    st.markdown("---")
    
    # Memory monitor (optional - remove in production)
    st.caption(f"📊 Data: {len(df):,} rows | {df_size_mb:.1f}MB")
    
    # Filtered State List
    state_list = sorted([s for s in df['state'].unique() if s != 'OTHER/UNCATEGORIZED'])
    sel_state = st.selectbox("Select State", ["INDIA"] + state_list, on_change=reset_search)
    
    st.markdown("---")
    st.markdown("### Risk Profiles")
    f1 = st.checkbox("Adult Entry Spikes", value=True)
    f2 = st.checkbox("Child Biometric Lags", value=True)
    f3 = st.checkbox("Unusual Activity Bursts", value=True)
    f4 = st.checkbox("Suspicious Profile Creation", value=True)
    
    risk_map = {
        'age_18_greater': f1, 
        'service_delivery_rate': f2, 
        'demo_age_17_': f3, 
        'security_anomaly_score': f4
    }
    active_drivers = [k for k, v in risk_map.items() if v]

    st.markdown("---")
    
    # Pincode Search Form
    with st.form("pincode_scan_form", clear_on_submit=False):
        st.markdown("### Pincode Enquiry")
        search_pin = st.text_input(
            "Enter 6-digit PIN", 
            placeholder=" ", 
            key="pincode_query"
        )
        submit_search = st.form_submit_button("Analyze Pincode")

    # Initial view_df state
    if sel_state == "INDIA":
        view_df = df.copy()
    else:
        view_df = df[df['state'] == sel_state]
   
    target_obj = None
    
    # Process deep scan if button clicked or value exists
    if st.session_state.pincode_query:
        search_str = str(st.session_state.pincode_query).strip()
        match = df[df['pincode_str'] == search_str]
        
        if not match.empty:
            target_obj = match.iloc[0]
            sel_state = target_obj['state']
            view_df = df[df['state'] == sel_state]
            
            if submit_search:
                st.sidebar.success(f"PINCODE : {target_obj['district']}")
        else:
            if submit_search:
                st.sidebar.error("PINCODE not found in forensic database")
    
    # Apply forensic risk filters
    view_df = view_df[view_df['primary_risk_driver'].isin(active_drivers)]

    # --- DATE FILTER ---
    st.markdown("---")
    st.markdown("### Select Month")
    
    all_periods = df['date'].dt.to_period('M').dropna().unique()
    all_months = sorted(all_periods)
    month_labels = [m.strftime('%B %Y') for m in all_months]
    
    col_from, col_to = st.columns(2)
    
    with col_from:
        start_label = st.selectbox("From", options=month_labels, index=0)
        
    with col_to:
        end_label = st.selectbox("To", options=month_labels, index=len(month_labels)-1)

    start_period = all_months[month_labels.index(start_label)]
    end_period = all_months[month_labels.index(end_label)]

    if start_period > end_period:
        st.error("Error: 'From' date must be before 'To' date.")
    else:
        start_date = start_period.start_time
        end_date = end_period.end_time
        view_df = view_df[(view_df['date'] >= start_date) & (view_df['date'] <= end_date)]
        st.caption(f"Showing data from {start_label} to {end_label}")
    
    # --- PDF EXPORT (OPTIMIZED) ---
    st.markdown("---")
    st.markdown("### Export Final Report")
    
    # Warning for large datasets
    if len(view_df) > 100000:
        st.warning(f"⚠️ Large dataset ({len(view_df):,} rows). PDF may take 30-60 seconds.")
    
    if st.button("Download Report"):
        try:
            with st.spinner("Compiling National & Tactical Evidence..."):
                # CRITICAL FIX: Pass filtered view_df instead of full df
                pdf_bytes = generate_forensic_dossier(
                    df=view_df,  # Changed from df to view_df
                    state_name=sel_state, 
                    root_path=root_path, 
                    search_pin=st.session_state.pincode_query,
                    team_id="UIDAI_11060"
                )
                
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download Submission PDF",
                        data=pdf_bytes,
                        file_name=f"AadhaarSetu_ProjectReport.pdf",
                        mime="application/pdf"
                    )
                    st.success("Report Compiled. Ready for Submission.")
                else:
                    st.warning("PDF generation returned empty.")
        except MemoryError:
            st.error("⚠️ Memory limit exceeded. Try filtering to a smaller region or date range.")
        except Exception as e:
            st.error(f"System Error: {str(e)}")

# --- 8. MAIN DASHBOARD ---
st.markdown('<p class="main-title">Aadhaar National Integrity Dashboard</p>', unsafe_allow_html=True)

# --- 6-KPI COMMAND ROW ---
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: 
    st.metric("Audit Scope", sel_state if sel_state != "NATIONAL OVERVIEW" else "INDIA")
with k2: 
    st.metric("Unique Pincodes", f"{view_df['pincode'].nunique():,}")
with k3: 
    st.metric("High Risk Sites", len(view_df[view_df['integrity_risk_pct'] > 75]))
with k4: 
    st.metric("Integrity", f"{100 - view_df['integrity_risk_pct'].mean():.1f}%")
with k5: 
    st.metric("Child Biometric Updates", f"{view_df['service_delivery_rate'].mean():.1f}%")
with k6: 
    st.metric("Records Analyzed", f"{len(view_df):,}") 

st.markdown("---")

# --- 9. TABS ---
t1, t2, t3, t4, t5 = st.tabs(["Executive Overview", "Behavioral DNA", "Strategic Action", "Risk Drives", "Pincode Drilldown"])

with t1:
    st.markdown('<div class="section-header">National Service Demand vs. Forensic Risk Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # LIVE DEMOGRAPHIC LIFECYCLE CHART
        infant_gen = view_df['age_0_5'].sum()
        infant_upd = 0 
        
        child_gen = view_df['age_5_17'].sum()
        child_upd = view_df['bio_age_5_17'].sum() + view_df['demo_age_5_17'].sum()
        
        adult_gen = view_df['age_18_greater'].sum()
        adult_upd = view_df['bio_age_17_'].sum() + view_df['demo_age_17_'].sum()

        lifecycle_data = pd.DataFrame({
            'Age Group': ['Infants (0-5)', 'Infants (0-5)', 'Children (5-17)', 'Children (5-17)', 'Adults (18+)', 'Adults (18+)'],
            'Activity': ['New Enrolment', 'Maintenance/Updates', 'New Enrolment', 'Maintenance/Updates', 'New Enrolment', 'Maintenance/Updates'],
            'Volume': [infant_gen, infant_upd, child_gen, child_upd, adult_gen, adult_upd]
        })

        fig_life = px.bar(
            lifecycle_data, 
            x='Age Group', 
            y='Volume', 
            color='Activity', 
            barmode='group',
            text_auto='.3s', 
            title=f"<b>Lifecycle Service Demand: {sel_state}</b>",
            color_discrete_map={'New Enrolment': '#3498DB', 'Maintenance/Updates': '#2ECC71'},
            template="plotly_white"
        )

        fig_life.update_layout(
            height=500,
            margin=dict(t=50, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            yaxis_title="Transaction Volume",
            xaxis_title=""
        )

        st.plotly_chart(fig_life,width='stretch')      
    
    with col2:
        pie_data = view_df['risk_diagnosis'].value_counts().reset_index()
        fig_pie = px.pie(
            pie_data, 
            values='count', 
            names='risk_diagnosis', 
            hole=0.4, 
            height=550, 
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title=f"Risk Profile Composition: {sel_state}"
        )
        st.plotly_chart(fig_pie,width='stretch')

    # TREEMAP
    st.markdown('<div class="section-header">Regional Integrity Hierarchy</div>', unsafe_allow_html=True)
    tree_view_df = view_df[view_df['state'] != 'OTHER/UNCATEGORIZED']

    if not tree_view_df.empty:
        tree_agg = tree_view_df.groupby(['state', 'district']).agg({
            'integrity_risk_pct': 'mean',
            'risk_diagnosis': lambda x: x.mode()[0] if not x.empty else "Stable",
            'pincode': 'count' 
        }).reset_index()

        tree_agg = tree_agg.rename(columns={
            'integrity_risk_pct': 'risk',
            'risk_diagnosis': 'driver',
            'pincode': 'volume'
        })

        color_max = tree_agg['risk'].quantile(0.95)
        if color_max < 15: 
            color_max = 15 

        fig_tree = px.treemap(
            tree_agg, 
            path=[px.Constant("INDIA"), 'state', 'district'], 
            values='volume',           
            color='risk',              
            color_continuous_scale='RdYlGn_r', 
            range_color=[0, color_max],
            custom_data=['state', 'district', 'risk', 'driver'],
            height=750                 
        )

        fig_tree.update_traces(
            textinfo="label+value",
            texttemplate="<b>%{label}</b>",
            hovertemplate="""
            <b>State:</b> %{customdata[0]}<br>
            <b>District:</b> %{customdata[1]}<br>
            <b>Risk Intensity:</b> %{customdata[2]:.2f}%<br>
            <b>Primary Threat:</b> %{customdata[3]}
            <extra></extra>""",
            insidetextfont_size=14,
            textposition="middle center"
        )

        fig_tree.update_layout(
            margin=dict(t=30, l=10, r=10, b=10),
            coloraxis_colorbar=dict(title="Relative Risk %", ticksuffix="%")
        )

        st.plotly_chart(fig_tree, width='stretch')
    
    # LIVE TREND ANALYSIS
    st.markdown('<div class="section-header">Administrative Pulse: Risk & Compliance Trends</div>', unsafe_allow_html=True)
    
    pulse_df = view_df.groupby(view_df['date'].dt.to_period('M')).agg({
        'integrity_risk_pct': 'mean',
        'service_delivery_rate': 'mean'
    }).reset_index()
    pulse_df['Risk'] = pulse_df['integrity_risk_pct'].clip(0, 100).round(1)
    pulse_df['Compliance'] = pulse_df['service_delivery_rate'].clip(0, 100).round(1)
    pulse_df['Month'] = pulse_df['date'].astype(str)

    fig_pulse = go.Figure()
    fig_pulse.add_trace(go.Bar(
        x=pulse_df['Month'], 
        y=pulse_df['Compliance'],
        name='MBU Compliance % (Efficiency)',
        marker_color='#27AE60',
        opacity=0.7,
        text=pulse_df['Compliance'].apply(lambda x: f"{x}%"),
        textposition='inside'
    ))
    fig_pulse.add_trace(go.Scatter(
        x=pulse_df['Month'], 
        y=pulse_df['Risk'],
        name='Risk Intensity % (Security Threat)',
        mode='lines+markers',
        line=dict(color='#E74C3C', width=4),
        marker=dict(size=10, symbol='diamond')
    ))

    fig_pulse.update_layout(
        title=f"<b>Forensic Pulse: {sel_state} Operational Health</b>",
        xaxis_title="Audit Month",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 115], 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig_pulse,width='stretch')

with t2:
    st.markdown('<div class="section-header">Automated Risk Profiling: Characterizing Systemic Anomalies</div>', unsafe_allow_html=True)
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        img2_path = os.path.join(root_path, "output", "charts", "02_risk_leaderboard.png")
        if os.path.exists(img2_path):
            st.image(img2_path, width='stretch', caption="Chart 02: Top 25 Priority Audit Districts")
    
    with row1_col2:
        img3_path = os.path.join(root_path, "output", "charts", "03_risk_treemap.png")
        if os.path.exists(img3_path):
            st.image(img3_path, width='stretch', caption="Chart 03: National Integrity Hierarchy")

    st.markdown('<div class="section-header">Regional Forensic DNA (Live Investigative Matrix)</div>', unsafe_allow_html=True)
    
    heat_df = view_df.groupby('district').agg({
        'age_18_greater': 'mean',
        'service_delivery_rate': 'mean',
        'demo_age_17_': 'mean',
        'security_anomaly_score': 'mean'
    }).tail(20)

    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())

    fig7 = px.imshow(
        heat_norm,
        labels=dict(x="Forensic Driver", y="District", color="Relative Intensity"),
        x=['Adult Spikes', 'Child Compliance', 'Activity Bursts', 'Fraud Index'],
        y=heat_norm.index,
        color_continuous_scale='YlOrRd',
        aspect="auto",
        title=f"<b>Chart 07: Normalized DNA Scorecard (Fingerprinting Fraud Types) {sel_state}</b>"
    )
    fig7.update_traces(hovertemplate="District: %{y}<br>Driver: %{x}<br>Relative Intensity: %{z:.2f}")
    st.plotly_chart(fig7,width='stretch')

    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        img5_path = os.path.join(root_path, "output", "ML_Anomaly_charts", "05_ml_threat_radar.png")
        if os.path.exists(img5_path):
            st.image(img5_path, width='stretch', caption="Chart 05: Behavioral Signature Radar")
    
    with row3_col2:
        img6_path = os.path.join(root_path, "output", "ML_Anomaly_charts", "06_ml_forensic_scorecard.png")
        if os.path.exists(img6_path):
            st.image(img6_path, width='stretch', caption="Chart 06: Forensic Magnitude Scorecard")

with t3:
    st.markdown('<div class="section-header">Root-Cause Modelling & Strategic Resource Allocation</div>', unsafe_allow_html=True)
    img11_path = os.path.join(root_path, "output", "Deep_Analysis", "11_strategic_portfolio.png")
    if os.path.exists(img11_path):
        st.image(img11_path, width='stretch', caption="Chart 11: National Policy Zones")

    st.markdown('<div class="section-header">Regional Risk Driver Impact (Live Analysis)</div>', unsafe_allow_html=True)
    driver_impact = view_df['risk_diagnosis'].value_counts().reset_index()
    fig8 = px.bar(
        driver_impact, 
        x='risk_diagnosis', 
        y='count', 
        color='risk_diagnosis',
        title=f"<b>Chart 08: Volume of Primary Threat Drivers in {sel_state} Active Scope</b>",
        labels={'risk_diagnosis': 'ML Diagnosis', 'count': 'Number of Impacted Records'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig8,width='stretch')

    st.markdown('<div class="section-header">Chart 10: High-Priority Forensic Audit List</div>', unsafe_allow_html=True)
    st.write("The following sites have been flagged by the **Isolation Forest** model for manual document verification.")
    
    action_plan = {
        'Adult Entry Spikes': 'Enrolment Form Audit',
        'Child Biometric Lags': 'Deploy Mobile Van',
        'Activity Bursts': 'Operator ID Freeze',
        'Suspicious Creation': 'Manual ID Verification'
    }
    
    audit_table = view_df.sort_values('integrity_risk_pct', ascending=False).head(45).copy()
    audit_table['Recommended Action'] = audit_table['risk_diagnosis'].map(action_plan)

    st.dataframe(
        audit_table[['district', 'pincode', 'integrity_risk_pct', 'risk_diagnosis', 'Recommended Action']],
        column_config={
            "integrity_risk_pct": st.column_config.ProgressColumn(
                "Risk Severity",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "risk_diagnosis": "Forensic Diagnosis"
        },
        width='stretch',
        hide_index=True
    )
    
    st.download_button(
        label="Download Regional Action Plan",
        data=audit_table.to_csv(index=False),
        file_name=f"Audit_Plan_{sel_state}.csv",
        mime='text/csv',
    )

with t4:
    st.markdown('<div class="section-header">National Service Baseline [Static Benchmarks]</div>', unsafe_allow_html=True)
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        img9_path = os.path.join(root_path, "output", "Deep_Analysis", "09_state_anomaly_concentration.png")
        if os.path.exists(img9_path):
            st.image(img9_path,width='stretch')
    with row1_col2:
        img8_path = os.path.join(root_path, "output", "Deep_Analysis", "08_global_feature_importance.png")
        if os.path.exists(img8_path):
            st.image(img8_path,width='stretch')

    st.markdown('<div class="section-header">The Administrative Pressure Index: Workload vs. Security Oversight</div>', unsafe_allow_html=True)
    st.info("""
    **Forensic Narrative:** This chart identifies **'Burnout Zones'**. 
    When Maintenance Workload (Blue) and Forensic Risk (Red) are both high, the system is at critical friction.
    """)

    friction_df = view_df.groupby(['state', 'district']).agg({
        'integrity_risk_pct': 'mean',
        'demo_age_17_': 'mean',
        'age_18_greater': 'mean'
    }).reset_index()

    friction_df['display_name'] = friction_df['state'] + " - " + friction_df['district']
    friction_df['Forensic_Pressure'] = friction_df['integrity_risk_pct'].clip(0, 100).round(1)
    friction_df['Workload_Pressure'] = ((friction_df['demo_age_17_'] / (friction_df['demo_age_17_'] + friction_df['age_18_greater'] + 1)) * 100).clip(0, 100).round(1)
    friction_df['Total_Friction'] = friction_df['Forensic_Pressure'] + friction_df['Workload_Pressure']
    friction_df = friction_df.sort_values('Total_Friction', ascending=False).head(15)

    fig_friction = go.Figure()

    fig_friction.add_trace(go.Bar(
        x=friction_df['display_name'],
        y=friction_df['Workload_Pressure'],
        name='Maintenance Workload (Operator Stress)',
        marker_color="#197ADB",
        text=friction_df['Workload_Pressure'].apply(lambda x: f"{x}%"),
        textposition='outside',
        textangle=-90
    ))

    fig_friction.add_trace(go.Bar(
        x=friction_df['display_name'],
        y=friction_df['Forensic_Pressure'],
        name='Forensic Risk (Security Threat)',
        marker_color='#E74C3C',
        text=friction_df['Forensic_Pressure'].apply(lambda x: f"{x}%"),
        textposition='outside',
        textangle=-90
    ))

    fig_friction.update_layout(
        barmode='group',
        title="Operational Friction: Contextual District Analysis",
        xaxis_title="Region (State - District)",
        yaxis_title="Pressure Index (%)",
        yaxis_range=[0, 130], 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=650, 
        xaxis=dict(tickangle=45, tickfont=dict(size=11)) 
    )
    
    st.plotly_chart(fig_friction, width='stretch')

    st.success("""
    **Administrative Directive:**
    - **State-Level Hotspots:** Look for states appearing multiple times in this Top 15. This indicates a systemic state-wide failure in balancing workload with security.
    """)

with t5:
    if st.session_state.pincode_query and target_obj is not None:
        st.markdown(f'<div class="section-header">Forensic Investigation: PIN {st.session_state.pincode_query}</div>', unsafe_allow_html=True)
        
        # RESOURCE PIVOT
        district_all = df[df['district'] == target_obj['district']].copy()
        safe_haven = district_all.sort_values('integrity_risk_pct', ascending=True).iloc[0]

        if target_obj['integrity_risk_pct'] > 60:
            st.warning(f"**BREACH DETECTED:** Suspend Adult Enrolment at {st.session_state.pincode_query}. Reroute to **PIN {safe_haven['pincode_str']}**.")
        
        # 15-PIN CLUSTER
        district_agg = district_all.groupby('pincode_str').agg({
            'state': 'first',
            'district': 'first',
            'integrity_risk_pct': 'mean',
            'risk_diagnosis': lambda x: x.mode()[0] if not x.empty else "N/A"
        }).reset_index()

        action_map = {
            'Adult Entry Spikes': 'Forensic Audit: Verify 18+ Form Authenticity',
            'Child Biometric Lags': 'Outreach: Deploy Mobile Update Van',
            'Activity Bursts': 'Technical: Inspect Operator Software Logs',
            'Suspicious Creation': 'Security: Manual Identity Cross-Verification'
        }
        district_agg['Required Action'] = district_agg['risk_diagnosis'].map(action_map).fillna("Monitor Activity")

        peers = district_agg.sort_values('integrity_risk_pct', ascending=False).reset_index(drop=True)
        
        match_indices = peers.index[peers['pincode_str'] == st.session_state.pincode_query].tolist()
        
        if not match_indices:
            st.error("PIN found in master but missing in district aggregation.")
        else:
            t_idx = match_indices[0]
            start, end = max(0, t_idx - 7), min(len(peers), t_idx + 8)
            cluster = peers.iloc[start:end].copy()

            cluster['color_logic'] = cluster['pincode_str'].apply(
                lambda x: '#ef4444' if x == st.session_state.pincode_query else "#1278DF"
            )

            fig_grad = px.bar(
                cluster.sort_values('integrity_risk_pct', ascending=True), 
                x='integrity_risk_pct', 
                y='pincode_str', 
                orientation='h',
                text_auto='.1f',
                title=f"Risk Hierarchy: {target_obj['district']} Cluster (Period Average)",
                labels={'integrity_risk_pct': 'Average Risk %', 'pincode_str': 'PIN'}
            )
            
            fig_grad.update_traces(marker_color=cluster.sort_values('integrity_risk_pct', ascending=True)['color_logic'])
            fig_grad.update_layout(
                xaxis_range=[0, 100], 
                yaxis_type='category',
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig_grad,width='stretch')
            
            st.markdown("**Field Investigative Evidence**")
            
            display_table = cluster[['state', 'district', 'pincode_str', 'integrity_risk_pct', 'risk_diagnosis', 'Required Action']].rename(columns={
                'state': 'State', 
                'district': 'District', 
                'pincode_str': 'Pincode', 
                'integrity_risk_pct': 'Risk Score %', 
                'risk_diagnosis': 'Forensic Diagnosis',
                'Required Action': 'Required Action'
            })
            
            def highlight_target(row):
                is_target = str(row['Pincode']).strip() == st.session_state.pincode_query
                return ['background-color: #fee2e2; font-weight: bold' if is_target else '' for _ in row]

            st.table(display_table.style.apply(highlight_target, axis=1))

            st.download_button(
                label="Download Field Work-Order",
                data=cluster.to_csv(index=False),
                file_name=f"Forensic_Audit_{st.session_state.pincode_query}.csv",
                mime='text/csv'
            )
    else:
        st.info("**National Aadhar Portal Ready.** Please enter a Pincode in the sidebar.")
