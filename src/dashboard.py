import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from project_pdf import generate_forensic_dossier
import os
import warnings

# --- 1. INITIALIZATION ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Aadhaar Integrity Analytics | National Forensic Suite", layout="wide")

# --- 2. SESSION STATE MANAGEMENT ---
if 'pincode_val' not in st.session_state:
    st.session_state.pincode_val = ""

def clear_pincode():
    st.session_state.pincode_val = ""

# --- 2. CSS---
st.markdown("""
<style>
    .main-title { 
        font-size: 2.8rem !important; 
        font-weight: 900 !important; 
        color: #1e3a8a; 
        margin-bottom: 0.5rem; 
        letter-spacing: -2px; /* Professional tight kerning */
        line-height: 1.1;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 1.2rem !important; 
        font-weight: 800 !important;   
        margin-bottom: 10px !important;
    }

    .section-header { font-size: 1.6rem; font-weight: 700; color: #1e3a8a; border-left: 10px solid #ef4444; padding-left: 15px; margin: 25px 0; background: #f1f5f9; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important;white-space: nowrap; font-weight: 800 !important; color: #1e3a8a !important; }
</style>
""", unsafe_allow_html=True)

# mapping for internal keys
label_fix = {
    'age_18_greater': 'Adult Entry Spikes', 
    'service_delivery_rate': 'Child Biometric Lags',
    'demo_age_17_': 'Activity Bursts', 
    'security_anomaly_score': 'Suspicious Creation'
}

# --- 3. GEOGRAPHY & DATA ENGINE ---
    # --- 1. DEFINE ROOT PATH GLOBALLY ---
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. LOAD PRIMARY AUDIT DATA
    audit_path = os.path.join(project_root, "output", "final_audit_report.parquet")
    if not os.path.exists(audit_path): return None
    df = pd.read_parquet(audit_path)

    fcols = data.select_dtypes('float').columns
    icols = data.select_dtypes('integer').columns
    data[fcols] = data[fcols].apply(pd.to_numeric, downcast='float')
    data[icols] = data[icols].apply(pd.to_numeric, downcast='integer')
    # 2. LOAD MASTER REFERENCE (The Rescue File)
    # Expected columns: pincode, district, statename
    master_path = os.path.join(project_root, "datasets", "pincode_master_clean.csv")
    state_lookup, dist_lookup = {}, {}
    
    if os.path.exists(master_path):
        m_df = pd.read_csv(master_path)
        m_df['pincode_str'] = m_df['pincode'].astype(str).str.split('.').str[0].str.zfill(6)
        state_lookup = m_df.set_index('pincode_str')['statename'].to_dict()
        dist_lookup = m_df.set_index('pincode_str')['district'].to_dict()
    
    # 3. NORMALIZE AUDIT DATA
    df['pincode_str'] = df['pincode'].astype(str).str.split('.').str[0].str.zfill(6)
    
    for col in ['state', 'district']:
        df[col] = df[col].astype(str).str.upper().str.strip()

    # 4. THE MASTER RESCUE (Mapping Unknowns to Reality)
    invalid_tags = ['UNKNOWN', 'NAN', 'NONE', '0', 'NULL', '', 'UNDEFINED', 'UNCATEGORIZED']
    
    # If state is unknown, try to get it from the Master lookup
    df['state'] = np.where(
        df['state'].isin(invalid_tags), 
        df['pincode_str'].map(state_lookup), 
        df['state']
    )
    
    # If district is unknown, try to get it from the Master lookup
    df['district'] = np.where(
        df['district'].isin(invalid_tags), 
        df['pincode_str'].map(dist_lookup), 
        df['district']
    )

    # 5. FINAL CLEANUP 
    # Standardize names one last time to fix fragmentation
    fragment_map = {'SPSR NELLORE': 'S.P.S. NELLORE', 'NELLORE': 'S.P.S. NELLORE', 'GURGAON': 'GURUGRAM'}
    df['district'] = df['district'].replace(fragment_map)
    df['state'] = df['state'].replace({'TAMILNADU': 'TAMIL NADU', 'ORISSA': 'ODISHA', 'WESTBENGAL': 'WEST BENGAL'})

    # 6. FILL TRULY LOST DATA (If not in Master file either)
    df['state'] = df['state'].fillna('OTHER/UNCATEGORIZED')
    df['district'] = df['district'].fillna('OTHER/UNCATEGORIZED')

    # 7. PINCODE INTEGRITY LOCK (PIL)
    pil_state = df.groupby('pincode_str')['state'].agg(lambda x: x.mode()[0] if not x.empty else 'UNCATEGORIZED').to_dict()
    pil_dist = df.groupby('pincode_str')['district'].agg(lambda x: x.mode()[0] if not x.empty else 'UNCATEGORIZED').to_dict()
    df['state'] = df['pincode_str'].map(pil_state)
    df['district'] = df['pincode_str'].map(pil_dist)
    
    # 8. ADD DASHBOARD METRICS
    df['integrity_risk_pct'] = (df['integrity_score'] * 10).clip(0, 100).round(2)
    # Map technical keys to sidebar labels
    label_map = {
        'age_18_greater': 'Adult Entry Spikes', 
        'service_delivery_rate': 'Child Biometric Lags',
        'demo_age_17_': 'Activity Bursts', 
        'security_anomaly_score': 'Suspicious Creation'
    }
    df['risk_diagnosis'] = df['primary_risk_driver'].map(label_map).fillna("Systemic Risk")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df
df = load_data()

# --- 3. SIDEBAR ---
if df is not None:
    with st.sidebar:
        # Ensure view_df is initialized so warnings stop
        view_df = df.copy() 
else:
    st.error("Dataset not found. Check your file paths.")
if 'pincode_query' not in st.session_state:
    st.session_state.pincode_query = ""

# This function will run EVERY time you change the state dropdown
def reset_search():
    st.session_state.pincode_query = ""

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=120)
    st.markdown("---")
    if df is not None:
        # Filtered State List
        state_list = sorted([s for s in df['state'].unique() if s != 'OTHER/UNCATEGORIZED'])
        sel_state = st.selectbox("Select State", ["INDIA"] + state_list,on_change=reset_search)
        
        st.markdown("---")
        st.markdown("### Risk Profiles")
        f1 = st.checkbox("Adult Entry Spikes", value=True)
        f2 = st.checkbox("Child Biometric Lags", value=True)
        f3 = st.checkbox("Unusual Activity Bursts", value=True)
        f4 = st.checkbox("Suspicious Profile Creation", value=True)
        
        risk_map = {'age_18_greater': f1, 'service_delivery_rate': f2, 'demo_age_17_': f3, 'security_anomaly_score': f4}
        active_drivers = [k for k, v in risk_map.items() if v]
        
        st.markdown("---")
        
        search_pin = st.text_input("Pincode Enquery: ", placeholder="Enter 6-digit PIN",key="pincode_query")
        if sel_state == "INDIA":
            view_df = df.copy()
        else:
            view_df = df[df['state'] == sel_state]
       
        target_obj = None
        if st.session_state.pincode_query:
            search_str = str(st.session_state.pincode_query).strip()
            df_pins = df['pincode_str'] 

            # Find the match
            match = df[df_pins == search_str]
            
            if not match.empty:
                target_obj = match.iloc[0]
                sel_state = target_obj['state']
                view_df = df[df['state'] == sel_state]
            else:
                st.sidebar.error("PIN not found in database")
        view_df = view_df[view_df['primary_risk_driver'].isin(active_drivers)]
        # --- SIDEBAR DATE FILTER ---
        st.markdown("---")
        st.markdown("### Select Month")
        
        # 1. Prepare the sorted list of unique months in Month-Year format
        all_periods = df['date'].dt.to_period('M').dropna().unique()
        all_months = sorted(all_periods)
        month_labels = [m.strftime('%B %Y') for m in all_months]
        
        # 2. Layout: Two columns for Start and End
        col_from, col_to = st.columns(2)
        
        with col_from:
            start_label = st.selectbox("From", options=month_labels, index=0)
            
        with col_to:
            # Default 'To' is the last month (index=len-1)
            end_label = st.selectbox("To", options=month_labels, index=len(month_labels)-1)

        # 3. Convert selected labels back to actual dates for filtering
        start_period = all_months[month_labels.index(start_label)]
        end_period = all_months[month_labels.index(end_label)]

        # 4. Error Handling: If user picks 'From' date after 'To' date
        if start_period > end_period:
            st.error("Error: 'From' date must be before 'To' date.")
        else:
            # Apply Filter to view_df
            start_date = start_period.start_time
            end_date = end_period.end_time
            view_df = view_df[(view_df['date'] >= start_date) & (view_df['date'] <= end_date)]
            
            st.caption(f"Showing data from {start_label} to {end_label}")
        # --- SIDEBAR PDF EXPORT ---
        st.markdown("---")
        st.markdown("### Export Final Report")
        if st.button("Download Report"):
            try:
                with st.spinner("Compiling National & Tactical Evidence..."):
                    pdf_bytes = generate_forensic_dossier(
                        df=df, 
                        state_name=sel_state, 
                        root_path=root_path, 
                        search_pin=st.session_state.pincode_query,
                        team_id="UIDAI_11060"
                    )
                    
                    if pdf_bytes:
                        st.download_button(
                            label="📥 Download Submission PDF",
                            data=pdf_bytes,
                            file_name=f"UIDAI_11060_AadhaarSetu_ProjectReport.pdf",
                            mime="application/pdf"
                        )
                        st.success("Report Compiled. Ready for Submission.")
            except Exception as e:
                st.error(f"System Error: {str(e)}")
        

st.markdown('<p class="main-title">Aadhaar National Integrity Dashboard</p>', unsafe_allow_html=True)
if df is not None:
    total_unique_pins = df['pincode'].nunique() 
    # --- 6-KPI COMMAND ROW ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.metric("Audit Scope", sel_state if sel_state != "NATIONAL OVERVIEW" else "INDIA")
    with k2: st.metric("Unique Pincodes", f"{view_df['pincode'].nunique():,}")
    with k3: st.metric("High Risk Sites", len(view_df[view_df['integrity_risk_pct'] > 75]))
    with k4: st.metric("Integrity", f"{100 - view_df['integrity_risk_pct'].mean():.1f}%")
    child_upd = view_df['service_delivery_rate'].mean()
    with k5: st.metric("Child Biometric Updates", f"{view_df["service_delivery_rate"].mean():.1f}%")
    
    with k6: st.metric("Records Analyzed", f"{len(view_df):,}") 
    st.markdown("---")

    t1, t2, t3, t4,t5= st.tabs(["Executive Overview", "Behavioral DNA", "Strategic Action", "Risk Drives","Pincode Drilldown"])

    with t1:
        st.markdown('<div class="section-header">National Service Demand vs. Forensic Risk Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # --- LIVE DEMOGRAPHIC LIFECYCLE CHART ---
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

            # 2. Create the Interactive Bar Chart
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

            # 3.Styling
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
            fig_pie = px.pie(pie_data, values='count', names='risk_diagnosis', hole=0.4, height=550, 
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             title=f"Risk Profile Composition: {sel_state}")
            st.plotly_chart(fig_pie, width='stretch')

        # --- THE TREEMAP ---
        st.markdown('<div class="section-header">Regional Integrity Hierarchy </div>', unsafe_allow_html=True)
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

        # 2. Dynamic Color Anchoring
        color_max = tree_agg['risk'].quantile(0.95)
        if color_max < 15: color_max = 15 

        # 3. Create the Treemap
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

        # 4. The Trace Fix (This solves the NaN and the label issue)
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
        # --- TAB 1: LIVE TREND ANALYSIS ---
        st.markdown('<div class="section-header">Administrative Pulse: Risk & Compliance Trends</div>', unsafe_allow_html=True)
        # Aggregate by month for the current state/selection
        pulse_df = view_df.groupby(view_df['date'].dt.to_period('M')).agg({
            'integrity_risk_pct': 'mean',
            'service_delivery_rate': 'mean'
        }).reset_index()
        pulse_df['Risk'] = pulse_df['integrity_risk_pct'].clip(0, 100).round(1)
        pulse_df['Compliance'] = pulse_df['service_delivery_rate'].clip(0, 100).round(1)
        pulse_df['Month'] = pulse_df['date'].astype(str)

        # 2. CREATE THE COMBO CHART
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

        # 3. PROFESSIONAL FORMATTING
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

        st.plotly_chart(fig_pulse, width='stretch')
    # --- TAB 2: ANOMALY CLUSTERING [HYBRID MODE] ---
    with t2:
        # --- TOP ROW: National Baselines---
        st.markdown('<div class="section-header">Automated Risk Profiling: Characterizing Systemic Anomalie</div>', unsafe_allow_html=True)
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            # Chart 2: Priority Leaderboard
            img2_path = os.path.join(root_path, "output", "charts", "02_risk_leaderboard.png")
            if os.path.exists(img2_path):
                st.image(img2_path, width='stretch', caption="Chart 02: Top 25 Priority Audit Districts")
        
        with row1_col2:
            # Chart 3: National Treemap
            img3_path = os.path.join(root_path, "output", "charts", "03_risk_treemap.png")
            if os.path.exists(img3_path):
                st.image(img3_path,width='stretch', caption="Chart 03: National Integrity Hierarchy")

        st.markdown('<div class="section-header">Regional Forensic DNA (Live Investigative Matrix)</div>', unsafe_allow_html=True)
        # We use a heatmap to show how different districts compare across the 4 risk drivers        
        heat_df = view_df.groupby('district').agg({
            'age_18_greater': 'mean',
            'service_delivery_rate': 'mean',
            'demo_age_17_': 'mean',
            'security_anomaly_score': 'mean'
        }).tail(20)

        # Analyst's Trick: Normalize each column (Min-Max Scaling)
        heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())

        fig7 = px.imshow(
            heat_norm,
            labels=dict(x="Forensic Driver", y="District", color="Relative Intensity"),
            x=['Adult Spikes', 'Child Compliance', 'Activity Bursts', 'Fraud Index'],
            y=heat_norm.index,
            color_continuous_scale='YlOrRd',
            aspect="auto",
            title=f"<b>Chart 07: Normalized DNA Scorecard of (Fingerprinting Fraud Types) {sel_state}</b>"
        )
        fig7.update_traces(hovertemplate="District: %{y}<br>Driver: %{x}<br>Relative Intensity: %{z:.2f}")
        st.plotly_chart(fig7, width='stretch')

        row3_col1, row3_col2 = st.columns(2)
        
        with row3_col1:
            # Chart 5: Threat Radar
            img5_path = os.path.join(root_path, "output", "ML_Anomaly_charts", "05_ml_threat_radar.png")
            if os.path.exists(img5_path):
                st.image(img5_path, width='stretch', caption="Chart 05: Behavioral Signature Radar")
        
        with row3_col2:
            # Chart 6: Forensic Scorecard 
            img6_path = os.path.join(root_path, "output", "ML_Anomaly_charts", "06_ml_forensic_scorecard.png")
            if os.path.exists(img6_path):
                st.image(img6_path, width='stretch', caption="Chart 06: Forensic Magnitude Scorecard")


# --- TAB 3: STRATEGIC ACTION (Decision Support System) ---
    with t3:
        # 1. TOP ROW: Static National Strategy
        st.markdown('<div class="section-header">Root-Cause Modelling & Strategic Resource Allocation</div>', unsafe_allow_html=True)
        img11_path = os.path.join(root_path, "output", "Deep_Analysis", "11_strategic_portfolio.png")
        if os.path.exists(img11_path):
            st.image(img11_path, width='stretch', caption="Chart 11: National Policy Zones")

        # 2. MIDDLE ROW: Live Risk Drivers (Chart 08)
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
        st.plotly_chart(fig8, width='stretch')

        # 3. BOTTOM ROW: The Audit Master-List (Chart 10)
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
        
        # Download Button for the Work Order
        st.download_button(
            label="Download Regional Action Plan",
            data=audit_table.to_csv(index=False),
            file_name=f"Audit_Plan_{sel_state}.csv",
            mime='text/csv',
        )
    # --- TAB 4: OPERATIONAL FRICTION ---
    with t4:
        st.markdown('<div class="section-header">National Service Baseline [Static Benchmarks]</div>', unsafe_allow_html=True)
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            img9_path = os.path.join(root_path, "output", "Deep_Analysis", "09_state_anomaly_concentration.png")
            if os.path.exists(img9_path):
                st.image(img9_path, width='stretch')
        with row1_col2:
            img8_path = os.path.join(root_path, "output", "Deep_Analysis", "08_global_feature_importance.png")
            if os.path.exists(img8_path):
                st.image(img8_path, width='stretch')

        st.markdown('<div class="section-header">The Administrative Pressure Index: Workload vs. Security Oversight</div>', unsafe_allow_html=True)
        st.info("""
        **Forensic Narrative:** This chart identifies **'Burnout Zones.'**. 
        When Maintenance Workload (Blue) and Forensic Risk (Red) are both high, the system is at critical friction.
        """)

        # --- 1. DATA PREPARATION (The Context Fix) ---
        friction_df = view_df.groupby(['state', 'district']).agg({
            'integrity_risk_pct': 'mean',
            'demo_age_17_': 'mean',
            'age_18_greater': 'mean'
        }).reset_index()

        # THE NUCLEAR FIX: Create a Unique Display Name
        friction_df['display_name'] = friction_df['state'] + " - " + friction_df['district']

        # Metric 1: Forensic Pressure
        friction_df['Forensic_Pressure'] = friction_df['integrity_risk_pct'].clip(0, 100).round(1)

        # Metric 2: Workload Pressure
        friction_df['Workload_Pressure'] = ((friction_df['demo_age_17_'] / (friction_df['demo_age_17_'] + friction_df['age_18_greater'] + 1)) * 100).clip(0, 100).round(1)

        # Sort and take top 15
        friction_df['Total_Friction'] = friction_df['Forensic_Pressure'] + friction_df['Workload_Pressure']
        friction_df = friction_df.sort_values('Total_Friction', ascending=False).head(15)

        # --- 2. THE VISUAL (Using Display Name) ---
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

        # --- 3. PROFESSIONAL FORMATTING ---
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
        - **State-Level Hotspots:** Look for states appearing multiple times in this Top 15 (e.g., DELHI). This indicates a systemic state-wide failure in balancing workload with security.
        """)
    # --- TAB 5: FORENSIC PIVOT & DRILLDOWN ---
    with t5:
        if search_pin and target_obj is not None:
            st.markdown(f'<div class="section-header">Forensic Investigation: PIN {search_pin}</div>', unsafe_allow_html=True)
            
            # --- THE RESOURCE PIVOT ---
            district_all = df[df['district'] == target_obj['district']].copy()
            safe_haven = district_all.sort_values('integrity_risk_pct', ascending=True).iloc[0]

            if target_obj['integrity_risk_pct'] > 60:
                st.warning(f"**BREACH DETECTED:** Suspend Adult Enrolment at {search_pin}. Reroute to **PIN {safe_haven['pincode_str']}**.")
            
            # --- THE 15-PIN CLUSTER ---
                        # 1. First, group the district data by pincode to prevent SUMMING months
            district_agg = district_all.groupby('pincode_str').agg({
                'state': 'first','district': 'first',
                'integrity_risk_pct': 'mean',
                'risk_diagnosis': lambda x: x.mode()[0] if not x.empty else "N/A"
            }).reset_index()

            action_map = {
                'Adult Entry Spikes': ' Forensic Audit: Verify 18+ Form Authenticity',
                'Child Biometric Lags': 'Outreach: Deploy Mobile Update Van',
                'Activity Bursts': 'Technical: Inspect Operator Software Logs',
                'Suspicious Creation': 'Security: Manual Identity Cross-Verification'
            }
            district_agg['Required Action'] = district_agg['risk_diagnosis'].map(action_map).fillna("Monitor Activity")

            # 2. Now sort and find the target index in the AGGREGATED list
            peers = district_agg.sort_values('integrity_risk_pct', ascending=False).reset_index(drop=True)
            
            match_indices = peers.index[peers['pincode_str'] == search_pin].tolist()
            
            if not match_indices:
                st.error("PIN found in master but missing in district aggregation.")
            else:
                t_idx = match_indices[0]
                # Target-in-middle logic (15 pins)
                start, end = max(0, t_idx - 7), min(len(peers), t_idx + 8)
                cluster = peers.iloc[start:end].copy()

                # Highlight Logic: Red for target, Blue-Grey for peers
                cluster['color_logic'] = cluster['pincode_str'].apply(
                    lambda x: '#ef4444' if x == search_pin else "#1278DF"
                )

                # 3. Create the Bar Chart with fixed scale
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
                st.plotly_chart(fig_grad, width='stretch')
                st.markdown("**Field Investigative Evidence**")
                
                display_table = cluster[['state', 'district', 'pincode_str', 'integrity_risk_pct', 'risk_diagnosis', 'Required Action']].rename(columns={
                    'state': 'State', 'district': 'District', 'pincode_str': 'Pincode', 
                    'integrity_risk_pct': 'Risk Score %', 'risk_diagnosis': 'Forensic Diagnosis','Required Action': 'Required Action'
                })
                def highlight_target(row):
                    is_target = str(row['Pincode']).strip() == search_pin
                    return ['background-color: #fee2e2; font-weight: bold' if is_target else '' for _ in row]

                st.table(display_table.style.apply(highlight_target, axis=1))

                # --- 5. WORK-ORDER DOWNLOAD ---
                st.download_button(
                    label="Download Field Work-Order",
                    data=cluster.to_csv(index=False),
                    file_name=f"Forensic_Audit_{search_pin}.csv",
                    mime='text/csv'
                )
        else:
            st.info("**National Aadhar Portal Ready.** Please enter a Pincode in the sidebar.")
           
