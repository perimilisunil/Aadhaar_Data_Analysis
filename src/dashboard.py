import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from project_pdf import generate_forensic_dossier
import os
import warnings
import traceback

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
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audit_path = os.path.join(project_root, "output", "final_audit_report.parquet")
master_path = os.path.join(project_root, "datasets", "pincode_master_clean.csv")

# --- Safe loader settings
PARQUET_SIZE_THRESHOLD = 200 * 1024 * 1024  # 200 MB threshold - tune to host

def safe_read_parquet(path, columns=None, nrows=None):
    """
    Try to read a parquet file safely:
    - If small, read fully
    - If large, attempt column-limited read; fallback to sampling via pyarrow or CSV
    """
    try:
        fsize = os.path.getsize(path)
    except Exception:
        fsize = 0

    try:
        if fsize == 0 or fsize <= PARQUET_SIZE_THRESHOLD:
            return pd.read_parquet(path, columns=columns)
        else:
            if columns:
                try:
                    return pd.read_parquet(path, columns=columns)
                except Exception:
                    pass
            try:
                import pyarrow.parquet as pq
                tbl = pq.read_table(path, columns=columns)
                df = tbl.to_pandas()
                if nrows and len(df) > nrows:
                    return df.head(nrows)
                return df
            except Exception:
                csv_path = path.replace('.parquet', '.csv')
                if os.path.exists(csv_path):
                    return pd.read_csv(csv_path, usecols=columns, nrows=nrows)
                raise
    except Exception as e:
        raise RuntimeError(f"safe_read_parquet failed for {path} : {e}")

# --- UPDATE: cached safe loader (prevents re-reading on each rerun)
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_safe(parquet_path, master_path=None, sample_when_large=True):
    """
    Loads audit data with safe fallbacks and returns dataframe plus a status dict.
    Cached so host restarts/re-runs don't re-load arbitrarily.
    """
    status = {"sampled": False, "msg": ""}
    if not os.path.exists(parquet_path):
        status["msg"] = f"Parquet file not found: {parquet_path}"
        return None, status

    required_cols = [
        "pincode", "integrity_score", "primary_risk_driver", "date", "state", "district",
        "age_0_5", "age_5_17", "age_18_greater",
        "bio_age_5_17", "demo_age_5_17", "bio_age_17_", "demo_age_17_"
    ]

    try:
        fsize = os.path.getsize(parquet_path)
    except Exception:
        fsize = 0

    try:
        if fsize and fsize > PARQUET_SIZE_THRESHOLD and sample_when_large:
            try:
                df = safe_read_parquet(parquet_path, columns=[c for c in required_cols if c is not None], nrows=None)
            except Exception:
                df = safe_read_parquet(parquet_path, columns=None, nrows=200000)
                status["sampled"] = True
                status["msg"] = "Large file: loaded sample (first 200k rows). For full analysis, precompute charts offline."
        else:
            df = safe_read_parquet(parquet_path, columns=None)
    except Exception as e:
        status["msg"] = f"Failed to read parquet: {e}"
        return None, status

    # cheap normalization
    try:
        df['pincode_str'] = df['pincode'].astype(str).str.split('.').str[0].str.zfill(6)
    except Exception:
        df['pincode_str'] = df.get('pincode', pd.Series(dtype=str)).astype(str).str.zfill(6)

    # load master lookup (minimal)
    state_lookup, dist_lookup = {}, {}
    if master_path and os.path.exists(master_path):
        try:
            m_df = pd.read_csv(master_path, usecols=['pincode','statename','district'], dtype=str)
            m_df['pincode_str'] = m_df['pincode'].astype(str).str.zfill(6)
            state_lookup = m_df.set_index('pincode_str')['statename'].to_dict()
            dist_lookup = m_df.set_index('pincode_str')['district'].to_dict()
        except Exception:
            pass

    # standardize columns cheaply
    for col in ['state','district']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
        else:
            df[col] = 'OTHER/UNCATEGORIZED'

    # master rescue for unknowns
    invalid_tags = ['UNKNOWN', 'NAN', 'NONE', '0', 'NULL', '', 'UNDEFINED', 'UNCATEGORIZED']
    try:
        df['state'] = np.where(df['state'].isin(invalid_tags), df['pincode_str'].map(state_lookup), df['state'])
        df['district'] = np.where(df['district'].isin(invalid_tags), df['pincode_str'].map(dist_lookup), df['district'])
    except Exception:
        pass

    # final cleanup + PIL (cheap ops)
    fragment_map = {'SPSR NELLORE': 'S.P.S. NELLORE', 'NELLORE': 'S.P.S. NELLORE', 'GURGAON': 'GURUGRAM'}
    df['district'] = df['district'].replace(fragment_map)
    df['state'] = df['state'].replace({'TAMILNADU': 'TAMIL NADU', 'ORISSA': 'ODISHA', 'WESTBENGAL': 'WEST BENGAL'})
    df['state'] = df['state'].fillna('OTHER/UNCATEGORIZED')
    df['district'] = df['district'].fillna('OTHER/UNCATEGORIZED')

    try:
        pil_state = df.groupby('pincode_str')['state'].agg(lambda x: x.mode()[0] if not x.empty else 'UNCATEGORIZED').to_dict()
        pil_dist = df.groupby('pincode_str')['district'].agg(lambda x: x.mode()[0] if not x.empty else 'UNCATEGORIZED').to_dict()
        df['state'] = df['pincode_str'].map(pil_state)
        df['district'] = df['pincode_str'].map(pil_dist)
    except Exception:
        pass

    # add dashboard-friendly metrics
    try:
        df['integrity_risk_pct'] = (df['integrity_score'] * 10).clip(0, 100).round(2)
    except Exception:
        df['integrity_risk_pct'] = 0.0
    label_map = {
        'age_18_greater': 'Adult Entry Spikes',
        'service_delivery_rate': 'Child Biometric Lags',
        'demo_age_17_': 'Activity Bursts',
        'security_anomaly_score': 'Suspicious Creation'
    }
    df['risk_diagnosis'] = df.get('primary_risk_driver', pd.Series(dtype=str)).map(label_map).fillna("Systemic Risk")
    df['date'] = pd.to_datetime(df.get('date', pd.Series(dtype='datetime64[ns]')), errors='coerce')

    status["msg"] = status.get("msg", "Loaded data successfully")
    return df, status

# --- UPDATE: caching small-view computation (already present in your code but kept & used)
@st.cache_data(show_spinner=False)
def compute_view_df(df, sel_state, start_date, end_date, active_drivers):
    """
    Cached heavy filtering / light aggregation for UI views.
    """
    if df is None:
        return None
    d = df.copy()

    # apply state filter
    if sel_state and sel_state != "INDIA":
        d = d[d['state'] == sel_state]

    # active drivers: the data stores 'primary_risk_driver' keys; filter if provided
    if active_drivers:
        d = d[d['primary_risk_driver'].isin(active_drivers)]

    # date range
    if start_date is not None and end_date is not None:
        try:
            d = d[(d['date'] >= start_date) & (d['date'] <= end_date)]
        except Exception:
            pass

    # ensure KPI columns exist cheaply
    try:
        d['integrity_risk_pct'] = (d['integrity_score'] * 10).clip(0,100).round(2)
    except Exception:
        d['integrity_risk_pct'] = d.get('integrity_risk_pct', 0.0)

    if 'service_delivery_rate' not in d.columns:
        try:
            d['service_delivery_rate'] = ((d.get('bio_age_5_17', 0) + d.get('demo_age_5_17', 0)) / (d.get('age_5_17', 0) + 1)) * 100
        except Exception:
            d['service_delivery_rate'] = 0.0

    return d

# --- UPDATE: additional cached aggregate builders to avoid recomputing heavy groupbys on reruns
@st.cache_data(show_spinner=False)
def build_tree_agg(view_df):
    if view_df is None or view_df.empty:
        return pd.DataFrame()
    tree_view_df = view_df[view_df['state'] != 'OTHER/UNCATEGORIZED']
    if tree_view_df.empty:
        return pd.DataFrame()
    tree_agg = tree_view_df.groupby(['state', 'district']).agg({
        'integrity_risk_pct': 'mean',
        'risk_diagnosis': lambda x: x.mode()[0] if not x.empty else "Stable",
        'pincode': 'count'
    }).reset_index()
    tree_agg = tree_agg.rename(columns={'integrity_risk_pct': 'risk', 'risk_diagnosis': 'driver', 'pincode': 'volume'})
    return tree_agg

@st.cache_data(show_spinner=False)
def build_pulse_df(view_df):
    if view_df is None or view_df.empty:
        return pd.DataFrame(columns=['Month','Risk','Compliance'])
    try:
        pulse_raw = view_df.groupby(view_df['date'].dt.to_period('M')).agg({
            'integrity_risk_pct': 'mean',
            'service_delivery_rate': 'mean'
        }).reset_index()
        pulse_raw['Risk'] = pulse_raw['integrity_risk_pct'].clip(0,100).round(1)
        pulse_raw['Compliance'] = pulse_raw['service_delivery_rate'].clip(0,100).round(1)
        pulse_raw['Month'] = pulse_raw['date'].astype(str)
        return pulse_raw[['Month','Risk','Compliance','date']]
    except Exception:
        return pd.DataFrame(columns=['Month','Risk','Compliance'])

@st.cache_data(show_spinner=False)
def build_heat_df(view_df):
    if view_df is None or view_df.empty:
        return pd.DataFrame()
    heat_df = view_df.groupby('district').agg({
        'age_18_greater': 'mean',
        'service_delivery_rate': 'mean',
        'demo_age_17_': 'mean',
        'security_anomaly_score': 'mean'
    }).tail(20)
    if heat_df.empty:
        return pd.DataFrame()
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())
    return heat_norm

@st.cache_data(show_spinner=False)
def build_friction_df(view_df):
    if view_df is None or view_df.empty:
        return pd.DataFrame()
    friction_df = view_df.groupby(['state', 'district']).agg({
        'integrity_risk_pct': 'mean',
        'demo_age_17_': 'mean',
        'age_18_greater': 'mean'
    }).reset_index()
    friction_df['display_name'] = friction_df['state'] + " - " + friction_df['district']
    friction_df['Forensic_Pressure'] = friction_df['integrity_risk_pct'].clip(0, 100).round(1)
    friction_df['Workload_Pressure'] = ((friction_df['demo_age_17_'] / (friction_df['demo_age_17_'] + friction_df['age_18_greater'] + 1)) * 100).clip(0, 100).round(1)
    friction_df['Total_Friction'] = friction_df['Forensic_Pressure'] + friction_df['Workload_Pressure']
    friction_top = friction_df.sort_values('Total_Friction', ascending=False).head(15)
    return friction_top

@st.cache_data(show_spinner=False)
def build_audit_table(view_df, top_n=45):
    if view_df is None or view_df.empty:
        return pd.DataFrame()
    audit_table = view_df.sort_values('integrity_risk_pct', ascending=False).head(top_n).copy()
    action_plan = {
        'Adult Entry Spikes': 'Enrolment Form Audit',
        'Child Biometric Lags': 'Deploy Mobile Van',
        'Activity Bursts': 'Operator ID Freeze',
        'Suspicious Creation': 'Manual ID Verification'
    }
    audit_table['Recommended Action'] = audit_table['risk_diagnosis'].map(action_plan)
    return audit_table

@st.cache_data(show_spinner=False)
def build_district_agg(df, district):
    if df is None or df.empty or district is None:
        return pd.DataFrame()
    district_all = df[df['district'] == district].copy()
    if district_all.empty:
        return pd.DataFrame()
    district_agg = district_all.groupby('pincode_str').agg({
        'state': 'first',
        'district': 'first',
        'integrity_risk_pct': 'mean',
        'risk_diagnosis': lambda x: x.mode()[0] if not x.empty else "N/A"
    }).reset_index()
    return district_agg

# Load data safely into session_state so subsequent reruns reuse it
if "df" not in st.session_state:
    try:
        df_loaded, load_status = load_data_safe(audit_path, master_path=master_path)
        st.session_state.df = df_loaded
        st.session_state.load_status = load_status
    except Exception as e:
        st.error(f"Starting problem while loading data: {e}")
        st.exception(e)
        st.stop()

df = st.session_state.get('df', None)
load_status = st.session_state.get('load_status', {"msg": ""})

if df is None or df.empty:
    st.warning("Data could not load or dataset is empty. Check file paths and precompute step.")
    if load_status.get("msg"):
        st.info(load_status.get("msg"))
    st.stop()

# --- Sidebar widgets: collect selections but avoid heavy operations here ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=120)
    st.markdown("---")

    # State select
    state_list = sorted([s for s in df['state'].unique() if s != 'OTHER/UNCATEGORIZED'])
    sel_state = st.selectbox("Select State", ["INDIA"] + state_list, key="sel_state")

    st.markdown("---")
    st.markdown("### Risk Profiles")
    f1 = st.checkbox("Adult Entry Spikes", value=True, key="f1")
    f2 = st.checkbox("Child Biometric Lags", value=True, key="f2")
    f3 = st.checkbox("Unusual Activity Bursts", value=True, key="f3")
    f4 = st.checkbox("Suspicious Profile Creation", value=True, key="f4")

    risk_map = {'age_18_greater': f1, 'service_delivery_rate': f2, 'demo_age_17_': f3, 'security_anomaly_score': f4}
    active_drivers = [k for k, v in risk_map.items() if v]

    st.markdown("---")
    # --- UPDATE: Debounced PIN input using a form so typing does NOT trigger multiple reruns
    with st.form("pin_search_form"):
        pin_input = st.text_input("Pincode Enquery: ", placeholder="Enter 6-digit PIN", key="pin_input")
        pin_submitted = st.form_submit_button("Search PIN")
        if pin_submitted:
            st.session_state['pincode_query'] = pin_input.strip()

    # maintain previous PIN value if present
    if 'pincode_query' not in st.session_state:
        st.session_state['pincode_query'] = ""

    # --- Date selection
    st.markdown("---")
    st.markdown("### Select Month")
    all_periods = df['date'].dt.to_period('M').dropna().unique()
    all_months = sorted(all_periods) if len(all_periods) else []
    if not all_months:
        this_month = pd.Timestamp.now().to_period('M')
        all_months = [this_month]
    month_labels = [m.strftime('%B %Y') for m in all_months]
    col_from, col_to = st.columns(2)
    with col_from:
        start_label = st.selectbox("From", options=month_labels, index=0, key="start_label")
    with col_to:
        end_label = st.selectbox("To", options=month_labels, index=len(month_labels)-1, key="end_label")

    # PDF export button
    st.markdown("---")
    st.markdown("### Export Final Report")
    if st.button("Download Report"):
        try:
            with st.spinner("Compiling National & Tactical Evidence..."):
                pdf_bytes = generate_forensic_dossier(
                    df=df, 
                    state_name=sel_state, 
                    root_path=root_path, 
                    search_pin=st.session_state.get('pincode_query',''),
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
            st.error("System Error while generating PDF. See logs.")
            st.exception(e)

# Convert selected labels back to actual dates for filtering
try:
    start_period = all_months[month_labels.index(start_label)]
    end_period = all_months[month_labels.index(end_label)]
except Exception:
    start_period = all_months[0]
    end_period = all_months[-1]

if start_period > end_period:
    st.error("Error: 'From' date must be before 'To' date.")
    st.stop()
else:
    start_date = start_period.start_time
    end_date = end_period.end_time

# Compute view_df with caching - this does the heavy filtering/ KPI augmentation
view_df = compute_view_df(df, sel_state, start_date, end_date, active_drivers)
if view_df is None or view_df.empty:
    st.warning("No data available for the selected filters. Try selecting INDIA or expanding the date range.")
    # continue so UI shows structure but charts will be empty

# If user entered a pincode, locate it in the full df to keep pivot logic identical to original app
target_obj = None
if st.session_state.get('pincode_query'):
    search_str = str(st.session_state['pincode_query']).strip()
    match = df[df['pincode_str'] == search_str]
    if not match.empty:
        target_obj = match.iloc[0]
        # set sel_state to the pin's state and recompute view_df to match original UX
        sel_state = target_obj['state']
        view_df = compute_view_df(df, sel_state, start_date, end_date, active_drivers)
    else:
        st.sidebar.error("PIN not found in database")

# Notify if we loaded a sampled dataset
if st.session_state.get('load_status', {}).get('sampled'):
    st.warning(st.session_state['load_status'].get('msg', "Large dataset loaded as sample to keep app responsive."))

# --- Main page content (unchanged logic) ---
st.markdown('<p class="main-title">Aadhaar National Integrity Dashboard</p>', unsafe_allow_html=True)

# avoid divide by zero / nan in KPIs
try:
    total_unique_pins = df['pincode'].nunique()
except Exception:
    total_unique_pins = 0

# --- 6-KPI COMMAND ROW ---
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Audit Scope", sel_state if sel_state != "NATIONAL OVERVIEW" else "INDIA")
with k2:
    st.metric("Unique Pincodes", f"{view_df['pincode'].nunique():,}")
with k3:
    st.metric("High Risk Sites", len(view_df[view_df['integrity_risk_pct'] > 75]))
with k4:
    integrity_val = 100 - (view_df['integrity_risk_pct'].mean() if not view_df['integrity_risk_pct'].isna().all() else 0)
    st.metric("Integrity", f"{integrity_val:.1f}%")
with k5:
    child_upd_val = view_df['service_delivery_rate'].mean() if 'service_delivery_rate' in view_df.columns else 0.0
    st.metric("Child Biometric Updates", f"{child_upd_val:.1f}%")
with k6:
    st.metric("Records Analyzed", f"{len(view_df):,}") 

st.markdown("---")

t1, t2, t3, t4, t5 = st.tabs(["Executive Overview", "Behavioral DNA", "Strategic Action", "Risk Drives","Pincode Drilldown"])

# --- Tab 1 (Executive Overview) ---
with t1:
    st.markdown('<div class="section-header">National Service Demand vs. Forensic Risk Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        # --- LIVE DEMOGRAPHIC LIFECYCLE CHART ---
        infant_gen = view_df['age_0_5'].sum() if 'age_0_5' in view_df.columns else 0
        infant_upd = 0 
        child_gen = view_df['age_5_17'].sum() if 'age_5_17' in view_df.columns else 0
        child_upd = (view_df['bio_age_5_17'].sum() if 'bio_age_5_17' in view_df.columns else 0) + (view_df['demo_age_5_17'].sum() if 'demo_age_5_17' in view_df.columns else 0)
        adult_gen = view_df['age_18_greater'].sum() if 'age_18_greater' in view_df.columns else 0
        adult_upd = (view_df['bio_age_17_'].sum() if 'bio_age_17_' in view_df.columns else 0) + (view_df['demo_age_17_'].sum() if 'demo_age_17_' in view_df.columns else 0)

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

        fig_life.update_layout(height=500, margin=dict(t=50, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), yaxis_title="Transaction Volume", xaxis_title="")
        st.plotly_chart(fig_life, width='stretch')
    with col2:
        pie_data = view_df['risk_diagnosis'].value_counts().reset_index() if not view_df.empty else pd.DataFrame(columns=['risk_diagnosis','count'])
        fig_pie = px.pie(pie_data, values='count', names='risk_diagnosis', hole=0.4, height=550, color_discrete_sequence=px.colors.qualitative.Pastel, title=f"Risk Profile Composition: {sel_state}")
        st.plotly_chart(fig_pie, width='stretch')

    # --- THE TREEMAP ---
    st.markdown('<div class="section-header">Regional Integrity Hierarchy </div>', unsafe_allow_html=True)
    tree_agg = build_tree_agg(view_df)
    if not tree_agg.empty:
        color_max = tree_agg['risk'].quantile(0.95)
        if color_max < 15: color_max = 15
        fig_tree = px.treemap(tree_agg, path=[px.Constant("INDIA"), 'state', 'district'], values='volume', color='risk', color_continuous_scale='RdYlGn_r', range_color=[0, color_max], custom_data=['state', 'district', 'risk', 'driver'], height=750)
        fig_tree.update_traces(textinfo="label+value", texttemplate="<b>%{label}</b>", hovertemplate="<b>State:</b> %{customdata[0]}<br><b>District:</b> %{customdata[1]}<br><b>Risk Intensity:</b> %{customdata[2]:.2f}%<br><b>Primary Threat:</b> %{customdata[3]}<extra></extra>", insidetextfont_size=14, textposition="middle center")
        fig_tree.update_layout(margin=dict(t=30, l=10, r=10, b=10), coloraxis_colorbar=dict(title="Relative Risk %", ticksuffix="%"))
        st.plotly_chart(fig_tree,width='stretch')
    else:
        st.info("No treemap data available for this selection.")

    # --- TAB 1: LIVE TREND ANALYSIS ---
    st.markdown('<div class="section-header">Administrative Pulse: Risk & Compliance Trends</div>', unsafe_allow_html=True)
    pulse_df = build_pulse_df(view_df)
    fig_pulse = go.Figure()
    fig_pulse.add_trace(go.Bar(x=pulse_df['Month'] if not pulse_df.empty else [], y=pulse_df['Compliance'] if not pulse_df.empty else [], name='MBU Compliance % (Efficiency)', marker_color='#27AE60', opacity=0.7, text=pulse_df['Compliance'].apply(lambda x: f"{x}%") if not pulse_df.empty else [], textposition='inside'))
    fig_pulse.add_trace(go.Scatter(x=pulse_df['Month'] if not pulse_df.empty else [], y=pulse_df['Risk'] if not pulse_df.empty else [], name='Risk Intensity % (Security Threat)', mode='lines+markers', line=dict(color='#E74C3C', width=4), marker=dict(size=10, symbol='diamond')))
    fig_pulse.update_layout(title=f"<b>Forensic Pulse: {sel_state} Operational Health</b>", xaxis_title="Audit Month", yaxis_title="Percentage (%)", yaxis_range=[0, 115], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template="plotly_white", height=500, hovermode="x unified")
    st.plotly_chart(fig_pulse, width='stretch')

# --- Tab 2: Anomaly Clustering ---
with t2:
    st.markdown('<div class="section-header">Automated Risk Profiling: Characterizing Systemic Anomalie</div>', unsafe_allow_html=True)
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
    heat_norm = build_heat_df(view_df)
    if not heat_norm.empty:
        fig7 = px.imshow(heat_norm, labels=dict(x="Forensic Driver", y="District", color="Relative Intensity"), x=['Adult Spikes', 'Child Compliance', 'Activity Bursts', 'Fraud Index'], y=heat_norm.index, color_continuous_scale='YlOrRd', aspect="auto", title=f"<b>Chart 07: Normalized DNA Scorecard of (Fingerprinting Fraud Types) {sel_state}</b>")
        fig7.update_traces(hovertemplate="District: %{y}<br>Driver: %{x}<br>Relative Intensity: %{z:.2f}")
        st.plotly_chart(fig7, width='stretch')
    else:
        st.info("No DNA heatmap data to display for this scope.")

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        img5_path = os.path.join(root_path, "output", "ML_Anomaly_charts", "05_ml_threat_radar.png")
        if os.path.exists(img5_path):
            st.image(img5_path, width='stretch', caption="Chart 05: Behavioral Signature Radar")
    with row3_col2:
        img6_path = os.path.join(root_path, "output", "ML_Anomaly_charts", "06_ml_forensic_scorecard.png")
        if os.path.exists(img6_path):
            st.image(img6_path, width='stretch', caption="Chart 06: Forensic Magnitude Scorecard")

# --- Tab 3: Strategic Action ---
with t3:
    st.markdown('<div class="section-header">Root-Cause Modelling & Strategic Resource Allocation</div>', unsafe_allow_html=True)
    img11_path = os.path.join(root_path, "output", "Deep_Analysis", "11_strategic_portfolio.png")
    if os.path.exists(img11_path):
        st.image(img11_path, width='stretch', caption="Chart 11: National Policy Zones")

    st.markdown('<div class="section-header">Regional Risk Driver Impact (Live Analysis)</div>', unsafe_allow_html=True)
    driver_impact = view_df['risk_diagnosis'].value_counts().reset_index() if not view_df.empty else pd.DataFrame(columns=['risk_diagnosis','count'])
    fig8 = px.bar(driver_impact, x='risk_diagnosis', y='count', color='risk_diagnosis', title=f"<b>Chart 08: Volume of Primary Threat Drivers in {sel_state} Active Scope</b>", labels={'risk_diagnosis': 'ML Diagnosis', 'count': 'Number of Impacted Records'}, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig8, width='stretch')

    st.markdown('<div class="section-header">Chart 10: High-Priority Forensic Audit List</div>', unsafe_allow_html=True)
    st.write("The following sites have been flagged by the Isolation Forest model for manual document verification.")
    audit_table = build_audit_table(view_df)
    if not audit_table.empty:
        st.dataframe(audit_table[['district', 'pincode', 'integrity_risk_pct', 'risk_diagnosis', 'Recommended Action']], width='stretch', hide_index=True)
        st.download_button(label="Download Regional Action Plan", data=audit_table.to_csv(index=False), file_name=f"Audit_Plan_{sel_state}.csv", mime='text/csv')
    else:
        st.info("No audit candidates for the current selection.")

# --- Tab 4: Operational Friction ---
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
    st.info("Forensic Narrative: This chart identifies Burnout Zones. When Maintenance Workload and Forensic Risk are both high, the system is at critical friction.")
    friction_top = build_friction_df(view_df)
    if not friction_top.empty:
        fig_friction = go.Figure()
        fig_friction.add_trace(go.Bar(x=friction_top['display_name'], y=friction_top['Workload_Pressure'], name='Maintenance Workload (Operator Stress)', marker_color="#197ADB", text=friction_top['Workload_Pressure'].apply(lambda x: f"{x}%"), textposition='outside', textangle=-90))
        fig_friction.add_trace(go.Bar(x=friction_top['display_name'], y=friction_top['Forensic_Pressure'], name='Forensic Risk (Security Threat)', marker_color='#E74C3C', text=friction_top['Forensic_Pressure'].apply(lambda x: f"{x}%"), textposition='outside', textangle=-90))
        fig_friction.update_layout(barmode='group', title="Operational Friction: Contextual District Analysis", xaxis_title="Region (State - District)", yaxis_title="Pressure Index (%)", yaxis_range=[0, 130], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template="plotly_white", height=650, xaxis=dict(tickangle=45, tickfont=dict(size=11)))
        st.plotly_chart(fig_friction,width='stretch')
    else:
        st.info("No friction data available for this scope.")
    st.success("Administrative Directive: State-Level Hotspots indicate systemic state-wide failures in balancing workload with security.")

# --- Tab 5: Forensic Pivot & Drilldown ---
with t5:
    if st.session_state.get('pincode_query') and target_obj is not None:
        search_pin = st.session_state['pincode_query']
        st.markdown(f'<div class="section-header">Forensic Investigation: PIN {search_pin}</div>', unsafe_allow_html=True)
        district_all = df[df['district'] == target_obj['district']].copy()
        if district_all.empty:
            st.info("No district-level data available for this PIN.")
        else:
            safe_haven = district_all.sort_values('integrity_risk_pct', ascending=True).iloc[0]
            if target_obj['integrity_risk_pct'] > 60:
                st.warning(f"BREACH DETECTED: Suspend Adult Enrolment at {search_pin}. Reroute to PIN {safe_haven['pincode_str']}.")
            district_agg = build_district_agg(df, target_obj['district'])
            if district_agg.empty:
                st.info("No pincode aggregation available for this district.")
            else:
                action_map = {'Adult Entry Spikes': ' Forensic Audit: Verify 18+ Form Authenticity','Child Biometric Lags': 'Outreach: Deploy Mobile Update Van','Activity Bursts': 'Technical: Inspect Operator Software Logs','Suspicious Creation': 'Security: Manual Identity Cross-Verification'}
                district_agg['Required Action'] = district_agg['risk_diagnosis'].map(action_map).fillna("Monitor Activity")
                peers = district_agg.sort_values('integrity_risk_pct', ascending=False).reset_index(drop=True)
                match_indices = peers.index[peers['pincode_str'] == search_pin].tolist()
                if not match_indices:
                    st.error("PIN found in master but missing in district aggregation.")
                else:
                    t_idx = match_indices[0]
                    start, end = max(0, t_idx - 7), min(len(peers), t_idx + 8)
                    cluster = peers.iloc[start:end].copy()
                    cluster['color_logic'] = cluster['pincode_str'].apply(lambda x: '#ef4444' if x == search_pin else "#1278DF")
                    fig_grad = px.bar(cluster.sort_values('integrity_risk_pct', ascending=True), x='integrity_risk_pct', y='pincode_str', orientation='h', text_auto='.1f', title=f"Risk Hierarchy: {target_obj['district']} Cluster (Period Average)", labels={'integrity_risk_pct': 'Average Risk %', 'pincode_str': 'PIN'})
                    fig_grad.update_traces(marker_color=cluster.sort_values('integrity_risk_pct', ascending=True)['color_logic'])
                    fig_grad.update_layout(xaxis_range=[0, 100], yaxis_type='category', height=500, template="plotly_white")
                    st.plotly_chart(fig_grad, width='stretch')
                    st.markdown("**Field Investigative Evidence**")
                    display_table = cluster[['state', 'district', 'pincode_str', 'integrity_risk_pct', 'risk_diagnosis', 'Required Action']].rename(columns={'state': 'State', 'district': 'District', 'pincode_str': 'Pincode', 'integrity_risk_pct': 'Risk Score %', 'risk_diagnosis': 'Forensic Diagnosis','Required Action': 'Required Action'})
                    def highlight_target(row):
                        is_target = str(row['Pincode']).strip() == search_pin
                        return ['background-color: #fee2e2; font-weight: bold' if is_target else '' for _ in row]
                    st.table(display_table.style.apply(highlight_target, axis=1))
                    st.download_button(label="Download Field Work-Order", data=cluster.to_csv(index=False), file_name=f"Forensic_Audit_{search_pin}.csv", mime='text/csv')
    else:
        st.info("National Aadhar Portal Ready. Please enter a Pincode in the sidebar.")
