import pandas as pd
import numpy as np
import plotly.express as px
import os
def standardize_states(df):
    state_map = {
        # --- TYPOS & OLD NAMES ---
        'ANDAMAN & NICOBAR ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
        'CHHATISGARH': 'CHHATTISGARH',
        'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
        'ORISSA': 'ODISHA',
        'PONDICHERRY': 'PUDUCHERRY',
        'TAMILNADU': 'TAMIL NADU',
        'UTTARANCHAL': 'UTTARAKHAND',
        'WESTBENGAL': 'WEST BENGAL',
        'WEST BENGL': 'WEST BENGAL',
        'WEST BANGAL': 'WEST BENGAL',
        'WEST BENGLI': 'WEST BENGAL',
        'WEST  BENGAL': 'WEST BENGAL',
        
        # --- MERGED UT ---
        'DADRA & NAGAR HAVELI': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU',
        'DADRA AND NAGAR HAVELI': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU',
        'DAMAN & DIU': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU',
        'DAMAN AND DIU': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU',
        'THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU',

        # --- CITY/LOCALITY (Mapping them to their actual states) ---
        'DELHI STATE': 'DELHI',
        'NCT OF DELHI': 'DELHI',
        'GREATER KAILASH 2': 'DELHI',
        'GURGAON': 'HARYANA',
        'JAIPUR': 'RAJASTHAN',
        'PUNE CITY': 'MAHARASHTRA',
        'NAGPUR': 'MAHARASHTRA',
        'BALANAGAR': 'TELANGANA',
        'MADANAPALLE': 'ANDHRA PRADESH',
        'PUTTENAHALLI': 'KARNATAKA',
        'PUTHUR': 'ANDHRA PRADESH',
        'RAJA ANNAMALAI PURAM': 'TAMIL NADU',
        'DARBHANGA': 'BIHAR'
    }

    if 'state' in df.columns:
        df['state'] = df['state'].str.upper().str.strip()
        df['state'] = df['state'].replace(state_map)
    return df


def run_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, "output", "cleaned_master_data.csv")
    chart_dir = os.path.join(project_root, "output", "charts")
    if not os.path.exists(chart_dir): os.makedirs(chart_dir)
    
    # --- 1. DATA LOADING ---
    df = pd.read_csv(input_path, low_memory=False)
    df = standardize_states(df)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # --- 2. METRIC CALCULATION ---
    df['security_anomaly_score'] = df['age_18_greater'] / (df['demo_age_17_'] + 1)
    
    # Service Delivery Rate: Ratio of Child Updates vs Child Enrolments
    df['service_delivery_rate'] = (df['bio_age_5_17'] / (df['age_5_17'] + 1)) * 100

    df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']

    df['total_updates'] = df['bio_age_5_17'] + df['demo_age_5_17'] + df['bio_age_17_'] + df['demo_age_17_']


    # --- 3. ANOMALY MAXIMIZER LOGIC ---
    print("Surfacing high-volume forensic leads...")
    
    # Calculate state-level sensitivity (local normal)
    state_means = df.groupby('state')['security_anomaly_score'].transform('mean')
    state_stds = df.groupby('state')['security_anomaly_score'].transform('std')
    
    # SURFACING CRITERIA: 
    # 1. Z-Score > 1.5 (Significant local deviation)
    # 2. OR top 10% of scores in the entire country
    z_score = (df['security_anomaly_score'] - state_means) / (state_stds + 0.1)
    top_10_threshold = df['security_anomaly_score'].quantile(0.90)
    
    df['is_anomaly'] = (z_score > 1.5) | (df['security_anomaly_score'] > top_10_threshold)
    anomalies = df[df['is_anomaly'] == True].copy()

    # --- UPGRADED CHART 1: THREE-TIER DEMOGRAPHIC SPLIT ---
    print("Generating Chart 1: 3-Tier Demographic Demand...")

    # 1. Age 0-5: (Only has Enrolment, Maintenance is logically 0 in this schema)
    age_0_5_gen = df['age_0_5'].sum()
    age_0_5_upd = 0 

    # 2. Age 5-17: (School-going / MBU Group)
    age_5_17_gen = df['age_5_17'].sum()
    age_5_17_upd = df['bio_age_5_17'].sum() + df['demo_age_5_17'].sum()

    # 3. Age 18+: (Adult / Saturated Group)
    age_18_gen = df['age_18_greater'].sum()
    age_18_upd = df['bio_age_17_'].sum() + df['demo_age_17_'].sum()

    # 4. Prepare the Dataframe
    seg_data = pd.DataFrame({
        'Age_Group': ['Infants (0-5)', 'Infants (0-5)', 'Children (5-17)', 'Children (5-17)', 'Adults (18+)', 'Adults (18+)'],
        'Activity': ['New Enrolment', 'Maintenance/Updates', 'New Enrolment', 'Maintenance/Updates', 'New Enrolment', 'Maintenance/Updates'],
        'Volume': [age_0_5_gen, age_0_5_upd, age_5_17_gen, age_5_17_upd, age_18_gen, age_18_upd]
    })

    # 5. Create the Grouped Bar Chart
    fig1 = px.bar(seg_data, 
                  x='Age_Group', 
                  y='Volume', 
                  color='Activity', 
                  barmode='group',
                  text_auto='.3s', 
                  title="<b>National Service Demand: Lifecycle Segmentation </b>",
                  color_discrete_map={'New Enrolment': '#3498DB', 'Maintenance/Updates': '#2ECC71'},
                  template="plotly_white")

    # 6. Adjust layout for readability
    fig1.update_layout(
        yaxis_title="Transaction Volume",
        xaxis_title="Demographic Life-Stage",
        legend_title_text="Activity Type",
        font=dict(size=14)
    )
    
    fig1.write_image(os.path.join(chart_dir, "01_service_demand_split.png"), width=1000, height=600)

    # --- CHART 2: DISTRICT PURIFIER & LEADERBOARD ---
    print("Generating Chart 2: Priority Audit List...")
    banned = ['NEAR', 'ROAD', 'THANA', 'CROSS', 'STREET', 'COLONY', 'UNIVERSITY', 'POST', 'OFFICE', '5TH']
    pure_df = anomalies[~anomalies['district'].str.contains('|'.join(banned), na=False, case=False)]
    
    risk_list = pure_df.groupby(['state', 'district'])['security_anomaly_score'].mean().sort_values(ascending=False).head(25).reset_index()
    risk_list['scaled_score'] = (risk_list['security_anomaly_score'] / risk_list['security_anomaly_score'].max()) * 10

    fig2 = px.bar(risk_list, x='scaled_score', y='district', color='state', orientation='h',
                  text_auto='.2f', title="<b>Regional Security Audit: Top 25 Districts by Anomaly Score (0-10)</b>",
                  labels={'scaled_score': 'Security Anomaly Score', 'district': 'District'})
    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
    fig2.write_image(os.path.join(chart_dir, "02_risk_leaderboard.png"), width=1000, height=800)

# --- CHART 3: THE TREEMAP (GEOGRAPHICALLY HEALED & SCALED) ---
    print("Generating Chart 3: Proportional Map...")

    # 1. Standardize text strictly
    df['district'] = df['district'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)
    df['state'] = df['state'].str.upper().str.strip()
    df['pincode'] = df['pincode'].astype(str).str.zfill(6)

    # 2. THE NATIONAL FIX: Pincode-to-District Mapping 
    pincode_map = df.groupby('pincode')['district'].agg(lambda x: x.mode()[0] if not x.empty else "UNKNOWN").to_dict()
    df['healed_district'] = df['pincode'].map(pincode_map)

    # 3. AGGREGATION: We use MEAN for Risk and NUNIQUE for Districts
    tree_raw = df.groupby('state').agg({
        'security_anomaly_score': 'mean', 
        'total_updates': 'sum', 
        'total_enrolments': 'sum',
        'healed_district': 'nunique'     
    }).reset_index()

    # 4. SCALE THE RISK: Normalize to 0-10 scale
    max_score = tree_raw['security_anomaly_score'].max()
    tree_raw['scaled_risk'] = (tree_raw['security_anomaly_score'] / max_score * 10).round(2)
    
    # 5. Calculate Compliance / Velocity
    tree_raw['velocity'] = (tree_raw['total_updates'] / (tree_raw['total_enrolments'] + 1)).round(1)
    tree_raw['color_metric'] = tree_raw['velocity'].clip(upper=100)

    # 6. Create Professional Labels
    tree_raw['label'] = (
        "<b>" + tree_raw['state'] + "</b><br>" +
        "Risk Intensity: " + tree_raw['scaled_risk'].astype(str) + "/10<br>" +
        "Compliance: " + tree_raw['velocity'].astype(str) + "%<br>" +
        "Districts: " + tree_raw['healed_district'].astype(str)
    )

    # 7. Generate the Treemap
    fig3 = px.treemap(
        tree_raw, 
        path=[px.Constant("India"), 'state'], 
        values='scaled_risk',         
        color='color_metric',          
        color_continuous_scale='RdYlGn', 
        range_color=[0, 30],
        title="<b>National Integrity Map: Size=Risk Intensity (0-10), Color=Compliance</b>",
        labels={'color_metric': 'Compliance %'}
    )
    
    fig3.update_traces(
        texttemplate="%{customdata[0]}", 
        customdata=tree_raw[['label']].values,
        textposition='middle center',
        textfont_size=14
    )

    fig3.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    fig3.write_image(os.path.join(chart_dir, "03_risk_treemap.png"), width=1200, height=800)

   # --- CHART 4: THE CHRONOLOGICAL SERVICE RIBBON ---
    print("Generating Chart 4: Chronological Service Ribbon...")

    # 1. Normalize dates to the 1st of the month for perfect grouping
    df['temp_date'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # 2. Aggregate National Totals by the actual date (to keep them in order)
    pulse_raw = df.groupby('temp_date').agg({
        'total_updates': 'sum', 
        'total_enrolments': 'sum'
    }).sort_index().reset_index()

    # 3. Create the "Month-YY" label for the timeline (e.g., March-25)
    pulse_raw['month_year'] = pulse_raw['temp_date'].dt.strftime('%B-%y')

    # 4. Calculate Velocity with Safety Cap (100x)
    pulse_raw['velocity_raw'] = pulse_raw['total_updates'] / (pulse_raw['total_enrolments'] + 1)
    pulse_raw['velocity_capped'] = pulse_raw['velocity_raw'].clip(upper=100).round(1)

    # 5. Create the Ribbon Visual 
    fig4 = px.bar(
        pulse_raw, 
        x='month_year', 
        y=[1] * len(pulse_raw), 
        color='velocity_capped',
        text=pulse_raw['velocity_capped'].apply(lambda x: f"{x}x" if x < 100 else "100x+"),
        color_continuous_scale='RdYlGn_r',
        range_color=[0, 100],
        title="<b>Administrative Pulse: Monthly Maintenance-to-Enrolment Velocity</b>",
        labels={'velocity_capped': 'Velocity', 'month_year': 'Timeline'}
    )

    # 6. Professional Formatting
    fig4.update_traces(
        textposition='inside', 
        textfont_size=14,
        marker_line_width=2,
        marker_line_color="white"
    )

    fig4.update_layout(
        xaxis_title="",
        yaxis_showticklabels=False,
        yaxis_title="",
        xaxis_tickangle=-45,
        height=400,
        template="plotly_white",
        coloraxis_colorbar=dict(title="Velocity Index")
    )

    # 7. Final Save
    fig4.write_image(os.path.join(chart_dir, "04_seasonal_performance_ribbon.png"), width=1200, height=400)

    # --- ANOMALY EXPORT ---
    # We use a state-aware Z-score to find the leads
    state_mean = df.groupby('state')['security_anomaly_score'].transform('mean')
    state_std = df.groupby('state')['security_anomaly_score'].transform('std')
    df['z'] = (df['security_anomaly_score'] - state_mean) / (state_std + 0.1)
    
    anomalies = df[df['z'] > 1.5]
    anomalies.to_csv(os.path.join(project_root, "output", "anomaly_report.csv"), index=False)
    
    print(f"SUCCESS. Anomalies Surfaced: {len(anomalies)}")

if __name__ == "__main__":
    run_analysis()