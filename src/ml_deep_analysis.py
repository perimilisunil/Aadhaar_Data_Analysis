import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os


def run_forensic_intelligence():
    # 1. SETUP PATHS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, "output", "cleaned_master_data.csv")
    output_path = os.path.join(project_root, "output", "final_audit_report.csv")
    ml_chart_dir = os.path.join(project_root, "output", "Deep_Analysis")
    if not os.path.exists(ml_chart_dir): os.makedirs(ml_chart_dir)

    # 2. LOAD DATA
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run cleaner.py first!")
        return
    
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df)} records for intelligence processing...")

    # 3. CALCULATE METRICS 
    print(" Calculating Forensic Metrics.")
    df['service_delivery_rate'] = np.where(df['age_5_17'] > 0, (df['bio_age_5_17'] / df['age_5_17']) * 100, 0)
    df['security_anomaly_score'] = np.where(df['demo_age_17_'] > 0, (df['age_18_greater'] / df['demo_age_17_']), 0)

    # 4. SELECT THE 4 CORE FEATURES 
    features = ['age_18_greater', 'demo_age_17_', 'service_delivery_rate', 'security_anomaly_score']
    
    # Handle Infinities/NaNs before ML
    X_data = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 5. SCALE DATA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    # 6. ML MODEL: ISOLATION FOREST
    print(" Training Isolation Forest model...")
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_signal'] = model.fit_predict(X_scaled) 
    
    # Calculate raw scores
    raw_scores = model.decision_function(X_scaled)
    
    # 7. CALCULATE 0-10 INTEGRITY SCORE
    df['integrity_score'] = np.interp(raw_scores, (raw_scores.min(), raw_scores.max()), (10, 0)).round(2)
    
    # CRIMES Based on Drivers
    crime_map = {
        'age_18_greater': 'Identity Spoofing Risk',
        'security_anomaly_score': 'Synthetic Identity Alert',
        'demo_age_17_': 'Update Velocity Anomaly',
        'service_delivery_rate': 'Child Compliance Gap'
    }

    # 8. DEFINE RISK PROFILES
    def categorize_risk(score):
        if score > 8: return "High-Risk Fraud Cluster"
        if score > 5: return "Service Delivery Gap"
        return "Operational Normal"
    
    df['risk_profile'] = df['integrity_score'].apply(categorize_risk)

    # 9. CALCULATE PRIMARY RISK DRIVER
    mean_vals = X_data.mean()
    std_vals = X_data.std()
    deviations = (X_data - mean_vals) / std_vals
    df['primary_risk_driver'] = deviations.abs().idxmax(axis=1)

    # 9. GENERATE GLOBAL DRIVER CHART
    print("Generating Global Driver Chart...")
    high_risk_df = df[df['integrity_score'] > 7]
    
    if not high_risk_df.empty:
        driver_counts = high_risk_df['primary_risk_driver'].value_counts().reset_index()
        driver_counts.columns = ['Risk_Driver', 'Impact_Count']
        
        fig = px.bar(driver_counts, x='Risk_Driver', y='Impact_Count',
                     title="<b>National Forensic Summary: Primary Drivers of Systemic Risk</b>",
                     color='Risk_Driver',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     template="plotly_white")
        
        fig.update_traces(
            texttemplate='%{y:.3s}', 
            textposition='outside',   
            cliponaxis=False          
        )
        
        fig.update_layout(
            yaxis_title="Pincode Impact Count",
            xaxis_title="Primary Risk Driver (ML Classified)",
            uniformtext_minsize=8,
            margin=dict(t=80) 
        )
        fig.write_image(os.path.join(ml_chart_dir, "08_global_feature_importance.png"))
    else:
        print("No high-risk anomalies found to plot.")

    # --- CHART 09: STATE ANOMALY CONCENTRATION ---
    print("Generating Chart 09:")
    high_risk_only = df[df['risk_profile'] == "High-Risk Fraud Cluster"]
    state_concentration = pd.DataFrame(columns=['state', 'anomaly_count'])    
    if not high_risk_only.empty:
        state_concentration = high_risk_only.groupby('state').size().reset_index(name='anomaly_count')
        state_concentration = state_concentration.sort_values('anomaly_count', ascending=False).head(20)

        fig9 = px.bar(
            state_concentration,
            x='anomaly_count',
            y='state',
            orientation='h',
            color='anomaly_count',
            text='anomaly_count', 
            color_continuous_scale='Reds',
            title="<b>Audit Volume: States with Highest Count of Suspicious Pincodes</b>",
            labels={'anomaly_count': 'Total Suspicious Pincodes', 'state': 'State'}
        )
        fig9.update_traces(textposition='outside')
        fig9.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        fig9.write_image(os.path.join(ml_chart_dir, "09_state_anomaly_concentration.png"))

    # --- CHART 10: AUDIT SUMMARY TABLE  ---
    print("Generating Chart 10:")    
    diverse_list = []
    for driver in features:
        subset = df[df['primary_risk_driver'] == driver].sort_values(by='integrity_score', ascending=False).head(10)
        diverse_list.append(subset)
    
    audit_df = pd.concat(diverse_list).drop_duplicates().sort_values(by='integrity_score', ascending=False)
    audit_df['Forensic_Diagnosis'] = audit_df['primary_risk_driver'].map(crime_map)

    fig10 = go.Figure(data=[go.Table(
        header=dict(values=['State', 'District', 'Pincode', 'Integrity Score', 'Forensic Diagnosis'],
                    fill_color='midnightblue', font=dict(color='white', size=14), align='left'),
        cells=dict(values=[audit_df.state, audit_df.district, audit_df.pincode, 
                           audit_df.integrity_score, audit_df.Forensic_Diagnosis], 
                   fill_color='lavender', align='left'))
    ]) 
    fig10.update_layout(title="Executive Audit: Top Verification Sites")
    fig10.write_image(os.path.join(ml_chart_dir, "10_final_audit_summary.png"), width=1200, height=1200)

    # CHART 11: THE STATE STRATEGIC PORTFOLIO 
    print("Generating Chart 11: ")

    avg_risk = df['integrity_score'].median()
    df['eng_calc'] = (df['demo_age_17_'] + df['bio_age_17_']) / (df['age_18_greater'] + 1)
    avg_eng = df['eng_calc'].median()

    def get_strategy_zone(row):
        engagement = (row['demo_age_17_'] + row['bio_age_17_']) / (row['age_18_greater'] + 1)
        if row['integrity_score'] > avg_risk and engagement > avg_eng:
            return "ZONE A: Forensic Audit"
        if row['integrity_score'] > avg_risk and engagement <= avg_eng:
            return "ZONE B: Ghost ID Alerts"
        if row['integrity_score'] <= avg_risk and engagement > avg_eng:
            return "ZONE C: Model Districts"
        return "ZONE D: Awareness Camps"

    df['strategic_zone'] = df.apply(get_strategy_zone, axis=1)

    portfolio_df = df.groupby(['state', 'strategic_zone']).size().reset_index(name='count')
    state_total = portfolio_df.groupby('state')['count'].transform('sum')
    portfolio_df['percentage'] = (portfolio_df['count'] / state_total) * 100

    fig11 = px.bar(
        portfolio_df, 
        x='state', 
        y='percentage', 
        color='strategic_zone',
        title="<b>National Strategy Portfolio: Administrative Requirements by State (%)</b>",
        labels={'percentage': 'Percentage of Districts (%)', 'state': 'State', 'strategic_zone': 'Strategic Zone'},
        color_discrete_map={
            "ZONE A: Forensic Audit": "#E74C3C",
            "ZONE B: Ghost ID Alerts": "#E67E22",
            "ZONE C: Model Districts": "#27AE60",
            "ZONE D: Awareness Camps": "#2980B9"
        },
        template="plotly_white"
    )

    fig11.update_layout(
        barmode='stack', 
        xaxis={'categoryorder':'total descending'}, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig11.write_image(os.path.join(ml_chart_dir, "11_strategic_portfolio.png"), width=1200, height=800)

    # 10. SAVE THE FINAL AUDIT REPORT
    extra_cols = ['age_0_5', 'age_5_17', 'bio_age_5_17', 'demo_age_5_17','bio_age_17_']
    final_cols = ['date', 'state', 'district', 'pincode'] + features + extra_cols + ['integrity_score', 'primary_risk_driver']
    
    df[final_cols].to_csv(output_path, index=False)

    print(f"Deep Analysis Complete! Charts saved to: {ml_chart_dir}")

if __name__ == "__main__":
    run_forensic_intelligence()