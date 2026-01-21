import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os

def run_ml_analysis():
    # --- PATH SETUP ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, "output", "anomaly_report.csv")
    ml_chart_dir = os.path.join(project_root, "output", "ML_Anomaly_charts")
    if not os.path.exists(ml_chart_dir): os.makedirs(ml_chart_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Run analysis.py first.")
        return
    df = pd.read_csv(input_path)
    
    features = ['age_18_greater', 'demo_age_17_', 'service_delivery_rate', 'security_anomaly_score']
    
    # Mapping to English names for the Jury (Same labels, new underlying data)
    display_names = {
        'age_18_greater': 'Adult Spikes',
        'demo_age_17_': 'System Activity',
        'service_delivery_rate': 'Child Compliance',
        'security_anomaly_score': 'Fraud Index'
    }
    
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. ML CLUSTERING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 3. DYNAMIC LABELING (Updated to use new column names)
    high_risk_id = df.groupby('cluster')['security_anomaly_score'].mean().idxmax()
    low_comp_id = df.groupby('cluster')['service_delivery_rate'].mean().idxmin()
    
    def label_clusters(row):
        if row == high_risk_id: return "High-Risk (Fraud)"
        if row == low_comp_id: return "Service Gap (Children)"
        return "Active/Migration Zones"

    df['risk_profile'] = df['cluster'].apply(label_clusters)

    # CHART 5: THE "THREAT PERIMETER" RADAR
    print("Generating Chart 5: Threat Radar...")
    cluster_summary = df.groupby('risk_profile')[features].mean().reset_index()
    
    # Normalize by Max to show "Deviation"
    for f in features:
        cluster_summary[f] = cluster_summary[f] / cluster_summary[f].max()

    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(
        r=[0.2, 0.2, 0.2, 0.2], 
        theta=list(display_names.values()), 
        fill='toself', 
        name='Safe Baseline', 
        line=dict(color='gray', dash='dash')
    ))

    for profile in cluster_summary['risk_profile'].unique():
        data = cluster_summary[cluster_summary['risk_profile'] == profile]
        fig4.add_trace(go.Scatterpolar(
            r=[data[f].values[0] for f in features],
            theta=list(display_names.values()),
            fill='toself', 
            name=profile
        ))

    fig4.update_layout(title="<b>Threat Signature: How Clusters deviate from Safety</b>", template="plotly_white")
    fig4.write_image(os.path.join(ml_chart_dir, "05_ml_threat_radar.png"))

    # CHART 6: THE FORENSIC SCORECARD
    print("Generating Chart 6: Forensic Scorecard...")
    
    # 1. Calculate the MEAN for each group
    scorecard_df = df.groupby('risk_profile')[features].mean().reset_index()
    scorecard_df.columns = ['Profile', 'Adult Spikes', 'System Activity', 'Child Compliance', 'Fraud Index']
    
    # 2. Melt the data for side-by-side bars
    melted_scorecard = scorecard_df.melt(id_vars='Profile', var_name='Metric', value_name='Average Value')

    # 3. Create a Grouped Bar Chart
    fig5 = px.bar(melted_scorecard, 
                  x='Metric', 
                  y='Average Value', 
                  color='Profile',
                  barmode='group',
                  title="<b>Forensic Scorecard: Comparing Normal vs. High-Risk Behavior</b>",
                  color_discrete_map={
                      "High-Risk (Fraud)": "#E74C3C", 
                      "Service Gap (Children)": "#F1C40F", 
                      "Active/Migration Zones": "#2E86C1"
                  },
                  template="plotly_white")
    
    # ADDING LABELS
    fig5.update_traces(
            texttemplate='%{y:.3s}', 
            textposition='outside',   
            cliponaxis=False          
        )
    
    fig5.update_layout(
                      yaxis_title="Average Value",
                      legend_title_text="District Profile")
    
    fig5.write_image(os.path.join(ml_chart_dir, "06_ml_forensic_scorecard.png"), width=1000, height=600)


    # CHART 7: THE "DNA SCORECARD" (Heatmap Table)
    print("Generating Chart 7: DNA Scorecard...")
    matrix_df = df.groupby('risk_profile')[features].mean()
    matrix_norm = (matrix_df - matrix_df.min()) / (matrix_df.max() - matrix_df.min())
    
    fig6 = px.imshow(matrix_norm,
                     labels=dict(x="Metric", y="Cluster Profile", color="Risk Level"),
                     x=list(display_names.values()),
                     y=matrix_df.index,
                     color_continuous_scale='YlOrRd',
                     aspect="auto",
                     title="<b>Risk DNA Scorecard: Average Forensic Values</b>")

    fig6.update_traces(text=matrix_df.round(2).values, texttemplate="%{text}")
    fig6.write_image(os.path.join(ml_chart_dir, "07_ml_dna_scorecard.png"))

if __name__ == "__main__":
    run_ml_analysis()