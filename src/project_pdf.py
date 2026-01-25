import pandas as pd
import numpy as np
from fpdf import FPDF
import os
import gc
from datetime import datetime
from PIL import Image

# --- 1. SYSTEM THEME (FIXED) ---
THEME = {
    "primary": (30, 58, 138),
    "muted": (100, 116, 139),
    "alert": (180, 83, 9),
    "background": (248, 250, 252)
}

# --- 2. STABILITY HELPERS ---
def clean_text(text):
    if not isinstance(text, str): return str(text)
    text = text.replace("—", "-").replace("–", "-").replace("•", "-")
    text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    return "".join(i for i in text if ord(i) < 128)

def safe_pincode(val):
    try:
        if val is None or pd.isna(val): return "000000"
        s = str(val).split('.')[0]
        s = "".join(ch for ch in s if ch.isdigit())
        return s.zfill(6)[:6]
    except: return "000000"

def safe_mode(series, fallback="Baseline Activity"):
    if series is None or series.empty: return fallback
    try:
        res = series.dropna().mode()
        return res.iat[0] if not res.empty else fallback
    except: return fallback

def ensure_image_size(path, max_px=1200):
    """Optimized: Reduced max_px from 1400 to 1200 to save memory"""
    if not os.path.exists(path): return path
    try:
        tmp_dir = os.path.join("output", "tmp_reports")
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, os.path.basename(path))
        
        with Image.open(path) as im:
            if max(im.size) > max_px:
                im.thumbnail((max_px, max_px))
            # Increased compression to save memory
            im.save(out_path, optimize=True, quality=85)
        return out_path
    except: 
        return path

def cleanup_temp_files():
    """Clean up old temporary report files"""
    try:
        tmp_dir = os.path.join("output", "tmp_reports")
        if os.path.exists(tmp_dir):
            for f in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except:
                    pass
    except:
        pass

class AadhaarSetuPDF(FPDF):
    def __init__(self, team_id="UIDAI_11060"):
        super().__init__()
        self.team_id = team_id

    def header(self):
        self.set_line_width(0.3)
        self.set_draw_color(*THEME["primary"]) 
        self.rect(8, 8, 194, 281)
        if self.page_no() > 1:
            self.set_font('Helvetica', '', 10)
            self.set_text_color(*THEME["muted"])
            self.cell(0, 10, f"TEAM ID: {self.team_id} | NATIONAL INTEGRITY AUDIT DOSSIER", 0, 1, 'R')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*THEME["muted"])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10, f'Aadhaar Setu | UIDAI | Generated: {ts}', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')

# --- 3. MEMORY-SAFE GENERATOR ---
def generate_forensic_dossier(df, state_name, root_path, search_pin=None, team_id="UIDAI_11060"):
    """
    CRITICAL OPTIMIZATION: This function now works with filtered dataframes
    passed from the dashboard (view_df), not the full dataset.
    """
    try:
        # Clean up old temp files first
        cleanup_temp_files()
        
        # CRITICAL FIX: Don't copy! Just filter directly
        # The df passed here should already be view_df from dashboard
        if state_name == "INDIA":
            macro_df = df  # No copy needed
        else:
            # Only filter if needed, and use a view not a copy
            macro_df = df[df['state'] == state_name] if 'state' in df.columns else df
        
        # Process pincode search
        target_obj = None
        if search_pin:
            p_match = df[df['pincode_str'] == str(search_pin).strip()] if 'pincode_str' in df.columns else pd.DataFrame()
            if not p_match.empty: 
                target_obj = p_match.iloc[0]

        pdf = AadhaarSetuPDF(team_id)
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.set_font('Helvetica', '', 12)

        # --- PAGE 1: COVER ---
        pdf.add_page()
        pdf.set_y(100)
        pdf.set_font('Helvetica', 'B', 48)
        pdf.set_text_color(*THEME["primary"])
        pdf.cell(0, 25, "AADHAAR SETU", 0, 1, 'C')
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(*THEME["muted"])
        pdf.cell(0, 12, "National Integrity Audit & Strategic Intelligence Dossier", 0, 1, 'C')
        pdf.ln(30)
        pdf.set_y(230)
        pdf.set_font('Helvetica', '', 12)
        pdf.multi_cell(0, 8, "A high-fidelity analytical engine designed to detect administrative outliers, diagnose behavioral risks, and optimize jurisdictional integrity across Million transactional records.", align='C')
        
        # --- PAGE 2: INDEX ---
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(*THEME["primary"])
        pdf.cell(0, 20, "Document Index", 0, 1)
        pdf.set_font('Helvetica', '', 13)
        pdf.set_text_color(0)
        toc = ["Executive Summary", "Technical KPI Definitions", "Dashboard Interaction Logic", "System Architecture & AI Pipeline", "Chart 1: National Service Demand Analysis", "Chart 2: Audit Priority Leaderboard", "Chart 3: National Risk Hierarchy", "Chart 4: Temporal Pulse & Seasonal Trends", "Chart 5: Behavioral DNA Signature", "Chart 6: Forensic Cluster Scorecard", "Chart 7: DNA Risk Heatmap", "Chart 8: Systemic Risk Driver Impact", "Chart 9: Regional Anomaly Concentration", "Chart 10: Executive Audit Master-List", "Chart 11: Strategic Policy Portfolio", "Interface Proof: National Command UI", "Interface Proof: Regional Heatmap Interaction", "Innovation Case Study: Tactical Security Pivot", "Technical Compliance & Data Lineage", "Source Code Appendix"]
        for i, item in enumerate(toc, 1): 
            pdf.cell(0, 10, f"{i}. {item}", 0, 1)

        # --- PAGE 3: EXECUTIVE SUMMARY ---
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(*THEME["primary"])
        pdf.cell(0, 20, "1. Executive Summary", 0, 1)
        pdf.set_font('Helvetica', '', 12)
        pdf.set_text_color(0)
        pdf.multi_cell(180, 10, clean_text(
            "Aadhaar Setu addresses the critical challenge of maintaining database integrity across millions of records. "
            "Administrative data at this scale inherently contains geographic fragmentation and behavioral outliers that can compromise jurisdictional trust. "
            "Our system provides an automated framework that standardizes nomenclature, isolates risk via Isolation Forest ML, and generates prescriptive field actions. "
            "By implementing a Regional Security Index (RSI), we provide UIDAI directors with a single source of truth for jurisdictional health. "
            "This dossier serves as proof of system readiness, demonstrating a complete pipeline from raw data healing to tactical field rerouting for UIDAI officers."
        ))

        # --- PAGE 4: KPI DEFINITIONS ---
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 20, "2. Technical KPI Blueprint", 0, 1)
        kpis = [
            ("Audit Scope", "Use Case: Recalibrates the engine baseline for localized accuracy."),
            ("Unique Pincodes", "Use Case: Confirms 100% ETL coverage of the geographic dataset."),
            ("High Risk Sites", "Use Case: Flags locations in the 90th percentile of behavioral deviation."),
            ("Integrity Score (RSI)", "Use Case: 0-10 index balancing enrollment spikes vs maintenance velocity."),
            ("MBU Compliance", "Use Case: Monitors biometric update rates to prevent future identity exclusion."),
            ("Records Analyzed", "Use Case: Proof of system performance in high-volume environments.")
        ]
        for k, d in kpis:
            pdf.set_font('Helvetica', 'B', 13)
            pdf.set_text_color(*THEME["primary"])
            pdf.cell(0, 10, k, 0, 1)
            pdf.set_font('Helvetica', '', 12)
            pdf.set_text_color(0)
            pdf.multi_cell(180, 8, d)
            pdf.ln(2)

        # MEMORY OPTIMIZATION: Force cleanup before loading images
        gc.collect()

        # --- CHART PAGES (Optimized - only include essential charts) ---
        essential_charts = [
            ("charts/01_service_demand_split.png", "3. Chart 1: National Service Demand", "- Segments demand into infants, children, and adults.\n- Compares new enrolment versus system updates.", "Analysis: This chart visualizes the fundamental load on the system. By segmenting demand into lifecycle stages, we can see if resource allocation matches population needs. In high-volatility areas, we often observe a deficit in maintenance activity relative to new registrations. This insight allows the UIDAI to rebalance the distribution of enrolment kits. When child update activity lags behind enrolment in the 5-17 bracket, it signals a systemic failure in the MBU mandate. This chart acts as the primary triage tool for national-level planning, ensuring that the next generation of citizens is not excluded from biometric functional services due to administrative bottlenecks. It identifies the 'Service Identity' of a state: whether it is an expansion zone or a maintenance hub. By ensuring that updates stay proportional to enrolments, we maintain a healthy database lifecycle."),
            ("charts/02_risk_leaderboard.png", "4. Chart 2: Audit Priority Leaderboard", "- Ranks districts by security anomaly scores.\n- Normalizes risk independent of population size.", "Analysis: The leaderboard provides the immediate priority list for regional directors. this visual normalizes for population, ensuring that a small district with high-intensity fraud is not ignored. It serves as the daily list for field audit teams. By monitoring which districts repeatedly appear in the top 25, administrators can identify systemic local corruption versus one-time technical errors. The scale ensures that different states can be compared on an even playing field, allowing the center to allocate  officers to the most volatile regions effectively. It provides a visual of regional improvements; as audits are performed and protocols fixed, districts drop out of the ranking, providing clear visual proof of security enhancement over time."),
            ("charts/03_risk_treemap.png", "5. Chart 3: National Integrity Treemap", "- National risk hierarchy of Million records.\n- Size denotes volume; color denotes intensity.", "Analysis: The Treemap is the executive 'Command Center' visual. It allows a Director to see the entire nation's health in one glance. A large green block represents a high-volume, high-trust jurisdiction, whereas a dark red block signals a critical failure zone. This solves the problem of data fragmentation by grouping all sub-districts under their healed state names. It is the tool for quarterly budget planning. Directors use this to decide which state hubs require new server infrastructure to handle maintenance load and which regions require increased technical oversight to reduce the overall risk intensity score. It handles the full Million record scale without visual clutter, making it the primary tool for cross-departmental security briefings."),
            ("charts/04_seasonal_performance_ribbon.png", "6. Chart 4: Temporal Pulse Analysis", "- Tracks maintenance trends over a 12-month period.\n- Exposes seasonal security pressure windows.", "Analysis: The Pulse Ribbon tracks the heartbeat of the administrative system. By correlating time with maintenance-to-enrolment ratios, we can identify seasonal risk waves. We have observed that integrity risks often spike during months where compliance velocity drops. This visual allows for 'Predictive Staffing.' If UIDAI knows that maintenance velocity traditionally falls in March, they can deploy additional verification layers during that window. It transforms the audit process from a static historical review into a dynamic, time-aware security strategy that protects the database during peak demand cycles throughout the year. The color-coded bars provide an instant 'Temperature Check' of national operations, allowing for early detection of multi-month administrative burnout."),
            ("ML_Anomaly_charts/05_ml_threat_radar.png", "7. Chart 5: Behavioral Radar", "- Identifies the DNA fingerprint of each risk cluster.\n- Distinguishes between technical and procedural gaps.", "Analysis: The Radar chart is the output of our behavioral archetyping model. It maps four forensic drivers to create a unique threat signature for each region. If the signature spikes in Adult Spikes, it indicates potential illicit registrations. If it spikes in Child Lags, it indicates a service delivery failure. This insight is critical because it tells the UIDAI what type of officer to send: a technical support officer for service gaps or an auditor for security risks. This chart ensures that the system doesn't just find problems, but diagnoses their nature, saving thousands of hours in unnecessary manual field inspections. The 'Safe Baseline' acts as a constant benchmark, allowing us to see how far a suspicious cluster has moved from the expected standard of operation."),
            ("ML_Anomaly_charts/06_ml_forensic_scorecard.png", "8. Chart 6: Forensic Cluster Scorecard", "- Side-by-side comparison of cluster magnitudes.\n- Provides numeric evidence for policy justification.", "Analysis: This scorecard quantifies the differences between our three ML-identified clusters. It serves as the numeric backbone of the report. When an administrator needs to justify a budget for new mobile vans or additional training, this chart provides the empirical proof: it shows the exact magnitude of the Service Gap cluster compared to the national average. It converts complex multi-dimensional ML results into a simple bar format that non-technical decision-makers can understand instantly. It is the primary reference used when translating data science into formal government policy memos and resource deployment orders for regional centers."),
            ("ML_Anomaly_charts/07_ml_dna_scorecard.png", "9. Chart 7: DNA Risk Heatmap", "- Normalized matrix of cluster-driver intersections.\n- Facilitates rapid pattern spotting for triage.", "Analysis: The DNA Heatmap provides a high-density view of how behavioral clusters interact with forensic metrics. By normalizing each column, we ensure that small-scale fraud is visible alongside large-scale trends. It allows triage teams to spot 'Metric Pairing' - where high adult registration meets low child compliance. This specific pairing is a high-confidence indicator of identity spoofing. The heatmap acts as the training baseline for our automated flagging logic, ensuring that the prescribed actions are based on a statistically significant intersection of forensic evidence. During a field audit, an officer can use this heatmap to see exactly which Driver is most corrupted in their specific cluster."),
            ("Deep_Analysis/08_global_feature_importance.png", "10. Chart 8: Risk Driver Impact", "- Measures the weight of each driver in the pool.\n- Identifies the primary cause of national anomalies.", "Analysis: This chart identifies what is breaking the national integrity system. If the primary driver is Child Biometric Lags, then the national strategy must pivot to MBU camps. If it is Adult Entry Spikes, the strategy must pivot to manual document verification. By identifying the dominant driver across Million records, we provide the UIDAI with a clear National Priority. It prevents regional offices from wasting resources on low-impact fixes and focuses the entire organization on the most significant threats to database integrity. This is the Executive Summary of our Machine Learning findings, telling the jury exactly which features of the dataset are the most predictive of risk."),
            ("Deep_Analysis/09_state_anomaly_concentration.png", "11. Chart 9: Regional Anomaly Concentration", "- Maps geographic density of suspicious sites.\n- Exposes risk corridors across state borders.", "Analysis: The concentration map reveals Fraud Corridors. Often, administrative anomalies move across borders due to migration or shared operator networks. This visual allows the UIDAI to see if a risk is localized to one rogue operator or systemic to a whole state. This insight is vital for border districts and high-migration states. It allows for Cluster Auditing where multiple centers in a geographic corridor are audited simultaneously to prevent the movement of illicit activity. It also assists in logistics planning for mobile update units, ensuring they are sent to the center of the highest-density risk clusters."),
            ("Deep_Analysis/10_final_audit_summary.png", "12. Chart 10: Executive Audit Master-List", "- High-priority table for regional verification.\n- Directly links ML scores to physical pincodes.", "Analysis: This is the execution layer of Aadhaar Setu. It is the final result of the entire Million record analysis. It lists the top sites identified by the Isolation Forest model as requiring immediate field intervention. For each site, it provides the RSI score and the forensic diagnosis. This is the canonical document that a Regional Director hands to their field audit team. It moves the project from the dashboard into the real world. By providing a clear, ranked, and diagnosed list, it eliminates administrative guesswork and ensures that every field visit has a high probability of success."),
            ("Deep_Analysis/11_strategic_portfolio.png", "13. Chart 11: Strategic Portfolio", "- Classifies districts into four deployment zones.\n- Guides long-term infrastructure and training policy.", "Analysis: Here it is the long-term policy roadmap. It classifies jurisdictions into four quadrants based on their Integrity Score and Engagement Level. Zone A districts require forensic audits; Zone D districts require awareness camps. This chart is used for long-term planning (1-3 years). It helps the UIDAI decide where to build new permanent centers and where to retire legacy enrolment kits. It is the ultimate proof of our system's ability to act as a Decision Support System (DSS) for the highest levels of government, transforming complex data into a clear administrative strategy.")
        ]

        for path, title, bullets, para in essential_charts:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 20)
            pdf.set_text_color(*THEME["primary"])
            pdf.cell(0, 15, title, 0, 1)
            pdf.set_font('Helvetica', '', 12)
            pdf.set_text_color(0)
            pdf.multi_cell(180, 8, clean_text(bullets))
            pdf.ln(5)
            
            full_path = os.path.join(root_path, "output", path)
            if os.path.exists(full_path): 
                optimized_img = ensure_image_size(full_path)
                pdf.image(optimized_img, x=15, w=180)
                # Clean up immediately after use
                try:
                    if optimized_img != full_path and os.path.exists(optimized_img):
                        os.remove(optimized_img)
                except:
                    pass
            else: 
                pdf.cell(180, 40, "[Visual Evidence Pending]", 1, 1, 'C')
            
            pdf.ln(10)
            pdf.multi_cell(180, 7, clean_text(para))
            
            # Force garbage collection after each chart
            gc.collect()

        # --- DASHBOARD PROOFS ---
        pdf.add_page()
        pdf.set_font('Helvetica', '', 20)
        pdf.cell(0, 20, "14. Dashboard Interactivity Proof", 0, 1)
        ui_img = os.path.join(root_path, "output", "charts", "dashboard_ui.png")
        if os.path.exists(ui_img): 
            pdf.image(ensure_image_size(ui_img), x=30, w=180)
        pdf.set_y(160)
        pdf.set_font('Helvetica','',12)
        pdf.multi_cell(180, 8, clean_text(
            "The dashboard confirms the system is live and tracking over 2.2 million records. It displays key metrics, including a 92.4% integrity score and 13,969 high-risk sites that need attention. Administrators can use the sidebar to filter data. The main charts clearly identify states with the most suspicious activity, specifically highlighting states like MAHARASHTRA and ASSAM . It also ranks the primary causes of these risks using machine learning categories. The interface updates instantly to show how these risks affect different regions. This tool effectively turns complex data into a clear action plan for field operations."
        ))
        pdf.add_page();
        pdf.set_font('Helvetica', '', 20);
        pdf.cell(0, 20, "15. Heatmap Intelligence Proof", 0, 1)
        st_img2 = os.path.join(root_path, "output", "charts", "state_evidence_map.png")
        if os.path.exists(st_img2): pdf.image(ensure_image_size(st_img2), x=30, w=180)
        pdf.set_y(160);
        pdf.set_font('Helvetica','',12)
        pdf.multi_cell(180, 8, clean_text("The Regional Integrity Hierarchy visualization functions as a heat map for state-level surveillance, specifically isolating high-variance districts within Uttar Pradesh. By utilizing a divergent color scale representing Relative Risk %, the heatmap immediately distinguishes compliant zones from critical administrative anomalies; notably, districts such as Sitapur, Bareilly, and Shahjahanpur are rendered in deep crimson, indicating risk saturation levels approaching the 20% upper threshold. Conversely, regions like Ayodhya and Lucknow appear in low-risk gradients, establishing a clear baseline for comparative analysis. This spatial data is corroborated by the "Administrative Pulse" temporal graph below, which tracks the intersection of MBU Compliance Efficiency—peaking in September 2025—against fluctuating Risk Intensity trends. This dual-layer visibility empowers regional managers to bypass low-priority areas and dynamically re-route field audit teams directly to the "Red Zone" hotspots identified in the treemap, optimizing the deployment of limited ground assets."))
        # --- INNOVATION CASE STUDY ---
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 20, "16. Innovation: Security Pivot", 0, 1)
        
        # Use limited sample for table
        sample = target_obj if target_obj is not None else (
            macro_df.sort_values('integrity_risk_pct', ascending=False).iloc[0] 
            if not macro_df.empty and 'integrity_risk_pct' in macro_df.columns 
            else None
        )
        
        if sample is not None and 'district' in macro_df.columns:
            safe_h = macro_df[macro_df['district'] == sample['district']].sort_values('integrity_risk_pct').iloc[0]
            pdf.set_font('Helvetica', '', 12)
            pdf.multi_cell(180, 8, clean_text(
                f"Innovation: When PIN {safe_pincode(sample['pincode'])} is flagged, "
                f"the engine reroutes citizens to PIN {safe_pincode(safe_h['pincode'])} (Verified Center). "
                f"Below is the tactical data table for this cluster:"
            ))
            
            pdf.ln(5)
            # MEMORY OPTIMIZATION: Limit table to 8 rows instead of 10
            dist_all = macro_df[macro_df['district'] == sample['district']].sort_values(
                'integrity_risk_pct', ascending=False
            ).head(8)
            table_data = [("PINCODE", "DISTRICT", "RISK %", "ML DIAGNOSIS", "REQUIRED ACTION")]

            for _, row in dist_all.iterrows():
                table_data.append((
                    safe_pincode(row['pincode']), 
                    str(row['district'])[:15],  # Shorten to prevent overlap
                    f"{row['integrity_risk_pct']:.1f}%", 
                    str(row.get('risk_diagnosis', 'N/A'))[:20], 
                    str(row.get('action_map', 'Review Required'))[:30] 
                ))

            with pdf.table(
                borders_layout="SINGLE_TOP_LINE", 
                cell_fill_color=(248, 250, 252), 
                cell_fill_mode="ROWS", 
                line_height=8, 
                width=190
            ) as table:
                for i, row_d in enumerate(table_data):
                    row_c = table.row()
                    pdf.set_font('Helvetica', 'B' if i==0 else '', 8)
                    if i==0: 
                        pdf.set_fill_color(*THEME["primary"])
                        pdf.set_text_color(255)
                    else: 
                        pdf.set_text_color(0)
                    for it in row_d: 
                        row_c.cell(it)

        # Clean up before reading source files
        gc.collect()

        # --- COMPLIANCE ---
        pdf.add_page()
        pdf.set_y(50)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 15, "17. Compliance and Ethics", 0, 1)
        pdf.set_font('Helvetica', '', 12)
                # Set the common width and alignment
        pdf.set_x(15) 
        
        # 1. Print the first three lines as a regular multi_cell
        pdf.multi_cell(180, 9, 
            "Privacy: No direct identifiers (Names, DOB) were processed.\n"
            "Logic: Unsupervised Machine Learning Algorithms.\n"
            "Maintenance: Commits to one-year support.", 
            border=0, align='L')
        
        # 2. Print the first part of the last line
        pdf.set_x(15)
        pdf.write(9, "Provenance: Official UIDAI datasets. Provided by ")
        
        # 3. Print the clickable link in Blue/Underlined
        pdf.set_text_color(0, 0, 255)
        pdf.set_font("helvetica", "U", 10)
        pdf.write(9, "data.gov.in", "https://data.gov.in")
        
        # 4. Reset font to normal for whatever comes next
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "", 10)
        pdf.ln(10)
                
        # Create the clickable link
        pdf.set_text_color(0, 0, 255)  # Set color to blue for standard link look
        pdf.set_font("helvetica", "U", 10)  # Add 'U' for underline
        repo_url = "https://github.com/perimilisunil/Aadhaar_Data_Analysis"
        # The 'link' parameter makes it clickable
        pdf.cell(0, 10, f"Technical provenance and full source code repository: {repo_url}", 0, 1, 'C', link=repo_url)
        
        # --- SOURCE CODE (OPTIMIZED) ---
        # OPTIMIZATION: Only include dashboard.py to save memory
        code_files = [("cleaner.py", "ETL Engine"), ("ml_deep_analysis.py", "Risk Engine "), ("dashboard.py", "Interface controller")]

        for fname, fshort in code_files:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 22)
            pdf.cell(0, 15, f"Appendix: {fname}", 0, 1)
            fpath = os.path.join(root_path, "src", fname)
            
            if os.path.exists(fpath):
                # CRITICAL FIX: Check file size before reading
                file_size = os.path.getsize(fpath)
                if file_size > 100000:  # Skip files over 100KB
                    pdf.set_font('Helvetica', '', 12)
                    pdf.cell(0, 10, "[File too large - see repository]", 0, 1)
                else:
                    pdf.set_font('Courier', '', 8)
                    pdf.set_text_color(50, 50, 50)
                    pdf.set_x(15)
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            # Read only first 40 lines instead of 60
                            lines = [f.readline() for _ in range(40)]
                            pdf.multi_cell(180, 4, clean_text("".join(lines) + "\n... [TRUNCATED]"))
                    except:
                        pdf.cell(0, 10, "[Could not read file]", 0, 1)

        # Final cleanup
        cleanup_temp_files()
        gc.collect()

        return bytes(pdf.output())
        
    except Exception as e:
        # Error fallback
        pdf = AadhaarSetuPDF(team_id)
        pdf.add_page()
        pdf.set_font('Helvetica','B',12)
        pdf.cell(0,10, clean_text(f"Generation Error: {str(e)}"), 0, 1, 'C')
        return bytes(pdf.output())
