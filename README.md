# 🛡️Aadhaar Setu: National Integrity Audit & Anomaly Detection Framework

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white&style=for-the-badge)
![Backend](https://img.shields.io/badge/Backend-Streamlit-red?logo=streamlit&logoColor=white&style=for-the-badge)
![Engine](https://img.shields.io/badge/Engine-DuckDB-black?logo=duckdb&logoColor=yellow&style=for-the-badge)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn&logoColor=white&style=for-the-badge)
![Data](https://img.shields.io/badge/Data-Pandas-green?logo=pandas&logoColor=white&style=for-the-badge)
![Charts](https://img.shields.io/badge/Charts-Plotly-purple?logo=plotly&logoColor=white&style=for-the-badge)
![PDF](https://img.shields.io/badge/PDF-FPDF2-darkred?style=for-the-badge)


> **A production-grade forensic analytics engine for safeguarding Digital Public Infrastructure (DPI) at national scale.**

Aadhaar Setu is an advanced data intelligence platform that processes millions of transactional records to identify administrative anomalies, diagnose behavioral fraud patterns, and deliver actionable intelligence for ground-level verification teams.


## 🚀 Live demo

🔗 **Dashboard (Live):** [https://aadhaar-data-analysis-audit.streamlit.app](https://aadhaar-data-analysis-audit.streamlit.app)

> **NOTE:** The dashboard is hosted on Streamlit's free tier and may occasionally crash or run out of memory.
> If the site is down, please email `perimilisunil@gmail.com` and I will restart the app and ensure it runs as expected.
> For better experience switch to **LIGHT MODE**

---

## 📖 Project Overview

Aadhaar Setu is a national-scale analytics platform built to convert large volumes of administrative data into clear, actionable audit intelligence. It ingests raw transactional metadata, normalizes inconsistent geographic identifiers, derives operational metrics, and applies unsupervised machine learning to identify unusual or high-risk patterns across regions.

The system is designed for real-world operations rather than research output. Its primary goal is to help administrators answer three practical questions: *Where should we look? What is abnormal? What action should be taken first?* The pipeline produces ranked risk lists, district and pincode level summaries, and exportable work orders that can be directly used by field teams.

Key characteristics:

* End-to-end pipeline from raw data to deployable audit outputs.
* Scales to millions of records using DuckDB and columnar storage.
* Privacy-first design operating only on aggregated, non-PII metadata.
* Outputs focused on operations: ranked targets, dashboards, and downloadable reports.

---

## 💻 Technology Stack

| Core Technologies | Component      | Purpose                                       |
| ----------------- | -------------- | --------------------------------------------- |
| Runtime           | Python 3.12+   | Optimized for memory efficiency               |
| Data Engineering  | Pandas, NumPy  | Vectorized operations                         |
| Machine Learning  | Scikit-Learn   | IsolationForest, KMeans, StandardScaler       |
| Visualization     | Plotly Express | Interactive charts, combo visualizations      |
| Database          | DuckDB         | In-memory analytical queries on Parquet       |
| PDF Engine        | FPDF2          | Custom AadhaarSetuPDF class for dossiers      |
| Web Framework     | Streamlit      | Real-time dashboard with `@st.cache_resource` |

# 🎯 The Three-Tier Architecture

## 1. The Pincode Integrity Lock (PIL) Engine
**Before analysis, the system runs a Geographic Self-Healing Engine.**  
**Logic:** Standardizes fragmented naming metadata across millions of rows using a referential postal master list.  
**Impact:** Ensures a single source of truth — e.g., a district recorded as both `SPSR Nellore` and `Nellore` will be merged automatically to prevent data dilution.

## 2. The Forensic  Engine

**Algorithm 1 — Isolation Forest**  
An ensemble method that isolates anomalies rather than profiling normal data. It identifies pincodes with suspicious ratios of new adult entries versus maintenance velocity.

**Algorithm 2 — K-Means Clustering**  
Every identified anomaly is fingerprinted and grouped into behavioral archetypes:

- **Identity Spoofing Risk:** High adult entry spikes  
- **Maintenance Latency:** Biometric compliance gaps  
- **Operational Velocity Anomaly:** Technical bulk-upload spikes

## 3. The Regional Security Index (RSI)
We developed the RSI (0–10 score). This metric balances enrollment spikes against maintenance health, enabling directors to prioritize audits based on forensic intensity rather than raw population size.

---

## 🏗️ System Architecture

```
                        ┌────────────────────────────────────────────────────────────┐
                        │                     Technology Stack                       │
                        │ Core Technologies    | Component            | Purpose      │
                        └────────────────────────────────────────────────────────────┘
                                              ↓
                        ┌────────────────────────────────────────────────────────────┐
                        │ Raw Data (CSV/Parquet)                                     │
                        │ └─ "Schizophrenic" Locality Data                           │
                        └────────────────────────────────────────────────────────────┘
                                              ↓
                        ┌────────────────────────────────────────────────────────────┐
                        │ PIL Engine (cleaner.py)                                    │
                        │ ├─ master_reference_healing()                              │
                        │ └─ Golden Reference Merge (SSOT)                           │
                        └────────────────────────────────────────────────────────────┘
                                              ↓
                        ┌────────────────────────────────────────────────────────────┐
                        │ Forensic AI Pipeline                                       │
                        │ ├─ Analysis Metrics (analysis.py)                          │
                        │ ├─ K-Means Clustering (ml_analysis.py)                     │
                        │ └─ Isolation Forest (ml_deep_analysis.py)                  │
                        └────────────────────────────────────────────────────────────┘
                                              ↓
                        ┌────────────────────────────────────────────────────────────┐
                        │ Intelligence Layer (dashboard.py)                          │
                        │ ├─ Streamlit Interface (DuckDB Backend)                    │
                        │ └─ PDF Dossier Generator (project_pdf.py)                  │
                        └────────────────────────────────────────────────────────────┘
```

---


## 📐 Installation & Setup

**Prerequisites**

* Python 3.12 or higher
* 4GB RAM minimum 
* Git

**Quick Start**

```bash
# Clone the repository
git clone https://github.com/perimilisunil/Aadhaar_Data_Analysis.git
cd Aadhaar_Data_Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# 1. Run the Cleaner (ETL)
python src/cleaner.py

# 2. Run the Analysis Modules
python src/analysis.py
python src/ml_analysis.py
python src/ml_deep_analysis.py

# 3. PDF Report
python src/project_pdf.py

# 4. Launch the Dashboard
streamlit run src/dashboard.py
```

**Dependencies**

```
streamlit>=1.52.2
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
scikit-learn>=1.3.0
duckdb>=0.9.2
fpdf2>=2.7.0
pillow>=10.0.0
```

---

## ✍️ Usage Guide

**Dashboard Navigation**

1. **Sidebar Controls**

   * State Filter: `sel_state` (Selects specific state or "INDIA").
   * Risk Profile Filters: Adult Entry Spikes, Child Biometric Lags, etc.
   * Date Range: Select "From" and "To" months for temporal analysis.
   * Pincode Enquiry: Enter 6-digit PIN for Deep Scan.
   * Download Report: Generates confidential PDF dossie 

2. **Tab Organization**

   * **Tab 1:**
     ***Executive Overview:***
        - Administrative Pulse.
        - Service Demand Split.
        - KPI Command Row
   * **Tab 2:**
     ***ML-Powered Insights***
       - Static Charts (National baselines from ML pipeline)
      - Live DNA Heatmap (District-level forensic fingerprinting
      - Threat Signatures
    * **Tab 3:**
      ***Strategic Action :***
        - Root-Cause Analysis (Feature importance + policy zones)
        - Risk Driver Impact Chart (Live volume analysis)
        - High-Priority Audit List (Top 45 flagged sites)
        - CSV Export (Field work order download)
    * **Tab 4:**
        ***Operational Analysis***
      -  National Baselines (Static benchmarks)
      -  Pressure Index State-District (Live calculation)
    * **Tab 5:**
      ***Pincode Drilldown***
      - Pincode Search (6-digit input + form submission)
      - 15-PIN Cluster Chart (Risk hierarchy within district)
      - Field Evidence Table

3. **Export Options**

   - PDF Report: Full "National Integrity Audit Dossier" via `project_pdf.py`.
   - CSV Audit Plan: Field work orders with "Recommended Action".
   - Pincode Work Order: Cluster-specific verification tasks.

---

## API Example (Programmatic Access)

```python
import pandas as pd
from src.project_pdf import generate_forensic_dossier

# Load processed data
df = pd.read_parquet('output/final_audit_report.parquet')

# Filter for specific high-risk zone
zone_data = df[df['integrity_score'] > 8]

# Generate Evidence Dossier
pdf_bytes = generate_forensic_dossier(
    df=zone_data,
    state_name="Andhra Pradesh",
    root_path=".",
    team_id="UIDAI_11060"
)

with open('AadhaarSetu_Audit_Report.pdf', 'wb') as f:
    f.write(pdf_bytes)
```

---

## 🛎️ Key Features

###  Data Intelligence

* [x] PIL Engine (`cleaner.py`): "Nuclear Healing" of pincode/district mismatches.
* [x] Metrics (`analysis.py`): Calculates `service_delivery_rate` and `security_anomaly_score`.
* [x] Performance: Uses DuckDB for sub-second queries on 2.2M+ records.

###  Machine Learning

* [x] Isolation Forest (`ml_deep_analysis.py`): 5% contamination threshold for anomaly detection.
* [x] K-Means Clustering (`ml_analysis.py`): segments districts into "High-Risk", "Service Gap", "Migration Zones".
* [x] Root Cause: Feature importance analysis for primary risk drivers.

###  Visualization

* [x] Interactive Charts: Plotly Treemaps, Radar Charts (`go.Scatterpolar`), and Ribbon Charts.
* [x] Operational Friction: Visualizes workload vs. security pressure.
* [x] DNA Scorecard: Heatmap of normalized forensic values.

###  Operational Intelligence

* [x] Automated Reporting (`project_pdf.py`): Generates 20+ page investigative PDFs.
* [x] Tactical Pivot: "Target-in-middle" cluster logic for field rerouting.
* [x] Strategic Zones: Classification into Zones A, B, C, D for policy planning.

###  Production Readiness

* [x] Streamlit Optimization: Uses `@st.cache_data` and memory management.
* [x] Robust Error Handling: Safe mode for missing images/data in PDF generation.
* [x] Data Lineage: Parquet-based workflow from Raw -> Cleaned -> Analyzed.

---

## 🔏 Privacy & Compliance

**Data Protection Principles**

* No PII Processing: System operates on obfuscated metadata only.
* No Access To: Names, Date of Birth, Biometric strings, Aadhaar numbers.
* Unsupervised Learning: Removes demographic bias through algorithmic neutrality (K-Means/Isolation Forest).
* Transparency: All transformations logged in `cleaner.py` and `ml_deep_analysis.py`.

---

## 📂 Project Structure

```
Aadhaar_Data_Analysis/
├── src/
│   ├── cleaner.py             # PIL Engine (Nuclear Pincode Healing)
│   ├── analysis.py            # Base Metrics & Z-Score Analysis
│   ├── ml_analysis.py         # K-Means Clustering & DNA Scorecards
│   ├── ml_deep_analysis.py    # Isolation Forest & Strategic Portfolio
│   ├── dashboard.py           # Streamlit Command Center
│   └── project_pdf.py         # FPDF2 Report Generator
├── datasets/
│   ├── pincode_master_clean.csv  # Golden Reference (SSOT)
│   └── [Raw CSV Data Folders]    # Source Biometric/Demographic data
├── output/
│   ├── cleaned_master_data.csv   # Post-PIL healed data
│   ├── final_audit_report.csv    # Final ML-scored data
│   ├── charts/                   # Static visualizations
│   ├── ML_Anomaly_charts/        # Clustering visuals
│   └── Deep_Analysis/            # Strategic portfolios
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛞 Performance Benchmarks

**System Metrics (Streamlit Cloud - Free Tier)**

| Metric             |       Value | Notes                           |
| ------------------ | ----------: | ------------------------------- |
| Concurrent Users   |  5-7 active | With 60% data sampling (DuckDB) |
| Data Load Time     | < 3 seconds | Cached via `@st.cache_data`     |
| Dashboard Response |     < 200ms | Real-time filter updates        |
| PDF Generation     |      10-15s | Includes chart embedding        |
| Memory Footprint   |   350-500MB | Base data + per-user overhead   |

**Optimization Strategies**

* DuckDB: In-memory SQL queries on Parquet/CSV.
* Parquet Storage: Compressed, schema-aware storage.
* Lazy Loading: Images loaded on-demand per tab.
* Garbage Collection: Explicit `gc.collect()` usage.

---

## 🛠️ Development Roadmap

**Phase 1: Foundation**

* [x] PIL Engine implementation (`cleaner.py`).
* [x] Basic metric calculation (`analysis.py`).
* [x] PDF report generation (`project_pdf.py`).

**Phase 2: Intelligence**

* [x] K-Means behavioral clustering (`ml_analysis.py`).
* [x] Isolation Forest anomaly detection (`ml_deep_analysis.py`).
* [x] RSI scoring algorithm.
* [x] Interactive treemaps and radar charts.

**Phase 3: Production (Current)**

* [x] Multi-user optimization (DuckDB).
* [x] Memory management for free tier.
* [x] Pincode search with security pivot.
* [ ] API endpoint deployment.

---

## 🧳Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---
---

## 📃 Acknowledgments

**Built For:** National Aadhaar Hackathon 2026

**Safeguarding India's Digital Public Infrastructure**

**Technical Inspiration:**

* Isolation Forest paper by Liu, Ting & Zhou (2008).
* Streamlit's philosophy of "data apps in pure Python".

**Special Recognition:** Developed with a commitment to privacy-first design and operational excellence in public sector technology.


---
## 📥 Contact & Support

### Project Maintainer
**Sunil Kumar**  
🔗 [GitHub](https://github.com/perimilisunil)  
🔗 [LinkedIn](https://www.linkedin.com/in/perimili-sunil-kumar-bb22b3300?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
📧 [perimilisunil@gmail.com](mailto:perimilisunil@gmail.com)

### Community Resources
-  [Report Issues](https://github.com/perimilisunil/Aadhaar_Data_Analysis/issues) - Bug reports and feature requests
-  [Discussions](https://github.com/perimilisunil/Aadhaar_Data_Analysis/discussions) - General questions and ideas
-  [Wiki](https://github.com/perimilisunil/Aadhaar_Data_Analysis/wiki) - Extended documentation
-  [Star this repo](https://github.com/perimilisunil/Aadhaar_Data_Analysis) if you find it useful!

---


