import pandas as pd
import numpy as np
import glob
import os

# --- STEP 1: LOAD RAW DATA ---
def load_folder_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.CSV"))
    if not files:
        print(f" Warning: No CSV files found in: {folder_path}")
        return pd.DataFrame()

    all_data = []
    for f in files:
        try:
            temp_df = pd.read_csv(f, low_memory=False)
            temp_df.columns = temp_df.columns.str.strip().str.lower()
            if 'pincode' in temp_df.columns:
                temp_df['pincode'] = temp_df['pincode'].astype(str).str.split('.').str[0].str.zfill(6)
            all_data.append(temp_df)
        except Exception as e:
            print(f" Could not read file {f}: {e}")
            
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# --- STEP 2: THE NUCLEAR HEALING ---
def master_reference_healing(df, master_df):
    """
    This replaces the 'Schizophrenic' raw data with the 
    'Ground Truth' from your Master Pincode File.
    """
    if df.empty: return df
    
    # 1. Prepare Pincodes for a perfect join
    df['pincode'] = df['pincode'].astype(str).str.zfill(6)
    
    # 2. DROP messy state/district columns from raw data
    cols_to_drop = [c for c in ['state', 'district', 'statename'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. JOIN with the Golden Reference (Method 1)
    df = pd.merge(df, master_df, on='pincode', how='left')
    
    # Rename 'statename' to 'state' for compatibility with your other scripts
    df = df.rename(columns={'statename': 'state'})
    
    # Fill gaps for Pincodes not found in the master list
    df['state'] = df['state'].fillna('UNKNOWN')
    df['district'] = df['district'].fillna('UNKNOWN')
    
    return df

def process_and_sum(df, name, keys, master_df):
    if df.empty: return df
    print(f"Healing & Aggregating {name} Dataset...")
    
    # Apply the Master Healing here!
    df = master_reference_healing(df, master_df)
    
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce', format='mixed')
    
    # Now group by the 'Healed' keys
    return df.groupby(keys).sum(numeric_only=True).reset_index()

# --- STEP 3: MASTER FILE PREP (Run once) ---
def prepare_golden_reference(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return None

    print("Creating Golden Reference...")
    master = pd.read_csv(input_file, usecols=['pincode', 'district', 'statename', 'latitude', 'longitude'], low_memory=False)
    
    # Standardize Master names to one unique format
    master['district'] = master['district'].astype(str).str.upper().str.strip()
    master['statename'] = master['statename'].astype(str).str.upper().str.strip()
    master['pincode'] = master['pincode'].astype(str).str.strip().str.zfill(6)
    
    # DEDUPLICATE: 1 Pincode = 1 Record
    master_clean = master.drop_duplicates(subset=['pincode'], keep='first')
    master_clean.to_csv(output_file, index=False)
    return master_clean

# --- STEP 4: MAIN RUNNER ---
def run_cleaner():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 1. Prepare/Load Master Data
    raw_pincode_path = os.path.join(project_root, "all_india_pincode.csv")
    master_clean_path = os.path.join(project_root, "pincode_master_clean.csv")
    
    # If clean file doesn't exist, create it
    if not os.path.exists(master_clean_path):
        master_df = prepare_golden_reference(raw_pincode_path, master_clean_path)
    else:
        master_df = pd.read_csv(master_clean_path, dtype={'pincode': str})

    merge_keys = ['date', 'state', 'district', 'pincode']

    # 2. Load Raw Transaction Data
    base_path = os.path.join(project_root, "datasets")
    d_bio = load_folder_data(os.path.join(base_path, "AADHAR BIOMETRIC DATA"))
    d_demo = load_folder_data(os.path.join(base_path, "AADHAR DEMOGRAPHIC DATA"))
    d_enrol = load_folder_data(os.path.join(base_path, "AADHAR ENROLMENT DATA"))

    # 3. Process with Master Healing
    df_bio = process_and_sum(d_bio, "Biometric", merge_keys, master_df)
    df_demo = process_and_sum(d_demo, "Demographic", merge_keys, master_df)
    df_mon = process_and_sum(d_enrol, "Enrolment", merge_keys, master_df)

    # 4. Merge into Master File
    print("\n--- PERFORMING FINAL MEMORY-SAFE MERGE ---")
    df_updates = pd.merge(df_bio, df_demo, on=merge_keys, how='outer')
    df_master = pd.merge(df_mon, df_updates, on=merge_keys, how='outer').fillna(0)

    # 5. Save final cleaned data
    output_dir = os.path.join(project_root, "output")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    df_master.to_csv(os.path.join(output_dir, "cleaned_master_data.csv"), index=False)
    print(f"SUCCESS! Master file saved. Total rows: {len(df_master)}")

if __name__ == "__main__":
    run_cleaner()