import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Paths
data_folder = "bidmc_csv"
output_folder = "processed_csv"
os.makedirs(output_folder, exist_ok=True)

sampling_rate = 125  # Hz

for i in range(1, 54):
    subj_id = f"{i:02d}"
    base = f"bidmc_{subj_id}"
    try:
        # File paths
        signals_file = os.path.join(data_folder, f"{base}_Signals.csv")
        numerics_file = os.path.join(data_folder, f"{base}_Numerics.csv")
        breaths_file = os.path.join(data_folder, f"{base}_Breaths.csv")
        fix_file = os.path.join(data_folder, f"{base}_Fix.txt")

        # --- Load Data ---
        signals = pd.read_csv(signals_file)
        numerics = pd.read_csv(numerics_file)
        breaths = pd.read_csv(breaths_file, header=None, names=['ann1', 'ann2'])

        # --- Extract Age & Gender ---
        age, gender = None, None
        if os.path.exists(fix_file):
            with open(fix_file, 'r') as f:
                for line in f:
                    if "Age:" in line:
                        age = int(line.strip().split("Age:")[1].strip())
                    elif "Gender:" in line:
                        gender = line.strip().split("Gender:")[1].strip()
        else:
            print(f"[!] {base}_Fix.txt not found.")

        # --- Time Handling ---
        if 'Time [s]' not in signals.columns:
            signals['Time [s]'] = [i / sampling_rate for i in range(len(signals))]
        signals['Time'] = pd.to_timedelta(signals['Time [s]'], unit='s')
        signals.set_index('Time', inplace=True)

        if 'Time [s]' not in numerics.columns:
            numerics['Time [s]'] = [i for i in range(len(numerics))]
        numerics['Time'] = pd.to_timedelta(numerics['Time [s]'], unit='s')
        numerics.set_index('Time', inplace=True)

        # --- Standardize RESP and PLETH ---
        scaler = StandardScaler()
        for sig in ['RESP', 'PLETH']:
            if sig in signals.columns:
                signals[sig] = scaler.fit_transform(signals[[sig]])
            else:
                print(f"[!] {sig} not found in {base}, skipping.")

        # --- Resample to 1 Hz ---
        resampled = {}
        for sig in ['RESP', 'PLETH']:
            if sig in signals.columns:
                resampled[sig] = signals[sig].resample('1S').mean()

        # --- Create RR Features from Breaths ---
        breath_times_sec = breaths['ann1'] / sampling_rate
        rr_intervals = np.diff(breath_times_sec)
        rr_times = breath_times_sec[:-1]
        rr_df = pd.DataFrame({
            'RR_Time': pd.to_timedelta(rr_times, unit='s'),
            'RR_interval': rr_intervals,
            'Breathing_Rate_BPM': 60 / rr_intervals
        }).set_index('RR_Time')

        # --- Merge ---
        combined = numerics.copy()
        for sig in resampled:
            combined[sig] = resampled[sig]
        combined = combined.merge(rr_df, how='left', left_index=True, right_index=True)

        # Interpolate missing
        combined = combined.interpolate().dropna()

        # Add age & gender
        combined['Age'] = age
        combined['Gender'] = gender

        # --- Save ---
        out_path = os.path.join(output_folder, f"{base}_Processed.csv")
        combined.to_csv(out_path)
        print(f"[✓] Saved: {out_path}")

    except Exception as e:
        print(f"[X] Failed for {base}: {e}")
