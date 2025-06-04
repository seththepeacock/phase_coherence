import pandas as pd
from N_xi_fit_funcs import *



for species in ['Owl', 'Human', 'Anole']:
    for wf_idx in range(4):
        print(f"Checking {species} {wf_idx}")
        sheet_name = species if species != 'Anole' else 'Anolis'
        df = pd.read_excel(r'N_xi Fits/2024.07analysisSpreadsheetV8_RW.xlsx', sheet_name=sheet_name)
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx) 
        # This one has trailing whitespace in Becky's excel sheet
        if (species, wf_idx) == ('Owl', 3): 
            wf_fn += ' '
        df_wf = df[df['rootWF'].str.split(r'/').str[-1] == wf_fn].copy()
        if df_wf.empty:
            raise ValueError(f"No data for {wf_fn}!")
        for peak_freq in good_peak_freqs:

            # --- Step 1: Coerce 'CF' column to numeric, forcing non‐parsable strings → NaN ---
            df_wf['CF_numeric'] = pd.to_numeric(df_wf['CF'], errors='coerce')

            # Drop any rows where CF_numeric is NaN, because we can't compare them to peak_freq
            df_valid_cf = df_wf.dropna(subset=['CF_numeric']).copy()

            # --- Step 2: Compute |CF_numeric – peak_freq| for each row ---
            diff_series = (df_valid_cf['CF_numeric'] - peak_freq).abs()

            # --- Step 3: Find the index (in df_valid_cf) of the closest CF to peak_freq ---
            closest_idx = diff_series.idxmin()
            min_diff = diff_series.loc[closest_idx]

            # --- Step 4: If that difference > 10, raise an error ---
            if min_diff > 32:
                raise ValueError(
                    f"No row found within 10 units of peak_freq={peak_freq:.2f}; "
                    f"closest difference={min_diff:.2f}"
                )

            # --- Step 5: Grab that “best” row and check SNRfit, BW, FWHM ---
            row = df_valid_cf.loc[closest_idx]
            required_cols = ['SNRfit', 'FWHM']

            # First, coerce those three to numeric as well (if you haven’t already) 
            # so that empty strings or weird text become NaN:
            for col in required_cols:
                df_valid_cf.loc[:, col + '_numeric'] = pd.to_numeric(df_valid_cf[col], errors='coerce')


            # Now re-grab the row from df_valid_cf (with the new _numeric columns)
            row = df_valid_cf.loc[closest_idx]

            # Check for NaN in any of the three numeric columns
            missing_mask = row[[c + '_numeric' for c in required_cols]].isna()
            if missing_mask.any():
                missing_cols = [c for c, is_missing in missing_mask.items() if is_missing]
                # Strip the "_numeric" suffix for reporting
                missing_cols = [c.replace('_numeric', '') for c in missing_cols]
                raise ValueError(
                    f"Row at index {closest_idx} is missing or non‐numeric in columns: {missing_cols}"
                )

            # (Optional) If you want to be extra sure they’re true Python/NumPy numerics:
            for col in required_cols:
                val = row[f"{col}_numeric"]
                if not isinstance(val, (int, float, np.floating, np.integer)):
                    raise TypeError(
                        f"Value in column '{col}' at index {closest_idx} is not numeric: {val!r}"
                    )



            # If you reach here, everything is OK:
            print(f"Checking peak you chose at {peak_freq}")
            # print(f"Closest row index: {closest_idx}")
            print(f"CF value in that row: {row['CF']} (diff = {min_diff:.2f})")
            # print(rf"{required_cols} are all present and numeric:")
            # print(row[required_cols])