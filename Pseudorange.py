import pandas as pd
import numpy as np


# Constants

C = 299792458.0          
WEEKSEC = 604800.0       


# Pseudorange computation

def compute_pseudorange(row):
    """
    Compute raw and corrected pseudorange from Android GNSS raw measurements
    """

    try:
       
        # Receiver GPS time (ns)
       
        t_rx_nanos = (
            row["TimeNanos"]
            - row["FullBiasNanos"]
            - row.get("BiasNanos", 0.0)
        )

        # Convert to seconds
        t_rx_sec = t_rx_nanos * 1e-9

        # GPS time of week
        t_rx_tow = t_rx_sec % WEEKSEC

      
        # Satellite transmit time
       
        t_tx = (
            row["ReceivedSvTimeNanos"]
            + row.get("TimeOffsetNanos", 0.0)
        ) * 1e-9

       
        # Raw pseudorange
        
        rho_raw = (t_rx_tow - t_tx) * C

        
        # Satellite clock correction
       
        rho_corrected = rho_raw + row["SvClockBiasMeters"]

      
        # Sanity check
     
        if rho_corrected < 1e6 or rho_corrected > 1e8:
            return pd.Series([np.nan, np.nan, False])

        return pd.Series([rho_raw, rho_corrected, True])

    except Exception:
        return pd.Series([np.nan, np.nan, False])


# Main processing function

def process_excel(input_excel, output_excel):
    print(" Loading GNSS raw measurement Excel...")
    df = pd.read_excel(input_excel)

    print(f" Rows loaded: {len(df)}")

    # Optional: keep GPS only
    if "ConstellationType" in df.columns:
        df = df[df["ConstellationType"] == 1]  
        print(f" GPS-only rows: {len(df)}")

    # Compute pseudorange
    print(" Computing pseudorange...")
    df[["PseudorangeRaw_m", "PseudorangeCorrected_m", "ValidPR"]] = (
        df.apply(compute_pseudorange, axis=1)
    )

    # Drop invalid
    df_valid = df[df["ValidPR"] == True].copy()

    print(f" Valid pseudorange rows: {len(df_valid)}")

    # Export
    print(" Writing output Excel...")
    df_valid.to_excel(output_excel, index=False)

    print(" Done!")
    print(f"Output file saved as: {output_excel}")


# Run

if __name__ == "__main__":
    input_file = r"C:\Project\ไฟล์ข้อมูลดิบของมือถือแบบอยู่นิ่ง.xlsx"
    output_file = r"C:\project\gnss_pseudorange_sppแบบมือถือนิ่ง.xlsx"

    process_excel(input_file, output_file)
