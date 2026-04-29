import pandas as pd
import numpy as np
import os
import re

def add_nan_noise(df, noise_percent):
    df_noisy = df.copy()  # keep original safe

    total_cells = df.size
    num_nan_cells = int(total_cells * noise_percent / 100)

    rows = np.random.randint(0, df.shape[0], num_nan_cells)
    cols = np.random.randint(0, df.shape[1], num_nan_cells)

    for r, c in zip(rows, cols):
        col_name = df_noisy.columns[c]

        # ✅ Skip boolean columns to avoid dtype issues
        if df_noisy[col_name].dtype == bool:
            continue

        df_noisy.iat[r, c] = np.nan

    return df_noisy


def main():
    # Step 1: Input file
    file_path = input("Enter the CSV file path: ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found!")
        return

    # Step 2: Noise %
    try:
        noise_percent = float(input("Enter noise percentage (e.g., 5, 10, 15): "))
        if noise_percent < 0 or noise_percent > 100:
            print("❌ Enter a value between 0 and 100")
            return
    except:
        print("❌ Invalid input!")
        return

    # Step 3: Load data
    df = pd.read_csv(file_path)

    # Step 4: Add noise
    noisy_df = add_nan_noise(df, noise_percent)

    # Step 5: Create output folder
    output_folder = "noisy_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Step 6: Fix filename
    base = os.path.splitext(os.path.basename(file_path))[0]

    if "_sampled_" in base:
        # alarm_seed1_sampled_0 → alarm_seed1_sampled
        base = re.sub(r"_sampled_\d+$", "_sampled", base)
    else:
        # alarm_seed1 → alarm_seed1_sampled
        base = base + "_sampled"

    # Final output file
    output_file = os.path.join(
        output_folder,
        f"{base}_{int(noise_percent)}.csv"
    )

    # Step 7: Save
    noisy_df.to_csv(output_file, index=False)

    print(f"✅ File saved at: {output_file}")


if __name__ == "__main__":
    main()