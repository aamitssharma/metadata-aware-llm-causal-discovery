import argparse
import os
import pandas as pd


def generate_sample(input_file, output_dir, sample_size=500):
    # Load dataset
    df = pd.read_csv(input_file)

    # Sample data
    sampled_df = df.sample(n=sample_size, random_state=42)

    # Extract base filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Generate output filename
    output_file = f"{base_name}_sampled_0.csv"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Full output path
    output_path = os.path.join(output_dir, output_file)

    # Save sampled data
    sampled_df.to_csv(output_path, index=False)

    print(f"✅ Sampled file saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataset and create new CSV")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save sampled file"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of samples (default: 500)"
    )

    args = parser.parse_args()

    generate_sample(args.input, args.output_dir, args.n)