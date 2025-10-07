
"""Simple preprocessing script to be used by dvc pipeline.
Reads CSV, applies basic cleaning, and writes processed CSV."""
import argparse, os
import pandas as pd
from src.heart_pred.data_preprocessing import basic_cleaning

def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df = pd.read_csv(args.input_path)
    df_clean = basic_cleaning(df)
    df_clean.to_csv(args.output_path, index=False)
    print(f"Processed data written to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    main(args)
