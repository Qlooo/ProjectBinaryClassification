
"""Evaluate a saved model on a dataset."""
import argparse
import os
from src.heart_pred.data_preprocessing import load_csv, basic_cleaning, split_features_target, train_val_test_split, scale_data
from src.heart_pred.utils import load_model_pickle
from src.heart_pred.model import evaluate_model

def main(args):
    df = load_csv(args.data)
    df = basic_cleaning(df)
    X, y = split_features_target(df, target=args.target)
    _, _, X_test, _, _, y_test = train_val_test_split(X, y, test_size=args.test_size, val_size=args.val_size)
    X_train, X_val, X_test_s, _ = scale_data(X, X, X_test)
    model = load_model_pickle(args.model_path)
    eval_test = evaluate_model(model, X_test_s, y_test)
    print('Test eval:', eval_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV data')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--model-path', required=True, help='Path to saved model joblib')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
