
"""Train script for heart prediction project."""
import argparse
import os
from src.heart_pred.data_preprocessing import load_csv, basic_cleaning, split_features_target, train_val_test_split, scale_data
from src.heart_pred.model import get_model, train_model, evaluate_model
from src.heart_pred.utils import save_model_pickle

def main(args):
    df = load_csv(args.data)
    df = basic_cleaning(df)
    X, y = split_features_target(df, target=args.target)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, test_size=args.test_size, val_size=args.val_size)
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)
    model = get_model(args.model)
    model = train_model(model, X_train_s, y_train)
    eval_train = evaluate_model(model, X_train_s, y_train)
    eval_val = evaluate_model(model, X_val_s, y_val)
    print('Train eval:', eval_train)
    print('Val eval:', eval_val)
    os.makedirs(args.output_dir, exist_ok=True)
    save_model_pickle(model, os.path.join(args.output_dir, 'model.joblib'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV data')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--model', default='logistic', help='Model choice: logistic or rf')
    parser.add_argument('--output-dir', dest='output_dir', default='artifacts', help='Directory to save model')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
