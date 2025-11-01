import argparse
import ast
import itertools
import os
import random

import numpy as np
import tensorflow as tf

random.seed(42)

np.random.seed(42)

tf.random.set_seed(42)

os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def process_directory(train_path, test_path):
    train_data = {'con': [], 'non': []}
    test_data = {'con': [], 'non': []}

    with open(train_path, "r") as f:
        for t in f.readlines():
            label, feature_str = t.strip().split(':')
            ff = ast.literal_eval(feature_str)

            patterns = {
                'kw': ff[0:23],
                'ph': ff[23:35],
                'se': ff[35:52],
                'br': ff[52:]
            }

            train_data[label].append(patterns)

    with open(test_path, "r") as f:
        for t in f.readlines():
            label, feature_str = t.strip().split(':')
            ff = ast.literal_eval(feature_str)

            patterns = {
                'kw': ff[0:23],
                'ph': ff[23:35],
                'se': ff[35:52],
                'br': ff[52:]
            }

            test_data[label].append(patterns)

    all_combinations = []
    pattern_keys = ['kw', 'ph', 'se', 'br']

    for r in range(1, len(pattern_keys) + 1):
        all_combinations.extend(itertools.combinations(pattern_keys, r))

    all_results = {}

    for comb in all_combinations:
        comb_name = '+'.join(comb)
        train_vec_con, test_vec_con = [], []
        train_vec_non, test_vec_non = [], []

        for pattern_dict in train_data['con']:
            vec = sum([pattern_dict[key] for key in comb], [])
            train_vec_con.append(vec)

        for pattern_dict in train_data['non']:
            vec = sum([pattern_dict[key] for key in comb], [])
            train_vec_non.append(vec)

        for pattern_dict in test_data['con']:
            vec = sum([pattern_dict[key] for key in comb], [])
            test_vec_con.append(vec)

        for pattern_dict in test_data['non']:
            vec = sum([pattern_dict[key] for key in comb], [])
            test_vec_non.append(vec)

        all_results[comb_name] = (np.array(train_vec_con + train_vec_non),
                                  np.array(test_vec_con + test_vec_non),
                                  np.array([1] * len(train_vec_con) + [0] * len(train_vec_non)),
                                  np.array([1] * len(test_vec_con) + [0] * len(test_vec_non)))

    return all_results


def train_and_evaluate(X_train, X_test, y_train, y_test):
    X_train = X_train.astype(np.float32) / np.max(X_train)
    X_test = X_test.astype(np.float32) / np.max(X_test)

    input_shape = X_train.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=1000, output_dim=128, input_length=input_shape),
        tf.keras.layers.LSTM(units=128, return_sequences=False),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    history = model.fit(X_train, y_train, epochs=3, batch_size=4, validation_data=(X_test, y_test), verbose=0)

    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate linguistic pattern combinations for concurrency bug classification.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training data file (e.g., training.txt)")
    parser.add_argument("--test", type=str, required=True, help="Path to the testing data file (e.g., github.txt)")
    args = parser.parse_args()

    train_path = args.train
    test_path = args.test
    results = process_directory(train_path, test_path)

    print("\n=== Pattern Combination Evaluation ===\n")

    for pattern_comb, (X_train, X_test, y_train, y_test) in results.items():
        accuracy, precision, recall, f1 = train_and_evaluate(X_train, X_test, y_train, y_test)

        print(f"Pattern Combination: {pattern_comb}")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print("-" * 40)


if __name__ == '__main__':
    main()
