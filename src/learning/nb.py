import argparse
import ast
import itertools
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)


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
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

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
