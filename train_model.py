"""CLI для (пере)обучения ML-классификатора SALE / NOT_SALE на dataset.jsonl.

Использование:
    python train_model.py
    python train_model.py --dataset dataset.jsonl --model model.joblib
"""

import argparse

import ml_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset.jsonl", help="путь к JSONL-датасету")
    parser.add_argument("--model", default="model.joblib", help="куда сохранить обученную модель")
    parser.add_argument("--meta", default="model_meta.json", help="куда сохранить метаданные обучения")
    args = parser.parse_args()

    result = ml_classifier.train(args.dataset, args.model, args.meta)

    print(f"Примеров в датасете: {result.n_examples} (train {result.n_train} / test {result.n_test})")
    print(f"Точность на отложенной выборке: {result.accuracy:.1%}")
    print()
    print(result.report)
    print(f"Модель сохранена: {args.model}")


if __name__ == "__main__":
    main()
