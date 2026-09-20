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
    parser.add_argument("--min-sale", type=int, default=ml_classifier.MIN_SALE_EXAMPLES,
                        help="минимум SALE-примеров для калибровки порога удаления")
    args = parser.parse_args()

    result = ml_classifier.train(args.dataset, args.model, args.meta, min_sale=args.min_sale)

    print(f"Примеров в датасете: {result.n_examples} (SALE: {result.n_sale}; "
          f"train {result.n_train} / test {result.n_test})")
    print(f"Точность на отложенной выборке: {result.accuracy:.1%}")
    print()
    print(result.report)

    if result.delete_threshold is None:
        print(f"Калибровка НЕ пройдена: {result.note}.")
        print("В режиме USE_OLLAMA=false бот с такой моделью ничего не удаляет.")
    else:
        print(f"Порог удаления: P(NOT_SALE) >= {result.delete_threshold:.2f}")
        print(f"Кросс-валидация: ложно удалено SALE {result.false_deletions} из {result.n_sale}, "
              f"спама распознано {result.delete_recall:.0%}")

    print(f"Модель сохранена: {args.model}")


if __name__ == "__main__":
    main()
