"""Обучаемый ML-классификатор SALE / NOT_SALE.

Хранит датасет в JSONL (по одной строке — один пример), обучает
TF-IDF + логистическую регрессию и умеет отдавать модель для инференса.

Формат строки датасета:
    {"text": "...", "label": "SALE"|"NOT_SALE", "source": "ollama"|"admin", "ts": 1234567890.0}

При обучении примеры дедуплицируются по нормализованному тексту:
если один и тот же текст встречается с меткой от Ollama и меткой от
администратора — побеждает метка администратора (source="admin"),
так как это подтверждённая человеком разметка.
"""

import json
import os
import threading
import time
from dataclasses import dataclass

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _priority(record: dict) -> int:
    return 1 if record.get("source") == "admin" else 0


def load_dataset(path: str) -> list[dict]:
    """Читает JSONL и дедуплицирует записи по тексту (admin побеждает ollama)."""
    records: dict[str, dict] = {}
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "")
            key = normalize_text(text)
            if not key or record.get("label") not in ("SALE", "NOT_SALE"):
                continue

            existing = records.get(key)
            if existing is None or _priority(record) >= _priority(existing):
                records[key] = record

    return list(records.values())


def count_dataset(path: str) -> int:
    return len(load_dataset(path))


def append_example(path: str, text: str, label: str, source: str, ts: float | None = None) -> None:
    text = text.strip()
    if not text:
        return
    entry = {"text": text, "label": label, "source": source, "ts": ts if ts is not None else time.time()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class TrainResult:
    n_examples: int
    n_train: int
    n_test: int
    accuracy: float
    report: str


def train(dataset_path: str, model_path: str, meta_path: str) -> TrainResult:
    records = load_dataset(dataset_path)
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    if len(set(labels)) < 2:
        raise ValueError(
            "Нужны примеры обоих классов (SALE и NOT_SALE), сейчас датасет содержит только один класс "
            "или пуст. Дайте боту поработать в гибридном режиме или добавьте примеры через /correct."
        )

    n = len(texts)
    test_size = 0.2 if n >= 50 else max(1, round(n * 0.1))
    stratify = labels if n >= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=stratify
    )

    def make_pipeline() -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    eval_pipeline = make_pipeline()
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Финальная модель обучается на всём датасете (включая отложенную выборку),
    # чтобы не терять данные, которых и так немного.
    final_pipeline = make_pipeline()
    final_pipeline.fit(texts, labels)
    joblib.dump(final_pipeline, model_path)

    meta = {
        "n_examples": n,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy,
        "trained_at": time.time(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return TrainResult(n_examples=n, n_train=len(X_train), n_test=len(X_test), accuracy=accuracy, report=report)


class MLClassifier:
    """Потокобезопасная обёртка над обученным pipeline с горячей перезагрузкой."""

    def __init__(self, model_path: str, meta_path: str):
        self.model_path = model_path
        self.meta_path = meta_path
        self._lock = threading.Lock()
        self.pipeline = None
        self.meta: dict = {}
        self.reload()

    def reload(self) -> bool:
        with self._lock:
            if not os.path.exists(self.model_path):
                self.pipeline = None
                self.meta = {}
                return False

            self.pipeline = joblib.load(self.model_path)
            self.meta = {}
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            return True

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> tuple[str | None, float]:
        """Возвращает (метка, уверенность) или (None, 0.0), если модель не загружена."""
        pipeline = self.pipeline
        if pipeline is None:
            return None, 0.0

        proba = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        idx = proba.argmax()
        return str(classes[idx]), float(proba[idx])
