"""Обучаемый ML-классификатор SALE / NOT_SALE.

Хранит датасет в JSONL (по одной строке — один пример), обучает
TF-IDF + логистическую регрессию и умеет отдавать модель для инференса.

Формат строки датасета:
    {"text": "...", "label": "SALE"|"NOT_SALE", "source": "admin"|"claude"|"ollama"|"history", "ts": 1234567890.0}

При обучении примеры дедуплицируются по нормализованному тексту. Если один и тот же
текст встречается с разными метками, побеждает более надёжный источник:
admin (ваша правка) > claude (ручная проверка ассистентом) > ollama (решение LLM) >
history (метка выведена из того, что сообщение уцелело в экспорте чата).
"""

import json
import os
import re
import threading
import time
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# Правило трёх: если на 100 SALE-примерах не найдено ни одного ложного удаления,
# с 95% уверенностью реальная доля ложных удалений не превышает ~3%.
# Меньшая выборка не позволяет честно оценить риск, поэтому порог удаления не выдаётся.
MIN_SALE_EXAMPLES = 100
MAX_FALSE_DELETION_RATE = 0.01
# Оценка по кросс-валидации оптимистична: в датасете много близких копий и «лёгких» объявлений.
# Поэтому удалять начинаем не ниже этого порога, даже если калибровка разрешает меньше.
MIN_DELETE_THRESHOLD = 0.9


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


_SOURCE_PRIORITY = {"admin": 3, "claude": 2, "ollama": 1}


def _priority(record: dict) -> int:
    return _SOURCE_PRIORITY.get(record.get("source"), 0)


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


def known_texts(path: str, trusted_only: bool = False) -> set[str]:
    """Нормализованные тексты, уже лежащие в датасете.

    trusted_only=True исключает записи из истории чата: их метка выведена косвенно,
    поэтому реальное решение Ollama по тому же тексту должно уметь её перекрыть.
    """
    return {
        normalize_text(r["text"])
        for r in load_dataset(path)
        if not trusted_only or _priority(r) >= 1
    }


def append_example(path: str, text: str, label: str, source: str, ts: float | None = None, **extra) -> None:
    text = text.strip()
    if not text:
        return
    entry = {"text": text, "label": label, "source": source, "ts": ts if ts is not None else time.time(), **extra}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def merge_records(path: str, raw: str) -> tuple[int, int, int]:
    """Дописывает в датасет записи из присланного JSONL.

    Строки с одним текстом сначала сводятся по приоритету источников (как в load_dataset), поэтому
    сырой dataset.jsonl с несколькими строками на текст переносится с итоговыми метками.
    Запись пропускается, если такой текст уже есть с таким же или более надёжным источником.

    Возвращает (добавлено, уже было в датасете, некорректных строк).
    """
    known = {normalize_text(r["text"]): _priority(r) for r in load_dataset(path)}
    incoming: dict[str, dict] = {}
    invalid = 0

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            text = record["text"].strip()
            label = record["label"]
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
            invalid += 1
            continue
        if not text or label not in ("SALE", "NOT_SALE"):
            invalid += 1
            continue

        source = record.get("source") if isinstance(record.get("source"), str) else "history"
        ts = record.get("ts")
        entry = {**record, "text": text, "label": label, "source": source,
                 "ts": ts if isinstance(ts, (int, float)) else time.time()}
        key = normalize_text(text)
        existing = incoming.get(key)
        if existing is None or _priority(entry) >= _priority(existing):
            incoming[key] = entry

    added = duplicates = 0
    with open(path, "a", encoding="utf-8") as out:
        for key, entry in incoming.items():
            if key in known and _priority(entry) <= known[key]:
                duplicates += 1
                continue
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            added += 1

    return added, duplicates, invalid


@dataclass
class TrainResult:
    n_examples: int
    n_train: int
    n_test: int
    accuracy: float
    report: str
    n_sale: int = 0
    delete_threshold: float | None = None
    delete_recall: float | None = None
    false_deletions: int | None = None
    note: str = ""


def calibrate_delete_threshold(texts, labels, make_pipeline, min_sale: int, max_false_rate: float) -> dict:
    """Подбирает порог P(NOT_SALE), выше которого сообщение можно удалять без Ollama.

    Порог считается по out-of-fold предсказаниям: самый низкий, при котором доля
    ложно удалённых настоящих объявлений (SALE) не превышает max_false_rate.
    """
    n_sale = labels.count("SALE")
    n_not_sale = labels.count("NOT_SALE")

    # Тексты, отличающиеся только цифрами (цены, телефоны), — почти копии: они всегда
    # попадают в одну часть кросс-валидации, иначе модель "видела" бы проверяемое при обучении.
    groups = np.array([re.sub(r"\d+", "0", normalize_text(t)) for t in texts])
    y = np.array(labels)
    folds = min(5, len(set(groups[y == "SALE"])), len(set(groups[y == "NOT_SALE"])))

    if n_sale < min_sale or folds < 2:
        return {"delete_threshold": None, "note": f"SALE-примеров {n_sale}, для калибровки нужно минимум {min_sale}"}

    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    proba = cross_val_predict(make_pipeline(), texts, y, cv=cv, groups=groups, method="predict_proba")
    p_not = proba[:, sorted(set(labels)).index("NOT_SALE")]

    is_sale = y == "SALE"
    allowed = int(max_false_rate * n_sale)
    sale_p = np.sort(p_not[is_sale])[::-1]
    threshold = max(float(np.nextafter(sale_p[allowed], 1.0)), MIN_DELETE_THRESHOLD)
    deleted = p_not >= threshold

    return {
        "delete_threshold": threshold,
        "false_deletions": int((deleted & is_sale).sum()),
        "delete_recall": float((deleted & ~is_sale).sum() / n_not_sale),
        "note": "",
    }


def make_pipeline() -> Pipeline:
    # Символьные n-граммы ловят опечатки и подмену букв в спаме, словесные — смысл фраз.
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=30000)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=10)),
    ])


def train(
    dataset_path: str,
    model_path: str,
    meta_path: str,
    min_sale: int = MIN_SALE_EXAMPLES,
    max_false_rate: float = MAX_FALSE_DELETION_RATE,
) -> TrainResult:
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

    calibration = calibrate_delete_threshold(texts, labels, make_pipeline, min_sale, max_false_rate)
    n_sale = labels.count("SALE")

    meta = {
        "n_examples": n,
        "n_sale": n_sale,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy,
        "trained_at": time.time(),
        **calibration,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return TrainResult(
        n_examples=n,
        n_train=len(X_train),
        n_test=len(X_test),
        accuracy=accuracy,
        report=report,
        n_sale=n_sale,
        delete_threshold=calibration["delete_threshold"],
        delete_recall=calibration.get("delete_recall"),
        false_deletions=calibration.get("false_deletions"),
        note=calibration["note"],
    )


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

    @property
    def delete_threshold(self) -> float | None:
        return self.meta.get("delete_threshold")

    def p_not_sale(self, text: str) -> float | None:
        pipeline = self.pipeline
        if pipeline is None:
            return None
        proba = pipeline.predict_proba([text])[0]
        return float(proba[list(pipeline.classes_).index("NOT_SALE")])
