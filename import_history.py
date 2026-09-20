"""Импорт SALE-примеров из экспорта истории чата Telegram Desktop.

Сообщения, которые бот удалил, в экспорте отсутствуют, поэтому там остаются в основном
нормальные объявления. Скрипт берёт их как SALE-примеры с источником "history"
(самый низкий приоритет: метки Ollama и админа при обучении его перекрывают).

Экспорт: Telegram Desktop -> меню чата -> Export chat history -> формат JSON.
Нужен экспорт ОДНОГО чата (result.json с ключом "messages"), а не всего аккаунта.

Использование:
    python import_history.py result.json --dry-run
    python import_history.py result.json --since 2026-04-11 --exclude-user 123456789
"""

import argparse
import collections
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime

import ml_classifier

MIN_LENGTH = 20


def message_text(msg: dict) -> str:
    text = msg.get("text", "")
    if isinstance(text, list):
        text = "".join(part if isinstance(part, str) else part.get("text", "") for part in text)
    return text.strip()


def user_digits(value) -> str:
    return re.sub(r"\D", "", str(value or ""))


def message_ts(msg: dict) -> float | None:
    try:
        return datetime.fromisoformat(msg["date"]).timestamp()
    except (KeyError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="путь к result.json экспорта чата")
    parser.add_argument("--dataset", default="dataset.jsonl", help="путь к датасету")
    parser.add_argument("--since", help="брать сообщения с даты YYYY-MM-DD (например, с момента появления бота в чате)")
    parser.add_argument("--exclude-user", action="append", default=[],
                        help="user id, чьи сообщения пропустить (админы, whitelist); можно указать несколько раз")
    parser.add_argument("--include-photos", action="store_true",
                        help="брать и сообщения с фото (бот их не проверяет, поэтому по умолчанию они пропускаются)")
    parser.add_argument("--max", type=int, help="взять не больше N случайных примеров")
    parser.add_argument("--dry-run", action="store_true", help="только посчитать, ничего не записывать")
    args = parser.parse_args()

    with open(args.export, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("messages"), list):
        sys.exit("В файле нет ключа 'messages'. Нужен экспорт одного чата, а не всего аккаунта.")

    excluded = {user_digits(u) for u in args.exclude_user} - {""}
    existing = ml_classifier.known_texts(args.dataset)
    skipped: collections.Counter = collections.Counter()
    candidates: dict[str, tuple[str, float | None]] = {}

    for msg in data["messages"]:
        if msg.get("type") != "message":
            skipped["служебные сообщения"] += 1
            continue

        text = message_text(msg)
        if not text:
            skipped["без текста"] += 1
        elif args.since and str(msg.get("date", ""))[:10] < args.since:
            skipped["раньше --since"] += 1
        elif msg.get("reply_to_message_id"):
            skipped["ответы на чужие сообщения"] += 1
        elif user_digits(msg.get("from_id")) in excluded:
            skipped["от исключённых пользователей"] += 1
        elif "photo" in msg and not args.include_photos:
            skipped["с фото (бот их не проверял)"] += 1
        elif len(text) < MIN_LENGTH:
            skipped[f"короче {MIN_LENGTH} символов"] += 1
        else:
            key = ml_classifier.normalize_text(text)
            if key in existing:
                skipped["уже есть в датасете"] += 1
            elif key in candidates:
                skipped["дубли внутри экспорта"] += 1
            else:
                candidates[key] = (text, message_ts(msg))

    items = list(candidates.values())
    if args.max and len(items) > args.max:
        items = random.Random(42).sample(items, args.max)

    print(f"Сообщений в экспорте: {len(data['messages'])}")
    for reason, n in skipped.most_common():
        print(f"  пропущено — {reason}: {n}")
    print(f"К импорту как SALE: {len(items)}")

    if args.dry_run:
        print("--dry-run: ничего не записано")
        return

    if os.path.exists(args.dataset):
        shutil.copy2(args.dataset, args.dataset + ".bak")

    for text, ts in items:
        ml_classifier.append_example(args.dataset, text, "SALE", "history", ts)

    print(f"Записано в {args.dataset} (резервная копия: {args.dataset}.bak)")


if __name__ == "__main__":
    main()
