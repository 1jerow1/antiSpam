"""Вторая проверка SALE-меток более сильной LLM через Ollama.

Метки «уцелело в чате = SALE» и метки слабой модели содержат много пропущенного спама:
подработки, «есть тема», курсы, посредники, приветствия. Скрипт спрашивает вторую модель,
что это за сообщение: SELL (продают), BUY (ищут вещь) или OTHER (всё остальное). OTHER и
крипта переводятся в NOT_SALE, подтверждённые SALE помечаются проверенными и повторно
не спрашиваются. Покупка остаётся объявлением — как и раньше в работе бота.

Использование:
    python verify_labels.py --dry-run
    python verify_labels.py --model qwen2.5:14b-instruct --workers 2
"""

import argparse
import asyncio
import collections
import os
import re
import time

import aiohttp

import ml_classifier

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROMPT = """Ты — модератор Telegram-барахолки. Определи тип сообщения. Ответь одним словом.

SELL — предлагают продать, отдать или обменять конкретную вещь (товар).
BUY — хотят купить или найти конкретную вещь.
OTHER — всё остальное:
- работа, подработка, шабашка, разгрузка, смена, «нужны деньги», «заработок», «есть тема», вакансия
- услуги, курсы и обучение, доставка и выкуп товаров из Китая (посредники), ремонт
- крипта, обмен валют, займы, финансовая помощь
- реклама, ссылки, приглашения в чаты
- приветствия, правила чата, вопросы, разговоры, бессмысленный текст

Если человека зовут что-то ДЕЛАТЬ за деньги — это OTHER, даже без слов «работа» или «вакансия».
"""

# крипта и валюта — NOT_SALE по правилам бота, даже если модель сочла это «покупкой вещи»
CRYPTO = re.compile(
    r"(?i)usdt|юздт|юсдт|крипт|bitcoin|биткоин|\bbtc\b|ethereum|\beth\b|обмен\w* валют|\bнфт\b|\bnft\b|telegram stars"
)


async def classify_kind(session, url: str, model: str, text: str, max_len: int) -> str | None:
    payload = {
        "model": model,
        "prompt": PROMPT + "\nСообщение:\n" + text[:max_len],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 6},
    }
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        raw = (await resp.json()).get("response", "").strip().upper()
    for kind in ("SELL", "BUY", "OTHER"):
        if raw.startswith(kind):
            return kind
    return None


async def run(records: list[dict], url: str, model: str, workers: int, max_len: int) -> list[tuple[dict, str | None]]:
    sem = asyncio.Semaphore(workers)
    done = 0

    async with aiohttp.ClientSession() as session:
        async def one(record: dict) -> tuple[dict, str | None]:
            nonlocal done
            async with sem:
                try:
                    kind = await classify_kind(session, url, model, record["text"], max_len)
                except Exception:
                    kind = None
            done += 1
            if done % 25 == 0 or done == len(records):
                print(f"  проверено {done}/{len(records)}", flush=True)
            return record, kind

        return await asyncio.gather(*(one(r) for r in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset.jsonl", help="путь к датасету")
    parser.add_argument("--model", default="qwen2.5:14b-instruct", help="модель Ollama для второй проверки")
    parser.add_argument("--url", default=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--workers", type=int, default=2, help="параллельных запросов к Ollama")
    parser.add_argument("--max-len", type=int, default=500, help="до скольких символов обрезать текст")
    parser.add_argument("--dry-run", action="store_true", help="только показать итог, ничего не записывать")
    args = parser.parse_args()

    todo = [r for r in ml_classifier.load_dataset(args.dataset) if r["label"] == "SALE" and "verified_by" not in r]
    print(f"SALE-примеров на проверку: {len(todo)} (модель {args.model})")
    if not todo:
        return

    results = asyncio.run(run(todo, args.url, args.model, args.workers, args.max_len))

    stat: collections.Counter = collections.Counter()
    lines = []
    for record, kind in results:
        if kind is None:
            stat["нет ответа (пропущено)"] += 1
            continue
        label = "NOT_SALE" if kind == "OTHER" or CRYPTO.search(record["text"]) else "SALE"
        stat[f"{'подтверждено SALE' if label == 'SALE' else 'SALE → NOT_SALE'}"] += 1
        lines.append((record["text"], label, kind))

    for name, count in sorted(stat.items()):
        print(f"  {name}: {count}")

    if args.dry_run:
        print("--dry-run: ничего не записано")
        return

    now = time.time()
    for text, label, kind in lines:
        ml_classifier.append_example(args.dataset, text, label, "ollama", now, verified_by=args.model, kind=kind)
    print(f"Записано в {args.dataset}: {len(lines)} записей")


if __name__ == "__main__":
    main()
