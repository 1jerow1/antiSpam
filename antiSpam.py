import asyncio
import json
import os
import time
from datetime import date
from collections import OrderedDict

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command

import ml_classifier

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не обязателен, если переменные заданы в окружении

# ================== CONFIG ==================

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit("❌ Не задан BOT_TOKEN (см. .env.example)")

ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
if not ADMIN_ID:
    raise SystemExit("❌ Не задан ADMIN_ID (см. .env.example)")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

STATS_FILE = os.getenv("STATS_FILE", "stats.json")
WHITELIST_FILE = os.getenv("WHITELIST_FILE", "whitelist.json")

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))

# Гибридная ML-классификация: пока модель не обучена или не уверена в ответе,
# решение принимает Ollama (как раньше); каждое такое решение Ollama становится
# новым обучающим примером, так что датасет растёт сам по себе.
DATASET_FILE = os.getenv("DATASET_FILE", "dataset.jsonl")
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")
MODEL_META_FILE = os.getenv("MODEL_META_FILE", "model_meta.json")
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.85"))

# ================== IMPROVED PROMPT ==================

PROMPT = """Ты — модератор Telegram-барахолки.

Нужно определить: продаётся ли ТОВАР.

SALE — продаётся товар  
NOT_SALE — всё остальное  

====================
ТОВАР (SALE):
- физические вещи: техника, одежда, жидкости, вейпы, предметы
- есть конкретный объект продажи

====================
НЕ ТОВАР (NOT_SALE):
- работа, подработка, шабашка, халтура, смена, вахта, вакансия
- "нужен человек", "ищу человека", "есть дело", "требуется", "ищем сотрудника", "свободная смена"
- оплата за день/смену/час/выход (это оплата труда, а не цена товара)
- любые услуги
- крипта, обмен, валюты
- реклама
- ссылки

====================
ОЧЕНЬ ВАЖНО:
- любые предложения РАБОТЫ или ПОДРАБОТКИ → NOT_SALE, даже без слов "работа"/"вакансия" —
  если человека зовут что-то ДЕЛАТЬ (смена, выход, помощь, подработка) и платят ЗА ЭТО, а не за вещь → NOT_SALE
- даже если указана сумма (₽, $) → это НЕ товар, если нет конкретного продаваемого предмета
- если нет конкретного предмета → NOT_SALE

====================
Если сомневаешься → SALE

====================
Ответ:
только SALE или NOT_SALE
"""

# ================== INIT ==================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
session: aiohttp.ClientSession | None = None
ml = ml_classifier.MLClassifier(MODEL_FILE, MODEL_META_FILE)

# ================== STORAGE ==================

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

whitelist = set(map(int, load_json(WHITELIST_FILE, [])))
stats = load_json(STATS_FILE, {})

# ================== STATS ==================

stats_lock = asyncio.Lock()
stats_dirty = False

def get_today():
    return str(date.today())

async def update_stats(key: str):
    global stats_dirty
    async with stats_lock:
        today = get_today()
        day = stats.setdefault(today, {"checked": 0, "allowed": 0, "deleted": 0, "ml_used": 0, "ollama_used": 0})
        day[key] = day.get(key, 0) + 1
        stats_dirty = True

async def stats_saver():
    global stats_dirty
    while True:
        await asyncio.sleep(30)
        if stats_dirty:
            async with stats_lock:
                save_json(STATS_FILE, stats)
                stats_dirty = False
                print("💾 Статистика сохранена")

# ================== LRU CACHE ==================

class LRUCache:
    def __init__(self, max_size=500):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

ai_cache = LRUCache()

# ================== OLLAMA ==================

SEM = asyncio.Semaphore(3)

async def call_ollama(text: str) -> bool:
    text = text[:MAX_TEXT_LENGTH]
    key = " ".join(text.lower().split())

    cached = ai_cache.get(key)
    if cached is not None:
        print("🧠 CACHE HIT")
        return cached

    async with SEM:
        try:
            print("🤖 Запрос к Ollama...")

            async with session.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": PROMPT + "\n\nСообщение:\n" + text,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.1,
                        "num_predict": 15
                    }
                },
                timeout=60
            ) as resp:
                data = await resp.json()
                result = data.get("response", "").strip().upper()

                is_sale = result.startswith("SALE")
                ai_cache.set(key, is_sale)
                return is_sale

        except Exception as e:
            print(f"❌ Ошибка Ollama: {e}")
            return True  # fail-safe: лучше оставить, чем удалить

# ================== ГИБРИДНАЯ КЛАССИФИКАЦИЯ (ML + OLLAMA) ==================

async def classify(text: str) -> tuple[bool, str]:
    """Возвращает (is_sale, engine). engine — 'ml' или 'ollama'.

    Пока ML-модель не обучена или не уверена в ответе, решение принимает Ollama —
    как раньше. Каждое решение Ollama сохраняется как новый обучающий пример,
    поэтому датасет пополняется сам по себе по мере работы бота.
    """
    if ml.is_ready:
        label, confidence = await asyncio.to_thread(ml.predict, text)
        if label is not None and confidence >= ML_CONFIDENCE_THRESHOLD:
            return label == "SALE", "ml"

    is_sale = await call_ollama(text)
    return is_sale, "ollama"

# ================== ЛОГ РЕШЕНИЙ (для /correct и /pending) ==================

DECISION_LOG_MAX = 2000

decision_counter = 0
decision_log: "OrderedDict[int, dict]" = OrderedDict()

def log_decision(text: str, label: str, engine: str, chat_id: int, message_id: int, user_id: int) -> int:
    global decision_counter
    decision_counter += 1
    decision_log[decision_counter] = {
        "text": text,
        "label": label,
        "source": engine,
        "chat_id": chat_id,
        "message_id": message_id,
        "user_id": user_id,
    }
    if len(decision_log) > DECISION_LOG_MAX:
        decision_log.popitem(last=False)
    return decision_counter

ADMIN_CACHE_TTL = 300  # секунд

admin_cache: dict[int, tuple[set[int], float]] = {}

async def is_admin(message: Message) -> bool:
    chat_id = message.chat.id
    now = time.monotonic()

    cached = admin_cache.get(chat_id)
    if cached is None or now >= cached[1]:
        try:
            admins = await bot.get_chat_administrators(chat_id)
            admin_ids = {member.user.id for member in admins}
            cached = (admin_ids, now + ADMIN_CACHE_TTL)
            admin_cache[chat_id] = cached
        except Exception as e:
            print(f"❌ Ошибка получения списка админов: {e}")
            if cached is None:
                return False
            # используем протухший кеш, лучше так, чем дёргать API на каждое сообщение

    return message.from_user.id in cached[0]

# ================== ADMIN COMMANDS ==================
# Регистрируются раньше handle_message, иначе он перехватит команды
# как обычный текст и до Command()-хендлеров дело не дойдёт.

@dp.message(Command("whitelist_add"))
async def whitelist_add(message: Message):
    if message.from_user.id != ADMIN_ID:
        await message.reply("⛔ Нет доступа")
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("ℹ️ Нужно ответить на сообщение пользователя")
        return

    user_id = message.reply_to_message.from_user.id
    whitelist.add(user_id)
    save_json(WHITELIST_FILE, list(whitelist))

    await message.reply(f"✅ Добавлен в whitelist: {user_id}")


@dp.message(Command("stats"))
async def show_stats(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    today = get_today()
    today_stats = stats.get(today, {"checked": 0, "allowed": 0, "deleted": 0, "ml_used": 0, "ollama_used": 0})

    text = f"""📊 Статистика за сегодня ({today}):

Проверено сообщений: {today_stats['checked']}
Оставлено (SALE): {today_stats['allowed']}
Удалено (NOT_SALE): {today_stats['deleted']}

Решений через ML: {today_stats.get('ml_used', 0)}
Решений через Ollama: {today_stats.get('ollama_used', 0)}"""

    await message.reply(text)


@dp.message(Command("mlstats"))
async def show_ml_stats(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    n_examples = ml_classifier.count_dataset(DATASET_FILE)

    if ml.is_ready:
        meta = ml.meta
        trained_at = meta.get("trained_at")
        trained_at_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(trained_at)) if trained_at else "?"

        text = f"""🧠 ML-модель: активна
Обучена: {trained_at_str}
Примеров при обучении: {meta.get('n_examples', '?')}
Точность на отложенной выборке: {meta.get('accuracy', 0):.1%}
Порог уверенности: {ML_CONFIDENCE_THRESHOLD:.0%}
Примеров в датасете сейчас: {n_examples}

Переобучить: /train"""
    else:
        text = f"""🧠 ML-модель: ещё не обучена (все решения принимает Ollama)
Накоплено примеров в датасете: {n_examples}

Обучить: /train"""

    await message.reply(text)


@dp.message(Command("train"))
async def train_model_command(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    await message.reply("⏳ Обучение модели...")

    try:
        result = await asyncio.to_thread(ml_classifier.train, DATASET_FILE, MODEL_FILE, MODEL_META_FILE)
    except ValueError as e:
        await message.reply(f"⚠️ {e}")
        return
    except Exception as e:
        await message.reply(f"❌ Ошибка обучения: {e}")
        return

    ml.reload()

    await message.reply(
        f"✅ Модель обучена и загружена\n"
        f"Примеров: {result.n_examples} (train {result.n_train} / test {result.n_test})\n"
        f"Точность на отложенной выборке: {result.accuracy:.1%}\n\n"
        f"{result.report}"
    )


@dp.message(Command("pending"))
async def show_pending(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    entries = [(rid, rec) for rid, rec in reversed(decision_log.items()) if rec["label"] == "NOT_SALE"][:10]

    if not entries:
        await message.reply("Нет недавних удалений в логе решений")
        return

    lines = ["🗑 Последние удаления (используйте /correct <id> sale, если это ошибка):"]
    for rid, rec in entries:
        snippet = rec["text"][:80].replace("\n", " ")
        lines.append(f"#{rid} [{rec['source']}]: {snippet}")

    await message.reply("\n".join(lines))


@dp.message(Command("correct"))
async def correct_decision(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    args = (message.text or "").split(maxsplit=2)[1:]

    def parse_label(token: str) -> str | None:
        token = token.lower()
        if token in ("sale", "sell"):
            return "SALE"
        if token in ("spam", "not_sale"):
            return "NOT_SALE"
        return None

    usage = (
        "ℹ️ Использование:\n"
        "• Ответом на сообщение: /correct sale|spam\n"
        "• По ID из /pending: /correct <id> sale|spam"
    )

    if not args:
        await message.reply(usage)
        return

    # Вариант 1: ответом на сообщение, которое ещё есть в чате
    if message.reply_to_message and len(args) == 1:
        label = parse_label(args[0])
        if label is None:
            await message.reply(usage)
            return

        target = message.reply_to_message
        text = (target.text or target.caption or "").strip()
        if not text:
            await message.reply("⚠️ В этом сообщении нет текста")
            return

        ml_classifier.append_example(DATASET_FILE, text, label, "admin")

        if label == "NOT_SALE":
            try:
                await target.delete()
            except Exception as e:
                print(f"❌ Не удалось удалить по /correct: {e}")

        await message.reply(f"✅ Помечено как {label} и добавлено в датасет")
        return

    # Вариант 2: по ID из /pending (для уже удалённых сообщений)
    if len(args) == 2 and args[0].isdigit():
        decision_id = int(args[0])
        label = parse_label(args[1])
        record = decision_log.get(decision_id)

        if record is None:
            await message.reply("⚠️ Запись с таким ID не найдена (устарела или бот перезапускался)")
            return
        if label is None:
            await message.reply(usage)
            return

        ml_classifier.append_example(DATASET_FILE, record["text"], label, "admin")
        await message.reply(f"✅ Запись #{decision_id} переразмечена как {label} и добавлена в датасет")
        return

    await message.reply(usage)

# ================== HANDLER ==================

@dp.message(
    ~F.from_user.is_bot,
    F.text | F.caption
)
async def handle_message(message: Message):
    user_id = message.from_user.id

    if user_id in whitelist:
        return

    if await is_admin(message):
        return

    # 🆕 ЕСЛИ ЕСТЬ ФОТО — ПРОПУСКАЕМ
    if message.photo:
        print("🖼 Фото → пропускаем без проверки")
        return

    text = (message.text or message.caption or "").strip()
    if not text:
        return

    await update_stats("checked")

    is_sale, engine = await classify(text)
    label = "SALE" if is_sale else "NOT_SALE"
    await update_stats("ml_used" if engine == "ml" else "ollama_used")

    if engine == "ollama":
        ml_classifier.append_example(DATASET_FILE, text, label, "ollama")

    decision_id = log_decision(text, label, engine, message.chat.id, message.message_id, user_id)

    if is_sale:
        print(f"✅ SALE [{engine}] #{decision_id} → Оставляем | {text[:100]}...")
        await update_stats("allowed")
    else:
        print(f"🗑 NOT_SALE [{engine}] #{decision_id} → Удаляем | {text[:100]}...")
        await update_stats("deleted")
        try:
            await message.delete()
        except Exception as e:
            print(f"❌ Не удалось удалить: {e}")

# ================== MAIN ==================

async def main():
    global session
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=70))

    asyncio.create_task(stats_saver())

    print("🚀 Бот запущен...")

    try:
        await dp.start_polling(bot)
    finally:
        if session and not session.closed:
            await session.close()
        async with stats_lock:
            if stats_dirty:
                save_json(STATS_FILE, stats)
        print("👋 Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())