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
        if today not in stats:
            stats[today] = {"checked": 0, "allowed": 0, "deleted": 0}
        stats[today][key] += 1
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
    today_stats = stats.get(today, {"checked": 0, "allowed": 0, "deleted": 0})

    text = f"""📊 Статистика за сегодня ({today}):

Проверено сообщений: {today_stats['checked']}
Оставлено (SALE): {today_stats['allowed']}
Удалено (NOT_SALE): {today_stats['deleted']}"""

    await message.reply(text)

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

    is_sale = await call_ollama(text)

    if is_sale:
        print(f"✅ SALE → Оставляем | {text[:100]}...")
        await update_stats("allowed")
    else:
        print(f"🗑 NOT_SALE → Удаляем | {text[:100]}...")
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