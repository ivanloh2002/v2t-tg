import sys
from pathlib import Path

# эт проверка на наличие конфигов
if not (Path(__file__).parent / ".env").is_file() or not (Path(__file__).parent / "config.py").is_file():
    print("Конфигурация не найдена. Запустите первичную настройку: .venv/bin/python start.py")
    sys.exit(1)

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from aiogram.client.session.aiohttp import AiohttpSession

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from handlers import user
import short

from dotenv import load_dotenv
import os
import logging



load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
PROXY_URL = os.getenv("PROXY_URL")

if not API_TOKEN:
    print("API_TOKEN не найден в .env. Запустите .venv/bin/python start.py")
    sys.exit(1)

# меню команд
async def set_main_menu(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="info", description="Информация"),
        BotCommand(command="short", description="Пересказ")
    ]
    await bot.set_my_commands(commands)

# ну тут ясно
async def main():
    session = AiohttpSession(proxy=PROXY_URL)
    bot = Bot(token=API_TOKEN, session=session)
    dp = Dispatcher()
    dp.include_router(user)
    dp.shutdown.register(short.close_session)
    await set_main_menu(bot)
    await dp.start_polling(bot)

# защита от неочень умных
if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
