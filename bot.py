import sys
from pathlib import Path

# эт проверка на наличие конфигов
if (
    not (Path(__file__).parent / ".env").is_file()
    or not (Path(__file__).parent / "config.py").is_file()
):
    print("Конфигурация не найдена. Запустите первичную настройку: .venv/bin/python start.py")
    sys.exit(1)

import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import BotCommand

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import logging
import os

from dotenv import load_dotenv

import short
from handlers import user
from main import setup_proxy_env

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN", "")
PROXY_URL = os.getenv("PROXY_URL", "")

# пробрасываем прокси в окружение: их читает huggingface_hub при скачивании моделей
setup_proxy_env()

if not API_TOKEN:
    print("API_TOKEN не найден в .env. Запустите .venv/bin/python start.py")
    sys.exit(1)


# меню команд
async def set_main_menu(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="info", description="Информация"),
        BotCommand(command="short", description="Пересказ"),
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
