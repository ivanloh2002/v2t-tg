import asyncio
import logging

from aiogram import F, Router, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from faster_whisper import WhisperModel

import config
import short
from main import process_audio, transcribe

logger = logging.getLogger(__name__)

# роутер для пользовательских сообщений
# и чтобы связать main.py c handlers.py
user = Router()


# вайтлист: берём ID из конфига и нормализуем в int
try:
    ALLOWED_IDS = {(x) for x in config.ALLOWED_USERS}
except (TypeError, ValueError):
    ALLOWED_IDS = set()
    logger.warning("Не смог распарсить config.ALLOWED_USERS, вайтлист пуст")

logger.info("Режим доступа: %s", config.ACCESS)
if config.ACCESS == "whitelist":
    if ALLOWED_IDS:
        logger.info("Разрешённые ID: %s", sorted(ALLOWED_IDS))
    else:
        logger.warning("ACCESS=whitelist, но список пуст/не распарсен — доступ закрыт ВСЕМ!")
else:
    if ALLOWED_IDS:
        logger.warning(
            "ACCESS=public, но ALLOWED_USERS заполнен — вайтлист фактически выключен, доступ открыт всем"
        )


# как раз таки проверка доступа
def _check_access(message: types.Message) -> bool:
    if config.ACCESS != "whitelist":
        return True
    #  защита от сообщений, у которых нет отправител
    return bool(message.from_user) and message.from_user.id in ALLOWED_IDS


# модель whisper-а грузится лениво, чтобы бот стартовал даже без скачанной модели
_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type="int8_float16" if config.WHISPER_DEVICE == "cuda" else "int8",
        )
    return _model


# расшифровка сериализована: две одновременно могут уронить VRAM
_transcribe_lock = asyncio.Lock()


async def _transcribe_with_lock(file_path: str) -> str:
    async with _transcribe_lock:
        return await asyncio.to_thread(_run_transcribe, file_path)


def _run_transcribe(file_path: str) -> str:
    text = transcribe(_get_model(), file_path)
    if config.USE_QWEN:
        try:
            text = process_audio(text)
        except Exception as e:
            logger.warning("Qwen пост-обработка не удалась, отправлен сырой текст: %s", e)
    return text


_last_text: dict[int, str] = {}
_MAX_LAST = 50


def _store_last(chat_id: int, text: str) -> None:
    _last_text[chat_id] = text
    while len(_last_text) > _MAX_LAST:
        _last_text.pop(next(iter(_last_text)))


async def safe_delete(msg):
    try:
        await msg.delete()
    except Exception:
        pass


# уже телеграммное деление на чанки, т.к если длина ответа бота превысит 4096 символов,
# бот упадёт
def split_text(text: str, max_len: int = 4096) -> list[str]:
    chunks = []
    lines = text.splitlines(keepends=True)
    current = ""
    for line in lines:
        while len(line) > max_len:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:max_len])
            line = line[max_len:]
        if len(current) + len(line) > max_len:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [""]


# отправляет по чанкам
async def send_result(message: types.Message, text: str):
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        prefix = "Результат:\n" if i == 0 else ""
        await message.answer(prefix + chunk)


# отправляет соо с хтмл чтобы было красиво
async def send_html(message: types.Message, text: str):
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        prefix = "Пересказ сообщения:\n" if i == 0 else ""
        try:
            await message.answer(prefix + chunk, parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            await message.answer(prefix + chunk)


# отправляет сообщение об отказе доступа
async def _denied(message: types.Message):
    user = message.from_user
    uid = user.id if user else None
    uname = user.username if user else None
    logger.warning("Доступ закрыт: id=%s (username=%s), chat=%s", uid, uname, message.chat.id)
    await message.answer("Доступ закрыт.\nСвяжитесь с человеком, который дал вам сслыку на бот ")


@user.message(CommandStart())
async def start(message: types.Message):
    await message.answer(
        "Отправьте сюда видео/голосовое сообщение для расшифровки.\n"
        " Ваши сообщения не сохраняются, так что не бойтесь. "
        "Сразу после расшифровки они удаляются.\n"
        "Время распознавания зависит от длины видео/голосового сообщения."
    )


# обработка гс и кружочков
@user.message(F.voice | F.video_note)
async def media(message: types.Message):
    if not _check_access(message):
        await _denied(message)
        return

    media_file = message.voice or message.video_note
    if media_file is None:
        return

    if message.voice:
        file_path = f"{message.voice.file_unique_id}.ogg"
        waiting = "⏰Сообщение получено, идёт расшифровка..."
    else:
        file_path = f"{media_file.file_unique_id}.mp4"
        waiting = "⏰Видеосообщение получено, идёт расшифровка..."

    bot = message.bot
    if bot is None:
        return

    await bot.download(
        file=media_file,
        destination=file_path,
    )
    msg = await message.answer(waiting)

    try:
        text = await _transcribe_with_lock(file_path)
        _store_last(message.chat.id, text)
        await safe_delete(msg)
        await send_result(message, text)
    except Exception as e:
        await safe_delete(msg)
        await message.answer(f"Ошибка при расшифровке: {e}")


# пересказ
@user.message(Command("short"))
async def cmd_short(message: types.Message):
    if not _check_access(message):
        await _denied(message)
        return
    text = _last_text.get(message.chat.id)
    if not text:
        await message.answer("Сначала отправьте голосовое или видеосообщение для расшифровки.")
        return
    msg = await message.answer("⏰Готовлю краткий пересказ...")
    try:
        summary = await short.summarize(text)
        await safe_delete(msg)
        await send_html(message, summary)
    except Exception as e:
        await safe_delete(msg)
        await message.answer(f"Ошибка при пересказе: {e}")


# неудаляёте пжпжпжп😭😭😭😭
@user.message(Command("info"))
async def info(message: types.Message):
    await message.answer(
        "Информация о боте:\nтут используется либа whisper от openai\
        \nТакже используется локальная моедль unsloth/Qwen3.5-4B-GGUF для исправления текста\nдля пересказа используется nvidia/nemotron-3-super-120b-a12b:free по API через OpenRouter\nВ сумме всё это занимает менее 8 гигабайт VRAM\nБот тестировался на ноутбучной RTX3070ti \nУ меня получалось что-то окло 44 секнуд на расшифровку 7-ми минутного гс, на модели medium\nПримичательно, что расшифровщик который в тг прем не смог справиться с этой задачей\n\nМой тг, обращаться по вопросам: @larp13337\nМой гитхаб:https://github.com/ivanloh2002/\nПожайлуста, если берёте моего бота с гх не удаляйте этот текст, будьте людми\n\nСпасибо что используете моего бота!!!!"
    )
