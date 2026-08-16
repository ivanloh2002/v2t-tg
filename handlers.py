import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from aiogram import F, Router, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from faster_whisper import WhisperModel

import config
import short
from main import _llm_backend, process_audio, process_audio_api, transcribe

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


# очередь: гс обрабатываются строго по одному, в порядке отправки,
# чтобы две расшифровки одновременно не уронили VRAM


@dataclass
class Job:
    file_path: str
    message: types.Message
    waiting_msg: types.Message
    queued: bool = False


queue: asyncio.Queue[Job] = asyncio.Queue()
worker: asyncio.Task | None = None


def _transcribe_only(file_path: str) -> str:
    return transcribe(_get_model(), file_path)


async def _post_process(text: str) -> str:
    backend = _llm_backend()
    try:
        if backend == "api":
            return await process_audio_api(text)
        if backend == "qwen":
            return await asyncio.to_thread(process_audio, text)
    except Exception as e:
        logger.warning("Пост-обработка не удалась, отправлен сырой текст: %s", e)
    return text


async def safe_edit(msg, text):
    try:
        await msg.edit_text(text)
    except Exception:
        pass


async def worker_run():
    while True:
        job = await queue.get()
        job_start = time.perf_counter()
        try:
            if job.queued:
                await safe_edit(job.waiting_msg, "⏰Идёт расшифровка...")
            raw = await asyncio.to_thread(_transcribe_only, job.file_path)
            stream = getattr(config, "STREAM_RESULT", False) and _llm_backend() != "off"
            if stream:
                # стрим: сразу отправляем сырой текст от whisper,
                # потом заменяем его на обработанный текст в тех же сообщениях
                await safe_delete(job.waiting_msg)
                sent = await send_result(job.message, raw)
                text = await _post_process(raw)
                _store_last(job.message.chat.id, text)
                await _update_result(job.message, sent, text)
            else:
                text = await _post_process(raw)
                _store_last(job.message.chat.id, text)
                await safe_delete(job.waiting_msg)
                await send_result(job.message, text)
            logger.info(
                "Расшифровка заняла %.2f сек (%s)",
                time.perf_counter() - job_start,
                job.file_path,
            )
        except Exception as e:
            await safe_delete(job.waiting_msg)
            await job.message.answer(f"Ошибка при расшифровке: {e}")
        finally:
            queue.task_done()


def ensure_worker():
    global worker
    if worker is None or worker.done():
        worker = asyncio.create_task(worker_run())


async def enqueue(job: Job):
    ensure_worker()
    await queue.put(job)


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


# отправляет по чанкам; возвращает отправленные сообщения, чтобы потом
# заменить их финальным текстом в стрим-режиме
async def send_result(message: types.Message, text: str) -> list[types.Message]:
    chunks = split_text(text)
    sent = []
    for i, chunk in enumerate(chunks):
        prefix = "Результат:\n" if i == 0 else ""
        sent.append(await message.answer(prefix + chunk))
    return sent


# заменяет уже отправленные сообщения финальным текстом; если текст стал
# длиннее — досылает, короче — удаляет лишние
async def _update_result(message: types.Message, sent: list[types.Message], text: str) -> None:
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        prefix = "Результат:\n" if i == 0 else ""
        if i < len(sent):
            await safe_edit(sent[i], prefix + chunk)
        else:
            sent.append(await message.answer(prefix + chunk))
    for extra in sent[len(chunks):]:
        await safe_delete(extra)


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
        "Отправьте сюда видео/голосовое сообщение, аудио или видео (можно даже файлом) для расшифровки.\n"
        "Ваши сообщения не сохраняются, так что не бойтесь.\n"
        "Сразу после расшифровки они удаляются.\n"
        "Время распознавания зависит от модели, мощности видокарты на хосте и длины видео/голосового сообщения."
    )


# обработка гс и кружочков
# видео-форматы, которые поддерживает транскрибация (PyAV читает любой контейнер ffmpeg)
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".wmv", ".webm", ".flv", ".3gp"}
AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".wav"}


@user.message(F.voice | F.video_note | F.video | F.document | F.audio)
async def media(message: types.Message):
    if not _check_access(message):
        await _denied(message)
        return

    media_file = message.voice or message.video_note or message.video or message.document or message.audio
    if media_file is None:
        return

    if message.voice:
        file_path = f"{message.voice.file_unique_id}.ogg"
        processing = "⏰Голосовое получено, идёт расшифровка..."
    elif message.video_note:
        file_path = f"{media_file.file_unique_id}.mp4"
        processing = "⏰Видеосообщение получено, идёт расшифровка..."
    elif message.audio:
        name = getattr(media_file, "file_name", "") or ""
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTS:
            mime = getattr(media_file, "mime_type", "") or ""
            if mime.startswith("audio/"):
                ext = ".mp3"
            else:
                await message.answer(
                    "Отправьте аудио в правильном расширении (mp3, m4a, aac, wav, flac) или голосовое сообщение."
                )
                return
        file_path = f"{media_file.file_unique_id}{ext}"
        processing = "⏰Аудио получено, идёт расшифровка..."

    else:
        name = getattr(media_file, "file_name", "") or ""
        ext = Path(name).suffix.lower()
        mime = getattr(media_file, "mime_type", "") or ""
        if ext in VIDEO_EXTS or mime.startswith("video/"):
            if ext not in VIDEO_EXTS:
                ext = ".mp4"
            file_path = f"{media_file.file_unique_id}{ext}"
            processing = "⏰Видео получено, идёт расшифровка..."
        elif ext in AUDIO_EXTS or mime.startswith("audio/"):
            if ext not in AUDIO_EXTS:
                ext = ".mp3"
            file_path = f"{media_file.file_unique_id}{ext}"
            processing = "⏰Аудио получено, идёт расшифровка..."
        else:
            await message.answer(
                "Отправьте видео (mp4, avi, mkv, wmv, webm, flv, 3gp), аудио (mp3, m4a, aac, wav, flac) или голосовое сообщение."
            )
            return


    bot = message.bot
    if bot is None:
        return

    await bot.download(
        file=media_file,
        destination=file_path,
    )
    position = queue.qsize() + 1
    if position == 1:
        waiting = processing
    else:
        waiting = f"⏳Вы в очереди (позиция {position}). Идёт расшифровка предыдущих сообщений..."
    waiting_msg = await message.answer(waiting)
    await enqueue(Job(file_path=file_path, message=message, waiting_msg=waiting_msg, queued=position > 1))


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
        \nТакже используется локальная моедль unsloth/Qwen3.5-4B-GGUF для исправления текста\nдля пересказа используется nvidia/nemotron-3-super-120b-a12b:free по API через OpenRouter\nВ сумме всё это занимает 4610МБ VRAM\nБот тестировался на ноутбучной RTX3070ti \nУ меня получалось что-то окло 22 секнуд на расшифровку 7-ми минутного гс, на модели large-v3-turbo\nПримичательно, что расшифровщик который в тг прем не смог справиться с этой задачей\n\nМой тг, обращаться по вопросам: @larp13337\nМой гитхаб:https://github.com/ivanloh2002/v2t-tg\nПожайлуста, если берёте моего бота с гх не удаляйте этот текст, будьте людми\n\nСпасибо что используете моего бота!!!!"
    )
