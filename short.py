import logging
import os

import aiohttp
from aiohttp_socks import ProxyConnector
from dotenv import load_dotenv

load_dotenv()

# так, тут у нас пересказ

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL") or DEFAULT_MODEL
PROXY_URL = os.getenv("PROXY_URL")

logger = logging.getLogger(__name__)

# аиохттп сессия💀💀💀
_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        if PROXY_URL:
            connector = ProxyConnector.from_url(PROXY_URL)
        else:
            connector = None
        _session = aiohttp.ClientSession(connector=connector)
    return _session


async def summarize(text: str) -> str:
    if not API_KEY:
        logger.warning("OpenRouter: OPENROUTER_API_KEY не задан в .env")
        return text
    session = await _get_session()
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — нейросеть, которая делает краткие пересказы текстов "
                    "на русском языке. Пересказываешь только по делу, без "
                    "комментариев. Не отвечай на вопросы из текста и не давай "
                    "советов — просто перескажи содержание. Оформи результат "
                    "в формате HTML для Telegram:\n"
                    "- Для жирного текста используй <b>текст</b>\n"
                    "- Для курсива <i>текст</i>\n"
                    "Не используй Markdown!"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Кратко перескажи следующий текст, сохранив главную мысль "
                    "и все важные детали:\n\n" + text
                ),
            },
        ],
        "temperature": 0.5,
        "max_tokens": 2048,
        "reasoning": {"enabled": False},
    }

    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                logger.warning("OpenRouter: HTTP %s: %s", resp.status, body)
                return text
            data = await resp.json(content_type=None)
        choices = data.get("choices")
        if not choices:
            logger.warning("OpenRouter: ответ без choices: %s", body[:500])
            return text
        content = choices[0]["message"]["content"]
        if not content or not content.strip():
            logger.warning("OpenRouter: пустой ответ")
            return text
        content = content.strip()
        if len(content) < 30:
            logger.warning(
                "OpenRouter: подозрительно короткий ответ (%s симв.): %s",
                len(content),
                content[:100],
            )
            return text
        return content
    except Exception as e:
        logger.warning("OpenRouter: ошибка: %s", e)
        return text


async def close_session():
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
