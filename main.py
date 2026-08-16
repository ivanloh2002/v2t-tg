import logging
import os
import time
from pathlib import Path
from typing import cast

import aiohttp
from aiohttp_socks import ProxyConnector
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from llama_cpp import CreateChatCompletionResponse, Llama

import config

load_dotenv()
# некоторые говнососы говорят, что токен от хф необязателен, но это ложь. У меня банально не запустился бот
# так что придётся регаться на хг ахахахаххахаха
# этот токен нужен для whisper т.к я использую whisper-faster а не обычный
# т.к обычный нельзя квантовать
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))

# проброс PROXY_URL в env для huggingface_hub: httpx внутри него читает эти
# переменные при скачивании моделей
def setup_proxy_env() -> None:
    proxy = os.getenv("PROXY_URL")
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("ALL_PROXY", proxy)
        logger.info("Скачивание моделей с HuggingFace идёт через прокси: %s", proxy)
    else:
        logger.info("PROXY_URL не задан — модели качаются напрямую")

# ищет телеграмовские .ogg и .mp4 файлы
# .ogg - это гс
script_dir = Path(__file__).parent
target_extensions = {".ogg", ".mp4", ".mp3", ".flac", ".mov", ".avi", ".mkv", ".wmv", ".webm",".flv", ".3gp", ".m4a" , ".aac", ".wav"}

logger = logging.getLogger(__name__)

# чанки
CHUNK_SIZE = 3000

_llm = None


# совместимость со старым config.py (USE_QWEN без LLM_BACKEND)
def _llm_backend() -> str:
    if hasattr(config, "LLM_BACKEND"):
        return config.LLM_BACKEND
    return "qwen" if getattr(config, "USE_QWEN", False) else "off"


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    try:
        # качаем только нужный файл, а не весь снапшот репы (там 20+ квантований,
        # суммарно ~70 ГБ — snapshot_download без allow_patterns скачал бы всё)
        # имя файла и репа задаются в config.py (QWEN_REPO / QWEN_MODEL)
        _gguf_path = hf_hub_download(
            getattr(config, "QWEN_REPO", "unsloth/Qwen3.5-4B-GGUF"),
            filename=getattr(config, "QWEN_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
        )
        def is_flash_attn():
            return config.WHISPER_DEVICE == 'cuda'
        _llm = Llama(
            model_path=_gguf_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            flash_attn=is_flash_attn(),
            n_batch=1024,
            verbose=False
        )
    except Exception as e:
        # если модель не скачана/не грузится — не роняем всю расшифровку
        logger.warning("Не удалось загрузить Qwen: %s", e)
        _llm = False
    return _llm

# разбивает текст на чанки, сохраняя абзацы
def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    chunks = []
    current = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    for para in text.splitlines():
        words = para.split()
        if not words:
            flush()
            continue
        for word in words:
            word_len = len(word) + 1
            if current and current_len + word_len > size:
                flush()
            current.append(word)
            current_len += word_len
    flush()
    return chunks or [text]

# если модель выдаст «мышление» (шаблон Qwen3.5 подставляет префикс <think>/response)
# и оно утечёт в content — отрезаем его и оставляем только ответ
def _clean_output(content: str) -> str:
    content = content.strip()
    for marker in ("<think>", "</think>", " response\n\n"):
        if marker in content:
            parts = content.split(marker)
            content = parts[-1].strip()
    return content

# промпты для Qwen3.5-4B (детальный) и Qwen3.5-2B (короткий); chunk
# приклеивается к шаблону в цикле
_SYSTEM_4B = (
    "Ты редактор транскрипций русской речи. "
    "Твоя задача — только исправлять орфографию, очевидные ошибки распознавания "
    "и пунктуацию. Смысл, факты и порядок изложения должны сохраняться. "
    "Не пересказывай и не сокращай текст. "
    "Возвращай только исправленный текст."
)
_PROMPT_4B = (
    "Отредактируй текст, полученный из распознавания речи Whisper.\n\n"

    "ЗАДАЧА:\n"
    "1. Исправь орфографические ошибки.\n"
    "2. Исправь очевидные ошибки распознавания, если правильный вариант "
    "однозначно следует из контекста.\n"
    "3. Восстанови пунктуацию: точки, запятые, двоеточия, тире, "
    "вопросительные и восклицательные знаки.\n"
    "4. Раздели текст на логические абзацы.\n\n"

    "СТРОГО СОБЛЮДАЙ:\n"
    "- Сохраняй исходный смысл.\n"
    "- Не добавляй новую информацию.\n"
    "- Не удаляй информацию.\n"
    "- Не пересказывай и не сокращай текст.\n"
    "- Не меняй порядок изложения.\n"
    "- Не заменяй разговорную речь на литературный стиль без необходимости.\n"
    "- Не исправляй слова, если их значение неочевидно из контекста.\n"
    "- Сохраняй имена, названия, технические термины и аббревиатуры.\n"
    "- Слова-паразиты можно удалить только если они явно являются "
    "бессмысленными артефактами распознавания.\n\n"

    "ФОРМАТ ОТВЕТА:\n"
    "- Верни только отредактированный текст.\n"
    "- Не добавляй пояснений, комментариев или вступления.\n"
    "- Не используй Markdown.\n\n"

    "Исходный текст:\n"
)
_SYSTEM_2B = (
    "Редактируй транскрипцию русской речи. "
    "Исправляй орфографию, очевидные ошибки распознавания и пунктуацию. "
    "Сохраняй смысл, факты и порядок изложения. "
    "Не сокращай и не пересказывай текст. "
    "Возвращай только исправленный текст."
)
_PROMPT_2B = (
    "Исправь этот текст после распознавания Whisper. "
    "Сохрани разговорный стиль, но сделай пунктуацию естественной "
    "и раздели текст на логические абзацы.\n\n"
)

def process_audio(raw_text):
    # шаг 2: Пост-обработка локальной LLM
    if len(raw_text) == 0:
        return '[По всей видимости, в аудио нету речи]'
    if _llm_backend() == "off":
        return raw_text
    stripped = raw_text.strip()
    if not stripped:
        return raw_text
    if len(raw_text.split()) <= 100:
        return raw_text

    llm = _get_llm()
    if not llm:
        return raw_text
    if "2B" in getattr(config, "QWEN_MODEL", ""):
        system_prompt = _SYSTEM_2B
        prompt_tpl = _PROMPT_2B
    else:
        system_prompt = _SYSTEM_4B
        prompt_tpl = _PROMPT_4B
    results = []
    failed_chunks = 0
    total_chunks = 0
    start = time.perf_counter()
    for chunk in _chunk_text(stripped):
        total_chunks += 1
        prompt = prompt_tpl + chunk
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # низкая температура, чтобы модель не галюцинировала
            )
            response = cast(CreateChatCompletionResponse, response)
            fixed = _clean_output(response["choices"][0]["message"]["content"] or "")
        except Exception as e:
            logger.warning("Ошибка локальной модели: %s", e)
            fixed = chunk

        if len(fixed) < len(chunk) * 0.5:
            # защита на случай если текст будет слишком большим и квен не сможет переварить
            # у меня просто были большие пробелмы с этим ахахахахах
            logger.warning(
                "Ответ модели подозрительно короткий (%d из %d символов), "
                "оставлен исходный фрагмент",
                len(fixed),
                len(chunk),
            )
            fixed = chunk

        if fixed == chunk:
            failed_chunks += 1

        results.append(fixed)

    if failed_chunks == total_chunks:
        logger.warning(
            "Qwen не обработал ни один из %d чанков — возвращён сырой текст",
            total_chunks,
        )

    logger.info(
        "Qwen: пост-обработка заняла %.2f сек (%d чанков)",
        time.perf_counter() - start,
        total_chunks,
    )

    return "\n\n".join(results)


# ---------- пост-обработка LLM по API (OpenRouter) ----------

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_api_session: aiohttp.ClientSession | None = None


async def _get_api_session() -> aiohttp.ClientSession:
    global _api_session
    if _api_session is None or _api_session.closed:
        proxy = os.getenv("PROXY_URL")
        connector = ProxyConnector.from_url(proxy) if proxy else None
        _api_session = aiohttp.ClientSession(connector=connector)
    return _api_session


async def close_api_session():
    global _api_session
    if _api_session is not None and not _api_session.closed:
        await _api_session.close()


async def _post_chunk(session: aiohttp.ClientSession, system_prompt: str, prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = getattr(config, "API_LLM_MODEL", "") or os.getenv("MODEL") or "nvidia/nemotron-3-super-120b-a12b:free"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,  # низкая температура, чтобы модель не галюцинировала
        "max_tokens": 4096,  # чанки по 3000 символов могут не влезть в 2048
        "reasoning": {"enabled": False},  # без этого рассуждения модели «протекают» в content
    }
    async with session.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        body = await resp.text()
        if resp.status != 200:
            logger.warning("OpenRouter (пост-обработка): HTTP %s: %s", resp.status, body[:500])
            return ""
        data = await resp.json(content_type=None)
    choices = data.get("choices")
    if not choices:
        logger.warning("OpenRouter (пост-обработка): ответ без choices: %s", body[:500])
        return ""
    content = choices[0]["message"]["content"]
    if not content or not content.strip():
        logger.warning("OpenRouter (пост-обработка): пустой ответ")
        return ""
    return _clean_output(content)


async def process_audio_api(raw_text: str) -> str:
    # шаг 2: Пост-обработка LLM по API (OpenRouter), промпты те же, что у Qwen 4B
    if len(raw_text) == 0:
        return '[По всей видимости, в аудио нету речи]'
    if _llm_backend() != "api":
        return raw_text
    stripped = raw_text.strip()
    if not stripped:
        return raw_text
    if len(raw_text.split()) <= 100:
        return raw_text
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("OpenRouter: OPENROUTER_API_KEY не задан — сырой текст")
        return raw_text

    session = await _get_api_session()
    results = []
    failed_chunks = 0
    total_chunks = 0
    start = time.perf_counter()
    for chunk in _chunk_text(stripped):
        total_chunks += 1
        prompt = _PROMPT_4B + chunk
        try:
            fixed = await _post_chunk(session, _SYSTEM_4B, prompt)
        except Exception as e:
            logger.warning("Ошибка OpenRouter (пост-обработка): %s", e)
            fixed = ""

        if len(fixed) < len(chunk) * 0.5:
            # защита: ошибка/короткий ответ — оставляем исходный фрагмент
            logger.warning(
                "Ответ модели подозрительно короткий (%d из %d символов), "
                "оставлен исходный фрагмент",
                len(fixed),
                len(chunk),
            )
            fixed = chunk

        if fixed == chunk:
            failed_chunks += 1

        results.append(fixed)

    if failed_chunks == total_chunks:
        logger.warning(
            "API-LLM не обработал ни один из %d чанков — возвращён сырой текст",
            total_chunks,
        )

    logger.info(
        "API-LLM: пост-обработка заняла %.2f сек (%d чанков, модель=%s)",
        time.perf_counter() - start,
        total_chunks,
        getattr(config, "API_LLM_MODEL", ""),
    )

    return "\n\n".join(results)

# сама расшифровка
def transcribe(model, file_path):
    start = time.perf_counter()
    try:
        segments, _ = model.transcribe(
            str(file_path),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        # собираем через пробел, чтобы слова не слиплись на стыках сегментов
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.remove(file_path)
    logger.info("Whisper: распознавание заняло %.2f сек", time.perf_counter() - start)
    return text


# если хотите использовать без тг бота (почему-то)
if __name__ == "__main__":
    setup_proxy_env()
    files = [
        f.name for f in script_dir.iterdir()
        if f.is_file() and f.suffix.lower() in target_extensions
    ]
    print(files)
    try:
        file_path = Path(files[0])
    except IndexError:
        print("Файл .ogg не найден.")
        exit(0)

    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type="int8_float16" if config.WHISPER_DEVICE == "cuda" else "int8",
    )
    text = transcribe(model, file_path)
    if _llm_backend() == "api":
        import asyncio

        print(asyncio.run(process_audio_api(text)))
    else:
        print(process_audio(text))
