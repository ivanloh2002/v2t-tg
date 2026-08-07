from faster_whisper import WhisperModel
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
from llama_cpp import Llama
from huggingface_hub import snapshot_download
import config


load_dotenv()
# некоторые говнососы говорят, что токен от хф необязателен, но это ложь. У меня банально не запустился бот
# так что придётся регаться на хг ахахахаххахаха
# этот токен нужен для whisper т.к я использую whisper-faster а не обычный
# т.к обычный нельзя квантовать
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN"))

# ищет телеграмовские .ogg и .mp4 файлы
# .ogg - это гс
script_dir = Path(__file__).parent
target_extensions = {".ogg", ".mp4"}

logger = logging.getLogger(__name__)

# чанки
CHUNK_SIZE = 600

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    try:
        _qwen_dir = snapshot_download(
            "unsloth/Qwen3.5-4B-GGUF",
            local_files_only=True,
        )
        _llm = Llama(
            model_path=os.path.join(_qwen_dir, "Qwen3.5-4B-Q5_K_M.gguf"),
            n_gpu_layers=-1,
            n_ctx=4096
        )
    except Exception as e:
        # если модель не скачана/не грузится — не роняем всю расшифровку
        logger.warning("Не удалось загрузить Qwen: %s", e)
        _llm = False
    return _llm

#  разбивает текст на чанки, сохраняя абзацы
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

#  если модель выдаст «мышление» (шаблон Qwen3.5 подставляет префикс <think>/response)
#  и оно утечёт в content — отрезаем его и оставляем только ответ
def _clean_output(content: str) -> str:
    content = content.strip()
    for marker in ("<think>", "</think>", " response\n\n"):
        if marker in content:
            parts = content.split(marker)
            content = parts[-1].strip()
    return content

def process_audio(raw_text):
    # Шаг 2: Пост-обработка локальной LLM
    if not config.USE_QWEN:
        return raw_text
    stripped = raw_text.strip()
    if not stripped:
        return raw_text

    llm = _get_llm()
    if not llm:
        return raw_text
    results = []
    failed_chunks = 0
    total_chunks = 0
    for chunk in _chunk_text(stripped):
        total_chunks += 1
        prompt = (
            "Тебе дан сырой текст из системы распознавания речи (Whisper).\n"
            "Исправь орфографические ошибки и ОБЯЗАТЕЛЬНО расставь знаки препинания:\n"
            "каждое предложение заканчивай точкой или другим нужным знаком, добавь запятые.\n"
            "Разбей текст на логические абзацы.\n"
            "НЕ меняй смысл, НЕ добавляй и НЕ убирай детали.\n"
            "Отвечай сразу итоговым текстом, без комментариев и рассуждений.\n\n"
            f"Текст:\n{chunk}"
        )

        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Ты профессиональный редактор текста. Исправляешь распознанную речь: орфография, пунктуация, абзацы."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # низкая температура, чтобы модель не галюцинировала
            )
            fixed = _clean_output(response["choices"][0]["message"]["content"])
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

    return "\n\n".join(results)

# сама расшифровка
def transcribe(model, file_path):
    try:
        segments, _ = model.transcribe(str(file_path), beam_size=1, vad_filter=True)
        # собираем через пробел, чтобы слова не слиплись на стыках сегментов
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.remove(file_path)
    return text


# если хотите использовать без тг бота (почему-то)
if __name__ == "__main__":
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
        local_files_only=True
    )
    text = transcribe(model, file_path)
    print(process_audio(text))
