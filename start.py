#!/usr/bin/env python3
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ENV_FILE = SCRIPT_DIR / ".env"
CONFIG_FILE = SCRIPT_DIR / "config.py"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
QWEN_MODELS = [
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q5_K_M.gguf"),
    ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q5_K_M.gguf"),
]


def ask_str(prompt: str, *, default: str = "", required: bool = False) -> str:
    suffix = f" [по умолчанию: {default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default:
            return default
        if not value and required:
            print("Ошибка: поле обязательно.")
            continue
        return value


def ask_choice(prompt: str, options: list[str], default: int = 0) -> int:
    print(prompt)
    for i, opt in enumerate(options, 1):
        marker = " *" if i - 1 == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        raw = input(f"Выберите 1-{len(options)} [по умолчанию {default + 1}]: ").strip()
        if not raw:
            return default
        try:
            idx = int(raw)
        except ValueError:
            print("Введите число.")
            continue
        if 1 <= idx <= len(options):
            return idx - 1
        print("Неверный номер.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " (Y/n)" if default else " (y/N)"
    while True:
        value = input(prompt + suffix + ": ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes", "да", "д"):
            return True
        if value in ("n", "no", "нет", "н"):
            return False
        print("Ответьте y или n.")


def ask_ids(prompt: str) -> list[int]:
    while True:
        raw = input(prompt + ": ").strip()
        ids = []
        for part in re.split(r"[,; ]+", raw):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                print(f"'{part}' — не число, пропущено.")
        if ids:
            return ids
        print("Введите хотя бы один ID.")


def escape_env(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_env(keys: dict[str, str]) -> None:
    if ENV_FILE.exists():
        backup = SCRIPT_DIR / (ENV_FILE.name + ".bak")
        backup.write_text(ENV_FILE.read_text(encoding="utf-8"))
        print(f"Старый .env сохранён в {backup.name}")
    lines = "\n".join(f"{k}={escape_env(v)}" for k, v in keys.items() if v)
    ENV_FILE.write_text(lines + "\n", encoding="utf-8")


def write_config(cfg: dict) -> None:
    content = f'''"""Сгенерировано start.py — первичная настройка бота."""

# Модель Whisper: tiny / base / small / medium / large-v3 / large-v3-turbo
WHISPER_MODEL = "{cfg['whisper_model']}"
# Устройство: cuda / cpu
WHISPER_DEVICE = "{cfg["whisper_device"]}"
# Пост-обработка: "qwen" — локальная GGUF, "api" — LLM по API (OpenRouter), "off" — только whisper
LLM_BACKEND = "{cfg["llm_backend"]}"
# Модель OpenRouter для пост-обработки (при LLM_BACKEND="api")
API_LLM_MODEL = "{cfg["api_llm_model"]}"
# Режим вывода: True — сразу сырой текст whisper, затем замена на обработанный; False — только финальный
STREAM_RESULT = {cfg["stream_result"]}
# Qwen: репозиторий HuggingFace и файл GGUF-квантования (при LLM_BACKEND="qwen")
# unsloth/Qwen3.5-4B-GGUF / unsloth/Qwen3.5-2B-GGUF
QWEN_REPO = "{cfg["qwen_repo"]}"
# Qwen3.5-4B-Q4_K_M.gguf / Qwen3.5-4B-Q5_K_M.gguf / Qwen3.5-2B-Q5_K_M.gguf
QWEN_MODEL = "{cfg["qwen_model"]}"
# Доступ: "public" — все, "whitelist" — только перечисленные ID
ACCESS = "{cfg["access"]}"
ALLOWED_USERS = {cfg["allowed_users"]}
'''
    CONFIG_FILE.write_text(content, encoding="utf-8")


def main() -> None:
    print("=== Настройка транскрайб-бота ===\n")

    missing = [f for f in (".env", "config.py") if not (SCRIPT_DIR / f).is_file()]
    if missing:
        print("Обнаружены отсутствующие файлы конфигурации: " + ", ".join(missing))
        print("Сейчас они будут созданы.\n")
    else:
        if not ask_yes_no(".env и config.py уже существуют. Перезаписать настройки?"):
            print("Отменено.")
            sys.exit(0)

    print("--- API-ключи ---")
    api_token = ask_str("API_TOKEN (Telegram, от @BotFather)", required=True)
    proxy = ask_str("PROXY_URL (пусто — без прокси)")
    openrouter_key = ask_str("OPENROUTER_API_KEY (пусто — /short не будет работать)")
    model = ask_str("Модель OpenRouter для пересказов", default=DEFAULT_MODEL)
    hf_token = ask_str("HF_TOKEN (пусто — без токена HuggingFace)")

    print("\n--- Модель Whisper ---")
    whisper_model = WHISPER_MODELS[ask_choice("Модель Whisper:", WHISPER_MODELS, default=5)]
    whisper_device = ["cuda", "cpu"][ask_choice("Устройство:", ["cuda", "cpu"], default=0)]

    print("\n--- Пост-обработка ---")
    llm_backend = ["qwen", "api", "off"][
        ask_choice(
            "Режим обработки:",
            [
                "whisper + Qwen (локальная GGUF)",
                "whisper + LLM по API (OpenRouter)",
                "только whisper",
            ],
            default=0,
        )
    ]
    qwen_repo, qwen_model = QWEN_MODELS[0]
    api_llm_model = DEFAULT_MODEL
    stream_result = False
    if llm_backend != "off":
        if llm_backend == "api":
            api_llm_model = ask_str("Модель LLM по API для пост-обработки", default=DEFAULT_MODEL)
        else:
            qwen_repo, qwen_model = QWEN_MODELS[
                ask_choice("Модель Qwen:", [name for _, name in QWEN_MODELS], default=0)
            ]
        stream_result = ask_choice(
            "Режим вывода результата:",
            ["сразу сырой текст, затем заменить на обработанный", "показывать только после обработки"],
            default=0,
        ) == 0

    print("\n--- Доступ ---")
    access = ["whitelist", "public"][
        ask_choice("Доступ к боту:", ["вайтлист по ID", "свободный"], default=1)
    ]
    allowed_users = []
    if access == "whitelist":
        allowed_users = ask_ids("ID Telegram-пользователей (через запятую)")

    print("\n--- Подтверждение ---")
    print(f"API_TOKEN:            {api_token[:8]}...{api_token[-4:]}")
    print(f"PROXY_URL:            {proxy or '(нет)'}")
    print(f"OPENROUTER_API_KEY:   {'указан' if openrouter_key else '(нет)'}")
    print(f"OpenRouter MODEL:     {model}")
    print(f"HF_TOKEN:             {'указан' if hf_token else '(нет)'}")
    print(f"Whisper:              {whisper_model} ({whisper_device})")
    backend_names = {"qwen": "whisper + Qwen (локально)", "api": "whisper + LLM по API", "off": "только whisper"}
    print(f"Обработка:            {backend_names[llm_backend]}")
    if llm_backend == "qwen":
        print(f"Qwen:                 {qwen_model}")
        print(f"Вывод результата:     {'стрим (сразу сырой, затем замена)' if stream_result else 'только после обработки'}")
    elif llm_backend == "api":
        print(f"Модель API:           {api_llm_model}")
        print(f"Вывод результата:     {'стрим (сразу сырой, затем замена)' if stream_result else 'только после обработки'}")
    if access == "whitelist":
        print(f"Доступ:               вайтлист: {', '.join(map(str, allowed_users))}")
    else:
        print("Доступ:               свободный")
    if not ask_yes_no("Всё верно? Записать конфигурацию?"):
        print("Отменено.")
        sys.exit(0)

    write_env(
        {
            "API_TOKEN": api_token,
            "PROXY_URL": proxy,
            "OPENROUTER_API_KEY": openrouter_key,
            "MODEL": model,
            "HF_TOKEN": hf_token,
        }
    )
    write_config(
        {
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "llm_backend": llm_backend,
            "api_llm_model": api_llm_model,
            "stream_result": stream_result,
            "qwen_repo": qwen_repo,
            "qwen_model": qwen_model,
            "access": access,
            "allowed_users": allowed_users,
        }
    )
    print("\nГотово! Запуск бота: ./run.sh  (или .venv/bin/python bot.py)")


if __name__ == "__main__":
    main()
