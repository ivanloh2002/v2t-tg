#!/usr/bin/env python3
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ENV_FILE = SCRIPT_DIR / ".env"
CONFIG_FILE = SCRIPT_DIR / "config.py"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


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

# Модель Whisper: tiny / base / small / medium / large-v3
WHISPER_MODEL = "{cfg['whisper_model']}"
# Устройство: cuda / cpu
WHISPER_DEVICE = "{cfg['whisper_device']}"
# Пост-обработка: True — whisper + qwen, False — только whisper
USE_QWEN = {cfg['use_qwen']}
# Доступ: "public" — все, "whitelist" — только перечисленные ID
ACCESS = "{cfg['access']}"
ALLOWED_USERS = {cfg['allowed_users']}
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
    whisper_model = WHISPER_MODELS[ask_choice("Модель Whisper:", WHISPER_MODELS, default=3)]
    whisper_device = ["cuda", "cpu"][ask_choice("Устройство:", ["cuda", "cpu"], default=0)]

    print("\n--- Пост-обработка ---")
    use_qwen = ask_choice("Режим обработки:", ["whisper + qwen", "только whisper"], default=0) == 0

    print("\n--- Доступ ---")
    access = ["whitelist", "public"][ask_choice("Доступ к боту:", ["вайтлист по ID", "свободный"], default=1)]
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
    print(f"Обработка:            {'whisper + qwen' if use_qwen else 'только whisper'}")
    if access == "whitelist":
        print(f"Доступ:               вайтлист: {', '.join(map(str, allowed_users))}")
    else:
        print("Доступ:               свободный")
    if not ask_yes_no("Всё верно? Записать конфигурацию?"):
        print("Отменено.")
        sys.exit(0)

    write_env({
        "API_TOKEN": api_token,
        "PROXY_URL": proxy,
        "OPENROUTER_API_KEY": openrouter_key,
        "MODEL": model,
        "HF_TOKEN": hf_token,
    })
    write_config({
        "whisper_model": whisper_model,
        "whisper_device": whisper_device,
        "use_qwen": use_qwen,
        "access": access,
        "allowed_users": allowed_users,
    })
    print("\nГотово! Запуск бота: ./run.sh  (или .venv/bin/python bot.py)")


if __name__ == "__main__":
    main()
