#!/usr/bin/env bash
#
# использовать этот файл для запуска бота ОБЯЗАТЕЛЬНО. т.к там какято хурма с кудами у faster_whisper
# изза чего у меня в проекте хахахаха, две развне версии куда.
# ну это плохо, ну а что я сделаю
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import os; import nvidia.cublas, nvidia.cudnn; print(os.path.join(nvidia.cublas.__path__[0], "lib") + ":" + os.path.join(nvidia.cudnn.__path__[0], "lib"))')"
exec .venv/bin/python bot.py "$@"
