[**Read in Russian**](README_ru.md)🇷🇺

# **v2t-tg Bot** #
**voice to text*
**Please**, read the README fully to avoid any misunderstandings.

This is a bot for transcribing your voice/video messages. It's powered by **faster_whisper** to convert **audio to text**.

There's also a local model **unsloth/Qwen3.5-4B-GGUF** used to polish the text after whisper.

You simply send the bot a voice or video message and it gives you the text. There's also a **retelling feature**, based on **nvidia/nemotron-3-super-120b-a12b:free**
The best part is that this project is **ABSOLUTELY FREE**

I just hope people will run it **locally**. I mean, either one person or a small group, since I've no idea who could afford renting a server with specs like these in 2026 without any benefit from it.

## **How this works in more detail** ##

It uses **faster_whisper**.

Requirements for different models (shown with **int8_float16** quantization):
| whisper | VRAM, GB | File size on disk | Parameters |
| :--- | :---: | :---: | :---: |
| tiny | ~0.3–0.4 | ~80 MB | 39M |
| base | ~0.4–0.5  | ~150 MB | 74M |
| small | ~0.6–0.8  | ~490 MB | 244M |
| medium | ~1.2–1.5  | 1.53 GB | 769M |
| large-v3 | ~2.0–2.5 | ~3.1 GB | 1550M |
| large-v3-turbo | ~1.2–1.6 | 1.62 GB | 809M |

(*I personally use the large-v3-turbo model, and I recommend it to you*)

After the transcription, the unsloth/Qwen3.5-4B-GGUF model is used to polish the text. It adds punctuation and fixes mistakes. Takes **~3.5 GB of VRAM**.

In total, during work it usually takes about **4610 MB of VRAM** for me (with the *qwen3.5-4b-q5* model). But, counting other apps, that's **6 gigs out of 8**.

How does the retelling work? Very simple: a prompt with the already-processed text is sent to nvidia/nemotron-3-super-120b-a12b:free via the OpenRouter API.

## Performance ##

I use a laptop with an RTX 3070 Ti with 8 gigs of VRAM. It handled a **7-minute-long message in 44 seconds** (on the *medium* model) (I counted it from the aiogram logs). And on the *large-v3-turbo* model the same message took **22 seconds** for me. That's, if anything, the full cycle time, i.e. transcription + polishing.

There's also support for *qwen3.5-2b-q5*, which is way faster — about **160 tokens per second**, compared to *qwen3.5-4b-q4* (**82 tokens per second**) and *qwen3.5-4b-q5* (**72 tokens per second**). Ultimately, decoding with *qwen2.5-2b-q5* is **27% faster**. But the 2-billion-parameter qwen is **much worse in quality**, so choose for yourselves.



The funniest thing is that TG Premium transcription couldn't handle this same message hahaha

That's a good and telling result, if you ask me.
## **Project structure** ##

```
v2t-tg/
├── start.py        # initial setup
├── run.sh          # launch
├── bot.py          # main file
├── main.py         # transcription
├── handlers.py     # handlers
├── short.py        # retelling
│
├── config.py       # configuration
├── .env            # environment variables
│
├── pyproject.toml  # dependencies and project config
├── uv.lock         # pinned dependency versions
├── .python-version # Python version
├── .gitignore      # ignored files
├── LICENSE         # license
└── README.md       # description
```
## **How to run it?** ##

1. Install the dependencies: `uv sync`
2. Go to the bot [**BotFather**](https://web.telegram.org/k/#@BotFather) and create a new bot. Copy the token.
3. Register on [**Hugging Face**](https://huggingface.co/) and get an API key. 
4. Also register and get an API key on [**OpenRouter**]( https://openrouter.ai/).
5. Run **start.py**, enter the API keys, the proxy, and configure everything the way you need.
6. Run **run.sh** to start the bot.



## **Links** ##

nvidia/nemotron-3-super-120b-a12b:free model: https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free

Qwen: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF

## **Code notes** ##

If a huge number of meaningless logs gets on your nerves, you can write *verbose=False,* in main.py in the instance of the Llama class _llm on line 58. But keep in mind that it disables **all** llama_cpp logs, including the important ones.
