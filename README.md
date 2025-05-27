# Transcript to WebVTT

This repository provides `convert_to_vtt.py`, a script that converts a voice-to-text transcript in JSON format into an English or bilingual WebVTT subtitle file.

## Requirements

- Python 3.10+
- See `requirements.txt` for Python package dependencies.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Set these environment variables before running:

- `OPENAI_API_KEY` – your OpenAI API key.
- `OPENAI_API_BASE_URL` – base URL for the API (optional).
- `OPENAI_MODEL` – model name used for translation.

Then run:

```bash
python convert_to_vtt.py transcript.json
```

Add `--model MODEL_NAME` to override the model, `--concurrency N` to change concurrency, `--lang LANGUAGE` to translate into an additional language (omit the flag for English only), `--font-size SIZE` to change the subtitle font size in `em` units, or `--light-bg` for dark text on a light translucent background.

The script writes `<transcript>.vtt` to the current directory.
