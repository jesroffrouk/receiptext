# Local Receipt Extractor

A tiny script that reads receipt images with an open-weight vision model running **locally via Ollama** and returns structured data: vendor, date, total, currency, and expense category.

No cloud API, no per-page fees, and receipts never leave your machine.

## Setup

```bash
# 1. Install Ollama: https://ollama.com
ollama pull qwen2.5vl:7b

# 2. Install dependencies
pip install ollama pydantic
```

## Usage

```bash
python receipt_extract.py receipt.jpg
python receipt_extract.py ./receipts --csv results.csv
python receipt_extract.py ./receipts --model gemma3:4b
```

Example output (one JSON line per receipt):

```json
{"file": "receipt.jpg", "vendor": "Starbucks", "date": "2024-03-12", "total": 8.45, "currency": "USD", "category": "meals"}
```

## Why open models?

- **Privacy:** receipts contain client names and payment details; nothing is uploaded.
- **Cost:** free to run, no per-receipt charges.
- **Offline:** works with no internet after the model is downloaded.
- **Swappable:** change the model with `--model`, no code changes.

## Limitations

Small local models can misread blurry or unusual receipts. Always spot-check the output before using it for taxes.

## License

MIT
