# Emotion-Aware Response Generator

## Purpose

This repository builds an emotion-aware dialogue response system that:

- classifies user text into GoEmotions emotion labels,
- maps those labels to DailyDialog emotions,
- generates candidate responses conditioned on the predicted emotion,
- produces multiple top-k sampled responses with explanation templates,
- provides an input-only baseline response generator for comparison.

The goal is to demonstrate an emotion-conditioned response pipeline with training, inference, and evaluation components.

## Quick Start

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Run inference with a sample text.

```powershell
python -m emotion_response.infer --text "I cannot believe you forgot my birthday."
```

5. If you have model checkpoints, run inference with explicit files.

```powershell
python -m emotion_response.infer --text "I cannot believe you forgot my birthday." --classifier checkpoints/emotion.pt --generator checkpoints/generator.pt --input-generator checkpoints/input_generator.pt
```

6. For interactive testing, use:

```powershell
python chat_with_models.py
```

## File Descriptions

- `README.md`
  - Project overview, setup instructions, and usage guidance.

- `requirements.txt`
  - Python package dependencies required to run training and inference.

- `chat_with_models.py`
  - Interactive console script for chatting with the trained models.

- `evaluate_main_model.py`
  - Evaluates the emotion classifier and generation quality using held-out data.

- `format_evaluation_report.py`
  - Converts evaluation JSON into CSV and PNG confusion matrix reports.

- `COLAB.md`
  - Notes for running the project in Google Colab.

- `Emotion_Response_Colab.ipynb`
  - Colab notebook for training, evaluation, and demonstration.

- `checkpoints/`
  - Stored model checkpoint files, metrics, and evaluation outputs.

- `emotion_response/`
  - Main package containing model training, inference, data processing, and prompts.

Inside `emotion_response/`:

- `__init__.py`
  - Package initializer.

- `data.py`
  - Dataset loading, preprocessing, and batching utilities.

- `explanations.py`
  - Templates and logic for generating explanation text.

- `fallbacks.py`
  - Lightweight fallback behavior for inference before checkpoints exist.

- `infer.py`
  - Inference module for emotion classification and response generation.

- `labels.py`
  - Label definitions and GoEmotions-to-DailyDialog mappings.

- `metrics.py`
  - Evaluation metrics and classification scoring utilities.

- `models.py`
  - Transformer-based encoder and generator model definitions.

- `prompts.py`
  - Prompt formatting for emotion-conditioned generation.

- `tokenizer.py`
  - Tokenization utilities used by training and inference.

- `train_emotion_classifier.py`
  - Training script for the GoEmotions emotion classifier.

- `train_input_response_generator.py`
  - Training script for the input-only response generator baseline.

- `train_response_generator.py`
  - Training script for the emotion-conditioned response generator.

## Training

Train the emotion classifier:

```powershell
python -m emotion_response.train_emotion_classifier --resume --epochs 12 --batch-size 128 --max-len 96 --max-vocab 30000 --d-model 192 --nhead 6 --layers 3 --class-weights inverse_sqrt --out checkpoints/emotion.pt --metrics-out checkpoints/emotion_metrics.json
```

Train the emotion-aware response generator:

```powershell
python -m emotion_response.train_response_generator --resume --balance-emotions --epochs 40 --batch-size 128 --max-len 128 --max-vocab 50000 --d-model 256 --nhead 8 --layers 6 --out checkpoints/generator.pt
```

Train the input-only response baseline:

```powershell
python -m emotion_response.train_input_response_generator --resume --epochs 40 --batch-size 128 --max-len 128 --max-vocab 50000 --d-model 256 --nhead 8 --layers 6 --out checkpoints/input_generator.pt
```

### Smoke test

Use shorter runs to verify the training pipeline without producing full models:

```powershell
python -m emotion_response.train_emotion_classifier --epochs 1 --limit 200 --val-limit 80 --out checkpoints/emotion_smoke.pt --metrics-out checkpoints/emotion_smoke_metrics.json
python -m emotion_response.train_response_generator --epochs 1 --limit 200 --val-limit 80 --out checkpoints/generator_smoke.pt
python -m emotion_response.train_input_response_generator --epochs 1 --limit 200 --val-limit 80 --out checkpoints/input_generator_smoke.pt
```

## Evaluation

Generate a full evaluation report after training:

```powershell
python evaluate_main_model.py --classifier checkpoints/emotion.pt --generator checkpoints/generator.pt --out checkpoints/evaluation_report.json
```

Convert the evaluation report into CSV and PNG confusion matrices:

```powershell
python format_evaluation_report.py --report checkpoints/evaluation_report.json
```

Expected outputs:

- `checkpoints/classifier_confusion_matrix.csv`
- `checkpoints/generation_confusion_matrix.csv`
- `checkpoints/classifier_confusion_matrix.png`
- `checkpoints/generation_confusion_matrix.png`

## Notes

- The project uses Hugging Face `datasets` to load `google-research-datasets/go_emotions` and `daily_dialog`.
- If the canonical `daily_dialog` loader fails with newer `datasets` versions, the code falls back to `OpenRL/daily_dialog`.
- Checkpoints are saved to `checkpoints/` after training, making it easy to resume or continue from interrupted runs.
- Use Google Colab GPU for full training, and refer to `COLAB.md` or `Emotion_Response_Colab.ipynb` for notebook-based execution.
