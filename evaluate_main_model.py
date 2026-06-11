import argparse
import json
import os
from collections import defaultdict

from emotion_response.data import load_daily_dialog_emotion_pairs, load_goemotions_rows
from emotion_response.infer import generate_one, load_classifier, load_generator
from emotion_response.labels import DAILYDIALOG_EMOTIONS, GOEMOTIONS_LABELS, map_goemotion_to_daily_dialog
from emotion_response.metrics import classification_report
from emotion_response.models import require_torch, torch
from emotion_response.prompts import format_response_prompt


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def require_checkpoint(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {name} checkpoint: {path}\n"
            "Train the model first, copy the Colab checkpoint into this local checkpoints folder, "
            "or pass a different checkpoint path with the command-line argument."
        )


def classify_batch(model, tokenizer, texts, max_len, device):
    input_ids = [tokenizer.encode(text, max_len=max_len) for text in texts]
    x = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask = (x != tokenizer.pad_id).long()
    with torch.no_grad():
        logits = model(x, mask)
        predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().tolist()


def evaluate_classifier(classifier_path, split, limit, batch_size, device):
    model, tokenizer, max_len = load_classifier(classifier_path, device)
    rows = load_goemotions_rows(split)
    if limit:
        rows = rows[:limit]

    predictions = []
    targets = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [text for text, _ in batch]
        labels = [label for _, label in batch]
        predictions.extend(classify_batch(model, tokenizer, texts, max_len, device))
        targets.extend(labels)

    return classification_report(predictions, targets, GOEMOTIONS_LABELS)


def make_daily_confusion():
    return {target: {pred: 0 for pred in DAILYDIALOG_EMOTIONS} for target in DAILYDIALOG_EMOTIONS}


def evaluate_generation_alignment(
    classifier_path,
    generator_path,
    split,
    limit,
    samples_per_input,
    top_k,
    temperature,
    device,
    seed,
):
    classifier, classifier_tokenizer, classifier_max_len = load_classifier(classifier_path, device)
    generator, generator_tokenizer, generator_max_len = load_generator(generator_path, device)
    rows = load_daily_dialog_emotion_pairs(split)
    if limit:
        rows = rows[:limit]
    if seed is not None:
        torch.manual_seed(seed)

    confusion = make_daily_confusion()
    per_target = defaultdict(lambda: {"matches": 0, "total": 0})
    examples = []
    total = 0
    matches = 0

    for target_emotion, user_message, _reference_reply in rows:
        prompt = format_response_prompt(target_emotion, user_message)
        for _ in range(samples_per_input):
            response = generate_one(
                generator,
                generator_tokenizer,
                prompt,
                generator_max_len,
                top_k,
                temperature,
                device,
            )
            predicted_go_index = classify_batch(
                classifier,
                classifier_tokenizer,
                [response],
                classifier_max_len,
                device,
            )[0]
            predicted_go = GOEMOTIONS_LABELS[predicted_go_index]
            predicted_daily = map_goemotion_to_daily_dialog(predicted_go)
            is_match = predicted_daily == target_emotion

            confusion[target_emotion][predicted_daily] += 1
            per_target[target_emotion]["total"] += 1
            total += 1
            if is_match:
                per_target[target_emotion]["matches"] += 1
                matches += 1

            if len(examples) < 20:
                examples.append(
                    {
                        "target_daily_dialog_emotion": target_emotion,
                        "input_message": user_message,
                        "generated_response": response,
                        "classified_goemotion": predicted_go,
                        "mapped_generated_emotion": predicted_daily,
                        "matches_target": is_match,
                    }
                )

    per_target_accuracy = {
        emotion: values["matches"] / values["total"] if values["total"] else 0.0
        for emotion, values in per_target.items()
    }
    return {
        "alignment_accuracy": matches / total if total else 0.0,
        "matches": matches,
        "total_generated_responses": total,
        "samples_per_input": samples_per_input,
        "per_target_accuracy": per_target_accuracy,
        "confusion_matrix": confusion,
        "examples": examples,
    }


def main():
    require_torch()
    parser = argparse.ArgumentParser(description="Grade classifier and emotion-aware generation model.")
    parser.add_argument("--classifier", default="checkpoints/emotion.pt")
    parser.add_argument("--generator", default="checkpoints/generator.pt")
    parser.add_argument("--classifier-split", default="validation")
    parser.add_argument("--generation-split", default="validation")
    parser.add_argument("--classifier-limit", type=int, help="Optional cap for quick classifier evaluation.")
    parser.add_argument("--generation-limit", type=int, default=200, help="Optional cap for generation evaluation.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-input", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="checkpoints/evaluation_report.json")
    args = parser.parse_args()

    require_checkpoint(args.classifier, "classifier")
    require_checkpoint(args.generator, "generator")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "classifier": evaluate_classifier(
            args.classifier,
            args.classifier_split,
            args.classifier_limit,
            args.batch_size,
            device,
        ),
        "generation_emotion_alignment": evaluate_generation_alignment(
            args.classifier,
            args.generator,
            args.generation_split,
            args.generation_limit,
            args.samples_per_input,
            args.top_k,
            args.temperature,
            device,
            args.seed,
        ),
    }

    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to {args.out}")


if __name__ == "__main__":
    main()
