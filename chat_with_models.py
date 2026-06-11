import argparse
import json
from pathlib import Path

from emotion_response.infer import build_output


def existing_path(path):
    return str(path) if path and Path(path).exists() else None


def print_result(result):
    classification = result["emotion_classification_before_mapping"]
    print("\nEmotion classification")
    print(f"- GoEmotions label: {classification['label']}")
    print(f"- Confidence: {classification['confidence']:.4f}")
    print(f"- Source: {classification['source']}")
    print(f"- DailyDialog emotion: {result['mapped_daily_dialog_emotion']}")

    print("\nEmotion-aware responses")
    for index, item in enumerate(result["sample_responses"], start=1):
        print(f"{index}. {item['response']}")
        print(f"   Why: {item['why_it_corresponds_to_emotion']}")

    if "input_only_sample_responses" in result:
        print("\nInput-only responses")
        for index, item in enumerate(result["input_only_sample_responses"], start=1):
            print(f"{index}. {item['response']}")
            print(f"   Why: {item['why_it_uses_the_input']}")


def main():
    parser = argparse.ArgumentParser(description="Chat with the trained response models.")
    parser.add_argument("--classifier", default="checkpoints/emotion.pt")
    parser.add_argument("--generator", default="checkpoints/generator.pt")
    parser.add_argument("--input-generator", default="checkpoints/input_generator.pt")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of readable text.")
    args = parser.parse_args()

    classifier = existing_path(args.classifier)
    generator = existing_path(args.generator)
    input_generator = existing_path(args.input_generator)

    print("Interactive model chat")
    print("Type a message and press Enter. Type quit, exit, or q to stop.")
    if not classifier:
        print(f"Using classifier fallback because {args.classifier} was not found.")
    if not generator:
        print(f"Using response fallback because {args.generator} was not found.")
    if not input_generator:
        print(f"Skipping input-only generator because {args.input_generator} was not found.")

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue

        result = build_output(
            text,
            classifier_path=classifier,
            generator_path=generator,
            input_generator_path=input_generator,
            samples=args.samples,
            top_k=args.top_k,
            temperature=args.temperature,
            seed=args.seed,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(result)


if __name__ == "__main__":
    main()
