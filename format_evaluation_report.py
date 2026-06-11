import argparse
import csv
import glob
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "checkpoints", "matplotlib-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
for lock_path in glob.glob(os.path.join(os.environ["MPLCONFIGDIR"], "*.matplotlib-lock")):
    try:
        os.remove(lock_path)
    except OSError:
        pass

import matplotlib.pyplot as plt


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_classifier_confusion(report, path):
    labels = report["classifier"]["labels"]
    matrix = report["classifier"]["confusion_matrix"]
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label \\ predicted_label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def write_generation_confusion(report, path):
    matrix = report["generation_emotion_alignment"]["confusion_matrix"]
    labels = sorted(matrix.keys())
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["target_emotion \\ predicted_emotion", *labels])
        for target in labels:
            writer.writerow([target, *[matrix[target].get(predicted, 0) for predicted in labels]])


def write_confusion_heatmap(matrix, labels, path, title, x_label, y_label):
    ensure_parent_dir(path)
    figure_width = max(7, len(labels) * 0.45)
    figure_height = max(6, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)

    max_value = max((max(row) for row in matrix), default=0)
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_classifier_heatmap(report, path):
    labels = report["classifier"]["labels"]
    matrix = report["classifier"]["confusion_matrix"]
    write_confusion_heatmap(
        matrix,
        labels,
        path,
        "Classifier Confusion Matrix",
        "Predicted label",
        "True label",
    )


def write_generation_heatmap(report, path):
    confusion = report["generation_emotion_alignment"]["confusion_matrix"]
    labels = sorted(confusion.keys())
    matrix = [[confusion[target].get(predicted, 0) for predicted in labels] for target in labels]
    write_confusion_heatmap(
        matrix,
        labels,
        path,
        "Generation Emotion Alignment Confusion Matrix",
        "Predicted emotion",
        "Target emotion",
    )


def print_summary(report):
    classifier = report["classifier"]
    generation = report["generation_emotion_alignment"]
    print("Classifier")
    print(f"- Accuracy: {classifier['accuracy']:.4f}")
    print(f"- Macro F1: {classifier['macro_f1']:.4f}")
    print("\nGeneration emotion alignment")
    print(f"- Accuracy: {generation['alignment_accuracy']:.4f}")
    print(f"- Matches: {generation['matches']} / {generation['total_generated_responses']}")
    print("\nPer-target generation accuracy")
    for emotion, value in sorted(generation["per_target_accuracy"].items()):
        print(f"- {emotion}: {value:.4f}")


def write_output(path, writer):
    try:
        writer(path)
    except PermissionError:
        print(f"Could not write {path}; close it if it is open in another app and run this script again.")
        return False
    print(f"Wrote {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Format evaluation report confusion matrices as labeled CSV files and colored heatmaps.")
    parser.add_argument("--report", default="checkpoints/evaluation_report.json")
    parser.add_argument("--classifier-out", default="checkpoints/classifier_confusion_matrix.csv")
    parser.add_argument("--generation-out", default="checkpoints/generation_confusion_matrix.csv")
    parser.add_argument("--classifier-plot-out", default="checkpoints/classifier_confusion_matrix.png")
    parser.add_argument("--generation-plot-out", default="checkpoints/generation_confusion_matrix.png")
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as file:
        report = json.load(file)

    outputs = [
        (args.classifier_out, lambda path: write_classifier_confusion(report, path)),
        (args.generation_out, lambda path: write_generation_confusion(report, path)),
        (args.classifier_plot_out, lambda path: write_classifier_heatmap(report, path)),
        (args.generation_plot_out, lambda path: write_generation_heatmap(report, path)),
    ]

    print_summary(report)
    print()
    for path, writer in outputs:
        write_output(path, writer)


if __name__ == "__main__":
    main()
