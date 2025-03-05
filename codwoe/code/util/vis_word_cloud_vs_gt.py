import json
import os
import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def load_ground_truth(ground_truth_file):
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)
    ground_truth_glosses = " ".join([item["gloss"] for item in ground_truth])
    return ground_truth_glosses


def load_predictions(pred_file):
    with open(pred_file, "r") as f:
        predictions = json.load(f)
    predicted_glosses = " ".join([item["gloss"] for item in predictions])
    return predicted_glosses


def generate_word_clouds(ground_truth_text, predicted_text, save_dir):
    """Generates side-by-side word clouds for ground truth and predicted glosses."""

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Define output path
    output_path = os.path.join(save_dir, "wordcloud.png")

    wordcloud_ground_truth = WordCloud(width=800, height=400, background_color="white").generate(ground_truth_text)
    wordcloud_predictions = WordCloud(width=800, height=400, background_color="white").generate(predicted_text)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Ground Truth Word Cloud
    axes[0].imshow(wordcloud_ground_truth, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Most Frequent Words in Ground Truth Glosses")

    # Prediction Word Cloud
    axes[1].imshow(wordcloud_predictions, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Most Frequent Words in Model Predictions")

    plt.tight_layout()
    plt.savefig(output_path) 
    print(f"Word clouds saved to {output_path}")


def main():
    """Main function to parse arguments and generate word clouds."""
    parser = argparse.ArgumentParser(description="Generate Word Clouds for Ground Truth vs. Predicted Glosses")
    parser.add_argument("--ground_truth_file", type=str, help="Path to the ground truth JSON file, 'en.test.defmod.complete.json'")
    parser.add_argument("--pred_file", type=str, help="Path to the predicted JSON file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated word cloud image")
    args = parser.parse_args()

    # Load glosses
    ground_truth_text = load_ground_truth(args.ground_truth_file)
    predicted_text = load_predictions(args.pred_file)

    # Generate word clouds
    generate_word_clouds(ground_truth_text, predicted_text, args.save_dir)


if __name__ == "__main__":
    main()