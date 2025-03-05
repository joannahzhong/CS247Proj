import json
import os
import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def load_glosses(file):
    with open(file, "r") as f:
        json_file = json.load(f)
    glosses = " ".join([item["gloss"] for item in json_file])
    return glosses


def generate_word_clouds(left_text, right_text, save_dir, left_label, right_label):
    """Generates side-by-side word clouds for two different glosses."""

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Define output path
    output_path = os.path.join(save_dir, "wordcloud.png")

    left_title = left_label if left_label else 'Left'
    right_title = right_label if right_label else 'Right'

    wordcloud_left = WordCloud(width=800, height=400, background_color="white").generate(left_text)
    wordcloud_right = WordCloud(width=800, height=400, background_color="white").generate(right_text)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Ground Truth Word Cloud
    axes[0].imshow(wordcloud_left, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title(left_title)

    # Prediction Word Cloud
    axes[1].imshow(wordcloud_right, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title(right_title)

    plt.tight_layout()
    plt.savefig(output_path) 
    print(f"Word clouds saved to {output_path}")


def main():
    """Main function to parse arguments and generate word clouds."""
    parser = argparse.ArgumentParser(description="Generate side by side Word Clouds for glosses")
    parser.add_argument("--file_left", type=str, help="Path to the 1st JSON file")
    parser.add_argument("--label_left", type=str, help="Label of the 1st JSON file")
    parser.add_argument("--file_right", type=str, help="Path to the 2nd JSON file")
    parser.add_argument("--label_right", type=str, help="Label of the 2nd JSON file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated word cloud image")
    args = parser.parse_args()

    # Load glosses
    left_text = load_glosses(args.file_left)
    right_text = load_glosses(args.file_right)

    # Generate word clouds
    generate_word_clouds(left_text, right_text, args.save_dir, args.label_left, args.label_right)


if __name__ == "__main__":
    main()