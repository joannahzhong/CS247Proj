import argparse
import json
import os


def clean_predictions(pred_file):
    # Open prediction file
    with open(pred_file, "r") as f:
        preds = json.load(f)

    # Iterate through predictions
    for pred in preds:
        # Remove all whitespace
        clean_gloss = "".join(pred["gloss"].split())

        # Replace unicode with whitespace
        clean_gloss = clean_gloss.replace("\u2581", " ").strip()
        clean_gloss = clean_gloss.replace("<seq>", "")

        # Update gloss
        pred["gloss"] = clean_gloss

    # Update filename to write predictions
    filename, ext = os.path.basename(pred_file).split(".")
    filename = filename + "_clean"
    filename = ".".join([filename, ext])
    filename = os.path.join(os.path.dirname(pred_file), filename)

    # Write to file
    with open(filename, "w") as f:
        json.dump(preds, f)

    # Print output
    # print(json.dumps(preds, indent=2))
    print(f"Saving JSON as: {filename}")


if __name__ == "__main__":
    # Argument parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("pred_file", help="Path to prediction file")
    args = argparser.parse_args()

    # Clean predictions
    clean_predictions(args.pred_file)
