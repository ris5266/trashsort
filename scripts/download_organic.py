import os
import sys
import shutil
import collections

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset

OUT = "data/trashnet/organic"
PER_CLASS = 22
MAXSIDE = 512

def main():
    dataset = load_dataset(
        "Nattakarn/fruit-and-vegetable-image-recognition",
        split="train",
    )
    class_names = dataset.features["label"].names

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # keep a small balanced sample from each food class
    saved_per_class = collections.Counter()
    saved_count = 0
    for index, example in enumerate(dataset):
        class_name = class_names[example["label"]]
        if saved_per_class[class_name] >= PER_CLASS:
            continue

        image = example["image"].convert("RGB")
        if max(image.size) > MAXSIDE:
            scale = MAXSIDE / max(image.size)
            image = image.resize(
                (int(image.size[0] * scale), int(image.size[1] * scale))
            )

        safe_name = class_name.replace(" ", "_")
        output_path = os.path.join(OUT, f"organic_{safe_name}_{index:04d}.jpg")
        image.save(output_path)
        saved_per_class[class_name] += 1
        saved_count += 1

    print(
        "saved %d real organic images (<=%d per type) -> %s"
        % (saved_count, PER_CLASS, OUT)
    )

if __name__ == "__main__":
    main()
