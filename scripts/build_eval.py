import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import hf_hub_download
from trashsort import config

REPO = "omasteam/waste-garbage-management-dataset"

def main():
    csv_path = os.path.join(config.EVAL_DIR, "labels.csv")
    img_dir = os.path.join(config.EVAL_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    # download the exact images named in labels.csv
    downloaded_count = 0
    with open(csv_path) as label_file:
        for row in csv.DictReader(label_file):
            name = row["file"]
            orig = name.split("__", 1)[1]
            source_path = hf_hub_download(
                REPO,
                f'{row["material"]}/{orig}',
                repo_type="dataset",
            )
            output_path = os.path.join(img_dir, name)
            with open(source_path, "rb") as source_file:
                with open(output_path, "wb") as output_file:
                    output_file.write(source_file.read())
            downloaded_count += 1
    print("downloaded %d eval images -> %s" % (downloaded_count, img_dir))

if __name__ == "__main__":
    main()
