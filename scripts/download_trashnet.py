import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import hf_hub_download

from trashsort import config

def main():
    zip_path = hf_hub_download(repo_id="garythung/trashnet", filename="dataset-resized.zip", repo_type="dataset")

    os.makedirs(config.TRASHNET_DIR, exist_ok=True)
    # unpack only the material folders used by this project
    counts = {class_name: 0 for class_name in config.CLASSES}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            base = os.path.basename(name)
            if base.startswith(".") or not base.lower().endswith(".jpg"):
                continue
            class_name = name.split("/")[-2]
            if class_name not in config.CLASSES:
                continue

            folder = os.path.join(config.TRASHNET_DIR, class_name)
            os.makedirs(folder, exist_ok=True)
            output_path = os.path.join(folder, os.path.basename(name))
            with archive.open(name) as source_file:
                with open(output_path, "wb") as output_file:
                    output_file.write(source_file.read())
            counts[class_name] += 1

    print(counts)
    print("done ->", config.TRASHNET_DIR)

if __name__ == "__main__":
    main()
