import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import hf_hub_download

from trashsort import config

def main():
    zip_path = hf_hub_download(repo_id="garythung/trashnet", filename="dataset-resized.zip", repo_type="dataset")

    os.makedirs(config.TRASHNET_DIR, exist_ok=True)
    counts = {c: 0 for c in config.CLASSES}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base.startswith(".") or not base.lower().endswith(".jpg"):
                continue
            cls = name.split("/")[-2]
            if cls not in config.CLASSES:
                continue
            folder = os.path.join(config.TRASHNET_DIR, cls)
            os.makedirs(folder, exist_ok=True)
            out = os.path.join(folder, os.path.basename(name))
            with z.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
            counts[cls] += 1

    print(counts)
    print("done ->", config.TRASHNET_DIR)


if __name__ == "__main__":
    main()
