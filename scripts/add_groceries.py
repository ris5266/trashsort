import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

SRC = "data/groceries/images"
DST = "data/trashnet"
PER_CLASS = 100
MAXSIDE = 512

GROC_TO_MATERIAL = {
    # plastic, foil bags, metal cans, and bottles
    "CHIPS": "plastic", "CANDY": "plastic", "SODA": "plastic", "WATER": "plastic",
    "BEANS": "metal", "CORN": "metal", "FISH": "metal", "TOMATO_SAUCE": "metal",
    # cardboard and paper boxes
    "CEREAL": "cardboard", "PASTA": "cardboard", "RICE": "cardboard",
    "FLOUR": "cardboard", "SUGAR": "cardboard", "TEA": "cardboard",
    # glass jars
    "JAM": "glass", "HONEY": "glass",
}

def main():
    added = {}
    for grocery_class, material in GROC_TO_MATERIAL.items():
        files = sorted(
            glob.glob(os.path.join(SRC, grocery_class, "*"))
        )[:PER_CLASS]
        output_dir = os.path.join(DST, material)
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0

        for path in files:
            image = cv2.imread(path)
            if image is None:
                continue

            height, width = image.shape[:2]
            if max(height, width) > MAXSIDE:
                scale = MAXSIDE / max(height, width)
                image = cv2.resize(
                    image,
                    (int(width * scale), int(height * scale)),
                )

            filename = f"groc_{grocery_class}_{saved_count:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), image)
            saved_count += 1

        added[material] = added.get(material, 0) + saved_count
    print("added per material:", added)
    print("(remove later with: find data/trashnet -name 'groc_*' -delete)")

if __name__ == "__main__":
    main()
