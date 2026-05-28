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
    # plastic / foil bags + metal cans + bottles -> Gelbe Tonne
    "CHIPS": "plastic", "CANDY": "plastic", "SODA": "plastic", "WATER": "plastic",
    "BEANS": "metal", "CORN": "metal", "FISH": "metal", "TOMATO_SAUCE": "metal",
    # cardboard / paper boxes -> Papier
    "CEREAL": "cardboard", "PASTA": "cardboard", "RICE": "cardboard",
    "FLOUR": "cardboard", "SUGAR": "cardboard", "TEA": "cardboard",
    # glass jars -> Altglas
    "JAM": "glass", "HONEY": "glass",
}

def main():
    added = {}
    for groc, material in GROC_TO_MATERIAL.items():
        files = sorted(glob.glob(os.path.join(SRC, groc, "*")))[:PER_CLASS]
        out_dir = os.path.join(DST, material)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for f in files:
            im = cv2.imread(f)
            if im is None:
                continue
            h, w = im.shape[:2]
            if max(h, w) > MAXSIDE:
                s = MAXSIDE / max(h, w)
                im = cv2.resize(im, (int(w * s), int(h * s)))
            cv2.imwrite(os.path.join(out_dir, "groc_%s_%03d.jpg" % (groc, n)), im)
            n += 1
        added[material] = added.get(material, 0) + n
    print("added per material:", added)
    print("(remove later with: find data/trashnet -name 'groc_*' -delete)")


if __name__ == "__main__":
    main()
