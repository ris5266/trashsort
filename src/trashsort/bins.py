BINS = {
    "restmuell": {
        "name": "Restmuell",
        "color_name": "schwarz/grau",
        "color": (60, 60, 60),
        "law": "KrWG",
    },
    "biomuell": {
        "name": "Biomuell",
        "color_name": "braun",
        "color": (33, 67, 101),
        "law": "BioAbfV",
    },
    "papier": {
        "name": "Papier",
        "color_name": "blau",
        "color": (200, 120, 30),
        "law": "VerpackG",
    },
    "gelbe_tonne": {
        "name": "Gelbe Tonne / Gelber Sack",
        "color_name": "gelb",
        "color": (40, 200, 230),
        "law": "VerpackG",
    },
    "altglas": {
        "name": "Altglas (Glascontainer)",
        "color_name": "weiss/gruen/braun",
        "color": (120, 180, 120),
        "law": "VerpackG",
    },
    "pfand": {
        "name": "Pfand",
        "color_name": "-",
        "color": (0, 215, 255),
        "law": "VerpackG",
    },
    "sondermuell": {
        "name": "Sondermuell",
        "color_name": "-",
        "color": (0, 0, 220),
        "law": "KrWG",
    },
    "elektroschrott": {
        "name": "Elektroschrott",
        "color_name": "-",
        "color": (180, 0, 180),
        "law": "ElektroG",
    },
}

# which bin each trashnet class belongs to
CLASS_TO_BIN = {
    "cardboard": "papier",
    "paper": "papier",
    "glass": "altglas",
    "metal": "gelbe_tonne",
    "plastic": "gelbe_tonne",
    "organic": "biomuell",
    "trash": "restmuell",
}


def get_bin(class_name):
    # unknown stuff goes to restmuell
    key = CLASS_TO_BIN.get(class_name.lower(), "restmuell")
    return BINS[key]


# which bin each coco object goes into
COCO_TO_BIN = {
    # electronics -> Elektroschrott
    "tv": "elektroschrott", "laptop": "elektroschrott", "mouse": "elektroschrott",
    "remote": "elektroschrott", "keyboard": "elektroschrott", "cell phone": "elektroschrott",
    "microwave": "elektroschrott", "oven": "elektroschrott", "toaster": "elektroschrott",
    "refrigerator": "elektroschrott", "hair drier": "elektroschrott", "clock": "elektroschrott",
    # food -> Biomuell
    "banana": "biomuell", "apple": "biomuell", "orange": "biomuell", "broccoli": "biomuell",
    "carrot": "biomuell", "sandwich": "biomuell", "hot dog": "biomuell", "pizza": "biomuell",
    "donut": "biomuell", "cake": "biomuell",
    # paper -> Papier
    "book": "papier",
    # packaging -> Gelbe Tonne
    "bottle": "gelbe_tonne",
    # drinking glasses / ceramics / cutlery etc -> Restmuell
    "wine glass": "restmuell", "cup": "restmuell", "bowl": "restmuell", "fork": "restmuell",
    "knife": "restmuell", "spoon": "restmuell", "scissors": "restmuell", "vase": "restmuell",
    "toothbrush": "restmuell", "teddy bear": "restmuell",
}


def coco_bin(class_name):
    key = COCO_TO_BIN.get(class_name.lower())
    if key is None:
        return None
    return BINS[key]
