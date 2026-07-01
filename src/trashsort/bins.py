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
    "altkleider": {
        "name": "Altkleider",
        "color_name": "-",
        "color": (200, 200, 200),
        "law": "KrWG",
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
