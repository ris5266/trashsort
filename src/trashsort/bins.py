BINS = {
    "restmuell": {
        "name": "Restmuell",
        "color": (60, 60, 60),
        "law": "KrWG",
    },
    "biomuell": {
        "name": "Biomuell",
        "color": (33, 67, 101),
        "law": "BioAbfV",
    },
    "papier": {
        "name": "Papier",
        "color": (200, 120, 30),
        "law": "VerpackG",
    },
    "gelbe_tonne": {
        "name": "Gelbe Tonne / Gelber Sack",
        "color": (40, 200, 230),
        "law": "VerpackG",
    },
    "altglas": {
        "name": "Altglas (Glascontainer)",
        "color": (120, 180, 120),
        "law": "VerpackG",
    },
    "sondermuell": {
        "name": "Sondermuell",
        "color": (0, 0, 220),
        "law": "KrWG",
    },
    "elektroschrott": {
        "name": "Elektroschrott",
        "color": (180, 0, 180),
        "law": "ElektroG",
    },
    "altkleider": {
        "name": "Altkleider",
        "color": (200, 200, 200),
        "law": "KrWG",
    },
}

BIN_NAME_TO_KEY = {info["name"]: key for key, info in BINS.items()}

# map each material class to its usual bin
MATERIAL_TO_BIN = {
    "cardboard": "papier",
    "paper": "papier",
    "glass": "altglas",
    "metal": "gelbe_tonne",
    "plastic": "gelbe_tonne",
    "organic": "biomuell",
    "trash": "restmuell",
}

def get_bin_for_material(material):
    # use restmuell when the model returns an unknown material
    key = MATERIAL_TO_BIN.get(material.lower(), "restmuell")
    return BINS[key]
