# the knowledge base: known household items -> german bin.
# "en" is the phrase clip scores against, "de" is shown to the user.
# "cond" marks items where the bin depends on something we must ask about.
#   pfand   -> ask for a deposit logo, then pfand instead
#   greasy  -> if dirty/greasy it goes to restmuell
#   residue -> if not fully empty it goes to sondermuell

ITEMS = [
    # --- gelbe tonne (plastic / metal / composite packaging) ---
    {"en": "an empty plastic yogurt cup", "de": "Joghurtbecher", "bin": "gelbe_tonne"},
    {"en": "a plastic drink bottle", "de": "Plastikflasche", "bin": "gelbe_tonne", "cond": "pfand"},
    {"en": "a plastic shampoo bottle", "de": "Shampooflasche", "bin": "gelbe_tonne"},
    {"en": "a plastic detergent bottle", "de": "Waschmittelflasche", "bin": "gelbe_tonne"},
    {"en": "an empty chip bag", "de": "Chipstüte", "bin": "gelbe_tonne"},
    {"en": "a plastic shopping bag", "de": "Plastiktüte", "bin": "gelbe_tonne"},
    {"en": "a plastic food wrapper", "de": "Plastikverpackung", "bin": "gelbe_tonne"},
    {"en": "plastic cling film", "de": "Frischhaltefolie", "bin": "gelbe_tonne"},
    {"en": "bubble wrap", "de": "Luftpolsterfolie", "bin": "gelbe_tonne"},
    {"en": "a styrofoam food tray", "de": "Styroporschale", "bin": "gelbe_tonne"},
    {"en": "a sheet of aluminium foil", "de": "Alufolie", "bin": "gelbe_tonne"},
    {"en": "a drinks can", "de": "Getränkedose", "bin": "gelbe_tonne", "cond": "pfand"},
    {"en": "a food tin can", "de": "Konservendose", "bin": "gelbe_tonne"},
    {"en": "a metal bottle cap", "de": "Kronkorken", "bin": "gelbe_tonne"},
    {"en": "an aluminium food tray", "de": "Aluschale", "bin": "gelbe_tonne"},
    {"en": "a tetra pak drink carton", "de": "Getränkekarton (Tetrapak)", "bin": "gelbe_tonne"},
    {"en": "a milk carton", "de": "Milchkarton", "bin": "gelbe_tonne"},
    {"en": "a juice carton", "de": "Saftkarton", "bin": "gelbe_tonne"},
    {"en": "a coffee capsule", "de": "Kaffeekapsel", "bin": "gelbe_tonne"},
    {"en": "a toothpaste tube", "de": "Zahnpastatube", "bin": "gelbe_tonne"},
    {"en": "a plastic margarine tub", "de": "Margarinebecher", "bin": "gelbe_tonne"},
    {"en": "a plastic meat tray", "de": "Plastikschale", "bin": "gelbe_tonne"},
    {"en": "a plastic lid", "de": "Plastikdeckel", "bin": "gelbe_tonne"},
    {"en": "a plastic net bag for fruit", "de": "Obstnetz", "bin": "gelbe_tonne"},
    {"en": "an empty aerosol spray can", "de": "Spraydose", "bin": "gelbe_tonne", "cond": "residue"},

    # --- papier (paper / cardboard) ---
    {"en": "a cardboard box", "de": "Karton", "bin": "papier"},
    {"en": "a sheet of corrugated cardboard", "de": "Wellpappe", "bin": "papier"},
    {"en": "a newspaper", "de": "Zeitung", "bin": "papier"},
    {"en": "a magazine", "de": "Zeitschrift", "bin": "papier"},
    {"en": "a sheet of printer paper", "de": "Druckerpapier", "bin": "papier"},
    {"en": "a school notebook", "de": "Heft", "bin": "papier"},
    {"en": "a book", "de": "Buch", "bin": "papier"},
    {"en": "a paper bag", "de": "Papiertüte", "bin": "papier"},
    {"en": "a cardboard egg carton", "de": "Eierkarton", "bin": "papier"},
    {"en": "a paper envelope", "de": "Briefumschlag", "bin": "papier"},
    {"en": "a cereal box", "de": "Müslikarton", "bin": "papier"},
    {"en": "a pizza box", "de": "Pizzakarton", "bin": "papier", "cond": "greasy"},
    {"en": "a cardboard tube", "de": "Pappröhre", "bin": "papier"},
    {"en": "an advertising flyer", "de": "Prospekt", "bin": "papier"},
    {"en": "a shoe box", "de": "Schuhkarton", "bin": "papier"},
    {"en": "sheets of wrapping paper", "de": "Geschenkpapier", "bin": "papier", "cond": "greasy"},

    # --- altglas (bottle/jar glass only, by container) ---
    {"en": "a glass wine bottle", "de": "Weinflasche", "bin": "altglas"},
    {"en": "a glass beer bottle", "de": "Bierflasche", "bin": "altglas", "cond": "pfand"},
    {"en": "an empty glass bottle", "de": "Glasflasche", "bin": "altglas", "cond": "pfand"},
    {"en": "a glass jam jar", "de": "Marmeladenglas", "bin": "altglas"},
    {"en": "a glass pickle jar", "de": "Gurkenglas", "bin": "altglas"},
    {"en": "a glass food jar", "de": "Einmachglas", "bin": "altglas"},
    {"en": "a small glass baby food jar", "de": "Babygläschen", "bin": "altglas"},
    {"en": "a glass olive oil bottle", "de": "Ölflasche", "bin": "altglas"},

    # --- biomuell (organic) ---
    {"en": "a banana peel", "de": "Bananenschale", "bin": "biomuell"},
    {"en": "an apple core", "de": "Apfelgehäuse", "bin": "biomuell"},
    {"en": "an orange peel", "de": "Orangenschale", "bin": "biomuell"},
    {"en": "potato peelings", "de": "Kartoffelschalen", "bin": "biomuell"},
    {"en": "an eggshell", "de": "Eierschale", "bin": "biomuell"},
    {"en": "used coffee grounds", "de": "Kaffeesatz", "bin": "biomuell"},
    {"en": "a used tea bag", "de": "Teebeutel", "bin": "biomuell"},
    {"en": "vegetable scraps", "de": "Gemüsereste", "bin": "biomuell"},
    {"en": "leftover bread", "de": "Brotreste", "bin": "biomuell"},
    {"en": "wilted flowers", "de": "welke Blumen", "bin": "biomuell"},
    {"en": "grass clippings", "de": "Rasenschnitt", "bin": "biomuell"},
    {"en": "dead leaves", "de": "Laub", "bin": "biomuell"},
    {"en": "nut shells", "de": "Nussschalen", "bin": "biomuell"},
    {"en": "leftover meat scraps", "de": "Fleischreste", "bin": "biomuell"},
    {"en": "moldy food", "de": "verschimmeltes Essen", "bin": "biomuell"},

    # --- restmuell (not packaging, not recyclable) ---
    {"en": "a drinking glass", "de": "Trinkglas", "bin": "restmuell"},
    {"en": "a broken ceramic plate", "de": "kaputter Teller", "bin": "restmuell"},
    {"en": "a ceramic mug", "de": "Tasse", "bin": "restmuell"},
    {"en": "a porcelain cup", "de": "Porzellantasse", "bin": "restmuell"},
    {"en": "a mirror", "de": "Spiegel", "bin": "restmuell"},
    {"en": "an incandescent light bulb", "de": "Glühbirne", "bin": "restmuell"},
    {"en": "a toothbrush", "de": "Zahnbürste", "bin": "restmuell"},
    {"en": "a disposable razor", "de": "Einwegrasierer", "bin": "restmuell"},
    {"en": "a cigarette butt", "de": "Zigarettenstummel", "bin": "restmuell"},
    {"en": "a used diaper", "de": "Windel", "bin": "restmuell"},
    {"en": "a vacuum cleaner bag", "de": "Staubsaugerbeutel", "bin": "restmuell"},
    {"en": "a used paper tissue", "de": "benutztes Taschentuch", "bin": "restmuell"},
    {"en": "a ballpoint pen", "de": "Kugelschreiber", "bin": "restmuell"},
    {"en": "a cd disc", "de": "CD", "bin": "restmuell"},
    {"en": "a kitchen sponge", "de": "Spülschwamm", "bin": "restmuell"},
    {"en": "cat litter", "de": "Katzenstreu", "bin": "restmuell"},
    {"en": "a paper coffee cup", "de": "Pappbecher", "bin": "restmuell"},
    {"en": "a ceramic flower pot", "de": "Keramiktopf", "bin": "restmuell"},
    {"en": "a used cotton pad", "de": "Wattepad", "bin": "restmuell"},
    {"en": "an adhesive bandage", "de": "Pflaster", "bin": "restmuell"},
    {"en": "a plastic drinking straw", "de": "Trinkhalm", "bin": "restmuell"},
    {"en": "a broken plastic toy", "de": "kaputtes Plastikspielzeug", "bin": "restmuell"},
    {"en": "a wax candle", "de": "Kerze", "bin": "restmuell"},
    {"en": "a leather belt", "de": "Ledergürtel", "bin": "restmuell"},
    {"en": "a rubber glove", "de": "Gummihandschuh", "bin": "restmuell"},

    # --- sondermuell (hazardous) ---
    {"en": "a battery", "de": "Batterie", "bin": "sondermuell"},
    {"en": "a rechargeable battery pack", "de": "Akku", "bin": "sondermuell"},
    {"en": "a button cell battery", "de": "Knopfzelle", "bin": "sondermuell"},
    {"en": "a paint can with leftover paint", "de": "Farbeimer", "bin": "sondermuell"},
    {"en": "a bottle of motor oil", "de": "Motoröl", "bin": "sondermuell"},
    {"en": "a packet of medication pills", "de": "Medikamente", "bin": "sondermuell"},
    {"en": "a bottle of nail polish", "de": "Nagellack", "bin": "sondermuell"},
    {"en": "a tube of glue", "de": "Klebstofftube", "bin": "sondermuell"},
    {"en": "an energy saving lamp", "de": "Energiesparlampe", "bin": "sondermuell"},
    {"en": "a fluorescent tube", "de": "Leuchtstoffröhre", "bin": "sondermuell"},
    {"en": "a bottle of cleaning chemical", "de": "Reinigungsmittel", "bin": "sondermuell"},
    {"en": "a pesticide bottle", "de": "Pflanzenschutzmittel", "bin": "sondermuell"},

    # --- elektroschrott (e-waste) ---
    {"en": "a smartphone", "de": "Smartphone", "bin": "elektroschrott"},
    {"en": "a laptop", "de": "Laptop", "bin": "elektroschrott"},
    {"en": "a computer mouse", "de": "Computermaus", "bin": "elektroschrott"},
    {"en": "a computer keyboard", "de": "Tastatur", "bin": "elektroschrott"},
    {"en": "a charging cable", "de": "Ladekabel", "bin": "elektroschrott"},
    {"en": "a pair of headphones", "de": "Kopfhörer", "bin": "elektroschrott"},
    {"en": "a tv remote control", "de": "Fernbedienung", "bin": "elektroschrott"},
    {"en": "an electric toothbrush", "de": "elektrische Zahnbürste", "bin": "elektroschrott"},
    {"en": "a hair dryer", "de": "Föhn", "bin": "elektroschrott"},
    {"en": "a toaster", "de": "Toaster", "bin": "elektroschrott"},
    {"en": "an led light bulb", "de": "LED-Lampe", "bin": "elektroschrott"},
    {"en": "a power adapter", "de": "Netzteil", "bin": "elektroschrott"},
    {"en": "a usb stick", "de": "USB-Stick", "bin": "elektroschrott"},
    {"en": "a game controller", "de": "Controller", "bin": "elektroschrott"},
    {"en": "a pocket calculator", "de": "Taschenrechner", "bin": "elektroschrott"},
    {"en": "a wristwatch", "de": "Armbanduhr", "bin": "elektroschrott"},

    # --- altkleider (wearable textiles / shoes) ---
    {"en": "a t-shirt", "de": "T-Shirt", "bin": "altkleider"},
    {"en": "a pair of jeans", "de": "Jeans", "bin": "altkleider"},
    {"en": "a jacket", "de": "Jacke", "bin": "altkleider"},
    {"en": "a knitted sweater", "de": "Pullover", "bin": "altkleider"},
    {"en": "a pair of shoes", "de": "Schuhe", "bin": "altkleider"},
    {"en": "a pair of sneakers", "de": "Turnschuhe", "bin": "altkleider"},
    {"en": "a handbag", "de": "Handtasche", "bin": "altkleider"},
    {"en": "a bed sheet", "de": "Bettlaken", "bin": "altkleider"},
    {"en": "a bath towel", "de": "Handtuch", "bin": "altkleider"},
    {"en": "a scarf", "de": "Schal", "bin": "altkleider"},
]


# follow-up question for ambiguous items.
# "yes" -> use the yes bin, "no" -> keep the item's normal bin
CONDITIONS = {
    "pfand": {"q": "Hat der Artikel ein Pfand-Logo?", "yes": "pfand"},
    "greasy": {"q": "Ist es stark verschmutzt oder fettig?", "yes": "restmuell"},
    "residue": {"q": "Ist die Dose nicht restlos leer?", "yes": "sondermuell"},
}


# non-trash things so clip can say "not an item" instead of forcing a bin
DECOYS = [
    "a wooden desk",
    "a table surface",
    "a human hand",
    "a person",
    "a wall",
    "a floor",
    "a kitchen counter",
    "a carpet",
    "a plain background",
]
