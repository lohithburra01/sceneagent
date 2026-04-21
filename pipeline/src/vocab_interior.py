"""Open interior vocabulary. Not derived from InteriorGS labels — any
similar scene (residential, commercial bar, restaurant) should segment
into one of these classes."""
INTERIOR_VOCAB: list[str] = [
    # structural — still predicted, filtered out at eval
    "wall", "floor", "ceiling", "window", "door", "stairs", "column",
    # residential furniture
    "bed", "sofa", "couch", "armchair", "chair", "stool", "bench",
    "table", "dining table", "coffee table", "side table", "desk",
    "nightstand", "wardrobe", "cabinet", "shelf", "bookshelf",
    "dresser", "console", "rug", "curtain", "mirror", "painting",
    "picture frame", "wall clock", "lamp", "floor lamp", "table lamp",
    "chandelier", "pendant light", "spotlight", "downlight",
    "ceiling fan", "television", "speaker", "radiator", "fireplace",
    # kitchen + bath
    "sink", "toilet", "bathtub", "shower", "stove", "oven",
    "microwave", "refrigerator", "dishwasher", "coffee maker",
    "kettle", "pot", "pan",
    # decor + small items
    "plant", "potted plant", "vase", "candle", "book", "tray",
    "ornament", "sculpture", "box", "basket", "bowl", "cup", "mug",
    "plate", "glass", "wine glass", "bottle", "wine bottle",
    "jar", "fruit", "chocolate", "placemat",
    # commercial / bar
    "bar counter", "bar stool", "high chair", "cash register",
    "billboard", "menu board", "signboard", "shelf of bottles",
    "pillow", "cushion", "throw blanket", "doormat",
]
