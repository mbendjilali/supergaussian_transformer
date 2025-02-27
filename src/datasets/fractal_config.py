import numpy as np


########################################################################
#                                Labels                                #
########################################################################

FRACTAL_NUM_CLASSES = 9

MAP_ID_TO_TRAINID = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 9:4, 17: 5, 64: 6, 65: 7, 66: 8}

CLASS_NAMES = [
    'Unclassified',
    'Ground',
    'Vegetation',
    'Building',
    'Water',
    'Bridge deck',
    'Permanent structure',
    "Artifact",
    "Synthetic",
    ]

CLASS_COLORS = np.asarray([
    [255, 255, 255],  # Unclassified
    [255, 128,   0],  # Ground
    [  0, 255,   0],  # Vegetation
    [255,   0,   0],  # Building
    [  0, 225, 225],  # Water
    [255, 255,   0],  # Bridge
    [128,   8,  64],  # Permanent structure
    [ 64,   0, 128],  # Artifact
    [255,   0, 255],  # Synthetic
    ])

