# instrument_defaults.py
Manual = {
    "supermirror": {
        "enabled": False,
        "m_value": 1.2
    },

    "collimator": {
        "1st_h": 80,
        "1st_v": 120,
        "2nd_h": 80,
        "2nd_v": 120,
        "3rd_h": 80,
        "3rd_v": 120,
        "4th_h": 80,
        "4th_v": 120,
    },

    "monochromator": {
        "crystal": "PG002",
        "mosaic_h": 30,
        "mosaic_v": 30,
        "width": 0.02,
        "height": 0.020,
        "thickness": 0.002,
        "vfocus": False,
        "hfocus": False,
        "blade_h": 7,
        "blade_v": 7
    },

    "analyzer": {
        "crystal": "PG002",
        "mosaic_h": 60,
        "mosaic_v": 60,
        "width": 0.020,
        "height": 0.020,
        "thickness": 0.002,
        "vfocus": False,
        "hfocus": False,
        "blade_h": 7,
        "blade_v": 7
    },

    "detector": {
        "width": 0.05,
        "height": 0.10
    },

    "configuration": {
        "energy_mode": "Ef fixed",
        "Ef": 5.0,
        "geometry": "W",
        "sign": "+-+"
    },

    "distance": {
        "L0": 50,
        "L1": 2.0,
        "L2": 1.0,
        "L3": 1.0
    },

    "beam": {
        "width": 0.08,
        "height": 0.15
    },
}