# instrument_defaults.py
Manual = {
    "supermirror": {
        "enabled": False,
        "m_value": 1.0
    },

    "collimator": {
        "1st_h": 60,
        "1st_v": 180,
        "2nd_h": 60,
        "2nd_v": 180,
        "3rd_h": 60,
        "3rd_v": 180,
        "4th_h": 60,
        "4th_v": 180,
    },

    "monochromator": {
        "crystal": "PG002",
        "mosaic_h": 40,
        "mosaic_v": 40,
        "width": 0.08,
        "height": 0.03,
        "thickness": 0.002,
        "vfocus": False,
        "hfocus": False,
        "blade_h": 5,
        "blade_v": 5
    },

    "analyzer": {
        "crystal": "PG002",
        "mosaic_h": 40,
        "mosaic_v": 40,
        "width": 0.025,
        "height": 0.025,
        "thickness": 0.002,
        "vfocus": False,
        "hfocus": False,
        "blade_h": 5,
        "blade_v": 5
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
        "L2": 2.0,
        "L3": 2.0
    },

    "beam": {
        "width": 0.08,
        "height": 0.15
    },
}