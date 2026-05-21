# instrument_defaults.py

CTAX = {
    "supermirror": {
        "enabled": True,
        "m_value": 1.2
    },

    "collimator": {
        "1st_h": 80,
        "1st_v": 240,
        "2nd_h": 120,
        "2nd_v": 240,
        "3rd_h": 80,
        "3rd_v": 240,
        "4th_h": 120,
        "4th_v": 240,
    },

    "monochromator": {
        "crystal": "PG002",
        "mosaic_h": 30,
        "mosaic_v": 30,
        "width": 0.07,
        "height": 0.02,
        "thickness": 0.002,
        "vfocus": True,
        "hfocus": False,
        "blade_h": 1,
        "blade_v": 7
    },

    "analyzer": {
        "crystal": "PG002",
        "mosaic_h": 30,
        "mosaic_v": 30,
        "width": 0.022,
        "height": 0.022,
        "thickness": 0.002,
        "vfocus": True,
        "hfocus": False,
        "blade_h": 9,
        "blade_v": 7
    },

    "detector": {
        "width": 0.032,
        "height": 0.120
    },

    "configuration": {
        "energy_mode": "Ef fixed",
        "Ef": 4.8,
        "geometry": "W",
        "sign": "-+-"
    },

    "distance": {
        "L0": 53,
        "L1": 1.6,
        "L2": 1.06,
        "L3": 0.5
    },

    "beam": {
        "width": 0.07,
        "height": 0.14
    },
}

MANUAL = {
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

INSTRUMENTS = {
    "CTAX": CTAX,
    "Manual": MANUAL
}