"""
ui_constants.py
===============

Constants and mapping for the Pro War Room UI, including hero roles 
and radar chart attribute mappings.
"""

# Mapping of Hero Class/Role to 5-axis attributes for Radar Chart
# Axes: Damage, Durability, Control, Mobility, Utility
ROLE_ATTRIBUTES = {
    "Marksman": {"Damage": 5, "Durability": 1, "Control": 2, "Mobility": 3, "Utility": 2},
    "Tank": {"Damage": 1, "Durability": 5, "Control": 5, "Mobility": 2, "Utility": 3},
    "Fighter": {"Damage": 4, "Durability": 4, "Control": 3, "Mobility": 3, "Utility": 1},
    "Mage": {"Damage": 5, "Durability": 1, "Control": 4, "Mobility": 2, "Utility": 3},
    "Assassin": {"Damage": 5, "Durability": 1, "Control": 2, "Mobility": 5, "Utility": 1},
    "Support": {"Damage": 2, "Durability": 2, "Control": 3, "Mobility": 3, "Utility": 5},
}

# General heuristic mapping from Class to Primary Lane
CLASS_TO_LANE = {
    "Marksman": "Gold Lane",
    "Tank": "Roam",
    "Fighter": "EXP Lane",
    "Mage": "Mid Lane",
    "Assassin": "Jungle",
    "Support": "Roam"
}

DEFAULT_ATTRS = {"Damage": 1, "Durability": 1, "Control": 1, "Mobility": 1, "Utility": 1}

CSS_STYLE = """
<style>
    /* Main Background and Text */
    .main {
        background-color: #050a14;
        color: #e0e6ed;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a1120;
        border-right: 1px solid #1c2c4c;
    }
    
    /* Header Polish */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #0d1b33 0%, #050a14 100%);
        border: 1px solid #1c2c4c;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }

    /* Hero Cards */
    .hero-container {
        background: #0d1b33;
        border: 1px solid #1c2c4c;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .hero-container:hover {
        transform: scale(1.02);
        border-color: #00d4ff;
    }

    /* Tabs and Buttons */
    .stButton>button {
        background-color: #00d4ff;
        color: #050a14;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #008fb3;
        color: white;
    }
</style>
"""
