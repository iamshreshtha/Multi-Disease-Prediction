import numpy as np

def safe_input(value):
    """Convert missing or invalid input safely"""
    try:
        return float(value)
    except:
        return 0.0


def risk_level(prob):
    """Convert probability → medical risk level"""
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"