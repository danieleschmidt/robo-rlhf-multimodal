"""
Human preference collection system for RLHF.
"""

from robo_rlhf.preference.pair_generator import PreferencePairGenerator
from robo_rlhf.preference.models import PreferencePair, PreferenceLabel, PreferenceChoice, Segment

# Optional server import
try:
    from robo_rlhf.preference.server import PreferenceServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False
    PreferenceServer = None
    print("Warning: PreferenceServer not available (missing web dependencies)")

__all__ = [
    "PreferencePairGenerator",
    "PreferenceServer",
    "PreferencePair",
    "PreferenceLabel",
]