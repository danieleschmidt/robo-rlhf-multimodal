"""Basic tests to verify package structure."""

import pytest

import robo_rlhf


def test_package_import():
    """Test that package imports correctly."""
    assert robo_rlhf.__version__ == "0.1.0"
    assert robo_rlhf.__author__ == "Daniel Schmidt"


def test_package_exports():
    """Test that main classes are available."""
    expected_exports = [
        "TeleOpCollector",
        "PreferencePairGenerator", 
        "PreferenceServer",
        "MultimodalRLHF",
        "VisionLanguageActor",
    ]
    
    for export in expected_exports:
        assert hasattr(robo_rlhf, export), f"Missing export: {export}"