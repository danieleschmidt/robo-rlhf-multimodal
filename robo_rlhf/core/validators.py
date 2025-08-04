"""
Input validation utilities for robo-rlhf-multimodal.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import re

from robo_rlhf.core.exceptions import ValidationError


def validate_observations(
    observations: Dict[str, np.ndarray],
    required_modalities: Optional[List[str]] = None,
    max_sequence_length: int = 10000,
    image_modalities: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Validate observation data.
    
    Args:
        observations: Dictionary of observation arrays
        required_modalities: Required modality keys
        max_sequence_length: Maximum allowed sequence length
        image_modalities: List of modalities that should be images
        
    Returns:
        Validated observations
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(observations, dict):
        raise ValidationError("Observations must be a dictionary", field="observations")
    
    if not observations:
        raise ValidationError("Observations cannot be empty", field="observations")
    
    # Check required modalities
    if required_modalities:
        missing = set(required_modalities) - set(observations.keys())
        if missing:
            raise ValidationError(
                f"Missing required modalities: {missing}",
                field="observations",
                details={"missing_modalities": list(missing)}
            )
    
    # Validate each modality
    sequence_length = None
    for modality, data in observations.items():
        if not isinstance(data, np.ndarray):
            raise ValidationError(
                f"Observation data must be numpy arrays, got {type(data)} for {modality}",
                field=f"observations.{modality}"
            )
        
        if data.size == 0:
            raise ValidationError(
                f"Empty observation data for modality: {modality}",
                field=f"observations.{modality}"
            )
        
        # Check sequence length consistency
        current_length = data.shape[0]
        if sequence_length is None:
            sequence_length = current_length
        elif sequence_length != current_length:
            raise ValidationError(
                f"Inconsistent sequence lengths: {sequence_length} vs {current_length} for {modality}",
                field="observations",
                details={"modality": modality}
            )
        
        if current_length > max_sequence_length:
            raise ValidationError(
                f"Sequence too long: {current_length} > {max_sequence_length} for {modality}",
                field=f"observations.{modality}"
            )
        
        # Validate image modalities
        if image_modalities and modality in image_modalities:
            if len(data.shape) not in [3, 4]:  # [H, W, C] or [T, H, W, C]
                raise ValidationError(
                    f"Invalid image shape for {modality}: {data.shape}",
                    field=f"observations.{modality}"
                )
    
    return observations


def validate_actions(
    actions: np.ndarray,
    expected_dim: Optional[int] = None,
    action_bounds: Optional[tuple] = None,
    max_sequence_length: int = 10000
) -> np.ndarray:
    """
    Validate action data.
    
    Args:
        actions: Action array
        expected_dim: Expected action dimension
        action_bounds: (min, max) bounds for actions
        max_sequence_length: Maximum allowed sequence length
        
    Returns:
        Validated actions
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(actions, np.ndarray):
        raise ValidationError(
            f"Actions must be numpy array, got {type(actions)}",
            field="actions"
        )
    
    if actions.size == 0:
        raise ValidationError("Actions cannot be empty", field="actions")
    
    if len(actions.shape) != 2:
        raise ValidationError(
            f"Actions must be 2D [sequence_length, action_dim], got shape {actions.shape}",
            field="actions"
        )
    
    sequence_length, action_dim = actions.shape
    
    if sequence_length > max_sequence_length:
        raise ValidationError(
            f"Action sequence too long: {sequence_length} > {max_sequence_length}",
            field="actions"
        )
    
    if expected_dim is not None and action_dim != expected_dim:
        raise ValidationError(
            f"Wrong action dimension: expected {expected_dim}, got {action_dim}",
            field="actions"
        )
    
    # Check for NaN or inf values
    if not np.isfinite(actions).all():
        raise ValidationError("Actions contain NaN or inf values", field="actions")
    
    # Check bounds
    if action_bounds is not None:
        min_bound, max_bound = action_bounds
        if np.any(actions < min_bound) or np.any(actions > max_bound):
            raise ValidationError(
                f"Actions outside bounds [{min_bound}, {max_bound}]",
                field="actions",
                details={
                    "min_value": float(np.min(actions)),
                    "max_value": float(np.max(actions))
                }
            )
    
    return actions


def validate_preferences(
    preferences: List[Dict[str, Any]],
    required_fields: Optional[List[str]] = None,
    valid_choices: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Validate preference data.
    
    Args:
        preferences: List of preference dictionaries
        required_fields: Required fields in each preference
        valid_choices: Valid preference choices
        
    Returns:
        Validated preferences
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(preferences, list):
        raise ValidationError(
            f"Preferences must be a list, got {type(preferences)}",
            field="preferences"
        )
    
    if not preferences:
        raise ValidationError("Preferences cannot be empty", field="preferences")
    
    required_fields = required_fields or ["annotator_id", "choice", "confidence"]
    valid_choices = valid_choices or ["a", "b", "equal", "unclear"]
    
    for i, pref in enumerate(preferences):
        if not isinstance(pref, dict):
            raise ValidationError(
                f"Preference {i} must be a dictionary, got {type(pref)}",
                field=f"preferences[{i}]"
            )
        
        # Check required fields
        missing = set(required_fields) - set(pref.keys())
        if missing:
            raise ValidationError(
                f"Preference {i} missing required fields: {missing}",
                field=f"preferences[{i}]",
                details={"missing_fields": list(missing)}
            )
        
        # Validate choice
        if "choice" in pref and pref["choice"] not in valid_choices:
            raise ValidationError(
                f"Invalid choice '{pref['choice']}' in preference {i}",
                field=f"preferences[{i}].choice",
                value=pref["choice"]
            )
        
        # Validate confidence
        if "confidence" in pref:
            confidence = pref["confidence"]
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValidationError(
                    f"Confidence must be a number between 0 and 1, got {confidence}",
                    field=f"preferences[{i}].confidence",
                    value=confidence
                )
    
    return preferences


def validate_episode_id(episode_id: str) -> str:
    """
    Validate episode ID format.
    
    Args:
        episode_id: Episode identifier
        
    Returns:
        Validated episode ID
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(episode_id, str):
        raise ValidationError(
            f"Episode ID must be a string, got {type(episode_id)}",
            field="episode_id"
        )
    
    if not episode_id.strip():
        raise ValidationError("Episode ID cannot be empty", field="episode_id")
    
    # Check format (alphanumeric, underscores, hyphens, max 100 chars)
    if not re.match(r'^[a-zA-Z0-9_-]+$', episode_id):
        raise ValidationError(
            "Episode ID can only contain alphanumeric characters, underscores, and hyphens",
            field="episode_id",
            value=episode_id
        )
    
    if len(episode_id) > 100:
        raise ValidationError(
            f"Episode ID too long: {len(episode_id)} > 100 characters",
            field="episode_id",
            value=episode_id
        )
    
    return episode_id


def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate file/directory path.
    
    Args:
        path: File or directory path
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(
            f"Path must be string or Path object, got {type(path)}",
            field="path"
        )
    
    path = Path(path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}", field="path", value=str(path))
    
    if path.exists():
        if must_be_file and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}", field="path", value=str(path))
        
        if must_be_dir and not path.is_dir():
            raise ValidationError(f"Path is not a directory: {path}", field="path", value=str(path))
    
    if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationError(
            f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}",
            field="path",
            value=str(path)
        )
    
    return path


def validate_numeric(
    value: Union[int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    must_be_positive: bool = False,
    must_be_integer: bool = False
) -> Union[int, float]:
    """
    Validate numeric value.
    
    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        must_be_positive: Whether value must be positive
        must_be_integer: Whether value must be an integer
        
    Returns:
        Validated numeric value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"Value must be numeric, got {type(value)}",
            field="value",
            value=value
        )
    
    if not np.isfinite(value):
        raise ValidationError("Value must be finite (not NaN or inf)", field="value", value=value)
    
    if must_be_integer and not isinstance(value, int):
        if not float(value).is_integer():
            raise ValidationError(
                f"Value must be an integer, got {value}",
                field="value",
                value=value
            )
    
    if must_be_positive and value <= 0:
        raise ValidationError(
            f"Value must be positive, got {value}",
            field="value",
            value=value
        )
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Value {value} below minimum {min_value}",
            field="value",
            value=value
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Value {value} above maximum {max_value}",
            field="value",
            value=value
        )
    
    return value