"""
Input device controllers for teleoperation.
"""

import numpy as np
from typing import Optional, Union, Any
import threading
import time
from abc import ABC, abstractmethod

try:
    import pygame
except ImportError:
    pygame = None

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None


class BaseController(ABC):
    """Abstract base class for teleoperation controllers."""
    
    def __init__(self, action_space):
        """Initialize controller with action space."""
        self.action_space = action_space
        self.action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else 1
        self.action_low = action_space.low if hasattr(action_space, 'low') else -1.0
        self.action_high = action_space.high if hasattr(action_space, 'high') else 1.0
        self.running = False
        self.current_action = np.zeros(self.action_dim)
        
    @abstractmethod
    def get_action(self) -> Optional[np.ndarray]:
        """Get current action from controller."""
        pass
    
    @abstractmethod
    def wait_for_start(self) -> None:
        """Wait for user to indicate start of recording."""
        pass
    
    def normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Normalize action to environment's action space."""
        # Clip to [-1, 1]
        normalized = np.clip(raw_action, -1.0, 1.0)
        
        # Scale to action space
        scaled = self.action_low + (normalized + 1.0) * 0.5 * (self.action_high - self.action_low)
        
        return scaled
    
    def shutdown(self) -> None:
        """Clean up controller resources."""
        self.running = False


class KeyboardController(BaseController):
    """
    Keyboard-based teleoperation controller.
    
    Uses arrow keys for 2D movement, WASD for additional axes.
    """
    
    def __init__(self, action_space, sensitivity: float = 0.1):
        """
        Initialize keyboard controller.
        
        Args:
            action_space: Environment's action space
            sensitivity: Movement sensitivity (0-1)
        """
        super().__init__(action_space)
        self.sensitivity = sensitivity
        
        if pygame is None:
            raise ImportError("pygame is required for keyboard control. Install with: pip install pygame")
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Teleoperation Keyboard Control")
        self.clock = pygame.time.Clock()
        
        # Key mapping
        self.key_map = {
            pygame.K_LEFT: (0, -1),   # X-axis negative
            pygame.K_RIGHT: (0, 1),    # X-axis positive
            pygame.K_UP: (1, 1),       # Y-axis positive
            pygame.K_DOWN: (1, -1),    # Y-axis negative
            pygame.K_w: (2, 1),        # Z-axis positive
            pygame.K_s: (2, -1),       # Z-axis negative
            pygame.K_a: (3, -1),       # Rotation negative
            pygame.K_d: (3, 1),        # Rotation positive
            pygame.K_q: (4, -1),       # Gripper close
            pygame.K_e: (4, 1),        # Gripper open
        }
        
        self.recording = False
    
    def get_action(self) -> Optional[np.ndarray]:
        """Get action based on current keyboard state."""
        action = np.zeros(self.action_dim)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_SPACE:
                    self.recording = not self.recording
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Update action based on pressed keys
        for key, (axis, direction) in self.key_map.items():
            if axis < self.action_dim and keys[key]:
                action[axis] += direction * self.sensitivity
        
        # Display status
        self._update_display(action)
        
        # Control loop rate
        self.clock.tick(30)
        
        return self.normalize_action(action) if self.recording else np.zeros(self.action_dim)
    
    def wait_for_start(self) -> None:
        """Wait for space key to start recording."""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("User closed window")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.recording = True
                        waiting = False
            
            self._update_display(np.zeros(self.action_dim))
            self.clock.tick(30)
    
    def _update_display(self, action: np.ndarray) -> None:
        """Update pygame display with current status."""
        self.screen.fill((0, 0, 0))
        
        # Draw status text
        font = pygame.font.Font(None, 36)
        status_text = "RECORDING" if self.recording else "PAUSED"
        status_color = (0, 255, 0) if self.recording else (255, 255, 0)
        text = font.render(status_text, True, status_color)
        self.screen.blit(text, (150, 50))
        
        # Draw action values
        font_small = pygame.font.Font(None, 24)
        for i, val in enumerate(action[:min(5, len(action))]):
            axis_text = f"Axis {i}: {val:.3f}"
            text = font_small.render(axis_text, True, (255, 255, 255))
            self.screen.blit(text, (50, 120 + i * 30))
        
        # Draw instructions
        inst_font = pygame.font.Font(None, 20)
        instructions = [
            "Arrow Keys: X/Y movement",
            "W/S: Z movement",
            "A/D: Rotation",
            "Q/E: Gripper",
            "SPACE: Start/Stop",
            "ESC: Finish"
        ]
        for i, inst in enumerate(instructions):
            text = inst_font.render(inst, True, (128, 128, 128))
            self.screen.blit(text, (220, 120 + i * 25))
        
        pygame.display.flip()
    
    def shutdown(self) -> None:
        """Clean up pygame resources."""
        super().shutdown()
        pygame.quit()


class SpaceMouseController(BaseController):
    """
    3D SpaceMouse controller for precise 6-DOF control.
    """
    
    def __init__(self, action_space, deadzone: float = 0.05):
        """
        Initialize SpaceMouse controller.
        
        Args:
            action_space: Environment's action space
            deadzone: Deadzone threshold for input
        """
        super().__init__(action_space)
        self.deadzone = deadzone
        
        if pyspacemouse is None:
            raise ImportError("pyspacemouse is required. Install with: pip install pyspacemouse")
        
        # Open SpaceMouse connection
        self.device = pyspacemouse.open()
        if self.device is None:
            raise RuntimeError("No SpaceMouse device found")
        
        self.state = None
        self.recording = False
        
        # Start reading thread
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.running = True
        self.read_thread.start()
    
    def _read_loop(self) -> None:
        """Continuously read SpaceMouse state."""
        while self.running:
            try:
                self.state = pyspacemouse.read()
                time.sleep(0.01)
            except Exception as e:
                print(f"SpaceMouse read error: {e}")
                break
    
    def get_action(self) -> Optional[np.ndarray]:
        """Get action from SpaceMouse state."""
        if self.state is None:
            return np.zeros(self.action_dim)
        
        # Check buttons
        if self.state.buttons[0]:  # Left button - toggle recording
            self.recording = not self.recording
            time.sleep(0.2)  # Debounce
        
        if self.state.buttons[1]:  # Right button - stop
            return None
        
        if not self.recording:
            return np.zeros(self.action_dim)
        
        # Extract 6-DOF values
        action = np.zeros(self.action_dim)
        
        # Translation (x, y, z)
        if self.action_dim > 0:
            action[0] = self._apply_deadzone(self.state.x / 350.0)
        if self.action_dim > 1:
            action[1] = self._apply_deadzone(self.state.y / 350.0)
        if self.action_dim > 2:
            action[2] = self._apply_deadzone(self.state.z / 350.0)
        
        # Rotation (roll, pitch, yaw)
        if self.action_dim > 3:
            action[3] = self._apply_deadzone(self.state.roll / 350.0)
        if self.action_dim > 4:
            action[4] = self._apply_deadzone(self.state.pitch / 350.0)
        if self.action_dim > 5:
            action[5] = self._apply_deadzone(self.state.yaw / 350.0)
        
        # Gripper (if available)
        if self.action_dim > 6:
            # Use button for gripper
            action[6] = 1.0 if self.state.buttons[2] else -1.0
        
        return self.normalize_action(action)
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to input value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def wait_for_start(self) -> None:
        """Wait for button press to start."""
        print("Press left SpaceMouse button to start...")
        while not self.recording:
            if self.state and self.state.buttons[0]:
                self.recording = True
                time.sleep(0.2)  # Debounce
            time.sleep(0.1)
    
    def shutdown(self) -> None:
        """Clean up SpaceMouse connection."""
        super().shutdown()
        if hasattr(self, 'device'):
            pyspacemouse.close()


class VRController(BaseController):
    """
    VR controller for immersive teleoperation.
    
    Supports Oculus/Meta Quest and SteamVR controllers.
    """
    
    def __init__(self, action_space, controller_hand: str = "right"):
        """
        Initialize VR controller.
        
        Args:
            action_space: Environment's action space
            controller_hand: Which hand controller to use ('left' or 'right')
        """
        super().__init__(action_space)
        self.hand = controller_hand
        
        # Note: Actual VR implementation would require OpenXR or SteamVR bindings
        # This is a simplified mock implementation
        self.position = np.zeros(3)
        self.rotation = np.zeros(4)  # Quaternion
        self.trigger = 0.0
        self.grip = 0.0
        self.buttons = {}
        
        print(f"VR Controller initialized (mock mode) - using {controller_hand} hand")
        self.recording = False
    
    def get_action(self) -> Optional[np.ndarray]:
        """Get action from VR controller state."""
        # In a real implementation, this would read from OpenXR/SteamVR
        # For now, return simulated values
        
        action = np.zeros(self.action_dim)
        
        if not self.recording:
            return action
        
        # Map VR controller pose to robot action
        if self.action_dim >= 3:
            action[:3] = self.position * 0.1  # Scale position
        
        if self.action_dim >= 6:
            # Convert quaternion to euler angles for rotation axes
            euler = self._quat_to_euler(self.rotation)
            action[3:6] = euler * 0.1
        
        if self.action_dim >= 7:
            # Map trigger to gripper
            action[6] = self.trigger * 2.0 - 1.0  # Convert [0,1] to [-1,1]
        
        return self.normalize_action(action)
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to euler angles."""
        # Simplified conversion (would use proper library in production)
        w, x, y, z = quat
        
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return np.array([roll, pitch, yaw])
    
    def wait_for_start(self) -> None:
        """Wait for VR trigger press to start."""
        print(f"Pull {self.hand} trigger to start recording...")
        # In real implementation, would wait for actual trigger press
        import time
        time.sleep(2)  # Simulate wait
        self.recording = True
    
    def update_state(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        trigger: float,
        grip: float,
        buttons: dict
    ) -> None:
        """
        Update VR controller state.
        
        This would be called by the VR runtime in a real implementation.
        """
        self.position = position
        self.rotation = rotation
        self.trigger = trigger
        self.grip = grip
        self.buttons = buttons
        
        # Check for recording toggle
        if buttons.get('A', False) or buttons.get('X', False):
            self.recording = not self.recording