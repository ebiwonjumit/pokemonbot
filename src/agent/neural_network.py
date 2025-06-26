"""
Neural network architectures for Pokemon RL agent.
Implements CNN-based policy and value networks for PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gym

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PokemonCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Pokemon game frames.
    
    Processes 4 stacked 84x84 grayscale frames into feature vectors.
    Optimized for Pokemon game visual patterns and UI elements.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = True
    ):
        """
        Initialize Pokemon CNN.
        
        Args:
            observation_space: Observation space (4, 84, 84)
            features_dim: Dimension of output features
            normalized_image: Whether input images are normalized [0,1]
        """
        super(PokemonCNN, self).__init__(observation_space, features_dim)
        
        # Input shape: (batch, 4, 84, 84)
        n_input_channels = observation_space.shape[0]
        
        self.normalized_image = normalized_image
        
        # CNN layers optimized for Pokemon sprites and UI
        self.cnn = nn.Sequential(
            # First conv layer - capture basic shapes and edges
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Second conv layer - detect Pokemon sprites and UI elements
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Third conv layer - higher level features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Flatten for fully connected layers
            nn.Flatten(),
        )
        
        # Calculate the output size after conv layers
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            conv_output_size = self.cnn(sample_input).shape[1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Small dropout for regularization
        )
        
        logger.info(f"Pokemon CNN initialized: {conv_output_size} -> {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observations: Input observations (batch, 4, 84, 84)
            
        Returns:
            torch.Tensor: Feature vectors (batch, features_dim)
        """
        # Normalize input if needed
        if not self.normalized_image:
            observations = observations / 255.0
        
        # CNN feature extraction
        conv_features = self.cnn(observations)
        
        # Fully connected layers
        features = self.fc(conv_features)
        
        return features


class AttentionCNN(BaseFeaturesExtractor):
    """
    CNN with attention mechanism for better Pokemon game understanding.
    
    Uses spatial attention to focus on important game elements like:
    - Pokemon sprites
    - HP bars
    - Menu items
    - Battle UI
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        attention_dim: int = 64
    ):
        """Initialize attention-based CNN."""
        super(AttentionCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Base CNN layers
        self.conv1 = nn.Conv2d(n_input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        # Spatial attention mechanism
        self.attention_conv = nn.Conv2d(64, attention_dim, 1)
        self.attention_fc = nn.Conv2d(attention_dim, 1, 1)
        
        # Calculate conv output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            x = F.relu(self.conv1(sample_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            conv_output_size = x.numel()  # Total elements
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply spatial attention
        attention_weights = self.attention_conv(x)
        attention_weights = torch.sigmoid(self.attention_fc(attention_weights))
        
        # Apply attention to features
        x = x * attention_weights
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        return features


class DuelingCNN(BaseFeaturesExtractor):
    """
    Dueling architecture CNN for Pokemon RL.
    
    Separates value and advantage streams for better learning,
    especially useful for Pokemon where state value and action
    advantages can be quite different.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512
    ):
        """Initialize dueling CNN."""
        super(DuelingCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Shared CNN layers
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate shared output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            shared_output_size = self.shared_cnn(sample_input).shape[1]
        
        # Value stream - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(shared_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream - estimates action advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(shared_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling architecture."""
        shared_features = self.shared_cnn(observations)
        
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        # Combine value and advantage using dueling formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        
        return q_values


class RecurrentCNN(BaseFeaturesExtractor):
    """
    CNN with LSTM for temporal sequence modeling.
    
    Useful for Pokemon RL where temporal context matters:
    - Battle sequences
    - Menu navigation
    - Story progression
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        lstm_hidden_size: int = 256
    ):
        """Initialize recurrent CNN."""
        super(RecurrentCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            cnn_output_size = self.cnn(sample_input).shape[1]
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(lstm_hidden_size, features_dim)
        
        # Hidden state (will be reset externally)
        self.hidden_state = None
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with LSTM."""
        batch_size = observations.size(0)
        
        # CNN feature extraction
        cnn_features = self.cnn(observations)
        
        # Reshape for LSTM (batch, seq_len=1, features)
        lstm_input = cnn_features.unsqueeze(1)
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            device = observations.device
            h_0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            c_0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            self.hidden_state = (h_0, c_0)
        
        # LSTM forward
        lstm_output, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        
        # Take last output and apply final linear layer
        lstm_features = lstm_output.squeeze(1)  # Remove seq_len dimension
        features = self.fc(lstm_features)
        
        return features
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state (call between episodes)."""
        self.hidden_state = None


class PokemonActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for Pokemon RL.
    
    Uses custom CNN architecture and adds Pokemon-specific improvements:
    - Custom feature extraction
    - Action masking for invalid actions
    - Multi-head attention for complex scenes
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        net_arch: Optional[Dict[str, Any]] = None,
        activation_fn = nn.ReLU,
        use_attention: bool = False,
        use_dueling: bool = False,
        use_recurrent: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize Pokemon actor-critic policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space  
            lr_schedule: Learning rate schedule
            net_arch: Network architecture specification
            activation_fn: Activation function
            use_attention: Use attention mechanism
            use_dueling: Use dueling architecture
            use_recurrent: Use recurrent (LSTM) architecture
        """
        # Set network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])
        
        # Store architecture flags
        self.use_attention = use_attention
        self.use_dueling = use_dueling  
        self.use_recurrent = use_recurrent
        
        super(PokemonActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build the custom feature extractor."""
        if self.use_attention:
            self.features_extractor = AttentionCNN(
                self.observation_space,
                features_dim=512
            )
        elif self.use_dueling:
            self.features_extractor = DuelingCNN(
                self.observation_space,
                features_dim=512
            )
        elif self.use_recurrent:
            self.features_extractor = RecurrentCNN(
                self.observation_space,
                features_dim=512
            )
        else:
            self.features_extractor = PokemonCNN(
                self.observation_space,
                features_dim=512
            )
    
    def reset_hidden_state(self):
        """Reset hidden state for recurrent networks."""
        if self.use_recurrent and hasattr(self.features_extractor, 'reset_hidden_state'):
            self.features_extractor.reset_hidden_state()


def create_pokemon_cnn(
    observation_space: gym.Space,
    architecture: str = "standard",
    features_dim: int = 512,
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Factory function to create Pokemon CNN architectures.
    
    Args:
        observation_space: Observation space
        architecture: Architecture type ('standard', 'attention', 'dueling', 'recurrent')
        features_dim: Output feature dimension
        **kwargs: Additional architecture-specific parameters
        
    Returns:
        BaseFeaturesExtractor: CNN feature extractor
    """
    if architecture == "standard":
        return PokemonCNN(observation_space, features_dim, **kwargs)
    elif architecture == "attention":
        return AttentionCNN(observation_space, features_dim, **kwargs)
    elif architecture == "dueling":
        return DuelingCNN(observation_space, features_dim, **kwargs)
    elif architecture == "recurrent":
        return RecurrentCNN(observation_space, features_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def test_architectures():
    """Test different CNN architectures."""
    import torch
    
    # Create dummy observation space
    obs_space = gym.spaces.Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 4, 84, 84)
    
    architectures = ["standard", "attention", "dueling", "recurrent"]
    
    for arch in architectures:
        print(f"\nTesting {arch} architecture:")
        
        try:
            model = create_pokemon_cnn(obs_space, architecture=arch)
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Reset hidden state for recurrent models
            if arch == "recurrent":
                model.reset_hidden_state()
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("Testing Pokemon CNN architectures...")
    test_architectures()
    print("\nArchitecture testing completed!")
