"""Tests for the config module."""

import pytest
from ai_image_detector.config import AppConfig


class TestAppConfig:
    """Test the AppConfig dataclass."""
    
    def test_app_config_defaults(self):
        """Test default values for AppConfig."""
        config = AppConfig()
        
        assert config.size == 224
        assert config.mean == (0.5, 0.5, 0.5)
        assert config.std == (0.5, 0.5, 0.5)
        assert config.threshold == 0.5
        assert config.ensemble_weights == (0.4, 0.6)
    
    def test_app_config_custom_values(self):
        """Test custom values for AppConfig."""
        config = AppConfig(
            size=256,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            threshold=0.7,
            ensemble_weights=(0.3, 0.7)
        )
        
        assert config.size == 256
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)
        assert config.threshold == 0.7
        assert config.ensemble_weights == (0.3, 0.7)
    
    def test_app_config_immutable(self):
        """Test that AppConfig is immutable (frozen dataclass)."""
        config = AppConfig()
        
        with pytest.raises(AttributeError):
            config.size = 256
        
        with pytest.raises(AttributeError):
            config.threshold = 0.8
    
    def test_app_config_tuple_types(self):
        """Test that tuple fields maintain correct types."""
        config = AppConfig()
        
        assert isinstance(config.mean, tuple)
        assert isinstance(config.std, tuple)
        assert isinstance(config.ensemble_weights, tuple)
        
        assert len(config.mean) == 3
        assert len(config.std) == 3
        assert len(config.ensemble_weights) == 2
        
        assert all(isinstance(x, float) for x in config.mean)
        assert all(isinstance(x, float) for x in config.std)
        assert all(isinstance(x, float) for x in config.ensemble_weights)
    
    def test_app_config_equality(self):
        """Test AppConfig equality comparison."""
        config1 = AppConfig()
        config2 = AppConfig()
        config3 = AppConfig(size=256)
        
        assert config1 == config2
        assert config1 != config3
    
    def test_app_config_hash(self):
        """Test that AppConfig is hashable (frozen dataclass)."""
        config1 = AppConfig()
        config2 = AppConfig()
        config3 = AppConfig(size=256)
        
        # Should be hashable
        hash1 = hash(config1)
        hash2 = hash(config2)
        hash3 = hash(config3)
        
        assert isinstance(hash1, int)
        assert isinstance(hash2, int)
        assert isinstance(hash3, int)
        
        # Equal objects should have same hash
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_app_config_repr(self):
        """Test AppConfig string representation."""
        config = AppConfig()
        repr_str = repr(config)
        
        assert "AppConfig" in repr_str
        assert "size=224" in repr_str
        assert "threshold=0.5" in repr_str
    
    def test_app_config_edge_values(self):
        """Test AppConfig with edge case values."""
        # Test with extreme values
        config = AppConfig(
            size=1,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            threshold=0.0,
            ensemble_weights=(0.0, 1.0)
        )
        
        assert config.size == 1
        assert config.mean == (0.0, 0.0, 0.0)
        assert config.std == (1.0, 1.0, 1.0)
        assert config.threshold == 0.0
        assert config.ensemble_weights == (0.0, 1.0)
        
        # Test with maximum values
        config2 = AppConfig(
            size=1024,
            mean=(1.0, 1.0, 1.0),
            std=(0.1, 0.1, 0.1),
            threshold=1.0,
            ensemble_weights=(1.0, 0.0)
        )
        
        assert config2.size == 1024
        assert config2.mean == (1.0, 1.0, 1.0)
        assert config2.std == (0.1, 0.1, 0.1)
        assert config2.threshold == 1.0
        assert config2.ensemble_weights == (1.0, 0.0)
    
    def test_app_config_validation(self):
        """Test AppConfig with various valid input types."""
        # Test with different float representations
        config = AppConfig(
            size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            threshold=0.5,
            ensemble_weights=(0.4, 0.6)
        )
        
        assert config.size == 224
        assert config.mean == (0.5, 0.5, 0.5)
        assert config.std == (0.5, 0.5, 0.5)
        assert config.threshold == 0.5
        assert config.ensemble_weights == (0.4, 0.6)
    
    def test_app_config_usage_in_dict(self):
        """Test AppConfig can be used as dictionary key (hashable)."""
        config1 = AppConfig()
        config2 = AppConfig()
        config3 = AppConfig(size=256)
        
        config_dict = {
            config1: "default",
            config3: "custom"
        }
        
        assert config_dict[config1] == "default"
        assert config_dict[config2] == "default"  # Equal to config1
        assert config_dict[config3] == "custom"
    
    def test_app_config_usage_in_set(self):
        """Test AppConfig can be used in sets (hashable)."""
        config1 = AppConfig()
        config2 = AppConfig()
        config3 = AppConfig(size=256)
        
        config_set = {config1, config2, config3}
        
        # Should only have 2 unique configs (config1 == config2)
        assert len(config_set) == 2
