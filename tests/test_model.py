import pytest
import torch
import torch.nn as nn
from model import NetArch

def test_parameter_count():
    """Test that model has less than 20000 parameters"""
    model = NetArch()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params <= 20000, f'Model has {total_params} parameters, which exceeds the limit of 20000'

def test_batch_norm_layers():
    """Test that model uses Batch Normalization layers"""
    model = NetArch()
    batch_norm_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(batch_norm_layers) > 0, 'Model does not use Batch Normalization'
    # Additional check: Each conv layer should be followed by BatchNorm
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(batch_norm_layers) >= len(conv_layers), 'Not all Conv layers are followed by BatchNorm'

def test_dropout_layers():
    """Test that model uses Dropout layers"""
    model = NetArch()
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dropout_layers) > 0, 'Model does not use Dropout'
    # Additional check: Model should have exactly 3 dropout layers as per architecture
    assert len(dropout_layers) == 3, f'Model should have exactly 3 dropout layers, but found {len(dropout_layers)}'

def test_gap_layer():
    """Test that model uses Global Average Pooling"""
    model = NetArch()
    gap_layers = [m for m in model.modules() if isinstance(m, nn.AvgPool2d)]
    assert len(gap_layers) > 0, 'Model does not use Global Average Pooling'
    # Additional check: GAP should be at the end with kernel size 3
    assert gap_layers[-1].kernel_size == 3, 'Final GAP layer should have kernel size 3'

def test_model_structure():
    """Test overall model structure and output shape"""
    model = NetArch()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # MNIST input shape
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f'Expected output shape (4, 10), got {output.shape}'

def test_model_forward_pass():
    """Test that model can perform a forward pass"""
    model = NetArch()
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f'Expected output shape (1, 10), got {output.shape}'
        assert not torch.isnan(output).any(), 'Model output contains NaN values'
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {str(e)}") 