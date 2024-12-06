import pytest
import torch
from model import NetArch

@pytest.fixture
def model():
    """Fixture to provide a fresh model instance for each test"""
    return NetArch()

@pytest.fixture
def sample_input():
    """Fixture to provide a sample input tensor"""
    return torch.randn(1, 1, 28, 28)  # Single MNIST image shape

@pytest.fixture
def batch_input():
    """Fixture to provide a batch of input tensors"""
    return torch.randn(4, 1, 28, 28)  # Batch of MNIST images 