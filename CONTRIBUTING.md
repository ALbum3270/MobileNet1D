# Contributing to MobileNet-1D ECG Biometric

Thank you for your interest in contributing! 🎉

We welcome contributions from the community to make this project better. This document provides guidelines for contributing.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of:
- Age, body size, disability
- Ethnicity, gender identity and expression
- Level of experience, nationality
- Personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behavior includes:**
- ✅ Using welcoming and inclusive language
- ✅ Being respectful of differing viewpoints
- ✅ Gracefully accepting constructive criticism
- ✅ Focusing on what is best for the community

**Unacceptable behavior includes:**
- ❌ Trolling, insulting/derogatory comments
- ❌ Public or private harassment
- ❌ Publishing others' private information
- ❌ Other unprofessional conduct

---

## 🤝 How Can I Contribute?

### 1. Reporting Bugs 🐛

**Before submitting a bug report:**
- Check existing [issues](https://github.com/ALbum3270/MobileNet1D/issues)
- Try to reproduce the bug with the latest version
- Collect relevant information (OS, Python version, GPU, etc.)

**When submitting a bug report, include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details

**Template:**
```markdown
## Bug Description
[Clear description of the bug]

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- PyTorch: [e.g., 1.10.0]
- CUDA: [e.g., 11.3]
- GPU: [e.g., RTX 3090]

## Additional Context
[Any other relevant information]
```

---

### 2. Suggesting Enhancements 💡

**Before suggesting an enhancement:**
- Check if it's already implemented in the latest version
- Search existing feature requests

**When suggesting an enhancement, include:**
- Clear use case
- Expected benefits
- Potential implementation approach
- Examples from other projects (if applicable)

**Template:**
```markdown
## Feature Request

**Problem**: [What problem does this solve?]

**Proposed Solution**: [How should it work?]

**Alternatives**: [Other approaches considered?]

**Benefits**: [Why is this useful?]

**Example**: [Code or diagram if possible]
```

---

### 3. Contributing Code 💻

We welcome code contributions! Here's how:

#### Types of Contributions

**🌟 High Priority:**
- Bug fixes
- Performance improvements
- Documentation improvements
- Test coverage
- Cross-dataset evaluation
- Model quantization

**🔮 Nice to Have:**
- New architectures
- Advanced augmentation techniques
- Visualization improvements
- Tutorial notebooks
- Docker support

#### Before You Start

1. **Open an issue** to discuss your idea
2. **Wait for feedback** before writing code
3. **Fork the repository** and create a branch

---

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/MobileNet1D.git
cd MobileNet1D

# Add upstream remote
git remote add upstream https://github.com/ALbum3270/MobileNet1D.git
```

### 2. Create Environment

```bash
# Create virtual environment
conda create -n ecg-bio-dev python=3.8
conda activate ecg-bio-dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

---

## 🔄 Pull Request Process

### 1. Make Your Changes

```bash
# Write code
# ...

# Format code
black model.py train.py

# Check code style
flake8 model.py train.py

# Run tests
pytest tests/
```

### 2. Commit Your Changes

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example:**
```bash
git add .
git commit -m "feat: add multi-lead fusion support

- Implement multi-lead MobileNet architecture
- Add fusion strategies (early, late, hybrid)
- Update config to support multiple leads
- Add tests for multi-lead dataloader

Closes #42"
```

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then:
1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Fill in the PR template
4. Wait for review

### 4. PR Template

```markdown
## Description
[Brief description of changes]

## Motivation
[Why is this change needed?]

## Changes Made
- [ ] Change 1
- [ ] Change 2
- [ ] ...

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new features
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Changelog updated

## Related Issues
Closes #[issue_number]
```

---

## 📏 Coding Standards

### Python Style Guide

**Follow PEP 8** with some modifications:
- Line length: 100 characters (not 79)
- Use `black` for auto-formatting
- Use type hints when possible

**Example:**
```python
from typing import Tuple, Optional
import torch
import torch.nn as nn


class MobileNet1D(nn.Module):
    """
    1D MobileNet for ECG biometric identification.
    
    Args:
        num_classes: Number of subjects
        embedding_dim: Embedding dimension (default: 128)
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # ... rest of initialization
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # ... implementation
        return logits
```

### Code Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports
from model import MobileNet1D
from utils import seed_everything
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `MobileNet1D`)
- **Functions**: `snake_case` (e.g., `load_dataset`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- **Private**: prefix with `_` (e.g., `_internal_function`)

---

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

**Test file structure:**
```python
# tests/test_model.py
import pytest
import torch
from model import MobileNet1D


class TestMobileNet1D:
    """Test suite for MobileNet1D model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MobileNet1D(num_classes=90)
        assert model.num_classes == 90
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = MobileNet1D(num_classes=90)
        x = torch.randn(8, 1, 500)
        output = model(x)
        assert output.shape == (8, 90)
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_batch_sizes(self, batch_size):
        """Test different batch sizes."""
        model = MobileNet1D(num_classes=90)
        x = torch.randn(batch_size, 1, 500)
        output = model(x)
        assert output.shape[0] == batch_size
```

### Test Coverage

Aim for:
- ✅ **80%+** overall code coverage
- ✅ **100%** for critical functions (loss, metrics)
- ✅ Edge cases and error handling

---

## 📚 Documentation

### Code Documentation

**Use Google-style docstrings:**

```python
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    EER is the point where FAR equals FRR on the DET curve.
    
    Args:
        scores: Similarity scores, shape (n_pairs,)
        labels: Ground truth labels (1 for genuine, 0 for impostor), shape (n_pairs,)
    
    Returns:
        eer: Equal Error Rate as a float in [0, 1]
    
    Raises:
        ValueError: If scores and labels have different lengths
    
    Example:
        >>> scores = np.array([0.9, 0.8, 0.3, 0.2])
        >>> labels = np.array([1, 1, 0, 0])
        >>> eer = compute_eer(scores, labels)
        >>> print(f"EER: {eer:.2%}")
        EER: 0.00%
    """
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length")
    
    # ... implementation
    return eer
```

### README Updates

When adding features, update:
- [ ] Main README.md (feature description)
- [ ] Quick Start section (if applicable)
- [ ] Configuration examples
- [ ] Performance tables (if improved)

---

## 🎯 Areas for Contribution

### High Priority

1. **Cross-Dataset Evaluation** 🔥
   - Train on PTB-XL, test on MIT-BIH
   - Evaluate domain adaptation

2. **Model Optimization** ⚡
   - Quantization (INT8, FP16)
   - ONNX export
   - TensorRT optimization

3. **Multi-Lead Fusion** 🔬
   - Early fusion (concatenate signals)
   - Late fusion (combine predictions)
   - Attention-based fusion

4. **Robustness Analysis** 🛡️
   - Adversarial attacks
   - Noise robustness
   - Signal quality impact

### Medium Priority

5. **Advanced Architectures**
   - Transformer-based models
   - Hybrid CNN-RNN
   - Neural Architecture Search

6. **Visualization Tools**
   - Attention maps
   - Embedding space (t-SNE)
   - Interactive demo (Gradio)

7. **Documentation**
   - Tutorial notebooks
   - Video tutorials
   - API documentation

### Nice to Have

8. **Deployment**
   - Docker container
   - REST API
   - Mobile app (TFLite)

9. **Benchmarking**
   - Comparison with commercial systems
   - Speed benchmarks
   - Memory profiling

---

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: album3270@gmail.com (for private matters)

### Response Time

- Issues: Usually within 2-3 days
- Pull Requests: Usually within 1 week
- Emergency bugs: Within 24 hours

---

## 🙏 Recognition

### Contributors

All contributors will be:
- Listed in [README.md](README.md)
- Acknowledged in releases
- Credited in papers (if significant contribution)

### Attribution

If you use code from other sources:
- ✅ Add proper attribution in code comments
- ✅ Include license information
- ✅ Mention in PR description

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🎉 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort! 🙏

**Questions?** Open an issue or reach out!

**Happy Coding!** 🚀

