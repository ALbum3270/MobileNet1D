# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-19

### 🎉 Initial Release

#### Added
- **MobileNet-1D Architecture**: Lightweight 1D CNN for ECG biometric identification (~2M parameters)
- **PTB-XL Support**: Large-scale dataset with 16,039 subjects
- **ECG-ID Support**: Smaller dataset for quick experiments
- **Biometric Evaluation**: Comprehensive evaluation metrics (EER, AUC, FAR, FRR)
- **Data Preprocessing**: Complete pipeline for ECG signal processing
- **Training Script**: Full training pipeline with mixed precision support
- **Visualization Tools**: 11 high-quality visualizations for results analysis

#### Features
- ✅ **State-of-the-art Performance**: EER 2.08%, AUC 99.64% on PTB-XL
- ✅ **Open-Set Evaluation**: Completely disjoint train/val/test splits
- ✅ **ArcFace Loss**: Angular margin loss for better feature learning
- ✅ **Data Augmentation**: Time shifting, amplitude scaling, noise addition
- ✅ **Mixed Precision Training**: AMP support for 2× speedup
- ✅ **Comprehensive Logging**: JSONL metrics logging + visualization
- ✅ **Multi-GPU Support**: DataParallel training (experimental)

#### Models
- **Best Model (PTB-XL, Lead II)**:
  - Training subjects: 11,227
  - Test subjects: 2,831 (unseen)
  - EER: 2.08%
  - AUC: 99.64%
  - Model size: 74 MB

#### Datasets
- **PTB-XL**: 16,039 subjects, 314,653 samples
  - 70% training, 12.4% validation, 17.6% test
  - Subject-disjoint splits
  - Lead II, 100 Hz, 2-second windows
  
- **ECG-ID**: 90 subjects (for quick experiments)
  - Lead I, 250 Hz, 2-second windows

#### Documentation
- ✅ Comprehensive README with quick start guide
- ✅ Detailed results summary (RESULTS_SUMMARY_PTBXL.md)
- ✅ Visualization guide (visualizations/README.md)
- ✅ Code comments and docstrings

#### Performance Benchmarks
- **Training time**: ~2 hours on NVIDIA RTX 3090 (15 epochs)
- **Inference time**: ~2ms per sample (GPU), ~10ms (CPU)
- **Memory usage**: ~8 GB GPU memory for batch size 512

---

## [Unreleased]

### Planned Features
- [ ] Multi-lead fusion (combine multiple ECG leads)
- [ ] Cross-dataset evaluation (train on PTB-XL, test on MIT-BIH)
- [ ] Model quantization (INT8) for mobile deployment
- [ ] ONNX export for production deployment
- [ ] Real-time inference optimization
- [ ] Adversarial robustness evaluation
- [ ] Attention mechanism visualization
- [ ] Web demo with Gradio
- [ ] Docker containerization
- [ ] Pre-trained model zoo

### Known Issues
- Mixed precision training may cause NaN loss on some GPUs (use `--no-amp` flag)
- Large batch sizes (>512) may require gradient accumulation
- Some ECG signals with extreme artifacts may affect performance

---

## Version History

### [1.0.0] - 2025-10-19
- **Major milestone**: First public release with state-of-the-art results
- **Achievement**: EER 2.08% on 2,831 unseen subjects
- **Scale**: Largest test set in ECG biometric literature

---

## Comparison with Previous Work

| Version | Dataset | Test Subjects | EER | AUC | Notes |
|---------|---------|---------------|-----|-----|-------|
| **v1.0.0** | PTB-XL | **2,831** | **2.08%** | **99.64%** | Current release |
| Literature avg | Various | <1,000 | 5-10% | 95-98% | Typical results |

**Improvement**: 
- EER reduced by **72.3%**
- Test scale increased by **3-5×**
- Generalization gap reduced to **<0.05%**

---

## How to Update

### From scratch
```bash
git clone https://github.com/yourusername/MobileNet1D-ECG.git
cd MobileNet1D-ECG
pip install -r requirements.txt
```

### Update to latest version
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## Migration Guide

### For new users
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: See [Dataset](#dataset) section in README
3. Run preprocessing: `python preprocess.py`
4. Start training: `python train.py --config config_ptbxl.yaml`

### For researchers
- Use pre-trained model for evaluation
- Customize `config_ptbxl.yaml` for your experiments
- Check `eval_biometric.py` for evaluation protocols

---

## Contributors

- **Lead Developer**: Album (album3270@gmail.com)
- **Contributors**: See [GitHub contributors](https://github.com/ALbum3270/MobileNet1D/graphs/contributors)

---

## Acknowledgments

Special thanks to:
- PTB-XL dataset creators
- PyTorch team
- Open-source community

---

**For detailed usage instructions, see [README.md](README.md)**

**For contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)**

