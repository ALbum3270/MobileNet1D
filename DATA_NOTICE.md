# 📦 Data Notice

## ⚠️ Important: Data Not Included

This repository **does NOT include the ECG datasets** used for training and evaluation. The data must be downloaded separately from the original sources.

---

## 📊 Required Datasets

### PTB-XL Dataset (Primary)

**Source**: PhysioNet  
**URL**: https://physionet.org/content/ptb-xl/1.0.3/  
**License**: Creative Commons Attribution 4.0 International Public License  
**Size**: ~16 GB  
**Citation**:
```bibtex
@article{wagner2020ptbxl,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={154},
  year={2020},
  publisher={Nature Publishing Group}
}
```

### ECG-ID Dataset (Optional, for quick experiments)

**Source**: PhysioNet  
**URL**: https://physionet.org/content/ecgiddb/1.0.0/  
**License**: Open Data Commons Attribution License v1.0  
**Size**: ~100 MB (90 subjects)

---

## 📥 How to Download

### Method 1: Direct Download (Recommended)

1. Visit the PhysioNet page: https://physionet.org/content/ptb-xl/1.0.3/
2. Create a free account if you don't have one
3. Download the dataset files
4. Extract to `data/raw/ptbxl/`

### Method 2: Command Line (with wget)

```bash
# Note: Requires PhysioNet account
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
     https://physionet.org/files/ptb-xl/1.0.3/
     
# Move to correct location
mv physionet.org/files/ptb-xl/1.0.3/* data/raw/ptbxl/
```

### Method 3: Using WFDB tools

```bash
pip install wfdb
python -c "
import wfdb
# Download PTB-XL database
wfdb.dl_database('ptb-xl', dl_dir='data/raw/ptbxl')
"
```

---

## 📁 Expected Directory Structure

After downloading and preprocessing:

```
data/
├── raw/                          # Raw data (gitignored)
│   └── ptbxl/
│       ├── records100/           # 100 Hz ECG recordings
│       ├── records500/           # 500 Hz ECG recordings
│       └── ptbxl_database.csv    # Metadata
│
└── processed/                    # Preprocessed data (gitignored)
    └── ptbxl/
        └── fixed_II/
            ├── train.csv         # Training metadata
            ├── val.csv           # Validation metadata
            ├── test.csv          # Test metadata
            └── segments/         # Preprocessed ECG segments (.npy)
```

---

## 🔒 Data Privacy & Copyright

### Why Data is NOT Included

1. **Size**: PTB-XL is ~16 GB, too large for GitHub
2. **Copyright**: Data belongs to PhysioNet/PTB
3. **License**: Requires users to accept PhysioNet license
4. **Best Practice**: Users should download from official source

### Data Usage Terms

- ✅ **Academic Research**: Allowed
- ✅ **Non-commercial Use**: Allowed  
- ✅ **Citation Required**: Always cite original paper
- ❌ **Commercial Use**: Requires special permission from PTB
- ❌ **Redistribution**: Not allowed without permission

---

## 🚫 What is Gitignored

The following are automatically ignored by `.gitignore`:

```gitignore
# All raw data
data/raw/

# All preprocessed data
data/processed/

# Data files
*.csv
*.npy
*.npz
*.h5
*.hdf5

# Model checkpoints (too large)
*.pt
*.pth
*.ckpt
```

---

## ✅ What IS Included in This Repository

- ✅ **Code**: All Python scripts
- ✅ **Configs**: YAML configuration files
- ✅ **Documentation**: README, guides, tutorials
- ✅ **Visualizations**: Result figures (PNG)
- ❌ **Data**: NOT included (download separately)
- ❌ **Models**: NOT included (use GitHub Releases)

---

## 📋 Quick Start Checklist

To get started with this project:

- [ ] Clone this repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Download PTB-XL from PhysioNet
- [ ] Extract to `data/raw/ptbxl/`
- [ ] Run preprocessing (`python preprocess.py`)
- [ ] Download pre-trained model from Releases (optional)
- [ ] Start training or evaluation

---

## 🆘 Troubleshooting

### Issue: Cannot download from PhysioNet

**Solution**: 
1. Create a free PhysioNet account
2. Sign the data use agreement
3. Use your credentials to download

### Issue: Download is very slow

**Solution**:
1. Use a download manager (e.g., aria2c)
2. Download during off-peak hours
3. Consider using a VPS closer to servers

### Issue: Not enough disk space

**Solution**:
- PTB-XL requires ~16 GB raw + ~50 GB processed
- Use external hard drive or cloud storage
- Consider using only a subset for testing

---

## 📞 Questions?

If you have questions about data access:

- **PhysioNet Support**: https://physionet.org/about/contact/
- **Project Issues**: https://github.com/Album3270/MobileNet1D-ECG/issues
- **Email**: album3270@gmail.com

---

## 🙏 Acknowledgments

We are grateful to:

- **PhysioNet** for hosting the datasets
- **PTB** (Physikalisch-Technische Bundesanstalt) for creating PTB-XL
- All contributors to the ECG biometric research community

---

**Remember**: Always cite the original dataset papers when using this code! 📚

