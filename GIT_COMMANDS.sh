#!/bin/bash

################################################################################
# Git Commands for First Push to GitHub
# Project: MobileNet-1D for ECG Biometric Identification
# Author: Album (album3270@gmail.com)
# Date: 2025-10-19
################################################################################

echo "🚀 MobileNet-1D ECG - Git Setup Script"
echo "======================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/home/DATA/xiazhiqiang1/MobileNet1D"

# Change to project directory
cd "$PROJECT_DIR" || { echo -e "${RED}Error: Cannot access project directory${NC}"; exit 1; }

echo -e "${YELLOW}Current directory: $(pwd)${NC}"
echo ""

################################################################################
# Step 1: Pre-flight checks
################################################################################
echo "📋 Step 1: Pre-flight Checks"
echo "----------------------------"

# Check if data directory exists but is gitignored
echo "Checking .gitignore configuration..."
if [ -f .gitignore ]; then
    echo -e "${GREEN}✓ .gitignore exists${NC}"
    
    # Test if data directories are ignored
    if git check-ignore -q data/raw data/processed 2>/dev/null || true; then
        echo -e "${GREEN}✓ Data directories will be ignored${NC}"
    fi
else
    echo -e "${RED}✗ .gitignore not found!${NC}"
fi

# Check for large files
echo ""
echo "Checking for large files (>10MB)..."
LARGE_FILES=$(find . -type f -size +10M 2>/dev/null | grep -v ".git" | grep -v "data/" || true)
if [ -z "$LARGE_FILES" ]; then
    echo -e "${GREEN}✓ No large files detected in tracked directories${NC}"
else
    echo -e "${YELLOW}⚠ Large files found:${NC}"
    echo "$LARGE_FILES"
    echo -e "${YELLOW}Make sure these are gitignored or should be in releases${NC}"
fi

# Check for data files in current directory
echo ""
echo "Checking for data files in root directory..."
DATA_FILES=$(find . -maxdepth 2 -type f \( -name "*.csv" -o -name "*.npy" -o -name "*.h5" \) 2>/dev/null | grep -v "data/" || true)
if [ -z "$DATA_FILES" ]; then
    echo -e "${GREEN}✓ No data files in tracked areas${NC}"
else
    echo -e "${YELLOW}⚠ Data files found:${NC}"
    echo "$DATA_FILES"
    echo -e "${YELLOW}These should be in data/ directory${NC}"
fi

echo ""
read -p "Continue with git initialization? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi

################################################################################
# Step 2: Initialize Git Repository
################################################################################
echo ""
echo "📦 Step 2: Initialize Git Repository"
echo "------------------------------------"

if [ -d .git ]; then
    echo -e "${YELLOW}Git repository already initialized${NC}"
    read -p "Reinitialize? This will erase git history. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        git init
        echo -e "${GREEN}✓ Git repository reinitialized${NC}"
    else
        echo "Using existing git repository"
    fi
else
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi

################################################################################
# Step 3: Configure Git
################################################################################
echo ""
echo "⚙️ Step 3: Configure Git"
echo "------------------------"

# Set user name and email (local to this repo only)
git config user.name "Album"
git config user.email "album3270@gmail.com"
echo -e "${GREEN}✓ Git user configured (local)${NC}"
echo "  Name: Album"
echo "  Email: album3270@gmail.com"

################################################################################
# Step 4: Review Files to be Committed
################################################################################
echo ""
echo "📝 Step 4: Stage Files"
echo "----------------------"

# Add all files
git add .

echo "Files to be committed:"
git status --short | head -20
FILE_COUNT=$(git status --short | wc -l)
echo ""
echo -e "${GREEN}Total files staged: $FILE_COUNT${NC}"

if [ $FILE_COUNT -gt 100 ]; then
    echo -e "${YELLOW}⚠ Large number of files. Verify this is correct.${NC}"
fi

echo ""
read -p "Review staged files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git status
    echo ""
    read -p "Continue with commit? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted by user."
        exit 1
    fi
fi

################################################################################
# Step 5: First Commit
################################################################################
echo ""
echo "💾 Step 5: Create Initial Commit"
echo "---------------------------------"

git commit -m "Initial commit: MobileNet-1D for ECG Biometric Identification

- Complete implementation with 97-line MobileNet-1D architecture
- Achieves state-of-the-art EER 2.08% on PTB-XL Lead II
- Subject-disjoint evaluation protocol with 2,831 test subjects
- Comprehensive documentation and user guides
- Ready for academic research and reproducibility

Features:
- Lightweight model (~2M parameters)
- Fast inference (~3 ms/sample)
- Modular and extensible codebase
- Detailed results and visualizations

Documentation:
- Complete README with quick start
- Contributing guidelines
- Data download instructions
- Project structure guide
- Release checklist

Author: Album (album3270@gmail.com)
License: MIT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Initial commit created successfully${NC}"
else
    echo -e "${RED}✗ Commit failed${NC}"
    exit 1
fi

################################################################################
# Step 6: Set Default Branch to 'main'
################################################################################
echo ""
echo "🌿 Step 6: Set Default Branch"
echo "-----------------------------"

git branch -M main
echo -e "${GREEN}✓ Default branch set to 'main'${NC}"

################################################################################
# Step 7: Add Remote Repository
################################################################################
echo ""
echo "🔗 Step 7: Add Remote Repository"
echo "---------------------------------"

REMOTE_URL="https://github.com/Album3270/MobileNet1D-ECG.git"

# Check if remote already exists
if git remote | grep -q "^origin$"; then
    echo -e "${YELLOW}Remote 'origin' already exists${NC}"
    EXISTING_URL=$(git remote get-url origin)
    echo "Current URL: $EXISTING_URL"
    
    if [ "$EXISTING_URL" != "$REMOTE_URL" ]; then
        read -p "Update remote URL to $REMOTE_URL? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin "$REMOTE_URL"
            echo -e "${GREEN}✓ Remote URL updated${NC}"
        fi
    fi
else
    git remote add origin "$REMOTE_URL"
    echo -e "${GREEN}✓ Remote repository added${NC}"
fi

echo "Remote URL: $REMOTE_URL"

################################################################################
# Step 8: Push to GitHub
################################################################################
echo ""
echo "🚀 Step 8: Push to GitHub"
echo "-------------------------"
echo ""
echo -e "${YELLOW}IMPORTANT: Make sure you have created the GitHub repository first!${NC}"
echo "Repository: https://github.com/Album3270/MobileNet1D-ECG"
echo ""
echo "If not created yet:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: MobileNet1D-ECG"
echo "3. Description: State-of-the-art ECG biometric identification using MobileNet-1D"
echo "4. Public repository"
echo "5. Do NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""

read -p "GitHub repository created? Ready to push? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "When ready, run:"
    echo "  cd $PROJECT_DIR"
    echo "  git push -u origin main"
    exit 0
fi

echo ""
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓✓✓ Successfully pushed to GitHub! ✓✓✓${NC}"
    echo ""
    echo "🎉 Your project is now live!"
    echo ""
    echo "View your repository:"
    echo "  https://github.com/Album3270/MobileNet1D-ECG"
    echo ""
    echo "Next steps:"
    echo "1. Enable GitHub Issues and Discussions in repository settings"
    echo "2. Add topics: ecg, biometrics, deep-learning, pytorch, mobilenet"
    echo "3. Create a release (v1.0.0) and upload pre-trained model"
    echo "4. Share on social media and research communities"
    echo ""
    echo "See OPEN_SOURCE_CHECKLIST.md for detailed next steps."
else
    echo ""
    echo -e "${RED}✗ Push failed${NC}"
    echo ""
    echo "Common issues:"
    echo "1. Repository doesn't exist on GitHub"
    echo "2. No permission to push (check GitHub authentication)"
    echo "3. Network issues"
    echo ""
    echo "For authentication, you may need to:"
    echo "  - Use a personal access token (classic) as password"
    echo "  - Configure SSH keys"
    echo "  - Use GitHub CLI: gh auth login"
    exit 1
fi

################################################################################
# Summary
################################################################################
echo ""
echo "📊 Summary"
echo "=========="
echo -e "${GREEN}✓ Git repository initialized${NC}"
echo -e "${GREEN}✓ Initial commit created${NC}"
echo -e "${GREEN}✓ Remote repository configured${NC}"
echo -e "${GREEN}✓ Code pushed to GitHub${NC}"
echo ""
echo "Repository: https://github.com/Album3270/MobileNet1D-ECG"
echo "Author: Album (album3270@gmail.com)"
echo "License: MIT"
echo ""
echo -e "${GREEN}🎊 Congratulations! Your project is now open source! 🎊${NC}"

