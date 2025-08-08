# Pushing the project folder into github

# ICD Prediction Project - GitHub Push Guide

This guide explains how to push your local ICD prediction project to GitHub using Git and VS Code.

---

## ✅ Step-by-Step: Push Project to GitHub

### 🔹 Step 1: Open your project folder in VS Code
Make sure you're inside the folder you want to push:

```bash
cd C:/Users/vc/project/icd_automation/icd_production_code
```

Or open via menu:
```
File → Open Folder → icd_production_code
```

---

### 🔹 Step 2: Initialize Git (if not done already)

```bash
git init
```

If you see:
```
Reinitialized existing Git repository...
```
That’s okay — it means Git is already initialized.

---

### 🔹 Step 3: Set your Git identity (if not already set)

```bash
git config --global user.name "Azentyk"
git config --global user.email "azentyk01@gmail.com"
```

---

### 🔹 Step 4: Add all files to staging area

```bash
git add .
```

---

### 🔹 Step 5: Commit your changes

```bash
git commit -m "Initial commit for ICD prediction project"
```

---

### 🔹 Step 6: Add your GitHub remote repo

```bash
git remote add origin https://github.com/Azentyk/icd_prediction.git
```

If `origin` already exists, reset it:

```bash
git remote set-url origin https://github.com/Azentyk/icd_prediction.git
```

---

### 🔹 Step 7: Set main branch and push to GitHub

```bash
git branch -M main
git push -u origin main
```

If prompted, enter your **GitHub username** and **personal access token (PAT)** instead of your password.

🔐 Generate your PAT at:  
https://github.com/settings/tokens

---

## ✅ Done!

Visit your repository:  
👉 [https://github.com/Azentyk/icd_prediction](https://github.com/Azentyk/icd_prediction)

You should now see your project files.

---

## 🛠 Recommended: Add a `.gitignore` File

Create a `.gitignore` file to avoid pushing unnecessary files:

```gitignore
__pycache__/
*.pyc
.env
*.h5
*.parquet
.ipynb_checkpoints/
```

---

## 📌 Notes

- Use `git status` to see what’s staged or untracked.
- Use `git log` to view commit history.
- Use `git pull origin main --allow-unrelated-histories` if push fails due to remote history mismatch.

---

Happy Coding! 🚀  
Maintained by [Azentyk](https://github.com/Azentyk)


