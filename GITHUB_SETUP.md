# GitHub Setup Instructions

## After creating your GitHub repository, run these commands:

### Configure Git (if not already done):
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Add the remote repository and push:

Replace `YOUR_USERNAME` with your actual GitHub username:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/facematric.git
git branch -M main
git push -u origin main
```

### Or if you prefer SSH:
```powershell
git remote add origin git@github.com:YOUR_USERNAME/facematric.git
git branch -M main
git push -u origin main
```

## Current Status:
✅ Git repository initialized
✅ All files committed (except celebrity_dataset which is too large)
✅ README.md created with project documentation
✅ .gitignore configured to exclude large datasets and temporary files

## Next Steps After Pushing:

1. Go to your repository on GitHub
2. Verify all files are uploaded correctly
3. Consider adding:
   - GitHub Actions for CI/CD
   - Issues and project boards
   - GitHub Pages for documentation
   - Branch protection rules

## Note about the Dataset:

The `celebrity_dataset/` folder (275 directories with images) is excluded from git.
If you need to share the dataset:
- Consider using Git LFS for large files
- Or upload to a cloud storage service (Google Drive, AWS S3, etc.)
- Or provide download instructions in the README
