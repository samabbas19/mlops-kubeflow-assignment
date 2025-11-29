# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `mlops-kubeflow-assignment`
5. Description: "MLOps Pipeline with DVC, MLflow, and CI/CD"
6. Select: **Public**
7. Do NOT initialize with README (we already have one)
8. Click "Create repository"

## Step 2: Push Local Code to GitHub

After creating the repository, run these commands in your terminal:

```bash
# Navigate to your project directory
cd "c:\Users\PC\Desktop\sem 7\MLops\A4"

# Add GitHub remote (replace <your-username> with your GitHub username)
git remote add origin https://github.com/<your-username>/mlops-kubeflow-assignment.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all files including:
   - README.md
   - src/
   - components/
   - .github/workflows/
   - requirements.txt
   - etc.

## Step 4: Enable GitHub Actions

1. Go to the "Actions" tab in your GitHub repository
2. GitHub Actions should automatically detect the workflow file
3. You may need to enable workflows if prompted
4. Wait for the workflow to run (it should start automatically)

## Step 5: Take Screenshots

Now take the required screenshots for your submission:

### Task 1:
- Screenshot of GitHub repo showing file structure
- Screenshot of terminal with `dvc status` and `dvc push`

### Task 2:
- Screenshot of [src/pipeline_components.py](src/pipeline_components.py) (open in GitHub)
- Screenshot of [components/](components/) directory

### Task 3:
- Run `mlflow ui` and take screenshot of the UI
- Screenshot showing pipeline run details and metrics

### Task 4:
- Screenshot of GitHub Actions workflow run (green checkmark)

### Task 5:
- Screenshot of GitHub repository main page with README

## Step 6: Create Screenshots Folder

```bash
# Create screenshots directory
mkdir screenshots

# Place all your screenshots here with names:
# - task1_repo_structure.png
# - task1_dvc_status.png
# - task2_pipeline_components.png
# - task2_components_directory.png
# - task3_mlflow_ui.png
# - task3_pipeline_graph.png
# - task3_metrics_output.png
# - task4_github_actions.png
# - task5_github_repo.png
```

## Step 7: Create PDF Document

1. Create a new Word/Google Doc document
2. Add your name and student ID at the top
3. For each task (1-5), add:
   - Task title
   - Repository URL: `https://github.com/<your-username>/mlops-kubeflow-assignment`
   - Required screenshots
   - Any required text/code snippets

4. Save as PDF with filename: `ROLLNUM_SECTION.pdf`

## Step 8: Create Final Submission Zip

```bash
# Copy only source files (no cache/binaries)
# Create a new folder
mkdir ROLLNUM_SECTION

# Copy files
cp -r src components data .github .dvc ROLLNUM_SECTION/
cp requirements.txt pipeline.py compile_components.py README.md Dockerfile Jenkinsfile .gitignore .dvcignore ROLLNUM_SECTION/
cp ROLLNUM_SECTION.pdf ROLLNUM_SECTION/

# Create zip
# On Windows: Right-click the folder > Send to > Compressed (zipped) folder
# On Linux/Mac: zip -r ROLLNUM_SECTION.zip ROLLNUM_SECTION/
```

## Important Notes

1. **Do NOT include**:
   - `__pycache__/` directories
   - `.git/` directory
   - `mlruns/` directory
   - `models/*.pkl` files (large binaries)
   - Any other cache or binary files

2. **DO include**:
   - All source code (.py files)
   - Configuration files (.yaml, .yml, .txt)
   - Documentation (.md files)
   - Your PDF document
   - Small data files (if needed for demonstration)

3. **Repository URL**: Make sure to update `<your-username>` in all documentation

4. **Test your zip**: Extract it in a different location and verify all files are present

## Troubleshooting

### Issue: Git push rejected

```bash
# Pull first, then push
git pull origin main --rebase
git push origin main
```

### Issue: DVC remote not accessible on GitHub

DVC data is stored separately from Git. For the assignment, you can:
- Use local DVC storage (already configured)
- Or set up Google Drive/S3 remote storage

The `.dvc` files in Git track the data versions.

### Issue: Large files

If you get errors about large files:

```bash
# Add them to .gitignore
echo "models/*.pkl" >> .gitignore
echo "data/*.csv" >> .gitignore
git rm --cached models/*.pkl data/*.csv
git commit -m "Remove large files"
```

## Questions?

Refer to:
- [README.md](README.md) for setup instructions
- [DELIVERABLES.md](DELIVERABLES.md) for what to submit
- Assignment PDF for grading criteria
