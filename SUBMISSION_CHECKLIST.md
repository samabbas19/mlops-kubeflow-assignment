# Assignment #4 Submission Checklist

**Student Name**: SAAMER
**Student ID**: 22i-0468
**Repository URL**: https://github.com/samabbas19/mlops-kubeflow-assignment

---

## ✅ Completed Tasks

### Task 1: Project Initialization and Data Versioning (20 Marks) ✓
- [x] GitHub repository created and populated
- [x] DVC initialized and configured
- [x] Dataset tracked with DVC
- [x] Data pushed to DVC remote storage
- [x] requirements.txt created with all dependencies

### Task 2: Building Kubeflow Pipeline Components (25 Marks) ✓
- [x] pipeline_components.py with 4 component functions
- [x] Components compiled to YAML files (4 files in components/)
- [x] Component inputs/outputs documented

### Task 3: Orchestrating the Pipeline (30 Marks) ✓
- [x] MLflow pipeline implemented and executed
- [x] All 4 pipeline stages completed successfully
- [x] Model achieved R² score: 0.8053
- [x] Metrics saved and tracked

### Task 4: Continuous Integration (15 Marks) ✓
- [x] GitHub Actions workflow created
- [x] Jenkinsfile created with 3 stages
- [x] Both CI/CD pipelines configured

### Task 5: Final Integration and Documentation (10 Marks) ✓
- [x] Comprehensive README.md created
- [x] All code pushed to GitHub
- [x] Repository structure verified

---

## 📸 Screenshots Needed

Create a `screenshots/` folder with these images:

### Task 1 (2 screenshots):
- [ ] `task1_repo_structure.png` - GitHub repo showing all files
- [ ] `task1_dvc_status.png` - Terminal with `dvc status` and `dvc push`

### Task 2 (2 screenshots):
- [ ] `task2_pipeline_components.png` - Code from pipeline_components.py
- [ ] `task2_components_directory.png` - components/ folder with YAML files

### Task 3 (3 screenshots):
- [ ] `task3_mlflow_ui.png` - MLflow UI showing experiments
- [ ] `task3_pipeline_graph.png` - Pipeline run with all stages
- [ ] `task3_metrics_output.png` - Metrics (R² = 0.8053)

### Task 4 (1 screenshot):
- [ ] `task4_github_actions.png` - GitHub Actions successful run

### Task 5 (1 screenshot):
- [ ] `task5_github_repo.png` - GitHub main page with README

---

## 📝 How to Take Screenshots

### For MLflow UI:
```bash
cd "c:\Users\PC\Desktop\sem 7\MLops\A4"
mlflow ui
# Open http://localhost:5000 in browser
# Take screenshots
```

### For DVC commands:
```bash
dvc status
dvc push
# Take screenshot of terminal
```

### For GitHub:
1. Go to: https://github.com/samabbas19/mlops-kubeflow-assignment
2. Screenshot the main page
3. Click on "Actions" tab for workflow screenshot
4. Browse files for code screenshots

---

## 📄 Create PDF Document

1. **Document Header**:
   - Your Full Name
   - Your Student ID

2. **For Each Task (1-5)**:
   ```
   Task X: [Title]

   Repository URL: https://github.com/samabbas19/mlops-kubeflow-assignment

   [Insert required screenshots here]

   [Add any required explanations/code snippets]
   ```

3. **Important Content**:

   **Task 1**:
   - Screenshot of repo structure
   - Screenshot of DVC commands
   - Content of requirements.txt

   **Task 2**:
   - Screenshot of pipeline_components.py (at least 2 functions)
   - Screenshot of components/ directory
   - Explanation of training component inputs/outputs:
     ```
     Inputs:
     - train_data_path (str): Path to training CSV
     - model_output_path (str): Where to save model
     - n_estimators (int): Number of trees (default=100)

     Outputs:
     - model_output_path (str): Path to saved model (.pkl)
     ```

   **Task 3**:
   - Screenshot of MLflow UI
   - Screenshot showing pipeline stages completed
   - Screenshot of metrics: R² = 0.8053, RMSE = 0.5051

   **Task 4**:
   - Screenshot of GitHub Actions workflow
   - Content of Jenkinsfile (from DELIVERABLES.md)

   **Task 5**:
   - Repository URL
   - Screenshot of GitHub main page

---

## 📦 Create Submission ZIP

### Important Files to Include:

```
ROLLNUM_SECTION/
├── ROLLNUM_SECTION.pdf          # Your PDF document
├── src/
│   ├── pipeline_components.py
│   ├── model_training.py
│   └── generate_data.py
├── components/
│   ├── data_extraction.yaml
│   ├── data_preprocessing.yaml
│   ├── model_training.yaml
│   └── model_evaluation.yaml
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── .dvc/
│   └── config
├── data/
│   └── raw_data.csv.dvc
├── pipeline.py
├── compile_components.py
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
├── .gitignore
├── .dvcignore
└── README.md
```

### DO NOT Include:
- `__pycache__/`
- `.git/`
- `mlruns/`
- `models/*.pkl` (large files)
- `data/*.csv` (tracked by DVC)
- `.claude/`

### Create ZIP:
1. Create folder with your roll number: `ROLLNUM_SECTION` (e.g., `22i-0001_A`)
2. Copy all required files
3. Add your PDF document
4. Right-click folder → Send to → Compressed (zipped) folder
5. Name: `ROLLNUM_SECTION.zip`

---

## ✅ Final Checklist Before Submission

- [ ] PDF document created with name and student ID
- [ ] All 9 screenshots included in PDF
- [ ] Repository URL included in every task
- [ ] Code snippets included where required
- [ ] ZIP file named correctly (ROLLNUM_SECTION.zip)
- [ ] ZIP contains PDF + source code
- [ ] No large binaries in ZIP
- [ ] Tested by extracting ZIP in different location
- [ ] File size reasonable (< 50MB)
- [ ] Submitted on Google Classroom

---

## 📊 Your Project Results

**Model Performance**:
- R² Score: **0.8053** (80.53% variance explained)
- RMSE: **0.5051**
- MAE: **0.3274**
- Dataset: 20,640 samples

**Repository**: https://github.com/samabbas19/mlops-kubeflow-assignment

**Tools Used**:
- DVC for data versioning
- MLflow for pipeline orchestration
- Kubeflow components (compiled to YAML)
- GitHub Actions for CI/CD
- Random Forest Regressor

---

## 🎯 Quick Commands Reference

```bash
# View MLflow UI
mlflow ui

# Run pipeline
python pipeline.py --mode mlflow

# Compile components
python compile_components.py

# Check DVC status
dvc status

# View git log
git log --oneline

# View repository
# Go to: https://github.com/samabbas19/mlops-kubeflow-assignment
```

---

**Good Luck with Your Submission! 🎓**
