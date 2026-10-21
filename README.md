
# PentAI — Multi-Livestock Disease Detection System

**Institution:** Moringa School  
**Group Name:** PentAI  
**Repository:** livestock-disease-detection  
**Technical Mentor:** Samwel Jane  
**Project Deadline:** 20 November 2026  
**Project Status:** Setup and Data Preparation

## 1. Project Overview

PentAI is a Data Science capstone project that explores
the use of computer vision and deep learning for
image-based livestock disease classification.

The project focuses on three livestock species:
- Goats
- Cattle
- Chickens

The proposed system will classify supported disease
categories from uploaded livestock images.

## 2. Problem Statement

Livestock farmers may experience difficulties identifying
visible disease symptoms early, especially where access
to veterinary services is limited.

Delayed identification can affect livestock productivity,
animal welfare, and farm income.

This project aims to develop an image-classification
prototype to support preliminary livestock disease
screening and awareness.

The system is not a substitute for veterinary diagnosis.

For the complete problem statement, refer to the
official capstone proposal.

## 3. Project Objectives

### Main Objective
Develop and evaluate an image-based machine learning
prototype for classifying supported livestock disease
categories in goats, cattle, and chickens.

### Specific Objectives
1. Collect and assess available livestock image datasets.
2. Clean and preprocess images for model development.
3. Conduct exploratory data analysis.
4. Train and compare image-classification models.
5. Evaluate models using accuracy, precision, recall,
   F1-score, and confusion matrices.
6. Develop an interface for submitting images and
   displaying classification results.

## 4. Datasets

The original datasets are stored as ZIP archives on
Google Drive. Processed versions will be maintained
separately as they become available.

| Species | Dataset | Google Drive |
|---|---|---|
| Goat | Goat Pox Detection | [Goat Dataset](PASTE_GOAT_FOLDER_URL) |
| Cattle | Healthy and Lumpy Skin Disease | [Cattle Dataset](PASTE_CATTLE_FOLDER_URL) |
| Chicken | Chicken Disease Images | [Chicken Dataset](PASTE_CHICKEN_FOLDER_URL) |

### Known Dataset Information

**Cattle**
- Lumpy skin disease: 324 images
- Healthy: 700 images
- Total: 1,024 images
- Processed image size: 256 × 256 pixels
- Processed format: PNG

**Goat and Chicken**
- Detailed counts and classes: to be verified
  in the dataset inventory.

The original dataset sources, licenses, class
distributions, and preprocessing details are recorded
in [Dataset Inventory](docs/dataset_inventory.md).

## 5. Repository Structure

```text
livestock-disease-detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── models/
├── app/
├── reports/
│   ├── figures/
│   └── presentations/
├── docs/
│   ├── proposal/
│   ├── dataset_inventory.md
│   └── CONTRIBUTING.md
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

Large image datasets are stored externally and
excluded from normal Git tracking.

## 6. Project Team

| Member | Responsibility |
|---|---|
| Dan Wamatu | Project Manager |
| Linnet Kinyua | Scrum Master |
| Meshack Kimanzi | Team Member |
| Grace Muthoni | Team Member |
| Cleophas Ndunda | Team Member |

All team members contribute to technical development
according to their assigned ClickUp tasks.

## 7. Technologies

Planned tools and technologies include:

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib
- OpenCV or Pillow
- Scikit-learn
- TensorFlow/Keras or PyTorch
- Git and GitHub
- Streamlit or another suitable application framework

Final choices will be documented as implementation
progresses.

## 8. Collaboration Workflow

ClickUp is used for task assignments, deadlines,
dependencies, progress tracking, and reviews.

GitHub is used for version control and code review.

Contributors should:
1. Synchronize their local repository.
2. Create a task-specific branch.
3. Develop and test their changes.
4. Commit and push the changes.
5. Open a pull request.
6. Request review before merging.

Detailed guidance will be maintained in
`docs/CONTRIBUTING.md`.

## 9. Project Documentation

- Capstone Proposal: [Open Proposal](PASTE_PROPOSAL_URL)
- Shared Project Drive: [Open Drive](PASTE_MAIN_DRIVE_URL)
- Dataset Inventory: [View Inventory](docs/dataset_inventory.md)
- ClickUp Workspace: [Open ClickUp](PASTE_CLICKUP_URL)

## 10. Installation and Usage

Detailed installation instructions will be added
when the project's environment and application
dependencies are confirmed.

## 11. Limitations

- The system will classify only supported categories.
- Predictions depend on dataset quality and coverage.
- Model confidence is not clinical certainty.
- The prototype is intended for academic research.
- Veterinary professionals remain responsible for
  clinical diagnosis and treatment decisions.

## 12. Project Deliverables

- Cleaned and documented datasets
- Exploratory data analysis
- Reproducible preprocessing scripts
- Trained machine learning models
- Model evaluation and comparison
- Working prediction prototype
- Final capstone report and presentation
