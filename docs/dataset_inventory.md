# Capstone
 https://drive.google.com/drive/folders/1BYzqsYn1JDBOm_b33nyU3Ryullv-p8XI?usp=drive_link

# Dataset/Chicken
https://drive.google.com/file/d/1tb_HGvGJqAEttBV6c8yLFxRwMmWP7g7f/view?usp=drive_link

# Dataset/Cow
https://drive.google.com/file/d/1COCT4TeyrFEKH6oooxD62CzGXFHs67Ku/view?usp=drive_link

# Dataset/Goat
https://drive.google.com/file/d/1Dpw1SHO_P0QVEiOo_VnCMDRE10Sae8Ed/view?usp=drive_link


# PentAI — Livestock Dataset Inventory

**Project:** Multi-Livestock Disease Detection System  
**Repository:** livestock-disease-detection  
**Maintained by:** PentAI Team  
**Last Updated:** 21 October 2026

## 1. Dataset Overview

| Species | Dataset | Images | Status |
|---|---|---:|---|
| Goat | Goat Pox Detection | TBD | Initial dataset available |
| Cattle | Lumpy Skin Disease / Healthy | 1024 | Processed |
| Chicken | Chicken Disease Dataset | TBD | Requires quality review |

Note: Counts refer to the current known dataset versions.
Update them after verification.

## 2. Goat Dataset

**Dataset Name:** Goat Pox Detection  
**Original Filename:** Goat Pox Detection.v1i.tensorflow.zip  
**Original Source:** [Add original source URL]  
**Version:** v1  
**License:** [Verify license]  
**Drive Folder:** [Add goat folder URL]

### Dataset Composition

| Class | Image Count |
|---|---:|
| [Actual class label] | TBD |
| [Actual class label] | TBD |
| Total | TBD |

### Technical Details
- Image format: To verify
- Image resolution: To verify
- Dataset split: To verify
- Preprocessing: To document
- Duplicates: Not yet assessed
- Quality issues: To verify

## 3. Cattle Dataset

**Dataset Name:** Cattle Disease Dataset  
**Original Filename:** Cow data.zip  
**Original Source:** [Add original source URL]  
**License:** [Verify license]  
**Drive Folder:** [Add cattle folder URL]

### Dataset Composition

| Class | Image Count |
|---|---:|
| Lumpy Skin Disease | 324 |
| Healthy | 700 |
| Total | 1024 |

### Technical Details
- Processed resolution: 256 x 256 pixels
- Processed format: PNG
- Preprocessing completed: Image resizing
- Duplicate screening: To verify
- Dataset split: To verify
- Class imbalance: Present (324 versus 700)

### Preparation Notes
The current processed dataset contains 1,024 images.
The preprocessing script or notebook must be linked
when available to document the transformation process.

## 4. Chicken Dataset

**Dataset Name:** Chicken Disease Dataset  
**Original Filename:** Chicken data.zip  
**Original Source:** [Add original source URL]  
**License:** [Verify license]  
**Drive Folder:** [Add chicken folder URL]

### Dataset Composition

| Class | Image Count |
|---|---:|
| [Actual class label] | TBD |
| [Actual class label] | TBD |
| Total | TBD |

### Technical Details
- Image format: To verify
- Resolution: To verify
- Dataset split: To verify
- Preprocessing: To document
- Quality issues: Requires further review
- Preliminary usability assessment: 5.5/10

## 5. Data Storage

Original ZIP archives are stored in Google Drive.

- Main Drive Folder: [Add link]
- Goat Dataset: [Add link]
- Cattle Dataset: [Add link]
- Chicken Dataset: [Add link]

Processed datasets will be stored separately
with appropriate version names.

## 6. Data Cleaning and Preprocessing

For each species, record:
1. Original number of images.
2. Removed corrupted or unreadable files.
3. Removed duplicate images, if any.
4. Image resizing and format conversion.
5. Final number of images by class.
6. Training, validation, and test split method.
7. Script or notebook used.
8. Date and responsible contributor.

Do not overwrite original data.

## 7. Dataset Risks and Limitations

- Cattle classes are imbalanced.
- Chicken dataset quality needs further review.
- Goat image counts and classes require verification.
- Dataset licensing must be confirmed.
- Data leakage must be prevented during splitting.
- Dataset suitability must be evaluated before training.

## 8. Pending Verification

- [ ] Confirm original source URL for each dataset.
- [ ] Confirm the license for each dataset.
- [ ] Verify goat class names and counts.
- [ ] Verify chicken class names and counts.
- [ ] Verify cattle preprocessing code.
- [ ] Record image quality findings.
- [ ] Document train/validation/test splits.
- [ ] Verify team access to Google Drive.
