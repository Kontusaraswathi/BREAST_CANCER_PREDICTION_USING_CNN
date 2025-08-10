Breast Cancer Prediction using Deep Learning (CNN):
This project predicts whether a breast tumor is **benign**, **malignant**, or **normal** using a **Convolutional Neural Network (CNN)** trained on breast ultrasound images.

---
Problem Statement:
Breast cancer is one of the most common cancers in women worldwide. Early detection plays a crucial role in treatment success.  
This project aims to **automate breast ultrasound image classification** into:
- Benign
- Malignant
- Normal

---

Dataset
- **Name**: [Breast Ultrasound Images Dataset (BUSI)][(https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Number of Classes**: 3  
  - `benign/` – Images of benign tumors
  - `malignant/` – Images of malignant tumors
  - `normal/` – Images of normal breast tissue
- **Image Format**: JPG
- **Ground Truth**: Segmentation masks also provided (not used in classification)

---

## Approach

1. **Data Preprocessing**
   - Load images and resize to fixed dimensions
   - Normalize pixel values between 0 and 1
   - Encode class labels
   - Train-test split (80%-20%)

2. **Model Architecture (CNN)**
   - Convolutional layers for feature extraction
   - MaxPooling layers for dimensionality reduction
   - Dropout for regularization
   - Fully connected dense layers
   - Softmax output for 3-class classification

3. **Training**
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Evaluation Metrics: Accuracy, Precision, Recall

4. **Evaluation**
   - Confusion Matrix
   - Accuracy & loss plots
   - Example predictions

---

## model Files
- `breast_cancer_model.h5` – Trained CNN model
- `breast_cancer_prediction.h5` – Saved model for deployment

---

## Results
| Metric      | Value  |
|-------------|--------|
| Accuracy    | ~95%   |
| Precision   | ~94%   |
| Recall      | ~94%   |

---
## 🚀 Installation & Usage

### **1. Clone the repository**
```bash
git clone https://github.com/Kontusaraswathi/BREAST_CANCER_PREDICTION_USING_CNN/tree/main
cd Breast_cancer_prediction_using_ML
