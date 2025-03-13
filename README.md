# Hyperdimensional Computing for Covertype Classification

## Introduction
This project explores the use of **Hyperdimensional Computing (HDC)**, specifically **Fourier Holographic Reduced Representation (FHRR)**, as an encoding method for a traditional machine learning classifier, **Random Forest**, on the **Covertype dataset**. The main goal is to evaluate whether **HDC-based feature encoding can outperform traditional ML techniques**, particularly in the context of high-dimensional representations.

### What is Hyperdimensional Computing (HDC)?
Hyperdimensional Computing (HDC) represents data using high-dimensional vectors (hypervectors). A hypervector is a randomly generated, high-dimensional vector that encodes information in a way that preserves similarity and allows algebraic operations. This method of encoding places data in high dimensional space so that data are now separable.

### Fourier Holographic Reduced Representation (FHRR)
FHRR is a type of hyperdimensional representation that leverages **complex numbers** and **Fourier transform properties** to encode information. Instead of using binary or bipolar representations, FHRR encodes data using **complex-valued hypervectors**, where the phase and magnitude capture essential feature properties.

## Dataset: Covertype
The **Covertype dataset** classifies forest cover types based on cartographic features (elevation, slope, soil type, etc.). The dataset has **581,012 samples and 54 features**, but for this experiment, **only the first 10 features were used** to simplify processing and focus on encoding efficiency.

## Preprocessing Steps
To prepare the dataset for HDC encoding and Random Forest classification, the following preprocessing steps were applied:

1. **Feature Selection**: Used only the **first 10 features** instead of the full 54 to reduce computational complexity.
2. **Data Normalization**: Applied **StandardScaler** to normalize the feature values.
3. **Handling Class Imbalance**: Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples and balance class distribution.
4. **Dataset Reduction**: Since Covertype is a large dataset, **only 10% of the data** was used to make training and encoding manageable.
5. **Train-Test Split**: The dataset was split into **training (80%)**, **validation (10%)**, and **test (10%)** sets.

## Hyperdimensional Encoding
To transform the dataset into an HDC-friendly format, the following steps were applied:

1. **Hypervector Dimensionality (D)**: Set **D = 10,000**, meaning each sample is represented by a **10,000-dimensional complex-valued hypervector**.
2. **Basis Generation**: Generated a **random Gaussian matrix W (D × n_features)** for encoding.
3. **Encoding Method (FHRR)**:
   - Computed **dot product between W and feature vectors**.
   - Applied the **Fourier transform (complex exponentiation)** to generate complex-valued hypervectors.
   - Used the **real part** of the encoded hypervectors as input for classification.

## Model and Performance Evaluation
### **Classifier: Random Forest**
A **Random Forest model** was trained on the **HDC-encoded dataset** with the following configuration:
- **n_estimators = 100** (100 decision trees)
- **Random state = 42**

### **Performance Results**
| Model | Validation Accuracy | Test Accuracy |
|--------|--------------------|--------------|
| **Random Forest (HDC-encoded features)** | **92.25%** | **92.02%** |
| **Traditional ML (Best Known RF Performance)** | ~88% | ~88% |

### **Observations**
- The **HDC-encoded features outperformed the traditional Random Forest baseline (88%)**, achieving **92.02% test accuracy**.
- The encoding dimensionality **D = 10,000** contributed significantly to model performance.
- Using **FHRR encoding** preserved key feature interactions, improving separability.

## Conclusion
This project demonstrated that **HDC-based encoding (FHRR) can significantly boost Random Forest classification performance on the Covertype dataset**. The results suggest that high-dimensional representations **enhance feature expressiveness and improve classification accuracy**, surpassing traditional feature engineering techniques.

Future work could involve:
- **Comparing FHRR with other HDC encoding methods (e.g., Binary HDC, MAP encoding).**
- **Exploring the effect of varying hypervector dimensionality (D) on accuracy and efficiency.**
- **Applying HDC to other ML models (e.g., SVM, XGBoost) to assess its generalization ability.**

