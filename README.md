# AI Health Risk Predictor

---

## 🎯 **Project Highlights**

- Developed a binary classification model to predict **smoking status** using health-related metrics.
- Explored multiple algorithms including **Logistic Regression** and **Random Forest Classifier**.
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance.
- Achieved competitive **AUC-ROC scores** on validation data, enabling better prediction of smoking likelihood.
- Delivered a **submission file** containing predicted probabilities for the test dataset.

---

## 👩🏽‍💻 **Setup and Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place `train.csv` and `test.csv` from the **BTT DataSpark 2025** dataset in the working directory.
   - Update file paths in `BinaryClassificationProject.py` or `team-3.ipynb` if necessary.

4. **Run the notebook or script:**
   - For the notebook:
     ```bash
     jupyter notebook team-3.ipynb
     ```
   - For the Python script:
     ```bash
     python BinaryClassificationProject.py
     ```

---

## 🏗️ **Project Overview**

This project is part of the **Break Through Tech AI Studio Challenge**, aimed at applying data science and machine learning techniques to real-world datasets.  
Our objective was to build a robust binary classification model capable of predicting whether an individual is a smoker based on their health examination data.

**Real-world significance:**  
Accurate predictions of smoking status can support healthcare providers in targeted interventions, preventive care, and public health strategies.

---

## 📊 **Data Exploration**

- **Dataset:**  
  The training and test datasets include medical and demographic features such as:
  - `Gtp`, `hemoglobin`, `serum creatinine`, `weight(kg)`, `triglyceride`, `height(cm)`, `waist(cm)`, `age`  
  - Target variable: `smoking` (1 = smoker, 0 = non-smoker)

- **Key steps in EDA:**
  - Checked for missing values and data types.
  - Identified and visualized outliers using boxplots.
  - Explored target distribution to understand class imbalance.
  - Calculated smoker-to-non-smoker ratio.

**Sample Visualizations:**
![Distribution of Smoking Status](images/smoking_distribution.png)  
![Boxplot of Numerical Features](images/numerical_features_boxplot.png)

---

## 🧠 **Model Development**

- **Algorithms Used:**
  - Logistic Regression (baseline model)
  - Random Forest Classifier (final model)

- **Preprocessing:**
  - Feature selection based on domain relevance.
  - Applied **SMOTE** to handle class imbalance.
  - Train-test split: 80% training, 20% validation.

- **Evaluation Metrics:**
  - **AUC-ROC Score**
  - Classification Report (Precision, Recall, F1-score)

---

## 📈 **Results & Key Findings**

- **Logistic Regression (with SMOTE):** Improved recall for the minority class.
- **Random Forest Classifier:** Achieved the highest AUC-ROC score on validation data.
- Final output: CSV file containing IDs and predicted probabilities of smoking.

**Sample Output:**
```csv
id,smoking
1001,0.8421
1002,0.1245
1003,0.5602
...
```

---

## 🚀 **Next Steps**

- Experiment with gradient boosting algorithms (e.g., XGBoost, LightGBM).
- Perform hyperparameter optimization for further accuracy gains.
- Incorporate additional features if available for better predictive power.

---

## 📝 **License**

This project is licensed under the MIT License.

---

## 🙏 **Acknowledgements**

Special thanks to the **Break Through Tech AI Program**, our challenge advisors, and the BTT DataSpark 2025 dataset providers.
