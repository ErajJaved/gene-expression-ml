# 🧬 Gene Expression Analysis 

## 📌 Project Overview  
This project explores whether **gene expression profiles** can be used to **predict disease risk and treatment response**.  
Using machine learning models, we can classify patients based on their gene expression data and evaluate which genes are most predictive of disease outcomes.  

📊 **Dataset:** [Gene Expression Analysis and Disease Relationship (Kaggle)](https://www.kaggle.com/datasets/ylmzasel/gene-expression-analysis-and-disease-relationship)

---

## 🚀 Objectives  
- Preprocess and transform raw gene expression data.  
- Train multiple machine learning classifiers.  
- Compare model performance (accuracy, precision, recall, F1).  
- Identify the most important genes influencing disease status.  
- Provide insights into potential biomarkers for disease risk prediction.  

---

## 🛠️ Tech Stack  
- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost)  
- **Machine Learning Models:**  
  - Random Forest  
  - Logistic Regression  
  - Support Vector Classifier (SVC)  
  - XGBoost  

---

## 📂 Workflow  
1. **Data Preprocessing**  
   - Log2 transformation of gene expression features.  
   - Removal of identifiers and categorical features not suitable for modeling.  
   - Splitting data into training and test sets.  

2. **Model Training & Evaluation**  
   - Models: RandomForest, LogisticRegression, SVC, XGBClassifier.  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-score.  
   - Confusion matrices plotted for each model.  
   - Cross-validation to ensure model generalization.  

3. **Feature Importance**  
   - Extracted from tree-based models (RandomForest, XGB).  
   - Identified key genes that contribute most to disease classification.  

---

## 📊 Results  

| Model                 | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
| Random Forest          | 1.000    | 1.000     | 1.000  | 1.000    |
| Logistic Regression    | 0.960    | 0.922     | 0.960  | 0.940    |
| Support Vector Machine | 0.976    | 0.977     | 0.976  | 0.970    |
| XGBoost                | 0.996    | 0.997     | 0.996  | 0.996    |


✅ **Tree-based models (RF, XGB)** outperform simpler linear models.  
✅ **Gene X expression** was found to be more predictive of disease outcome than Gene Y.  

---

## 📈 Visualizations  
- Confusion matrices for each model.  
- Feature importance plots (top contributing genes).  
- Model comparison bar chart (Accuracy, Precision, Recall, F1).  

---

## 🎯 Key Insights  
- Gene expression profiles can reliably predict disease status.  
- Certain genes (e.g., Gene X) show much higher predictive importance.  
- Tree-based models generalize extremely well with cross-validation.  
- Potential applications in **biomarker discovery** and **personalized medicine**.  

---


