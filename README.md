# Bertelsmann_Arvato_Project

This repository contains the solution for the Bertelsmann Arvato Financial Solutions project, developed as part of the Udacity Data Science Nanodegree Program. The project focuses on customer segmentation using demographic data and building a predictive machine learning model.

## GitHub Repository

The source code for this project is hosted on GitHub:  
[Arvato Project Repository](https://github.com/Makarand7/Bertelsmann_Arvato_Project)

## Web Application 

Web Application for this project is hosted on Streamlit:  
[Bertelsmann Arvato Project Web App](<link>)

## Project Overview

Arvato Financial Solutions aims to revolutionize customer acquisition for a mail-order company specializing in organic products. The company faces a significant challenge: efficiently identifying potential customers from the general population of Germany while minimizing costs and maximizing conversion rates.  

### Data Overview
The project uses four key datasets:  
- **General Population Data (AZDIAS)**: Demographic and lifestyle data representing the German population.  
- **Customer Data (CUSTOMERS)**: Demographic and behavioral data of existing customers.  
- **Training Dataset (MAILOUT_TRAIN)**: Contains labeled data indicating customer responses to past marketing campaigns.  
- **Test Dataset (MAILOUT_TEST)**: Unlabeled data used to predict and evaluate responses.  

### Objectives
- **Customer Segmentation Reports**:  
   - Performed using unsupervised learning (PCA, KMeans) on both general population and customer datasets.  
   - Identified distinct demographic segments with actionable business insights.  

- **Supervised Learning Model**:  
   - Built an XGBoost classifier using the labeled training data.  
   - The model predicts the likelihood of a customer responding to a marketing campaign.  

### Business Relevance
- **Optimize Marketing Campaigns**: Reduce costs by targeting only high-potential customers.  
- **Enhance Customer Satisfaction**: Personalize marketing to customer preferences.  
- **Improve ROI**: Increase conversion rates and long-term customer loyalty through data-driven targeting.  

---

## Key Features

### Data Cleaning and Preparation:
- Handled missing values through imputation and column dropping where necessary.  
- Encoded categorical variables and scaled numerical features.  
- Prepared datasets for segmentation and modeling.  

### Customer Segmentation Reports:
- **Population Segmentation Report**:  
   - Performed PCA and KMeans on the AZDIAS dataset.  
   - Identified 4 distinct clusters based on demographics and purchasing behavior.  

- **Customer Segmentation Report**:  
   - Performed similar clustering on the customer dataset.  
   - Extracted customer behavior insights to guide model feature selection.  

### Dimensionality Reduction (PCA):
- Reduced feature dimensions while retaining over 95% variance.  
- Helped identify key components for both clustering and predictive modeling.  

### Model Building:
- Developed and fine-tuned an **XGBoost classifier** using GridSearchCV.  
- Trained on balanced data after handling class imbalance.  

### Model Evaluation:
- Evaluated using accuracy, precision, recall, F1_Score, ROC-AUC and Precision-Recall AUC.  
- Analyzed feature importance.  

### Results:
- Generated predictions for potential positive responders.  
- Exported **positive_predicted_users.csv** for marketing use.  

---

## Project Structure
```
Bertelsmann_Arvato_Project
├── app.py
├── data
│   ├── Udacity_AZDIAS_052018.csv
│   ├── Udacity_CUSTOMERS_052018.csv
│   ├── Udacity_MAILOUT_052018_TRAIN.csv
│   ├── Udacity_MAILOUT_052018_TEST.csv
├── model
│   ├── trained_ct.pkl
│   ├── scaler.pkl
│   ├── train_features.pkl
│   └── xgboost_model.pkl
├── notebooks
│   ├── Bertelsmann_Arvato_Project.ipynb
│   ├── Bertelsmann_Arvato_Project.py
│   ├── Bertelsmann_Arvato_Project.html
│   └── positive_predicted_users.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

### Set up the environment:
Install dependencies from `requirements.txt` using:

```
pip install -r requirements.txt
```

### Run the notebook:
Open `Bertelsmann_Arvato_Project.ipynb` in Jupyter Notebook or JupyterLab and run cells step-by-step.

## Results and Recommendations

- XGBoost model performed well after hyperparameter tuning.
- PCA helped in understanding key components.
- Feature importance plots showed the top influencing features.
- Generated prediction output (`positive_predicted_users.csv`) with customer IDs most likely to respond positively.

## Improvements and Next Steps

- Explore alternative machine learning models (LightGBM, Random Forest).
- Fine-tune PCA and feature engineering for better predictive performance.
- Consider ensemble approaches.

## How to Contribute
We welcome contributions to improve this project! Feel free to fork the repository, raise issues, or submit pull requests.

## Acknowledgments
This project was completed as part of the Udacity Data Science Nanodegree Program. Special thanks to Arvato Financial Solutions for providing the data and Udacity for the learning opportunity.
