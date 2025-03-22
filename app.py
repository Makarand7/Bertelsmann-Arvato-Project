import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from notebooks.Bertelsmann_Arvato_Project import clean_dataframes
from notebooks.Bertelsmann_Arvato_Project import preprocess_dataframe

# Define paths
DATA_PATH = "data/Udacity_MAILOUT_052018_TEST.csv"
MODEL_PATH = "model/xgboost_model.pkl"  # Update with actual path
SCALER_PATH = "model/scaler.pkl"  # Update with actual path
FEATURES_PATH = "model/train_features.pkl"
TRAIN_CT_PATH = "model/trained_ct.pkl"

# Load required variables
train_features = joblib.load(FEATURES_PATH)

# Load trained model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    tarined_ct = joblib.load(TRAIN_CT_PATH)
else:
    st.error("Model or Scaler file missing! Upload them in the 'model' directory.")
    st.stop()

# Streamlit UI
st.markdown(
    '<h1 style="color:#800080; text-align:left; margin-top:-20px;">Bertelsmann Arvato Project</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h3 style="margin-top: -10px; margin-bottom: 30px;">Customer Response Prediction Web App</h3>',
    unsafe_allow_html=True,
)


##Mak
# Sidebar for navigation
st.sidebar.markdown(
    """
    <style>
        .sidebar-images {
            margin-top: -20px; /* moves logos upward */
        }
        .sidebar-images table {
            border: none;
        }
        .sidebar-images img {
            width: 40px;
            height: 40px;
            object-fit: contain;
            transition: transform 0.2s;
            border: none; /* removes any box lines */
        }
        .sidebar-images img:hover {
            transform: scale(1.15);
        }
    </style>
    <div class="sidebar-images">
        <table>
            <tr>
                <td>
                    <a href="https://github.com/Makarand7/Recommendations_with_IBM" target="_blank">
                        <img src="https://github.com/Makarand7/assets/blob/main/GitHub_logo.jpg?raw=true" alt="GitHub">
                    </a>
                </td>
                <td>
                    <a href="https://www.linkedin.com/in/makarand-52b930324/" target="_blank">
                        <img src="https://github.com/Makarand7/assets/blob/main/LinkedIn_logo.png?raw=true" alt="LinkedIn">
                    </a>
                </td>
                <td>
                    <a href="https://www.udacity.com/" target="_blank">
                        <img src="https://github.com/Makarand7/assets/blob/main/Udacity_logo.png?raw=true" alt="Udacity">
                    </a>
                </td>
            </tr>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    [
        "📊 Project Overview",
        "🗺️ Customer Segmentation Report - POPULATION",
        "👥 Customer Segmentation Report - CUSTOMERS",
        "📈 Prediction & Analysis",
        "✅ Conclusion and Key Outcomes",
    ],
)

# Project Overview
if section == "📊 Project Overview":
    st.markdown('<h2 style="color:#a770a0;">📊 Project Overview</h2>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-size:18px">
<strong style="color:#0d47a1;">Arvato Financial Solutions</strong> aims to revolutionize customer acquisition for a mail-order company specializing in organic products.<br><br>
The company faces a significant challenge: efficiently identifying potential customers from the general population of Germany while minimizing costs and maximizing conversion rates.
</p>

---

### 🎯 <span style="color:#0d47a1;">Objectives:</span>
<ul>
<li><strong style="color:#4caf50;">Customer Segmentation Report:</strong> Using unsupervised learning methods (PCA, KMeans) to analyze and segment both the general population and existing customers.</li>
<li><strong style="color:#4caf50;">Supervised Learning Model:</strong> Building a machine learning model to predict the likelihood of an individual responding to a marketing campaign.</li>
</ul>

---

### 💡 <span style="color:#0d47a1;">Business Relevance:</span>
<ul>
<li><strong style="color:#4caf50;">Optimize Marketing Campaigns:</strong> Efficient resource allocation reduces advertising costs.</li>
<li><strong style="color:#4caf50;">Enhance Customer Satisfaction:</strong> Tailored product recommendations improve engagement.</li>
<li><strong style="color:#4caf50;">Improve ROI:</strong> By focusing efforts on high-potential individuals, the company can increase conversion rates and build a base of loyal customers.</li>
</ul>

---

<p style="font-size:16px">
✅ Through this data-driven approach, <strong style="color:#0d47a1;">Arvato Financial Solutions</strong> can ensure its marketing strategies are both effective and impactful.<br>
Industries such as retail, e-commerce, and finance have successfully used similar methods to achieve measurable improvements in customer engagement and profitability.
</p>
""", unsafe_allow_html=True)


# Customer Segmentation Report - POPULATION
elif section == "🗺️ Customer Segmentation Report - POPULATION":
    st.markdown("""
<h2 style="color:#a770a0; text-align:left;">🗺️ Customer Segmentation Final Report for General Population (AZDIAS)</h2>

## 1️⃣ Objective  
This report presents the findings from the segmentation analysis of the AZDIAS dataset, which contains demographic and lifestyle data representing the general population of Germany. The analysis employed unsupervised learning techniques to identify clusters of individuals with similar attributes, providing actionable insights for targeted marketing.  

<hr style="border: 1px solid #ccc;">  

## 2️⃣ Dataset Overview  
The AZDIAS dataset underwent extensive preprocessing, including:  

✔ **Scaling and imputing missing values**  
✔ **Application of PCA** to reduce dimensions while preserving 95% of the variance  
✔ **Post-PCA Data Shape:** The reduced dataset contained over **890,000 records** with significantly fewer features, enhancing computational efficiency and interpretability  

<hr style="border: 1px solid #ccc;">  

## 3️⃣ Methodology and Key Steps  

### ➡️ 3.1 Dimensionality Reduction  
PCA was used to retain essential information and eliminate redundancy in the dataset. This step ensured efficient clustering of the data.  

### ➡️ 3.2 Optimal Clustering  
The optimal number of clusters was determined using the **Elbow Method**, where the Within-Cluster Sum of Squares (WCSS) was plotted against the number of clusters. The "elbow" observed at **4 clusters** indicated the most suitable balance between complexity and interpretability.  

### ➡️ 3.3 KMeans Clustering  
Using **KMeans with 4 clusters**, individuals were grouped into distinct segments that share common demographic and lifestyle characteristics.  

<hr style="border: 1px solid #ccc;">  

## 4️⃣ Findings from Clustering Analysis  
The general population was categorized into **4 key clusters**, described below:  

- **Cluster 0:**  
   - ✔ **Size:** Largest group  
   - ✔ **Demographics:** Dominated by middle-aged individuals  
   - ✔ **Key Features:** Strong inclination towards stability in residence duration  

- **Cluster 1:**  
   - ✔ **Size:** Medium-sized segment  
   - ✔ **Demographics:** Younger demographic, active in urban settings  
   - ✔ **Distinct Traits:** Interest in modern, technology-driven products  

- **Cluster 2:**  
   - ✔ **Size:** Smaller cluster  
   - ✔ **Key Behaviors:** Strong alignment with traditional values  
   - ✔ **Key Insights:** Higher representation in rural or suburban areas  

- **Cluster 3:**  
   - ✔ **Size:** Smallest cluster  
   - ✔ **Behavior:** High diversity in purchasing preferences, from luxury to budget-conscious items  

<hr style="border: 1px solid #ccc;">  

## 5️⃣ Business Implications  
The segmentation insights from the general population can support business strategies as follows:  

- 🎯 **Personalized Outreach:** Design marketing campaigns tailored to the needs and preferences of each cluster  
- 📈 **Enhanced Customer Acquisition:** Focus resources on clusters more likely to convert into loyal customers  
- 🌍 **Market Penetration:** Develop region-specific strategies for clusters predominantly residing in particular geographic areas  

<hr style="border: 1px solid #ccc;">  

## 6️⃣ Limitations and Recommendations  

- ⚠ **Data Imputation:** Mean imputation for missing values may oversimplify the dataset and overlook subtle trends  
- ⚠ **Clustering Assumptions:** The use of Euclidean distance in KMeans may not capture complex, non-linear relationships in the data  
- 🔎 **Further Exploration:** Incorporate additional clustering techniques, such as hierarchical clustering or DBSCAN, to validate and refine these results  

<hr style="border: 1px solid #ccc;">  

<p style="font-size:16px; color:#4a148c;"><strong>✅ This segmentation provides a foundational understanding of Germany's general population and guides subsequent analyses, including predictive modeling.</strong></p>
""", unsafe_allow_html=True)


# Customer Segmentation Report - CUSTOMERS
elif section == "👥 Customer Segmentation Report - CUSTOMERS":
    st.markdown("""
<h2 style="color:#a770a0; text-align:left;">👥 Customer Segmentation Final Report for Customer Data (CUSTOMERS)</h2>

## 1️⃣ Objective  
The aim of this report is to summarize the findings from the customer segmentation analysis performed on the `df_customers` dataset. This analysis utilized clustering methods to identify distinct customer groups, leveraging unsupervised learning techniques like Principal Component Analysis (PCA) and KMeans clustering.  

<hr style="border: 1px solid #ccc;">  

## 2️⃣ Dataset Overview  
The `df_customers` dataset includes attributes representing demographic and behavioral data of existing customers. After preprocessing, the dataset was scaled, imputed for missing values, and subjected to dimensionality reduction.  

✔ **Post-PCA Data Shape:** The reduced dataset contained **191,652 samples** with **329 principal components**, preserving 95% of the variance.  

<hr style="border: 1px solid #ccc;">  

## 3️⃣ Methodology and Key Steps  

### ➡️ 3.1 Dimensionality Reduction  
PCA was applied to reduce data dimensionality while retaining key patterns. This step addressed multicollinearity and improved computational efficiency.  

### ➡️ 3.2 Optimal Clustering  
The optimal number of clusters was determined using the **elbow method**, which plots within-cluster sum of squares (WCSS) for various values of *k*.  

### ➡️ 3.3 KMeans Clustering  
The **KMeans algorithm** was employed to partition customers into distinct clusters. Each cluster represents a unique segment with shared characteristics.  

<hr style="border: 1px solid #ccc;">  

## 4️⃣ Findings from Clustering Analysis  
The customer base was segmented into **4 primary clusters**, each described below:  

- **Cluster 0:**  
   - ✔ **Size:** Largest cluster  
   - ✔ **Demographics:** Predominantly older age groups  
   - ✔ **Key Features:** High variance in `LNR` and stability in `WOHNDAUER` (residence duration)  

- **Cluster 1:**  
   - ✔ **Size:** Moderate-sized segment  
   - ✔ **Demographics:** Diverse age distribution with higher income stability  
   - ✔ **Distinct Traits:** Consistently high representation of luxury buyers  

- **Cluster 2:**  
   - ✔ **Size:** Medium-sized cluster  
   - ✔ **Key Behaviors:** Active in purchasing insurance products  
   - ✔ **Geographic Pattern:** Predominantly Western Germany (`OST_WEST_KZ` = 'W')  

- **Cluster 3:**  
   - ✔ **Size:** Smallest group  
   - ✔ **Behavior:** Limited purchase activity but notable for unique niche product interests  

<hr style="border: 1px solid #ccc;">  

### ✅ Key Categorical Insights Across Clusters  
- Variables such as **`CAMEO_DEU_2015`** (socioeconomic classification) and **`D19_LETZTER_KAUF_BRANCHE`** (last purchase category) played a significant role in differentiating clusters.  

<hr style="border: 1px solid #ccc;">  

## 5️⃣ Business Implications  
This segmentation provides actionable insights for marketing strategies:  

- 🎯 **Targeted Campaigns:** Design cluster-specific marketing strategies to cater to the unique preferences and purchasing behaviors  
- 📊 **Resource Allocation:** Allocate resources efficiently to high-value clusters, such as those with higher income stability and active purchasing patterns  
- 🤝 **Customer Retention:** Use insights to enhance personalized experiences for the most loyal and lucrative customer segments  

<hr style="border: 1px solid #ccc;">  

## 6️⃣ Limitations and Recommendations  

- ⚠ **Data Limitations:** Missing data imputation may oversimplify real-world customer behaviors, potentially affecting the clustering outcome  
- ⚠ **Model Limitations:** KMeans' reliance on Euclidean distance may struggle with non-linear relationships. Exploring other clustering techniques like **DBSCAN** could yield additional insights  

<hr style="border: 1px solid #ccc;">  

<p style="font-size:16px; color:#4a148c;"><strong>✅ This segmentation serves as a foundation for predictive modeling in subsequent phases, aiming to predict campaign responses and further optimize customer engagement.</strong></p>
""", unsafe_allow_html=True)


# Prediction & Analysis
elif section == "📈 Prediction & Analysis":
    st.markdown('<h2 style="color:#a770a0;">📈 Prediction & Analysis</h2>', unsafe_allow_html=True)

    st.write("Upload a CSV file or use test data to predict positive response users.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df_test = pd.read_csv(uploaded_file, delimiter=';')
    elif os.path.exists(DATA_PATH):
        df_test = pd.read_csv(DATA_PATH, delimiter=';')
        st.write("Using default test dataset.")
    else:
        st.error("No data file found!")
        st.stop()

    df_train_feature_names = train_features
    #Data Cleaning
    cleaned_df_test = clean_dataframes(df_test)
 
    # Load the transformer
    #ct = joblib.load('model/column_transformer.pkl')
    preprocessed_df_test = preprocess_dataframe(cleaned_df_test, is_test=True, df_train_feature_names = df_train_feature_names)

    # Feature Scaling
    preprocessed_df_test_scaled = scaler.transform(preprocessed_df_test)

    predictions = model.predict(preprocessed_df_test_scaled)

    df_test['Prediction'] = predictions
    df_positive = df_test[df_test['Prediction'] == 1]
    st.write("### Predicted Positive Response Users:")
    st.dataframe(df_positive)

    csv = df_positive.to_csv(index=False).encode('utf-8')
    st.download_button("Download Positive Users CSV", csv, "positive_predicted_users.csv", "text/csv")

    #Printing No of Positive Response Results
    st.write("No. of Customers in Test Dataset(df_test) = ",df_test.shape[0])
    st.write("No. of Positive Response Results predicted = ",df_positive.shape[0])

# Conclusion and Key Outcomes
elif section == "✅ Conclusion and Key Outcomes":
    st.markdown("""
<h2 style="color:#a770a0; text-align:left;">✅ Conclusion and Key Outcomes</h2>

<p style="font-size:16px">
The predictive model successfully identified customers within the test dataset who are likely to respond positively to the upcoming marketing campaign.
</p>

### 🔎 Key Results:
- **Total customers in test dataset:** 42,833  
- **Predicted positive responses:** 6,117  

### 💡 Business Interpretation:
- Approximately **14.3%** of the test population (6,117 out of 42,833) are flagged as high-potential responders.  
- These individuals can be **targeted with tailored marketing efforts**, allowing the company to optimize resources and maximize ROI.  
- By focusing on these identified segments, the marketing team can **significantly reduce customer acquisition costs** while improving **conversion rates**.  

### 📈 Strategic Impact for Arvato Financial Solutions:
- The combination of **customer segmentation** and **predictive modeling** provides a robust framework for **data-driven decision-making**.  
- This approach enables **smarter targeting**, **higher customer engagement**, and **increased profitability** in future campaigns.  

<p style="color:#800080; font-weight:bold">
📂 The identified high-potential customers have been exported to <code>positive_predicted_users.csv</code> for immediate use in marketing initiatives.
</p>
""", unsafe_allow_html=True)

