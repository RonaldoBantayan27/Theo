
**Breast Cancer Diagnostics**  

**1.	Problem Statement**
   
Early diagnosis of breast cancer saves lives.

In a medical context like breast cancer diagnosis, a False Negative is the most dangerous error because it means a patient has cancer but is told they are healthy.  In order to mitigate this, several machine learning and artificial intelligence models are built to predict breast cancer to an accuracy that reduces the probability of False Negatives to almost zero although the number of false positives tends to go up. The number of False Positives is further minimized. However, where lives are on the line, it is prudent to err on the side of caution.

This analysis is based on a tabular dataset where the features are calculated from a digitized image of a fine needle aspirate (FNA) of a breast mass.     

It is a supervised learning dataset used for binary classification: predicting whether a breast tumor is malignant or benign based on features computed from a digitized image of a breast mass.     

-Overview of dataset-    
Source: UCI Machine Learning Repository     
Original creators: Dr. William H. Wolberg and collaborators     
Common use: teaching classification, feature selection, model evaluation, medical ML examples

**2.	Model Outcomes or Predictions**
   
The type of learning is classification. The expected output of the selected model is the prediction of benign or malignant cases. **False Negatives** are reduced to zero by tuning the threshold. Feature selection brings the number of **False Positives** to a minimum. 

Supervised machine learning algorithms are used to build predictive models. 

**3.	Data**
   
The **Wisconsin Breast Cancer** dataset is a classic binary classification dataset where **30 numeric cell-image features** are used to predict whether tumors are malignant or benign.

While the data is suitable for teaching machine learning and artificial intelligence, it is relatively small and highly curated, meaning that the data is less than random and less representative of modern hospital pipelines. The data is simpler than real clinical data and not enough alone for production medical diagnosis.

For this particular dataset, the classes are fairly separable, therefore, accuracy is often > 95% and with a proper train/test split, many models can achieve strong performance.

The **label**	means the following:     

**Malignant**	Cancerous tumor     
**Benign**	Non-cancerous tumor

The typical distribution is **357** benign and **212** malignant so the dataset is only **mildly imbalanced**.

**What the Features Represent**

They describe cell nuclei characteristics such as:

- size
- shape
- smoothness
- compactness
- symmetry
- texture
- perimeter
- area

There are **10 core measurements**, each summarized in 3 ways:

1.  radius
2.  texture
3.  perimeter
4.  area
5.  smoothness
6.  compactness
7.  concavity
8.  concave points
9.  symmetry
10. fractal dimension

The **suffixes** mean as follows:     
**mean**	average value     
**se**	standard error     
**worst**	worst / largest value

**4.	Data Preprocessing/Preparation**
   
The column **id** is dropped because it contains unique values which will not affect the performance of the model.  
 
The column '**Unnamed: 32**' containing no information is also dropped.  

The target variables are renamed '**0**' and '**1**'.

There are no duplicate rows.      

Ensure lowercase, replace spaces in column names with _, and strip leading/trailing whitespace.   

**Histograms**    

There are outliers. Some of the data are right skewed.

**Histograms of Classes**

The difference in the mean and median of the benign and malignant cases lend to the strength of the predictive property of the features.

**Box Plots**   

The outliers are not removed to preserve data. Most of the positive and negative classes of the features have diverging medians which means that there are quite a number of features which are good predictors.

**Medians**     

Based on percentage differences on medians between the classes, the top predictors are analyzed.

**Heat Map**

Heat maps show the features strongly correlated with '**diagnosis**'.

**Pair Plot**    

The classes are distinct and separable which make most of the features to be very good predictors. The same is true for the diagonals distributions of benign and malignant cases. 

**5.	Modeling**
   
Eight (8) supervised machine learning classification algorithms are used to build predictive models:  
`LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, SVC, RandomForestClassifier, Keras Classifier, GaussianNB Classifier` and `XGBoost Classifier`. 

The parameter 'class_weight="balanced"' is used to address the class imbalance. For the KNeighbors Classifier, SMOTE (Synthetic Minority Over-sampling Technique) is used to address the class imbalance. 
    
The features are scaled as required.      

The model parameters are optimized using the appropriate grid search function. 

Scoring is based on 'recall' to try to reduce the number of **False Negatives** to zero.     

In line with this, the threshold is tuned to the absolute minimum threshold.  

Feature selection brings **False Positives** to a minimum. Feature selection reduces noise and improves the interpretation of feature importance.
Feature selection is about removing noise, improving interpretability, improving logistic regression stability, and slightly reducing overfitting.  

The model performance metrics and feature importances are output and compared with other models in a separate notebook.

Sample predictions using the model are demonstrated.  

**6.	Model Evaluation**
   
The GenAI model provided the following feature importance analysis on the results of the top four (4) classification models which are averaged.  

"Feature Importance Analysis for Breast Cancer Diagnostics The provided feature importance data suggests a ranking of the most critical features used in a breast cancer diagnosis model. The top features are: ### Top 2 Features: 1. texture_worst (0.640281): This feature, which measures the worst-case texture of the tumor, appears to be the most critical factor in distinguishing between cancerous and non-cancerous tissues. A high importance score indicates that the model relies heavily on this feature to make diagnoses. 2. perimeter_worst (0.664062): The perimeter of the worst-case shape of the tumor also plays a significant role in breast cancer diagnostics. Models that incorporate this feature are likely to improve their accuracy in predicting the presence of cancer. ### Other Important Features: 1. radius_worst (0.448413): The radius of the worst-case shape of the tumor is also an important factor, ranking third in the list. This suggests that the size and shape of the tumor are crucial indicators of cancerous tissues. 2. perimeter_mean (0.414885): While not as critical as the worst-case perimeter, the average perimeter of the tumor still contributes significantly to the model's decision-making process. 3. concavity_mean (0.400984): The concavity of the tumor, measured on average, also ranks as one of the most critical features. This highlights the importance of the tumor's shape and structure in diagnosing breast cancer. ### Less Important Features: 1. area_mean (0.383176): The average area of the tumor seems to be a less critical factor in breast cancer diagnostics, ranking sixth in the list. 2. area_worst (0.378347): Similarly, the worst-case area of the tumor has a relatively lower importance score. 3. radius_mean (0.350140): The average radius of the tumor is another feature with a moderate importance score. 4. smoothness_mean (0.344295): The smoothness of the tumor, on average, ranks ninth in the list, indicating its relatively lower impact on the model's decision-making process. Conclusion: The analysis suggests that the breast cancer diagnosis model relies heavily on the worst-case texture, perimeter, and radius of the tumor to make accurate predictions. While other features, such as concavity and perimeter on average, are also significant, they have a lower impact on the model's decision-making process."

**Overall Model Summary**     

Based on the metrics of ROC_AUC , Accuracy, Precision, Recall, and False_Positives, `RandomForestClassifier` and `XGBoost Classifier` are tied for the top spot (see comparison table below). 

For this particular dataset, the `RandomForestClassifier` and `XGBoost Classifier` are the recommended machine learning algorithms.

| Model | ROC_AUC | Accuracy | Precision | Recall | False_Positives |
|:---------|:---------|:---------|:---------|:---------|:---------|
| Random Forest | 0.997 | 0.974 | 0.933 | 1.0 | 3 | 
| XGBoost  | 0.997 | 0.974 | 0.933 | 1.0 | 3 | 
| GaussianNB | 0.991 | 0.956 | 0.894 | 1.0 | 5 | 
| SVC | 0.993 | 0.947 | 0.875 | 1.0 | 6 | 
| Decision Tree | 0.988 | 0.939 | 0.957 | 1.0 | 7 | 
| Logistic Regression | 0.995 | 0.930 | 0.840 | 1.0 | 8 | 
| K-Neighbors | 0.995 | 0.930 | 0.840 | 1.0 | 8 | 
| Keras | 0.994 | 0.868 | 0.737 | 1.0 | 15 |

**Next Steps and Further Recommendations**  

- Confirm the models suit real and larger clinical data.    
- Continue model development to further reduce false positives.

**Notebook**    
You can view the full analysis here:

[Exploratory Data Analysis]01_EDA_Breast_Cancer_Wisconsin_Diagnostic


**Reference:** 
Ronaldo Bantayan (Author) Email: one01bant@yahoo.com     


