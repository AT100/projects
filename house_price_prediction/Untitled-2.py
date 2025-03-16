# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
#load the data 
data = pd.read_csv('Social_Network_Ads.csv')

data.head()

# %%
data.info()

# %%
data.shape

# %%
data.describe(include="all")

# %%
#initial exploration
#re-checking missing values
data.isna().sum()

# %%
#unique values present in columns
for col in data.columns:
    print(f"Unique values in {col}: {data[col].nunique()}")

# %%
# drop the user id field which is unique identifier for each user
data.drop('User ID', inplace=True, axis=1)

# %%
# check if there are duplicate rows
data[data.duplicated()]

# %%
# drop duplicate rows
data.drop_duplicates(inplace=True)

# %%
data.columns

# %%
#detecting outliers in numerical columns
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data, x='Age')
plt.title("Boxplot for Age")

plt.subplot(1, 2, 2)
sns.boxplot(data, x='EstimatedSalary')
plt.title("Boxplot for EstimatedSalary")


# %%
# visualize target
sns.countplot(data, x='Purchased', hue='Purchased', palette="Greens", legend=False)

# %%
sns.countplot(data, x='Purchased', hue='Purchased', palette="Greens", legend=False, stat='percent')

# %%
# visualize gender field
sns.countplot(data, x='Purchased', hue='Gender', palette="Blues", stat='percent')

# %%
sns.countplot(data, x='Gender')

# %%
#analyze the numerical variables
g = sns.PairGrid(data, hue="Purchased", palette="Greens", diag_sharey=False)
g.map_diag(sns.histplot, hue=None)
g.map_lower(sns.scatterplot)
g.map_upper(sns.kdeplot)
g.add_legend()

# %%
#feature engineering
import numpy as np

bin_edges = np.histogram_bin_edges(data['Age'], bins="fd")
print(f"Bin edges for Age: {np.round(bin_edges).astype(int)}")
print(f"Number of bins: {len(np.round(bin_edges).astype(int))}")

# %%
sns.histplot(data['Age'], bins=10)

# %%
data['Age'] = pd.cut(data['Age'], bins=np.round(bin_edges).astype(int))

# %%
data['Age']

# %%
#preprocessing pipelines
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

num_col = ['EstimatedSalary']
num_pipe = make_pipeline(StandardScaler())

cat_cols=['Age', 'Gender']
cat_pipe = make_pipeline(OneHotEncoder())

preprocess = ColumnTransformer(transformers=[
    ('num_pipe', num_pipe, num_col),
    ('cat_pipe', cat_pipe, cat_cols)
])

# %%
# split data into train, test
from sklearn.model_selection import train_test_split

X = data[num_col + cat_cols]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# %%
#Modeling 
#Logistic Regression
from sklearn.linear_model import LogisticRegression

log_model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('log_model', LogisticRegression(random_state=42, class_weight="balanced"))
])

log_model.fit(X_train, y_train)

# %%
# Predict on train and test data
y_pred_train = log_model.predict(X_train)
y_pred_test = log_model.predict(X_test)

# %%
# evaluation
from sklearn import metrics

# for test data
def eval_metrics(test, pred):
#print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_train)}")
    print(f"\nClassification report for Tesing dataset: \n{metrics.classification_report(y_test, y_pred_test)}")
    cm = metrics.confusion_matrix(y_test, y_pred_test)
    metrics.ConfusionMatrixDisplay(cm).plot()
    print(f"ROC AUC score: {metrics.roc_auc_score(y_test, y_pred_test)}")
    metrics.RocCurveDisplay.from_predictions(y_test, y_pred_test)

# %%

print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_train)}")
eval_metrics(y_test, y_pred_test)

# %%
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(log_model, X, y, cv=10, scoring='roc_auc')
print(f"CV scores for logistic regression model is {cv_scores.mean()}")

# %%
from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('rf_model', RandomForestClassifier(random_state=42))
])

rf_model.fit(X_train, y_train)

# %%
# Predict on train and test data
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='roc_auc')
print(f"CV scores for Random Forest Classifier is {cv_scores.mean()}")


#eval results
print(f"\n\nClassification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_train)}")
eval_metrics(y_test, y_pred_test)


