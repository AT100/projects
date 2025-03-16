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
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.countplot(data, x='Purchased', hue='Purchased', legend=False, palette="Greens")
plt.title("Countplot for Purchased")

plt.subplot(1, 2, 2)
sns.countplot(data, x='Purchased', hue='Purchased', legend=False, palette="Greens", stat='percent')
plt.title("Count (%) for Purchased")



# %%
# visualize gender field
# visualize target
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.countplot(data, x='Gender', hue='Gender', legend=False, palette="Blues", stat='percent')
plt.title("Countplot for Gender")

plt.subplot(1, 2, 2)
sns.countplot(data, x='Purchased', hue='Gender', legend=True, palette="Blues")
plt.title("Countplot of Purchased based on Gender")


# %%
#analyze the numerical variables
plt.figure(figsize=(12,8))
g = sns.PairGrid(data, hue="Purchased", palette="Greens", diag_sharey=False, height=3)
g.map_diag(sns.histplot, hue=None, kde=True)
g.map_lower(sns.scatterplot)
g.map_upper(sns.kdeplot)
g.add_legend()
plt.show()

# %%
#detecting outliers in numerical columns
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data, x='Purchased', y='Age', hue='Purchased')
plt.title("Boxplot for Purchased by Age")

plt.subplot(1, 2, 2)
sns.boxplot(data, x='Purchased', y='EstimatedSalary', hue='Purchased')
plt.title("Boxplot for Purchased by EstimatedSalary")


# %%
data['Age'].describe()

# %%
bins=[18, 26, 40, 52, 60]
labels=['young adult', 'adult', 'middle aged', 'senior']
data['Age_groups'] = pd.cut(data['Age'], bins=bins, labels=labels)
data['Age_groups']

# %%
#detecting outliers in numerical columns
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data, x='Age_groups', y='EstimatedSalary', hue='Age_groups', palette="Blues")
plt.title("Boxplot for Age_groups by EstimatedSalary")

plt.subplot(1, 2, 2)
sns.countplot(data, x='Age_groups', hue='Purchased', palette="Greens")
plt.title("Boxplot for Age_groups by Purchased")

# %%
#preprocessing pipelines
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

num_col = ['EstimatedSalary']
num_pipe = make_pipeline(StandardScaler())

cat_cols=['Age_groups', 'Gender']
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models={
    'Logistic Regression': LogisticRegression(),
    'Random forest classifier': RandomForestClassifier()
}

pipelines = {name: Pipeline(steps=[('preprocess', preprocess), ('classifier', model)]) for name, model in models.items()}

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for name, pipe in pipelines.items():
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    print(f"\nCV AUC score for {name}: {cv_scores.mean()}")
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"CV accuracy score for {name}: {cv_scores.mean()}")

# %%
rf_model = pipelines['Random forest classifier']
rf_model.fit(X_train, y_train)

# %%
# Predict on train and test data
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)
y_probs_train = rf_model.predict_proba(X_train)[:, 1]
y_probs = rf_model.predict_proba(X_test)[:, 1]

# %%
# evaluation
from sklearn import metrics

# for test data
def eval_metrics(test, pred):
#print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_train)}")
    print(f"\nClassification report for Tesing dataset: \n{metrics.classification_report(test, pred)}")
    cm = metrics.confusion_matrix(test, pred)
    metrics.ConfusionMatrixDisplay(cm).plot()

def eval_prob(test, probs):
    print(f"ROC AUC score: {metrics.roc_auc_score(test, probs)}")
    # metrics.RocCurveDisplay.from_predictions(test, probs)
    # plt.plot([0,1], [0,1], linestyle="--", label = "chance level")
    # plt.legend()

    metrics.PrecisionRecallDisplay.from_predictions(test, probs)
    metrics.balanced_accuracy_score

# %%

print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_train)}")
eval_metrics(y_test, y_pred_test)
eval_prob(y_test, y_probs)

# %%
import numpy as np
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_probs)
f1_scores = 2*(precision*recall)/(precision+recall)
threshold = thresholds[np.argmax(f1_scores)]
threshold

# %%
y_pred_new_threshold_train = (y_probs_train >= 0.7).astype(int)
y_pred_new_threshold = (y_probs >= 0.7).astype(int)

print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_new_threshold_train)}")

eval_metrics(y_test, y_pred_new_threshold)

# %%
from sklearn.model_selection import GridSearchCV

param={
    'classifier__max_depth':[6,8,10,12,15],
    'classifier__min_samples_leaf':[2, 5, 6, 8, 10]
}

grid = GridSearchCV(estimator=rf_model, param_grid=param, cv=cv, scoring='roc_auc')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

y_probs_train = best_model.predict_proba(X_train)[:, 1]
y_probs = best_model.predict_proba(X_test)[:, 1]


print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_pred_train)}")
eval_metrics(y_test, y_pred_test)
eval_prob(y_test, y_probs)

# %%
fpr, tpr, thresholds = metrics.precision_recall_curve(y_test, y_probs)
f1_scores = 2*(precision*recall)/(precision+recall)
threshold = thresholds[np.argmax(f1_scores)]
threshold

# %%
y_threshold_train=(y_probs_train>0.6).astype(int)
y_threshold=(y_probs>0.6).astype(int)

print(f"Classification report for Training dataset: \n{metrics.classification_report(y_train, y_threshold_train)}")
eval_metrics(y_test, y_threshold)


