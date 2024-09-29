#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, Binarizer, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/mnt/data/Food Waste data and research - by country.csv'
data = pd.read_csv('Food Waste data and research - by country.csv')

print(data.head())

print("Missing values:\n", data.isnull().sum())
data.fillna(data.mean(), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)

label_encoder = LabelEncoder()
data['Confidence in estimate'] = label_encoder.fit_transform(data['Confidence in estimate'])
data = pd.get_dummies(data, columns=['Region'], drop_first=True)

features = data.drop(columns=['Country', 'Source', 'combined figures (kg/capita/year)'])
target = data['combined figures (kg/capita/year)']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

normalizer = Normalizer()
features_normalized = normalizer.fit_transform(features)

binarizer = Binarizer()
features_binarized = binarizer.fit_transform(features)

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

model_80 = LinearRegression().fit(X_train_80, y_train_80)
model_70 = LinearRegression().fit(X_train_70, y_train_70)

y_pred_80 = model_80.predict(X_test_20)
y_pred_70 = model_70.predict(X_test_30)
mse_80 = mean_squared_error(y_test_20, y_pred_80)
mse_70 = mean_squared_error(y_test_30, y_pred_70)

print(f"80:20 MSE: {mse_80:.2f}")
print(f"70:30 MSE: {mse_70:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = pd.DataFrame(features_scaled).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

selected_features = features_scaled[:, [0, 1, 3]]

print("Model Coefficients:", model_80.coef_)
print("Model Intercept:", model_80.intercept_)

import numpy as np

random_indices = np.random.choice(len(X_test_20), 5, replace=False)
for i in random_indices:
    print(f"Actual: {y_test_20.iloc[i]}, Predicted: {y_pred_80[i]}")

train_errors = []
test_errors = []

for m in range(1, len(X_train_80)):
    model_partial = LinearRegression().fit(X_train_80[:m], y_train_80[:m])
    train_errors.append(mean_squared_error(y_train_80[:m], model_partial.predict(X_train_80[:m])))
    test_errors.append(mean_squared_error(y_test_20, model_partial.predict(X_test_20)))

plt.plot(train_errors, label='Training Error')
plt.plot(test_errors, label='Testing Error')
plt.legend()
plt.title('Overfitting/Underfitting Analysis')
plt.show()


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

penalties = ['none', 'l1', 'l2', 'elasticnet']

for penalty in penalties:
    if penalty == 'elasticnet':
        logistic_model = LogisticRegression(penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=5000)
    elif penalty == 'l1':
        logistic_model = LogisticRegression(penalty=penalty, solver='saga', max_iter=5000)
    else:
        logistic_model = LogisticRegression(penalty=penalty, solver='lbfgs' if penalty == 'l2' else 'saga', max_iter=5000)

    logistic_model.fit(X_train_80, y_train_80)

    y_pred_80 = logistic_model.predict(X_test_20)
    accuracy = accuracy_score(y_test_20, y_pred_80)

    print(f"Penalty: {penalty}, Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




