# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:47:38 2021

@author: doguilmak

dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29

"""
#%%
# 1. Importing Libraries

from sklearn.impute import SimpleImputer
from keras.models import load_model
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
data = pd.read_csv('heart.data', header=None, sep=" ")

# 2.2. Looking for anomalies and duplicated datas
print(data.info())
print(data.isnull().sum())
print("\n", data.head(10))
print("\n", data.describe().T)
print("\n{} duplicated.".format(data.duplicated().sum()))

# 2.3. Looking for '?' mark
from sklearn.preprocessing import LabelEncoder
data = data.apply(LabelEncoder().fit_transform)
print("data:\n", data)
data.replace('?', -999999, inplace=True)

imputer = SimpleImputer(missing_values= -999999, strategy='mean')
newData = imputer.fit_transform(data)

# 2.3. Determination of dependent and independent variables
X = newData[:, 0:13]
y = newData[:, 13]

#%%
# 3 Artificial Neural Network

# 3.1 Loading Created Model
model = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
model.summary()
"""
# 3.3. Creating layers
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# Input layer(nöron) + Output layer / 2 ----- (11 + 1) / 2
# First hidden layer:
model.add(Dense(6, init="uniform", activation="relu", input_dim=13))
# Second hidden layer:
model.add(Dense(6, init="uniform", activation="relu"))
# Output layer:
model.add(Dense(1, init="uniform", activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_history = model.fit(X, y, epochs=128, batch_size=32, validation_split=0.13)

# Plot accuracy and val_accuracy
print(model_history.history.keys())
model.summary()
model.save('model.h5')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""

# 3.4. Visualize ANN
"""
from ann_visualizer.visualize import ann_viz

try:
    ann_viz(model, view=True, title="", filename="ann")
except:
    print("PDF saved.")
"""

# 3.5. Predicting class
predict = np.array([70.0, 1.0, 4.0, 130.0, 322.0, 0.0, 2.0, 109.0, 0.0, 2.4, 2.0, 3.0, 3.0]).reshape(1, 13)
print(f'Model predicted class as {model.predict_classes(predict)}.')

#%%   
# 4. XGBoost

from xgboost import XGBClassifier
classifier= XGBClassifier()

# 4.1. Splitting test and train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 4.2. Scaling datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 4.3. Prediction
print('\nXGBoost Prediction')
predict_model_XGBoost = newData[0:1, 1:17]
if classifier.predict(predict_model_XGBoost) == 0:
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')
else:
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')

# 4.4. Confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test)
print("\nConfusion Matrix(XGBoost):\n", cm2)

from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")


end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
