# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:41:26 2021

@author: User
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#%%
data = pd.read_csv('C:\\Users\\User\\Desktop\\kmeans\\test.csv')
data.head()
#############################################
radiofeature = data.loc[:, 'ADC(10-3mm2/s)':'Kpa']
label = data['Customlabel']
radiofeature.insert(0, "Customlabel", label, True)
radiofeature['Customlabel'] = radiofeature['Customlabel'].map({'Normal': 0, 'Fat': 1, 'Fibrosis': 2})
radiofeature.head()

#################################################
sns.countplot(data['Customlabel'])
################################################

#radiofeature.replace('inf', np.nan)
#nullvalue = radiofeature.isnull().sum()

'''
i = 0
nullindex = []

for i in range(len(nullvalue)):
    if not nullvalue[i] == 0:
        nullindex.append(i)
'''

dropRF = radiofeature.dropna(axis=1)
dropRF = radiofeature.dropna(axis=0)

#########################################################
train, test = train_test_split(dropRF, test_size=0.2, random_state=2019)

x_train = train.drop(['CustomLabel'], axis=1)
y_train = train.CustomLabel

x_test = test.drop(['CustomLabel'], axis=1)
y_test = test.CustomLabel

print(len(train), len(test))


###################################################
#%%
#SVM
model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('SVM: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

#%%
#DecisionTree
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

#%%
#KNeighbors
model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('KNeighborsClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

#%%
#LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=2000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('LogisticRegression: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

#%%
#RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))


#%%
#Compute Feature Importances
features = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)

print(features)

#%%
# Extract Top 5 Features
top_5_features = features.keys()[:10]

print(top_5_features)

#%%
#SVM with top 5
model = svm.SVC(gamma='scale')
model.fit(x_train[top_5_features], y_train)

y_pred = model.predict(x_test[top_5_features])

print('SVM(Top 5): %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

#%%

models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=1000)
    }


for name, model in models.items():
    model.fit(x_train[top_5_features], y_train)
    y_pred = model.predict(x_test[top_5_features])
    print('%s: %.2f' % (name, (metrics.accuracy_score(y_pred, y_test) * 100)))


#%%
#Cross Validation
model = svm.SVC(gamma='scale')

cv = KFold(n_splits=5, random_state = 0, shuffle = True)

accs = cross_val_score(model, radiofeature[top_5_features], radiofeature.CustomLabel, cv=cv)

print(accs)

#%%
#testModel
models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

cv = KFold(n_splits=5, random_state=2019)

for name, model in models.items():
    scores = cross_val_score(model, radiofeature[top_5_features], radiofeature.CustomLabel, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))

#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(radiofeature[top_5_features])

models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

cv = KFold(n_splits=3, random_state=2019, shuffle = True)

for name, model in models.items():
    scores = cross_val_score(model, scaled_data, radiofeature.CustomLabel, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))



from keras import utils, models, layers, optimizers
from keras.models import Model, load_model, Sequential

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Add
from keras.callbacks import ReduceLROnPlateau


# define the keras model
model = Sequential()
model.add(Dense(2048, input_dim=1213, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(optimizer = optimizers.Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
model.summary()


history = model.fit(x_train, y_train, epochs=300, batch_size=64, validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)])

trained_weight = "C:\\Users\\User\\Desktop\\weight.hdf5"
model.save_weights(trained_weight)
        
fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('binary_accuracy')
ax[0, 1].plot(history.history['binary_accuracy'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val__accuracy')
ax[1, 1].plot(history.history['val_binary_accuracy'], 'b--')