#import the required packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
#Reading the dataset “Iris.csv”.

data = pd.read_csv("data.csv")


#Displaying up the top rows of the dataset with their columns
data.head()

#Displaying the number of columns and names of the columns.
data.columns

#Displaying the shape of the dataset.
data.shape

#Display the whole dataset
print(data)


data['sepal_length'].max()
data['sepal_length'].min()



data['sepal_width'].max()
data['sepal_width'].min()

data['petal_length'].max()
data['petal_length'].min()

data['petal_width'].max()
data['petal_width'].min()

#Train-Test Split

train,test = train_test_split(data,test_size=0.4,stratify=data['species'],random_state=42)

#splitting the data
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species


#Gaussian Naive Bayes Classifier
model_nb = GaussianNB()
y_pred_nb = model_nb.fit(X_train,y_train).predict(X_test)

print('The accuracy of the Gaussian Naive Bayes Classifier on test data is',"{:3f}".format(metrics.accuracy_score(y_pred_nb,y_test)))

#LDA Classifier

model_lda = LinearDiscriminantAnalysis()
y_pred_lda =  model_lda.fit(X_train,y_train).predict(X_test)
print('The accuracy of the LDA classifier on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred_lda,y_test)))

#SVC with linear kernel

model_linear_svc = SVC(kernel='linear').fit(X_train,y_train)
y_pred_linear_svc = model_linear_svc.predict(X_test)
print('The accuracy of the linear SVC is',"{:.3f}".format(metrics.accuracy_score(y_pred_linear_svc,y_test)))


#creating a pickle file for the classifier

filename = 'SVC-linear-model.pkl'

pickle.dump(model_linear_svc,open(filename,'wb'))

