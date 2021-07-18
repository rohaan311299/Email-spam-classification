# Importing the libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reading the data
df=pd.read_csv("./emails.csv")
print(df.head(10))

# getting some basic info about the data
print(df.isnull().sum())

print(df.describe())

print(df.corr())

X=df.iloc[:, 1:3001]
y=df.iloc[:, -1].values

# splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Using Naive Bayes model on the dataset
mnb=MultinomialNB(alpha=1.9)
mnb.fit(X_train, y_train)
mnb_pred=mnb.predict(X_test)
print("Accuracy Score for Naive Bayes: ", accuracy_score(mnb_pred, y_test))

# Using SVC model on the dataset
svc=SVC(C=1.0, kernel="rbf", gamma="auto")
svc.fit(X_train, y_train)
svc_pred=svc.predict(X_test)
print("Accuracy Score for SVC: ", accuracy_score(svc_pred, y_test))

# Using Random Forest Classification model on the dataset
rfc=RandomForestClassifier(n_estimators=10, criterion="gini")
rfc.fit(X_train, y_train)
rfc_pred=rfc.predict(X_test)
print("Accuracy Score for Random Forest Classification: ", accuracy_score(rfc_pred, y_test))
