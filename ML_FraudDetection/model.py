import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import pickle

#Reading the csv file
df=pd.read_csv(r"C:/Documents/git/ML_FraudDetection/MLfrauddetection.csv")

print(df)
print(df.columns)

df.drop(['isFlaggedFraud'],axis=1,inplace=True)

print(df["isFraud"].value_counts())

df.loc[df["isFraud"]==0,"isFraud"] = "is not Fraud"
df.loc[df["isFraud"]==1,"isFraud"] = "is Fraud"

df.drop(["nameOrig","nameDest"], axis=1, inplace=True)
df.drop(["type"], axis=1, inplace=True)
print(df)

from scipy import stats
print(stats.mode(df["amount"]))
print(np.mean(df["amount"]))

df["amount"]=np.log(df["amount"])

print(df)

# Dividing the dataset into dependent and independent y and x respectively
x=df.drop("isFraud",axis=1)
y=df["isFraud"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
rfc=RandomForestClassifier() 
rfc.fit(x_train, y_train) 

y_test_predict=rfc.predict(x_test)
test_accuracy=accuracy_score(y_test,y_test_predict)
print(test_accuracy)

y_train_predict=rfc.predict(x_train)
train_accuracy=accuracy_score(y_train,y_train_predict)
print(train_accuracy)

print(pd.crosstab(y_test,y_test_predict))

print(classification_report(y_test,y_test_predict))

pickle.dump(rfc,open("model.pkl","wb"))