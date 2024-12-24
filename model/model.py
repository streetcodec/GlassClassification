#IMPORT LIBRARIES
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("glass.csv")

x=data.iloc[:,:-1].values 
y=data.iloc[:,9].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, 
                                          random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
# xtest=sc.fit(xtest)
x_test=sc_x.transform(x_test)

#creating model
from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(n_estimators=300,criterion='entropy',random_state=42)
cls.fit(x_train,y_train)


print("Accuracy is ",cls.score(x_test,y_test)*100 , '%')

filename='finalized_model.sav'
joblib.dump(cls, filename)