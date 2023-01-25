# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:04:00 2022

@author: ASUS
"""

import numpy as np
import pandas as pd

df=pd.read_csv('cclass.csv') 

df.info()
'''
i have in datset 
values float:1
values int :3
values object :3
'''
o=df['year'].value_counts()
m=df['model'].value_counts()
a=df.isnull().sum()#no missing value

#encoding categorical for number
df= pd.get_dummies(df,columns=['fuelType'],drop_first=True)
df= pd.get_dummies(df,columns=['year'],drop_first=True)
df= pd.get_dummies(df,columns=['transmission'],drop_first=True)
#df= pd.get_dummies(df,columns=['model'],drop_first=True)



#split data

X = df.loc[:,df.columns.difference(['price'],sort=False)].values
y = df.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

'''
labelEncoder=LabelEncoder()
X[:,0]=labelEncoder.fit_transform(X[:,0])
'''

columntransformer=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder= 'passthrough')
#columntransformer=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder= 'passthrough')
X=np.array(columntransformer.fit_transform(X))



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=(0))

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)
y_pred=linearRegression.predict(X_test)


#A function for the account of the Accuracy for data

def accuracy(a,b,n):
  import sklearn.metrics as sm
  print(f"\n------------ alograthim  {n} ----------------")

 # print(f"\n################# ACCURACY TEST NO.{c} ###################")
  
  print("Mean absolute error =", round(sm.mean_absolute_error(a,b), 2)) 
  print("Mean squared error =", round(sm.mean_squared_error(a,b), 2)) 
  print("Median absolute error =", round(sm.median_absolute_error(a, b), 2)) 
  print("Explain variance score =", round(sm.explained_variance_score(a, b), 2)) 
    
 
accuracy(y_test,y_pred,'linear')


import statsmodels.api as sm
X=np.append(np.ones((len(X),1)).astype(int),values=X,axis=1)#CONSTANT
#import statsmodels.api as sm


#A function to determine the veggies that have an effect on the results
def reg_ols(X,y):
    columns=list(range(X.shape[1]))
    
    for i in range(X.shape[1]):
        X_opt=np.array(X[:,columns],dtype=float) 
        regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
        pvalues = list(regressor_ols.pvalues)
        d=max(pvalues)
        if (d>0.05):
            for k in range(len(pvalues)):
                if(pvalues[k] == d):
                    del(columns[k])  
    
    return(X_opt,regressor_ols)

X_opt,regressor_ols=reg_ols(X, y)
regressor_ols.summary()


from sklearn.model_selection import train_test_split
X_train_opt,X_test_opt,y_train_opt,y_test_opt=train_test_split(X_opt,y,test_size=0.2,random_state=(0))

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train_opt,y_train_opt)
y_pred1=linearRegression.predict(X_test_opt)


#function accurcy 
accuracy(y_test_opt,y_pred1,'X_opt')

#################################

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)
y_pred=linearRegression.predict(X_test)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =3)
X_poly = poly_reg.fit_transform(X_test)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_test)

y_pred_2=lin_reg_2.predict(poly_reg.fit_transform(X_test))

accuracy(y_test,y_pred_2,'PolynomialFeatures')


######################################

from sklearn.preprocessing import StandardScaler

standardscaler = StandardScaler()

X = standardscaler.fit_transform(X)

y = np.ravel(standardscaler.fit_transform(y.reshape(-1,1)))

from sklearn.svm import SVR

regressor=SVR(kernel='linear')

from sklearn.model_selection import train_test_split

X_linear_train,X_linear_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_linear_train,y_train)

y_linear_pred=regressor.predict(X_linear_test)


accuracy(y_test,y_linear_pred,'svr linear')

######################################

#RBF:
    
regressor=SVR()

from sklearn.model_selection import train_test_split

X_rbf_train,X_rbf_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_rbf_train,y_train)

y_rbf_pred=regressor.predict(X_rbf_test)


accuracy(y_test,y_rbf_pred,'svr /rbf')



regressor=SVR(kernel='poly',degree=4)

from sklearn.model_selection import train_test_split

X_poly_train,X_poly_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_poly_train,y_train)

y_poly_pred=regressor.predict(X_poly_test)

accuracy(y_test,y_poly_pred,'poly')


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

X_forset_train,X_forest_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor=RandomForestRegressor(n_estimators=40,random_state=(0))

regressor.fit(X_forset_train,y_train)

y_forest_pred=regressor.predict(X_forest_test)

    
accuracy(y_test,y_forest_pred,'RandomForestRegressor')

    


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X_tree_train,X_tree_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
reg=DecisionTreeRegressor(random_state=(42))
reg.fit(X_tree_train,y_train)
y_tree_pred=reg.predict(X_tree_test)

accuracy(y_test,y_tree_pred,'DecisionTreeRegressor')










