#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sklearn
#from sklearn.cross_validation import ShuffleSplit
#from sklearn.model_selection import ShuffleSplit
#import sklearn.model_selection import learning_curve
#import sklearn.model_selection.learning_curve #experimenting the accessbility of other modules
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import train_test_split
from visuals import * #visuals.py

get_ipython().run_line_magic('matplotlib', 'inline #for better presentation')

df = pd.read_csv('housing.txt')
features = df.drop('MEDV', axis = 1)
    
print("The Boston housing dataset has {} data points. Each data point has {} variables.".format(*df.shape))
#lets have a review at our data set
print("A look at the data");df[:10] 
print("A look at the features as well"); features 
#now working on data distribution
import matplotlib.pyplot as plt

f = plt.figure()
one = f.add_subplot(1, 1, 1)
one.hist(df['RM'], bins = 30)  #using bins=30 here instead of usual conventional bin=15 
plt.title("1: Average no of rooms")
plt.xlabel("RM");plt.ylabel("freq");plt.show() #graph 1

f = plt.figure()
two = f.add_subplot(1, 1, 1)
two.hist(df['LSTAT'], bins = 30)  
plt.title("2: Homeowners low class")
plt.xlabel("LSTAT");plt.ylabel("freq");plt.show() #graph 2

f = plt.figure()
three = f.add_subplot(1, 1, 1)
three.hist(df['PTRATIO'], bins = 30)  
plt.title("3: Students to Teachers ratio")
plt.xlabel("PTRATIO");plt.ylabel("freq");plt.show() #graph 3

minp = np.min(df['MEDV']); maxp = np.max(df['MEDV']); meanp = np.mean(df['MEDV'])
medp = np.median(df['MEDV']); stdp = np.std(df['MEDV'])

print("#Some stats for our Boston housing dataset are as follows:\n")
print("\tMinimum price = ${}".format(minp)) 
print("\tMaximum price = ${}".format(maxp))
print("\tMean price = ${}".format(meanp))
print("\tMedian price = ${}".format(medp))
print("\tStandard deviation of prices = ${}".format(stdp))

#plotting relations btw features and given y now
f = plt.figure()
one = f.add_subplot(1, 1, 1)
one.scatter(df['RM'], df['MEDV']) 
plt.title("1. Avg SP vs Avg no of rooms")
plt.xlabel("RM");plt.ylabel("Prices");plt.show()# graph 1

f = plt.figure()
two = f.add_subplot(1, 1, 1)
two.scatter(df['LSTAT'], df['MEDV'])  
plt.title("2. Avg SP vs % of low class Homeowners")
plt.xlabel("LSTAT");plt.ylabel("Prices");plt.show()# graph 2

f = plt.figure()
three = f.add_subplot(1, 1, 1)
three.scatter(df['PTRATIO'], df['MEDV'])  
plt.title("3. Avg SP vs Studs:Trs")
plt.xlabel("PTRATIO");plt.ylabel("Prices");plt.show()# graoh 3

from sklearn.metrics import r2_score
def func(y_actual, y_prediction):
    score = r2_score(y_actual,y_prediction)
    
    return score

#score = func([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3]) #verifying how R2 works
#print("Coefficient of relationshio, R^2, of {:.3f}.".format(score))

#splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, df['MEDV'], test_size=0.2, random_state=0)
print("Training and testing split done :)")
      
# Produce learning curves for varying training set sizes and maximum depths
visuals.ModelLearning(features, df['MEDV'])
      
visuals.ModelComplexity(X_train, y_train)

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
def func2(X, y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor(random_state=0)
    parameters = {'max_depth':list(range(1,11))}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, parameters, cv=cv_sets, scoring=scoring_fnc)
    grid = grid.fit(X, y)
    return grid.best_estimator_
      
reg = func2(X_train, y_train)
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_parameters()['max_depth']))
 #now applying model to a given data of a client     
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Showing predictions now
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
      
visuals.PredictTrials(features, df['MEDV'], fit_model, client_data)
      


# In[45]:


pip install sklearn


# In[12]:


from sklearn.model_selection import learning_curve


# In[14]:


#from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit


# In[ ]:


pip install skl


# In[22]:


pip install scikit-learn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




