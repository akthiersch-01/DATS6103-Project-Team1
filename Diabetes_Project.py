#%%
#Import packages
import numpy as np
import pandas as pd
import os
import mlxtend
import seaborn as sns
import matplotlib.pyplot as plt
import math 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#%%
#Let's define some key functions here that'll help us throughout the rest of this

#Violin plot function


#Scatter plot function


#Contingency table/heat map functions


#Chi-square test function (for testing impact of categorical data on our outcome variables)


#two sample Z-test function (for testing impact of continuous data based on diabetes_012)

#%%
#Read in csv
diabetes = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
# %%
#Let's add basic summary information here (proportions, averages, etc.)

#%%
#Let's add some summary visualizations here using our few continuous variables. We can put Diabetes_012 as the color and use shape for some other things, or we can make violin plots with diabetes_012 as the splits and maybe do double splits

#%%
#Let's do some contingency tables/heat maps here and could consider proportions.

#%%
#Test/Train split - we have sufficient data to do a 9/1 or a 4/1 (probably a 4/1 since pre-diabetes is a relatively small category). Make sure we set the random state here so we can repeat it
#%%
#First, let's build a basic logistic regression, we'll need to either use sklearn or the function Prof. Lo gave us in quiz 3 for a multinomial response variable

#Model Building - Logistic

#Model summary information (including pseudo-R^2)
#Loop with different cutoff values showing score and confusion matrix

#ROC-AUC


#%%
#Start building more complicated models
#Model Building - Trees, SVM, etc.

#Information regarding fit and accuracy


#%%
#Comparison of all models to determine which variables had the largest impacts and which model was best