#%%
#Import packages
from audioop import mul
from tkinter import Label
import numpy as np
import pandas as pd
import os
import mlxtend
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import tabulate
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statsmodels.stats.weightstats import ztest as ztest
import statsmodels.api as sm
from statsmodels.formula.api import mnlogit, glm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import warnings
warnings.filterwarnings('always')

#%%
#Let's define some key functions here that'll help us throughout the rest of this

#Violin plot function
def violin_plot_func(data, cat_col, cont_col):
    '''
    Function will take in a dataset and two column names, one categorical and one continuous
    Data: name of pandas dataframe containing the columns
    cat_col: name of column that will be used to split the violin plots
    cont_col: continuous function'''
    for i in range(len(data[cat_col].unique())):
        globals()['group%s' % i] = data[data[cat_col]==i]
    cat_list = []
    for i in range(len(data[cat_col].unique())):
        cat_list.append(list(globals()['group%s' % i][cont_col]))
    pos = np.arange(1,len(data[cat_col].unique())+1)
    pos_list = np.ndarray.tolist(pos)
    plt.violinplot(cat_list, positions = pos_list)
    plt.xlabel(cat_col)
    plt.ylabel(cont_col)
    plt.show()


#Category plot function
def sns_catplot(data, cat_col, cont_col, kind='violin', hue=None, split=False, col=None, col_wrap=2, legend_labels=None,
                xticks=None):
    '''
    Function will take in a datatset and two column names, one categorical and one continuous. Other parameters are
    optional to assist in plot customization.
    :param data: name of pandas dataframe containing the columns
    :param cat_col: x axis value
    :param cont_col: y axis value
    :param kind: kind of sns plot. Default set to violin
    :param hue: name of column to split the violin plot into categories. Default set to None
    :param split: bool to set hue split into two halves. Default set to False
    :param col: categorical variable to facet the grid
    :param col_wrap: wraps the columns at this width. Default set to 2
    :param legend_labels: labels to set for legend. Default set to None
    :param xticks: labels to set for xticks. Default set to None
    '''
    # check if hue is set
    if hue:
        hue = hue
        if data[hue].unique().size == 2:
            split = True

    # check if col is set
    if col:
        col = col

    # create catplot
    sns.set_palette('hls')
    chart = sns.catplot(data=data, x=cat_col, y=cont_col, kind=kind, hue=hue, split=split, col=col, col_wrap=col_wrap)

    # edit legend labels if provided
    if legend_labels:
        for index in range(len(legend_labels)):
            chart._legend.texts[index].set_text(legend_labels[index])

    # edit xticks if provided
    if xticks:
        plt.xticks(xticks[0], xticks[1])

    plt.xlabel(cat_col)
    plt.ylabel(cont_col)
    plt.show()

#Contingency table/heat map functions - non-proportional
def categorical_contigency_base(group1, group2):
    '''
    Function will combine two categorical variables into a contingency table, and then output a heatmap showing both the numbers and a color scheme to represent equality across the cells. Includes margins
    input group1, group2
    group1: categorical variable
    group2: categorical variable
    output: heatmap showing the contingency table with appropriate coloring
    Group here refers to categorical, but not individual level'''
    data_contingency = pd.crosstab(group1, group2, margins = True, margins_name = 'Total')
    print(data_contingency)
    data_contingency=pd.crosstab(group1, group2, margins = False, margins_name = 'Total')
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(data_contingency, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.show()
    return

#Contingency table/heat map functions - overall proportional
def categorical_contigency_prop_whole(group1, group2):
    '''
    Function will combine two categorical variables into a contingency table, and then output a heatmap showing both the numbers and a color scheme to represent equality across the cells. Includes margins
    input group1, group2
    group1: categorical variable
    group2: categorical variable
    output: heatmap showing the contingency table with appropriate coloring
    Group here refers to categorical, but not individual level
    If there is an error, try switching group1 and group2'''
    data_contingency = pd.crosstab(group1, group2, margins = True, margins_name = 'Total')
    columns = group1.unique()
    rows = group2.unique()
    df = pd.DataFrame()
    for i in rows:
        for j in columns:
            proportion = data_contingency[i][j]/data_contingency['Total']["Total"]
            df.loc[i,j]=proportion
    df=df.transpose()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df, annot=True, fmt="f", linewidths=.5, ax=ax)
    plt.show()
    return

#Contingency table/heat map functions - column proportional
def categorical_contigency_prop_col(group1, group2):
    '''
    Function will combine two categorical variables into a contingency table, and then output a heatmap showing both the numbers and a color scheme to represent equality across the cells. Includes margins
    input group1, group2
    group1: categorical variable
    group2: categorical variable
    output: heatmap showing the contingency table with appropriate coloring
    Group here refers to categorical, but not individual level'''
    data_contingency = pd.crosstab(group1, group2, margins = True, margins_name = 'Total')
    columns = group1.unique()
    rows = group2.unique()
    df = pd.DataFrame()
    for i in rows:
        for j in columns:
            proportion = data_contingency[i][j]/data_contingency[i]["Total"]
            df.loc[i,j]=proportion
    df=df.transpose()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df, annot=True, fmt="f", linewidths=.5, ax=ax)
    plt.show()
    return


#Contingency table/heat map functions - row proportional
def categorical_contigency_prop_row(group1, group2):
    '''
    Function will combine two categorical variables into a contingency table, and then output a heatmap showing both the numbers and a color scheme to represent equality across the cells. Includes margins
    input group1, group2
    group1: categorical variable
    group2: categorical variable
    output: heatmap showing the contingency table with appropriate coloring
    Group here refers to categorical, but not individual level'''
    data_contingency = pd.crosstab(group1, group2, margins = True, margins_name = 'Total')
    columns = group1.unique()
    rows = group2.unique()
    df = pd.DataFrame()
    for i in rows:
        for j in columns:
            proportion = data_contingency[i][j]/data_contingency['Total'][j]
            df.loc[i,j]=proportion
    df=df.transpose()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df, annot=True, fmt="f", linewidths=.5, ax=ax)
    plt.show()
    return

#Chi-square test function (for testing impact of categorical data on our outcome variables)
def chi_square_test(group1, group2, alpha = 0.05, decimals = 3):
    '''
    Function will combine two categorical variables into a contingency table, and then output a two sided hypothesis test for independence with the chi-square statistic, p-value, and hypothesis test conclusion
    group1: categorical variable
    group2: categorical variable
    alpha: cutoff for p-value, number between 0 and 1, defaults to 0.05 or a 5% cutoff
    decimals: preferred rounding number for chi-square and p-value
    output: chi-square statistic, p-value, and hypothesis test conclusion
    Group here refers to categorical, but not individual level'''
    data_contingency = pd.crosstab(group1, group2, margins = True, margins_name = 'Total')
    chi_square = 0
    columns = group1.unique()
    rows = group2.unique()
    for i in rows:
        for j in columns:
            O = data_contingency[i][j]
            E = data_contingency[i]['Total'] * data_contingency['Total'][j] / data_contingency['Total']['Total']
            chi_square += (O-E)**2/E
    p_value = 1 - stats.chi2.cdf(chi_square, (len(rows)-1)*(len(columns)-1))
    conclusion = "Failed to reject the null hypothesis."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected."
        
    print("chisquare-score is:", round(chi_square,decimals), " and p value is:", round(p_value,decimals))
    return(conclusion)

#two sample Z-test function (for testing impact of continuous data based on diabetes_012)
def two_sample_test(group1, group2, alpha = 0.05, decimals = 3):
    '''
    input group1, group 2, alpha, decimals
    group 1: qualitative variable corresponding to the first group (ie female)
    group 2: qualitative variable corresponding to the second group (ie male)
    alpha: cutoff for p-value, number between 0 and 1, defaults to 0.05 or a 5% cutoff
    decimals: preferred rounding number for z-score and p-value
    outputs: z_score and p_value, plus hypothesis testing determination
    Note: If there are more than 2 levels of a category, it is necessary to run the function for each respective pair of values
    '''
    ztest_vals = ztest(group1, group2)
    z_stat = round(ztest_vals[0],decimals)
    p_value = round(ztest_vals[1],decimals)
    if p_value < 0.05:
        
        print (f"Your z-score was {z_stat} and your p-value was  {p_value}, which is less than 0.05. We therefore reject our null hypothesis")
    else:
        print (f"Your z-score was {z_stat} and your p-value was  {p_value}, which is greater than 0.05. We therefore fail to reject our null hypothesis")
    return 




#%%
#Read in csv
diabetes = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

#Testing the functions work
# categorical_contigency_base(diabetes['Diabetes_012'], diabetes['HighBP'])
# categorical_contigency_prop_whole(diabetes['Diabetes_012'], diabetes['HighBP'])
# categorical_contigency_prop_col(diabetes['Diabetes_012'], diabetes['HighBP'])
# categorical_contigency_prop_row(diabetes['Diabetes_012'], diabetes['HighBP'])
# chi_square_test(diabetes['Diabetes_012'], diabetes['HighBP'])
# violin_plot_func(diabetes, 'Diabetes_012', 'Age')
# two_sample_test(diabetes[diabetes['Diabetes_012']==0]['Age'], diabetes[diabetes['Diabetes_012']==1]['Age'])


# %%
#Let's add basic summary information here (proportions, averages, etc.)
summary_stats = pd.DataFrame(diabetes.describe())
summary_stats = summary_stats.reset_index()
print(summary_stats.to_markdown())


#%%
#Let's add some summary visualizations here using our few continuous variables. We can put Diabetes_012 as the color and use shape for some other things, or we can make violin plots with diabetes_012 as the splits and maybe do double splits
# BMI vs. Diabetes_012 by Sex and Income
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='Income', col_wrap=4, legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# BMI vs. Diabetes_012 by Sex and HighBP
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='HighBP', legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# BMI vs. Diabetes_012 by Sex and HighChol
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='HighChol', legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# BMI vs. Diabetes_012 by Sex and PhysActivity
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='PhysActivity', legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# Age vs. Diabetes_012 by Sex and Income
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='Income', col_wrap=4, legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# Age vs. Diabetes_012 by Sex and HighBP
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='HighBP', legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# Age vs. Diabetes_012 by Sex and HighChol
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='HighChol', legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))

# Age vs. Diabetes_012 by Sex and PhysActivity
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='PhysActivity', legend_labels=['Male', 'Female'],
            xticks=([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']))
            
#%%
#Let's do some contingency tables/heat maps here and could consider proportions.

#Diabetes status by blood pressure - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HighBP)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HighBP)
chi_square_test(diabetes.Diabetes_012, diabetes.HighBP)

#Diabetes status by Sex - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Sex)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Sex)
chi_square_test(diabetes.Diabetes_012, diabetes.Sex)

#Diabetes status by HighChol - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HighChol)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HighChol)
chi_square_test(diabetes.Diabetes_012, diabetes.HighChol)

#Diabetes status by CholCheck - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.CholCheck)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.CholCheck)
chi_square_test(diabetes.Diabetes_012, diabetes.CholCheck)

#Diabetes status by Smoker - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Smoker)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Smoker)
chi_square_test(diabetes.Diabetes_012, diabetes.Smoker)

#Diabetes status by Stroke - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Stroke)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Stroke)
chi_square_test(diabetes.Diabetes_012, diabetes.Stroke)

#Diabetes status by HeartDiseaseorAttack - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HeartDiseaseorAttack)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HeartDiseaseorAttack)
chi_square_test(diabetes.Diabetes_012, diabetes.HeartDiseaseorAttack)

#Diabetes status by PhysActivity - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.PhysActivity)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.PhysActivity)
chi_square_test(diabetes.Diabetes_012, diabetes.PhysActivity)

#Diabetes status by Fruits - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Fruits)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Fruits)
chi_square_test(diabetes.Diabetes_012, diabetes.Fruits)

#Diabetes status by Veggies - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Veggies)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Veggies)
chi_square_test(diabetes.Diabetes_012, diabetes.Veggies)

#Diabetes status by HvyAlcoholConsump - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HvyAlcoholConsump)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HvyAlcoholConsump)
chi_square_test(diabetes.Diabetes_012, diabetes.HvyAlcoholConsump)

#Diabetes status by AnyHealthcare - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.AnyHealthcare)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.AnyHealthcare)
chi_square_test(diabetes.Diabetes_012, diabetes.AnyHealthcare)

#Diabetes status by NoDocbcCost - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.NoDocbcCost)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.NoDocbcCost)
chi_square_test(diabetes.Diabetes_012, diabetes.NoDocbcCost)

#Diabetes status by GenHlth - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.GenHlth)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.GenHlth)
chi_square_test(diabetes.Diabetes_012, diabetes.GenHlth)

#Diabetes status by DiffWalk - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.DiffWalk)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.DiffWalk)
chi_square_test(diabetes.Diabetes_012, diabetes.DiffWalk)

#Diabetes status by Education - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Education)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Education)
chi_square_test(diabetes.Diabetes_012, diabetes.Education)

#Diabetes status by Income - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Income)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Income)
chi_square_test(diabetes.Diabetes_012, diabetes.Income)

#Diabetes by Age - Z-Tests (All Significant)
two_sample_test(diabetes[diabetes['Diabetes_012']==0]['Age'], diabetes[diabetes['Diabetes_012']==1]['Age'])
two_sample_test(diabetes[diabetes['Diabetes_012']==0]['Age'], diabetes[diabetes['Diabetes_012']==2]['Age'])
two_sample_test(diabetes[diabetes['Diabetes_012']==1]['Age'], diabetes[diabetes['Diabetes_012']==2]['Age'])

#Diabetes by BMI - Z-Tests (All Significant)
two_sample_test(diabetes[diabetes['Diabetes_012']==0]['BMI'], diabetes[diabetes['Diabetes_012']==1]['BMI'])
two_sample_test(diabetes[diabetes['Diabetes_012']==0]['BMI'], diabetes[diabetes['Diabetes_012']==2]['BMI'])
two_sample_test(diabetes[diabetes['Diabetes_012']==1]['BMI'], diabetes[diabetes['Diabetes_012']==2]['BMI'])

#%%
#Test/Train split - we have sufficient data to do a 9/1 or a 4/1 (probably a 4/1 since pre-diabetes is a relatively small category). Make sure we set the random state here so we can repeat it

xdiabetes = diabetes[
    ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
     'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
     'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']]
ydiabetes = diabetes['Diabetes_012'].values

class_le = LabelEncoder()
ydiabetes = class_le.fit_transform(ydiabetes)

ydiabetes1 = label_binarize(ydiabetes, classes=[0,1,2])
xdiabetestrain, xdiabetestest, ydiabetestrain, ydiabetestest = train_test_split(xdiabetes, ydiabetes, train_size=.8,
                                                                                random_state=12345)
xdiabetestrain1, xdiabetestest1, ydiabetestrain1, ydiabetestest1 = train_test_split(xdiabetes, ydiabetes1, train_size=.8,
                                                                                random_state=12345)

#%%
#First, let's build a basic logistic regression, we'll need to either use sklearn or the function Prof. Lo gave us in quiz 3 for a multinomial response variable
#Model Building - Logistic
model_diabetes1 = mnlogit(formula='Diabetes_012 ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth + PhysHlth + DiffWalk + Sex + Age + Education + Income', data=diabetes)

#Model summary information (including pseudo-R^2)
model_diabetes1_fit = model_diabetes1.fit()
print( model_diabetes1_fit.summary() )
modelpredicitons = pd.DataFrame(model_diabetes1_fit.predict(diabetes)) 
modelpredicitons.rename(columns={0:'No Diabetes', 1:'Pre Diabetes', 2:'Has Diabetes'}, inplace=True)
print(modelpredicitons.head())


#Sklearn
diabetes_logit = LogisticRegression()
diabetes_logit.fit(xdiabetestrain, ydiabetestrain)
print('Logit model accuracy (with the test set):', diabetes_logit.score(xdiabetestest, ydiabetestest))
print('Logit model accuracy (with the train set):', diabetes_logit.score(xdiabetestrain, ydiabetestrain))
print(diabetes_logit.predict(xdiabetestest))
print(diabetes_logit.predict_proba(xdiabetestrain[:8]))
print(diabetes_logit.predict_proba(xdiabetestest[:8]))

#Loop with different cutoff values showing score and confusion matrix
def predictcutoff(arr, cutoff):
  arrbool = arr[:,1]>cutoff
  arr= arr[:,1]*arrbool/arr[:,1]
  return arr.astype(int)

test = diabetes_logit.predict_proba(xdiabetestest)
p = predictcutoff(test, 0.1)
print(p)

predictcutoff(test, 0.2)

predictcutoff(test, 0.5)

cut_off = 1
predictions = (diabetes_logit.predict_proba(xdiabetestest)[:,1]>cut_off).astype(int)
print(predictions)


# Classification Report
#
y_true, y_pred = ydiabetestest, diabetes_logit.predict(xdiabetestest)
print(classification_report(y_true, y_pred))
#%%
#ROC-AUC
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(ydiabetestest))]
# predict probabilities
lr_probs = diabetes_logit.predict_proba(xdiabetestest)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate for all response types
for response in range(3):
    # calculate roc scores
    ns_auc = roc_auc_score(ydiabetestest1[:,response], ns_probs)
    lr_auc = roc_auc_score(ydiabetestest1[:, response], lr_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ydiabetestest1[:,response], ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(ydiabetestest1[:, response], lr_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


#%%
#Start building more complicated models
#Model Building - Trees, SVM, etc.


#%%
#=====================KNN======================
#This takes a VERY long time to run when we include the train parameters. I will comment them for sake of ease, but we can attempt them in the future if we need the scores



for i in [3,5,7,9,11]:
    knn_Diet = KNeighborsClassifier(n_neighbors=i) # instantiate with n value given
    knn_Diet.fit(xdiabetestrain, ydiabetestrain)
    print(f'{i}-NN model accuracy (with the test set):', knn_Diet.score(xdiabetestest, ydiabetestest))
    #print(f'{i}-NN model accuracy (with the train set):', knn_Diet.score(xdiabetestrain, ydiabetestrain))
    y_pred_score1 = knn_Diet.predict_proba(xdiabetestest)
    n_classes=3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(n_classes):
        fpr[j], tpr[j], _ = roc_curve(ydiabetestest1[:, j], y_pred_score1[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])
        print(f'AUC value of {j} class:{roc_auc[j]}')

    # Plot of a KNN ROC curve for a specific class
    for j in range(n_classes):
        plt.figure()
        plt.plot(fpr[j], tpr[j], label='ROC curve (area = %0.2f)' % roc_auc[j])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{i}-NN model ROC for Diabetes_012 = {j}')
        plt.legend(loc="lower right")
        plt.show()


#%%
#================Decision Tree=================

rf1 = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=0)
# Fit dt to the training set
rf1 = rf1.fit(xdiabetestrain, ydiabetestrain)
y_test_pred = rf1.predict(xdiabetestest)
y_pred_score = rf1.predict_proba(xdiabetestest)
importance = rf1.feature_importances_
feature_importance = np.array(importance)
feature_names = np.array(xdiabetestrain.columns)

#Create a DataFrame using a Dictionary
data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)

#Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
print(fi_df)

rf2 = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3, criterion='entropy'))
# Fit dt to the training set
rf2.fit(xdiabetestrain1, ydiabetestrain1)
y_test_pred1 = rf2.predict(xdiabetestest1)
y_pred_score1 = rf2.predict_proba(xdiabetestest1)



print('Decision Tree results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ", accuracy_score(ydiabetestest, y_test_pred) * 100)
print("Confusion Matrix: \n", confusion_matrix(ydiabetestest, y_test_pred,))
print("Classification report:\n", classification_report(ydiabetestest, y_test_pred))



n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ydiabetestest1[:, i], y_pred_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'AUC value of {i} class:{roc_auc[i]}')

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree ROC')
    plt.legend(loc="lower right")
    plt.show()

#%%
#==================Random Forest====================


# Instantiate dtree
rf1 = RandomForestClassifier(n_estimators=100)
# Fit dt to the training set
rf1.fit(xdiabetestrain, ydiabetestrain)
y_test_pred = rf1.predict(xdiabetestest)
y_pred_score = rf1.predict_proba(xdiabetestest)


# importance = rf1.estimators_[2].feature_importances_
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
rf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))

# Fit dt to the training set
rf2.fit(xdiabetestrain1, ydiabetestrain1)
y_test_pred1 = rf2.predict(xdiabetestest1)
y_pred_score1 = rf2.predict_proba(xdiabetestest1)
importance = rf2.estimators_[2].feature_importances_

feature_importance = np.array(importance)
feature_names = np.array(xdiabetestrain.columns)

#Create a DataFrame using a Dictionary
data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)
#Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
print(fi_df)


rf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))

# Fit dt to the training set
rf2.fit(xdiabetestrain1, ydiabetestrain1)
y_test_pred1 = rf2.predict(xdiabetestest1)
y_pred_score1 = rf2.predict_proba(xdiabetestest1)


print('Random forest results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ", accuracy_score(ydiabetestest, y_test_pred) * 100)
print("Confusion Matrix: \n", confusion_matrix(ydiabetestest, y_test_pred))
print("Classification report:\n", classification_report(ydiabetestest, y_test_pred))


n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ydiabetestest1[:, i], y_pred_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'AUC value of {i} class:{roc_auc[i]}')

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC')
    plt.legend(loc="lower right")
    plt.show()

#%%
#=================SVM(SVC)====================
#Sometimes SVM can't solve the equation if there's a huge amount of data and/or predictors. We may be running into that here, so if this doesn't solve in a reasonable amount of time, we'll need to just leave the code commetned and write an acknowledgment of the computational limitations of the technique

xdiabetestrain = preprocessing.scale(xdiabetestrain)

xdiabetestest = preprocessing.scale(xdiabetestest)
rf1 = SVC(kernel='linear', C=1.0, random_state=0)
# Fit dt to the training set
rf1.fit(xdiabetestrain, ydiabetestrain)
y_test_pred = rf1.predict(xdiabetestest)
y_pred_score = rf1.decision_function(xdiabetestest)

rf2 = OneVsRestClassifier(SVC(kernel='linear', C=1.0, random_state=0))

# Fit dt to the training set
#This one doesn't run, compute needs are too high
rf2.fit(xdiabetestrain1, ydiabetestrain1)
y_test_pred1 = rf2.predict(xdiabetestest1)
y_pred_score1 = rf2.decision_function(xdiabetestest1)

print('SVC results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ", accuracy_score(ydiabetestest, y_test_pred) * 100)
print("Confusion Matrix: \n", confusion_matrix(ydiabetestest, y_test_pred))
print("Classification report:\n", classification_report(ydiabetestest, y_test_pred))




n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ydiabetestest[:, i], y_pred_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'AUC value of {i} class:{roc_auc[i]}')

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVC ROC')
    plt.legend(loc="lower right")
    plt.show()

#%%
#Comparison of all models to determine which variables had the largest impacts and which model was best