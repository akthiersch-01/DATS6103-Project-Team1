#%%
#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.weightstats import ztest as ztest
from statsmodels.formula.api import mnlogit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
warnings.filterwarnings('always')

#%%
#Let's define some key functions here that'll help us throughout the rest of this

#Category plot function
def sns_catplot(data, cat_col, cont_col, kind='violin', hue=None, split=False, col=None, col_wrap=2, legend_labels=None,
                xticks=None, title=None, xlabel=None, col_labels=None):
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
    :param title: title for entire chart 
    :param xlabel: x axis label
    :param col_labels: column labels
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

    if col_labels:
        col_titles = chart.axes.flatten()
        for i in range(len(col_labels)):
            col_titles[i].set_title(col_labels[i])

    # edit xticks if provided
    if xticks:
        plt.xticks(xticks[0], xticks[1])

    # add title if provided
    if title:
        plt.suptitle(title, fontweight='bold')
        plt.subplots_adjust(top=0.9)
    if xlabel:
        chart.set_xlabels(xlabel)
    else:
        chart.set_xlabels(cat_col)
    plt.ylabel(cont_col)
    plt.show()

#Contingency table/heat map functions - non-proportional
def categorical_contigency_base(group1, group2, title=None, yticks=None, ylabel=None, xticks=None):
    '''
    Function will combine two categorical variables into a contingency table, and then output a heatmap showing both the numbers and a color scheme to represent equality across the cells. Includes margins
    input group1, group2
    group1: categorical variable
    group2: categorical variable
    title: title for entire chart 
    yticks: labels for yticks
    ylabel: y axis label
    xticks: labels for xticks
    output: heatmap showing the contingency table with appropriate coloring
    Group here refers to categorical, but not individual level'''
    data_contingency = pd.crosstab(group1, group2, margins = True, margins_name = 'Total')
    data_contingency=pd.crosstab(group1, group2, margins = False, margins_name = 'Total')

    f, ax = plt.subplots(dpi=100, figsize=(14, 7))
    #sns.heatmap(data_contingency, annot=True, fmt="d", linewidths=.5, ax=ax)
    #res = sns.heatmap(pd.crosstab(group1, group2,), annot=True, fmt="d", linewidths = 0.5, cbar=False, ax=ax)
    #res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 18)
    #res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 18)
    #plt.savefig('data_table.png')
    if yticks:
        data_contingency = data_contingency.rename(index=yticks)
    if xticks:
        data_contingency = data_contingency.rename(columns=xticks)
    f, ax = plt.subplots(figsize=(9, 6))
    chart = sns.heatmap(data_contingency, annot=True, fmt="d", linewidths=.5, ax=ax)
    if title:
        plt.title(title, fontweight='bold')

    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(group1.name)

    plt.xlabel(group2.name)
    plt.show()
    return

#Contingency table/heat map functions - overall proportional
def categorical_contigency_prop_whole(group1, group2, title=None, yticks=None, ylabel=None, xticks=None):
    '''
    Function will combine two categorical variables into a contingency table, and then output a heatmap showing both the numbers and a color scheme to represent equality across the cells. Includes margins
    input group1, group2
    group1: categorical variable
    group2: categorical variable
    title: title for entire chart 
    yticks: labels for yticks
    ylabel: y axis label
    xticks: labels for xticks
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

    if yticks:
        df = df.rename(index=yticks)
    if xticks:
        df = df.rename(columns=xticks)
    
    f, ax = plt.subplots(figsize=(9, 6))
    chart = sns.heatmap(df, annot=True, fmt="f", linewidths=.5, ax=ax)
    if title:
        plt.title(title, fontweight='bold')

    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(group1.name)

    plt.xlabel(group2.name)
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

#%%
#Read in csv
diabetes = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# %%
#Let's add basic summary information here (proportions, averages, etc.)
summary_stats = pd.DataFrame(diabetes.describe())
summary_stats = summary_stats.reset_index()
print(summary_stats.to_markdown())
diabetes.head()
diabetes.Diabetes_012.value_counts()
#%%
#Let's add some summary visualizations here using our few continuous variables. We can put Diabetes_012 as the color and use shape for some other things, or we can make violin plots with diabetes_012 as the splits and maybe do double splits
xticks = ([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes'])
legend_labels_sex = ['Female', 'Male']
xlabel = 'Diabetes Status'

#BMI vs. Diabetes_012 by Sex
violin_plot_func(diabetes, 'Diabetes_012', 'BMI', hue='Sex', legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by Sex', xlabel=xlabel)

#PhysHlth vs. Diabetes_012 by Sex
violin_plot_func(diabetes, 'Diabetes_012', 'PhysHlth', hue='Sex', legend_labels=legend_labels_sex,
            xticks=xticks, title='PhysHlth vs. Diabetes Status by Sex', xlabel=xlabel)

#MentHlth vs. Diabetes_012 by Sex
violin_plot_func(diabetes, 'Diabetes_012', 'MentHlth', hue='Sex', legend_labels=legend_labels_sex,
            xticks=xticks, title='MentHlth vs. Diabetes Status by Sex', xlabel=xlabel)

# BMI vs. Diabetes_012 by Sex and Income
col_labels_inc = ['Income: < $10,000', 'Income: < $15,000', 'Income: < $20,000', 'Income: < $25,000', 'Income: < $35,000', 'Income: < $50,000', 'Income: < $75,000', 'Income: > $75,000']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='Income', col_wrap=4, legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by Income and Sex', xlabel=xlabel, col_labels=col_labels_inc)

# BMI vs. Diabetes_012 by Sex and HighBP
col_labels_bp = ['No High Blood Pressure', 'Has High Blood Pressure']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='HighBP', legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by High Blood Pressure and Sex', xlabel=xlabel, col_labels=col_labels_bp)

# BMI vs. Diabetes_012 by Sex and HighBP
col_labels_bp = ['No High Blood Pressure', 'Has High Blood Pressure']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='HighBP', legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by High Blood Pressure and Sex', xlabel=xlabel, col_labels=col_labels_bp)


# BMI vs. Diabetes_012 by Income and HighBP
col_labels_inc = ['Income: < $10,000', 'Income: < $15,000', 'Income: < $20,000', 'Income: < $25,000', 'Income: < $35,000', 'Income: < $50,000', 'Income: < $75,000', 'Income: > $75,000']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='HighBP', col='Income', col_wrap=4, legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by High Blood Pressure and Income', xlabel=xlabel, col_labels=col_labels_inc)



# BMI vs. Diabetes_012 by Sex and HighChol
col_labels_chol = ['No High Cholesterol', 'Has High Cholesterol']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='HighChol', legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by High Cholesterol and Sex', xlabel=xlabel, col_labels=col_labels_chol)

# BMI vs. Diabetes_012 by Sex and PhysActivity
col_labels_phys = ['Not Physically Active', 'Physically Active']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='PhysActivity', legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by Physical Activity and Sex', xlabel=xlabel, col_labels=col_labels_phys)

# PhysHlth vs. Diabetes_012 by Sex and PhysActivity
col_labels_phys = ['Not Physically Active', 'Physically Active']
sns_catplot(diabetes, 'Diabetes_012', 'PhysHlth', hue='Sex', col='PhysActivity', legend_labels=legend_labels_sex,
            xticks=xticks, title='PhysHlth vs. Diabetes Status by Physical Activity and Sex', xlabel=xlabel, col_labels=col_labels_phys)


# Age vs. Diabetes_012 by Sex and Income
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='Income', col_wrap=4, legend_labels=legend_labels_sex,
            xticks=xticks, title='Age vs. Diabetes Status by Income and Sex', xlabel=xlabel, col_labels=col_labels_inc)

# Age vs. Diabetes_012 by Sex and HighBP
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='HighBP', legend_labels=legend_labels_sex,
            xticks=xticks, title='Age vs. Diabetes Status by High Blood Pressure and Sex', xlabel=xlabel, col_labels=col_labels_bp)

# Age vs. Diabetes_012 by Sex and HighChol
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='HighChol', legend_labels=legend_labels_sex,
            xticks=xticks, title='Age vs. Diabetes Status by High Cholesterol and Sex', xlabel=xlabel, col_labels=col_labels_chol)

# Age vs. Diabetes_012 by Sex and PhysActivity

sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='PhysActivity', legend_labels=legend_labels_sex,
            xticks=xticks, title='Age vs. Diabetes Status by Physical Activity and Sex', xlabel=xlabel, col_labels=col_labels_phys)


# BMI vs. Diabetes_012 by Sex and Education
col_labels_edu = ['Never Attended', 'Elementary', 'Some High School', 'High School Graduate', 'Some College', 'College Graduate']
sns_catplot(diabetes, 'Diabetes_012', 'BMI', hue='Sex', col='Education', col_wrap=3, legend_labels=legend_labels_sex,
            xticks=xticks, title='BMI vs. Diabetes Status by Education and Sex', xlabel=xlabel, col_labels=col_labels_edu)


# Age vs. Diabetes_012 by Sex and Education
col_labels_edu = ['Never Attended', 'Elementary', 'Some High School', 'High School Graduate', 'Some College', 'College Graduate']
sns_catplot(diabetes, 'Diabetes_012', 'Age', hue='Sex', col='Education', col_wrap=3, legend_labels=legend_labels_sex,
            xticks=xticks, title='Age vs. Diabetes Status by Education and Sex', xlabel=xlabel, col_labels=col_labels_edu)


#%%
#Let's do some contingency tables/heat maps here and could consider proportions.
yticks={0: 'No Diabetes', 1: 'Pre Diabetes', 2: 'Has Diabetes'} #([0, 1, 2], ['No Diabetes', 'Pre Diabetes', 'Has Diabetes'])
ylabel = 'Diabetes Status'

#Diabetes status by blood pressure - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. High Blood Pressue'
xticks = {0:'No High Blood Pressure', 1: 'Has High Blood Pressure'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HighBP, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HighBP, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.HighBP)

#Diabetes status by Sex - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Sex'
xticks = {0:'Female', 1:'Male'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Sex, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Sex, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Sex)

#Diabetes status by HighChol - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. High Cholesterol'
xticks = {0: 'No High Cholesterol', 1: 'Has High Cholesterol'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HighChol, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HighChol, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.HighChol)

#Diabetes status by CholCheck - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Cholesterol Check'
xticks = {0: 'Not Checked in Last 5yrs', 1: 'Has Checked in Last 5yrs'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.CholCheck, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.CholCheck, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.CholCheck)

#Diabetes status by Smoker - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Smoker Status'
xticks = {0: 'Non Smoker' , 1: 'Smoker'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Smoker, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Smoker, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Smoker)

#Diabetes status by Stroke - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Stroke'
xticks = {0: 'No Stroke' , 1: 'Stroke'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Stroke, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Stroke, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Stroke)

#Diabetes status by HeartDiseaseorAttack - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Heart Disease'
xticks = {0: 'No Heart Disease' , 1: 'Heart Disease'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HeartDiseaseorAttack, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HeartDiseaseorAttack, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.HeartDiseaseorAttack)

#Diabetes status by PhysActivity - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Physical Activity'
xticks = {0: 'Not Physically Activity' , 1: 'Physically Active'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.PhysActivity, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.PhysActivity, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.PhysActivity)

#Diabetes status by Fruits - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Fruits'
xticks = {0: 'Does Not Eat Fruit' , 1: 'Eats Fruit'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Fruits, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Fruits, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Fruits)

#Diabetes status by Veggies - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Veggies'
xticks = {0: 'Does Not Eat Veggies' , 1: 'Eats Veggies'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Veggies, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Veggies, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Veggies)

#Diabetes status by HvyAlcoholConsump - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Heavy Alcohol Consumption'
xticks = {0: 'Non Heavy Drinker' , 1: 'Heavy Drinker'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.HvyAlcoholConsump, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.HvyAlcoholConsump, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.HvyAlcoholConsump)

#Diabetes status by AnyHealthcare - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Health Care Status'
xticks = {0: 'No Health Care' , 1: 'Has Health Care'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.AnyHealthcare, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.AnyHealthcare, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.AnyHealthcare)

#Diabetes status by NoDocbcCost - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Medical/Doctor Costs'
xticks = {0: 'Can Afford' , 1: 'Cannot Afford'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.NoDocbcCost, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.NoDocbcCost, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.NoDocbcCost)

#Diabetes status by GenHlth - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. General Health'
xticks = {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.GenHlth, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.GenHlth, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.GenHlth)

#Diabetes status by DiffWalk - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Difficulty Walking'
xticks = {0: 'No Difficulty' , 1: 'Has Difficulty'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.DiffWalk, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.DiffWalk, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.DiffWalk)

#Diabetes status by Education - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Education'
xticks = {1: 'Never Attended', 2: 'Elementary', 3: 'Some High School', 4: 'High School Graduate', 5: 'Some College', 6: 'College Graduate'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Education, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Education, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Education)

#Diabetes status by Income - Heat Map and Chi-Square test (Significant)
chart_title = 'Diabetes Status vs. Income'
xticks = {1: '< $10,000', 2: '< $15,000', 3: '< $20,000', 4: '< $25,000', 5: '< $35,000', 6: '< $50,000', 7: '< $75,000', 8: '≥ $75,000'}
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Income, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Income, title=chart_title, yticks=yticks, ylabel=ylabel, xticks=xticks)
chi_square_test(diabetes.Diabetes_012, diabetes.Income)

#Diabetes status by Age - Heat Map and Chi-Square test (Significant)
categorical_contigency_base(diabetes.Diabetes_012, diabetes.Age)
categorical_contigency_prop_whole(diabetes.Diabetes_012, diabetes.Age)
categorical_contigency_prop_row(diabetes.Diabetes_012, diabetes.Age)
chi_square_test(diabetes.Diabetes_012, diabetes.Age)


#Diabetes by Mental Health - 2 significant
#One-Way ANOVA
f_oneway(diabetes[diabetes['Diabetes_012']==0]['MentHlth'], diabetes[diabetes['Diabetes_012']==1]['MentHlth'], diabetes[diabetes['Diabetes_012']==2]['MentHlth'])

#Given significance, we run a Tukey to control for family variance
tukey = pairwise_tukeyhsd(endog=diabetes['MentHlth'], groups=diabetes['Diabetes_012'], alpha=0.05)
print(tukey)

#Diabetes by Physical Health - all significant
#One-Way ANOVA
f_oneway(diabetes[diabetes['Diabetes_012']==0]['PhysHlth'], diabetes[diabetes['Diabetes_012']==1]['PhysHlth'], diabetes[diabetes['Diabetes_012']==2]['PhysHlth'])


#Given significance, we run a Tukey to control for family variance
tukey = pairwise_tukeyhsd(endog=diabetes['PhysHlth'], groups=diabetes['Diabetes_012'], alpha=0.05)
print(tukey)

#Diabetes by BMI - all significant
#One-Way ANOVA
f_oneway(diabetes[diabetes['Diabetes_012']==0]['BMI'], diabetes[diabetes['Diabetes_012']==1]['BMI'], diabetes[diabetes['Diabetes_012']==2]['BMI'])


#Given significance, we run a Tukey to control for family variance
tukey = pairwise_tukeyhsd(endog=diabetes['BMI'], groups=diabetes['Diabetes_012'], alpha=0.05)
print(tukey)


#%%
#Test/Train split, we will do a 4:1 split with a set random state

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
model_diabetes1 = mnlogit(formula='Diabetes_012 ~ C(HighBP) + C(HighChol) + C(CholCheck) + BMI + C(Smoker) + C(Stroke) + C(HeartDiseaseorAttack) + C(PhysActivity) + C(Fruits) + C(Veggies) + C(HvyAlcoholConsump) + C(AnyHealthcare) + C(NoDocbcCost) + C(GenHlth) + MentHlth + PhysHlth + C(DiffWalk) + C(Sex) + C(Age) + C(Education) + C(Income)', data=diabetes)

#Model summary information
model_diabetes1_fit = model_diabetes1.fit()
print( model_diabetes1_fit.summary() )
modelpredictions = pd.DataFrame(model_diabetes1_fit.predict(diabetes)) 
modelpredictions.rename(columns={0:'No Diabetes', 1:'Pre Diabetes', 2:'Has Diabetes'}, inplace=True)
print(modelpredictions.head())


#Sklearn for mor complex analysis
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
  return arr.asStatus(int)

test = diabetes_logit.predict_proba(xdiabetestest)
p = predictcutoff(test, 0.1)
print(p)

predictcutoff(test, 0.2)

predictcutoff(test, 0.5)

cut_off = 1
predictions = (diabetes_logit.predict_proba(xdiabetestest)[:,1]>cut_off).asStatus(int)
print(predictions)


# Classification Report
#
y_true, y_pred = ydiabetestest, diabetes_logit.predict(xdiabetestest)
print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
#%%
#ROC-AUC
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(ydiabetestest))]
# predict probabilities
lr_probs = diabetes_logit.predict_proba(xdiabetestest)
# keep probabilities for the positive outcome only

# calculate for all response Status
diabetes_response = ['No Diabetes', 'Pre Diabetes', 'Has Diabetes']

for response in range(3):
    # calculate roc scores
    ns_auc = roc_auc_score(ydiabetestest1[:,response], ns_probs)
    lr_auc = roc_auc_score(ydiabetestest1[:, response], lr_probs[:,response])

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ydiabetestest1[:,response], ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(ydiabetestest1[:, response], lr_probs[:,response])

    # calculate auc
    area_curve = auc(lr_fpr, lr_tpr)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic (area = %0.2f)' % area_curve)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # title
    plt.title('ROC Curve: ' + diabetes_response[response],fontweight='bold')
    # show the plot
    plt.show()


#%%
#Non-Full Logistic Regression Model (run several times to simplify down to these variables)
model_diabetes_small = mnlogit(formula='Diabetes_012 ~ C(HighBP) + C(HighChol) + C(CholCheck) + BMI + C(Stroke) + C(Veggies) + C(HvyAlcoholConsump) + C(NoDocbcCost) + C(GenHlth) + MentHlth + C(Sex) + C(Age)+ C(Income)', data=diabetes)

#Model summary information
model_diabetes_small_fit = model_diabetes_small.fit()
print( model_diabetes_small_fit.summary() )
modelpredictions = pd.DataFrame(model_diabetes_small_fit.predict(diabetes)) 
modelpredictions.rename(columns={0:'No Diabetes', 1:'Pre Diabetes', 2:'Has Diabetes'}, inplace=True)
print(modelpredictions.head())


xdiabetes_small = diabetes[
    ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Stroke', 'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost', 
     'GenHlth', 'MentHlth', 'Sex', 'Age', 'Income']]
ydiabetes_small = diabetes['Diabetes_012'].values

class_le = LabelEncoder()
ydiabetes_small = class_le.fit_transform(ydiabetes_small)

ydiabetes_small1 = label_binarize(ydiabetes_small, classes=[0,1,2])
xdiabetes_smalltrain, xdiabetes_smalltest, ydiabetes_smalltrain, ydiabetes_smalltest = train_test_split(xdiabetes_small, ydiabetes_small, train_size=.8,
                                                                                random_state=12345)
xdiabetes_smalltrain1, xdiabetes_smalltest1, ydiabetes_smalltrain1, ydiabetes_smalltest1 = train_test_split(xdiabetes_small, ydiabetes_small1, train_size=.8,
                                                                                random_state=12345)


#Sklearn
diabetes_small_logit = LogisticRegression()
diabetes_small_logit.fit(xdiabetes_smalltrain, ydiabetes_smalltrain)
print('Logit model accuracy (with the test set):', diabetes_small_logit.score(xdiabetes_smalltest, ydiabetes_smalltest))
print('Logit model accuracy (with the train set):', diabetes_small_logit.score(xdiabetes_smalltrain, ydiabetes_smalltrain))
print(diabetes_small_logit.predict(xdiabetes_smalltest))
print(diabetes_small_logit.predict_proba(xdiabetes_smalltrain[:8]))
print(diabetes_small_logit.predict_proba(xdiabetes_smalltest[:8]))

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
y_true_small, y_pred_small = ydiabetes_smalltest, diabetes_small_logit.predict(xdiabetes_smalltest)
print("Confusion Matrix: \n", confusion_matrix(y_true_small, y_pred_small))
print(classification_report(y_true_small, y_pred_small))
#%%
#ROC-AUC
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(ydiabetes_smalltest))]
# predict probabilities
lr_probs = diabetes_small_logit.predict_proba(xdiabetes_smalltest)

# calculate for all response types
for response in range(3):
    # calculate roc scores
    ns_auc = roc_auc_score(ydiabetes_smalltest1[:,response], ns_probs)
    lr_auc = roc_auc_score(ydiabetes_smalltest1[:, response], lr_probs[:,response])

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ydiabetes_smalltest1[:,response], ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(ydiabetes_smalltest1[:, response], lr_probs[:,response])

    area_curve = auc(lr_fpr, lr_tpr)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic (area = %0.2f)' % area_curve)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # plot title
    plt.title('ROC Curve: ' + diabetes_response[response],fontweight='bold')
    # show the plot
    plt.show()

#%%
#=====================KNN======================
#This takes a VERY long time to run when we include the train parameters. The code is available in the event someone wishes to run it, but I don't recommend it unless necessary

for i in [3,5,7,9,11]:
    knn_Diet = KNeighborsClassifier(n_neighbors=i) # instantiate with n value given
    knn_Diet.fit(xdiabetestrain, ydiabetestrain)
    print(f'{i}-NN model accuracy (with the test set):', knn_Diet.score(xdiabetestest, ydiabetestest))
    #print(f'{i}-NN model accuracy (with the train set):', knn_Diet.score(xdiabetestrain, ydiabetestrain))
    y_preds=knn_Diet.predict(xdiabetestest)
    y_pred_score1 = knn_Diet.predict_proba(xdiabetestest)
    print (confusion_matrix(ydiabetestest, y_preds))
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
        plt.title(f'{i}-NN model ROC for Diabetes_012 = {j}', fontweight='bold')
        plt.legend(loc="lower right")
        plt.show()


#%%
#================Decision Tree=================
#Initialize the model
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


#Plotting ROC curves
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
    plt.title('Decision Tree ROC', fontweight='bold')
    plt.legend(loc="lower right")
    plt.show()


# Max depth = 11 (10 was ok, 11 was better, and 12 didn't show improvements from 11)
rf1 = DecisionTreeClassifier(max_depth=11, criterion='entropy', random_state=0)
# Fit dt to the training set
rf1 = rf1.fit(xdiabetestrain, ydiabetestrain)
y_test_pred = rf1.predict(xdiabetestest)
y_train_pred = rf1.predict(xdiabetestrain)
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

rf2 = OneVsRestClassifier(DecisionTreeClassifier(max_depth=11, criterion='entropy'))
# Fit dt to the training set
rf2.fit(xdiabetestrain1, ydiabetestrain1)
y_test_pred1 = rf2.predict(xdiabetestest1)
y_pred_score1 = rf2.predict_proba(xdiabetestest1)

print('Decision Tree results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score (Test): ", accuracy_score(ydiabetestest, y_test_pred) * 100)
print("Accuracy score (Train): ", accuracy_score(ydiabetestrain, y_train_pred) * 100)
print("Confusion Matrix (Test): \n", confusion_matrix(ydiabetestest, y_test_pred,))
print("Confusion Matrix (Train): \n", confusion_matrix(ydiabetestrain, y_train_pred,))
print("Classification report:\n", classification_report(ydiabetestest, y_test_pred))

#Plotting ROC curves
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
# Instantiate Random Forest model
rf1 = RandomForestClassifier(n_estimators=100)
# Fit dt to the training set
rf1.fit(xdiabetestrain, ydiabetestrain)
y_test_pred = rf1.predict(xdiabetestest)
y_pred_score = rf1.predict_proba(xdiabetestest)

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

#Plotting ROC Curves
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
    plt.title('Random Forest ROC', fontweight='bold')
    plt.legend(loc="lower right")
    plt.show()

#Estimators = 200
# Instantiate dtree
rf1 = RandomForestClassifier(n_estimators=200)
# Fit dt to the training set
rf1.fit(xdiabetestrain, ydiabetestrain)
y_test_pred = rf1.predict(xdiabetestest)
y_pred_score = rf1.predict_proba(xdiabetestest)

rf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=200))

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

rf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=50))

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

#Plotting ROC Curves
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
#It is possible to run the rf1 code in ~2 hours on an M1 chip, but the OneVsRestClassifier is too heavy without cloud computing power. The code is left here in the event someone wants to run this with higher computing power.

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

#Plotting ROC Curves
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
    plt.title('SVC ROC', fontweight='bold')
    plt.legend(loc="lower right")
    plt.show()
