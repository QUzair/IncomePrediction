# Data Analysis
import pandas as pd
import numpy as np

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Data Prediction
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import mean_squared_error

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input/inc-wlabels"))

# Importing the dataset
dataset = pd.read_csv('../input/inc-wlabels/inc_wlabels.csv') # 111993
dt = pd.read_csv('../input/inc-wlabels/inc_wo_labels.csv') #73230

# Remove negative incomes
# dataset.drop(dataset.loc[dataset['Income in EUR']<0].index, inplace=True)

org_dt = dataset # 111827
# print(dataset.shape, dt.shape)
desc_X = dataset.describe()

y = org_dt.iloc[:,11].values.reshape(-1,1) # 111827
# y = np.log(y)

# Merge both labelled and unlabelles so they may be processed together
dataset = pd.concat([dataset,dt],axis='rows')
print(dataset.shape)
print(org_dt.shape)
print(y.shape)
desc_X
y

dataset.nunique()

## Null Counting Fucntion
def null_values(df):
    
    sum_null = df.isnull().sum()
    total = df.isnull().count()
    percent_nullvalues = 100* sum_null / total 
    df_null = pd.DataFrame()
    df_null['Total'] = total
    df_null['Null_Count'] = sum_null
    df_null['Percent'] = round(percent_nullvalues,2)
    df_null = df_null.sort_values(by='Null_Count',ascending = False)
    df_null = df_null[df_null.Null_Count > 0]
    
    return(df_null)

print(null_values(org_dt))
null_values(dataset)

null_values(dataset)

# Fill missing values
dataset = dataset.fillna({
    'University Degree':'No',
    'Gender':'unknown',
    'Hair Color':'None',
    'Profession':'None',
    'Age': org_dt['Age'].mean(),
    'Year of Record': org_dt['Year of Record'].mean()
})

dataset['Gender'] = dataset['Gender'].replace({'0': 'unknown'})
dataset['Hair Color'] = dataset['Hair Color'].replace({'0': 'Unknown'})
dataset['University Degree'] = dataset['University Degree'].replace({'0': 'No'})

dataset.columns = ['Instance', 'Year of Record', 'Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height cm','Income in EUR']
y = pd.DataFrame.from_records(y)

null_values(dataset)

comm_profess = pd.Series()
comm_profess['comm_profess'] = pd.Series(' '.join(dataset['Profession']).lower().split()).value_counts()[:500].index.tolist()
comm_profess['comm_profess'].remove('and')
comm_profess['comm_profess'].remove('&')
comm_profess['comm_profess'].remove('-')

def replaceProf(x):
    arr = x.split() 
    for p in range(0,len(comm_profess['comm_profess'])) :
        for a in range(0,len(arr)):
            if (comm_profess['comm_profess'][p]==arr[a]):
                # print(x,comm_profess['comm_profess'][p])
                x = comm_profess['comm_profess'][p]
                return comm_profess['comm_profess'][p]
            else:
                x = 'worker'

# Extracts the most common words from the Profession

"""
dataset['Profession'] = dataset['Profession'].apply(lambda x: replaceProf(x))
# org_dt['Profession'].apply(lambda x: replaceProf(x))
dataset['Profession'].to_csv('datprof.csv')
"""
dataset['Uni Degree'] = pd.Categorical(dataset['University Degree']).rename_categories({'No': 0, 'Bachelor': 1,'Master':2,'PhD':3})
dataset['Uni Degree'].value_counts()
dataset['Uni Degree'] = dataset['Uni Degree'].astype('int64')
dataset['Uni Degree']

# Creating Dummy Variables
dummy_gender = (pd.get_dummies(dataset['Gender'])).iloc[:, :-1] # 4 gender
dummy_country = pd.get_dummies(dataset['Country']).iloc[:, :-1] #159 countries
dummy_profession = pd.get_dummies(dataset['Profession']).iloc[:, :-1] # 1339 professions
# dummy_degree = pd.get_dummies(dataset['University Degree']).iloc[:, :-1] # 4 degrees 
dummy_haircolor = pd.get_dummies(dataset['Hair Color']).iloc[:, :-1] # 5 hair


dataset['CitySizeLog'] = np.log(dataset['Size of City'])
dataset['IncomeLog'] = np.log(dataset['Income in EUR'])
# dataset['Year of Record'] = dataset['Year of Record'].apply(lambda x: x - 1980)
dataset['Year of Record']

"""dummy_gender = (pd.get_dummies(dataset['Gender'])) # 4 gender
dummy_country = pd.get_dummies(dataset['Country']) #159 countries
dummy_profession = pd.get_dummies(dataset['Profession']) # 1339 professions
# dummy_degree = pd.get_dummies(dataset['University Degree']).iloc[:, :-1] # 4 degrees 
dummy_haircolor = pd.get_dummies(dataset['Hair Color']) # 5 hair"""

y.columns = ['Income']
countryAves = pd.concat([org_dt['Country'],
                        org_dt['Size of City'],
                        org_dt['Age'],
                        y['Income']], axis=1)

b = countryAves.groupby('Country').mean()
b = b.rename({ 'Size of City':'CountryAveCitySize',
               'Income':'CountryAveEarning',
               'Age':'CountryAveAge'}, axis=1)

# Empty spaces can be added in later by autoclean...
X = org_dt['Country']
#X = pd.merge(X, b, on='Profession', how='left', sort=False)
X = X.reset_index().merge(b, on='Country', how="left").set_index('index')
X = X.drop('Country', axis=1)
X

prof = org_dt['Profession'].str.lower()
X = pd.concat([X, prof], axis=1)
professionAves = pd.concat([org_dt['Profession'].str.lower(),
                            org_dt['Size of City'],
                            org_dt['Age'],
                            X['CountryAveCitySize'],
                            X['CountryAveEarning'],
                            X['CountryAveAge'],
                            y['Income']], axis=1)

c = professionAves.groupby('Profession').mean().reset_index()
c = c.rename({ 'Size of City':'ProfessionAveCitySize',
               'Income':'ProfessionAveEarning',
               'Age':'ProfessionAveAge',
             'CountryAveAge':'CountryProfAveAge',
             'CountryAveEarning':'CountryProfAveEarning',
             'CountryAveAge':'CountryProfAveAge'}, axis=1)

X = X.reset_index().merge(c, on='Profession', how="left").set_index('index')
X = X.drop('Profession', axis=1)
X

import seaborn as sns
# Identify Numeric features
numeric_features = ['Year of Record','Age','Size of City','Uni Degree','Wears Glasses','Body Height cm','Income in EUR','IncomeLog']

# Identify Categorical features
cat_features = ['Gender','Country','Profession','Hair Color']

# Correlation matrix between numerical values
g = sns.heatmap(dataset[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

# Dataset copy to ocmpare 
dataset2 = dataset

#Drop categorical columns
dataset2 = dataset2.drop(['Gender','University Degree','IncomeLog','Country','Profession','Size of City','Hair Color','Instance','Wears Glasses','Income in EUR'],axis='columns')
# dataset2 = dataset2.drop(['Gender','Country','Profession','University Degree','Hair Color','Instance'],axis='columns')

# Merge dummy variables
merged  = pd.concat([dataset2,dummy_gender],axis="columns")
# merged  = pd.concat([dataset2,dummy_gender,dummy_degree,dummy_haircolor],axis="columns")

merged = pd.concat([merged[0:111993],X],axis=1)
merged

# Finding Outliers, +0.7 or higher: Very strong positive relationship

"""
org_dt = merged.iloc[0:111827,:]
org_dt = pd.DataFrame.from_records(org_dt)
org_dt['IncomeLog'] = np.log(org_dt['Income in EUR'])

corr = org_dt.corr().abs()
corr.IncomeLog[corr.IncomeLog >= 0.005].sort_values(ascending=False)
"""

"""org_dt.plot.scatter(x='Age', y='IncomeLog', color = 'green')
plt.show()"""

# Splitting the dataset into the Training set and Test set manually

"""X_train = merged.iloc[0:78000,:] 
X_test = merged.iloc[78000:111827,:]
y_train = y[0:78000]
y_test = y[78000:111827]"""


X_train = merged.iloc[0:111993,:] 
X_test = merged.iloc[100000:111993,:]
y_train = y[0:111993]
y_test = y[100000:111993]


"""
xgb = xgboost.XGBRegressor(colsample_bytree=0.8, subsample=0.5,
                             learning_rate=0.05, max_depth=8, 
                             min_child_weight=2.8, n_estimators=1000,
                             reg_alpha=0.3, reg_lambda=0.75, gamma=0.01, 
                             silent=1, random_state =7, nthread = -1)
"""

xgb = xgboost.XGBRegressor(colsample_bytree=0.4,
                   eval_metric='rmse',
                 gamma=0.01,
                   nthread=-1,
                 learning_rate=0.05,
                 max_depth=4, 
                 min_child_weight=1.8,
                 n_estimators=3000,                                                     
                 reg_alpha=0.3,
                 reg_lambda=0.75,
                 subsample=0.5,
                 seed=42)

"""# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
xgb = LinearRegression()"""

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
y_t2 = xgb.predict(X_train)

RMSE = np.sqrt(mean_squared_error(y_train, y_t2))
print(np.sqrt(mean_squared_error(xgb_pred, y_test)))
print(RMSE.round(4))
score = xgb.score(X_test,y_test)
print(score)

y_dt_pred = xgb.predict(merged.iloc[111827:,:])
corrected_pred = y_dt_pred
for i in range(0,len(corrected_pred)):
    if corrected_pred[i] < 0:
        corrected_pred[i] = corrected_pred[i]*(-1)
        
corrected_pred.flatten()        
y_dt_pred.flatten()

submission = pd.DataFrame({
        "Instance": dt['Instance'],
        "Income": y_dt_pred.flatten()
    })
submission = pd.DataFrame.from_records(submission)
submission.to_csv('submission.csv')

submission_corr = pd.DataFrame({
        "Instance": dt['Instance'],
        "Income": corrected_pred.flatten()
    })
submission_corr = pd.DataFrame.from_records(submission)
submission.to_csv('submission_corr.csv')
print(submission_corr)