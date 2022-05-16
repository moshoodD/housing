# ---
title: |
  <center>A Python Jupyter Notebook</center>  
  <center>Converted to RMarkdown</center>
  <br />
author: |
  <p style="float: right">Dayo Moshood</p>
  <br />
date: |
  <p style="float: right">May 10th, 2022</p>
  <br />
output:
  html_document:
    highlight: textmate
    theme: flatly
    number_sections: yes
    toc: yes
    toc_float:
      collapse: yes
      smooth_scroll: yes
---
```{python}
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
```



```{python}
pd.options.display.max_rows = 100

%matplotlib inline
```



### DATA LOADING AND EXPLORATION

```{python}
data = pd.read_csv("miami-housing.csv")

data.head()
```



```{python}
data.info()
```



```{python}
sns.set_style("whitegrid")
```



### EXPLORATORY ANALYSIS

```{python}
data.describe().T.to_csv("description.csv")
```



```{python}
from sklearn.cluster import KMeans
```



#### UNSUPERVISED MACHINE LEARNING TO FIND THE POSITION OF HOUSES 

```{python}
kmeans = KMeans(n_clusters=2)

geo = data[["LATITUDE", "LONGITUDE"]]

kmeans.fit(geo)
```



```{python}
kmeans.labels_
```



```{python}
data["area"] = kmeans.labels_
```



```{python}
sns.distplot(data["SALE_PRC"], bins=15)
plt.title("Distribution of Price of Houses")
plt.xlabel("Price of Houses (unit is in power of 10^6)")
```



```{python}
sns.boxplot(data["SALE_PRC"])
```



```{python}
months = data["month_sold"].value_counts()

ms = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
new_ms = months.sort_index()

new_ms.index = ms

new_ms
```



```{python}
prices = []
averages = []
target = data[["SALE_PRC", "month_sold"]]
mon = list(data["month_sold"].value_counts().index)

for m in sorted(mon):
    tot = 0
    count = 0
    for u,v in zip(target["SALE_PRC"], target["month_sold"]):
        if m == int(v):
            tot += u
            count += 1
    avg = tot / count
    prices.append(tot)
    averages.append(avg)
```



```{python}
sns.displot(data["age"], bins=15)
plt.xlabel("Age")
```



```{python}
sns.scatterplot(x="LND_SQFOOT", y="TOT_LVG_AREA", hue="area", data=data)
plt.xlabel("Land Area (sq feet)")
plt.ylabel("Floor Area (sq feet)")
```



```{python}
sns.scatterplot(x="LATITUDE", y="LONGITUDE", hue="area", data=data)
```



```{python}
sns.heatmap(data.corr())
```



```{python}
sns.scatterplot(x="LATITUDE",y="SALE_PRC", hue="area",data=data)
```



```{python}
sns.scatterplot(x="LONGITUDE",y="SALE_PRC", hue="area",data=data)
```



```{python}
sns.scatterplot(x="structure_quality",y="SALE_PRC", hue="area",data=data)
```



```{python}
sns.scatterplot(x="TOT_LVG_AREA",y="SALE_PRC", hue="area",data=data)
```



```{python}
sns.scatterplot(x="SUBCNTR_DI",y="CNTR_DIST", hue="area",data=data)
```



```{python}
sns.scatterplot(x="age",y="SALE_PRC", hue="area",data=data)
plt.ylabel("HOUSE PRICE")
```



```{python}
sns.scatterplot(x="CNTR_DIST",y="WATER_DIST", hue="area",data=data)
```



```{python}
sns.scatterplot(x="LONGITUDE",y="WATER_DIST", hue="area",data=data)
```



```{python}
sns.scatterplot(x="LONGITUDE",y="CNTR_DIST", hue="area",data=data)
```



```{python}
sns.scatterplot(x="LATITUDE",y="age", hue="area",data=data)
```



```{python}
sns.scatterplot(x="LONGITUDE",y="age", hue="area",data=data)
```



```{python}
sns.scatterplot(x="SPEC_FEAT_VAL",y="SALE_PRC", hue="area",data=data)
```



```{python}
sns.scatterplot(x="LND_SQFOOT",y="SALE_PRC", hue="area",data=data)
```



```{python}
data.corr()
```



## DATA PREPARATION

```{python}
pps = []
count = 1
for p,s in list(zip(data["SALE_PRC"], data["TOT_LVG_AREA"])):
    ps = p / s
    pps.append(ps)
len(pps)

data["price_per_sqfoot"] = pps
```



```{python}
sns.boxplot(x=data["price_per_sqfoot"])
```



```{python}
data["price_per_sqfoot"].median()
```



## CREATION OF TRAIN AND TEST DATA

```{python}
from sklearn.model_selection import train_test_split

X = data.drop(["SALE_PRC"],axis=1)
Y = data["SALE_PRC"]
```



```{python}
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
```



```{python}
metrics.r2_score(y_test, cat_pred)
```



```{python}
X_test.shape
```



```{python}
X_train.shape
```



### SELECTION AND CREATION OF MODELS FOR TRAINING AND TESTING

```{python}
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
```



```{python}
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# from lightgbm import LGBMRegressor
```



```{python}
standard_sc = StandardScaler()
linear_reg = LinearRegression()
random_f = RandomForestRegressor(n_estimators=500, random_state=42)
gradient = GradientBoostingRegressor(n_estimators=500, random_state=42)
ada = AdaBoostRegressor(n_estimators=500, random_state=42)
cat = CatBoostRegressor(random_state=42, n_estimators=500)
xgb = XGBRegressor()
```



### SCALING OF DATASET USING STANDARD SCALING

```{python}
X_train_scaled = standard_sc.fit_transform(X_train)
X_test_scaled = standard_sc.transform(X_test)
```



```{python}
linear_reg.fit(X_train_scaled,y_train)

random_f.fit(X_train_scaled, y_train)

gradient.fit(X_train_scaled,y_train)

ada.fit(X_train_scaled,y_train)

cat.fit(X_train_scaled, y_train)

xgb.fit(X_train_scaled,y_train)
```



```{python}
from sklearn import metrics
```



```{python}
linear_pred = linear_reg.predict(X_test_scaled)

random_pred = random_f.predict(X_test_scaled)

gradient_pred = gradient.predict(X_test_scaled)

ada_pred = ada.predict(X_test_scaled)

cat_pred = cat.predict(X_test_scaled)

xgb_pred = xgb.predict(X_test_scaled)
```



### EVALUATION

```{python}
metrics.mean_absolute_error(y_test, cat_pred)
```



```{python}
metrics.mean_squared_error(y_test, cat_pred)
```



```{python}
metrics.r2_score(y_test, cat_pred)
```



```{python}
metrics.mean_absolute_error(y_test, linear_pred)
```



```{python}
metrics.mean_squared_error(y_test, linear_pred)
```



```{python}
metrics.r2_score(y_test, linear_pred)
```



```{python}
metrics.mean_absolute_error(y_test, random_pred)
```



```{python}
metrics.mean_squared_error(y_test, random_pred)
```



```{python}
metrics.r2_score(y_test, random_pred)
```



```{python}
metrics.mean_absolute_error(y_test, ada_pred)
```



```{python}
metrics.mean_squared_error(y_test, ada_pred)
```



```{python}
metrics.r2_score(y_test, ada_pred)
```



```{python}
metrics.mean_absolute_error(y_test, xgb_pred)
```



```{python}
metrics.mean_squared_error(y_test, xgb_pred)
```



```{python}
metrics.r2_score(y_test, xgb_pred)
```



### REGRESSION PLOT FOR EACH MODEL CHOSEN

```{python}
sns.regplot(x=y_test, y=gradient_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices for Gradient Boosting Regressor")
```



```{python}
sns.regplot(x=y_test, y=cat_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices for CatBoost Regressor")
```



```{python}
sns.regplot(x=y_test, y=xgb_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices for XGBoost Regressor")
```



```{python}
sns.scatterplot(x=y_test, y=xgb_pred)
```



