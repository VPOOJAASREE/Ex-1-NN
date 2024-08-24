<H3>ENTER YOUR NAME: V. POOJAA SREE</H3>
<H3>ENTER YOUR REGISTER NO.: 212223040147</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 24.08.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv")
df
df.isnull().sum()
df.duplicated()
print(df['CreditScore'].describe())
df.info()
df.drop(['Surname','CustomerId','Geography','Gender'],axis=1,inplace=True)
df
scaler=MinMaxScaler()
df=pd.DataFrame(scaler.fit_transform(df))
df
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))

```


## OUTPUT:

![1](https://github.com/user-attachments/assets/cc422c21-6d36-4473-aed7-cd66466ac24f)

![2](https://github.com/user-attachments/assets/d913280b-2aa6-4e6e-84ca-826774d3a9f9)

![3](https://github.com/user-attachments/assets/77bd8f52-7394-4fad-ba30-96bb903b7bc7)

![4,5](https://github.com/user-attachments/assets/f191e6f2-dd4c-4490-8777-64bfa4b42153)

![6](https://github.com/user-attachments/assets/f85415c2-db11-4723-a16a-51f9757db9f0)

![7](https://github.com/user-attachments/assets/14b2ceae-4abb-42d4-a5c1-0bbdae5662a5)

![8,9](https://github.com/user-attachments/assets/8c4dca0d-5473-4b16-b097-c8f3b57975f9)

![10](https://github.com/user-attachments/assets/f9e64ecd-d6c2-4049-b16b-086d61e560c5)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


