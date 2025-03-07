<H3>ENTER YOUR NAME : Arshitha MS</H3>
<H3>ENTER YOUR REGISTER NO: 212223240015</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 07/03/2025</H3>
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
### Import Libraries:
```py
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
### Read the dataset:
```py
df=pd.read_csv("Churn_Modelling.csv")
df.head()
df.tail()
df.columns
```
### Check the missing data
```py
df.isnull().sum()
df.duplicated()
```
### Assigning Y
```py
y = df.iloc[:, -1].values
print(y)
```
### Check for duplicates
```py
df.duplicated()
```
### Check for outliers
```py
df.describe()
```
### Dropping string values data from dataset
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
### Checking datasets after dropping string values data from dataset
```py
data.head()
```
### Normalize the dataset
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
### Split the dataset
```py
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
### Training and testing model
```py
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
### Data checking

![image](https://github.com/user-attachments/assets/90c232b8-fce8-444b-9aab-7d8e96caa2cf)


### Missing Data 

![image](https://github.com/user-attachments/assets/db0bfcf1-87c6-46f6-b966-a9eb81df055e)


### Duplicates identification

![image](https://github.com/user-attachments/assets/4fee6095-5299-4333-9e0a-09c29d1da55c)


### Vakues of 'Y'

![image](https://github.com/user-attachments/assets/6a580db6-6225-4abb-a35b-a01e91e56d06)

### Outliers

![image](https://github.com/user-attachments/assets/591455ff-c359-45c2-875b-a6c71b460903)


### Checking datasets after dropping string values data from dataset

![image](https://github.com/user-attachments/assets/e5fbdb47-9f27-47f6-b614-3ddbc5f014a9)


### Normalize the dataset

![image](https://github.com/user-attachments/assets/6794a3eb-ca45-4ef9-8aa4-770d4a696a7a)


### Split the dataset

![image](https://github.com/user-attachments/assets/8016c148-8ec2-4234-abb9-19c6ebfb03bf)


### Training and testing model

![image](https://github.com/user-attachments/assets/e13b0f42-1109-478d-a20e-74cf77107d5f)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


