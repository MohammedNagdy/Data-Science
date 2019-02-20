# here we prepare the data for training
import pandas as pd
import numpy as np
import os


# load the data
# get the data from https://www.kaggle.com/c/titanic/data
df = pd.read_csv("train.csv")
# print(df.head())
# print(df.dtypes)
# print(df["Embarked"].head())


# drop ticket passId
# drop nan
df = df.drop(["Ticket", "PassengerId"], axis=1)
df = df.dropna()

# encode every categorical column
df_cat = df.select_dtypes(include="object").copy()
# binary encoding
def bin_encode(data):
    newData = []
    newData = pd.DataFrame(newData)
    for col in data.columns:
        data[col] = np.array(data[col].values) # tranforming from data frame to numpy array to ge unique cat-values
        cat = np.unique(data[col]) # every unique categorical variable
        for childcat in cat:
            newData[childcat] = [1 if category == childcat else 0 for category in data[col]] # transform every unique variable to binary
    return newData

cat_vars = bin_encode(df_cat)
print(cat_vars.head())

# concatenate the columns
df_num = df.select_dtypes(include=["float64", "int64"])
X = np.concatenate([df_num, cat_vars],axis=1)
print(X[:5])

if not os.path.exists("trainable"):
    os.mkdir("trainable")
np.savetxt(os.path.join("trainable","trainit.txt"), X)
