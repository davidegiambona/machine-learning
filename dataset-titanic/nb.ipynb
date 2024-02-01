import numpy as np
import pandas as pd #Load Pandas library

print("Versione di pandas: ", pd.__version__)
#Load Dataset (read_csv(), function):

df = pd.read_csv('./data/titanic.csv')

df
df.shape #attribute shape (nrows, ncolumns)
head() / tail() :

head() : Return the first n rows. (5 by default)

tail(): Return the last n rows. (5 by default)
df.head() #first 5 rows by DEFAULT
df.tail() #last 5 rows bt DEFAULT
df['Survived'] #Series
#Return a Series containing counts of unique values.

#The resulting object will be in descending order.

df['Survived'].value_counts()
df[['Name', 'Survived']] #Df Subset
df_survived_2 = df[["Pclass", "Survived"]].groupby('Pclass').mean().reset_index()

#print(df_survived_2)

print(df_survived_2.plot(kind='bar'))
# analisi di sopravvivenza delle persone per sesso

df[['Survived', 'Sex']].groupby('Sex').mean().plot(kind='bar')

# print(df[['Survived', 'Sex']].groupby('Sex').mean())

survivors_df = df[['Survived', 'Sex']].groupby('Sex').sum() #SUM VALUES: 0 (NOT SURVIVED) , 1 (SURVIVED)
survivors_df['Total'] = df[['Survived', 'Sex']].groupby('Sex').count() #COUNT TOTAL NUMBER OF PEOPLE PER CLASS (MALE,FEMALE)
print(survivors_df)
# analisi per verificare la difficoltà di sopravvivenza delle grandi famiglie 

df['family_size'] = 1 + df['SibSp'] + df['Parch']

# print(df.groupby('family_size').mean()['Survived'])

# analisi per verificare il tasso di morte in base alla distinzione dei titoli

df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

print("\n", df.groupby('Title').count()['PassengerId'].sort_values().plot(kind='bar'))

# df.groupby('Title').mean()['Survived'].sort_values()
# df.groupby('Title').count()[['PassengerId']]
df.head() # first 5 rows by DEFAULT

df.tail() # last 5 rows bt DEFAULT
