
import pandas as pd
""" import matplotlib.pyplot as plt """


# Importing the dataset
df = pd.read_csv("megaGymDataset.csv")
df = df.rename(columns={'Unnamed: 0': 'index'})
df


#Cheking if there is any NULL or missing values
df.isna().sum()


# DATA ANALYSIS

# Some exercises has the same title - Should remove duplicates?
# df = df.drop_duplicates('Title', keep='last')
df['Title'].value_counts()


""" # Sorted bv level
df['Level'].value_counts().plot.barh()


# sorted by type
df['Type'].value_counts().plot.barh()


# sorted by bodypart
df['BodyPart'].value_counts().plot.barh() """


# top rated exercises
ratingSorted = df.sort_values(by='Rating',ascending=False)
ratingSorted = ratingSorted.head(10)


# Prints the row of the given Title to find the index
print(df[df["Title"] == "Bench press"])
df.loc[df['Title'] == "Bench press", 'Rating'] = 10
print(df[df["Title"] == "Bench press"])


""" import matplotlib.pyplot as plt """
""" 
# Your code to create the bar chart
plt.figure(figsize=(6, 6))
df['Rating'].value_counts().plot.barh()
plt.yticks([])
plt.ylabel('Rating')
plt.xlabel('Amount')

# Add labels at the highest and lowest data points on the y-axis
plt.text(-5, 70, 0, ha='center')
plt.text(-5, 0, 10, ha='center') """


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import numpy as np

# Datasett for trening. Gjør om strenger til kategorier (int)
x = deepcopy(df)
x = x.drop(["Title"], axis = 1)
x = x.drop(["Desc"], axis = 1)
x = x.drop(["RatingDesc"], axis = 1)
x['Level'] = pd.factorize(x['Level'])[0]
x['Type'] = pd.factorize(x['Type'])[0]
x['BodyPart'] = pd.factorize(x['BodyPart'])[0]
x['Equipment'] = pd.factorize(x['Equipment'])[0]
x = x[x['Rating'].notna()]
x = x[df["Rating"] != 0]
# Verdier som skal predikeres, brukes for trening og testing
y = x["Rating"]
#y = y.round(0)
#y = y.astype(int)
x = x.drop(["Rating"], axis = 1)
#y=y.replace(0,1)
#y=y.replace(0.0,1)
#y=y.replace(np.nan,1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Grid search for å finne beste params
from sklearn.model_selection import GridSearchCV
'''
param_grid = {
    'n_neighbors': [3,5,7,9,11,13,15,17],
    'p': [1, 2]
}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
params = grid_search.best_params_
'''
param_grid = {
    'n_neighbors': [3,5,7,9,11,13,15,17],
    'p': [1, 2]
}
grid_search = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
params = grid_search.best_params_
print(params)

# Traiing
knn = KNeighborsRegressor(n_neighbors = params['n_neighbors'], p = params["p"])
knn.fit(X_train, y_train)


# Ny variabel X. Alle rader fra dataframe som ikke har rating
x = deepcopy(df)
# Ekskluderer øvelser med ratings
x = x[x["Rating"].isna()]

# Gjør om strenger til kategorier (int) for prediction
x = x.drop(["Rating"], axis = 1)
x = x.drop(["Title"], axis = 1)
x = x.drop(["Desc"], axis = 1)
x = x.drop(["RatingDesc"], axis = 1)
x['Level'] = pd.factorize(x['Level'])[0]
x['Type'] = pd.factorize(x['Type'])[0]
x['BodyPart'] = pd.factorize(x['BodyPart'])[0]
x['Equipment'] = pd.factorize(x['Equipment'])[0]

# Antall nonvalues
print("Nonvalues rating før:",df["Rating"].isna().sum())

# Predikerer en rating for hver rad i dataframe som ikke har rating
for index, row in x.iterrows():
    rating = knn.predict([row])
    #print(row["index"], rating)
    df.loc[df['index'] == index, 'Rating'] = rating

print("Nonvalues rating etter",df["Rating"].isna().sum())

filtered_df = df[df["Rating"] == 0]
print(len(filtered_df))


""" import matplotlib.pyplot as plt

# Your code to create the bar chart
plt.figure(figsize=(6, 6))
df['Rating'].value_counts().plot.barh()
plt.yticks([])
plt.ylabel('Rating')
plt.xlabel('Amount')

# Add labels at the highest and lowest data points on the y-axis
plt.text(-5, 172, 0, ha='center')
plt.text(-5, 0, 10, ha='center')

plt.show() """



#print(x["Rating"])
#import numpy as np
#print(x["Rating"].dtypes)
#x = x[x["Rating"].isna()]
#print(x["Rating"])


# Removing columns with lots of nonvalues
#df = df.drop('Rating', axis=1)
df = df.drop('RatingDesc', axis=1)
# Removing all rows containing nonvalues in description
df = df[df['Desc'].notna()]
#df = df[df['Rating'].notna()]
# Removing ID column
df.pop(df.columns[0])




# Checking datatypes
df.dtypes


# Merging columns for cosign similarity and dropping excess columns
df["Merged"] = df["Type"].astype(str) + '|' + \
  df["BodyPart"].astype(str) + '|' + df["Equipment"].astype(str) + '|' + \
  df["Level"]

df = df.drop('Type', axis=1)
df = df.drop('BodyPart', axis=1)
df = df.drop('Equipment', axis=1)
df = df.drop('Level', axis=1)


# The merged columns
df["Merged"]


# Converting values of the merged column into vectors

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df.loc[:,"Merged"])

liste = count_matrix.toarray()


# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(count_matrix, count_matrix)


#sim_matrix


# Resetting the index to avoid indexing errors and NAN values in recommender
# This makes the previous indexes invalid
# "drop" avoids adding the old index as a column
df = df.reset_index(drop = False)


def recommender(data_frame, exercise_id, sim_matrix):
    sim_df = pd.DataFrame(sim_matrix[exercise_id],
                         columns=["similarity"])
    exercise_titles = data_frame.loc[:, "Title"]
    exercise_rec = pd.concat([sim_df, exercise_titles], axis = 1)

    exercise_rec = exercise_rec.sort_values(by=["similarity"], ascending = False)

    return exercise_rec.iloc[1:20,:]


# Prints the row of the given Title to find the index
with open('entries.txt', 'r') as file:
  entry = file.readline()
row = df[df["Title"] == entry]
print(row)
index = row.index


# Exercises similar to bench press
df_recommended = recommender(df, index[0], sim_matrix)

df_recommended.to_csv('recommended.csv', index=False)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


tfidf = TfidfVectorizer(stop_words="english")
overview_matrix = tfidf.fit_transform(df["Desc"])
overview_matrix.shape


similarity_matrix = linear_kernel(overview_matrix, overview_matrix)
print(similarity_matrix[0:5,0:5])


mapping = pd.Series(df.index, index = df["Desc"])
mapping


def recommender_by_desc(exercise_input):
    exercise_index = mapping[exercise_input]
    similarity_score = list(enumerate(similarity_matrix[exercise_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[0:10]

    exercise_indices = [i[0] for i in similarity_score]
    return df["Title"].iloc[exercise_indices]


recommender_by_desc(df["Desc"][0])

print('Done')