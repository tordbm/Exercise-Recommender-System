
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Importing the dataset
df = pd.read_csv("megaGymDataset.csv")
df = df.rename(columns={'Unnamed: 0': 'index'})

df = df.drop_duplicates('Title', keep='last')

# top rated exercises
ratingSorted = df.sort_values(by='Rating',ascending=False)
ratingSorted = ratingSorted.head(10)

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
x = x.drop(["Rating"], axis = 1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Grid search for å finne beste params
param_grid = {
    'n_neighbors': [3,5,7,9,11,13,15,17],
    'p': [1, 2]
}
grid_search = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
params = grid_search.best_params_

# Traiing
knn = KNeighborsRegressor(n_neighbors = params['n_neighbors'], p = params["p"])
knn.fit(X_train, y_train)


# Ny variabel X. Alle rader fra dataframe som ikke har rating
x = deepcopy(df)
# Ekskluderer øvelser med ratings
x = df[df['Rating'].isin([0, np.nan])]

# Gjør om strenger til kategorier (int) for prediction
x = x.drop(["Rating"], axis = 1)
x = x.drop(["Title"], axis = 1)
x = x.drop(["Desc"], axis = 1)
x = x.drop(["RatingDesc"], axis = 1)
x['Level'] = pd.factorize(x['Level'])[0]
x['Type'] = pd.factorize(x['Type'])[0]
x['BodyPart'] = pd.factorize(x['BodyPart'])[0]
x['Equipment'] = pd.factorize(x['Equipment'])[0]

# Predikerer en rating for hver rad i dataframe som ikke har rating
for index, row in x.iterrows():
    rating = knn.predict([row]).round(decimals=1)
    df.loc[df['index'] == index, 'Rating'] = rating

filtered_df = df[df["Rating"] == 0]

# Removing irrelevant columns
df = df.drop('RatingDesc', axis=1)
# Removing all rows containing nonvalues in description
df = df[df['Desc'].notna()]
# Removing ID column
df.pop(df.columns[0])

# Dataset after preprocessing
clean_df = deepcopy(df)

# Merging columns for cosign similarity and dropping excess columns
df["Merged"] = df["Type"].astype(str) + '|' + \
  df["BodyPart"].astype(str) + '|' + df["Equipment"].astype(str) + '|' + \
  df["Level"]

df = df.drop('Type', axis=1)
df = df.drop('BodyPart', axis=1)
df = df.drop('Equipment', axis=1)
df = df.drop('Level', axis=1)

# Converting values of the merged column into vectors
count = CountVectorizer()
count_matrix = count.fit_transform(df.loc[:,"Merged"])

# liste = count_matrix.toarray()

# Cosine similarity
sim_matrix = cosine_similarity(count_matrix, count_matrix)

# Resetting the index to avoid indexing errors and NAN values in recommender
# This makes the previous indexes invalid
# "drop" avoids adding the old index as a column
df = df.reset_index(drop = False)

# Sets the index of the input from txt file
with open('entries.txt', 'r') as file:
  entry = file.readline()
if entry.lower() in df['Title'].str.lower().values:
    row = df[df['Title'].str.lower() == entry.lower()]
    index = row.index
else:
    print(f"{entry} not found in dataframe")
    df_empty = pd.DataFrame()
    df_empty.to_csv('recommended.csv', index=False)
    quit()

def recommender(data_frame, exercise_id, sim_matrix):
    sim_df = pd.DataFrame(sim_matrix[exercise_id],
                         columns=["Similarity"])
    exercise_titles = data_frame.loc[:, "Title"]
    exercise_rec = pd.concat([sim_df, exercise_titles], axis = 1)
    return exercise_rec

df_by_cat = recommender(df, index[0], sim_matrix)

tfidf = TfidfVectorizer(stop_words="english")
overview_matrix = tfidf.fit_transform(df["Desc"])

similarity_matrix = linear_kernel(overview_matrix, overview_matrix)

mapping = pd.Series(df.index, index = df["Desc"])

def recommender_by_desc(exercise_input, df, similarity_matrix, mapping):
    exercise_index = mapping[exercise_input]
    if not isinstance(exercise_index, np.int64):
        exercise_index = exercise_index[0]
    similarity_score = list(enumerate(similarity_matrix[exercise_index]))
    score = [tup[1] for tup in similarity_score]
    exercise_indices = [i[0] for i in similarity_score]
    df2 = df["Title"].iloc[exercise_indices].to_frame()
    df2["Similarity"] = score
    return df2

df_by_desc = recommender_by_desc(df["Desc"].iloc[index[0]], df, similarity_matrix, mapping)

merged_df = df_by_cat.copy()
merged_df["Similarity"] = (df_by_cat["Similarity"] + df_by_desc["Similarity"]) / 2
merged_df = merged_df.sort_values(by=["Similarity"], ascending = False)
merged_df[0:10].to_csv("recommended.csv")
print("Done")
