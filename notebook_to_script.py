# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Importing the dataset
df = pd.read_csv("megaGymDataset.csv")
df

# %%
#Cheking if there is any NULL or missing values
df.isna().sum()

# %%
# DATA ANALYSIS

# Some exercises has the same title - Should remove duplicates?
# df = df.drop_duplicates('Title', keep='last')
df['Title'].value_counts()

# %%
# Sorted bv level
df['Level'].value_counts().plot.barh()

# %%
# sorted by type
df['Type'].value_counts().plot.barh()

# %%
# sorted by bodypart
df['BodyPart'].value_counts().plot.barh()

# %%
# top rated exercises
ratingSorted= df.sort_values(by='Rating',ascending=False)
ratingSorted =ratingSorted.head(10)
ratingSorted

# %%
# Removing columns with lots of nonvalues
df = df.drop('Rating', axis=1)
df = df.drop('RatingDesc', axis=1)
# Removing all rows containing nonvalues in description
df = df[df['Desc'].notna()]
# Removing ID column
df.pop(df.columns[0])



# %%
# Checking datatypes
df.dtypes

# %%
# Merging columns for cosign similarity and dropping excess columns
df["Merged"] = df["Type"].astype(str) + '|' + \
  df["BodyPart"].astype(str) + '|' + df["Equipment"].astype(str) + '|' + \
  df["Level"]

df = df.drop('Type', axis=1)
df = df.drop('BodyPart', axis=1)
df = df.drop('Equipment', axis=1)
df = df.drop('Level', axis=1)

# %%
# The merged columns
df["Merged"]

# %%
# Converting values of the merged column into vectors

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df.loc[:,"Merged"])

liste = count_matrix.toarray()

# %%
# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(count_matrix, count_matrix)

# %%
#sim_matrix

# %%
# Resetting the index to avoid indexing errors and NAN values in recommender
# This makes the previous indexes invalid
# "drop" avoids adding the old index as a column
df = df.reset_index(drop = False)

# %%
def recommender(data_frame, exercise_id, sim_matrix):
    sim_df = pd.DataFrame(sim_matrix[exercise_id],
                         columns=["similarity"])
    exercise_titles = data_frame.loc[:, "Title"]
    exercise_rec = pd.concat([sim_df, exercise_titles], axis = 1)

    exercise_rec = exercise_rec.sort_values(by=["similarity"], ascending = False)

    return exercise_rec.iloc[1:20,:]

# %%
# Prints the row of the given Title to find the index
print(df[df["Title"] == "Bench press"])

# %%
# Exercises similar to bench press
recommender(df, 454, sim_matrix)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# %%
tfidf = TfidfVectorizer(stop_words="english")
overview_matrix = tfidf.fit_transform(df["Desc"])
overview_matrix.shape

# %%
similarity_matrix = linear_kernel(overview_matrix, overview_matrix)
print(similarity_matrix[0:5,0:5])

# %%
mapping = pd.Series(df.index, index = df["Desc"])
mapping

# %%
def recommender_by_desc(exercise_input):
    exercise_index = mapping[exercise_input]
    similarity_score = list(enumerate(similarity_matrix[exercise_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[0:10]

    exercise_indices = [i[0] for i in similarity_score]
    return df["Title"].iloc[exercise_indices]

# %%
recommender_by_desc(df["Desc"][0])
