import pandas as pd

filename = "user_ratings.csv"
data = pd.read_csv(filename)

def userbased_cf(user_index, data, display: int):

    data = data.drop(columns="Users")
    data.index.name = "Users"
    data.columns.name = "Exercises"
    ratings_norm = data.copy()
    ratings_norm = ratings_norm.subtract(data.mean(axis=1), axis = "rows")
    
    user_similarity = ratings_norm.T.corr(min_periods=3)
    picked_user = user_index

    user_similarity.drop(index=picked_user, inplace=True)
    
    n = 10

    user_similarity_threshold = 0.3

    similar_users = user_similarity[user_similarity[picked_user]>user_similarity_threshold][picked_user].sort_values(ascending=False)[:n]
    picked_user_done = ratings_norm[ratings_norm.index == picked_user].dropna(axis=1, how="all")
    similar_user_exercises = ratings_norm[ratings_norm.index.isin(similar_users.index)].dropna(axis=1, how="all")
    similar_user_exercises.drop(picked_user_done.columns, axis=1, inplace=True, errors='ignore')

    item_score = {}

    for i in similar_user_exercises.columns:
        movie_rating = similar_user_exercises[i]
        total = 0
        count = 0
        for u in similar_users.index:
            if pd.isna(movie_rating[u]) == False:
                score = similar_users[u] * movie_rating[u]
                total += score
                count +=1
        item_score[i] = total / count

    item_score = pd.DataFrame(item_score.items(), columns=['exercise', 'exercise_score'])
    
    ranked_item_score = item_score.sort_values(by="exercise_score", ascending=False)

    return(ranked_item_score.head(display))

with open('entries.txt', 'r') as file:
  entry = file.readline()
  if len(data.index) >= int(entry):
      index = int(entry)
  else:
      print(f"Index {entry} is not in dataframe")
      df_empty = pd.Dataframe()
      df_empty.to_csv("usr_recommended.csv", index=False)
      quit()

df_usr = userbased_cf(index, data, 10)

df_usr.to_csv("usr_recommended.csv")
print("Done")
