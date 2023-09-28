
import pandas as pd
import random

df = pd.read_csv("megaGymDataSet.csv")

df = df.drop('Rating', axis=1)
df = df.drop('RatingDesc', axis=1)

df = df[df['Desc'].notna()]
df.pop(df.columns[0])

all_categories = df["Type"].unique()
dict_all_exercises = {}

for cat in all_categories:
    cat_df = df[df['Type'] == cat]
    
    titles = cat_df['Title'].tolist()
    if not None:
        dict_all_exercises[cat] = titles

total_users = 1000

def gen_strength_users(total_users, all_categories):
    all_users = []

    for user_id in range(1, total_users + 1):
        user_exercises = gen_ratings(all_categories, user_id, dict_all_exercises)
        all_users.extend(user_exercises)

    return all_users

def gen_ratings(all_categories, user_id,  dict_all_exercises):
    user_exercises = []
    rated_exercises = set()

    for exercise_type in all_categories:
        available_exercises = list(set(dict_all_exercises[exercise_type]) - rated_exercises)
        if not available_exercises:
            continue
        exercise_title = random.choice(available_exercises)
        exercise_rating = round(random.uniform(1.0, 5.0), 1)
        user_exercises.append([user_id, exercise_title, exercise_rating])
        rated_exercises.add(exercise_title)

    return user_exercises

strength_users_data = gen_strength_users(total_users, all_categories)

df = pd.DataFrame(strength_users_data, columns=['userId', 'title', 'rating'])

df.to_csv("user_exercises.csv", index=False)
