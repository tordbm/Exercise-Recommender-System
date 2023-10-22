import pandas as pd
import random as rand

df = pd.read_csv("100_exercises.csv", sep=";", header=None)
df.columns = ['Index', 'Title']

exercises = set()
while len(exercises) < 20:
    random = rand.randint(0, 119)
    exercises.add(df.loc[random]['Title'] + " :")
    
for i in exercises:
    print(i)
    