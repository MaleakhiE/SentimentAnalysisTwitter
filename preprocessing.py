import pandas as pd
import nltk
import matplotlib.pyplot as plt
import warnings as wrn

train_set = pd.read_csv('Corona_NLP_train.csv',encoding="latin1")

print("DATASET AWAL : ","\n",train_set,"\n")

unrelevant_features = ["UserName","ScreenName","Location","TweetAt"]

train_set.drop(unrelevant_features,inplace=True,axis=1)

neutrals = train_set[train_set["Sentiment"] == "Neutral"]
negatives = train_set[(train_set["Sentiment"] == "Negative") | (train_set["Sentiment"] == "Extremely Negative")]
positives = train_set[(train_set["Sentiment"] == "Positive") | (train_set["Sentiment"] == "Extremely Positive")]


wrn.filterwarnings('ignore')
negatives["Sentiment"] = 0
positives["Sentiment"] = 2
neutrals["Sentiment"] = 1

data = pd.concat([positives,
                  neutrals,
                  negatives,
                 ],axis=0)
data.reset_index(inplace=True)
print(data.info(),"\n")
print(data.head(),"\n")

data.to_pickle("data_train.pkl")
print(data)