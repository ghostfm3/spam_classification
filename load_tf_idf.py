from doc_2_vec import MakeInputData
import numpy as np
import pandas as pd
import csv

pd.options.display.precision = 3

df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={"v1":"label", "v2":"text"}, inplace=True)
df_list = df['text'].tolist()
df = df.replace({"label":{"ham":0,"spam":1}})
df_label_list = df['label'].tolist()

const = MakeInputData(df_list)
vec, list_vec = const.sentence_bert()


print(vec)
print(list_vec)

with open("./st_spam_text_embeddings.csv", mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(list_vec)
    for embedding, label in zip(vec,df_label_list):
        vc = ','.join([str(i) for i in embedding])
        f.write(f"{vc},{label}\n")