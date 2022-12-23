from doc_2_vec import MakeInputData
import pandas as pd
import logging
import csv



def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.rename(columns={"v1":"label", "v2":"text"}, inplace=True)
    df_list = df['text'].tolist()
    df = df.replace({"label":{"ham":0,"spam":1}})
    df_label_list = df['label'].tolist()
    
    const = MakeInputData(df_list, 10, 2, 0, 4, 50)
    vec, header = const.make_vec_and_header()
    
    with open("./spam_text_embeddings.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for embedding, label in zip(vec,df_label_list):
            vc = ','.join([str(i) for i in embedding])
            f.write(f"{vc},{label}\n")

if __name__ == "__main__":
    main()