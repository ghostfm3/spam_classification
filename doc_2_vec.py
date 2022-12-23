from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pandas as pd

class MakeInputData:

    def __init__(self,data,vector_size=0, window=0, min_count=0, workers=0, epoches=0):
        self.data = data
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epoches = epoches
        self.sb_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def make_vec_and_header(self, num_count = 0):
        train_data = [TaggedDocument(words = data.split(),tags = [i]) for i, data in enumerate(self.data)]
        model = Doc2Vec(self.train_data, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epoches)

        vector_list = []
        header_list = []
        
        for i in range(len(train_data)):   
            vec = model.infer_vector(train_data[i].words) # 数値からなるベクトルに変換
            vector_list.append(vec.tolist())
        
        while True:
            text = f"vec{num_count+1}"
            header_list.append(text)
            num_count += 1

            if num_count == self.vector_size:
                text1 = "label"
                header_list.append(text1)
                break
        
        return vector_list, header_list
    
    def tf_idf_vector(self, norm_in='l2'):
        tfidf_model = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b", norm=norm_in)
        tfidf_vec = tfidf_model.fit_transform(self.data)
        return tfidf_vec
    
    def count_vectorizer(self, dim_num=10, min_count=3, num_count=0):
        header_list = []
        v_count = CountVectorizer(min_df=min_count)
        text_vec = v_count.fit_transform(self.data)
        # pca = PCA(n_components=dim_num)
        # pca.fit(df.values)
        # pca_cor = pca.transform(df.values)
        # print(pca.explained_variance_ratio_)
        
        while True:
            text = f"vec{num_count+1}"
            header_list.append(text)
            num_count += 1

            if num_count == len(text_vec.toarray()[0]):
                text1 = "label"
                header_list.append(text1)
                break


        return text_vec.toarray(), header_list
    
    def sentence_bert(self, num_count=0):
        embeddings = self.sb_model.encode(self.data)
        header_list = []
        while True:
                text = f"vec{num_count+1}"
                header_list.append(text)
                num_count += 1

                if num_count == len(embeddings[0]):
                    text1 = "label"
                    header_list.append(text1)
                    break

        return embeddings, header_list