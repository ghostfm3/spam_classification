from tensorflow.python.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from doc_2_vec import MakeInputData


def main():
 
    input_list = ["Free service! Please contact me immediately. But it is 300 US dollars next month."]

    model = load_model('spam_dim4_unit32_model.h5')
    label_list = ['ham','spam']
    print(input_list)
    const = MakeInputData(input_list)
    vec, list_vec = const.sentence_bert()
  

    # モデル読み込み・予測
    X_pred = vec
    y_pred = model.predict(X_pred)
    print(f"入力データのラベル予測：{label_list[y_pred.argmax()]}")

if __name__ == "__main__":
    main()


