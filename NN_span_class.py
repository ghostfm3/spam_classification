from sklearn import datasets
from sklearn.model_selection import train_test_split 
import tensorflow as tf 
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def NN_study(X_train,Y_train,X_test,Y_test,input_dim,middle_layer,output_dim):
      model = Sequential()
      model.add(Dense(middle_layer, Activation('sigmoid'), input_shape=(input_dim,)))
      model.add(Dense(middle_layer, Activation('sigmoid')))
      model.add(Dense(output_dim))
      model.add(Activation('softmax'))
      model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #adam

      history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=350, batch_size=1000, verbose=1)
      score = model.evaluate(X_test,Y_test,verbose=0,batch_size=1)
      model.save('spam_dim384_unit5_model.h5')
      
      return history, score

def plot_loss_accuracy(result):
    metrics = ['loss', 'accuracy']  

    plt.figure(figsize=(10, 5))  

    for i in range(len(metrics)):

        metric = metrics[i]

        plt.subplot(1, 2, i+1)  
        plt.title(metric) 
        
        plt_train = result.history[metric] 
        plt_test = result.history['val_' + metric] 
        
        plt.plot(plt_train, label='training')  
        plt.plot(plt_test, label='test') 
        plt.legend()  
        
    plt.savefig('./accuracy_loss.png') 

def main():
    df = pd.read_csv('st_spam_text_embeddings.csv', header=0)
    print(df)
    X = df.drop('label', axis=1)
    Y = np_utils.to_categorical(df['label'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.6,test_size=0.4)


    history_in, score_in = NN_study(x_train,y_train,x_test,y_test,384,5,2)
    print(f"正解率:{score_in[1]}")
    plot_loss_accuracy(history_in)

if __name__ == "__main__":
    main()