import numpy as np
from keras import backend as kb
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.utils import np_utils

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()
 
    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

def build_model():
    #ネットワークの構造を定義
    model = Sequential()
    model.add(Dense(128, input_shape=(784,)))#input_shapeは変えないで 
    #model.add(Activation('sigmoid'))    こんな感じで層をふやせる
    #model.add(Dense(10)) 
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    batch_size = 64
    epoch = 5

    # MNISTデータのロード
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # 画像を1次元配列化
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # 画素を0.0-1.0の範囲に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # one-hot-vectorに変換
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # 多層ニューラルネットワークモデルを構築
    model = build_model()

    # モデルのサマリを表示
    model.summary()

    # モデルをコンパイル
    sgd = optimizers.SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


    # モデルの訓練
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=1,
                        validation_split=0.1)

    # モデルの評価
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test loss:', loss)
    print('Test acc:', acc)  
    
    plot_history(history)      
    
    kb.clear_session()
