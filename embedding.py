import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
model = Sequential()
model.add(Embedding(4, 20,mask_zero=True,input_length=5))
input_array = np.array([[1,1,1,1,1],
                 [1,1,2,1,1],
                 [1,1,1,1,2],
                 [1,2,1,1,1],
                 [1,1,1,1,1],
                 [1,3,2,1,1],
                 [1,2,3,1,1],
                 [1,1,3,1,1],
                 [1,2,3,2,1],
                 [1,3,3,2,1],
                 [2,0,0,0,0],
                 [2,0,3,0,0],
                 [2,0,0,2,3],
                 [2,0,0,0,3],
                 [2,0,3,0,2],
                 [2,3,3,2,3],
                 [2,3,2,2,1],
                 [2,0,3,2,1],
                 [2,2,2,3,0],
                 [2,3,2,3,0]])
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(input_array)
print(output_array)
print(input_array[0:2])
print(output_array[0:2])
