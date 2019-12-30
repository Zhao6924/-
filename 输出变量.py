from keras.models import Sequential,Model
from keras.layers import Dense
import numpy as np
model = Sequential()
model.add(Dense(32,activation="relu",input_dim=100))
model.add(Dense(16,activation="relu",name="Dense_1"))
model.add(Dense(1, activation='sigmoid',name="Dense_2"))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#假设训练和测试使用同一组数据
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
model.fit(data,labels,epochs=10,batch_size=32)
#取某一层的输出为输出新建为model，采用函数模型
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
dense1_output = dense1_layer_model.predict(data)
print (dense1_output.shape)

#获得某一层的权重和偏置
weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
print(weight_Dense_1.shape)
print(bias_Dense_1.shape)