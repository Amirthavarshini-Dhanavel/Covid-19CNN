import cv2,os
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Activation, MaxPooling2D
from keras.utils import normalize
from keras.layers import Concatenate
from keras import Input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

#preprocess
paths='/content/drive/MyDrive/dataset' 
divisions=os.listdir(paths)

labels=[i for i in range(len(divisions))]

label_dict=dict(zip(divisions,labels)) 

print(label_dict)
print(divisions)
print(labels)
size=100 
data=[]
target=[]

for category in divisions:
    paths=os.path.join(paths,category)
    images=os.listdir(paths)
        
    for img_name in images:
        img_path=os.path.join(paths,img_name)
        image=cv2.imread(img_path)

        try:
            grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)           
            resize=cv2.resize(grayscale,(size,size))
            data.append(resize)
            target.append(label_dict[category])
           
        except Exception as e:
           
            print('Exception:',e)


data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],size,size,1))
target=np.array(target)



new_target=np_utils.to_categorical(target)

np.save('data',data)
np.save('target',new_target)

#training

nput_shape=data.shape[1:] #50,50,1
inp=Input(shape=input_shape)
convs=[]

parrallel_kernels=[3,5,7]

for k in range(len(parrallel_kernels)):

    conv = Conv2D(128, parrallel_kernels[k],padding ='same',activation='relu',input_shape=input_shape,strides=1)(inp)
    convs.append(conv)

out = Concatenate()(convs)
conv_model = Model(inputs=inp, outputs=out)

model = Sequential()
model.add(conv_model)

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,input_dim=128,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.1)