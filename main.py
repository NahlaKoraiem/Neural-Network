# -*- coding: utf-8 -*-

"""**Import the Libraries**"""


import keras
import tensorflow
import numpy as np
from keras.layers import Dense
from keras.layers import Conv2D
from keras.datasets import mnist
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn import preprocessing
from keras.layers import MaxPool2D
from keras.models import Sequential
from sklearn.svm import SVC,LinearSVC
from skimage.util.shape import view_as_blocks
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report

"""**Load the Dataset**"""

(train_x, train_y), (test_x, test_y) = mnist.load_data()

"""**Transform 3D Data to 2D**"""

nsamples, nx, ny = train_x.shape
tsamples, tx, ty = test_x.shape

d2_train = train_x.reshape((nsamples,nx*ny))
d2_test = test_x.reshape((tsamples,tx*ty))

print('Before: ', train_x.shape ,'\nAfter: ', d2_train.shape)
#First dimension is maintained and the other two dimensions are flattened (so 28x28 becomes 784).

"""# **MLP**

**Calculate the Centroid**
"""

def centroid(block):
    x = 0
    y = 0
    xy = 0
    for i in range(7):
        for j in range(7):
            x += i * block[i][j]
            y += j * block[i][j]
            xy += block[i][j]
    x = x / xy if xy > 0 else 0
    y = y / xy if xy > 0 else 0
    return x, y

"""**Extract the Feature Vector**"""

def extractfv(x):
    x_blocks = view_as_blocks(x[i], block_shape=(7, 7))
    fv_image = []
    for j in range(4):
        for k in range(4):
          x,y=centroid(x_blocks[j][k])
          fv_image.append(x)
          fv_image.append(y)
    fv_image = np.array(fv_image)
    return fv_image

"""**Calculate the Feature Vector for the Training and Testing Sets**"""

train_fv = []
for i in range(len(train_x)):
    train_fv.append(extractfv(train_x))
train_fv=np.array(train_fv)

test_fv = []
for i in range(len(test_x)):
    test_fv.append(extractfv(test_x))
test_fv=np.array(test_fv)

"""**Normalize the Features (Standardize features by removing the mean and scaling to unit variance) and Transform (Perform standardization by centering and scaling)**"""

normTrain = preprocessing.StandardScaler().fit(train_fv)
train_fv = normTrain.transform(train_fv)

normTest = preprocessing.StandardScaler().fit(test_fv)
test_fv = normTest.transform(test_fv)

"""**Implement the MLP Neural Network**"""

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
MLP_train=clf.fit(train_fv,train_y)
MLP_predict = clf.predict(test_fv)

"""**Classification Report (Classes Indicating Digits 0-9)**"""

print(classification_report(test_y, MLP_predict))

"""# **SVM**

**Create and Fit our LinearSVC Object**
"""

clf = LinearSVC(dual=False)
clf.fit(train_fv,train_y)

"""**Use our LinearSVC Object to Predict the Testing Samples**"""

pred = clf.predict(test_fv)

"""**Classification Report (Classes Indicating Digits 0-9)**"""

print(classification_report(test_y, pred))

"""# **KNN**

**Take 10% of the Training Data and Use That for Validation**
"""

(trainData, valData, trainLabels, valLabels) = train_test_split(d2_train, train_y,test_size=0.1, random_state=84)

"""**Initialize the Values of K (Odd Numbers Between 1 to 30) for KNN, and Loop Over the Values of K to Find the Best Accuracy**"""

accuracies = []

for k in range(1, 30, 2):
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(trainData, trainLabels)

  #Evaluate the Model and Update the Accuracies
  score = model.score(valData, valLabels)
  print("k= " , k , "accuracy= ", (score * 100))
  accuracies.append(score)

"""**Retrain Our Classifier Using the Value of K with the Highest Accuracy**"""

kVal = range(1, 30, 2)
k = np.argmax(accuracies)

model = KNeighborsClassifier(n_neighbors=kVal[k])
model.fit(trainData, trainLabels)
pred = model.predict(d2_test)

"""**Classification Report (Classes Indicating Digits 0-9)**"""

print(classification_report(test_y, pred))

"""# **CNN**

**Reshape the Inputs and Add 1 to Indicate the Images are Grayscale**
"""

X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
X_test = test_x.reshape((test_x.shape[0],test_x.shape[1],test_x.shape[2],1))

#Check the Shape
print('Before Reshaping: ', train_x.shape, test_x.shape)
print('After Reshaping: ' , X_train.shape, X_test.shape)

"""**Normalize the Pixel Values**"""

X_train=X_train/255
X_test=X_test/255

"""**Create a Sequential Model (Layer by Layer), and Add Model Layers**"""

model=Sequential()

#Convolution Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

#Pooling Layer
model.add(MaxPool2D(2,2))

#Fully Connected Layer
model.add(Flatten())
model.add(Dense(100,activation='relu'))

#Output Layer
model.add(Dense(10,activation='softmax'))

"""**Compile the Model**

"""

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

"""**Train the Model, then Use It to Make Predictions**"""

model.fit(X_train,train_y,epochs=10)
pred=model.predict(X_test)

"""**Evaluate the Model, and Show the Loss and Accuracy Percentages**"""

score = model.evaluate(X_test,test_y,verbose=1)

print('Loss = ', score[0], ', Accuracy = ', score[1])

"""**Classification Report (Classes Indicating Digits 0-9)**"""

#Convert labels to single-digits
pred_y = np.argmax(pred, axis=1)
print(classification_report(test_y, pred_y))