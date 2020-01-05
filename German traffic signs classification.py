import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
import warnings
import cv2
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras import backend as K
K.set_image_dim_ordering('th')

datadir = "/home/cyrine/Bureau/DL/ENIT/Train"
categories = [str(i) for i in range(43)]
ligne = []
y = []
for category in categories:
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        ligne.append(path + '/' + img)
        y.append(int(category))

df = pandas.DataFrame(ligne, columns=["aa"])
df2 = pandas.DataFrame(y, columns=["label"])
XX = pandas.concat([df, df2], axis=1)
XX.to_csv('results.csv')

ROWS = 64
COLS = 64
CHANNELS = 3


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 5 == 0: print('Processed {} of {}'.format(i, count))

    return data


X = XX['aa']
Y = XX['label']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)

train = prep_data(X_train)
val = prep_data(X_val)

# Understanding our Data
print(" Nbr of Training samples:", len(train))
print("Nbr of Validation samples:", len(val))
print("Number of classes:", len(np.unique(Y_train)))
n_classes = len(np.unique(Y_train))

# Checking the distribution of samples among classes
unique_elements, counts_elements = np.unique(Y_train, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

plt.bar(np.arange(43), counts_elements, align='center', color='blue')
plt.xlabel('Class')
plt.ylabel('Nbr of Training data')
plt.xlim([-1, 43])
plt.show()
print("We can see that certain classes are under represented, we have an unbalanced dataset")

Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)

print(Y_train)
print(len(Y_train))

print(train[1].shape)

# building a model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(3, 64, 64)))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, Y_train, batch_size=128, epochs=80, verbose=1)

##########################################

path = "/home/cyrine/Bureau/DL/ENIT/Test/"
a = range(300)
l1 = []
l2 = []

for i in a:
    s = str(i)
    while len(s) < 5:
        s = '0' + s
    l1.append(path + s + '.png')
    l2.append('Test/' + s + '.png')

dft = pandas.DataFrame(l1, columns=["path"])
test = prep_data(dft['path'])
prediction = model.predict_classes(test, batch_size=16, verbose=1)
dfr = pandas.DataFrame(l2, columns=["Path"])
dfr1 = pandas.DataFrame(prediction, columns=["ClassID"])
Final = pandas.concat([dfr, dfr1], axis=1)
Final.to_csv('results.csv', header=True, index=False)


