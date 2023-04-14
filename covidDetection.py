import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from os import listdir

vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(150,150,3)))


data_list_model2c = listdir('/home/narayana/wd/masterClass/covidDetection/train')
data_list_model2c

DATASET_PATH_model2c  = '/home/narayana/wd/masterClass/covidDetection/train'
train_dir_model2c =  '/home/narayana/wd/masterClass/covidDetection/train'
test_dir_model2c =  '/home/narayana/wd/masterClass/covidDetection/test'
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = len(data_list_model2c)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 100
LEARNING_RATE = 0.0001

#Train datagen here is a preprocessor
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=50,
                                   featurewise_center = True,
                                   featurewise_std_normalization = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.25,
                                   zoom_range=0.1,
                                   zca_whitening = True,
                                   channel_shift_range = 20,
                                   horizontal_flip = True ,
                                   vertical_flip = True ,
                                   fill_mode='constant' )

train_batches = train_datagen.flow_from_directory(DATASET_PATH_model2c,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical",)

valid_batches = train_datagen.flow_from_directory(DATASET_PATH_model2c,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "validation",
                                                  seed=42,
                                                  class_mode="categorical",)

vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(150,150,3)))
for layer in vgg16_model.layers[:-4]:
    layer.trainable = False
model2c = tf.keras.Sequential()
model2c.add(vgg16_model)
model2c.add(Flatten())
model2c.add(Dense(64, activation='relu'))
model2c.add(layers.Dense(2, activation='softmax'))

model2c.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=['accuracy'])

#Step Size
print("TrainingBatch: ", len(train_batches))
print("ValidationBatches: ",len(valid_batches))

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

print("STEP_SIZE_TRAIN: ", STEP_SIZE_TRAIN)
print("STEP_SIZE_VALID: ", STEP_SIZE_VALID)

model2c.fit_generator(train_batches, steps_per_epoch =STEP_SIZE_TRAIN, epochs= 5, verbose=True)
###########################################

Train_DataGeneration = ImageDataGenerator(rescale=1. / 255)
TrainBatch = Train_DataGeneration.flow_from_directory(train_dir_model2c, target_size=IMAGE_SIZE, batch_size=1, shuffle=False, seed=42, class_mode="categorical")

TrainBatch.reset()
trainResult = model2c.evaluate_generator(TrainBatch, steps = len(TrainBatch), use_multiprocessing = False, verbose = 1, workers=1)
print('Train loss:' , trainResult[0], 'Train accuracy: ', trainResult[1])

prediction = model2c.predict_generator(TrainBatch, steps = len(TrainBatch))
Predicted_class = np.argmax(prediction, axis=1)

print(confusion_matrix(TrainBatch.classes, Predicted_class))

## Test accuracy
Test_DataGeneration = ImageDataGenerator(rescale=1. / 255)
TestingBatch = Test_DataGeneration.flow_from_directory(test_dir_model2c, target_size=IMAGE_SIZE, batch_size=1, shuffle=False, seed=42, class_mode="categorical")

TestingBatch.reset()
testResult = model2c.evaluate_generator(TestingBatch, steps = len(TestingBatch), use_multiprocessing = False, verbose = 1, workers=1)
print('Test loss:' , testResult[0], 'Test accuracy: ', testResult[1])

prediction = model2c.predict_generator(TestingBatch, steps = len(TestingBatch))
Predicted_class = np.argmax(prediction, axis=1)

print(confusion_matrix(TestingBatch.classes, Predicted_class))