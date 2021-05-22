import tensorflow as tf
from tensorflow import keras
# from resnext import ResNext
# from resnext import ResNextImageNet

print("TensorFlow version is ", tf.__version__)

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()

%matplotlib inline



class_names = ['bulbasaur', 'charmander', 'eevee', 'gengar', 'gyarados',
               'meowth', 'mewtwo', 'pikachu', 'raichu', 'squirtle']
               
             
             
train_dir = './pokemon_data/train'

validation_dir = './pokemon_data/val'

test_dir = './pokemon_data/test'



train_bulbasaur_dir = os.path.join(train_dir, 'bulbasaur') 
train_charmander_dir = os.path.join(train_dir, 'charmander') 
train_eevee_dir = os.path.join(train_dir, 'eevee')  
train_gengar_dir = os.path.join(train_dir, 'gengar')  
train_gyarados_dir = os.path.join(train_dir, 'gyarados')  
train_meowth_dir = os.path.join(train_dir, 'meowth')  
train_mewtwo_dir = os.path.join(train_dir, 'mewtwo')  
train_pikachu_dir = os.path.join(train_dir, 'pikachu')  
train_raichu_dir = os.path.join(train_dir, 'raichu')  
train_squirtle_dir = os.path.join(train_dir, 'squirtle') 

validation_bulbasaur_dir = os.path.join(validation_dir, 'bulbasaur') 
validation_charmander_dir = os.path.join(validation_dir, 'charmander')  
validation_eevee_dir = os.path.join(validation_dir, 'eevee')  
validation_gengar_dir = os.path.join(validation_dir, 'gengar')  
validation_gyarados_dir = os.path.join(validation_dir, 'gyarados')  
validation_meowth_dir = os.path.join(validation_dir, 'meowth')  
validation_mewtwo_dir = os.path.join(validation_dir, 'mewtwo')  
validation_pikachu_dir = os.path.join(validation_dir, 'pikachu')  
validation_raichu_dir = os.path.join(validation_dir, 'raichu')  
validation_squirtle_dir = os.path.join(validation_dir, 'squirtle')  

test_bulbasaur_dir = os.path.join(test_dir, 'bulbasaur') 
test_charmander_dir = os.path.join(test_dir, 'charmander') 
test_eevee_dir = os.path.join(test_dir, 'eevee')  
test_gengar_dir = os.path.join(test_dir, 'gengar')  
test_gyarados_dir = os.path.join(test_dir, 'gyarados')  
test_meowth_dir = os.path.join(test_dir, 'meowth')  
test_mewtwo_dir = os.path.join(test_dir, 'mewtwo')  
test_pikachu_dir = os.path.join(test_dir, 'pikachu')  
test_raichu_dir = os.path.join(test_dir, 'raichu')  
test_squirtle_dir = os.path.join(test_dir, 'squirtle') 



num_bulbasaur_tr = len(os.listdir(train_bulbasaur_dir))
num_charmander_tr = len(os.listdir(train_charmander_dir))
num_eevee_tr = len(os.listdir(train_eevee_dir))
num_gengar_tr = len(os.listdir(train_gengar_dir))
num_gyarados_tr = len(os.listdir(train_gyarados_dir))
num_meowth_tr = len(os.listdir(train_meowth_dir))
num_mewtwo_tr = len(os.listdir(train_mewtwo_dir))
num_pikachu_tr = len(os.listdir(train_pikachu_dir))
num_raichu_tr = len(os.listdir(train_raichu_dir))
num_squirtle_tr = len(os.listdir(train_squirtle_dir))

num_bulbasaur_val = len(os.listdir(validation_bulbasaur_dir))
num_charmander_val = len(os.listdir(validation_charmander_dir))
num_eevee_val = len(os.listdir(validation_eevee_dir))
num_gengar_val = len(os.listdir(validation_gengar_dir))
num_gyarados_val = len(os.listdir(validation_gyarados_dir))
num_meowth_val = len(os.listdir(validation_meowth_dir))
num_mewtwo_val = len(os.listdir(validation_mewtwo_dir))
num_pikachu_val = len(os.listdir(validation_pikachu_dir))
num_raichu_val = len(os.listdir(validation_raichu_dir))
num_squirtle_val = len(os.listdir(validation_squirtle_dir))

num_bulbasaur_tst = len(os.listdir(test_bulbasaur_dir))
num_charmander_tst = len(os.listdir(test_charmander_dir))
num_eevee_tst = len(os.listdir(test_eevee_dir))
num_gengar_tst = len(os.listdir(test_gengar_dir))
num_gyarados_tst = len(os.listdir(test_gyarados_dir))
num_meowth_tst = len(os.listdir(test_meowth_dir))
num_mewtwo_tst = len(os.listdir(test_mewtwo_dir))
num_pikachu_tst = len(os.listdir(test_pikachu_dir))
num_raichu_tst = len(os.listdir(test_raichu_dir))
num_squirtle_tst = len(os.listdir(test_squirtle_dir))



print('total training bulbasaur images:', num_bulbasaur_tr)
print('total training charmander images:', num_charmander_tr)
print('total training eevee images:', num_eevee_tr)
print('total training gengar images:', num_gengar_tr)
print('total training gyarados images:', num_gyarados_tr)
print('total training meowth images:', num_meowth_tr)
print('total training mewtwo images:', num_mewtwo_tr)
print('total training pikachu images:', num_pikachu_tr)
print('total training raichu images:', num_raichu_tr)
print('total training squirtle images:', num_squirtle_tr)

print()

print('total validation bulbasaur images:', num_bulbasaur_val)
print('total validation charmander images:', num_charmander_val)
print('total validation eevee images:', num_eevee_val)
print('total validation gengar images:', num_gengar_val)
print('total validation gyarados images:', num_gyarados_val)
print('total validation meowth images:', num_meowth_val)
print('total validation mewtwo images:', num_mewtwo_val)
print('total validation pikachu images:', num_pikachu_val)
print('total validation raichu images:', num_raichu_val)
print('total validation squirtle images:', num_squirtle_val)

print()

print('total testing bulbasaur images:', num_bulbasaur_tst)
print('total testing charmander images:', num_charmander_tst)
print('total testing eevee images:', num_eevee_tst)
print('total testing gengar images:', num_gengar_tst)
print('total testing gyarados images:', num_gyarados_tst)
print('total testing meowth images:', num_meowth_tst)
print('total testing mewtwo images:', num_mewtwo_tst)
print('total testing pikachu images:', num_pikachu_tst)
print('total testing raichu images:', num_raichu_tst)
print('total testing squirtle images:', num_squirtle_tst)



train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



image_size = 160

# batch_size = (8, 16, 32)
batch_size = 16

train_generator = train_datagen.flow_from_directory(train_dir, 
                target_size = (image_size, image_size), batch_size = batch_size, class_mode = 'sparse') 

validation_generator = validation_datagen.flow_from_directory(validation_dir, 
                target_size = (image_size, image_size), batch_size = batch_size, class_mode = 'sparse')

test_generator = test_datagen.flow_from_directory(test_dir, 
                target_size = (image_size, image_size), batch_size = batch_size, class_mode = 'sparse') 
                


sample_training_images, _ = next(train_generator)



def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
plotImages(sample_training_images[:5])



IMG_SHAPE = (image_size, image_size, 3)

# Pre-Trained Model 선택

# MobileNetV2
# base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet') 

# ResNet50V2
base_model = tf.keras.applications.ResNet50V2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet') 

# InceptionV3
# base_model = tf.keras.applications.InceptionV3(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

# InceptionResNetV2
# base_model = tf.keras.applications.InceptionResNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

# VGG19
# base_model = tf.keras.applications.VGG19(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet') 

# ResNeXt101 -> 현재 버전에서는 사용 불가
'''
base_model = tf.keras.applications.ResNeXt101(
             input_tensor = pinp, input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet', 
             backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
'''
'''
base_model = ResNext(input_tensor = pinp, input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet', 
             backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
'''
# base_model = ResNextImageNet(image_shape)

# EfficientNetB7 -> 유의미한 학습 효과 없음
# base_model = tf.keras.applications.EfficientNetB7(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet') 



base_model.trainable = False



base_model.summary()



model = tf.keras.Sequential([base_model, keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(10, activation='softmax')]) 



# lr = (0.0001, 0.0005, 0.001)

# Optimizer 선택

# RMSprop
# model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Adam
# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Nesterov Momentum
'''
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''

# Adamax (기본값 사용 권장)
'''
model.compile(optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''

# Nadam (기본값 사용 권장)
# '''
model.compile(optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# '''



model.summary()



early_stopping = keras.callbacks.EarlyStopping(patience = 5)



# epochs = (10, 15, 20)
epochs = 50

steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps, 
                              callbacks=[early_stopping])
                              
                              
                              
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()



test_loss, test_acc = model.evaluate_generator(test_generator)

print('test accuracy :', test_acc)
print('test loss :', test_loss)



train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True, 
                    zoom_range=0.5
                    )
                    
                    
                    
train_generator = train_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(image_size, image_size),
                                                     class_mode='sparse')
                                                     
                                                     
                                                     
augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)



validation_datagen = ImageDataGenerator(rescale=1./255)



validation_generator = validation_datagen.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(image_size, image_size),
                                                 class_mode='sparse')
                                                 
                                                 
                                                 
model = tf.keras.Sequential([base_model, keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(10, activation='softmax')]) 



# lr = (0.0001, 0.0005, 0.001)

# Optimizer 선택

# RMSprop
# model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Adam
# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Nesterov Momentum
'''
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''

# Adamax (기본값 사용 권장)
'''
model.compile(optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''

# Nadam (기본값 사용 권장)
# '''
model.compile(optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# '''



model.summary()



# epochs = (10, 15, 20)
epochs = 50

steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps, 
                              callbacks=[early_stopping])
                              
                              
                              
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()



test_loss, test_acc = model.evaluate_generator(test_generator)

print('test accuracy :', test_acc)
print('test loss :', test_loss)



predictions = model.predict_generator(test_generator)



predictions[0]



np.argmax(predictions[0])



img, _ = next(test_generator)

print(img.shape)



plotImages(img[:5])



predictions_single = model.predict(img)

print(predictions_single)



prediction_result = np.argmax(predictions_single[4])

print(class_names[prediction_result])