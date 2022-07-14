import numpy as np
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_cifar10

(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

min_ = (min_-mean)/(std+1e-7)
max_ = (max_-mean)/(std+1e-7)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(x_train, y_train, num_classes=10, batch_size=64, epochs=25):
    model = Sequential([
        tf.keras.applications.ResNet50(input_shape=x_train.shape[1:], include_top=False, weights=None),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])

    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
        )
    datagen.fit(x_train)
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1)
    return model

model_path = "/home/taesung/projects/ai-robustness/models/cifar10-resnet50.h5"
if not os.path.exists(model_path):
    model = create_model(x_train, y_train, epochs=25)
    model.save(model_path)
else:
    model = tf.keras.models.load_model(model_path)

model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

print("Model and data preparation done.")

#%%

# model_path = "/home/taesung/projects/ai-robustness/models/cifar10-basic.h5"
# model = tf.keras.models.load_model(model_path)
# model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

#%% 

# x_train = x_train
# y_train = y_train
# epochs = 5
# num_classes = 10
# batch_size = 32

# model = Sequential([
#     tf.keras.applications.ResNet50(input_shape=x_train.shape[1:], include_top=False, weights=None),
#     Flatten(),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=False
#     )
# datagen.fit(x_train)
# model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1)


from tensorflow.keras.utils import to_categorical

# A trigger from class 0 will be classified into class 1.
class_source = 0
class_target = 1
index_target = np.where(y_test.argmax(axis=1)==class_source)[0][5]
y_poisoned = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.

# Trigger sample
x_trigger = x_test[index_target:index_target+1]
y_trigger  = to_categorical([class_target], num_classes=10)


#%%
print("Poisoning attack start...")
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack

# # Samples to poison
# percent_poison =  0.01
# P = int(len(x_train) * percent_poison)
# indices_poison = np.random.permutation(np.where(y_poisoned==class_target)[0])[:P]
# x_poison = x_train[indices_poison]
# y_poison = y_train[indices_poison]

attack = GradientMatchingAttack(model_art,
        percent_poison=0.01,
        max_trials=10,
        max_epochs=400,
        clip_values=(min_,max_),
        epsilon=0.3,
        verbose=False)


x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)

# x_train_poisoned = x_train.copy()
# x_raw_poisoned[indices_target] = x_poison

print("Poisoning data done.")

model_poisoned = create_model(x_poison, y_poison, epochs=25)
y_ = model_poisoned.predict(x_trigger)

print(y_trigger)
print(y_)


#%% [Witches' Brew paper experiment replication]
print("Witches' Brew paper experiment replication")

# 10 random poison-target pairs.

index_x_trigger = np.random.randint(0, len(x_test))
x_trigger = x_test[index_x_trigger:index_x_trigger+1]
class_source = y_test.argmax(axis=-1)[index_x_trigger]

class_target = list(range(10))
class_target.remove(class_source)
class_target = np.random.choice(class_target)
# index_x_trigger = np.where(y_test.argmax(axis=1)==class_source)[0][5]
# y_poisoned = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.
# y_poison = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.

# Trigger sample
y_trigger  = to_categorical([class_target], num_classes=10)


attack = GradientMatchingAttack(model_art,
        percent_poison=0.01,
        max_trials=10,
        max_epochs=400,
        clip_values=(min_,max_),
        epsilon=0.3,
        verbose=False)

x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)




# # poisoning_attack_witches_brew
# # %% [markdown]
# # # Creating Image Trigger Poison Samples with ART
# # 
# # This notebook shows how to create image triggers in ART with RBG and grayscale images.


# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# import os, sys

# module_path = os.path.abspath(os.path.join('..'))
# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
# from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttackKeras
# # from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
# from art.utils import  preprocess, load_cifar10


# # %%
# (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10()

# mean = np.mean(x_raw,axis=(0,1,2,3))
# std = np.std(x_raw,axis=(0,1,2,3))
# x_raw = (x_raw-mean)/(std+1e-7)
# x_raw_test = (x_raw_test-mean)/(std+1e-7)

# min_ = (min_-mean)/(std+1e-7)
# max_ = (max_-mean)/(std+1e-7)


# # %%
# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import regularizers, optimizers


# # tf.compat.v1.disable_eager_execution()

# # Create Keras convolutional neural network - basic architecture from Keras examples
# # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# def create_model(x_train, y_train, num_classes=10, batch_size=64, epochs=25):
#     # model = Sequential()
#     # model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Conv2D(64, (3, 3), activation='relu'))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.25))
#     # model.add(Flatten())
#     # model.add(Dense(128, activation='relu'))
#     # model.add(Dropout(0.5))
#     # model.add(Dense(10, activation='softmax'))    
#     baseMapNum = 32
#     weight_decay = 1e-4
#     model = Sequential()
#     model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.2))

#     model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.3))

#     model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.4))

#     model.add(Flatten())
#     model.add(Dense(num_classes, activation='softmax'))


#     model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#     datagen = ImageDataGenerator(
#         featurewise_center=False,
#         samplewise_center=False,
#         featurewise_std_normalization=False,
#         samplewise_std_normalization=False,
#         zca_whitening=False,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         horizontal_flip=True,
#         vertical_flip=False
#         )
#     datagen.fit(x_train)
#     # model.fit(x_train, y_train, batch_size=batch_size, epochs=25)
#     model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1)
#     return model

# #%%

# # TODO: Temporarily disabled.
# # model = create_model(x_raw, y_raw, epochs=25)
# # model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)
# # model_path = "/home/taesung/projects/ai-robustness/models/cifar10-basic.h5"
# # # model.save(model_path)

# #%%

# model_path = "/home/taesung/projects/ai-robustness/models/cifar10-basic.h5"
# model = tf.keras.models.load_model(model_path)
# model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

# #%% 

# from tensorflow.keras.utils import to_categorical

# P = int(len(x_raw) * 0.01)
# class_source = 0
# class_target = 1
# index_target = np.where(y_raw_test.argmax(axis=1)==class_source)[0][5]
# y_poisoned = y_raw.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.
# indices_target = np.random.permutation(np.where(y_poisoned==class_target)[0])[:P]

# x_target = x_raw_test[index_target:index_target+1]
# y_target = to_categorical([class_target], num_classes=10)

# x_poison = x_raw[indices_target]
# y_poison = y_raw[indices_target]

# #%%

# # plt.imshow(x_target[0])
# # print(y_raw_test[index_target], y_target)
# # x_poison.shape
# # y_poison

# #%%


# attack = GradientMatchingAttackKeras(model_art, 
#         # target: np.ndarray,
#         # feature_layer: Union[Union[str, int], List[Union[str, int]]],
#         # opt: str = "adam",
#         max_iter=400,
#         learning_rate=1e-1,
#         # momentum: float = 0.9,
#         # decay_iter: Union[int, List[int]] = 10000,
#         # decay_coeff: float = 0.5,
#         epsilon=0.3,
#         # dropout: float = 0.3,
#         # net_repeat: int = 1,
#         # endtoend: bool = True,
#         # batch_size: int = 128,
#         verbose=False)


# x_poison = attack.poison(x_target, y_target, x_poison, y_poison)

# #%%

# x_raw_poisoned = x_raw.copy()
# x_raw_poisoned[indices_target] = x_poison

# model_poisoned = create_model(x_raw_poisoned, y_raw, epochs=25)
# y_ = model_poisoned.predict(x_target)

# print("y_target:", y_target)
# print("y_:", y_)

