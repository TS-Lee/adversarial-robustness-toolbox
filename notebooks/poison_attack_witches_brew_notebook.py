# poison_attack_witches_brew_notebook.py
# # Gradient Matching Attack on a TF Classifier
#%%

#%% ResNet definition
# Tweaked the model from https://github.com/calmisential/TensorFlow2.0_ResNet
# License: MIT License

import tensorflow as tf

# batch_norm_momentum = 0.1
# batch_norm_momentum = 0.99

# class BasicBlock(tf.keras.layers.Layer):

#     def __init__(self, filter_num, stride=1):
#         super(BasicBlock, self).__init__()
#         self.filter_num = filter_num
#         self.stride = stride
#         self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
#                                             kernel_size=(3, 3),
#                                             strides=stride,
#                                             padding="same")
#         self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)
#         self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
#                                             kernel_size=(3, 3),
#                                             strides=1,
#                                             padding="same")
#         self.bn2 = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)
#         if stride != 1:
#             self.downsample = tf.keras.Sequential()
#             self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
#                                                        kernel_size=(1, 1),
#                                                        strides=stride))
#             self.downsample.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum))
#         else:
#             self.downsample = lambda x: x

#     def call(self, inputs, training=None, **kwargs):
#         residual = self.downsample(inputs)

#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x, training=training)

#         output = tf.nn.relu(tf.keras.layers.add([residual, x]))

#         return output

#     def get_config(self):
#         return {
#             # "conv1", self.conv1,
#             # "bn1", self.bn1,
#             # "conv2", self.conv2,
#             # "bn2", self.bn2,
#             # "downsample", self.downsample,
#             "filter_num": self.filter_num,
#             "stride": self.stride
#         }
    
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


def basic_block(x, filter_num, stride=1):
    conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        padding="same")
    bn1 = tf.keras.layers.BatchNormalization()
    conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding="same")
    bn2 = tf.keras.layers.BatchNormalization()
    if stride != 1:
        downsample = tf.keras.Sequential()
        downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=stride))
        downsample.add(tf.keras.layers.BatchNormalization())
    else:
        downsample = tf.keras.layers.Lambda(lambda x: x)

    residual = downsample(x)
    x = conv1(x)
    x = bn1(x)
    x = tf.nn.relu(x)
    x = conv2(x)
    x = bn2(x)

    output = tf.nn.relu(tf.keras.layers.add([residual, x]))

    return output



# class BottleNeck(tf.keras.layers.Layer):
#     def __init__(self, filter_num, stride=1):
#         super(BottleNeck, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
#                                             kernel_size=(1, 1),
#                                             strides=1,
#                                             padding='same')
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
#                                             kernel_size=(3, 3),
#                                             strides=stride,
#                                             padding='same')
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
#                                             kernel_size=(1, 1),
#                                             strides=1,
#                                             padding='same')
#         self.bn3 = tf.keras.layers.BatchNormalization()

#         self.downsample = tf.keras.Sequential()
#         self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
#                                                    kernel_size=(1, 1),
#                                                    strides=stride))
#         self.downsample.add(tf.keras.layers.BatchNormalization())

#     def call(self, inputs, training=None, **kwargs):
#         residual = self.downsample(inputs)

#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x, training=training)

#         output = tf.nn.relu(tf.keras.layers.add([residual, x]))

#         return output


# def make_basic_block_layer(filter_num, blocks, stride=1):
#     res_block = tf.keras.Sequential()
#     res_block.add(BasicBlock(filter_num, stride=stride))

#     for _ in range(1, blocks):
#         res_block.add(BasicBlock(filter_num, stride=1))

#     return res_block

def basic_block_layer(x, filter_num, blocks, stride=1):
    x = basic_block(x, filter_num, stride=stride)

    for _ in range(1, blocks):
        x = basic_block(x, filter_num, stride=1)

    return x


# def make_bottleneck_layer(filter_num, blocks, stride=1):
#     res_block = tf.keras.Sequential()
#     res_block.add(BottleNeck(filter_num, stride=stride))

#     for _ in range(1, blocks):
#         res_block.add(BottleNeck(filter_num, stride=1))

#     return res_block


def resnet(x, num_classes, layer_params):
    pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
    conv1 = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding="same")
    bn1 = tf.keras.layers.BatchNormalization()

    avgpool = tf.keras.layers.GlobalAveragePooling2D()
    fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    x = pad1(x)
    x = conv1(x)
    x = bn1(x)
    x = tf.nn.relu(x)
    x = basic_block_layer(x, filter_num=64,
                                        blocks=layer_params[0])
    x = basic_block_layer(x, filter_num=128,
                                        blocks=layer_params[1],
                                        stride=2)
    x = basic_block_layer(x, filter_num=256,
                                        blocks=layer_params[2],
                                        stride=2)
    x = basic_block_layer(x, filter_num=512,
                                        blocks=layer_params[3],
                                        stride=2)
    x = avgpool(x)
    output = fc(x)

    return output

# class ResNetTypeI(tf.keras.layers.Layer):
#     def __init__(self, num_classes, layer_params):
#         super(ResNetTypeI, self).__init__()

#         self.num_classes = num_classes
#         self.layer_params = layer_params

#         self.pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
#         self.conv1 = tf.keras.layers.Conv2D(filters=64,
#                                             kernel_size=(3, 3),
#                                             strides=1,
#                                             padding="same")
#         self.bn1 = tf.keras.layers.BatchNormalization()

#         self.layer1 = make_basic_block_layer(filter_num=64,
#                                              blocks=layer_params[0])
#         self.layer2 = make_basic_block_layer(filter_num=128,
#                                              blocks=layer_params[1],
#                                              stride=2)
#         self.layer3 = make_basic_block_layer(filter_num=256,
#                                              blocks=layer_params[2],
#                                              stride=2)
#         self.layer4 = make_basic_block_layer(filter_num=512,
#                                              blocks=layer_params[3],
#                                              stride=2)

#         self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

#     def call(self, inputs, training=None, mask=None):
#         x = self.pad1(inputs)
#         x = self.conv1(x)
#         x = self.bn1(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.layer1(x, training=training)
#         x = self.layer2(x, training=training)
#         x = self.layer3(x, training=training)
#         x = self.layer4(x, training=training)
#         x = self.avgpool(x)
#         output = self.fc(x)

#         return output

#     def get_config(self):
#         return {
#             "num_classes": self.num_classes,
#             "layer_params": self.layer_params
#             # "pad1", self.pad1,
#             # "conv1", self.conv1,
#             # "bn1", self.bn1,
#             # "layer1", self.layer1,
#             # "layer2", self.layer2,
#             # "layer3", self.layer3,
#             # "layer4", self.layer4,
#             # "avgpool", self.avgpool,
#             # "fc", self.fc
#         }

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

# class ResNetTypeII(tf.keras.Model):
#     def __init__(self, num_classes, layer_params):
#         super(ResNetTypeII, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters=64,
#                                             kernel_size=(7, 7),
#                                             strides=2,
#                                             padding="same")
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
#                                                strides=2,
#                                                padding="same")

#         self.layer1 = make_bottleneck_layer(filter_num=64,
#                                             blocks=layer_params[0])
#         self.layer2 = make_bottleneck_layer(filter_num=128,
#                                             blocks=layer_params[1],
#                                             stride=2)
#         self.layer3 = make_bottleneck_layer(filter_num=256,
#                                             blocks=layer_params[2],
#                                             stride=2)
#         self.layer4 = make_bottleneck_layer(filter_num=512,
#                                             blocks=layer_params[3],
#                                             stride=2)

#         self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

#     def call(self, inputs, training=None, mask=None):
#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.pool1(x)
#         x = self.layer1(x, training=training)
#         x = self.layer2(x, training=training)
#         x = self.layer3(x, training=training)
#         x = self.layer4(x, training=training)
#         x = self.avgpool(x)
#         output = self.fc(x)

#         return output


# def resnet_18(num_classes):
#     return ResNetTypeI(num_classes, layer_params=[2, 2, 2, 2])


def resnet_18(x, num_classes):
    return resnet(x, num_classes, layer_params=[2, 2, 2, 2])

# def resnet_34(num_classes):
#     return ResNetTypeI(num_classes, layer_params=[3, 4, 6, 3])


# def resnet_50(num_classes):
#     return ResNetTypeII(num_classes, layer_params=[3, 4, 6, 3])


# def resnet_101(num_classes):
#     return ResNetTypeII(num_classes, layer_params=[3, 4, 23, 3])


# def resnet_152(num_classes):
#     return ResNetTypeII(num_classes, layer_params=[3, 8, 36, 3])












# %% [markdown]
# In this notebook, we will learn how to use ART to run a clean-label gradient matching poisoning attack on a neural network trained with Tensorflow. We will be training our data on a subset of the CIFAR-10 dataset. The methods described are derived from [this paper](https://arxiv.org/abs/2009.02276) by Geiping, et. al. 2020.

# %% [markdown]
# ## Train a model to attack
# 
# In this example, we use a RESNET50 model on the CIFAR dataset.

# %%
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
from tqdm.keras import TqdmCallback

tf.get_logger().setLevel('ERROR')


def create_model(x_train, y_train, num_classes=10, batch_size=64, epochs=25):
    # model = Sequential([
    #     resnet_18(num_classes)
    # #     tf.keras.layers.UpSampling2D(size=(7,7)),
    # #     tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None),
    # #     tf.keras.layers.GlobalAveragePooling2D(),
    # #     tf.keras.layers.Flatten(),
    # #     tf.keras.layers.Dense(1024, activation="relu"),
    # #     # tf.keras.layers.Dense(512, activation="relu"),
    # #     tf.keras.layers.Dense(num_classes, activation="softmax")
    # ])
    # model = resnet_18(num_classes)
    inputs = tf.keras.layers.Input(shape=x_train.shape[1:])  # Specify the dimensions
    outputs = resnet_18(inputs, num_classes)
    model = tf.keras.models.Model(inputs, outputs)

    # opt = tf.keras.optimizers.SGD()
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

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
    # datagen = ImageDataGenerator()
    datagen.fit(x_train)
    callbacks = [TqdmCallback(verbose=0)]
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=0,callbacks=callbacks)
    return model
    
# model = create_model(x_train, y_train, epochs=40)
# model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)
# exit(0)

model_path = "../models/cifar10-resnet18-notebook.h5"
if not os.path.exists(model_path):
    model = create_model(x_train, y_train, epochs=80)
    model.save(model_path)
else:
    model = tf.keras.models.load_model(model_path)
# TODO: Sequentialize the model and remove the Layer subclassing.

model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

print("Model and data preparation done.")


# %% [markdown]
# ## Choose Target Image from Test Set

# %%
from tensorflow.keras.utils import to_categorical

# A trigger from class 0 will be classified into class 1.
class_source = 0
class_target = 1
index_target = np.where(y_test.argmax(axis=1)==class_source)[0][5]

# Trigger sample
x_trigger = x_test[index_target:index_target+1]
y_trigger  = to_categorical([class_target], num_classes=10)

# %% [markdown]
# ## Poison Training Images to Misclassify the Trigger Image
# 

# %%
x_trigger.shape, y_trigger.shape, x_train.shape, y_train.shape

# %%
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack

epsilson = 0.10/(std+1e-7)

attack = GradientMatchingAttack(model_art,
        percent_poison=0.30,
        max_trials=10,
        max_epochs=500,
        clip_values=(min_,max_),
        epsilon=epsilson,
        verbose=1)

x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)


# %% [markdown]
# ## Examples of the trigger, an original sample, and the poisoned sample

# %%
import matplotlib.pyplot as plt
plt.imshow(x_trigger[0]*(std+1e-7)+mean)
plt.title('Trigger image')
plt.show()

index_poisoned_example = np.where([np.any(p!=o) for (p,o) in zip(x_poison,x_train)])[0]
plt.imshow(x_train[index_poisoned_example[0]]*(std+1e-7)+mean)
plt.title('Original image')
plt.show()

plt.imshow(x_poison[index_poisoned_example[0]]*(std+1e-7)+mean)
plt.title('Poisoned image')
plt.show()


# %% [markdown]
# ## Training with Poison Images

# %% [markdown]
# These attacks allow adversaries who can poison your dataset the ability to mislabel any particular target instance of their choosing without manipulating labels.

# %%
model_poisoned = create_model(x_poison, y_poison, epochs=80)
y_ = model_poisoned.predict(x_trigger)

print("y_trigger:", y_trigger)
print("y_:", y_)

if np.argmax(y_trigger) == np.argmax(y_):
    print("Poisoning was successful.")
else:
    print("Poisoning failed.")


