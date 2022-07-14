#%%

import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

min_ = (min_-mean)/(std+1e-7)
max_ = (max_-mean)/(std+1e-7)


x_train = np.transpose(x_train, [0,3,1,2])
y_train = np.argmax(y_train, axis=-1)
x_test = np.transpose(x_test, [0,3,1,2])
y_test = np.argmax(y_test, axis=-1)
# x_train_torch = torch.tensor(np.transpose(x_train, [0,3,1,2]), dtype=torch.float32)
# y_train_torch = torch.tensor(np.argmax(y_train, axis=-1))
# x_test_torch = torch.tensor(np.transpose(x_test, [0,3,1,2]), dtype=torch.float32)

x_train_torch = torch.tensor(x_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train)



#%%

# TODO: Build a pytorch model.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            # nn.Softmax()
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        # return logits
        # pred_probab = nn.LogSoftmax(dim=1)(logits)
        y = self.sequential(x)
        return y

def create_model(x_train_torch, y_train_torch):
    model = NeuralNetwork().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(16):
        y = model(x_train_torch)
        loss = criterion(y, y_train_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d] loss: %.3f' %
                (epoch + 1, loss.item()))
        if epoch % 10 == 0:
            print("Training accuracy: %f" % np.average(np.argmax(y.clone().detach().numpy(),axis=-1)==np.argmax(y_train,axis=-1)))
    return model, criterion, optimizer

#%%

model, criterion, optimizer = create_model(x_train_torch, y_train_torch)

#%%



from art.utils import to_categorical
# INPUT: x_trigger, y_trigger, x_poison, y_poison : np.ndarray

# A trigger from class 0 will be classified into class 1.
class_source = 0
class_target = 1
index_target = np.where(y_test==class_source)[0][5]
# y_poisoned = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.

# Trigger sample
x_trigger = x_test[index_target:index_target+1]
# y_trigger = to_categorical([class_target], nb_classes=10)
y_trigger = np.array([class_target])

# TODO: Test the code with the above inputs.

indices_poison = np.where(y_train==class_target)
x_poison = x_train[indices_poison]
y_poison = y_train[indices_poison]


#%% [Running gradient matching with pytorch classifier.]

from art.estimators.classification import PyTorchClassifier
model_art = PyTorchClassifier(model, criterion, x_train.shape[1:], 10)

#%%
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack

attack = GradientMatchingAttack(model_art,
        percent_poison=0.01,
        max_trials=1,
        max_epochs=10, # max_epochs=400,
        clip_values=(min_,max_),
        epsilon=0.3,
        verbose=False)


x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)

model_poisoned, _, _ = create_model(torch.tensor(x_poison, dtype=torch.float32), torch.tensor(y_poison))
y_ = model_poisoned(torch.tensor(x_trigger))

print(y_trigger)
print(y_)




# #%%

# # y = model(x_train_torch)
# # np.average(np.argmax(y.detach().numpy(),axis=-1)==np.argmax(y_train,axis=-1))


# #%%

# #%% [Old test implementation starts]


# def grad(classifier, x, target):
#     classifier.model.zero_grad()
#     y = classifier.model(x)
#     loss_ = classifier.loss(y, target)
#     loss_.backward()
#     d_w = [w.grad for w in classifier.model.parameters()]
#     d_w = torch.cat([w.flatten() for w in d_w])
#     d_w_norm = d_w / torch.sqrt(torch.sum(torch.square(d_w)))
#     return d_w_norm

# classifier = PyTorchClassifier(model, criterion, x_train.shape[1:], 10)


# #%%


# #%%

# # x_trigger, y_trigger, x_train, y_train

# num_poison = len(x_poison)
# len_noise = np.prod(x_poison.shape[1:])

# #%%

# class NoiseEmbedding(nn.Module):
#     def __init__(self, num_poison, len_noise, epsilon, clip_values):
#         super(NoiseEmbedding, self).__init__()

#         # nn.Embedding(len(x_poison), np.prod(input_poison.shape[1:]))
#         self.embedding_layer = nn.Embedding(num_poison, len_noise)
#         self.epsilon = epsilon
#         self.clip_values = clip_values

#     def forward(self, input_poison, input_indices):
#         # input_indices.requires_grad = False
#         embeddings = self.embedding_layer(input_indices)
#         embeddings = torch.clip(embeddings, -self.epsilon, self.epsilon)
#         embeddings = embeddings.view(input_poison.shape)

#         input_noised = input_poison + embeddings
#         input_noised = torch.clip(input_noised, self.clip_values[0], self.clip_values[1])  # Make sure the poisoned samples are in a valid range.

#         return input_noised

# # def loss_fn(input_noised, target, grad_ws_norm):
# #     d_w2_norm = grad(self.substitute_classifier.model, input_noised, target)
# #     B = 1 - tf.reduce_sum(grad_ws_norm * d_w2_norm)  # pylint: disable=C0103
# #     return B

# class BackdoorModel(nn.Module):
#     def __init__(self, classifier, epsilon, num_poison, len_noise, min_, max_):
#         super(BackdoorModel, self).__init__()
#         self.classifier = classifier
#         self.ne = NoiseEmbedding(num_poison, len_noise, epsilon, (min_, max_))

#     def forward(self, x, indices_poison, y):
#         grad_ws_norm = grad(self.classifier, x, y)
#         d_w2_norm = grad(self.classifier, self.ne(x, indices_poison), y)
#         grad_ws_norm.requires_grad_(True)
#         d_w2_norm.requires_grad_(True)
#         B_score = 1 - nn.CosineSimilarity(dim=0)(grad_ws_norm, d_w2_norm)
#         return B_score


# #%%

# bm = BackdoorModel(classifier, 0.1, num_poison, len_noise, min_, max_)
# optimizer = torch.optim.Adam(bm.ne.embedding_layer.parameters(), lr = 0.0001)
# # ne.zero_grad()

# embeddings_original = bm.ne.embedding_layer.weight[0].clone().detach().numpy()

# # grad_ws_norm = grad(bm.classifier, torch.tensor(x_poison, dtype=torch.float), torch.tensor(y_poison))
# # loss = bm.ne(torch.tensor(x_poison, dtype=torch.float), torch.arange(0, len(x_poison), dtype=torch.int32))
# # grad_ws_norm = grad(classifier, torch.tensor(x_poison, dtype=torch.float), torch.tensor(y_poison))
# # d_w2_norm = grad(classifier, ne(torch.tensor(x_poison, dtype=torch.float), torch.arange(0, len(x_poison), dtype=torch.int32)), torch.tensor(y_poison))
# # loss = 1 - nn.CosineSimilarity(dim=0)(grad_ws_norm, d_w2_norm)
# # loss[0].backward()
# # grad_ws_norm[0].backward()

# # x = torch.tensor(x_poison, dtype=torch.float)
# # target = torch.tensor(y_poison)
# # # x.requires_grad = True
# # # target.requires_grad = True
# # bm.classifier.model.zero_grad()
# # y = bm.classifier.model(x)
# # loss_ = bm.classifier.loss(y, target)
# # loss_.backward()
# # d_w = [w.grad for w in bm.classifier.model.parameters()]
# # d_w = torch.cat([w.flatten() for w in d_w])
# # # d_w_norm = d_w / torch.sqrt(torch.sum(torch.square(d_w)))
# # loss = torch.sum(d_w) * 2
# # loss.backward()

# #%%

# for _ in range(100):
#     bm.zero_grad()
#     loss = bm(torch.tensor(x_poison[0:5], dtype=torch.float), torch.arange(0, len(x_poison[0:5]), dtype=torch.int32), torch.tensor(y_poison[0:5]))
#     # loss = bm(torch.tensor(x_poison, dtype=torch.float), torch.arange(0, len(x_poison), dtype=torch.int32), torch.tensor(y_poison))
#     loss.backward()
#     optimizer.step()

# embeddings_trained = bm.ne.embedding_layer.weight[0].clone().detach().numpy()

# # d_w = d_w.detach()
# # d_w.requires_grad = True
# # loss = torch.sum(d_w) * 2

# # loss.backward()
# # loss[0,0,0,0].backward()

# # loss.backward()
# # optimizer.step()

# #%%

# # np.all(embeddings_original == embeddings_trained)
# # np.sum(embeddings_original == embeddings_trained)

# #%%

# import matplotlib.pyplot as plt

# # plt.plot(bm.ne.embedding_layer.weight[0].view(x_poison.shape[1:]).transpose(1,2,0)))
# # print(bm.ne.embedding_layer.weight.shape)
# plt.imshow(np.transpose(bm.ne.embedding_layer.weight[0].view(x_poison.shape[1:]).clone().detach().numpy(), [1,2,0]))


# #%%

# # class CosineLoss(nn.Module):
# #     def __init__(self, weight=None, size_average=True):
# #         super(CosineLoss, self).__init__()
    
# #     def forward(self, d_w2_norm, grad_ws_norm):
# #         # batch_size = outputs.size()[0]            # batch_size
# #         # outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
# #         # outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
# #         B = 1 - torch.sum(grad_ws_norm * d_w2_norm)  # pylint: disable=C0103
# #         return B

# # TODO: lr_schedule = tf.keras.callbacks.LearningRateScheduler(PredefinedLRSchedule(*self.learning_rate_schedule))
# # m.fit(
# #     [x_poison, y_poison, np.arange(len(y_poison))],
# #     callbacks=callbacks,
# #     batch_size=self.batch_size,
# #     epochs=self.max_epochs,
# #     verbose=self.verbose,
# # )




# # input_poison = Input(batch_shape=self.substitute_classifier.model.input.shape)
# # input_indices = Input(shape=())
# # # y_true_poison = Input(shape=self.substitute_classifier.model.output.shape)
# # # y_true_poison = Input(shape=np.shape(y_trigger)[1:])
# # y_true_poison = Input(shape=np.shape(y_poison)[1:])

# # Poison embedding layer.
# # Clip the embedding, or apply the constrain.
# # nn.Embedding(len(x_poison), np.prod(input_poison.shape[1:]))

# # class ClipConstraint(tf.keras.constraints.MaxNorm):
# #     """
# #     Clip the tensor values.
# #     """

# #     def __init__(self, max_value=2):
# #         super().__init__(max_value=max_value)

# #     def __call__(self, w):
# #         return tf.clip_by_value(w, -self.max_value, self.max_value)

# # # Define the model to apply and optimize the poison.
# # input_poison = Input(batch_shape=self.substitute_classifier.model.input.shape)
# # input_indices = Input(shape=())
# # # y_true_poison = Input(shape=self.substitute_classifier.model.output.shape)
# # # y_true_poison = Input(shape=np.shape(y_trigger)[1:])
# # y_true_poison = Input(shape=np.shape(y_poison)[1:])
# # embedding_layer = Embedding(len(x_poison), np.prod(input_poison.shape[1:]))


# # embeddings = embedding_layer(input_indices)
# # embeddings = ClipConstraint(max_value=self.epsilon)(embeddings)
# # embeddings = tf.reshape(embeddings, tf.shape(input_poison))
# # input_noised = Add()([input_poison, embeddings])
# # input_noised = Lambda(lambda x: K.clip(x, self.clip_values[0], self.clip_values[1]))(
# #     input_noised
# # )  # Make sure the poisoned samples are in a valid range.

# # def loss_fn(input_noised, target, grad_ws_norm):
# #     d_w2_norm = grad_loss(self.substitute_classifier.model, input_noised, target)
# #     B = 1 - tf.reduce_sum(grad_ws_norm * d_w2_norm)  # pylint: disable=C0103
# #     return B




# #%% 
# # def grad_loss(model, x, target):
# #     with tf.GradientTape() as t:  # pylint: disable=C0103
# #         t.watch(model.weights)
# #         output = model(x)
# #         loss = model.compiled_loss(target, output)
# #     d_w = t.gradient(loss, model.trainable_weights)
# #     d_w = tf.concat([tf.reshape(d, [-1]) for d in d_w], 0)
# #     d_w_norm = d_w / tf.sqrt(tf.reduce_sum(tf.square(d_w)))
# #     return d_w_norm



# #%%

# def create_model(x_train, y_train, num_classes=10, batch_size=64, epochs=25):
#     model = Sequential([
#         tf.keras.applications.ResNet50(input_shape=x_train.shape[1:], include_top=False, weights=None),
#         Flatten(),
#         Dense(num_classes, activation='softmax')
#     ])

#     # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
#     model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1)
#     return model

# model_path = "/home/taesung/projects/ai-robustness/models/cifar10-resnet50.h5"
# if not os.path.exists(model_path):
#     model = create_model(x_train, y_train, epochs=25)
#     model.save(model_path)
# else:
#     model = tf.keras.models.load_model(model_path)

# model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

# #%%

# # model_path = "/home/taesung/projects/ai-robustness/models/cifar10-basic.h5"
# # model = tf.keras.models.load_model(model_path)
# # model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

# #%% 

# # x_train = x_train
# # y_train = y_train
# # epochs = 5
# # num_classes = 10
# # batch_size = 32

# # model = Sequential([
# #     tf.keras.applications.ResNet50(input_shape=x_train.shape[1:], include_top=False, weights=None),
# #     Flatten(),
# #     Dense(num_classes, activation='softmax')
# # ])

# # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# # datagen = ImageDataGenerator(
# #     featurewise_center=False,
# #     samplewise_center=False,
# #     featurewise_std_normalization=False,
# #     samplewise_std_normalization=False,
# #     zca_whitening=False,
# #     rotation_range=15,
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     horizontal_flip=True,
# #     vertical_flip=False
# #     )
# # datagen.fit(x_train)
# # model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1)


# from tensorflow.keras.utils import to_categorical

# # A trigger from class 0 will be classified into class 1.
# class_source = 0
# class_target = 1
# index_target = np.where(y_test.argmax(axis=1)==class_source)[0][5]
# y_poisoned = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.

# # Trigger sample
# x_trigger = x_test[index_target:index_target+1]
# y_trigger  = to_categorical([class_target], num_classes=10)

# indices_poison = np.where(np.argmax(y_train, axis=-1)==class_target)
# x_poison = x_train[indices_poison]
# y_poison = y_train[indices_poison]


# from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack

# # # Samples to poison
# # percent_poison =  0.01
# # P = int(len(x_train) * percent_poison)
# # indices_poison = np.random.permutation(np.where(y_poisoned==class_target)[0])[:P]
# # x_poison = x_train[indices_poison]
# # y_poison = y_train[indices_poison]

# attack = GradientMatchingAttack(model_art,
#         percent_poison=0.01,
#         max_trials=10,
#         max_epochs=400,
#         clip_values=(min_,max_),
#         epsilon=0.3,
#         verbose=False)


# x_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)


# # # poisoning_attack_witches_brew
# # # %% [markdown]
# # # # Creating Image Trigger Poison Samples with ART
# # # 
# # # This notebook shows how to create image triggers in ART with RBG and grayscale images.


# # # %%
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os, sys

# # module_path = os.path.abspath(os.path.join('..'))
# # module_path = os.path.abspath(os.path.join('.'))
# # if module_path not in sys.path:
# #     sys.path.append(module_path)

# # from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
# # from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttackKeras
# # # from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
# # from art.utils import  preprocess, load_cifar10


# # # %%
# # (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10()

# # mean = np.mean(x_raw,axis=(0,1,2,3))
# # std = np.std(x_raw,axis=(0,1,2,3))
# # x_raw = (x_raw-mean)/(std+1e-7)
# # x_raw_test = (x_raw_test-mean)/(std+1e-7)

# # min_ = (min_-mean)/(std+1e-7)
# # max_ = (max_-mean)/(std+1e-7)


# # # %%
# # import tensorflow.keras.backend as K
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras import regularizers, optimizers


# # # tf.compat.v1.disable_eager_execution()

# # # Create Keras convolutional neural network - basic architecture from Keras examples
# # # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# # def create_model(x_train, y_train, num_classes=10, batch_size=64, epochs=25):
# #     # model = Sequential()
# #     # model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
# #     # model.add(MaxPooling2D(pool_size=(2, 2)))
# #     # model.add(Conv2D(64, (3, 3), activation='relu'))
# #     # model.add(MaxPooling2D(pool_size=(2, 2)))
# #     # model.add(Dropout(0.25))
# #     # model.add(Flatten())
# #     # model.add(Dense(128, activation='relu'))
# #     # model.add(Dropout(0.5))
# #     # model.add(Dense(10, activation='softmax'))    
# #     baseMapNum = 32
# #     weight_decay = 1e-4
# #     model = Sequential()
# #     model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(MaxPooling2D(pool_size=(2,2)))
# #     model.add(Dropout(0.2))

# #     model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(MaxPooling2D(pool_size=(2,2)))
# #     model.add(Dropout(0.3))

# #     model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(MaxPooling2D(pool_size=(2,2)))
# #     model.add(Dropout(0.4))

# #     model.add(Flatten())
# #     model.add(Dense(num_classes, activation='softmax'))


# #     model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# #     datagen = ImageDataGenerator(
# #         featurewise_center=False,
# #         samplewise_center=False,
# #         featurewise_std_normalization=False,
# #         samplewise_std_normalization=False,
# #         zca_whitening=False,
# #         rotation_range=15,
# #         width_shift_range=0.1,
# #         height_shift_range=0.1,
# #         horizontal_flip=True,
# #         vertical_flip=False
# #         )
# #     datagen.fit(x_train)
# #     # model.fit(x_train, y_train, batch_size=batch_size, epochs=25)
# #     model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1)
# #     return model

# # #%%

# # # TODO: Temporarily disabled.
# # # model = create_model(x_raw, y_raw, epochs=25)
# # # model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)
# # # model_path = "/home/taesung/projects/ai-robustness/models/cifar10-basic.h5"
# # # # model.save(model_path)

# # #%%

# # model_path = "/home/taesung/projects/ai-robustness/models/cifar10-basic.h5"
# # model = tf.keras.models.load_model(model_path)
# # model_art = TensorFlowV2Classifier(model, nb_classes=10, input_shape=model.input_shape)

# # #%% 

# # from tensorflow.keras.utils import to_categorical

# # P = int(len(x_raw) * 0.01)
# # class_source = 0
# # class_target = 1
# # index_target = np.where(y_raw_test.argmax(axis=1)==class_source)[0][5]
# # y_poisoned = y_raw.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.
# # indices_target = np.random.permutation(np.where(y_poisoned==class_target)[0])[:P]

# # x_target = x_raw_test[index_target:index_target+1]
# # y_target = to_categorical([class_target], num_classes=10)

# # x_poison = x_raw[indices_target]
# # y_poison = y_raw[indices_target]

# # #%%

# # # plt.imshow(x_target[0])
# # # print(y_raw_test[index_target], y_target)
# # # x_poison.shape
# # # y_poison

# # #%%


# # attack = GradientMatchingAttackKeras(model_art, 
# #         # target: np.ndarray,
# #         # feature_layer: Union[Union[str, int], List[Union[str, int]]],
# #         # opt: str = "adam",
# #         max_iter=400,
# #         learning_rate=1e-1,
# #         # momentum: float = 0.9,
# #         # decay_iter: Union[int, List[int]] = 10000,
# #         # decay_coeff: float = 0.5,
# #         epsilon=0.3,
# #         # dropout: float = 0.3,
# #         # net_repeat: int = 1,
# #         # endtoend: bool = True,
# #         # batch_size: int = 128,
# #         verbose=False)


# # x_poison = attack.poison(x_target, y_target, x_poison, y_poison)

# # #%%

# # x_raw_poisoned = x_raw.copy()
# # x_raw_poisoned[indices_target] = x_poison

# # model_poisoned = create_model(x_raw_poisoned, y_raw, epochs=25)
# # y_ = model_poisoned.predict(x_target)

# # print("y_target:", y_target)
# # print("y_:", y_)

