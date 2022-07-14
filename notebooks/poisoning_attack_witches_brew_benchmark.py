#%% [Resnet18 borrowed from https://github.com/jimmyyhwu/resnet18-tf2/blob/master/resnet.py]
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
regularizer = L2(5e-4)

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name, kernel_regularizer=regularizer)(x)
    # x = layers.ZeroPadding2D(padding=3, name=f'{name}_pad')(x)
    # return layers.Conv2D(filters=out_planes, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name=name, kernel_regularizer=regularizer)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    # out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    # out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)
    out = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, kernel_regularizer=regularizer, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=f'{name}.0.downsample.1'),
            # layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding2D(padding=1, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, kernel_initializer=kaiming_normal, kernel_regularizer=regularizer, name='conv1')(x)
    # x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    # x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    # x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=regularizer, name='fc')(x)
    x = layers.Activation("softmax")(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)

    
#%%

import random
import threading
import tensorflow.compat.v2 as tf

_SEED_GENERATOR = threading.local()

from tensorflow.python.keras import backend_config
from tensorflow.python.keras.engine import base_layer
# from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util

# The below functions are kept accessible from backend for compatibility.
epsilon = backend_config.epsilon
floatx = backend_config.floatx

def is_tf_random_generator_enabled():
  """Check whether `tf.random.Generator` is used for RNG in Keras.
  Compared to existing TF stateful random ops, `tf.random.Generator` uses
  `tf.Variable` and stateless random ops to generate random numbers,
  which leads to better reproducibility in distributed training.
  Note enabling it might introduce some breakage to existing code,
  by producing differently-seeded random number sequences
  and breaking tests that rely on specific random numbers being generated.
  To disable the
  usage of `tf.random.Generator`, please use
  `tf.keras.backend.experimental.disable_random_generator`.
  We expect the `tf.random.Generator` code path to become the default, and will
  remove the legacy stateful random ops such as `tf.random.uniform` in the
  future (see the
  [TF RNG guide](https://www.tensorflow.org/guide/random_numbers)).
  This API will also be removed in a future release as well, together with
  `tf.keras.backend.experimental.enable_tf_random_generator()` and
  `tf.keras.backend.experimental.disable_tf_random_generator()`
  Returns:
    boolean: whether `tf.random.Generator` is used for random number generation
      in Keras.
  """
  return _USE_GENERATOR_FOR_RNG

def enable_tf_random_generator():
  """Enable the `tf.random.Generator` as the RNG for Keras.
  See `tf.keras.backend.experimental.is_tf_random_generator_enabled` for more
  details.
  """
  global _USE_GENERATOR_FOR_RNG
  _USE_GENERATOR_FOR_RNG = True


def disable_tf_random_generator():
  """Disable the `tf.random.Generator` as the RNG for Keras.
  See `tf.keras.backend.experimental.is_tf_random_generator_enabled` for more
  details.
  """
  global _USE_GENERATOR_FOR_RNG
  _USE_GENERATOR_FOR_RNG = False


class RandomGenerator(tf.__internal__.tracking.AutoTrackable):
  """Random generator that selects appropriate random ops.
  This class contains the logic for legacy stateful random ops, as well as the
  new stateless random ops with seeds and tf.random.Generator. Any class that
  relies on RNG (eg initializer, shuffle, dropout) should use this class to
  handle the transition from legacy RNGs to new RNGs.
  """

  def __init__(self, seed=None, force_generator=False):
    self._seed = seed
    self._force_generator = force_generator
    self._built = False

  def _maybe_init(self):
    """Lazily init the RandomGenerator.
    The TF API executing_eagerly_outside_functions() has some side effect, and
    couldn't be used before API like tf.enable_eager_execution(). Some of the
    client side code was creating the initializer at the code load time, which
    triggers the creation of RandomGenerator. Lazy init this class to walkaround
    this issue until it is resolved on TF side.
    """
    # TODO(b/167482354): Change this back to normal init when the bug is fixed.
    if self._built:
      return

    if (tf.compat.v1.executing_eagerly_outside_functions() and
        (is_tf_random_generator_enabled() or self._force_generator)):
      # In the case of V2, we use tf.random.Generator to create all the random
      # numbers and seeds.
      from keras.utils import tf_utils  # pylint: disable=g-import-not-at-top
      with tf_utils.maybe_init_scope(self):
        if self._seed is not None:
          self._generator = tf.random.Generator.from_seed(self._seed)
        else:
          if getattr(_SEED_GENERATOR, 'generator', None):
            seed = _SEED_GENERATOR.generator.randint(1, 1e9)
          else:
            seed = random.randint(1, 1e9)
          self._generator = tf.random.Generator.from_seed(seed)
    else:
      # In the v1 case, we use stateful op, regardless whether user provide a
      # seed or not. Seeded stateful op will ensure generating same sequences.
      self._generator = None
    self._built = True

  def make_seed_for_stateless_op(self):
    """Generate a new seed based on the init config.
    Note that this will not return python ints which will be frozen in the graph
    and cause stateless op to return the same value. It will only return value
    when generator is used, otherwise it will return None.
    Returns:
      A tensor with shape [2,].
    """
    self._maybe_init()
    if self._generator:
      return self._generator.make_seeds()[:, 0]
    return None

  def make_legacy_seed(self):
    """Create a new seed for the legacy stateful ops to use.
    When user didn't provide any original seed, this method will return None.
    Otherwise it will increment the counter and return as the new seed.
    Note that it is important the generate different seed for stateful ops in
    the `tf.function`. The random ops will return same value when same seed is
    provided in the `tf.function`.
    Returns:
      int as new seed, or None.
    """
    if self._seed is not None:
      result = self._seed
      self._seed += 1
      return result
    return None

  def random_normal(self, shape, mean=0., stddev=1., dtype=None):
    self._maybe_init()
    dtype = dtype or floatx()
    if self._generator:
      return self._generator.normal(
          shape=shape, mean=mean, stddev=stddev, dtype=dtype)
    return tf.random.normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype,
        seed=self.make_legacy_seed())

  def random_uniform(self, shape, minval=0., maxval=None, dtype=None):
    self._maybe_init()
    dtype = dtype or floatx()
    if self._generator:
      return self._generator.uniform(
          shape=shape, minval=minval, maxval=maxval, dtype=dtype)
    return tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=dtype,
        seed=self.make_legacy_seed())

  def truncated_normal(self, shape, mean=0., stddev=1., dtype=None):
    self._maybe_init()
    dtype = dtype or floatx()
    if self._generator:
      return self._generator.truncated_normal(
          shape=shape, mean=mean, stddev=stddev, dtype=dtype)
    return tf.random.truncated_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype,
        seed=self.make_legacy_seed())



class RandomFlip(base_layer.Layer):
  """A preprocessing layer which randomly flips images during training.
  This layer will flip the images horizontally and or vertically based on the
  `mode` attribute. During inference time, the output will be identical to
  input. Call the layer with `training=True` to flip the input.
  For an overview and full list of preprocessing layers, see the preprocessing
  [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
  Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.
  Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.
  Attributes:
    mode: String indicating which flip mode to use. Can be `"horizontal"`,
      `"vertical"`, or `"horizontal_and_vertical"`. Defaults to
      `"horizontal_and_vertical"`. `"horizontal"` is a left-right flip and
      `"vertical"` is a top-bottom flip.
    seed: Integer. Used to create a random seed.
  """

  def __init__(self,
                horizontal, vertical,
               seed=None,
               **kwargs):
    super(RandomFlip, self).__init__(**kwargs)
    # base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomFlip').set(True)
    self.horizontal = horizontal
    self.vertical = vertical
    self.seed = seed
    self._random_generator = RandomGenerator(seed, force_generator=True)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    def random_flipped_inputs():
      flipped_outputs = inputs
      if self.horizontal:
        seed = self._random_generator.make_seed_for_stateless_op()
        if seed is not None:
          flipped_outputs = tf.image.stateless_random_flip_left_right(
              flipped_outputs, seed=seed)
        else:
          flipped_outputs = tf.image.random_flip_left_right(
              flipped_outputs, self._random_generator.make_legacy_seed())
      if self.vertical:
        seed = self._random_generator.make_seed_for_stateless_op()
        if seed is not None:
          flipped_outputs = tf.image.stateless_random_flip_up_down(
              flipped_outputs, seed=seed)
        else:
          flipped_outputs = tf.image.random_flip_up_down(
              flipped_outputs, self._random_generator.make_legacy_seed())
      return flipped_outputs

    output = control_flow_util.smart_cond(training, random_flipped_inputs,
                                          lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'horizontal': self.horizontal,
        'vertical': self.vertical,
        'seed': self.seed,
    }
    base_config = super(RandomFlip, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))





#%%


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


def create_model(x_train, y_train, num_classes=10, batch_size=128, epochs=25, metrics=[]):
    # model = Sequential([
    #     tf.keras.applications.ResNet18(input_shape=x_train.shape[1:], include_top=False, weights=None),
    #     Flatten(),
    #     Dense(num_classes, activation='softmax')
    # ])
    data_augmentation = tf.keras.Sequential([
        RandomFlip(True, True),
        # RandomTranslation(0.1, 0.1)
        # layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # layers.experimental.preprocessing.RandomRotation(0.2),
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.random_shift(x, 0.1, 0.1)),
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.random_brightness(x, (-8, 8))),
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.random_channel_shift(x, 8, 0.1)),
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.random_rotation(x, 10)),
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.random_zoom(x, (0.1,0.1))),
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.random_shear(x, 10)),
        # # TODO: Do we need normalization? Put it here.
        # layers.Lambda(lambda x: tf.keras.preprocessing.image.smart_resize(x, (32, 32))),
        ])
    x = inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    # x = data_augmentation(x)
    outputs = resnet18(x, num_classes=num_classes)
    model = keras.Model(inputs, outputs)

    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    # opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] + metrics)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # datagen = ImageDataGenerator(
    #     featurewise_center=False,
    #     samplewise_center=True,
    #     featurewise_std_normalization=False,
    #     samplewise_std_normalization=True,
    #     zca_whitening=False,
    #     # rotation_range=0.05,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=False,
    #     vertical_flip=False
    #     )
    from tqdm.keras import TqdmCallback
    datagen = ImageDataGenerator()
    datagen.fit(x_train)
    callbacks = [TqdmCallback(verbose=0)]
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
      steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
      verbose=0,
      callbacks=callbacks)
    return model


model_path = "/home/taesung/projects/ai-robustness/models/cifar10-resnet18.h5"
# model = create_model(x_train, y_train, epochs=80)
# model.save(model_path)

if not os.path.exists(model_path):
    model = create_model(x_train, y_train, epochs=80)
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

# # A trigger from class 0 will be classified into class 1.
# class_source = 0
# class_target = 1
# index_target = np.where(y_test.argmax(axis=1)==class_source)[0][5]
# y_poisoned = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.

# # Trigger sample
# x_trigger = x_test[index_target:index_target+1]
# y_trigger  = to_categorical([class_target], num_classes=10)


# #%%
# print("Poisoning attack start...")
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack

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


# x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)

# # x_train_poisoned = x_train.copy()
# # x_raw_poisoned[indices_target] = x_poison

# print("Poisoning data done.")

# model_poisoned = create_model(x_poison, y_poison, epochs=25)
# y_ = model_poisoned.predict(x_trigger)

# print(y_trigger)
# print(y_)


#%% [Witches' Brew paper experiment replication]
print("Witches' Brew paper experiment replication")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

success = 0
fail = 0
fail_different_from_original = 0
fail_by_original = 0
result_history = []


def choose_triggers_from_all_classes(x_test, y_test, num_classes=10):
  x_triggers = []
  y_triggers = []
  y_originals = []
  for class_source in range(num_classes):
    index_x_trigger = np.random.choice(np.where(y_test.argmax(axis=-1) == class_source)[0])
    x_trigger = x_test[index_x_trigger:index_x_trigger+1]
    class_source_2 = y_test[index_x_trigger].argmax(axis=-1)
    assert(class_source_2==class_source)

    class_target = list(range(10))
    class_target.remove(class_source)
    class_target = np.random.choice(class_target)

    y_original = to_categorical([class_source], num_classes=num_classes) 
    y_trigger = to_categorical([class_target], num_classes=num_classes)

    x_triggers.append(x_trigger)
    y_triggers.append(y_trigger)
    y_originals.append(y_original)
  return np.asarray(x_triggers), np.asarray(y_originals), np.asarray(y_triggers)

for _ in range(10): # 10 random poison-target pairs.
    # index_x_trigger = np.random.randint(0, len(x_test))
    # x_trigger = x_test[index_x_trigger:index_x_trigger+1]
    # class_source = y_test[index_x_trigger].argmax(axis=-1)

    # class_target = list(range(10))
    # class_target.remove(class_source)
    # class_target = np.random.choice(class_target)
    # # index_x_trigger = np.where(y_test.argmax(axis=1)==class_source)[0][5]
    # # y_poisoned = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.
    # # y_poison = y_train.argmax(axis=-1)  # y_poisoned is not to be passed to the model trainer. It is only used to optimize x_poisoned.

    # # Trigger sample
    # y_trigger  = to_categorical([class_target], num_classes=10)

    x_trigger, y_source, y_trigger = choose_triggers_from_all_classes(x_test, y_test, num_classes=10)

    result_original = model_art.predict(x_trigger)
    fail_by_original += np.sum(np.argmax(result_original,axis=-1) != np.argmax(y_source, axis=-1))

    attack = GradientMatchingAttack(model_art,
            # percent_poison=0.10,
            percent_poison=1.00,
            # max_trials=10,
            max_trials=1,
            max_epochs=500,
            learning_rate_schedule=([1e-1, 1e-2, 1e-3, 1e-4], [94, 156, 219, 250]),
            # learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4])*(max_-min_), [200, 400, 500, 550]),
            clip_values=(min_,max_),
            # epsilon=255/255 * (max_ - min_),
            epsilon=16/255 * (max_ - min_),
            batch_size=500,
            verbose=1)

    x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train)
    # x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_trigger, y_trigger)

    model_poisoned = create_model(x_poison, y_poison, epochs=80)
    result_poisoned = model_poisoned.predict(x_trigger)

    print("y_trigger:", y_trigger)
    print("result_poisoned:", result_poisoned)
    print("result_original:", result_original)

    s = np.sum(np.argmax(result_poisoned,axis=-1) == np.argmax(y_trigger,axis=-1))
    success += s
    fail += len(y_trigger) - s
    fail_different_from_original += np.sum(np.argmax(result_poisoned) != np.argmax(y_source, axis=-1))

    print("success:", success)
    print("fail:", fail)
    print("fail_different_from_original:", fail_different_from_original)
    print("fail_by_original:", fail_by_original)
    result_history.append((y_trigger, result_poisoned, result_original, success, fail, fail_different_from_original, fail_by_original))


print("success:", success)
print("fail:", fail)
print("fail_different_from_original:", fail_different_from_original)
print("fail_by_original:", fail_by_original)

print("experiments done.")

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

