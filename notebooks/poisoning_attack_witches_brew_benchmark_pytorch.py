#%%
import math
from tqdm import trange
import numpy as np
import os, sys

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

from torchvision.models.resnet import BasicBlock, Bottleneck
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# Model from: https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# class ResNet(torchvision.models.ResNet):
#     """ResNet generalization for CIFAR-like thingies.
#     This is a minor modification of
#     https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
#     adding additional options.
#     This modification is from the authors of Witches' Brew.
#     Do NOT REDISTRIBUTE without checking the license.  
#     """

#     def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
#                  groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
#                  norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
#         """Initialize as usual. Layers and strides are scriptable."""
#         super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
#         self._norm_layer = norm_layer

#         self.dilation = 1
#         if len(replace_stride_with_dilation) != 4:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups

#         self.inplanes = base_width
#         self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
#         self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=initial_conv[0],
#                                      stride=initial_conv[1], padding=initial_conv[2], bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = torch.nn.ReLU(inplace=True)

#         layer_list = []
#         width = self.inplanes
#         for idx, layer in enumerate(layers):
#             layer_list.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
#             width *= 2
#         self.layers = torch.nn.Sequential(*layer_list)

#         self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = torch.nn.Linear(width // 2 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
#                 torch.nn.init.constant_(m.weight, 1)
#                 torch.nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     torch.nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     torch.nn.init.constant_(m.bn2.weight, 0)


#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.layers(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(model, test_loader):
    model_was_training = model.training
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    if model_was_training:
      model.train()
    return(accuracy)

def create_model(x_train, y_train, x_test=None, y_test=None, num_classes=10, batch_size=128, epochs=25):
    if x_test==None or y_test==None:
        x_test = x_train
        y_test = y_train
    initial_conv = [3, 1, 1]
    # model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    # model = torchvision.models.resnet18(num_classes=10)
    # model = ResNet18()

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0, nesterov=False)  # TODO: Test with a simpler optimizer.
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] + metrics)
    model.to(device)

    x_train = np.transpose(x_train, [0, 3,1,2])
    y_train = np.argmax(y_train, axis=1)
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device) # transform to torch tensor
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    x_test = np.transpose(x_test, [0, 3,1,2])
    y_test = np.argmax(y_test, axis=1)
    x_tensor_test = torch.tensor(x_test, dtype=torch.float32, device=device) # transform to torch tensor
    y_tensor_test = torch.tensor(y_test, dtype=torch.long, device=device)

    dataset_train = TensorDataset(x_tensor,y_tensor) # create your datset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

    dataset_test = TensorDataset(x_tensor_test,y_tensor_test) # create your datset
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

    for epoch in trange(epochs):
      running_loss = 0.0
      total = 0
      accuracy = 0
      for i, data in enumerate(dataloader_train, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # _, predicted = torch.max(outputs.data, 1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        accuracy += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
      train_accuracy = (100 * accuracy / total)
      print("Epoch %d train accuracy: %f" % (epoch, train_accuracy))
    test_accuracy = testAccuracy(model, dataloader_test)
    print("Final test accuracy: %f" % test_accuracy)
    return model, loss_fn, optimizer


model_path = "../models/cifar10-resnet18-pytorch.pth"
model, loss_fn, optimizer = create_model(x_train, y_train, epochs=80)
torch.save(model.state_dict(), model_path)

if not os.path.exists(model_path):
    model, loss_fn, optimizer = create_model(x_train, y_train, epochs=80)
    torch.save(model.state_dict(), model_path)
else:
    model, loss_fn, optimizer = create_model(x_train, y_train, epochs=0)
    model.load_state_dict(torch.load(model_path))

model_art = PyTorchClassifier(model, input_shape=x_train.shape[1:], loss=loss_fn, optimizer=optimizer, nb_classes=10)

print("Model and data preparation done.")


#%% [Witches' Brew paper experiment replication]
print("Witches' Brew paper experiment replication")

from art.utils import to_categorical
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack

success = 0
fail = 0
fail_different_from_original = 0
fail_by_original = 0
result_history = []

for _ in range(100): # 10 random poison-target pairs.
    index_x_trigger = np.random.randint(0, len(x_test))
    x_trigger = x_test[index_x_trigger:index_x_trigger+1]
    class_source = y_test[index_x_trigger].argmax(axis=-1)

    class_target = list(range(10))
    class_target.remove(class_source)
    class_target = np.random.choice(class_target)

    # Trigger sample
    y_trigger  = to_categorical([class_target], nb_classes=10)

    result_original = model_art.predict(torch.tensor(np.transpose(x_trigger, [0, 3,1,2]), dtype=torch.float32))
    if np.argmax(result_original) != class_source:
      fail_by_original += 1
      continue

    attack = GradientMatchingAttack(model_art,
            percent_poison=0.10,
            # max_trials=10,
            max_trials=1,
            max_epochs=500,
            # max_epochs=50,
            # learning_rate_schedule=([1e-1, 1e-2, 1e-3, 1e-4], [94, 156, 219, 250]),
            # learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4])/(max_-min_), [94, 156, 219, 250]),
            learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), [250, 350, 400, 430, 460]),
            clip_values=(min_,max_),
            epsilon=32/255 * (max_ - min_),
            batch_size=500,
            verbose=1)

    x_poison, y_poison = attack.poison(torch.tensor(np.transpose(x_trigger, [0, 3,1,2]), dtype=torch.float32), y_trigger, torch.tensor(np.transpose(x_train, [0, 3,1,2]), dtype=torch.float32), y_train)

    x_poison = np.transpose(x_poison, [0,2,3,1])
    model_poisoned, loss_fn, optimizer = create_model(x_poison, y_poison, epochs=80)
    model_poisoned.eval()
    result_poisoned = model_poisoned(torch.tensor(np.transpose(x_trigger, [0,3,1,2]), device=device, dtype=torch.float)).detach().cpu().numpy()

    print("y_trigger:", y_trigger)
    print("result_poisoned:", result_poisoned)
    print("result_original:", result_original)

    if np.argmax(result_poisoned) == class_target:
        success += 1
    else:
        fail += 1
        if np.argmax(result_poisoned) != class_source:
            fail_different_from_original += 1

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
