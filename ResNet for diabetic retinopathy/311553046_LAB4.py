import torch
import torch.nn as nn
from torchvision import models
import torch.optim 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import RetinopathyLoader
# In[]
class BasicBlock(nn.Module):
    def __init__(self, filter_nums, strides=1, expansion=False):
        super(BasicBlock, self).__init__()
        in_channels, out_channels = filter_nums
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=strides, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if expansion:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = lambda x:x
    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        identity = self.identity(inputs)
        output = self.relu(identity + output)
        return output
# In[]
def BasicBlock_build(filter_nums, block_nums, strides=1, expansion=False):
    build_model = []
    build_model.append(BasicBlock(filter_nums, strides, expansion=expansion))
    filter_nums[0] = filter_nums[1]
    for i in range(block_nums - 1):
        build_model.append(BasicBlock(filter_nums, strides=1))
    return nn.Sequential(*build_model)
# In[]
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            BasicBlock_build(filter_nums=[64, 64], block_nums=2, strides=1)
        )
        self.conv3 = BasicBlock_build(filter_nums=[64, 128], block_nums=2, strides=2, expansion=True)
        self.conv4 = BasicBlock_build(filter_nums=[128, 256], block_nums=2, strides=2, expansion=True)
        self.conv5 = BasicBlock_build(filter_nums=[256, 512], block_nums=2, strides=2, expansion=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc =nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 5)
            )

    def forward(self,inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avgpool(output)
        output = self.fc(output)
        return output
# In[]
class BottleneckBlock(nn.Module):
    def __init__(self, filter_nums, strides=1, expansion=False):
        super(BottleneckBlock, self).__init__()
        in_channels, mid_channels, out_channels = filter_nums
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (1, 1), stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if strides!=1 or expansion:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=strides, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = lambda inputs:inputs

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        identity = self.identity(inputs)
        output = self.relu(identity + output)
        return output
# In[]
def Bottleneck_build(filter_nums, block_nums, strides=1, expansion=False):
    build_model = []
    build_model.append(BottleneckBlock(filter_nums, strides, expansion=expansion))
    filter_nums[0] = filter_nums[2]
    for i in range(block_nums - 1):
        build_model.append(BottleneckBlock(filter_nums, strides=1))
    return nn.Sequential(*build_model)
# In[]
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            Bottleneck_build(filter_nums=[64, 64, 256], block_nums=3, strides=1, expansion=True)
        )
        self.conv3 = Bottleneck_build(filter_nums=[256, 128, 512], block_nums=4, strides=2, expansion=True)
        self.conv4 = Bottleneck_build(filter_nums=[512, 256, 1024], block_nums=6, strides=2, expansion=True)
        self.conv5 = Bottleneck_build(filter_nums=[1024, 512, 2048], block_nums=3, strides=2, expansion=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 5)
            )
    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avgpool(output)
        output = self.fc(output)
        return output
# In[]
class Pretrain_ResNet18(nn.Module):
    def __init__(self):
        super(Pretrain_ResNet18, self).__init__()
        pretrain=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = pretrain._modules['conv1']
        self.bn1 = pretrain._modules['bn1']
        self.relu = pretrain._modules['relu']
        self.maxpool =pretrain._modules['maxpool']

        self.layer1 = pretrain._modules['layer1']
        self.layer2 = pretrain._modules['layer2']
        self.layer3 = pretrain._modules['layer3']
        self.layer4 = pretrain._modules['layer4']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pretrain._modules['fc'].in_features,5)
            )
        del pretrain
    def forward(self,inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = self.fc(outputs)
        return outputs
# In[]
class Pretrain_ResNet50(nn.Module):
    def __init__(self):
        super(Pretrain_ResNet50, self).__init__()
        pretrain=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = pretrain._modules['conv1']
        self.bn1 = pretrain._modules['bn1']
        self.relu = pretrain._modules['relu']
        self.maxpool =pretrain._modules['maxpool']

        self.layer1 = pretrain._modules['layer1']
        self.layer2 = pretrain._modules['layer2']
        self.layer3 = pretrain._modules['layer3']
        self.layer4 = pretrain._modules['layer4']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pretrain._modules['fc'].in_features,5)
            )
        del pretrain
    def forward(self,inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = self.fc(outputs)
        return outputs
# In[]
def train(model,weight_path ,loss_func, optimizer, epochs, train_dataset, test_dataset):
    training_acc=[]
    testing_acc=[]
    best_acc=0.0
    for epoch in range(epochs):
        print(epoch)
        all_train_correct=0
        total_train=0
        # Train model
        model.train()
        for i, (inputs, labels) in enumerate(train_dataset):
            # 1.Define variables
            inputs=inputs.cuda()
            labels=labels.cuda().long()
            # 2.Forward propagation
            outputs=model.forward(inputs=inputs)
            # 3.Clear gradients
            optimizer.zero_grad()
            # 4.Calculate softmax and cross entropy loss
            loss=loss_func(outputs,labels)
            # 5. Calculate gradients
            loss.backward()
            # 6. Update parameters
            optimizer.step()
            # 7.Total correct predictions
            pred = torch.max(outputs.data, 1)[1]
            all_train_correct += (pred == labels).long().sum()
            total_train+=len(labels)
            torch.cuda.empty_cache()

        training_acc.append((100*all_train_correct/float(total_train)).cpu())
        print(100*all_train_correct/float(total_train))
        # Test model
        model.eval()
        total_test=0
        all_test_correct=0
        with torch.no_grad():
            for inputs, labels in test_dataset:
                inputs=inputs.cuda()
                labels=labels.cuda().long()
                outputs=model.forward(inputs=inputs)
                pred = torch.max(outputs.data, 1)[1]
                all_test_correct += (pred == labels).long().sum()
                total_test+=len(labels)
            now_testing_acc=(100* all_test_correct/float(total_test))
            testing_acc.append(now_testing_acc.cpu())
            if now_testing_acc>best_acc:
                best_acc=now_testing_acc
                torch.save(model.state_dict(),weight_path)
        print(now_testing_acc)
    return training_acc,testing_acc, best_acc
# In[]
def plot(model_name, R18_Ptraining, R18_Ptesting, R18_Ntraining, R18_Ntesting,epochs):
      plt.plot(range(1,epochs+1),R18_Ptraining) 
      plt.plot(range(1,epochs+1), R18_Ptesting)  
      plt.plot(range(1,epochs+1), R18_Ntraining)  
      plt.plot(range(1,epochs+1), R18_Ntesting)  
      plt.title(f'Result Comparison({model_name})')
      plt.ylabel('Accuracy(%)'), plt.xlabel('Epochs')
      plt.legend(['Train(with pretraining)','Test(with pretraining)', 'Train(w/o pretraining)','Test(w/o pretraining)'])
      plt.show()
# In[]
batch_size=10
epochs_18=10
epochs_50=5
lr=1e-3
loss_func=nn.CrossEntropyLoss()
# In[]
train_dataset=DataLoader(
    RetinopathyLoader("train","train"),
    batch_size=batch_size,
    shuffle=True
    )

test_dataset=DataLoader(
    RetinopathyLoader('test', 'test'),
    batch_size=batch_size
    )
# In[]
model=ResNet18()
model=model.cuda()
optimizer=torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
weight_path='weight/Res18_NotPretrain.pt'
NR18_Pre_training_acc,NR18_Pre_testing_acc,NR18_Pre_best_acc=train(model,weight_path ,loss_func, optimizer, epochs_18, train_dataset, test_dataset)
# In[]
model=Pretrain_ResNet18()
model=model.cuda()
optimizer=torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
weight_path='weight/Res18_Pretrain.pt'
R18_Pre_training_acc,R18_Pre_testing_acc,R18_Pre_best_acc=train(model,weight_path ,loss_func, optimizer, epochs_18, train_dataset, test_dataset)
# In[]
model=ResNet50()
model=model.cuda()
optimizer=torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
weight_path='weight/Res50_Not_Pretrain.pt'
NR50_Pre_training_acc,NR50_Pre_testing_acc,NR50_Pre_best_acc=train(model,weight_path ,loss_func, optimizer,epochs_50 , train_dataset, test_dataset)
# In[]
model=Pretrain_ResNet50()
model=model.cuda()
optimizer=torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
weight_path='weight/Res50_Pretrain.pt'
R50_Pre_training_acc,R50_Pre_testing_acc,R50_Pre_best_acc=train(model,weight_path ,loss_func, optimizer, epochs_50, train_dataset, test_dataset)
# In[]
model_name="Resnet18"
plot(model_name,R18_Pre_training_acc,R18_Pre_testing_acc,NR18_Pre_training_acc,NR18_Pre_testing_acc,epochs_18)
# In[]
model_name="Resnet50"
plot(model_name,R50_Pre_training_acc,R50_Pre_testing_acc,NR50_Pre_training_acc,NR50_Pre_testing_acc,epochs_50)
