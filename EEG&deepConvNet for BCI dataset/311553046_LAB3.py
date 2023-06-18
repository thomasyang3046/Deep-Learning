from dataloader import read_bci_data
from torch import Tensor, cuda, no_grad
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
# In[]
class EEGNet(nn.Module):
    def __init__(self,activation=nn.ELU(alpha=1.0)):
        super(EEGNet,self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1, 51),stride=(1, 1),padding=(0, 25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        self.depthwiseConv=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(2, 1),stride=(1, 1),groups=16 ,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(0.25)#drop 25% neuron
            )
        self.separableConv=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1, 15),stride=(1, 1),padding=(0, 7) ,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(0.25)#drop 25% neuron
            )
        self.classify=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
            )
    def forward(self,inputs):
        output=self.first_conv(inputs)
        output=self.depthwiseConv(output)
        output=self.separableConv(output)
        output=self.classify(output)
        return output   
# In[]
class DeepConvNet(nn.Module):
    def __init__(self,activation=nn.ELU(alpha=1.0)):
        super(DeepConvNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,25,kernel_size=(1,5)),
            nn.Conv2d(25,25,kernel_size=(2,1)),
            nn.BatchNorm2d(25),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(25,50,kernel_size=(1,5)),
            nn.BatchNorm2d(50),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
            )
        self.conv3=nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,5)),
            nn.BatchNorm2d(100),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
            )
        self.conv4=nn.Sequential(
            nn.Conv2d(100,200,kernel_size=(1,5)),
            nn.BatchNorm2d(200),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
            )
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(8600,2)
            )
    def forward(self,inputs):
        output=self.conv1(inputs)
        output=self.conv2(output)
        output=self.conv3(output)
        output=self.conv4(output)
        output=self.fc(output)
        return output
# In[]
def train(model,weight_path ,loss_func, optimizer, epochs, train_dataset, test_dataset):
    model=model.cuda()
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, len(test_dataset))
    training_acc=[]
    testing_acc=[]
    best_acc=0.0
    for epoch in range(epochs):
        all_train_correct=0
        # Train model
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # 1.Define variables
            inputs=inputs.cuda()
            labels=labels.cuda().long()
            # 2.Forward propagation
            outputs=model.forward(inputs=inputs)
            # 3.Clear gradients
            optimizer.zero_grad()
            # 4.Calculate softmax and cross entropy loss
            loss=loss_func(outputs,labels).cuda()
            # 5. Calculate gradients
            loss.backward()
            # 6. Update parameters
            optimizer.step()
            # 7.Total correct predictions
            pred = torch.max(outputs.data, 1)[1]
            all_train_correct += (pred == labels).long().sum()
            torch.cuda.empty_cache()
        training_acc.append((100*all_train_correct/len(train_dataset)).cpu())
        # Test model
        model.eval()
        all_test_correct=0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs=inputs.cuda()
                labels=labels.cuda().long()
                outputs=model.forward(inputs=inputs)
                pred = torch.max(outputs.data, 1)[1]
                all_test_correct += (pred == labels).long().sum()
            now_testing_acc=(100* all_test_correct/len(test_dataset))
            testing_acc.append(now_testing_acc.cpu())
            if now_testing_acc>best_acc:
                best_acc=now_testing_acc
                #torch.save(model.state_dict(),weight_path)
    return training_acc,testing_acc, best_acc
def plot(model_name,ELU_training,ELU_testing,ReLu_training,ReLu_testing,LReLu_training,LReLu_testing,epochs):
      plt.plot(range(epochs), ReLu_training) 
      plt.plot(range(epochs), ReLu_testing)  
      plt.plot(range(epochs), ELU_training)  
      plt.plot(range(epochs), ELU_testing)  
      plt.plot(range(epochs), LReLu_training)  
      plt.plot(range(epochs), LReLu_testing)  
      plt.title(f'Activation function comparison ({model_name})')
      plt.ylabel('Accuracy (%)'), plt.xlabel('Epoch')
      plt.legend(['ReLU_train', 'ReLU_test', 'ELU_train', 'ELU_test', 'LeakyReLU_train', 'LeakyReLU_test'])
      plt.show()
# In[]
train_data, train_label, test_data, test_label = read_bci_data()
train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))
# In[]
batch_size=64
epochs=300
lr=0.001111
loss_func=nn.CrossEntropyLoss()
# In[]
#EEGNet with ELU
model=EEGNet(activation=nn.ELU(alpha=1.0))
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
weight_path='weight/EEG_ELU.pt'
EEG_ELU_training_acc,EEG_ELU_testing_acc, Ebest_ELU_acc=train(model,weight_path, loss_func, optimizer, epochs, train_dataset, test_dataset)

#EEGNet with ReLu
model=EEGNet(activation=nn.ReLU())
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
weight_path='weight/EEG_ReLU.pt'
EEG_ReLu_training_acc,EEG_ReLU_testing_acc, Ebest_ReLu_acc=train(model,weight_path, loss_func, optimizer, epochs, train_dataset, test_dataset)
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))
#EEGNet with LeakyReLu
model=EEGNet(activation=nn.LeakyReLU())
optimizer=torch.optim.Adam(model.parameters(), lr=lr)

weight_path='weight/EEG_LeakyReLU.pt'
EEG_LReLU_training_acc,EEG_LReLU_testing_acc, Ebest_LReLU_acc=train(model,weight_path, loss_func, optimizer, epochs, train_dataset, test_dataset)
model_name='EEGNet'
plot(model_name,EEG_ELU_training_acc,EEG_ELU_testing_acc,EEG_ReLu_training_acc,EEG_ReLU_testing_acc,EEG_LReLU_training_acc,EEG_LReLU_testing_acc,epochs)
# In[]
#DeepConvNet with ELU
model=DeepConvNet(activation=nn.ELU(alpha=1.0))
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
weight_path='weight/DeepConvNet_ELU.pt'
DCN_ELU_training_acc,DCN_ELU_testing_acc, Dbest_ELU_acc=train(model,weight_path, loss_func, optimizer, epochs, train_dataset, test_dataset)

#DeepConvNet with ReLu
model=DeepConvNet(activation=nn.ReLU())
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
weight_path='weight/DeepConvNet_ReLU.pt'
DCN_ReLu_training_acc,DCN_ReLU_testing_acc, Dbest_ReLu_acc=train(model,weight_path, loss_func, optimizer, epochs, train_dataset, test_dataset)
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))
#DeepConvNet with LeakyReLu
model=DeepConvNet(activation=nn.LeakyReLU())
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
weight_path='weight/DeepConvNet_LeakyReLU.pt'
DCN_LReLU_training_acc,DCN_LReLU_testing_acc, Dbest_LReLU_acc=train(model,weight_path, loss_func, optimizer, epochs, train_dataset, test_dataset)

model_name='DeepConvNet'
plot(model_name,DCN_ELU_training_acc,DCN_ELU_testing_acc,DCN_ReLu_training_acc,DCN_ReLU_testing_acc,DCN_LReLU_training_acc,DCN_LReLU_testing_acc,epochs)
# In[]
print(f'EEGNet_ELU best acc：{Ebest_ELU_acc}')
print(f'EEGNet_ReLU best acc：{Ebest_ReLu_acc}')
print(f'EEGNet_LeakyReLU best acc：{Ebest_LReLU_acc}')
print(f'DeepConvNet_ELU best acc：{Dbest_ELU_acc}')
print(f'DeepConvNet_ReLU best acc：{Dbest_ReLu_acc}')
print(f'DeepConvNet_LeakyReLU best acc：{Dbest_LReLU_acc}')



