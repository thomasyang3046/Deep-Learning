import numpy as np
import matplotlib.pyplot as plt
import math
# In[]
def generate_linear(n=100):
    pts=np.random.uniform(0,1,(n,2))
    inputs=[]
    labels=[]
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance=(pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)
# In[]
def generate_XOR_easy():
    inputs=[]
    labels=[]
    
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)
# In[]
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
# In[]
def derivative_sigmoid(x):
    return np.multiply(x,1.0-x)
# In[]
def mse_loss(GT,y):
    return (GT-y)**2
# In[]
def derivative_mse_loss(GT,y):
    return -2*(GT-y)
# In[]
class TwoLayer_Net:
    def __init__(self,epoch=1500,lr=0.001,hidden_unit=5):
        self.epoch=epoch
        self.lr=lr
        self.hidden_unit=hidden_unit
        self.loss=[]
        self.record=[]
        
        self.X=np.zeros((2,1))
        self.W=[np.random.rand(hidden_unit,2),np.random.rand(hidden_unit,hidden_unit),np.random.rand(1,hidden_unit)]
        self.z=[np.zeros((hidden_unit,1)),np.zeros((hidden_unit,1)),np.zeros((1,1))]
        self.a=[np.zeros((hidden_unit,1)),np.zeros((hidden_unit,1)),np.zeros((1,1))]
        self.update_W=[np.zeros((hidden_unit,2)),np.zeros((hidden_unit,hidden_unit)),np.zeros((1,hidden_unit))]
    def forward(self,inputs):
        #Input Layer
        self.X=inputs
        #First Layer
        self.z[0]=np.matmul(self.W[0],self.X)
        self.a[0]=sigmoid(self.z[0])
        #Second Layer
        self.z[1]=np.matmul(self.W[1],self.a[0])
        self.a[1]=sigmoid(self.z[1])
        #Output Layer
        self.z[2]=np.matmul(self.W[2],self.a[1])
        self.a[2]=sigmoid(self.z[2])
        
        return self.a[2]
    def backward(self,GT,y):
        #算完L_W2
        gradL_a2=derivative_mse_loss(GT,y)
        gradL_z2=gradL_a2*derivative_sigmoid(self.a[2])#activation func
        gradL_W2=gradL_z2*self.a[1].reshape(1,self.hidden_unit)
        #算完L_W1
        gradz2_a1=self.W[2]#1*3
        grada1_z1=derivative_sigmoid(self.a[1]).T#1*3 activation func
        gradL_z1=gradL_z2*gradz2_a1*grada1_z1#1*3
        gradL_W1=np.zeros((hidden_unit,hidden_unit))#3*3
        for i in range(hidden_unit):
            gradL_W1[i,:]=gradL_z1[0,i]*self.a[0].T
        #算完L_W0
        gradz1_a0=self.W[1]#3*3
        grada0_z0=derivative_sigmoid(self.a[0]).T#1*3
        gradL_z0=np.matmul(gradL_z1,gradz1_a0)*grada0_z0
        gradL_W0=np.zeros((hidden_unit,2))
        for i in range(hidden_unit):
            for j in range(2):
                gradL_W0[i,j]=self.X[j,0]*gradL_z0[0,i]
        self.update_W[0]= gradL_W0
        self.update_W[1]= gradL_W1
        self.update_W[2]= gradL_W2
        
    def update(self):
        for i in range(3):
            self.W[i]-=self.lr*self.update_W[i]        
        
    def train(self,inputs,labels):
        for epoch in range(self.epoch):
            predict=np.zeros((labels.shape[0],1))
            for i in range(inputs.shape[0]):
                predict[i]=self.forward(inputs[i].reshape(2,1))
                self.backward(labels[i].reshape(1,1),predict[i].reshape(1,1))
                self.update()
            self.loss.append(np.mean(mse_loss(predict,labels)))
            self.record.append(epoch)    
            if (epoch+1) % 500==0:
                print('Epochs {} loss : {}'.format(epoch+1,self.loss[-1]))

    def testing(self,inputs,labels,prediction,pred_y):
        error=0
        for i in range(inputs.shape[0]):
            predict=self.forward(inputs[i].reshape(2,1)) 
            prediction.append(predict[0,0])#預測
            pred_y.append(np.round(predict[0,0]))
            error+=abs(np.round(predict[0,0])-labels[i,0])
        error/=inputs.shape[0]
        print('accuracy: %.2f' % ((1 - error)*100) + '%')
        print('prediction:')
        for i in range(inputs.shape[0]):
            print('use {} data prediction： {} '.format(i+1,prediction[i]))

    def show_loss(self):
        plt.plot(range(len(self.record)), self.loss, 'b-', label='Training_loss')
        plt.title('Training loss')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.legend(['Train MSE loss'],loc='upper right')
        plt.show()
# In[]
def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
# In[]
#x,y=generate_linear(n=100)
x,y=generate_XOR_easy()
hidden_unit=3
epoch=5000
lr=0.1 #XOR 0.1 linear 0.1
Model=TwoLayer_Net(epoch=epoch,lr=lr,hidden_unit=hidden_unit)
Model.train(x,y)
Model.show_loss()
prediction=[]
pred_y=[]
Model.testing(x,y,prediction,pred_y)
show_result(x,y,pred_y)