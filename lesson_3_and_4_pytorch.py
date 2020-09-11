import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim



train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)    #Input Layer
        self.fc2 = nn.Linear(64, 64)         #Hiden Layer
        self.fc3 = nn.Linear(64, 64)         #Hiden Layer
        self.fc4 = nn.Linear(64, 10)         #output 10 digits from 0 to 9

    def forward(self, x):
        x = F.relu(self.fc1(x))   # F.relu - activation function Relu
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()

X = torch.rand((28, 28))
X = X.view(-1, 28*28)

output = net(X)
print(output)

# Search Errors
# Optimizer create smaller steps for accuracy

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCH = 3

for epoch in range(EPOCH):
    for data in trainset:
        # data is batch of featursets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        #calculate how wrong we
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0


with torch.no_grad():
    for data in trainset:
        X,y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total, 3))

plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(net(X[3].view(-1, 784))[0]))

#Lesson 4 Done

'''FROM HABR
torch.rand: значения инициализируются из случайного равномерного распределения,
torch.randn: значения инициализируются из случайного нормального распределения,
torch.eye(n): единичная матрица вида n×nn×n,
torch.from_numpy(ndarray): тензор PyTorch на основе ndarray из NumPy
torch.linspace(start, end, steps): 1-D тензор со значениями steps, равномерно распределенными между start и end,
torch.ones : тензор с одними единицами,
torch.zeros_like(other): тензор такой же формы, что и other и с одними нулями,
torch.arange(start, end, step): 1-D тензор со значениями, заполненными из диапазона.

Операции

torch.add(x, y): поэлементное сложение
torch.mm(x, y): умножение матриц (не matmul или dot),
torch.mul(x, y): поэлементное умножение
torch.exp(x): поэлементная экспонента
torch.pow(x, power): поэлементное возведение в степень
torch.sqrt(x): поэлементное возведение в квадрат
torch.sqrt_(x): ситуативное поэлементное возведение в квадрат
torch.sigmoid(x): поэлементная сигмоида
torch.cumprod(x): произведение всех значений
torch.sum(x): сумма всех значений
torch.std(x): стандартное отклонение всех значений
torch.mean(x): среднее всех значений


Модуль torch.nn предоставляет пользователям PyTorch функционал, специфичный для нейронных сетей. 
Один из важнейших его членов — torch.nn.Module, представляющий многоразовый блок операций и связанные с ним (обучаемые) 
параметры, чаще всего используемые в слоях нейронных сетей.
Модули могут содержать иные модули и неявно получать функцию backward() для обратного распространения. 
Пример модуля — torch.nn.Linear(), представляющий линейный (плотный/полносвязный) 
слой (т.e. аффинное преобразование Wx+bWx+b):

'''