import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math
import random 

parser = argparse.ArgumentParser(description='plotting')
parser.add_argument('--epochs', default=10000, type=int, help='number of epochs for training')
parser.add_argument('--attack', '-a', action='store_true', help='attack')
parser.add_argument('--class2', default=40, type=int, help='number of epochs for training')
parser.add_argument('--inner', default=10, type=int, help='number of epochs for training')
parser.add_argument('--outer', default=80, type=int, help='number of epochs for training')
parser.add_argument('--gap', default=0.001, type=float, help='number of epochs for training')
parser.add_argument('--enlarge_gap', default=0.0, type=float, help='gap for plotting')
parser.add_argument('--save_path', type = int, help = 'plot number - REMOVE THIS ARGUMENT')

args = parser.parse_args()

device = 'cpu'
r = 0.1
def plot_grid(net, ax): 
    XX, YY = np.meshgrid(np.linspace(-1.0, 1.0, 200), np.linspace(-1.0, 1.0, 200))
    X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
    y0 = F.softmax(net(X0.to(device)))
    ZZ = (y0[:,0] - y0[:,1]).cpu().resize(200,200).data.numpy()
    

    ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-1,1,3))
    ax.scatter(enlarged.cpu().numpy()[:,0], enlarged.cpu().numpy()[:,1], c=y.cpu().numpy(), cmap="coolwarm", s=40)
    ax.axis("equal")
    ax.axis([-1,1,-1,1])


criterion1 = nn.KLDivLoss()
criterion2 = nn.CrossEntropyLoss()


inner_r = 0.4
outer_r = 0.95
class_1_outside = np.zeros((args.outer,2))
class_1_inside = np.zeros((args.inner, 2))
class_2_inside = np.zeros((args.class2, 2))

theta = 0
for i in range(args.class2):
    class_2_inside[i, :] = [(inner_r) * (math.sin(theta + (i * 2 * math.pi/args.class2))),(inner_r) * (math.cos(theta + (i * 2 * math.pi/args.class2)))]


for i in range(args.outer):  
    class_1_outside[i, :] = [outer_r * (math.sin(theta + (i * 2 * math.pi/args.outer))),outer_r * (math.cos(theta+(i * 2 * math.pi/args.outer)))]

for i in range(args.inner):
    class_1_inside[i, :] = [(inner_r + args.gap)* (math.sin(theta + (i * 2 * math.pi/args.inner) + math.pi/args.class2 )),(inner_r + args.gap) * (math.cos(theta + (i * 2 * math.pi/args.inner) + math.pi/args.class2))]

labels1 = np.ones(args.inner + args.outer)
labels2 = 0*np.ones(args.class2)


y = torch.Tensor(np.concatenate((labels2,labels1))).long().to(device)
X = torch.Tensor(np.concatenate((class_2_inside, np.concatenate((class_1_outside,class_1_inside))))).to(device)

#for visual purposes, we can enlarge the gap in the plot by passing a positive float for args.enlarge_gap
enlarge_X = np.zeros((args.class2, 2))
for i in range(args.class2):
    enlarge_X[i, :] = [(inner_r + args.enlarge_gap) * (math.sin(theta + (i * 2 * math.pi/args.class2))),(inner_r + args.enlarge_gap) * (math.cos(theta + (i * 2 * math.pi/args.class2)))]

enlarged = torch.Tensor(np.concatenate((enlarge_X, np.concatenate((class_1_outside,class_1_inside))))).to(device)

config = {
    'epsilon': 0.07,
    'num_steps': 10,
    'step_size': 0.015 ,
    'random_start': True,
    'loss_func': 'xent',
}


net = nn.Sequential(
nn.Linear(2,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,100),
nn.ReLU(),
nn.Linear(100,2)
)

net = net.to(device)

opt = optim.Adam(net.parameters(), lr=1e-3)

for i in range(args.epochs):
    out = net(Variable(X))
    l = nn.CrossEntropyLoss()(out, y)
    opt.zero_grad()
    l.backward()
    opt.step()

print("training accuracy: \t", torch.argmax(out, dim = 1).eq(Variable(y)).sum().item()/float(args.outer + args.inner + args.class2))


fig, ax = plt.subplots(figsize=(8,8))
plot_grid(net, ax)
plt.axis('on')
plt.show()




