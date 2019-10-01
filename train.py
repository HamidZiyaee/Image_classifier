import argparse

parser=argparse.ArgumentParser()

parser.add_argument('-d','--data_dir', help='Directory to data', default='flowers')
parser.add_argument('-s','--save_dir', help='Directory to save checkpoints', default="")
parser.add_argument('-a','--arch', help='Choose pretrained model architecture either vgg19_bn or densenet121', default='vgg19_bn')
parser.add_argument('-l','--learn_rate', help='Set learning rate', type=float)
parser.add_argument('-u','--hidden_units', help='Set hidden units number', type=int)
parser.add_argument('-e','--epochs', help='Set epochs', type=int)
parser.add_argument('-g','--gpu', help='Use GPU for training GPU / CPU', default='gpu')

args=parser.parse_args()

print(args)


import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle = True)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle = True)


#device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
       
    
if args.arch=='densenet121':
    model = models.densenet121(pretrained=True)
    input_units = 1024
elif args.arch == 'vgg19_bn' :
    model = models.vgg19_bn(pretrained=True)
    input_units = 25088
else:
    print('Your input was different than vgg19_bn / densenet121, so by default the vgg19_bn model was used')
    model = models.vgg19_bn(pretrained=True)
    input_units = 25088

    
for param in model.parameters():
    param.requires_grad = False
    
model.classifier=nn.Sequential(nn.Linear(input_units, args.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(args.hidden_units, 102),
                               nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learn_rate)

model.to(device)
epochs = args.epochs
step =0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in trainloaders:
        step +=1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion (logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if step % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for images, labels in validloaders:
                    images, labels = images.to(device), labels.to(device)
                
                    logps = model.forward(images)
                    
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class =ps.topk(1,dim=1)
                    equals=top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Validation loss: {valid_loss/len(validloaders):.3f}.. "
                f"Validation accuracy: {accuracy/len(validloaders):.3f}")
            running_loss = 0
            model.train()

    
checkpoint= {'model': model,
            'state_dict': model.state_dict(),
            'epochs': epochs,
            'optimizer': optimizer.state_dict}

torch.save(checkpoint, args.save_dir + 'checkpoints.pth')


