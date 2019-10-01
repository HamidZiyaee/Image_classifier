from PIL import Image
import numpy as np
import json    
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import argparse


def load_checkpoint(filename):
    
    parser=argparse.ArgumentParser()

    parser.add_argument('-i','--image_dir', help='Path to image', default='flowers/train/1/image_06734.jpg')
    parser.add_argument('-c','--check_dir', help='Directory to checkpoint', default="")
    parser.add_argument('-t','--top_k', help='Choose number of top K', type=int, default='5')
    parser.add_argument('-f','--category_name', help='The mapping full file name for converting categories to real      names', default='cat_to_name.json')
    parser.add_argument('-g','--gpu', help='Use GPU for training gpu / cpu', default='gpu')
    
    args=parser.parse_args()
    print(args)
    
    if args.gpu == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)                   
    checkpoint=torch.load(args.check_dir+filename)

    model=checkpoint['model']
    model.to(device)                   
    model.load_state_dict (checkpoint['state_dict'])
    optimizer= checkpoint['optimizer']
    epochs=checkpoint['epochs']
    model.to('cpu')
    
    return model, epochs, optimizer, args

model, epochs, optimizer, args=load_checkpoint('checkpoints.pth')                    
                    

with open(args.category_name, 'r') as f:
    cat_to_name = json.load(f)

    
def process_image(image):

    image = Image.open(image)
    image= image.resize((224, 224))
    #box=(224, 224, 224, 224)
    #image= image.crop(box)
    
    image = np.array(image)
       
    image = np.divide(image,255)
    mean= np.array([0.485, 0.456, 0.406])
    std= np.array([0.229, 0.224, 0.225])

    image= (image - mean)/ std
    
    image = image.transpose((2 ,0 ,1))
    image= torch.from_numpy(image)
    model.to('cpu')

    return image


def predict(image_path, model, topk):
    model.eval()
    #model.float()
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    
    image = image.type(torch.FloatTensor)
    image=torch.unsqueeze(image,dim=0)
    
    #print(image.shape)
    logps = model.forward(image)
    ps= torch.exp(logps)
    probs, classes = ps.topk(topk, dim=1)
    
    return probs, classes

file_path = args.image_dir

probs, classes = predict(file_path, model, args.top_k)

print(probs)
print(classes)

probs_np=probs.detach().numpy()
max_index = np.argmax(probs_np)
max_probability = probs[max_index]
label = classes[0][max_index]
label_np=label.numpy()
classes_np=classes.numpy()


fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)
#print (z)
image = Image.open(file_path)
ax1.axis('off')
ax1.set_title(cat_to_name[str(label_np+1)])
ax1.imshow(image)

labels = []
for i in classes_np[0]:
    labels.append(cat_to_name[str(i+1)])
    
y_pos = np.arange(args.top_k)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Probability')
ax2.invert_yaxis()
ax2.barh(y_pos, probs_np[0], xerr=0, align='center', color='blue')

plt.show()
