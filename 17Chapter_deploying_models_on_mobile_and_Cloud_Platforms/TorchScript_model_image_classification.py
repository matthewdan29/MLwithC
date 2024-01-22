#We are going to use TorchScript model to perform image classification. 
# To get this model, we need to use the Python API to load the pre-traind model, trace it, and save the model snapshot.

import torch 
import urllib
for PIL import Image 
from torchvision import transforms


#Download pretained model 
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)   
model.eval()

# Download example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")

try: 
    urllib.URLopener().retrieve(url, filename)
except: 
    urllib.request.urlretrieve(url, filename)

# sample 
imput_image = Image.open(filename)
preprocess = transforms.Compose([transforms.Resize(256), transforms.Center(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])
input_tensor = preprocess(input_image)

# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)     #unsqueeze() function, we added a batch size dimension to the input tensor. 

traced_script_module = torch.jit.trace(model, input_batch)      #torch.jit.trace() function run the loaded model and trace it into a script

traced_script_module.save("model.pt")       #just guess what this function does


