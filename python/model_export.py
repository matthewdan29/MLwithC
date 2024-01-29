# https://torch.org/hub/pytorch_version_resnet/

import torch
import urllib
from PIL import Image 
from torchvision import transforms

# Download pretrained model
model = torch.hub.load('pytorch/version:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Download an example image from the pytorch websit 
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "dog.jpg")

urllib.request.urlretrieve(url, filename)

# sample execution 
input_image = Image.open(filename)
preprocess = transform.Compose([
    transform.Resize(256), 
    transform.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model 

with torch.no_grad():
    output = model(input_batch)

print(output.squeeze().max(0))

traced_script_module = torch.jit.trace(model, input_batch)

traced_script_module.save("model.pt")
