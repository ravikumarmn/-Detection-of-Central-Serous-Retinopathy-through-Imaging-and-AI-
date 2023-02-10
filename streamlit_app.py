import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from models.cnn import SimpleCNN
import torchvision.transforms as transforms
import pickle
 

mapping = {1:"MOCULAR",0:"NORMAL"}
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



model_state_dict_file = "/home/Ravikumar/freelance/computer vision/CSR_detection/checkpoints/model_cnn_learning_rate_0.001_batch_size_8.pt"


st.markdown('<h1 style="color:white;">Image Detection</h1>', unsafe_allow_html=True)
@st.cache(allow_output_mutation=True)
def load_model(model_state_dict_file):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_state_dict_file)["model_state_dict"])
    model.eval()
    return model


    


file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png","jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(batch_t, model):
    with torch.no_grad():
        outputs = model(batch_t)
        prob,predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1)*100,1)
        return prob,predicted

if file is None:
    st.text("Please upload an image file")
else:
  img = Image.open(file)
  batch_t = torch.unsqueeze(data_transform(img), 0)
  model=load_model(model_state_dict_file)
  st.image(img, use_column_width=True)
  prob,predicted = upload_predict(batch_t, model)
  prints = f"The image is classified as {mapping.get(predicted.item())}"
  score = f"Score : {prob.item():.2f}"
  st.title(prints)
  st.title(score)
  print("The image is classified as ",mapping.get(predicted.item()))

