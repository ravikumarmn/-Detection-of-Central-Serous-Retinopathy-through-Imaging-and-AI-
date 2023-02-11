import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from models.cnn import SimpleCNN
import torchvision.transforms as transforms
import pickle
from src import config
st.set_page_config(
    page_title="Image_Detection",
    page_icon="👋",
)

st.markdown('<h1 style="color:white;">Image Detection</h1>', unsafe_allow_html=True)

mapping = {1:"CSR",0:"NORMAL"}



data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

data_option = st.selectbox(
  "Choose your dataset.",
  ("FUNDUS","MACULAR")
)
if data_option == "FUNDUS":
  #/home/Ravikumar/freelance/computer vision/CSR_detection/
  model_state_dict_file = "checkpoints/data_fundus_model_cnn_learning_rate_0.001_batch_size_8.pt"
  file = st.file_uploader(
    "Upload the image to be classified U0001F447",
    type=["jpg", "png","jpeg"],
    accept_multiple_files=False,
    key = 'fundus'
  )
  st.set_option('deprecation.showfileUploaderEncoding', False)

elif data_option == "MACULAR":
  model_state_dict_file = "checkpoints/data_macular_model_cnn_learning_rate_0.001_batch_size_8.pt"
  file = st.file_uploader(
    "Upload the image to be classified U0001F447",
    type=["jpg", "png","jpeg"],
    accept_multiple_files=False,
    key = 'macular'
  )
  st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model(model_state_dict_file):
  model = SimpleCNN(config.DATASET_NAME)
  model.load_state_dict(torch.load(model_state_dict_file)["model_state_dict"])
  model.eval()
  return model 


 
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
  prints = f"Detecting {mapping.get(predicted.item())} from  an image"
  score = f"Score : {prob.item():.2f}"

  st.title(prints)
  st.title(score)
  print("Detecting {} from  an image ".format(mapping.get(predicted.item()) ))

