import streamlit as st
from src import config
dataset_name_list = ["macular","fundus"]
for i in dataset_name_list:
    st.header(f"Confusion Matrix for {(i).upper()}" ,)

    images = [
        f"results/{i}/cnn_confustion.png",
        f"results/{i}/rf_confustion.png",
        f"results/{i}/svm_confustion.png"
    ]
    captions = [
        "CNN Confusion Matrix",
        "Randon Forest Confusion Matrix",
        "SVM Confustion Matrix"
    ]
    i = 0
    with st.container():
        for idx,col in enumerate(st.columns(1)):
            col.image(images, width=300,caption = captions)
            i+= 1

