
import streamlit as st

dataset_name_list = ["macular","fundus"]
for i in dataset_name_list:
    st.header(f"Loss and accuracy analysis for {(i).upper()}" ,)

    st.image(f"results/{i}/cnn_loss.png",caption="cnn loss ")
    st.image(f"results/{i}/cnn_acc.png",caption="cnn acc ")