import streamlit as st

dataset_name_list = ["macular","fundus"]
for i in dataset_name_list:
    st.header(f"Performance Analysis for {(i).upper()}" ,)

    st.image(f"results/{i}/performance_analysis.png",caption = "performance table")