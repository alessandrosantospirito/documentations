import streamlit as st
import os 
from matplotlib import pyplot as plt

st.markdown(
    r"""
    <style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;} 
    .stDeployButton {
            visibility: hidden;
        }
    .stApp {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# preprocessing_mode = st.selectbox("Game Pre-Processing", ["Base-Game", "Four-Images Gray-Scale"])
supermario_version = st.selectbox("Game Pre-Processing", ["SuperMarioBros-v0", "SuperMarioBros-v1", "SuperMarioBros-v2", "SuperMarioBros-v3"]) 

if st.button("Play-Video"):
    video_url = f"http://localhost:4000/images/supermario-rl/{supermario_version}.mp4"

    st.markdown(
        f"""
<video  muted autoplay loop>
      <source src="{video_url}" type="video/mp4" />
</video>
        """, unsafe_allow_html=True
    )
