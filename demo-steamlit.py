import streamlit as st
import time
import pandas as pd

st.write('Hello World')
name = st.text_input("Enter you Name")
st.write(name)
print(name)
progress = st.progress(0)

for i in range(0,100):
    time.sleep(0.1)
    progress.progress(i+1)
st.balloons()

