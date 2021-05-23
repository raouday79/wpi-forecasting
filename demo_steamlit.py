import streamlit as st
import time
import pandas as pd
import forecasting_wpi as fw
import altair as alt
from matplotlib import  pyplot as plt
from plotly import  graph_objs as go
import plotly.express as px

st.header('Wholesale Price Index Forecasting')
st.write("The new series of Wholesale Price Index(WPI) with base 2011-12 is effective from April 2017. Data for WPI(2011-12) has however been provided from April 2012 to March 2017 for the purpose of research and analysis only. Linking factor given for conversion of WPI(2011-12) indices to WPI(2004-05) series should be used from April 2017 onwards.")

temp_df = fw.getData()
col_values = fw.commodity(temp_df)
def showDataFrame(df):
    first_col.dataframe(df)

show_trend = st.button('Show Trend')
colNameSelected = st.selectbox("Please select Commodity",options=col_values)
first_col, second_col = st.beta_columns([1,1])

temp_df_selected = fw.filterData(colNameSelected,temp_df)
showDataFrame(temp_df_selected)

fig = px.line(temp_df_selected, x="TimeSeries", y="WPI", title='WPI Trend of '+colNameSelected)
second_col.plotly_chart(fig)


build_model = st.button('Forecast : Prophet')


def plotForecast(cmp_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cmp_df['TimeSeries'], y=cmp_df['y'],
                             mode='lines',
                             name='Actual'))
    fig.add_trace(go.Scatter(x=cmp_df['TimeSeries'], y=cmp_df['yhat'],
                             mode='lines+markers',
                             name='Forecast'))
    st.plotly_chart(fig)

if build_model:
    cmp_df = fw.modelBuildingStart(colNameSelected,temp_df)
    st.dataframe(cmp_df)
    plotForecast(cmp_df)
#if show_trend:
    # line_alt = alt.Chart(temp_df_selected).mark_line().encode(
    #     x='TimeSeries',
    #     y='WPI'
    # )
    # st.altair_chart(line_alt,use_container_width=True)


#
# st.write('Hello World')
# name = st.text_input("Enter you Name")
# home = st.sidebar.button("Home",)
# about = st.sidebar.button("About",)
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )
# if home:
#     print("Home clicked")
#     st.balloons()
# st.write(name)
# print(name)
# progress = st.progress(0)
#
#
# st.code('for i in range(8): foo()')
#
#
