import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Flexão de Vigas", page_icon="📐", layout="wide")

st.title("Flexão de Vigas — Teste Online")
st.write("Se você está vendo isso, o app está rodando.")

x = np.linspace(0, 2, 200)
w_mm = 5*np.sin(np.pi*x/2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=w_mm, mode="lines", name="w(x) (mm)"))
fig.update_layout(xaxis_title="x (m)", yaxis_title="Deslocamento (mm)", height=450)

st.plotly_chart(fig, use_container_width=True)