import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

r"""
# TSDS Neural Network Intuition

Some intuitions about neural networks - based on notes from Andrew Ng's
[Machine Learning Coursera course](https://www.coursera.org/learn/machine-learning).

## Forward Propagation

Lets say we have the single layer neural network shown below, where I have already indicated the weights that will be
used to map between each layer.
"""

st.image(
    image="app/assets/neural_network.jpg",
    caption='Example Single Layer Neural Network with Defined Weights',
    use_column_width=True
)

r"""
Notation:

- $x_i$: the $i^{th}$ feature of an observation, the **input layer**
- $a_j^{l}$: the **activation** of the $j^{th}$ parameter in the $l^{th}$ **hidden layer**
- $\theta_{ij}^{(l)}$: the **weight** used on the $i^{th}$ parameter of the $(l-1)^{th}$ layer to generate the $j^{th}$ parameter in the $l^{th}$ layer
- $h_{\theta}(x)$: our prediction, which is the activation value of the **output layer**

In matrix form, $\theta^{(l)}$ would represent a $s_{(l+1)}\times(s_j+1)$ matrix of the weights used on the parameters in the $l^{th}$ layer - $a^{(l)}$.

The **Sigmoid activation function**, as shown below, is used as our **logistic unit**.
"""

st.latex(r"g(x) = \frac{1}{1+e^{-x}}")

if st.checkbox("See the Sigmoid function"):
    sigmoid_df = pd.DataFrame({"x": np.linspace(-6, 6, 100)})
    sigmoid_df["y"] = sigmoid(sigmoid_df["x"])

    sigmoid_fig = px.line(sigmoid_df, "x", "y")
    sigmoid_fig.add_shape(type="line", x0=-6, y0=0.5, x1=6, y1=0.5)

    st.plotly_chart(sigmoid_fig)

r"""
The **activation** values in the $l^{th}$ layer are given by:
"""

st.latex(r"a^{(l)} = g(\theta^{(l-1)}a^{(l-1)})")
