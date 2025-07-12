import numpy as np
import scipy.stats as stats
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime

def black_scholes(S, K, T, r, σ, option_type="call"):
    """Return Black-Scholes price for a call or put."""
    d1 = (np.log(S/K) + (r + 0.5*σ**2)*T) / (σ * np.sqrt(T))
    d2 = d1 - σ * np.sqrt(T)
    if option_type == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")
st.title(" Gauri's Black-Scholes Option Pricing Dashboard")
st.markdown(
    """
    A few of the features:
    1. **Calculate** the price of a European _call_ and _put_ option
       using the classic Black-Scholes formula.
    2. **Explore** how those prices change when the underlying stock price
       and volatility change, using heatmaps.
    3. **Log** your single-point calculation.
    """
)


st.header(" Single-Point Price Calculator")
st.write(
    "Enter the five inputs below to compute the current value "
    "of a call and a put option"
)

st.write("**PS - What is a Strike Price?**  \n"
         "The **strike price (K)** is the agreed price at which you can exercise "
         "your option = buying the underlying asset for a call or selling it for a put.")

S = st.number_input(" Stock Price (S)",         value=100.0, step=1.0)
K = st.number_input(" Strike Price (K)",        value=100.0, step=1.0)
T = st.number_input(" Time to Expiry (years)",  value=0.5,   step=0.01)
r = st.number_input(" Risk-free Rate (annual)", value=0.01,  step=0.001)
σ = st.number_input(" Volatility (annual)",     value=0.20,  step=0.01)

if st.button("Calculate"):
    c = black_scholes(S, K, T, r, σ, "call")
    p = black_scholes(S, K, T, r, σ, "put")
    st.markdown(f"**Call Price:** {c:.4f}  \n"
                f"**Put Price:**  {p:.4f}")
    # Store for later
    st.session_state.last_c = c
    st.session_state.last_p = p


st.header(" Interactive Heatmaps")
st.write(
    "Watch how the _call_ and _put_ prices vary when you “shock” the underlying "
    "stock price and volatility up or down.  \n"
    "- **Horizontal axis**: shocked stock price  \n"
    "- **Vertical axis**: shocked volatility  \n"
    "- **Color**: option price (blue = low, white = mid, red = high)"
)

# Display last computed prices with definitions
if "last_c" in st.session_state:
    st.subheader("Last Calculated Option Prices & Definitions")
    st.write(f"**Call Price:** {st.session_state.last_c:.4f}  \n"
             "A **call option** gives you the right (but not the obligation) "
             "to **buy** the underlying asset at the strike price. "
             "This call price is what you pay today for that right.")
    st.write(f"**Put Price:** {st.session_state.last_p:.4f}  \n"
             "A **put option** gives you the right (but not the obligation) "
             "to **sell** the underlying asset at the strike price. "
             "This put price is what you pay today for that right.")

# Build the shock grid
n = 50
S_shocks = np.linspace(0.8 * S, 1.2 * S, n)
σ_shocks = np.linspace(0.10,    0.30,    n)

call_matrix = np.zeros((n, n))
put_matrix  = np.zeros((n, n))

for i, Si in enumerate(S_shocks):
    for j, σj in enumerate(σ_shocks):
        call_matrix[j, i] = black_scholes(Si, K, T, r, σj, "call")
        put_matrix[j, i]  = black_scholes(Si, K, T, r, σj, "put")

# Render heatmaps side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Call Price Heatmap")
    fig, ax = plt.subplots()
    cax = ax.imshow(
        call_matrix,
        extent=(S_shocks[0], S_shocks[-1], σ_shocks[0], σ_shocks[-1]),
        origin="lower",
        aspect="auto",
        cmap="bwr"
    )
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.set_title("Call Price")
    fig.colorbar(cax, label="Price")
    st.pyplot(fig)

with col2:
    st.subheader(" Put Price Heatmap")
    fig, ax = plt.subplots()
    cax = ax.imshow(
        put_matrix,
        extent=(S_shocks[0], S_shocks[-1], σ_shocks[0], σ_shocks[-1]),
        origin="lower",
        aspect="auto",
        cmap="bwr"
    )
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.set_title("Put Price")
    fig.colorbar(cax, label="Price")
    st.pyplot(fig)


st.header(" Log Your Calculation")
st.write(
    "If you’d like to keep a record of your last single point calculation, "
    "click below to append it to **bs_runs.csv**.(happy to share :)"
)

def log_run(inputs, outputs):
    with open("bs_runs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow(), *inputs, *outputs])

if st.button("Log This Run"):
    if not hasattr(st.session_state, "last_c"):
        st.error("Please **Calculate** first before logging.")
    else:
        inputs  = [S, K, T, r, σ]
        outputs = [st.session_state.last_c, st.session_state.last_p]
        log_run(inputs, outputs)
        st.success("Logged to bs_runs.csv")
