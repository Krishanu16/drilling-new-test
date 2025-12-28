import streamlit as st
import pandas as pd
import numpy as np
import math, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Drilling Optimization", layout="wide")
st.title("🛢️ AI & ML Integrated Drilling Optimization Application")

st.markdown("""
**Final Polished Version**
- Physics-based drilling models  
- Machine learning problem detection  
- Confusion matrix & metrics  
- Real-time streaming simulation  
""")

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_drilling.csv", parse_dates=["timestamp"])

df = load_data()

def compute_mse(WOB, Torque, RPM, ROP, bit_diam=8.5):
    WOB_lbf = WOB * 1000
    Torque_lbf_ft = Torque * 1000
    v = max(ROP/60, 1e-6)
    A = math.pi * bit_diam**2 / 4
    return (WOB_lbf/A) + (120*Torque_lbf_ft*RPM)/(A*v)

df["MSE"] = df.apply(lambda r: compute_mse(
    r["WOB_klbf"], r["Torque_klbf_ft"], r["RPM"], r["ROP_ftph"]), axis=1)
df["BitWear"] = df["MSE"].rolling(30, min_periods=1).mean()

df["Problem"] = (
    (df["BitWear"] > df["BitWear"].quantile(0.97)) |
    (df["ROP_ftph"] < df["ROP_ftph"].quantile(0.03))
).astype(int)

features = ["WOB_klbf","RPM","Torque_klbf_ft","ROP_ftph","MSE","BitWear"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(df[features], df["Problem"])
df["Prediction"] = model.predict(df[features])

st.subheader("📊 Drilling Performance Dashboard")
st.line_chart(df.set_index("timestamp")[["MSE","BitWear"]])

st.subheader("📈 Machine Learning Evaluation")
st.text(classification_report(df["Problem"], df["Prediction"]))

cm = confusion_matrix(df["Problem"], df["Prediction"])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("⚙️ What-if Drilling Analysis")
with st.form("what_if"):
    WOB = st.number_input("WOB (klbf)", value=20.0)
    RPM = st.number_input("RPM", value=120.0)
    Torque = st.number_input("Torque (klbf-ft)", value=10.0)
    ROP = st.number_input("ROP (ft/hr)", value=25.0)
    submit = st.form_submit_button("Analyze")

if submit:
    st.metric("Mechanical Specific Energy", round(compute_mse(WOB,Torque,RPM,ROP),2))

st.subheader("⏱️ Real-Time Drilling Simulation")
if st.button("Start Streaming"):
    placeholder = st.empty()
    for i in range(min(100,len(df))):
        row = df.iloc[i][features].values.reshape(1,-1)
        pred = model.predict(row)[0]
        placeholder.write(f"Time step {i} → Predicted State: {pred}")
        time.sleep(0.05)

st.download_button("Download Results", df.to_csv(index=False), "drilling_results.csv")
