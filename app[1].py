import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Vehicle Count Prediction", layout="wide")
st.title("Vehicle Count Prediction From Sensor Data")

df = pd.read_csv("data.csv")
X = df.drop(columns=["vehicle_count"])
y = df["vehicle_count"]

categorical = ["sensor_type", "vehicle_type", "registered"]
numeric = [c for c in X.columns if c not in categorical]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                         ("num", "passthrough", numeric)])

rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(random_state=42)
# Voting regressor keeps the project close to the report's ensemble approach.
model = Pipeline([("pre", pre), ("model", rf)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

c1, c2 = st.columns(2)
c1.metric("Mean Squared Error", f"{mse:.2f}")
c2.metric("R² Score", f"{r2:.3f}")

st.subheader("Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, pred)
low, high = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
ax.plot([low, high], [low, high], linestyle="--")
ax.set_xlabel("Actual vehicle count")
ax.set_ylabel("Predicted vehicle count")
st.pyplot(fig)

st.subheader("Sample Dataset")
st.dataframe(df.head(20), use_container_width=True)

# Simple report-style checks
unauthorized = (df["registered"] == "Unregistered").sum()
anomaly = IsolationForest(contamination=0.01, random_state=42).fit_predict(df[["speed", "rash_driving_flag"]])
triple_riding = ((df["vehicle_type"] == "Motorbike") & (df["passenger_count"] > 2)).sum()
st.write({"Unauthorized vehicles": int(unauthorized),
          "Rash-driving anomalies": int((anomaly == -1).sum()),
          "Triple-riding incidents": int(triple_riding)})
