"""
Cars Assistant
Extends my Cars EDA project with:
  1. An interactive EDA dashboard
  2. A Random Forest price predictor
  3. An "Ask AI" assistant powered by the Google Gemini API
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---------- CONFIG: edit these to match your dataset ----------
DATA_PATH = "data/cars.csv"
PRICE, YEAR, KM = "Selling_Price", "Year", "Kms_Driven"
FUEL = "Fuel_Type"
SHOWROOM = "Present_Price"   # showroom price of the car when new (in lakhs)
OWNER = "Owner"              # number of previous owners
CATEGORICAL = ["Fuel_Type", "Seller_Type", "Transmission"]
NUMERIC = [YEAR, KM, SHOWROOM, OWNER]
GEMINI_MODEL = "gemini-2.5-flash"  # check Google AI Studio for the current model name
# --------------------------------------------------------------

st.set_page_config(page_title="Cars Assistant", page_icon="🚗", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates().dropna()


@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL)
    y = df[PRICE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, list(X.columns), score


def ask_gemini(question, df, api_key):
    client = genai.Client(api_key=api_key)
    context = (
        f"Columns: {list(df.columns)}\n"
        f"Rows: {len(df)}\n"
        f"Sample rows:\n{df.head(5).to_string()}\n"
        f"Statistics:\n{df.describe(include='all').to_string()}\n"
        f"Average {PRICE} by {FUEL}:\n{df.groupby(FUEL)[PRICE].mean().round(0).to_string()}"
    )
    prompt = (
        "You are a friendly data analyst. Answer the question using ONLY the "
        "dataset summary below. Keep the answer short and clear.\n\n"
        f"{context}\n\nQuestion: {question}"
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found. Place your CSV at `{DATA_PATH}`.")
    st.stop()

model, model_columns, r2 = train_model(df)

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
year_min, year_max = int(df[YEAR].min()), int(df[YEAR].max())
year_range = st.sidebar.slider("Year", year_min, year_max, (year_min, year_max))
fuel_options = sorted(df[FUEL].unique())
fuels = st.sidebar.multiselect("Fuel type", fuel_options, default=fuel_options)
filtered = df[df[YEAR].between(*year_range) & df[FUEL].isin(fuels)]

st.title("🚗 Cars Assistant")
st.caption("EDA dashboard + ML price predictor + AI chat, built on my Cars EDA project")

if filtered.empty:
    st.warning("No cars match these filters. Adjust the sidebar.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💰 Price Predictor", "🤖 Ask AI"])

# ---------------- Tab 1: Dashboard ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Cars", f"{len(filtered):,}")
    c2.metric("Average price", f"₹{filtered[PRICE].mean():.2f} lakh")
    c3.metric("Average km driven", f"{filtered[KM].mean():,.0f}")

    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(filtered, x=PRICE, nbins=40, title="Selling price distribution (lakhs)"),
        use_container_width=True,
    )
    avg_fuel = filtered.groupby(FUEL)[PRICE].mean().reset_index()
    right.plotly_chart(
        px.bar(avg_fuel, x=FUEL, y=PRICE, title="Average selling price by fuel type (lakhs)"),
        use_container_width=True,
    )
    left.plotly_chart(
        px.scatter(filtered, x=KM, y=PRICE, color=FUEL, title="Selling price vs km driven"),
        use_container_width=True,
    )
    corr = filtered.select_dtypes("number").corr()
    right.plotly_chart(
        px.imshow(corr, text_auto=".2f", title="Correlation heatmap"),
        use_container_width=True,
    )

# ---------------- Tab 2: Price Predictor ----------------
with tab2:
    st.subheader("Estimate a car's price")
    st.caption(f"Model: Random Forest | Accuracy (R² on test data): {r2:.2f}")

    a, b = st.columns(2)
    year = a.number_input("Year", year_min, year_max, max(year_min, year_max - 5))
    km = b.number_input("Km driven", 0, 500000, 40000, step=1000)
    showroom = a.number_input("Showroom price when new (lakhs)", 0.1, 100.0, 7.0, step=0.5)
    owner = b.number_input("Previous owners", 0, 3, 0)
    inputs = {YEAR: year, KM: km, SHOWROOM: showroom, OWNER: owner}
    for i, col in enumerate(CATEGORICAL):
        widget = a if i % 2 == 0 else b
        inputs[col] = widget.selectbox(
            col.replace("_", " ").title(), sorted(df[col].unique())
        )

    if st.button("Predict price"):
        row = pd.get_dummies(pd.DataFrame([inputs]), columns=CATEGORICAL)
        row = row.reindex(columns=model_columns, fill_value=0)
        st.success(f"Estimated selling price: ₹{model.predict(row)[0]:.2f} lakh")

# ---------------- Tab 3: Ask AI ----------------
with tab3:
    st.subheader("Ask questions about the dataset")
    api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input(
        "Gemini API key", type="password"
    )
    question = st.text_input(
        "Your question", placeholder="Which fuel type has the highest average selling price?"
    )
    if st.button("Ask") and question:
        if not api_key:
            st.warning("Add your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    st.write(ask_gemini(question, df, api_key))
                except Exception as e:
                    st.error(f"AI request failed: {e}")
