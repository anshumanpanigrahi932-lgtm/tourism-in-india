# Tourism in India – Data Analysis & Visualization Dashboard (Streamlit)

B.Tech Data Analysis & Visualization project.
Stack: Python, Pandas, NumPy, Plotly, Streamlit.

> **Note:** the dataset is SIMULATED demo data (2019–2024), not official government statistics.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

If `tourism_india.csv` is placed next to `app.py` it is used automatically
(columns: Year, State, Region, Transport_Mode, Domestic_Tourists, Foreign_Tourists,
Revenue_Crore, Avg_Stay_Days, Hotel_Occupancy, Top_Attraction, Latitude, Longitude).
Otherwise the built-in demo data generator is used.

## What changed vs. the Dash version
Only the web framework: Dash (`app.layout` + `@app.callback`) became Streamlit
(script reruns on every widget change). The data generation, preprocessing, filters,
KPIs, insights, all charts, tables and the State Explorer are the same.
