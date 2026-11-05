# =====================================================================
#  TOURISM IN INDIA - Data Analysis & Visualization Dashboard
#  B.Tech Data Analysis & Visualization Project
#  Stack: Python, Pandas, NumPy, Plotly, Streamlit
#  NOTE: The dataset is SIMULATED demo data, NOT official government statistics.
#  Run with:  streamlit run app.py
# =====================================================================

# ---------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Tourism in India Dashboard", page_icon="🇮🇳", layout="wide")

# ---------------------------------------------------------------------
# 2. DATASET LOADING  (with a clearly separated sample-data generator)
# ---------------------------------------------------------------------
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tourism_india.csv")

# ---- SAMPLE DATA GENERATION SECTION (used only if the CSV is missing) ----
# state, region, lat, lon, top attraction, domestic (millions, 2019 base),
# foreign (millions, 2019 base), avg stay (days), hotel occupancy (%), water-transport share
STATE_INFO = [
    ("Rajasthan", "North", 26.91, 75.79, "Amber Fort", 50, 1.8, 3.2, 68, 0.01),
    ("Delhi", "North", 28.61, 77.21, "Red Fort", 35, 2.9, 2.4, 72, 0.01),
    ("Uttar Pradesh", "North", 26.85, 80.95, "Taj Mahal", 200, 3.1, 2.6, 62, 0.02),
    ("Punjab", "North", 31.15, 75.34, "Golden Temple", 22, 0.2, 1.8, 58, 0.01),
    ("Himachal Pradesh", "North", 31.10, 77.17, "Shimla Mall Road", 17, 0.4, 4.1, 64, 0.01),
    ("Uttarakhand", "North", 30.07, 79.02, "Rishikesh", 30, 0.2, 3.5, 60, 0.02),
    ("Jammu & Kashmir", "North", 33.78, 76.58, "Dal Lake", 15, 0.1, 4.0, 63, 0.03),
    ("Kerala", "South", 10.85, 76.27, "Alleppey Backwaters", 18, 1.2, 5.2, 70, 0.12),
    ("Tamil Nadu", "South", 11.13, 78.66, "Meenakshi Temple", 150, 3.5, 2.8, 66, 0.02),
    ("Karnataka", "South", 15.32, 75.71, "Hampi", 110, 0.5, 2.9, 65, 0.02),
    ("Andhra Pradesh", "South", 15.91, 79.74, "Tirupati Balaji", 100, 0.3, 2.2, 60, 0.03),
    ("Telangana", "South", 17.12, 79.21, "Charminar", 60, 0.3, 2.5, 67, 0.01),
    ("West Bengal", "East", 22.99, 87.85, "Victoria Memorial", 70, 1.4, 3.3, 61, 0.05),
    ("Odisha", "East", 20.95, 85.10, "Konark Sun Temple", 17, 0.1, 3.0, 57, 0.03),
    ("Bihar", "East", 25.10, 85.31, "Bodh Gaya", 30, 1.1, 2.1, 52, 0.02),
    ("Maharashtra", "West", 19.75, 75.71, "Gateway of India", 120, 4.4, 3.4, 71, 0.04),
    ("Gujarat", "West", 22.26, 71.19, "Statue of Unity", 45, 0.6, 2.7, 63, 0.02),
    ("Goa", "West", 15.30, 74.12, "Baga Beach", 8, 0.9, 4.8, 75, 0.08),
    ("Madhya Pradesh", "Central", 22.97, 78.66, "Khajuraho Temples", 65, 0.4, 2.5, 59, 0.01),
    ("Chhattisgarh", "Central", 21.28, 81.87, "Chitrakote Falls", 12, 0.05, 2.2, 54, 0.01),
    ("Assam", "North-East", 26.20, 92.94, "Kaziranga National Park", 6, 0.1, 3.6, 56, 0.07),
    ("Sikkim", "North-East", 27.53, 88.51, "Tsomgo Lake", 1.5, 0.06, 4.4, 62, 0.01),
    ("Meghalaya", "North-East", 25.47, 91.37, "Living Root Bridges", 1.2, 0.04, 3.8, 58, 0.01),
]
MODES = ["Air", "Train", "Road", "Water"]


def generate_sample_data(seed=42):
    """Create a simulated, internally consistent tourism dataset (2019-2024)."""
    rng = np.random.default_rng(seed)
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    dom_factor = {2019: 1.00, 2020: 0.35, 2021: 0.50, 2022: 0.85, 2023: 1.10, 2024: 1.20}
    frn_factor = {2019: 1.00, 2020: 0.12, 2021: 0.10, 2022: 0.40, 2023: 0.80, 2024: 1.00}
    occ_factor = {2019: 1.00, 2020: 0.55, 2021: 0.62, 2022: 0.85, 2023: 0.98, 2024: 1.02}
    stay_factor = {2019: 1.00, 2020: 0.85, 2021: 0.88, 2022: 0.95, 2023: 1.00, 2024: 1.03}
    # share of tourists by (Air, Train, Road) before adding water, by region
    region_share = {"North": (0.20, 0.30, 0.50), "South": (0.24, 0.28, 0.48),
                    "East": (0.18, 0.34, 0.48), "West": (0.26, 0.28, 0.46),
                    "Central": (0.12, 0.30, 0.58), "North-East": (0.35, 0.08, 0.57)}
    rows = []
    for state, region, lat, lon, attr, dom, frn, stay, occ, water in STATE_INFO:
        a, t, r = region_share[region]
        for y in years:
            dom_total = dom * 1e6 * dom_factor[y] * rng.normal(1, 0.04)
            frn_total = frn * 1e6 * frn_factor[y] * rng.normal(1, 0.06)
            occ_y = min(occ * occ_factor[y] + rng.normal(0, 1.5), 96)
            stay_y = stay * stay_factor[y] + rng.normal(0, 0.1)
            # domestic transport shares; foreign tourists favour air
            s = np.array([a, t, r]) * (1 - water)
            s = np.append(s, water) * rng.normal(1, 0.05, 4)
            s_dom = s / s.sum()
            s_frn = s * np.array([2.2, 0.6, 0.8, 0.7])
            s_frn = s_frn / s_frn.sum()
            for i, mode in enumerate(MODES):
                d = int(dom_total * s_dom[i])
                f = int(frn_total * s_frn[i])
                # revenue (Crore) = tourists x daily spend x stay; 1 Crore = 1e7 rupees
                revenue = (d * 1200 + f * 9000) * stay_y / 1e7 * rng.normal(1, 0.03)
                rows.append([y, state, region, mode, d, f, round(revenue, 2),
                             round(stay_y + rng.normal(0, 0.05), 2),
                             round(occ_y + rng.normal(0, 0.3), 1), attr, lat, lon])
    cols = ["Year", "State", "Region", "Transport_Mode", "Domestic_Tourists",
            "Foreign_Tourists", "Revenue_Crore", "Avg_Stay_Days", "Hotel_Occupancy",
            "Top_Attraction", "Latitude", "Longitude"]
    return pd.DataFrame(rows, columns=cols)
# ---- END OF SAMPLE DATA GENERATION SECTION ----


@st.cache_data(show_spinner=False)
def load_data():
    """Load the CSV (or generate demo data) and preprocess it with Pandas."""
    if os.path.exists(CSV_FILE):
        raw = pd.read_csv(CSV_FILE)
    else:
        print("tourism_india.csv not found - generating demo data instead.")
        raw = generate_sample_data()

    # -----------------------------------------------------------------
    # 3. DATA PREPROCESSING (Pandas)
    # -----------------------------------------------------------------
    d = raw.drop_duplicates().dropna().copy()
    num_cols = ["Year", "Domestic_Tourists", "Foreign_Tourists", "Revenue_Crore",
                "Avg_Stay_Days", "Hotel_Occupancy", "Latitude", "Longitude"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d["Year"] = d["Year"].astype(int)
    d["Total_Tourists"] = d["Domestic_Tourists"] + d["Foreign_Tourists"]
    return d


df = load_data()

# One row per state with fixed info (region, coordinates, attraction)
STATE_META = df.drop_duplicates("State")[["State", "Region", "Latitude", "Longitude", "Top_Attraction"]]
YEARS = sorted(df["Year"].unique())
REGIONS = ["North", "South", "East", "West", "Central", "North-East"]
METRICS = {"Total Tourists (M)": "Total", "Revenue (₹ Crore)": "Revenue",
           "Hotel Occupancy (%)": "Occupancy", "Average Stay (Days)": "Stay"}


def filter_data(year="All", region="All", transport="All"):
    """Apply the three dashboard filters to the main dataset."""
    d = df
    if year != "All":
        d = d[d["Year"] == int(year)]
    if region != "All":
        d = d[d["Region"] == region]
    if transport != "All":
        d = d[d["Transport_Mode"] == transport]
    return d


def summarize(d, by="State"):
    """Group data by State/Region/Top_Attraction. Tourists are returned in millions."""
    cols = [by, "Domestic", "Foreign", "Total", "Revenue", "Occupancy", "Stay"]
    if d.empty:
        return pd.DataFrame(columns=cols + ["Region", "Latitude", "Longitude", "Top_Attraction"])
    d = d.assign(StayW=d["Avg_Stay_Days"] * d["Total_Tourists"])  # weight stay by tourists
    out = d.groupby(by).agg(Domestic=("Domestic_Tourists", "sum"), Foreign=("Foreign_Tourists", "sum"),
                            Total=("Total_Tourists", "sum"), Revenue=("Revenue_Crore", "sum"),
                            Occupancy=("Hotel_Occupancy", "mean"), StayW=("StayW", "sum")).reset_index()
    out["Stay"] = out["StayW"] / out["Total"]
    out[["Domestic", "Foreign", "Total"]] = out[["Domestic", "Foreign", "Total"]] / 1e6
    out = out.drop(columns="StayW")
    if by == "State":
        out = out.merge(STATE_META, on="State", how="left")
    return out


# ---------------------------------------------------------------------
# 4-5. STREAMLIT PAGE STYLE + HELPER COMPONENTS
# ---------------------------------------------------------------------
KPI_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"]

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background-color: white; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);}
.kpi {background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);}
.kpi .t {font-size: 13px; color: #6b7280;}
.kpi .v {font-size: 26px; font-weight: 700; margin-top: 6px;}
.tile {background-color: #eef2ff; border-radius: 10px; padding: 10px;}
.tile .k {font-size: 12px; color: #6b7280;}
.tile .v {font-weight: 700; font-size: 17px;}
</style>
""", unsafe_allow_html=True)


def show(fig, key):
    """Render a Plotly figure full-width (works on older and newer Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def section_title(text, note=None):
    st.markdown(f"### {text}")
    if note:
        st.caption(note)


def kpi_card(title, value, color):
    st.markdown(f'<div class="kpi" style="border-left:6px solid {color}"><div class="t">{title}</div>'
                f'<div class="v" style="color:{color}">{value}</div></div>', unsafe_allow_html=True)


def style_fig(fig, title, height=380):
    fig.update_layout(title=title, template="plotly_white", height=height,
                      margin=dict(l=45, r=20, t=55, b=45), legend_title_text="")
    return fig


# ---------------------------------------------------------------------
# 6-9. OVERVIEW: KPIs, insights, six charts, scatter plot, India map
# ---------------------------------------------------------------------
def build_overview(year, region, transport):
    d = filter_data(year, region, transport)
    s = summarize(d, "State")

    # ---- KPI values (7. KPI cards) ----
    if d.empty:
        kpis = ["No data"] * 5
    else:
        dom, frn = d["Domestic_Tourists"].sum(), d["Foreign_Tourists"].sum()
        kpis = [f"{dom / 1e6:,.2f} M", f"{frn / 1e6:,.2f} M", f"₹ {d['Revenue_Crore'].sum():,.0f}",
                f"{np.average(d['Avg_Stay_Days'], weights=d['Total_Tourists']):.2f}",
                f"{d['Hotel_Occupancy'].mean():.1f} %"]

    # ---- Key insights (all computed, nothing hard-coded) ----
    if d.empty:
        insights = ["No data for the selected filters."]
        return kpis, insights, {}
    tot = d["Total_Tourists"].sum()
    modes = summarize(d, "Transport_Mode").sort_values("Total", ascending=False)
    regs = summarize(d, "Region").sort_values("Total", ascending=False)
    top = lambda col: s.loc[s[col].idxmax()]
    insights = [
        f"Most visited state: {top('Total').State} ({top('Total').Total:,.2f} M tourists).",
        f"Highest tourism revenue: {top('Revenue').State} (₹ {top('Revenue').Revenue:,.0f} Crore).",
        f"Highest hotel occupancy: {top('Occupancy').State} ({top('Occupancy').Occupancy:.1f}%).",
        f"Longest average stay: {top('Stay').State} ({top('Stay').Stay:.2f} days).",
        f"Most-used transport mode: {modes.iloc[0]['Transport_Mode']} "
        f"({modes.iloc[0]['Total'] * 1e6 / tot * 100:.1f}% of tourists).",
        f"Region with highest tourist volume: {regs.iloc[0]['Region']} ({regs.iloc[0]['Total']:,.2f} M).",
        f"Domestic share: {d['Domestic_Tourists'].sum() / tot * 100:.2f}% | "
        f"Foreign share: {d['Foreign_Tourists'].sum() / tot * 100:.2f}%.",
    ]
    # Year-wise growth uses all years (with Region/Transport filters applied)
    yt = filter_data("All", region, transport).groupby("Year")["Total_Tourists"].sum()
    if year == "All" and len(yt) > 1:
        g = (yt.iloc[-1] / yt.iloc[0] - 1) * 100
        insights.append(f"Total tourists changed by {g:+.1f}% from {yt.index[0]} to {yt.index[-1]}.")
    elif year != "All" and int(year) - 1 in yt.index:
        g = (yt[int(year)] / yt[int(year) - 1] - 1) * 100
        insights.append(f"Total tourists changed by {g:+.1f}% compared with {int(year) - 1}.")

    figs = {}
    # Chart 1 - Domestic vs Foreign
    long = s.melt(id_vars="State", value_vars=["Domestic", "Foreign"], var_name="Type", value_name="Tourists (M)")
    c1 = px.bar(long, x="State", y="Tourists (M)", color="Type", barmode="group",
                color_discrete_map={"Domestic": "#2563eb", "Foreign": "#f59e0b"})
    style_fig(c1, "1. Domestic vs Foreign Tourists by State (millions)").update_xaxes(tickangle=-45)
    figs["c1"] = c1

    # Chart 2 - Revenue by state
    r = s.sort_values("Revenue", ascending=False)
    c2 = px.bar(r, x="State", y="Revenue", color="Revenue", color_continuous_scale="Greens",
                labels={"Revenue": "Revenue (₹ Crore)"})
    style_fig(c2, "2. Tourism Revenue by State (₹ Crore)").update_xaxes(tickangle=-45)
    c2.update_coloraxes(showscale=False)
    figs["c2"] = c2

    # Chart 3 - Hotel occupancy
    o = s.sort_values("Occupancy", ascending=False)
    c3 = px.bar(o, x="State", y="Occupancy", color="Occupancy", color_continuous_scale="Purples",
                labels={"Occupancy": "Hotel Occupancy (%)"})
    style_fig(c3, "3. Average Hotel Occupancy by State (%)").update_xaxes(tickangle=-45)
    c3.update_coloraxes(showscale=False)
    figs["c3"] = c3

    # Chart 4 - Transport share donut
    m = summarize(d, "Transport_Mode")
    c4 = px.pie(m, names="Transport_Mode", values="Total", hole=0.5, color="Transport_Mode",
                color_discrete_map={"Air": "#2563eb", "Train": "#10b981", "Road": "#f59e0b", "Water": "#06b6d4"})
    style_fig(c4, "4. Transport Mode Share of Tourists")
    figs["c4"] = c4

    # Chart 5 - Average stay
    st_ = s.sort_values("Stay", ascending=False)
    c5 = px.bar(st_, x="State", y="Stay", color="Stay", color_continuous_scale="Oranges",
                labels={"Stay": "Avg Stay (Days)"})
    style_fig(c5, "5. Average Tourist Stay by State (days)").update_xaxes(tickangle=-45)
    c5.update_coloraxes(showscale=False)
    figs["c5"] = c5

    # Chart 6 - Top attractions by footfall
    a = summarize(d, "Top_Attraction").sort_values("Total", ascending=False).head(10).iloc[::-1]
    c6 = px.bar(a, x="Total", y="Top_Attraction", orientation="h", color="Total",
                color_continuous_scale="Reds", labels={"Total": "Footfall (M)", "Top_Attraction": ""})
    style_fig(c6, "6. Top Tourist Attractions by Footfall (millions)")
    c6.update_coloraxes(showscale=False)
    figs["c6"] = c6

    # Scatter plot - tourists vs revenue
    sc = px.scatter(s, x="Total", y="Revenue", color="Region", hover_name="State",
                    hover_data={"Total": ":.2f", "Revenue": ":,.0f", "Occupancy": ":.1f", "Region": True},
                    labels={"Total": "Total Tourists (M)", "Revenue": "Revenue (₹ Crore)",
                            "Occupancy": "Hotel Occupancy (%)"})
    sc.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
    title = "Revenue vs Total Tourists"
    if len(s) >= 3:  # add a simple linear trend line + correlation using NumPy
        k, b = np.polyfit(s["Total"], s["Revenue"], 1)
        xs = np.linspace(s["Total"].min(), s["Total"].max(), 20)
        sc.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="Trend",
                                line=dict(dash="dash", color="gray")))
        title += f" (correlation r = {np.corrcoef(s['Total'], s['Revenue'])[0, 1]:.2f})"
    style_fig(sc, title)
    figs["sc"] = sc

    # India map
    mp = px.scatter_geo(s, lat="Latitude", lon="Longitude", size="Revenue", color="Total",
                        hover_name="State", color_continuous_scale="YlOrRd", size_max=38,
                        hover_data={"Latitude": False, "Longitude": False, "Revenue": ":,.0f",
                                    "Domestic": ":.2f", "Foreign": ":.2f", "Occupancy": ":.1f", "Total": ":.2f"},
                        labels={"Revenue": "Revenue (₹ Cr)", "Domestic": "Domestic (M)", "Foreign": "Foreign (M)",
                                "Occupancy": "Hotel Occupancy (%)", "Total": "Total Tourists (M)"})
    mp.update_geos(scope="asia", lataxis_range=[6, 37], lonaxis_range=[67, 98], showcountries=True,
                   countrycolor="#9ca3af", showland=True, landcolor="#f3f4f6", showocean=True,
                   oceancolor="#e0f2fe")
    style_fig(mp, "India Tourism Map (size = revenue, colour = tourists)")
    figs["mp"] = mp
    return kpis, insights, figs


# ---------------------------------------------------------------------
# 10. ANALYSIS: Year trend, Top 5 states, Region comparison
# ---------------------------------------------------------------------
def build_trend(region, transport, trend_metrics):
    t = filter_data("All", region, transport).groupby("Year").agg(
        Domestic=("Domestic_Tourists", "sum"), Foreign=("Foreign_Tourists", "sum"),
        Revenue=("Revenue_Crore", "sum")).reset_index()
    fig_t = make_subplots(specs=[[{"secondary_y": True}]])
    styles = {"Domestic Tourists": ("Domestic", "#2563eb"), "Foreign Tourists": ("Foreign", "#f59e0b")}
    for name in trend_metrics or []:
        if name in styles:
            col, color = styles[name]
            fig_t.add_trace(go.Scatter(x=t["Year"], y=t[col] / 1e6, mode="lines+markers", name=name + " (M)",
                                       line=dict(color=color, width=3)), secondary_y=False)
        elif name == "Revenue":
            fig_t.add_trace(go.Scatter(x=t["Year"], y=t["Revenue"], mode="lines+markers",
                                       name="Revenue (₹ Crore)", line=dict(color="#059669", width=3, dash="dot")),
                            secondary_y=True)
    fig_t.update_yaxes(title_text="Tourists (millions)", secondary_y=False)
    fig_t.update_yaxes(title_text="Revenue (₹ Crore)", secondary_y=True)
    fig_t.update_xaxes(dtick=1)
    style_fig(fig_t, "Tourism Trend 2019–2024", 400)
    return fig_t


def build_top5(year, region, transport, top5_metric):
    col = METRICS[top5_metric]
    s = summarize(filter_data(year, region, transport), "State")
    top5 = s.sort_values(col, ascending=False).head(5).iloc[::-1]
    fig_5 = px.bar(top5, x=col, y="State", orientation="h", color=col, color_continuous_scale="Blues",
                   text=top5[col].round(2), labels={col: top5_metric, "State": ""})
    style_fig(fig_5, f"Top 5 States by {top5_metric}", 340)
    fig_5.update_coloraxes(showscale=False)
    return fig_5


def build_region(year, transport, region_metric):
    reg = summarize(filter_data(year, "All", transport), "Region")
    reg["Region"] = pd.Categorical(reg["Region"], categories=REGIONS, ordered=True)
    reg = reg.sort_values("Region")
    rcol = METRICS[region_metric]
    fig_r = px.bar(reg, x="Region", y=rcol, color="Region", text=reg[rcol].round(2),
                   labels={rcol: region_metric})
    style_fig(fig_r, f"Region-wise {region_metric}", 360).update_layout(showlegend=False)
    table = reg[["Region", "Total", "Revenue", "Occupancy", "Stay"]].round(2)
    table.columns = ["Region", "Tourists (M)", "Revenue (₹ Cr)", "Occupancy (%)", "Avg Stay (Days)"]
    table["Region"] = table["Region"].astype(str)
    return fig_r, table


# ---------------------------------------------------------------------
# 11. STATE EXPLORER
# ---------------------------------------------------------------------
def build_explorer(state, year, region, transport):
    d = filter_data(year, region, transport)
    d = d[d["State"] == state]
    if d.empty or state is None:
        return None, None, None

    s_all = summarize(filter_data(year, region, transport), "State").sort_values("Total", ascending=False)
    rank = int(np.where(s_all["State"].values == state)[0][0]) + 1
    row = summarize(d, "State").iloc[0]
    items = [("State", state), ("Region", row["Region"]), ("Rank (by tourists)", f"{rank} of {len(s_all)}"),
             ("Domestic tourists", f"{row['Domestic']:,.2f} M"), ("Foreign tourists", f"{row['Foreign']:,.2f} M"),
             ("Revenue", f"₹ {row['Revenue']:,.0f} Cr"), ("Average stay", f"{row['Stay']:.2f} days"),
             ("Hotel occupancy", f"{row['Occupancy']:.1f} %"), ("Top attraction", row["Top_Attraction"])]

    # Transport distribution (Transport filter not applied so all modes are visible)
    dm = filter_data(year, region, "All")
    dm = dm[dm["State"] == state]
    mode = summarize(dm, "Transport_Mode")
    f1 = px.pie(mode, names="Transport_Mode", values="Total", hole=0.45, color="Transport_Mode",
                color_discrete_map={"Air": "#2563eb", "Train": "#10b981", "Road": "#f59e0b", "Water": "#06b6d4"})
    style_fig(f1, f"{state}: Transport Distribution", 340)

    # State trend across years
    tr = filter_data("All", region, transport)
    tr = tr[tr["State"] == state].groupby("Year")[["Domestic_Tourists", "Foreign_Tourists"]].sum() / 1e6
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=tr.index, y=tr["Domestic_Tourists"], name="Domestic (M)", marker_color="#2563eb"))
    f2.add_trace(go.Bar(x=tr.index, y=tr["Foreign_Tourists"], name="Foreign (M)", marker_color="#f59e0b"))
    f2.update_layout(barmode="stack")
    f2.update_xaxes(dtick=1)
    style_fig(f2, f"{state}: Tourists per Year (millions)", 340)
    return items, f1, f2


# ---------------------------------------------------------------------
# 5. DASHBOARD LAYOUT  (Streamlit reruns this top-to-bottom on every interaction)
# ---------------------------------------------------------------------
st.markdown("<h1 style='text-align:center;margin:0 0 6px 0'>🇮🇳 Tourism in India – Data Analysis & "
            "Visualization Dashboard</h1>", unsafe_allow_html=True)


# ---- Filters ----
with st.container(border=True):
    f1c, f2c, f3c = st.columns(3)
    year = f1c.selectbox("Year", ["All"] + YEARS, key="year-dd")
    region = f2c.selectbox("Region", ["All"] + REGIONS, key="region-dd")
    transport = f3c.selectbox("Transport", ["All"] + MODES, key="transport-dd")

kpis, insights, figs = build_overview(year, region, transport)

# ---- KPI cards ----
k_titles = ["Domestic Tourists", "Foreign Tourists", "Tourism Revenue (₹ Crore)",
            "Average Stay (Days)", "Hotel Occupancy (%)"]
for col, title, val, color in zip(st.columns(5), k_titles, kpis, KPI_COLORS):
    with col:
        kpi_card(title, val, color)
st.write("")

# ---- Key insights ----
with st.container(border=True):
    section_title("💡 Key Insights", "Calculated automatically from the filtered data")
    st.markdown("\n".join(f"- {i}" for i in insights))

if not figs:           # empty selection -> nothing more to draw
    st.stop()

# ---- Six main charts ----
for left, right in [("c1", "c2"), ("c3", "c4"), ("c5", "c6")]:
    cl, cr = st.columns(2)
    with cl, st.container(border=True):
        show(figs[left], left)
    with cr, st.container(border=True):
        show(figs[right], right)

# ---- Year-wise trend ----
with st.container(border=True):
    section_title("📈 Year-wise Tourism Trend",
                  "Shows all years (Year filter ignored); Region & Transport filters apply")
    tcols = st.columns([1, 1, 1, 4])
    trend_metrics = [name for col, name in zip(tcols, ["Domestic Tourists", "Foreign Tourists", "Revenue"])
                     if col.checkbox(name, value=True, key=f"trend-{name}")]
    show(build_trend(region, transport, trend_metrics), "ch-trend")

# ---- Top 5 states ----
with st.container(border=True):
    section_title("🏆 Top 5 States")
    top5_metric = st.selectbox("Metric", list(METRICS.keys()), key="top5-metric",
                               label_visibility="collapsed")
    show(build_top5(year, region, transport, top5_metric), "ch-top5")

# ---- Region comparison ----
with st.container(border=True):
    section_title("🧭 Region Comparison",
                  "Compares all regions (Region filter ignored); Year & Transport filters apply")
    region_metric = st.selectbox("Metric", list(METRICS.keys()), key="region-metric",
                                 label_visibility="collapsed")
    fig_r, reg_table = build_region(year, transport, region_metric)
    rc1, rc2 = st.columns([1.3, 1])
    with rc1:
        show(fig_r, "ch-region")
    with rc2:
        try:
            st.dataframe(reg_table, hide_index=True, width="stretch")
        except TypeError:
            st.dataframe(reg_table, hide_index=True, use_container_width=True)

# ---- Scatter + Map ----
cl, cr = st.columns(2)
with cl, st.container(border=True):
    show(figs["sc"], "ch-scatter")
with cr, st.container(border=True):
    show(figs["mp"], "ch-map")

# ---- State explorer ----
with st.container(border=True):
    section_title("🔍 State Explorer",
                  "Uses Year, Region & Transport filters (the transport pie ignores the Transport filter)")
    # Only list states inside the selected region (keep the current pick if it is still valid)
    state_options = sorted(filter_data("All", region)["State"].unique())
    if st.session_state.get("state-dd") not in state_options:
        st.session_state["state-dd"] = state_options[0] if state_options else None
    state = st.selectbox("State", state_options, key="state-dd", label_visibility="collapsed")

    items, pie_fig, trend_fig = build_explorer(state, year, region, transport)
    if items is None:
        st.info("No data for this selection.")
    else:
        tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>'
                        for k, v in items)
        st.markdown(f'<div style="display:grid;gap:12px;margin:12px 0;'
                    f'grid-template-columns:repeat(auto-fit,minmax(170px,1fr))">{tiles}</div>',
                    unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            show(pie_fig, "ch-state-transport")
        with e2:
            show(trend_fig, "ch-state-trend")

st.markdown("<div style='text-align:center;font-size:12px;color:#6b7280;margin-top:8px'>B.Tech Data Analysis & "
            "Visualization Project | Built with Python, Pandas, NumPy, Plotly & Streamlit | Simulated demo data"
            "</div>", unsafe_allow_html=True)
