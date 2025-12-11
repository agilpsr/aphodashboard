import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import re
import urllib.parse
import folium
from streamlit_folium import st_folium
import xlsxwriter
from PIL import Image
import base64

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="APHO Tiruchirappalli Dashboard", layout="wide")

# --- STAFF NAME MAPPING & CONFIG (unchanged) ---
STAFF_NAMES = { ... }  # your existing dict

SECTION_CONFIG = { ... }  # your existing config

# --- ALL HELPER FUNCTIONS (unchanged) ---
# Keep all your existing helper functions (to_excel, load_kobo_data, plot_metric_bar, etc.)
# ... [paste all unchanged functions here] ...

# --- PASSWORD & HOME PAGE (unchanged except small tweak) ---
def check_password_on_home():
    # ... your existing password logic ...
    pass

def render_home_page():
    st.markdown("""
        <style>
        .stApp {background-image: none !important; background-color: white !important;}
        .main .block-container {background-color: transparent !important; border-radius: 0px !important; box-shadow: none !important;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>AIRPORT HEALTH ORGANISATION</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1E3A8A;'>TIRUCHIRAPPALLI INTERNATIONAL AIRPORT</h3>", unsafe_allow_html=True)
    st.divider()

    # Main Tabs
    tab_peri, tab_intra, tab_flights = st.tabs(["Outside Field (Peri)", "Inside Field (Intra)", "International Flights"])

    # === PERI TAB ===
    with tab_peri:
        st.session_state.current_tab = "peri"
        render_dashboard('peri')

    # === INTRA TAB ===
    with tab_intra:
        st.session_state.current_tab = "intra"
        render_dashboard('intra')

    # === FLIGHTS TAB ===
    with tab_flights:
        st.session_state.current_tab = "flights"
        render_dashboard('flights')


# --- MODIFIED render_dashboard() with per-tab sidebar ---
def render_dashboard(selected_key):
    st.markdown("""
        <style>
            .main .block-container {
                background-color: rgba(255, 255, 255, 0.90);
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    current_config = SECTION_CONFIG[selected_key]
    st.title(current_config['title'])

    # Clear previous sidebar filters when switching tabs
    if st.session_state.get("last_tab") != selected_key:
        for key in list(st.session_state.keys()):
            if key.startswith(("start_date_", "end_date_", "zone_filter_", "subzone_filter_", "personnel_filter_", "deputy_filter_")):
                del st.session_state[key]
        st.session_state.last_tab = selected_key

    # Load data
    with st.spinner('Fetching Surveillance data...'):
        df = load_kobo_data(current_config['surv_url'])
    if df.empty:
        st.info("No data found or error loading Kobo data.")
        return

    # ===================================================================
    # FLIGHTS SECTION – Enhanced with Deputy + Duty Personnel + Reset
    # ===================================================================
    if selected_key == 'flights':
        st.header("International Flights Screening Data Summary")

        # Column names (adjust if needed)
        staff1_col = "Flight_Duty_personnel"
        staff2_col = "Deputy"
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)

        df_filtered = df.copy()

        if date_col:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])

        # ====================== SIDEBAR FILTERS FOR FLIGHTS ======================
        with st.sidebar:
            st.subheader("Filters – International Flights")

            # Date range
            if date_col:
                min_date = df_filtered[date_col].min().date()
                max_date = df_filtered[date_col].max().date()
                col1, col2 = st.columns(2)
                start_date = col1.date_input("Start Date", min_date, key=f"start_date_flights")
                end_date = col2.date_input("End Date", max_date, key=f"end_date_flights")
                df_filtered = df_filtered[(df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)]

            # Duty Personnel Filter
            if staff1_col in df_filtered.columns:
                duty_options = sorted(df_filtered[staff1_col].dropna().astype(str).unique())
                selected_duty = st.multiselect("Flight Duty Personnel", duty_options, key="duty_filter_flights")
                if selected_duty:
                    df_filtered = df_filtered[df_filtered[staff1_col].astype(str).isin(selected_duty)]

            # Deputy Filter
            if staff2_col in df_filtered.columns:
                deputy_options = sorted(df_filtered[staff2_col].dropna().astype(str).unique())
                selected_deputy = st.multiselect("Deputy", deputy_options, key="deputy_filter_flights")
                if selected_deputy:
                    df_filtered = df_filtered[df_filtered[staff2_col].astype(str).isin(selected_deputy)]

            # Cancel Filters Button
            if st.button("Cancel All Filters", type="primary"):
                for key in ["start_date_flights", "end_date_flights", "duty_filter_flights", "deputy_filter_flights"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # === Rest of Flights Summary (unchanged) ===
        if df_filtered.empty:
            st.info("No data for selected filters.")
            st.stop()

        # Summary table logic (your existing code)
        summary_data = []
        total_entries = len(df_filtered)
        summary_data.append(["Total International Flights Screened", total_entries])
        total_days = df_filtered[date_col].dt.date.nunique() if date_col else 'N/A'
        summary_data.append(["Total Days of Screening", total_days])

        # ... rest of numeric columns sum ...

        summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
        st.table(summary_df)
        st.download_button("Download Raw Data", to_excel(df_filtered), "Flights_Raw_Data.xlsx")
        st.stop()

    # ===================================================================
    # PERI & INTRA SECTIONS – Isolated Filters
    # ===================================================================
    # Column mapping
    col_map_lower = {c.lower(): c for c in df.columns}
    col_zone = col_map_lower.get('zone')
    col_subzone = col_map_lower.get('subzone')
    col_street = col_map_lower.get('streetname')
    col_username = col_map_lower.get('username')
    col_premises = "Premises" if "Premises" in df.columns else col_map_lower.get('premises')
    col_pos_house_raw = "Among_the_wet_containers_how_"
    col_pos_cont_raw = "Among_the_wet_containers_how_"
    col_wet_cont_raw = "Number_of_wet_containers_found" if "Number_of_wet_containers_found" in df.columns else "Number_of_wet_containers_"
    col_dry_cont_raw = "number_of_dry_contai_tentially_hold_water"
    col_lat = "_Location_latitude"
    col_lon = "_Location_longitude"
    date_col = "Date" if "Date" in df.columns else col_map_lower.get('date')
    if not date_col:
        for c in ['today', 'start', '_submission_time']:
            if c in col_map_lower:
                date_col = col_map_lower[c]
                break

    df_filtered = df.copy()
    if date_col:
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')

    # ====================== SIDEBAR FILTERS FOR PERI/INTRA ======================
    with st.sidebar:
        st.subheader(f"Filters – {current_config['title']}")

        # Date Filter
        if date_col:
            min_date = df_filtered[date_col].dt.date.min()
            max_date = df_filtered[date_col].dt.date.max()
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start", min_date, key=f"start_date_{selected_key}")
            end_date = c2.date_input("End", max_date, key=f"end_date_{selected_key}")
            df_filtered = df_filtered[(df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)]

        # Zone Filter
        if col_zone and col_zone in df_filtered.columns:
            zones = sorted(df_filtered[col_zone].dropna().astype(str).unique())
            selected_zones = st.multiselect("Zone", zones, key=f"zone_filter_{selected_key}")
            if selected_zones:
                df_filtered = df_filtered[df_filtered[col_zone].astype(str).isin(selected_zones)]

        # Subzone Filter (Peri only)
        if selected_key == 'peri' and col_subzone and col_subzone in df_filtered.columns:
            subzones = sorted(df_filtered[col_subzone].dropna().astype(str).unique())
            selected_subzones = st.multiselect("Subzone", subzones, key=f"subzone_filter_{selected_key}")
            if selected_subzones:
                df_filtered = df_filtered[df_filtered[col_subzone].astype(str).isin(selected_subzones)]

        # Cancel Filters Button
        if st.button("Cancel All Filters", type="primary", key=f"clear_{selected_key}"):
            keys_to_clear = [k for k in st.session_state.keys() if k.endswith(f"_{selected_key}") or k in ["start_date_flights", "end_date_flights"]]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # === Rest of your existing Peri/Intra dashboard logic (unchanged) ===
    # Calculations, metrics, graphs, reports, etc.
    # ... (keep everything from your original code below the filters) ...

    # Example snippet – keep your existing metric calculations
    for col, raw_col in [('pos_house_calc', col_pos_house_raw), ('pos_cont_calc', col_pos_cont_raw), ('wet_cont_calc', col_wet_cont_raw)]:
        df_filtered[col] = pd.to_numeric(df_filtered[raw_col], errors='coerce').fillna(0) if raw_col in df_filtered.columns else 0
    # ... continue with the rest of your original code ...

    # (All the metric rows, graphs, maps, reports, etc. remain exactly the same)
    # Just make sure you use `df_filtered` everywhere

# === APP START ===
if 'page' not in st.session_state:
    st.session_state.page = 'home'

render_home_page()
