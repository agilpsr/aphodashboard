import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="Larvae Surveillance Dashboard", layout="wide")

# --- 2. PASSWORD PROTECTION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password_input = st.text_input("Enter Password to Login:", type="password")
        if st.button("Login"):
            if password_input == st.secrets["APP_PASSWORD"]:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect Password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- 3. DATA LOADING ---
@st.cache_data(ttl=300)
def load_kobo_data(url):
    try:
        token = st.secrets["KOBO_TOKEN"]
        headers = {"Authorization": f"Token {token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        df = pd.read_csv(
            io.StringIO(response.text), 
            sep=None, 
            engine='python', 
            on_bad_lines='skip'
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- 4. NAVIGATION ---
SECTION_CONFIG = {
    'outside': {
        'title': 'Peri-Airport Larvae Surveillance',
        'url': 'https://kf.kobotoolbox.org/api/v2/assets/aXM5aSjVEJTgt6z5qMvNFe/export-settings/es9zUAYU5f8PqCokaZSuPmg/data.csv'
    },
    'inside': {
        'title': 'Intra-Airport Larvae Surveillance',
        'url': 'https://kf.kobotoolbox.org/api/v2/assets/aEdcSxvmrBuXBmzXNECtjr/export-settings/esgYdEaEk79Y69k56abNGdW/data.csv'
    },
    'outid': {
        'title': 'Peri-Airport Larvae Identification',
        'url': 'https://kf.kobotoolbox.org/api/v2/assets/afU6pGvUzT8Ao4pAeX54QY/export-settings/esinGxnSujLzanzmAv6Mdb4/data.csv'
    },
    'inid': {
        'title': 'Intra-Airport Larvae Identification',
        'url': 'https://kf.kobotoolbox.org/api/v2/assets/anN9HTYvmLRTorb7ojXs5A/export-settings/esDnrutbSWfn8AbieZSqzdV/data.csv'
    }
}

st.sidebar.header("Navigation")
selected_key = st.sidebar.radio("Select Report:", list(SECTION_CONFIG.keys()), format_func=lambda x: SECTION_CONFIG[x]['title'])
current_config = SECTION_CONFIG[selected_key]
st.title(current_config['title'])

with st.spinner('Fetching data...'):
    df = load_kobo_data(current_config['url'])

if not df.empty:
    # --- A. CLEANING & MAPPING ---
    col_map = {c.lower(): c for c in df.columns}
    
    # Identify key columns safely
    col_zone = col_map.get('zone') or col_map.get('zone_name')
    col_subzone = col_map.get('subzone') or col_map.get('sub_zone')
    col_street = col_map.get('streetname') or col_map.get('street_name')
    col_positive = "Among_the_wet_containers_how_" # Exact name from user

    # --- B. FILTERS ---
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    
    df_filtered = df.copy()

    # 1. Date Filter
    date_col = None
    possible_date_cols = ['_submission_time', 'start', 'end', 'today', 'Date', 'date']
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
            
    if date_col:
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        min_date = df_filtered[date_col].min().date()
        max_date = df_filtered[date_col].max().date()
        
        d1, d2 = st.sidebar.columns(2)
        start_date = d1.date_input("Start", min_date)
        end_date = d2.date_input("End", max_date)
        
        mask = (df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask]

    # 2. Explicit Zone/Subzone Filters (Needed for graph logic)
    selected_zones = []
    selected_subzones = []

    if col_zone and col_zone in df_filtered.columns:
        options = sorted(df_filtered[col_zone].dropna().unique().astype(str))
        selected_zones = st.sidebar.multiselect(f"Filter by Zone", options)
        if selected_zones:
            df_filtered = df_filtered[df_filtered[col_zone].astype(str).isin(selected_zones)]

    if col_subzone and col_subzone in df_filtered.columns:
        # Filter options based on remaining data
        options = sorted(df_filtered[col_subzone].dropna().unique().astype(str))
        selected_subzones = st.sidebar.multiselect(f"Filter by SubZone", options)
        if selected_subzones:
            df_filtered = df_filtered[df_filtered[col_subzone].astype(str).isin(selected_subzones)]

    if col_street and col_street in df_filtered.columns:
        options = sorted(df_filtered[col_street].dropna().unique().astype(str))
        selected_streets = st.sidebar.multiselect(f"Filter by Street", options)
        if selected_streets:
            df_filtered = df_filtered[df_filtered[col_street].astype(str).isin(selected_streets)]

    # --- C. METRICS & INDICES ---
    total_entries = len(df_filtered)
    
    # Layout for metrics
    m1, m2 = st.columns(2)
    m1.metric("Total Houses/Entries", total_entries)

    # House Index Calculation (Only for Peri-Airport)
    if selected_key == 'outside':
        if col_positive in df_filtered.columns:
            # 1. Force column to numbers, turn errors into 0
            # 2. If value > 0, it counts as 1. If 0, counts as 0.
            positive_series = pd.to_numeric(df_filtered[col_positive], errors='coerce').fillna(0)
            positive_houses_count = (positive_series > 0).sum()
            
            # 3. Calculate Index
            if total_entries > 0:
                house_index = (positive_houses_count / total_entries) * 100
            else:
                house_index = 0
            
            m2.metric("House Index (HI)", f"{house_index:.2f}%", help="Positive Houses / Total Houses * 100")
        else:
            m2.warning(f"Column '{col_positive}' not found for House Index.")

    # --- D. SMART GRAPHS ---
    show_graphs = st.toggle("Show Graphs", value=True)
    
    if show_graphs and not df_filtered.empty:
        st.divider()
        
        # Decide which graphs to show based on filters
        # Logic: 
        # - Show Zone graph ONLY IF no specific zone is selected AND no subzone is selected.
        # - Show SubZone graph ONLY IF no specific subzone is selected.
        
        show_zone_graph = (len(selected_zones) == 0) and (len(selected_subzones) == 0)
        show_subzone_graph = (len(selected_subzones) == 0)

        if selected_key == 'outside':
            c1, c2 = st.columns(2)
            
            # 1. Zone Graph
            if show_zone_graph and col_zone in df_filtered.columns:
                counts = df_filtered[col_zone].value_counts().reset_index()
                counts.columns = ['Zone', 'Count']
                fig_z = px.bar(counts, x='Zone', y='Count', title="Houses by Zone", text='Count')
                c1.plotly_chart(fig_z, use_container_width=True)
            
            # 2. SubZone Graph
            if show_subzone_graph and col_subzone in df_filtered.columns:
                counts = df_filtered[col_subzone].value_counts().reset_index()
                counts.columns = ['SubZone', 'Count']
                fig_s = px.bar(counts, x='SubZone', y='Count', title="Houses by SubZone", text='Count')
                # Put in c2 if c1 is used, otherwise c1
                target_col = c2 if show_zone_graph else c1
                target_col.plotly_chart(fig_s, use_container_width=True)
            
            # 3. Street Graph (Always show if data exists)
            if col_street in df_filtered.columns:
                counts = df_filtered[col_street].value_counts().reset_index()
                counts.columns = ['Street Name', 'Count']
                fig_st = px.bar(counts, x='Street Name', y='Count', title="Houses by Street Name", text='Count')
                st.plotly_chart(fig_st, use_container_width=True)

        elif selected_key == 'inside':
             if col_zone in df_filtered.columns:
                counts = df_filtered[col_zone].value_counts().reset_index()
                counts.columns = ['Zone', 'Count']
                fig_z = px.bar(counts, x='Zone', y='Count', title="Premises by Zone", text='Count')
                st.plotly_chart(fig_z, use_container_width=True)

    # --- E. DATA TABLE ---
    st.divider()
    with st.expander("View Detailed Data"):
        st.dataframe(df_filtered)

else:
    st.info("No data found. Please check your Kobo connection.")
