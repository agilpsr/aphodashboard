import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="Larvae Surveillance Dashboard", layout="wide")

# --- 2. PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
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

# --- 3. DATA LOADING FUNCTION ---
@st.cache_data(ttl=300)
def load_kobo_data(url):
    try:
        token = st.secrets["KOBO_TOKEN"]
        headers = {"Authorization": f"Token {token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Robust loading: auto-detect separator, skip bad lines
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

# --- 4. NAVIGATION & TITLES ---
# Map the internal keys to the User-Friendly Titles you requested
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

# Sidebar Navigation
st.sidebar.header("Navigation")
# We use the keys (outside, inside, etc.) for the radio button options
selected_key = st.sidebar.radio("Select Report:", list(SECTION_CONFIG.keys()), format_func=lambda x: SECTION_CONFIG[x]['title'])

# Get current config
current_config = SECTION_CONFIG[selected_key]
st.title(current_config['title'])

# --- 5. MAIN LOGIC ---
with st.spinner('Fetching data...'):
    df = load_kobo_data(current_config['url'])

if not df.empty:
    # --- A. DATA CLEANING & NORMALIZATION ---
    # Try to standardize column names to Title Case for easier matching (e.g., 'zone' -> 'Zone')
    # This creates a copy of columns mapped to "Zone", "SubZone", etc. if they exist vaguely
    col_map = {c.lower(): c for c in df.columns}
    
    # Define the standard columns we want to filter by
    target_cols = {
        'zone': col_map.get('zone') or col_map.get('zone_name') or 'Zone',
        'subzone': col_map.get('subzone') or col_map.get('sub_zone') or 'SubZone',
        'street': col_map.get('streetname') or col_map.get('street_name') or 'StreetName'
    }

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
        
        # Two columns in sidebar for dates
        d1, d2 = st.sidebar.columns(2)
        start_date = d1.date_input("Start", min_date)
        end_date = d2.date_input("End", max_date)
        
        mask = (df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask]

    # 2. Zone / Subzone / Street Filters
    # We loop through our target columns. If the column exists in the Kobo data, we show a filter.
    for label, actual_col in target_cols.items():
        if actual_col in df_filtered.columns:
            # Get unique values, ignore NaNs
            options = sorted(df_filtered[actual_col].dropna().unique().astype(str))
            selected_options = st.sidebar.multiselect(f"Filter by {label.title()}", options)
            
            if selected_options:
                df_filtered = df_filtered[df_filtered[actual_col].astype(str).isin(selected_options)]

    # --- C. DISPLAY METRICS ---
    st.markdown(f"### Total Entries: **{len(df_filtered)}**")

    # --- D. GRAPHS ---
    # Using a toggle switch for the "Graph" button functionality
    show_graphs = st.toggle("Show Graphs", value=False)
    
    if show_graphs:
        st.divider()
        st.subheader("Visual Analysis")
        
        if df_filtered.empty:
            st.warning("No data available to plot.")
        else:
            # Logic for OUTSIDE (Peri-Airport)
            if selected_key == 'outside':
                c1, c2 = st.columns(2)
                
                # Zone Graph
                if target_cols['zone'] in df_filtered.columns:
                    counts = df_filtered[target_cols['zone']].value_counts().reset_index()
                    counts.columns = ['Zone', 'Count']
                    fig_z = px.bar(counts, x='Zone', y='Count', title="Houses Seen by Zone", text='Count')
                    c1.plotly_chart(fig_z, use_container_width=True)

                # SubZone Graph
                if target_cols['subzone'] in df_filtered.columns:
                    counts = df_filtered[target_cols['subzone']].value_counts().reset_index()
                    counts.columns = ['SubZone', 'Count']
                    fig_s = px.bar(counts, x='SubZone', y='Count', title="Houses Seen by SubZone", text='Count')
                    c2.plotly_chart(fig_s, use_container_width=True)
                
                # Street Graph (Full Width)
                if target_cols['street'] in df_filtered.columns:
                    counts = df_filtered[target_cols['street']].value_counts().reset_index()
                    counts.columns = ['Street Name', 'Count']
                    # Use a scrolling bar chart if there are many streets
                    fig_st = px.bar(counts, x='Street Name', y='Count', title="Houses Seen by Street Name", text='Count')
                    st.plotly_chart(fig_st, use_container_width=True)

            # Logic for INSIDE (Intra-Airport)
            elif selected_key == 'inside':
                if target_cols['zone'] in df_filtered.columns:
                    counts = df_filtered[target_cols['zone']].value_counts().reset_index()
                    counts.columns = ['Zone', 'Count']
                    fig_z = px.bar(counts, x='Zone', y='Count', title="Premises Seen by Zone", text='Count')
                    st.plotly_chart(fig_z, use_container_width=True)
                else:
                    st.info("Zone column not found in this dataset.")
            
            else:
                st.info("Graphs are specifically configured for Outside and Inside reports. Select those to view charts.")

    # --- E. DATA TABLE ---
    st.divider()
    with st.expander("View Detailed Data"):
        st.dataframe(df_filtered)

else:
    st.info("No data found. Please check your Kobo connection.")
