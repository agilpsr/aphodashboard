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
        # Centered login box
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.markdown("### 🔒 Secure Access")
            password_input = st.text_input("Enter Password:", type="password")
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

# --- HELPER FUNCTION FOR GRAPHS ---
def plot_metric_bar(data, x_col, y_col, title, color_col):
    """
    Creates a bar chart with Green->Red gradient based on risk level.
    """
    # RdYlGn_r = Red Yellow Green (Reversed), so Low=Green, High=Red
    fig = px.bar(
        data, x=x_col, y=y_col, 
        title=title, text=y_col,
        color=color_col,
        color_continuous_scale='RdYlGn_r', 
        range_color=[0, 20] # 0 is Green, 20+ is Red (Adjust threshold as needed)
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(coloraxis_showscale=False) # Hide the color bar to keep it clean
    return fig

if not df.empty:
    # --- A. CLEANING & MAPPING ---
    col_map = {c.lower(): c for c in df.columns}
    
    # Identify key columns safely
    col_zone = col_map.get('zone') or col_map.get('zone_name')
    col_subzone = col_map.get('subzone') or col_map.get('sub_zone')
    col_street = col_map.get('streetname') or col_map.get('street_name')
    
    # Columns for Indices (Exact names provided)
    col_pos_house_raw = "Among_the_wet_containers_how_"
    col_pos_cont_raw = "In_the_Number_of_wet_containe"
    col_wet_cont_raw = "Number_of_wet_containers_found"

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

    # 2. Explicit Zone/Subzone Filters
    selected_zones = []
    selected_subzones = []

    if col_zone and col_zone in df_filtered.columns:
        options = sorted(df_filtered[col_zone].dropna().unique().astype(str))
        selected_zones = st.sidebar.multiselect(f"Filter by Zone", options)
        if selected_zones:
            df_filtered = df_filtered[df_filtered[col_zone].astype(str).isin(selected_zones)]

    if col_subzone and col_subzone in df_filtered.columns:
        options = sorted(df_filtered[col_subzone].dropna().unique().astype(str))
        selected_subzones = st.sidebar.multiselect(f"Filter by SubZone", options)
        if selected_subzones:
            df_filtered = df_filtered[df_filtered[col_subzone].astype(str).isin(selected_subzones)]

    if col_street and col_street in df_filtered.columns:
        options = sorted(df_filtered[col_street].dropna().unique().astype(str))
        selected_streets = st.sidebar.multiselect(f"Filter by Street", options)
        if selected_streets:
            df_filtered = df_filtered[df_filtered[col_street].astype(str).isin(selected_streets)]

    # --- C. CALCULATIONS ---
    # Convert columns to numeric, forcing errors to 0
    if col_pos_house_raw in df_filtered.columns:
        df_filtered['pos_house_calc'] = pd.to_numeric(df_filtered[col_pos_house_raw], errors='coerce').fillna(0)
        # Convert any positive number to 1 for House Index calculation
        df_filtered['is_positive_house'] = df_filtered['pos_house_calc'].apply(lambda x: 1 if x > 0 else 0)
    
    if col_pos_cont_raw in df_filtered.columns:
        df_filtered['pos_cont_calc'] = pd.to_numeric(df_filtered[col_pos_cont_raw], errors='coerce').fillna(0)
    
    if col_wet_cont_raw in df_filtered.columns:
        df_filtered['wet_cont_calc'] = pd.to_numeric(df_filtered[col_wet_cont_raw], errors='coerce').fillna(0)

    # --- CALCULATION LOGIC ---
    total_entries = len(df_filtered)
    
    # 1. House/Premises Index (HI/PI)
    # Formula: (Positive Houses / Total Houses) * 100
    hi_val = 0
    if 'is_positive_house' in df_filtered.columns and total_entries > 0:
        pos_houses = df_filtered['is_positive_house'].sum()
        hi_val = (pos_houses / total_entries) * 100

    # 2. Container Index (CI)
    # Formula: (Positive Containers / Wet Containers) * 100
    ci_val = 0
    if 'pos_cont_calc' in df_filtered.columns and 'wet_cont_calc' in df_filtered.columns:
        total_pos_cont = df_filtered['pos_cont_calc'].sum()
        total_wet_cont = df_filtered['wet_cont_calc'].sum()
        if total_wet_cont > 0:
            ci_val = (total_pos_cont / total_wet_cont) * 100
            
    # 3. Breteau Index (BI)
    # Formula: (Positive Containers / Total Houses) * 100
    bi_val = 0
    if 'pos_cont_calc' in df_filtered.columns and total_entries > 0:
        total_pos_cont = df_filtered['pos_cont_calc'].sum()
        bi_val = (total_pos_cont / total_entries) * 100

    # --- D. TOP METRICS ---
    # Dynamic labels based on report type
    label_hi = "Premises Index (PI)" if selected_key == 'inside' else "House Index (HI)"
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Entries", total_entries)
    m2.metric(label_hi, f"{hi_val:.2f}")
    m3.metric("Container Index (CI)", f"{ci_val:.2f}")
    m4.metric("Breteau Index (BI)", f"{bi_val:.2f}")

    # --- E. ADVANCED VISUALIZATION ---
    # Only show graphs for Surveillance sheets (Outside/Inside)
    if selected_key in ['outside', 'inside']:
        st.divider()
        st.write("### 📊 Epidemiological Analysis")
        
        # Logic to hide graphs based on drill-down
        show_zone_graph = (len(selected_zones) == 0) and (len(selected_subzones) == 0)
        show_subzone_graph = (len(selected_subzones) == 0)

        # Prepare Grouping Function
        def get_grouped_data(groupby_col):
            """Aggregates data by the chosen column (Zone/Subzone/Street)"""
            g = df_filtered.groupby(groupby_col).agg({
                'is_positive_house': 'sum',
                'pos_cont_calc': 'sum',
                'wet_cont_calc': 'sum',
                groupby_col: 'count' # This counts total entries (houses)
            }).rename(columns={groupby_col: 'Total Entries'})
            
            # Calculate Indices
            g['HI'] = (g['is_positive_house'] / g['Total Entries']) * 100
            g['CI'] = (g['pos_cont_calc'] / g['wet_cont_calc'].replace(0, 1)) * 100 # Avoid div by 0
            g['BI'] = (g['pos_cont_calc'] / g['Total Entries']) * 100
            
            return g.reset_index()

        # --- GRAPH SECTION 1: HOUSE / PREMISES INDEX ---
        with st.container():
            st.markdown(f"#### 🏠 {label_hi} Analysis")
            st.info("Vector Density: Percentage of houses/premises found positive.")
            
            # 1. Zone Graph
            if show_zone_graph and col_zone in df_filtered.columns:
                data_z = get_grouped_data(col_zone)
                st.plotly_chart(plot_metric_bar(data_z, col_zone, 'HI', f"{label_hi} by Zone", 'HI'), use_container_width=True)

            # 2. SubZone Graph (Only for Outside)
            if selected_key == 'outside':
                if show_subzone_graph and col_subzone in df_filtered.columns:
                    data_s = get_grouped_data(col_subzone)
                    st.plotly_chart(plot_metric_bar(data_s, col_subzone, 'HI', f"{label_hi} by SubZone", 'HI'), use_container_width=True)
                
                # 3. Street Graph
                if col_street in df_filtered.columns:
                    data_st = get_grouped_data(col_street)
                    with st.expander("View Street Level Analysis"):
                        st.plotly_chart(plot_metric_bar(data_st, col_street, 'HI', f"{label_hi} by Street", 'HI'), use_container_width=True)

        st.divider()

        # --- GRAPH SECTION 2: CONTAINER INDEX ---
        with st.container():
            st.markdown("#### 🪣 Container Index (CI) Analysis")
            st.info("Breeding Preference: Percentage of wet containers found positive.")

            # 1. Zone Graph
            if show_zone_graph and col_zone in df_filtered.columns:
                data_z = get_grouped_data(col_zone)
                st.plotly_chart(plot_metric_bar(data_z, col_zone, 'CI', "Container Index by Zone", 'CI'), use_container_width=True)
            
            # 2. SubZone (Outside)
            if selected_key == 'outside' and show_subzone_graph and col_subzone in df_filtered.columns:
                data_s = get_grouped_data(col_subzone)
                st.plotly_chart(plot_metric_bar(data_s, col_subzone, 'CI', "Container Index by SubZone", 'CI'), use_container_width=True)
            
            # 3. Street
            if selected_key == 'outside' and col_street in df_filtered.columns:
                with st.expander("View Street Level Analysis"):
                    data_st = get_grouped_data(col_street)
                    st.plotly_chart(plot_metric_bar(data_st, col_street, 'CI', "Container Index by Street", 'CI'), use_container_width=True)

        st.divider()

        # --- GRAPH SECTION 3: BRETEAU INDEX ---
        with st.container():
            st.markdown("#### 🦟 Breteau Index (BI) Analysis")
            st.info("Breeding Risk: Number of positive containers per 100 houses.")

            # 1. Zone Graph
            if show_zone_graph and col_zone in df_filtered.columns:
                data_z = get_grouped_data(col_zone)
                st.plotly_chart(plot_metric_bar(data_z, col_zone, 'BI', "Breteau Index by Zone", 'BI'), use_container_width=True)
            
            # 2. SubZone (Outside)
            if selected_key == 'outside' and show_subzone_graph and col_subzone in df_filtered.columns:
                data_s = get_grouped_data(col_subzone)
                st.plotly_chart(plot_metric_bar(data_s, col_subzone, 'BI', "Breteau Index by SubZone", 'BI'), use_container_width=True)

            # 3. Street
            if selected_key == 'outside' and col_street in df_filtered.columns:
                with st.expander("View Street Level Analysis"):
                    data_st = get_grouped_data(col_street)
                    st.plotly_chart(plot_metric_bar(data_st, col_street, 'BI', "Breteau Index by Street", 'BI'), use_container_width=True)

    # --- E. DATA TABLE ---
    st.divider()
    with st.expander("View Raw Data Table"):
        st.dataframe(df_filtered)

else:
    st.info("No data found. Please check your Kobo connection or selection.")
