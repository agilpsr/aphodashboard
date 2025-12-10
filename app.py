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
        
        # We assume standard encoding, but we will fix artifacts later
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
    fig = px.bar(
        data, x=x_col, y=y_col, 
        title=title, text=y_col,
        color=color_col,
        color_continuous_scale='RdYlGn_r', 
        range_color=[0, 20] 
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(coloraxis_showscale=False) 
    return fig

if not df.empty:
    # --- A. CLEANING & MAPPING ---
    col_map = {c.lower(): c for c in df.columns}
    
    col_zone = col_map.get('zone') or col_map.get('zone_name')
    col_subzone = col_map.get('subzone') or col_map.get('sub_zone')
    col_street = col_map.get('streetname') or col_map.get('street_name')
    col_premises = col_map.get('premises') or col_map.get('premise') or col_map.get('location')
    
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

    # --- C. PRE-CALCULATIONS (Numeric Cleanup) ---
    if col_pos_house_raw in df_filtered.columns:
        df_filtered['pos_house_calc'] = pd.to_numeric(df_filtered[col_pos_house_raw], errors='coerce').fillna(0)
    
    if col_pos_cont_raw in df_filtered.columns:
        df_filtered['pos_cont_calc'] = pd.to_numeric(df_filtered[col_pos_cont_raw], errors='coerce').fillna(0)
    
    if col_wet_cont_raw in df_filtered.columns:
        df_filtered['wet_cont_calc'] = pd.to_numeric(df_filtered[col_wet_cont_raw], errors='coerce').fillna(0)

    # --- D. LOGIC BRANCHING (Intra vs Peri) ---
    
    if selected_key == 'inside':
        # --- INTRA-AIRPORT UNIQUE ID LOGIC ---
        if col_premises and date_col:
            # 1. Create the Key Part 1: Date String
            df_filtered['date_str_only'] = df_filtered[date_col].dt.date.astype(str)
            
            # 2. Create Key Part 2: CLEANED Premise Name
            # Start with lowercase
            s = df_filtered[col_premises].astype(str).str.lower()
            
            # FIX: REPLACE THE GARBAGE ENCODING CHARACTERS
            # We replace "Ã¢Â€Â“" and "â€“" and "–" (en-dash) and "—" (em-dash) with a simple "-"
            s = s.str.replace('ã¢â€â“', '-', regex=False) 
            s = s.str.replace('â€“', '-', regex=False)
            s = s.str.replace('–', '-', regex=False)
            s = s.str.replace('—', '-', regex=False)
            
            # Standardize whitespace (remove double spaces)
            s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
            
            df_filtered['premise_clean'] = s
            df_filtered['unique_premise_id'] = df_filtered['date_str_only'] + "_" + df_filtered['premise_clean']
            
            # 3. Group by this Unique ID
            agg_dict = {
                'pos_house_calc': 'max',
                'pos_cont_calc': 'sum',
                'wet_cont_calc': 'sum'
            }
            if col_zone in df_filtered.columns:
                agg_dict[col_zone] = 'first' 
                
            df_grouped = df_filtered.groupby('unique_premise_id', as_index=False).agg(agg_dict)
            
            # 4. Calculate Metrics
            total_unique_premises = df_grouped['unique_premise_id'].nunique()
            
            positive_premises_count = (df_grouped['pos_house_calc'] > 0).sum()
            hi_val = (positive_premises_count / total_unique_premises * 100) if total_unique_premises > 0 else 0
            
            total_pos_cont = df_grouped['pos_cont_calc'].sum()
            total_wet_cont = df_grouped['wet_cont_calc'].sum()
            ci_val = (total_pos_cont / total_wet_cont * 100) if total_wet_cont > 0 else 0
            
            bi_val = (total_pos_cont / total_unique_premises * 100) if total_unique_premises > 0 else 0
            
            df_for_graphs = df_grouped.copy()
            df_for_graphs['is_positive_premise'] = (df_for_graphs['pos_house_calc'] > 0).astype(int)
            
            display_count = total_unique_premises

        else:
            st.warning("⚠️ Could not find 'Premises' or 'Date' column.")
            display_count = len(df_filtered)
            hi_val, ci_val, bi_val = 0, 0, 0
            df_for_graphs = df_filtered.copy()

    else:
        # --- PERI-AIRPORT (STANDARD) LOGIC ---
        display_count = len(df_filtered)
        df_filtered['is_positive_house'] = df_filtered['pos_house_calc'].apply(lambda x: 1 if x > 0 else 0)
        
        if display_count > 0:
            hi_val = (df_filtered['is_positive_house'].sum() / display_count) * 100
            total_pos_cont = df_filtered['pos_cont_calc'].sum()
            total_wet_cont = df_filtered['wet_cont_calc'].sum()
            ci_val = (total_pos_cont / total_wet_cont * 100) if total_wet_cont > 0 else 0
            bi_val = (total_pos_cont / display_count * 100)
        else:
            hi_val, ci_val, bi_val = 0, 0, 0
            
        df_for_graphs = df_filtered.copy()

    # --- E. TOP METRICS DISPLAY ---
    label_hi = "Premises Index (PI)" if selected_key == 'inside' else "House Index (HI)"
    label_entries = "Unique Premises" if selected_key == 'inside' else "Total Entries"
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label_entries, display_count)
    m2.metric(label_hi, f"{hi_val:.2f}")
    m3.metric("Container Index (CI)", f"{ci_val:.2f}")
    m4.metric("Breteau Index (BI)", f"{bi_val:.2f}")

    # --- F. GRAPHICAL ANALYSIS ---
    st.divider()
    show_graphs = st.toggle("Show Graphical Analysis", value=False)
    
    if show_graphs and selected_key in ['outside', 'inside']:
        
        show_zone_graph = (len(selected_zones) == 0) and (len(selected_subzones) == 0)
        show_subzone_graph = (len(selected_subzones) == 0)

        def get_grouped_data(groupby_col):
            aggs = {
                'pos_cont_calc': 'sum',
                'wet_cont_calc': 'sum',
            }
            if selected_key == 'inside':
                aggs[groupby_col] = 'count'
                aggs['is_positive_premise'] = 'sum'
            else:
                aggs[groupby_col] = 'count'
                aggs['is_positive_house'] = 'sum'

            g = df_for_graphs.groupby(groupby_col).agg(aggs).rename(columns={groupby_col: 'Denominator'})
            
            if selected_key == 'inside':
                g['HI'] = (g['is_positive_premise'] / g['Denominator']) * 100
            else:
                g['HI'] = (g['is_positive_house'] / g['Denominator']) * 100
                
            g['CI'] = (g['pos_cont_calc'] / g['wet_cont_calc'].replace(0, 1)) * 100 
            g['BI'] = (g['pos_cont_calc'] / g['Denominator']) * 100
            
            return g.reset_index()

        with st.expander(f"📊 View {label_hi} Graphs"):
            st.info("Vector Density: Percentage of houses/premises found positive.")
            if show_zone_graph and col_zone in df_for_graphs.columns:
                data_z = get_grouped_data(col_zone)
                st.plotly_chart(plot_metric_bar(data_z, col_zone, 'HI', f"{label_hi} by Zone", 'HI'), use_container_width=True)

            if selected_key == 'outside':
                if show_subzone_graph and col_subzone in df_for_graphs.columns:
                    data_s = get_grouped_data(col_subzone)
                    st.plotly_chart(plot_metric_bar(data_s, col_subzone, 'HI', f"{label_hi} by SubZone", 'HI'), use_container_width=True)
                if col_street in df_for_graphs.columns:
                    st.plotly_chart(plot_metric_bar(get_grouped_data(col_street), col_street, 'HI', f"{label_hi} by Street", 'HI'), use_container_width=True)

        with st.expander("📊 View Container Index (CI) Graphs"):
            st.info("Breeding Preference: Percentage of wet containers found positive.")
            if show_zone_graph and col_zone in df_for_graphs.columns:
                data_z = get_grouped_data(col_zone)
                st.plotly_chart(plot_metric_bar(data_z, col_zone, 'CI', "Container Index by Zone", 'CI'), use_container_width=True)
            if selected_key == 'outside' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                data_s = get_grouped_data(col_subzone)
                st.plotly_chart(plot_metric_bar(data_s, col_subzone, 'CI', "Container Index by SubZone", 'CI'), use_container_width=True)
            if selected_key == 'outside' and col_street in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_street), col_street, 'CI', "Container Index by Street", 'CI'), use_container_width=True)

        with st.expander("📊 View Breteau Index (BI) Graphs"):
            st.info("Breeding Risk: Number of positive containers per 100 houses/premises.")
            if show_zone_graph and col_zone in df_for_graphs.columns:
                data_z = get_grouped_data(col_zone)
                st.plotly_chart(plot_metric_bar(data_z, col_zone, 'BI', "Breteau Index by Zone", 'BI'), use_container_width=True)
            if selected_key == 'outside' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                data_s = get_grouped_data(col_subzone)
                st.plotly_chart(plot_metric_bar(data_s, col_subzone, 'BI', "Breteau Index by SubZone", 'BI'), use_container_width=True)
            if selected_key == 'outside' and col_street in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_street), col_street, 'BI', "Breteau Index by Street", 'BI'), use_container_width=True)

    # --- F. DATA TABLE ---
    st.divider()
    with st.expander("📂 View Raw Data Table"):
        st.dataframe(df_filtered)

    # --- G. DEBUG TOOL ---
    if selected_key == 'inside':
        st.divider()
        with st.expander("🐞 Debug Tool (Check Unique IDs)"):
            st.write("Download this file to check which IDs are being counted.")
            csv = df_grouped.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Processed Data",
                csv,
                "debug_unique_premises.csv",
                "text/csv",
                key='download-csv'
            )

else:
    st.info("No data found. Please check your Kobo connection or selection.")
