import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import re

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

# --- 4. NAVIGATION & CONFIGURATION ---
# Refactored: Now we have 2 main sections, each has a Surveillance URL and an ID URL
SECTION_CONFIG = {
    'peri': {
        'title': 'Peri-Airport Larvae Surveillance',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aXM5aSjVEJTgt6z5qMvNFe/export-settings/es9zUAYU5f8PqCokaZSuPmg/data.csv',
        'id_url': 'https://kf.kobotoolbox.org/api/v2/assets/afU6pGvUzT8Ao4pAeX54QY/export-settings/esinGxnSujLzanzmAv6Mdb4/data.csv'
    },
    'intra': {
        'title': 'Intra-Airport Larvae Surveillance',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aEdcSxvmrBuXBmzXNECtjr/export-settings/esgYdEaEk79Y69k56abNGdW/data.csv',
        'id_url': 'https://kf.kobotoolbox.org/api/v2/assets/anN9HTYvmLRTorb7ojXs5A/export-settings/esDnrutbSWfn8AbieZSqzdV/data.csv'
    }
}

st.sidebar.header("Navigation")
# Simplified Radio Button (Just Peri vs Intra)
selected_key = st.sidebar.radio("Select Report:", list(SECTION_CONFIG.keys()), format_func=lambda x: SECTION_CONFIG[x]['title'])
current_config = SECTION_CONFIG[selected_key]
st.title(current_config['title'])

# --- 5. LOAD SURVEILLANCE DATA ---
with st.spinner('Fetching Surveillance data...'):
    df = load_kobo_data(current_config['surv_url'])

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

# --- HELPER: NUCLEAR CLEANING ---
def normalize_string(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

if not df.empty:
    # --- A. CLEANING & MAPPING (SURVEILLANCE) ---
    col_map_lower = {c.lower(): c for c in df.columns}
    
    col_zone = col_map_lower.get('zone') or col_map_lower.get('zone_name')
    col_subzone = col_map_lower.get('subzone') or col_map_lower.get('sub_zone')
    col_street = col_map_lower.get('streetname') or col_map_lower.get('street_name')
    col_premises = col_map_lower.get('premises') or col_map_lower.get('premise') or col_map_lower.get('location')
    
    col_pos_house_raw = "Among_the_wet_containers_how_"  
    col_pos_cont_raw = "Among_the_wet_containers_how_"  
    col_wet_cont_raw = "Number_of_wet_containers_found" 

    # --- UNIVERSAL DATE LOGIC ---
    date_col = col_map_lower.get('date')
    if not date_col:
        date_col = col_map_lower.get('today')
    if not date_col:
        fallback_cols = ['start', '_submission_time']
        for c in fallback_cols:
             if c in col_map_lower:
                 date_col = col_map_lower[c]
                 break

    # --- B. FILTERS ---
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    df_filtered = df.copy()

    # Apply Date Filter
    start_date, end_date = None, None
    if date_col:
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        min_date = df_filtered[date_col].min().date()
        max_date = df_filtered[date_col].max().date()
        
        d1, d2 = st.sidebar.columns(2)
        start_date = d1.date_input("Start", min_date)
        end_date = d2.date_input("End", max_date)
        
        mask = (df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask]
    else:
        st.warning("⚠️ CRITICAL: Could not find a column named 'Date'. Check your Kobo form.")

    # Explicit Zone/Subzone Filters
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

    # --- C. PRE-CALCULATIONS ---
    if col_pos_house_raw in df_filtered.columns:
        df_filtered['pos_house_calc'] = pd.to_numeric(df_filtered[col_pos_house_raw], errors='coerce').fillna(0)
    
    if col_pos_cont_raw in df_filtered.columns:
        df_filtered['pos_cont_calc'] = pd.to_numeric(df_filtered[col_pos_cont_raw], errors='coerce').fillna(0)
    
    if col_wet_cont_raw in df_filtered.columns:
        df_filtered['wet_cont_calc'] = pd.to_numeric(df_filtered[col_wet_cont_raw], errors='coerce').fillna(0)

    # --- D. LOGIC BRANCHING ---
    display_count = 0
    positive_count = 0
    hi_val = 0
    ci_val = 0
    bi_val = 0

    if selected_key == 'intra':
        if col_premises and date_col:
            df_filtered['date_str_only'] = df_filtered[date_col].dt.date.astype(str)
            df_filtered['premise_clean'] = df_filtered[col_premises].apply(normalize_string)
            df_filtered['unique_premise_id'] = df_filtered['date_str_only'] + "_" + df_filtered['premise_clean']
            
            agg_dict = {
                'pos_house_calc': 'max',
                'pos_cont_calc': 'sum',
                'wet_cont_calc': 'sum'
            }
            if col_zone in df_filtered.columns:
                agg_dict[col_zone] = 'first' 
                
            df_grouped = df_filtered.groupby('unique_premise_id', as_index=False).agg(agg_dict)
            
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
            positive_count = positive_premises_count
        else:
            st.warning("⚠️ Could not find 'Premises' or 'Date' column.")
            df_for_graphs = df_filtered.copy()

    else:
        # --- PERI-AIRPORT ---
        display_count = len(df_filtered)
        df_filtered['is_positive_house'] = df_filtered['pos_house_calc'].apply(lambda x: 1 if x > 0 else 0)
        positive_count = df_filtered['is_positive_house'].sum()
        
        if display_count > 0:
            hi_val = (positive_count / display_count) * 100
            total_pos_cont = df_filtered['pos_cont_calc'].sum()
            total_wet_cont = df_filtered['wet_cont_calc'].sum()
            ci_val = (total_pos_cont / total_wet_cont * 100) if total_wet_cont > 0 else 0
            bi_val = (total_pos_cont / display_count * 100)
        
        df_for_graphs = df_filtered.copy()

    # --- E. TOP METRICS DISPLAY ---
    label_hi = "Premises Index (PI)" if selected_key == 'intra' else "House Index (HI)"
    label_entries = "Unique Premises" if selected_key == 'intra' else "Total Entries"
    label_positive = "Positive Premises" if selected_key == 'intra' else "Positive Houses"
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(label_entries, display_count)
    m2.metric(label_positive, positive_count)
    m3.metric(label_hi, f"{hi_val:.2f}")
    m4.metric("Container Index (CI)", f"{ci_val:.2f}")
    m5.metric("Breteau Index (BI)", f"{bi_val:.2f}")

    # --- F. GRAPHICAL ANALYSIS ---
    st.divider()
    show_graphs = st.toggle("Show Graphical Analysis", value=False)
    
    if show_graphs:
        show_zone_graph = (len(selected_zones) == 0) and (len(selected_subzones) == 0)
        show_subzone_graph = (len(selected_subzones) == 0)

        def get_grouped_data(groupby_col):
            aggs = {'pos_cont_calc': 'sum', 'wet_cont_calc': 'sum'}
            if selected_key == 'intra':
                aggs[groupby_col] = 'count'
                aggs['is_positive_premise'] = 'sum'
            else:
                aggs[groupby_col] = 'count'
                aggs['is_positive_house'] = 'sum'

            g = df_for_graphs.groupby(groupby_col).agg(aggs).rename(columns={groupby_col: 'Denominator'})
            
            if selected_key == 'intra':
                g['HI'] = (g['is_positive_premise'] / g['Denominator']) * 100
            else:
                g['HI'] = (g['is_positive_house'] / g['Denominator']) * 100
                
            g['CI'] = (g['pos_cont_calc'] / g['wet_cont_calc'].replace(0, 1)) * 100 
            g['BI'] = (g['pos_cont_calc'] / g['Denominator']) * 100
            return g.reset_index()

        with st.expander(f"📊 View {label_hi} Graphs"):
            st.info("Vector Density: Percentage of houses/premises found positive.")
            if show_zone_graph and col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'HI', f"{label_hi} by Zone", 'HI'), use_container_width=True)
            if selected_key == 'peri':
                if show_subzone_graph and col_subzone in df_for_graphs.columns:
                    st.plotly_chart(plot_metric_bar(get_grouped_data(col_subzone), col_subzone, 'HI', f"{label_hi} by SubZone", 'HI'), use_container_width=True)
                if col_street in df_for_graphs.columns:
                    st.plotly_chart(plot_metric_bar(get_grouped_data(col_street), col_street, 'HI', f"{label_hi} by Street", 'HI'), use_container_width=True)

        with st.expander("📊 View Container Index (CI) Graphs"):
            st.info("Breeding Preference: Percentage of wet containers found positive.")
            if show_zone_graph and col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'CI', "Container Index by Zone", 'CI'), use_container_width=True)
            if selected_key == 'peri' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_subzone), col_subzone, 'CI', "Container Index by SubZone", 'CI'), use_container_width=True)

        with st.expander("📊 View Breteau Index (BI) Graphs"):
            st.info("Breeding Risk: Number of positive containers per 100 houses/premises.")
            if show_zone_graph and col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'BI', "Breteau Index by Zone", 'BI'), use_container_width=True)
            if selected_key == 'peri' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_subzone), col_subzone, 'BI', "Breteau Index by SubZone", 'BI'), use_container_width=True)

    # --- F. DATA TABLE ---
    st.divider()
    with st.expander("📂 View Raw Data Table"):
        st.dataframe(df_filtered)

    # --- G. DEBUG TOOL ---
    if selected_key == 'intra':
        st.divider()
        with st.expander("🐞 Debug Tool (Check Unique IDs)"):
            csv = df_grouped.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed Data", csv, "debug_unique_premises.csv", "text/csv")

    # --- H. LARVAE IDENTIFICATION SECTION ---
    st.divider()
    st.markdown("### 🔬 Larvae Identification")
    
    with st.expander("View Larvae Identification Data"):
        # Load ID Data
        with st.spinner('Fetching ID data...'):
            df_id = load_kobo_data(current_config['id_url'])
        
        if not df_id.empty:
            # 1. Apply SAME Date Filter as Main Data
            col_map_id = {c.lower(): c for c in df_id.columns}
            date_col_id = col_map_id.get('date') or col_map_id.get('today')
            
            # Find Address Column
            col_address_id = col_map_id.get('address') or col_map_id.get('location') or col_map_id.get('premise') or col_map_id.get('premises') or col_map_id.get('streetname')
            
            # Use specific columns user requested
            col_img = "Attach the microscopic image of the larva_URL"
            col_genus = "Select the Genus:"
            col_species = "Select the Species:"
            
            if date_col_id:
                df_id[date_col_id] = pd.to_datetime(df_id[date_col_id])
                if start_date and end_date: # From sidebar
                    mask_id = (df_id[date_col_id].dt.date >= start_date) & (df_id[date_col_id].dt.date <= end_date)
                    df_id = df_id.loc[mask_id]
            
            # 2. Select & Rename Columns
            # We construct the final table
            final_cols = []
            
            # Address
            if col_address_id and col_address_id in df_id.columns:
                final_cols.append(col_address_id)
            else:
                df_id['Address'] = 'N/A'
                final_cols.append('Address')
                
            # Date
            if date_col_id and date_col_id in df_id.columns:
                df_id['Date'] = df_id[date_col_id].dt.date
                final_cols.append('Date')
                
            # Image
            if col_img in df_id.columns:
                final_cols.append(col_img)
                
            # Genus
            if col_genus in df_id.columns:
                final_cols.append(col_genus)
                
            # Species
            if col_species in df_id.columns:
                final_cols.append(col_species)
                
            # Create Subset
            df_display = df_id[final_cols].copy()
            
            # Add Serial Number
            df_display.insert(0, 'Serial No', range(1, 1 + len(df_display)))
            
            # Rename for display
            rename_map = {
                col_address_id: 'Address',
                col_img: 'Image of Larva',
                col_genus: 'Genus',
                col_species: 'Species'
            }
            df_display = df_display.rename(columns=rename_map)
            
            # 3. Display with Image Column Configuration
            st.dataframe(
                df_display,
                column_config={
                    "Image of Larva": st.column_config.ImageColumn(
                        "Image of Larva", help="Microscopic Image"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No identification data available.")

else:
    st.info("No data found. Please check your Kobo connection or selection.")
