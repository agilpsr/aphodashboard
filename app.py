import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import re
import urllib.parse
import folium
from streamlit_folium import st_folium

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="Larvae Surveillance Dashboard", layout="wide")

# --- 2. PASSWORD PROTECTION (DISABLED) ---
def check_password():
    return True

if not check_password():
    st.stop()

# --- 3. DATA LOADING ---
@st.cache_data(ttl=300)
def load_kobo_data(url):
    try:
        # NOTE: Ensure secrets exist or replace with string for local testing
        if "KOBO_TOKEN" in st.secrets:
            token = st.secrets["KOBO_TOKEN"]
        else:
            token = "48554147c1847ddfe4c1c987a54b4196a03c1d9c"
            
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
SECTION_CONFIG = {
    'peri': {
        'title': 'Peri-Airport Larvae Surveillance',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aXM5aSjVEJTgt6z5qMvNFe/export-settings/es9zUAYU5f8PqCokaZSuPmg/data.csv',
        'id_url': 'https://kf.kobotoolbox.org/api/v2/assets/afU6pGvUzT8Ao4pAeX54QY/export-settings/esinGxnSujLzanzmAv6Mdb4/data.csv'
    },
    'intra': {
        'title': 'Intra-Airport Larvae Surveillance',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aEdcSxvmrBuXBmzXNECtjr/export-settings/esgYdEaEk79Y69k56abNGdW/data.csv',
        'id_url': 'https://kf.kobotoolbox.org/api/v2/assets/anN9HTYvmLRTorb7ojXs5A/export-settings/esLiqyb8KpPfeMX4ZnSoXSm/data.csv'
    }
}

st.sidebar.header("Navigation")
selected_key = st.sidebar.radio("Select Report:", list(SECTION_CONFIG.keys()), format_func=lambda x: SECTION_CONFIG[x]['title'])
current_config = SECTION_CONFIG[selected_key]
st.title(current_config['title'])

# --- 5. LOAD SURVEILLANCE DATA ---
with st.spinner('Fetching Surveillance data...'):
    df = load_kobo_data(current_config['surv_url'])

# --- HELPER FUNCTIONS ---
def plot_metric_bar(data, x_col, y_col, title, color_col):
    fig = px.bar(
        data, x=x_col, y=y_col, title=title, text=y_col,
        color=color_col, color_continuous_scale='RdYlGn_r', range_color=[0, 20]
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(coloraxis_showscale=False) 
    return fig

def normalize_string(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def get_thumbnail_url(original_url):
    if not isinstance(original_url, str) or not original_url.startswith("http"):
        return None
    encoded_url = urllib.parse.quote(original_url)
    return f"https://wsrv.nl/?url={original_url}&w=400&q=80"

@st.dialog("Microscopic View", width="large")
def show_image_popup(row_data):
    st.subheader(f"{row_data['Genus']} ({row_data['Species']})")
    c1, c2 = st.columns(2)
    c1.info(f"📍 **Address:** {row_data['Address']}")
    c2.warning(f"📅 **Date:** {row_data['Date']}")
    if row_data['Original Image URL'] and str(row_data['Original Image URL']).startswith('http'):
        st.image(row_data['Original Image URL'], caption="Microscopic View (Full Resolution)", use_container_width=True)
    else:
        st.error("Image not available or invalid URL.")

if not df.empty:
    # --- A. COLUMN MAPPING ---
    col_map_lower = {c.lower(): c for c in df.columns}
    
    col_zone = col_map_lower.get('zone')
    col_subzone = col_map_lower.get('subzone')
    col_street = col_map_lower.get('streetname')
    col_username = col_map_lower.get('username')
    col_premises = "Premises" if "Premises" in df.columns else col_map_lower.get('premises')
    
    col_pos_house_raw = "Among_the_wet_containers_how_"
    col_pos_cont_raw = "Among_the_wet_containers_how_"
    col_wet_cont_raw = "Number_of_wet_containers_found"
    col_dry_cont_raw = "number_of_dry_contai_tentially_hold_water"
    
    col_lat = "_Location_latitude"
    col_lon = "_Location_longitude"

    date_col = "Date" if "Date" in df.columns else col_map_lower.get('date')
    if not date_col:
        for c in ['today', 'start', '_submission_time']:
             if c in col_map_lower: date_col = col_map_lower[c]; break

    # --- B. FILTERS ---
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    df_filtered = df.copy()

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
        st.warning("⚠️ CRITICAL: Could not find a column named 'Date'.")

    selected_zones, selected_subzones = [], []
    if col_zone and col_zone in df_filtered.columns:
        options = sorted(df_filtered[col_zone].dropna().unique().astype(str))
        selected_zones = st.sidebar.multiselect(f"Filter by Zone", options)
        if selected_zones: df_filtered = df_filtered[df_filtered[col_zone].astype(str).isin(selected_zones)]

    if col_subzone and col_subzone in df_filtered.columns:
        options = sorted(df_filtered[col_subzone].dropna().unique().astype(str))
        selected_subzones = st.sidebar.multiselect(f"Filter by SubZone", options)
        if selected_subzones: df_filtered = df_filtered[df_filtered[col_subzone].astype(str).isin(selected_subzones)]

    if col_street and col_street in df_filtered.columns:
        options = sorted(df_filtered[col_street].dropna().unique().astype(str))
        selected_streets = st.sidebar.multiselect(f"Filter by Street", options)
        if selected_streets: df_filtered = df_filtered[df_filtered[col_street].astype(str).isin(selected_streets)]

    # --- C. PRE-CALCULATIONS ---
    df_filtered['pos_house_calc'] = pd.to_numeric(df_filtered[col_pos_house_raw], errors='coerce').fillna(0) if col_pos_house_raw in df_filtered.columns else 0
    df_filtered['pos_cont_calc'] = pd.to_numeric(df_filtered[col_pos_cont_raw], errors='coerce').fillna(0) if col_pos_cont_raw in df_filtered.columns else 0
    df_filtered['wet_cont_calc'] = pd.to_numeric(df_filtered[col_wet_cont_raw], errors='coerce').fillna(0) if col_wet_cont_raw in df_filtered.columns else 0
    df_filtered['dry_cont_calc'] = pd.to_numeric(df_filtered[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_filtered.columns else 0

    # --- D. LOGIC BRANCHING ---
    display_count, positive_count, hi_val, ci_val, bi_val = 0, 0, 0, 0, 0

    if selected_key == 'intra':
        if col_premises and date_col:
            df_filtered['date_str_only'] = df_filtered[date_col].dt.date.astype(str)
            df_filtered['premise_clean'] = df_filtered[col_premises].apply(normalize_string)
            df_filtered['unique_premise_id'] = df_filtered['date_str_only'] + "_" + df_filtered['premise_clean']
            
            agg_dict = {
                'pos_house_calc': 'max', 
                'pos_cont_calc': 'sum', 
                'wet_cont_calc': 'sum',
                'dry_cont_calc': 'sum'
            }
            for c in [col_zone, col_lat, col_lon, col_premises, col_username]:
                if c and c in df_filtered.columns: agg_dict[c] = 'first'
            
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
            display_count, positive_count = total_unique_premises, positive_premises_count
        else:
            st.warning("⚠️ For Intra-Airport, 'Premises' and 'Date' columns are required.")
            df_for_graphs = df_filtered.copy()
    else:
        # PERI
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

    # --- E. METRICS ---
    label_hi = "Premises Index (PI)" if selected_key == 'intra' else "House Index (HI)"
    label_entries = "Unique Premises" if selected_key == 'intra' else "Total Entries"
    label_positive = "Positive Premises" if selected_key == 'intra' else "Positive Houses"
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(label_entries, display_count)
    m2.metric(label_positive, positive_count)
    m3.metric(label_hi, f"{hi_val:.2f}")
    m4.metric("Container Index (CI)", f"{ci_val:.2f}")
    m5.metric("Breteau Index (BI)", f"{bi_val:.2f}")

    # --- F. GRAPHS ---
    st.divider()
    c_graph, c_report = st.columns([1,1])
    show_graphs = c_graph.toggle("Show Graphical Analysis", value=False)
    
    # --- REPORT HELPER ---
    def generate_report_df(df_source, report_period_name):
        # 1. Fetch ID Data for Genus
        with st.spinner("Fetching Identification Data for Report..."):
            df_id_rep = load_kobo_data(current_config['id_url'])
            id_date_col = next((c for c in df_id_rep.columns if 'date' in c.lower() or 'today' in c.lower()), None)
            if id_date_col:
                df_id_rep[id_date_col] = pd.to_datetime(df_id_rep[id_date_col])
                df_id_rep['join_date'] = df_id_rep[id_date_col].dt.date
        
        unique_dates = sorted(df_source[date_col].dt.date.unique())
        report_data = []
        
        for i, day in enumerate(unique_dates, 1):
            df_day = df_source[df_source[date_col].dt.date == day]
            
            staffs = ", ".join(df_day[col_username].dropna().unique().astype(str)) if col_username in df_day else ""
            
            loc_list = ""
            if selected_key == 'intra' and col_premises and col_premises in df_day:
                loc_list = ", ".join(df_day[col_premises].dropna().unique().astype(str))
            elif selected_key == 'peri' and col_subzone and col_subzone in df_day:
                loc_list = ", ".join(df_day[col_subzone].dropna().unique().astype(str))
                
            d_dry = df_day['dry_cont_calc'].sum()
            d_wet = df_day['wet_cont_calc'].sum()
            
            if selected_key == 'intra':
                if col_premises in df_day.columns:
                    df_day['premise_clean'] = df_day[col_premises].apply(normalize_string)
                    df_day_grp = df_day.groupby('premise_clean').agg({'pos_house_calc':'max', 'pos_cont_calc':'sum', 'wet_cont_calc':'sum'})
                    
                    cnt_entries = len(df_day_grp)
                    cnt_pos = (df_day_grp['pos_house_calc'] > 0).sum()
                    d_pos_cont = df_day_grp['pos_cont_calc'].sum()
                    d_wet_sum = df_day_grp['wet_cont_calc'].sum()
                    
                    idx_hi = (cnt_pos / cnt_entries * 100) if cnt_entries > 0 else 0
                    idx_ci = (d_pos_cont / d_wet_sum * 100) if d_wet_sum > 0 else 0
                    idx_bi = (d_pos_cont / cnt_entries * 100) if cnt_entries > 0 else 0
                else:
                    cnt_entries, cnt_pos, idx_hi, idx_ci, idx_bi = 0, 0, 0, 0, 0
            else:
                cnt_entries = len(df_day)
                cnt_pos = (df_day['pos_house_calc'] > 0).sum()
                d_pos_cont = df_day['pos_cont_calc'].sum()
                
                idx_hi = (cnt_pos / cnt_entries * 100) if cnt_entries > 0 else 0
                idx_ci = (d_pos_cont / d_wet * 100) if d_wet > 0 else 0
                idx_bi = (d_pos_cont / cnt_entries * 100) if cnt_entries > 0 else 0

            genus_list = ""
            if not df_id_rep.empty and 'join_date' in df_id_rep.columns:
                day_id = df_id_rep[df_id_rep['join_date'] == day]
                g_col = next((c for c in day_id.columns if "Genus" in c), None)
                if g_col: genus_list = ", ".join(day_id[g_col].dropna().astype(str).tolist())

            report_data.append({
                "Serial No": i,
                "Date": day,
                "Count": cnt_entries,
                "Staffs": staffs,
                "Locations": loc_list,
                "Dry": int(d_dry),
                "Wet": int(d_wet),
                "Positives": int(cnt_pos),
                "HI/PI": round(idx_hi, 2),
                "CI": round(idx_ci, 2),
                "BI": round(idx_bi, 2),
                "Genuses": genus_list
            })
        return pd.DataFrame(report_data)

    # --- G. MONTHLY REPORT ---
    with st.expander("📅 Monthly Report Generator", expanded=False):
        if date_col:
            df_report = df.copy()
            df_report[date_col] = pd.to_datetime(df_report[date_col])
            df_report['Month_Year'] = df_report[date_col].dt.strftime('%Y-%m')
            
            # Explicit Calc Columns
            df_report['pos_house_calc'] = pd.to_numeric(df_report[col_pos_house_raw], errors='coerce').fillna(0) if col_pos_house_raw in df_report.columns else 0
            df_report['pos_cont_calc'] = pd.to_numeric(df_report[col_pos_cont_raw], errors='coerce').fillna(0) if col_pos_cont_raw in df_report.columns else 0
            df_report['wet_cont_calc'] = pd.to_numeric(df_report[col_wet_cont_raw], errors='coerce').fillna(0) if col_wet_cont_raw in df_report.columns else 0
            df_report['dry_cont_calc'] = pd.to_numeric(df_report[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_report.columns else 0

            available_months = sorted(df_report['Month_Year'].unique(), reverse=True)
            selected_month = st.selectbox("Select Month to Generate Report:", available_months)
            
            if selected_month:
                df_month = df_report[df_report['Month_Year'] == selected_month].copy()
                st.dataframe(generate_report_df(df_month, selected_month), hide_index=True, use_container_width=True)

    # --- G2. FORTNIGHT REPORT ---
    with st.expander("📆 Fortnight Report Generator", expanded=False):
        if date_col:
            df_ft = df.copy()
            df_ft[date_col] = pd.to_datetime(df_ft[date_col])
            
            # Explicit Calc Columns
            df_ft['pos_house_calc'] = pd.to_numeric(df_ft[col_pos_house_raw], errors='coerce').fillna(0) if col_pos_house_raw in df_ft.columns else 0
            df_ft['pos_cont_calc'] = pd.to_numeric(df_ft[col_pos_cont_raw], errors='coerce').fillna(0) if col_pos_cont_raw in df_ft.columns else 0
            df_ft['wet_cont_calc'] = pd.to_numeric(df_ft[col_wet_cont_raw], errors='coerce').fillna(0) if col_wet_cont_raw in df_ft.columns else 0
            df_ft['dry_cont_calc'] = pd.to_numeric(df_ft[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_ft.columns else 0

            # Label Logic: 1-15 is First Half, 16+ is Second Half
            df_ft['Month_Str'] = df_ft[date_col].dt.strftime('%B %Y')
            df_ft['Day'] = df_ft[date_col].dt.day
            df_ft['Fortnight_Label'] = df_ft.apply(lambda x: f"First Half {x['Month_Str']}" if x['Day'] <= 15 else f"Second Half {x['Month_Str']}", axis=1)
            
            # Sort options chronologically
            df_ft = df_ft.sort_values(by=date_col, ascending=False)
            available_fortnights = df_ft['Fortnight_Label'].unique()
            
            selected_ft = st.selectbox("Select Fortnight:", available_fortnights)
            
            if selected_ft:
                df_selected_ft = df_ft[df_ft['Fortnight_Label'] == selected_ft].copy()
                st.dataframe(generate_report_df(df_selected_ft, selected_ft), hide_index=True, use_container_width=True)

    # --- H. VISUALS ---
    if show_graphs:
        show_zone_graph = (len(selected_zones) == 0) and (len(selected_subzones) == 0)
        show_subzone_graph = (len(selected_subzones) == 0)

        def get_grouped_data(groupby_col):
            aggs = {'pos_cont_calc': 'sum', 'wet_cont_calc': 'sum'}
            if selected_key == 'intra':
                aggs[groupby_col] = 'count'; aggs['is_positive_premise'] = 'sum'
            else:
                aggs[groupby_col] = 'count'; aggs['is_positive_house'] = 'sum'
            g = df_for_graphs.groupby(groupby_col).agg(aggs).rename(columns={groupby_col: 'Denominator'})
            if selected_key == 'intra': g['HI'] = (g['is_positive_premise'] / g['Denominator']) * 100
            else: g['HI'] = (g['is_positive_house'] / g['Denominator']) * 100
            g['CI'] = (g['pos_cont_calc'] / g['wet_cont_calc'].replace(0, 1)) * 100 
            g['BI'] = (g['pos_cont_calc'] / g['Denominator']) * 100
            return g.reset_index()

        with st.expander(f"📊 View {label_hi} Graphs", expanded=False):
            if show_zone_graph and col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'HI', f"{label_hi} by Zone", 'HI'), use_container_width=True)
            if selected_key == 'peri' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_subzone), col_subzone, 'HI', f"{label_hi} by SubZone", 'HI'), use_container_width=True)

        with st.expander("📊 View Container Index (CI) Graphs", expanded=False):
            if show_zone_graph and col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'CI', "Container Index by Zone", 'CI'), use_container_width=True)
            if selected_key == 'peri' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_subzone), col_subzone, 'CI', "Container Index by SubZone", 'CI'), use_container_width=True)

        with st.expander("📊 View Breteau Index (BI) Graphs", expanded=False):
            if show_zone_graph and col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'BI', "Breteau Index by Zone", 'BI'), use_container_width=True)
            if selected_key == 'peri' and show_subzone_graph and col_subzone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_subzone), col_subzone, 'BI', "Breteau Index by SubZone", 'BI'), use_container_width=True)

    # --- I. GEO MAP ---
    st.divider()
    with st.expander("🌍 View Geo-Spatial Mapping (Map)", expanded=False):
        if col_lat in df_for_graphs.columns and col_lon in df_for_graphs.columns:
            map_df = df_for_graphs.dropna(subset=[col_lat, col_lon]).copy()
            if not map_df.empty:
                avg_lat = map_df[col_lat].mean()
                avg_lon = map_df[col_lon].mean()
                m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
                for idx, row in map_df.iterrows():
                    larvae_count = int(row['pos_house_calc'])
                    if larvae_count == 0:
                        color = '#00ff00'; fill_opacity = 0.5
                    else:
                        color = '#ff0000'; fill_opacity = min(1.0, 0.4 + (larvae_count * 0.1))
                    popup_text = f"Larvae: {larvae_count}"
                    folium.CircleMarker(
                        location=[row[col_lat], row[col_lon]], radius=6,
                        color=color, fill=True, fill_color=color, fill_opacity=fill_opacity,
                        tooltip=popup_text
                    ).add_to(m)
                st_folium(m, width=None, height=500)
            else: st.warning("No GPS data.")
        else: st.warning("GPS columns missing.")

    # --- J. LARVAE IDENTIFICATION ---
    st.divider()
    st.markdown("### 🔬 Larvae Identification")
    with st.expander("View Larvae Identification Data", expanded=False):
        with st.spinner('Fetching ID data...'):
            df_id = load_kobo_data(current_config['id_url'])
        
        if not df_id.empty:
            col_map_id = {c.lower(): c for c in df_id.columns}
            date_col_id = next((c for c in df_id.columns if c in ['Date', 'today', 'date']), None)
            
            col_address_id = col_map_id.get('address') or col_map_id.get('location') or col_map_id.get('premise') or col_map_id.get('premises') or col_map_id.get('streetname')
            possible_img_cols = ["Attach the microscopic image of the larva _URL", "Attach the microscopic image of the larva_URL", "image_url", "url"]
            col_img = next((c for c in possible_img_cols if c in df_id.columns), None)
            
            col_genus = "Select the Genus:"
            col_species = "Select the Species:"
            col_container = "Type of container the sample was collected from"

            if date_col_id:
                df_id[date_col_id] = pd.to_datetime(df_id[date_col_id])
                if start_date and end_date: 
                    mask_id = (df_id[date_col_id].dt.date >= start_date) & (df_id[date_col_id].dt.date <= end_date)
                    df_id = df_id.loc[mask_id]

            c1, c2 = st.columns(2)
            if col_genus in df_id.columns:
                c1.write("#### Genus Distribution")
                genus_counts = df_id[col_genus].value_counts().reset_index()
                genus_counts.columns = ['Genus', 'Count']
                fig_pie_g = px.pie(genus_counts, values='Count', names='Genus', hole=0.4)
                c1.plotly_chart(fig_pie_g, use_container_width=True)
            
            if col_container in df_id.columns:
                c2.write("#### Container Distribution")
                cont_data = df_id[df_id[col_container].notna() & (df_id[col_container] != "")]
                cont_counts = cont_data[col_container].value_counts().reset_index()
                cont_counts.columns = ['Container Type', 'Count']
                fig_pie_c = px.pie(cont_counts, values='Count', names='Container Type', hole=0.4)
                c2.plotly_chart(fig_pie_c, use_container_width=True)

            df_display = pd.DataFrame()
            df_display['Serial No'] = range(1, 1 + len(df_id))
            df_display['Address'] = df_id[col_address_id] if col_address_id in df_id.columns else 'N/A'
            df_display['Date'] = df_id[date_col_id].dt.date if date_col_id in df_id.columns else 'N/A'
            df_display['Genus'] = df_id[col_genus] if col_genus in df_id.columns else 'N/A'
            df_display['Species'] = df_id[col_species] if col_species in df_id.columns else 'N/A'
            if col_img:
                df_display['Original Image URL'] = df_id[col_img]
                df_display['Thumbnail'] = df_id[col_img].apply(get_thumbnail_url)
            else:
                df_display['Original Image URL'] = None; df_display['Thumbnail'] = None

            st.info("💡 **Select a row** to view the **Mega-Size Image**.")
            event = st.dataframe(
                df_display,
                column_config={"Thumbnail": st.column_config.ImageColumn("Microscopic Image", width="large"), "Original Image URL": None},
                hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
            )
            if len(event.selection.rows) > 0:
                show_image_popup(df_display.iloc[event.selection.rows[0]])
        else: st.info("No identification data available.")

    # --- K. RAW DATA ---
    st.divider()
    with st.expander("📂 View Raw Data Table", expanded=False):
        st.dataframe(df_filtered)

else:
    st.info("No data found. Please check your Kobo connection or selection.")
