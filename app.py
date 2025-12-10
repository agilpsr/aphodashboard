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
import base64 # <-- NEW IMPORT FOR BACKGROUND

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="APHO Tiruchirappalli Dashboard", layout="wide")

# --- STAFF NAME MAPPING ---
STAFF_NAMES = {
    'abhiguptak': 'Abhishek Gupta',
    'arunhealthinspector': 'Arun',
    'chandru1426': 'Chandru',
    'dineshg': 'Dinesh',
    'iyyappank': 'Iyyapan',
    'kalaig': 'Kalaichelvan',
    'kishanth': 'Kishanth',
    'nitesh9896': 'Nitesh',
    'prabhahi': 'Prabhakaran',
    'rajaramha': 'Rajaram',
    'ramnareshfw': 'Ram naresh',
    'siddhik23': 'siddhik',
    'simbuha': 'Silambarasan',
    'souravmalik7055': 'sourav MAlik'
}

# --- CONFIGURATION DICTIONARY ---
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

# --- HELPER FUNCTIONS ---
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

@st.cache_data(ttl=300)
def load_kobo_data(url):
    try:
        if "KOBO_TOKEN" in st.secrets:
            token = st.secrets["KOBO_TOKEN"]
        else:
            token = "48554147c1847ddfe4c1c987a54b4196a03c1d9c"
        headers = {"Authorization": f"Token {token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), sep=None, engine='python', on_bad_lines='skip')
    except:
        return pd.DataFrame()

def plot_metric_bar(data, x_col, y_col, title, color_col):
    fig = px.bar(data, x=x_col, y=y_col, title=title, text=y_col, color=color_col, color_continuous_scale='RdYlGn_r', range_color=[0, 20])
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(coloraxis_showscale=False)
    return fig

def normalize_string(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def get_thumbnail_url(original_url):
    if not isinstance(original_url, str) or not original_url.startswith("http"): return None
    return f"https://wsrv.nl/?url={urllib.parse.quote(original_url)}&w=400&q=80"

@st.dialog("Microscopic View", width="large")
def show_image_popup(row_data):
    st.subheader(f"{row_data['Genus']} ({row_data['Species']})")
    c1, c2 = st.columns(2)
    c1.info(f"📍 **Address:** {row_data['Address']}")
    c2.warning(f"📅 **Date:** {row_data['Date']}")
    if row_data['Original Image URL'] and str(row_data['Original Image URL']).startswith('http'):
        st.image(row_data['Original Image URL'], caption="Full Resolution", use_container_width=True)
    else:
        st.error("Image not available.")

# --- NEW HELPER FOR BACKGROUND IMAGE ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- MAIN DASHBOARD LOGIC ---
def render_dashboard(selected_key):
    # Sidebar Navigation Back Button
    if st.sidebar.button("🏠 Back to Home"):
        st.session_state['page'] = 'home'
        st.rerun()
    
    current_config = SECTION_CONFIG[selected_key]
    st.title(current_config['title'])

    with st.spinner('Fetching Surveillance data...'):
        df = load_kobo_data(current_config['surv_url'])

    if df.empty:
        st.info("No data found or error loading Kobo data.")
        return

    # --- A. COLUMN MAPPING ---
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

    selected_zones = []
    if col_zone and col_zone in df_filtered.columns:
        options = sorted(df_filtered[col_zone].dropna().unique().astype(str))
        selected_zones = st.sidebar.multiselect(f"Filter by Zone", options)
        if selected_zones: df_filtered = df_filtered[df_filtered[col_zone].astype(str).isin(selected_zones)]

    selected_subzones = []
    if col_subzone and col_subzone in df_filtered.columns:
        options = sorted(df_filtered[col_subzone].dropna().unique().astype(str))
        selected_subzones = st.sidebar.multiselect(f"Filter by SubZone", options)
        if selected_subzones: df_filtered = df_filtered[df_filtered[col_subzone].astype(str).isin(selected_subzones)]

    if col_street and col_street in df_filtered.columns:
        options = sorted(df_filtered[col_street].dropna().unique().astype(str))
        selected_streets = st.sidebar.multiselect(f"Filter by Street", options)
        if selected_streets: df_filtered = df_filtered[df_filtered[col_street].astype(str).isin(selected_streets)]

    # --- C. PRE-CALCULATIONS ---
    for col, raw_col in [('pos_house_calc', col_pos_house_raw), ('pos_cont_calc', col_pos_cont_raw), ('wet_cont_calc', col_wet_cont_raw)]:
        df_filtered[col] = pd.to_numeric(df_filtered[raw_col], errors='coerce').fillna(0) if raw_col in df_filtered.columns else 0
    df_filtered['dry_cont_calc'] = pd.to_numeric(df_filtered[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_filtered.columns else 0

    # --- D. LOGIC BRANCHING ---
    display_count, positive_count, hi_val, ci_val, bi_val = 0, 0, 0, 0, 0

    if selected_key == 'intra':
        if col_premises and date_col:
            df_filtered['unique_premise_id'] = df_filtered[date_col].dt.date.astype(str) + "_" + df_filtered[col_premises].apply(normalize_string)
            agg_dict = {'pos_house_calc': 'max', 'pos_cont_calc': 'sum', 'wet_cont_calc': 'sum', 'dry_cont_calc': 'sum'}
            for c in [col_zone, col_lat, col_lon, col_premises, col_username]:
                if c and c in df_filtered.columns: agg_dict[c] = 'first'
            df_grouped = df_filtered.groupby('unique_premise_id', as_index=False).agg(agg_dict)
            
            total_unique_premises = df_grouped['unique_premise_id'].nunique()
            positive_premises_count = (df_grouped['pos_house_calc'] > 0).sum()
            hi_val = (positive_premises_count / total_unique_premises * 100) if total_unique_premises > 0 else 0
            ci_val = (df_grouped['pos_cont_calc'].sum() / df_grouped['wet_cont_calc'].sum() * 100) if df_grouped['wet_cont_calc'].sum() > 0 else 0
            bi_val = (df_grouped['pos_cont_calc'].sum() / total_unique_premises * 100) if total_unique_premises > 0 else 0
            
            df_for_graphs = df_grouped.copy()
            df_for_graphs['is_positive_premise'] = (df_for_graphs['pos_house_calc'] > 0).astype(int)
            display_count, positive_count = total_unique_premises, positive_premises_count
        else:
            df_for_graphs = df_filtered.copy()
    else:
        # PERI
        display_count = len(df_filtered)
        df_filtered['is_positive_house'] = df_filtered['pos_house_calc'].apply(lambda x: 1 if x > 0 else 0)
        positive_count = df_filtered['is_positive_house'].sum()
        if display_count > 0:
            hi_val = (positive_count / display_count) * 100
            ci_val = (df_filtered['pos_cont_calc'].sum() / df_filtered['wet_cont_calc'].sum() * 100) if df_filtered['wet_cont_calc'].sum() > 0 else 0
            bi_val = (df_filtered['pos_cont_calc'].sum() / display_count * 100)
        df_for_graphs = df_filtered.copy()

    # --- E. METRICS ---
    label_hi = "Premises Index (PI)" if selected_key == 'intra' else "House Index (HI)"
    label_entries = "Unique Premises" if selected_key == 'intra' else "Total Entries"
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(label_entries, display_count)
    m2.metric("Positive Found", positive_count)
    m3.metric(label_hi, f"{hi_val:.2f}")
    m4.metric("Container Index (CI)", f"{ci_val:.2f}")
    m5.metric("Breteau Index (BI)", f"{bi_val:.2f}")

    # --- F. GRAPHS ---
    st.divider()
    c_graph, c_report = st.columns([1,1])
    show_graphs = c_graph.toggle("Show Graphical Analysis", value=False)

    # --- REPORT GENERATOR FUNC ---
    def generate_report_df(df_source, report_period_name):
        with st.spinner("Fetching Identification Data..."):
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
            street_list = ""
            if selected_key == 'intra' and col_premises and col_premises in df_day:
                loc_list = ", ".join(df_day[col_premises].dropna().unique().astype(str))
            elif selected_key == 'peri' and col_subzone and col_subzone in df_day:
                loc_list = ", ".join(df_day[col_subzone].dropna().unique().astype(str))
            if col_street and col_street in df_day.columns:
                street_list = ", ".join(df_day[col_street].dropna().astype(str).unique())
                
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
                else: cnt_entries, cnt_pos, idx_hi, idx_ci, idx_bi = 0, 0, 0, 0, 0
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
                "Serial No": i, "Date": day, "Count": cnt_entries, "Staffs": staffs,
                "Locations": loc_list, "Streets": street_list, "Dry": int(d_dry), "Wet": int(d_wet),
                "Positives": int(cnt_pos), "HI/PI": round(idx_hi, 2), "CI": round(idx_ci, 2),
                "BI": round(idx_bi, 2), "Genuses": genus_list
            })
        return pd.DataFrame(report_data)

    # --- STAFF PERFORMANCE REPORT ---
    with st.expander("👮 Staff Performance Report", expanded=False):
        if col_username in df_filtered.columns:
            staff_group = df_filtered.groupby(col_username)
            staff_perf = pd.DataFrame(staff_group[date_col].apply(lambda x: x.dt.date.nunique()))
            staff_perf.columns = ['Days Worked']
            
            def get_staff_name(u):
                return STAFF_NAMES.get(str(u).strip().lower(), u)
            staff_perf['Name'] = staff_perf.index.map(get_staff_name)

            staff_perf['Total Entries'] = staff_group[col_username].count()
            staff_perf['Positive Found'] = staff_group['pos_house_calc'].apply(lambda x: (x > 0).sum())
            staff_perf['Positive Containers'] = staff_group['pos_cont_calc'].sum()
            total_searched = staff_group['wet_cont_calc'].sum()
            staff_perf['Container Index'] = (staff_perf['Positive Containers'] / total_searched.replace(0, 1) * 100).round(2)
            
            try:
                with st.spinner("Syncing Larvae ID Data..."):
                    df_id_sync = load_kobo_data(current_config['id_url'])
                    if not df_id_sync.empty and col_username in df_id_sync.columns:
                        df_id_sync['clean_user'] = df_id_sync[col_username].astype(str).str.strip().str.lower()
                        id_counts = df_id_sync.groupby('clean_user').size().rename('Larvae ID Entries')
                        temp_index = staff_perf.index.astype(str).str.strip().str.lower()
                        staff_perf['Larvae ID Entries'] = temp_index.map(id_counts).fillna(0).astype(int)
                    else: staff_perf['Larvae ID Entries'] = 0
            except: staff_perf['Larvae ID Entries'] = 0

            staff_perf = staff_perf.reset_index()
            staff_perf.index += 1
            staff_perf.index.name = 'S.No'
            staff_perf = staff_perf.reset_index()
            final_cols_staff = ['S.No', 'Name', 'Days Worked', 'Total Entries', 'Positive Found', 'Positive Containers', 'Container Index', 'Larvae ID Entries']
            staff_final = staff_perf[[c for c in final_cols_staff if c in staff_perf.columns]]
            st.dataframe(staff_final, use_container_width=True)
            st.download_button("Download Staff Excel", to_excel(staff_final), "Staff_Performance.xlsx")
        else: st.warning("Username column not found.")

    # --- MONTHLY & FORTNIGHTLY REPORTS ---
    c_month, c_fort = st.columns(2)
    with c_month:
        with st.expander("📅 Monthly Report", expanded=False):
            if date_col:
                df_rep_raw = df.copy()
                df_rep_raw[date_col] = pd.to_datetime(df_rep_raw[date_col])
                # Re-apply calcs
                for col, raw_col in [('pos_house_calc', col_pos_house_raw), ('pos_cont_calc', col_pos_cont_raw), ('wet_cont_calc', col_wet_cont_raw)]:
                    df_rep_raw[col] = pd.to_numeric(df_rep_raw[raw_col], errors='coerce').fillna(0) if raw_col in df_rep_raw.columns else 0
                df_rep_raw['dry_cont_calc'] = pd.to_numeric(df_rep_raw[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_rep_raw.columns else 0
                
                df_rep_raw['Month_Year'] = df_rep_raw[date_col].dt.strftime('%Y-%m')
                sel_mon = st.selectbox("Select Month:", sorted(df_rep_raw['Month_Year'].unique(), reverse=True))
                if sel_mon:
                    df_m = df_rep_raw[df_rep_raw['Month_Year'] == sel_mon].copy()
                    rep_df = generate_report_df(df_m, sel_mon)
                    st.dataframe(rep_df, hide_index=True)
                    st.download_button("Download Excel", to_excel(rep_df), "Monthly.xlsx")

    with c_fort:
        with st.expander("📆 Fortnight Report", expanded=False):
            if date_col:
                df_ft = df.copy()
                df_ft[date_col] = pd.to_datetime(df_ft[date_col])
                # Re-apply calcs
                for col, raw_col in [('pos_house_calc', col_pos_house_raw), ('pos_cont_calc', col_pos_cont_raw), ('wet_cont_calc', col_wet_cont_raw)]:
                    df_ft[col] = pd.to_numeric(df_ft[raw_col], errors='coerce').fillna(0) if raw_col in df_ft.columns else 0
                df_ft['dry_cont_calc'] = pd.to_numeric(df_ft[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_ft.columns else 0

                df_ft['Month_Str'] = df_ft[date_col].dt.strftime('%B %Y')
                df_ft['Label'] = df_ft.apply(lambda x: f"First Half {x['Month_Str']}" if x[date_col].day <= 15 else f"Second Half {x['Month_Str']}", axis=1)
                df_ft = df_ft.sort_values(by=date_col, ascending=False)
                sel_ft = st.selectbox("Select Fortnight:", df_ft['Label'].unique())
                if sel_ft:
                    df_sft = df_ft[df_ft['Label'] == sel_ft].copy()
                    ft_rep = generate_report_df(df_sft, sel_ft)
                    st.dataframe(ft_rep, hide_index=True)
                    st.download_button("Download Excel", to_excel(ft_rep), "Fortnightly.xlsx")

    # --- GRAPHS ---
    if show_graphs:
        with st.expander("📊 View Graphs", expanded=True):
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
                return g.reset_index()

            if col_zone in df_for_graphs.columns:
                st.plotly_chart(plot_metric_bar(get_grouped_data(col_zone), col_zone, 'HI', f"{label_hi} by Zone", 'HI'), use_container_width=True)

    # --- MAP ---
    with st.expander("🌍 Map", expanded=False):
        if col_lat in df_for_graphs.columns and col_lon in df_for_graphs.columns:
            map_df = df_for_graphs.dropna(subset=[col_lat, col_lon]).copy()
            if not map_df.empty:
                m = folium.Map(location=[map_df[col_lat].mean(), map_df[col_lon].mean()], zoom_start=13)
                for _, row in map_df.iterrows():
                    color = '#00ff00' if row['pos_house_calc'] == 0 else '#ff0000'
                    folium.CircleMarker([row[col_lat], row[col_lon]], radius=6, color=color, fill=True, fill_color=color).add_to(m)
                st_folium(m, height=400)

    # --- LARVAE ID TABLE ---
    with st.expander("🔬 Larvae Identification Data", expanded=False):
        df_id = load_kobo_data(current_config['id_url'])
        if not df_id.empty:
            st.dataframe(df_id, use_container_width=True)
        else: st.info("No ID Data")

# --- HOME PAGE LOGIC (UPDATED WITH FULL BACKGROUND) ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_home_page():
    # 1. SET FULL BACKGROUND IMAGE USING CSS
    try:
        bin_str = get_base64_of_bin_file("logo.png")
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Add a semi-transparent white backing to make text readable */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            padding: 3rem;
            margin-top: 5rem;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        st.warning("Background image 'logo.png' not found on GitHub.")

    # 2. HEADER AND BUTTONS
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>AIRPORT HEALTH ORGANISATION</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1E3A8A;'>TIRUCHIRAPPALLI INTERNATIONAL AIRPORT</h3>", unsafe_allow_html=True)
    
    st.divider()
    
    # Center buttons using columns with offsets
    _, col_buttons, _ = st.columns([1, 2, 1])
    with col_buttons:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🦟 Outside Field Activities (Peri)", use_container_width=True, type="primary"):
            st.session_state['page'] = 'peri'
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✈️ Inside Field Activities (Intra)", use_container_width=True, type="primary"):
            st.session_state['page'] = 'intra'
            st.rerun()

# --- APP ENTRY POINT ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    render_home_page()
else:
    render_dashboard(st.session_state['page'])
