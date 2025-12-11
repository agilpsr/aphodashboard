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
import datetime

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="APHO Tiruchirappalli Dashboard", layout="wide", page_icon="🦟")

# --- INITIALIZE SESSION STATE ---
if 'reports' not in st.session_state:
    st.session_state['reports'] = []

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# --- STAFF NAME MAPPING ---
STAFF_NAMES = {
    'abhiguptak': 'Abhishek Gupta', 'arunhealthinspector': 'Arun', 'chandru1426': 'Chandru',
    'dineshg': 'Dinesh', 'iyyappank': 'Iyyapan', 'kalaig': 'Kalaichelvan',
    'kishanth': 'Kishanth', 'nitesh9896': 'Nitesh', 'prabhahi': 'Prabhakaran',
    'rajaramha': 'Rajaram', 'ramnareshfw': 'Ram naresh', 'siddhik23': 'siddhik',
    'simbuha': 'Silambarasan', 'souravmalik7055': 'sourav MAlik'
}

# --- CONFIGURATION DICTIONARY ---
SECTION_CONFIG = {
    'peri': {
        'title': 'Peri-Airport Larvae Surveillance',
        'icon': '🦟',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aXM5aSjVEJTgt6z5qMvNFe/export-settings/es9zUAYU5f8PqCokaZSuPmg/data.csv',
        'id_url': 'https://kf.kobotoolbox.org/api/v2/assets/afU6pGvUzT8Ao4pAeX54QY/export-settings/esinGxnSujLzanzmAv6Mdb4/data.csv'
    },
    'intra': {
        'title': 'Intra-Airport Larvae Surveillance',
        'icon': '🏢',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aEdcSxvmrBuXBmzXNECtjr/export-settings/esgYdEaEk79Y69k56abNGdW/data.csv',
        'id_url': 'https://kf.kobotoolbox.org/api/v2/assets/anN9HTYvmLRTorb7ojXs5A/export-settings/esLiqyb8KpPfeMX4ZnSoXSm/data.csv'
    },
    'flights': {
        'title': 'International Flights Screened',
        'icon': '✈️',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aHdVBAGwFvJwpTaATAZN8v/export-settings/esFbR4cbEQXToCUwLfFGbV4/data.csv',
        'id_url': None
    },
    'anti_larval': {
        'title': 'Anti-Larval Action Reports',
        'icon': '🛡️',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/az3jC73Chq5yPKMhM73eMm/export-settings/esJCVJu8sXKCxUfywczgC4x/data.csv',
        'id_url': None
    },
    'sanitary': {
        'title': 'Sanitary & Toilet Inspection Reports',
        'icon': '🧹',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aCn73Fp8jaAPz3TcG5Y3jJ/export-settings/esBxawoybCQnoWtYJ5mbsEw/data.csv',
        'id_url': None
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
            token = "48554147c1c987a54b4196a03c1d9c"
        headers = {"Authorization": f"Token {token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), sep=None, engine='python', on_bad_lines='skip')
    except:
        return pd.DataFrame()

def plot_metric_bar(data, x_col, y_col, title, color_col, range_max=None):
    if data.empty: return None
    r_max = range_max if range_max else (data[y_col].max() * 1.1 if data[y_col].max() > 0 else 20)
    fig = px.bar(data, x=x_col, y=y_col, title=title, text=y_col, color=color_col, 
                 color_continuous_scale='RdYlGn_r', range_color=[0, 10]) # Green to Red
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=14, color="black"), 
        coloraxis_showscale=False
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    return fig

def normalize_string(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def get_thumbnail_url(original_url):
    if not isinstance(original_url, str) or not original_url.startswith("http"): return None
    return f"https://wsrv.nl/?url={urllib.parse.quote(original_url)}&w=150&q=60"

def get_high_res_url(original_url):
    if not isinstance(original_url, str) or not original_url.startswith("http"): return None
    return f"https://wsrv.nl/?url={urllib.parse.quote(original_url)}&w=1600&q=90"

@st.dialog("🔬 Larvae Microscopic View", width="large")
def show_image_popup(row_data):
    col_genus = "Select the Genus:"
    col_species = "Select the Species:"
    col_container = "Type of container the sample was collected from"
    col_submitted = "_submitted_by"
    
    genus = row_data.get(col_genus, 'N/A')
    species = row_data.get(col_species, 'N/A')
    container = row_data.get(col_container, 'N/A')
    submitted_by = row_data.get(col_submitted, 'N/A')
    address = row_data.get('Calculated_Address', 'N/A')
    original_url = row_data.get('Original_Image_URL')

    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1E3A8A;'>
            <h3 style='margin:0; color:#1E3A8A;'>Sample Details</h3>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.info(f"**📍 Address:**\n{address}")
    with c2: st.warning(f"**🪣 Container:**\n{container}")
    with c3: st.success(f"**👤 Submitted By:**\n{submitted_by}")

    st.markdown(f"### {genus} ({species})")

    if original_url:
        high_res_link = get_high_res_url(original_url)
        st.image(high_res_link, caption="Full Resolution Microscopic View", use_container_width=True)
    else:
        st.warning("⚠️ No image URL found.")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- FILE HANDLERS ---
def get_pdf_bytes(filename):
    """Reads a local PDF file into bytes for download/viewing."""
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        return None

# --- UTILS ---
def find_column_by_keywords(df, keywords):
    """Robustly finds a column name containing any of the keywords."""
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None

# --- GLOBAL REPORT FUNCTION ---
def generate_report_df(df_source, date_col, col_username, selected_key, col_premises, col_subzone, col_street, current_config):
    with st.spinner("Fetching Identification Data..."):
        if current_config.get('id_url'):
            df_id_rep = load_kobo_data(current_config['id_url'])
        else:
            df_id_rep = pd.DataFrame()
            
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
            street_list = ", ".join(df_day[col_street].dropna().unique().astype(str))
            
        d_dry = df_day['dry_cont_calc'].sum()
        d_wet = df_day['wet_cont_calc'].sum()
        
        cnt_entries, cnt_pos, idx_hi, idx_ci, idx_bi = 0, 0, 0, 0, 0
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

# --- SUMMARY GENERATOR ---
def generate_narrative_summary(df, selected_key, date_col, col_street, col_subzone, col_premises):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['Month_Year'] = df[date_col].dt.to_period('M')
    
    all_months = sorted(df['Month_Year'].unique())
    if not all_months: return "No data."
    
    curr_month = all_months[-1]
    prev_month = all_months[-2] if len(all_months) > 1 else None
    
    df_curr = df[df['Month_Year'] == curr_month]
    df_prev = df[df['Month_Year'] == prev_month] if prev_month else pd.DataFrame()
    
    narrative = [f"#### 📝 Executive Summary ({curr_month.strftime('%B %Y')})"]
    
    if col_street and col_street in df_curr.columns:
        street_stats = df_curr.groupby(col_street).agg(
            pos=('pos_house_calc', lambda x: (x>0).sum()),
            total=('pos_house_calc', 'count')
        )
        street_stats['HI'] = (street_stats['pos'] / street_stats['total'] * 100)
        top_streets = street_stats[street_stats['pos'] > 0].sort_values('HI', ascending=False).head(5)
        
        if not top_streets.empty:
            s_list = ", ".join([f"**{idx}** ({row['HI']:.1f}%)" for idx, row in top_streets.iterrows()])
            narrative.append(f"**🔴 High Risk Streets (House Index):** {s_list}")
        else:
            narrative.append(f"**🟢 Streets:** No positive streets found in {curr_month.strftime('%B')}.")
            
    loc_col = col_subzone if selected_key == 'peri' else col_premises
    if loc_col and loc_col in df_curr.columns:
        cont_stats = df_curr.groupby(loc_col)['pos_cont_calc'].sum().sort_values(ascending=False)
        high_cont = cont_stats[cont_stats > 0].head(3)
        if not high_cont.empty:
            c_list = ", ".join([f"**{idx}** ({int(val)} containers)" for idx, val in high_cont.items()])
            narrative.append(f"**🪣 High Positive Containers:** Found in {c_list}.")
            
    if prev_month and loc_col and loc_col in df_curr.columns:
        def calc_hi(d, g_col):
            g = d.groupby(g_col)
            return (g['pos_house_calc'].apply(lambda x: (x>0).sum()) / g[g_col].count() * 100).fillna(0)

        hi_c = calc_hi(df_curr, loc_col)
        hi_p = calc_hi(df_prev, loc_col)
        comp = pd.DataFrame({'Curr': hi_c, 'Prev': hi_p}).fillna(0)
        comp['Diff'] = comp['Curr'] - comp['Prev']
        inc = comp[comp['Diff'] > 0].sort_values('Diff', ascending=False).head(3)
        dec = comp[comp['Diff'] < 0].sort_values('Diff', ascending=True).head(3)
        if not inc.empty:
            i_str = ", ".join([f"**{idx}** (+{val:.1f}%)" for idx, val in inc['Diff'].items()])
            narrative.append(f"**📈 Worsening Trends (vs {prev_month.strftime('%b')}):** Indices increased in {i_str}.")
        if not dec.empty:
            d_str = ", ".join([f"**{idx}** ({val:.1f}%)" for idx, val in dec['Diff'].items()])
            narrative.append(f"**📉 Improving Trends:** Indices reduced in {d_str}.")
            
    return "\n\n".join(narrative)

# --- AUTHENTICATION ---
def check_password():
    """Returns `True` if the user had a correct password."""
    def password_entered():
        if st.session_state["password"] == "Aphotrz@2025":
            st.session_state["authenticated"] = True
            del st.session_state["password"] 
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated", False):
        return True

    st.markdown("""
        <style>
        .stApp { background-color: #f8fafc; }
        .login-box {
            max-width: 400px;
            margin: 100px auto;
            padding: 30px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-box"><h2>🔐 Access Restricted</h2><p>Please enter your credentials.</p></div>', unsafe_allow_html=True)
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    
    if "authenticated" in st.session_state and not st.session_state["authenticated"]:
        st.error("😕 Incorrect password")
        
    return False

# --- CUSTOM CSS INJECTION ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            font-size: 18px; /* Increased font size */
            font-weight: 500; /* Bolder */
            color: #0f172a;
        }

        /* Gradient Header */
        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 2.5rem;
            border-radius: 0 0 20px 20px;
            color: white;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .main-header h1 {
            font-weight: 800;
            font-size: 3rem !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            color: white !important;
        }
        .main-header h3 {
             color: #e2e8f0 !important;
             font-weight: 600 !important;
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
            text-align: center;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
            color: #475569;
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            color: #1e293b;
            font-weight: 800;
        }

        /* Buttons */
        div.stButton > button {
            width: 100%;
            height: 110px;
            font-size: 24px !important;
            font-weight: 700 !important;
            color: white !important;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            border: none !important;
            border-radius: 16px !important;
            box-shadow: 0 6px 12px rgba(30, 58, 138, 0.2) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        div.stButton > button:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 20px rgba(30, 58, 138, 0.3) !important;
        }

        /* Table */
        div[data-testid="stDataFrame"] {
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font-size: 16px;
        }

        </style>
    """, unsafe_allow_html=True)

@st.dialog("🌍 Expanded Geo-Spatial Map", width="large")
def show_large_map(m):
    # Added unique key to st_folium inside dialog to prevent ID collision
    st_folium(m, height=700, width=1200, use_container_width=True, key="large_map_dialog")

# --- MAIN DASHBOARD RENDERER ---
def render_dashboard(selected_key):
    inject_custom_css()
    
    current_config = SECTION_CONFIG[selected_key]
    
    st.markdown(f"""
        <div class="main-header">
            <h1>{current_config.get('icon', '')} {current_config['title']}</h1>
        </div>
    """, unsafe_allow_html=True)

    # --- ACTION REPORTS & SANITARY REPORTS LOGIC ---
    if selected_key in ['anti_larval', 'sanitary']:
        with st.spinner('Fetching Reports...'):
            df_action = load_kobo_data(current_config['surv_url'])
            
        if df_action.empty:
            st.info("No reports found.")
            st.stop()
            
        st.subheader("Reports Repository")
        
        column_config = {}
        clean_cols = {c.strip().lower(): c for c in df_action.columns}
        
        if selected_key == 'anti_larval':
            target_key = "upload action taken report (pdf) _url"
            pdf_col = clean_cols.get(target_key)
            if not pdf_col: pdf_col = next((c for c in df_action.columns if 'pdf' in c.lower() and 'url' in c.lower()), None)
            if pdf_col: column_config[pdf_col] = st.column_config.LinkColumn("Action Report", display_text="📥 Download PDF")

        elif selected_key == 'sanitary':
            target_sanitary = "upload sanitary inspection report (pdf) _url"
            target_toilet = "upload toilet inspection report(pdf) _url"
            sanitary_col = clean_cols.get(target_sanitary)
            if not sanitary_col: sanitary_col = next((c for c in df_action.columns if 'sanitary' in c.lower() and 'url' in c.lower()), None)
            toilet_col = clean_cols.get(target_toilet)
            if not toilet_col: toilet_col = next((c for c in df_action.columns if 'toilet' in c.lower() and 'url' in c.lower()), None)
            
            if sanitary_col:
                column_config[sanitary_col] = st.column_config.LinkColumn("Sanitary Report", display_text="📥 Download Sanitary")
            if toilet_col:
                column_config[toilet_col] = st.column_config.LinkColumn("Toilet Report", display_text="📥 Download Toilet")

        system_cols = ['start', 'end', '_id', '_uuid', '_submission_time', '_validation_status', '_notes', '_status', '_submitted_by', '__version__', '_tags', '_index']
        display_cols = [c for c in df_action.columns if c not in system_cols]
        
        st.dataframe(df_action[display_cols], column_config=column_config, use_container_width=True, hide_index=True)
        st.stop()

    # --- ZONING MAP BUTTON ---
    if selected_key == 'peri': pdf_file_name = "zoning.pdf"
    elif selected_key == 'intra': pdf_file_name = "zoninginside.pdf"
    else: pdf_file_name = None 
    
    if pdf_file_name:
        col_map, _ = st.columns([1, 5])
        with col_map:
            pdf_bytes = get_pdf_bytes(pdf_file_name)
            if pdf_bytes:
                st.download_button(label="🗺️ View Zoning Map (PDF)", data=pdf_bytes, file_name=pdf_file_name, mime="application/pdf", key=f'download_pdf_{selected_key}', use_container_width=True)

    with st.spinner('Fetching Surveillance data...'):
        df = load_kobo_data(current_config['surv_url'])

    if df.empty:
        st.info("No data found or error loading Kobo data.")
        return
    
    # --- ROBUST CALCULATION (The Fix for Peri Map) ---
    col_pos_house_raw = find_column_by_keywords(df, ["how_many_wet_containers_were_found_positive", "among_the_wet", "positive_premises"])
    
    # FIX: Explicit check for the exact string user provided
    col_pos_cont_raw = find_column_by_keywords(df, ["Among_the_wet_containers_how_", "how_many_wet_containers_were_found_positive", "positive_containers"])
    
    col_wet_cont_raw = find_column_by_keywords(df, ["wet_containers", "wet container", "wet_containers_inspected"])
    col_dry_cont_raw = find_column_by_keywords(df, ["dry_container", "dry container"])

    if col_pos_house_raw:
        df['pos_house_calc'] = pd.to_numeric(df[col_pos_house_raw], errors='coerce').fillna(0)
    else:
        df['pos_house_calc'] = 0

    if col_pos_cont_raw:
        df['pos_cont_calc'] = pd.to_numeric(df[col_pos_cont_raw], errors='coerce').fillna(0)
    else:
        df['pos_cont_calc'] = 0

    if col_wet_cont_raw:
        df['wet_cont_calc'] = pd.to_numeric(df[col_wet_cont_raw], errors='coerce').fillna(0)
    else:
        df['wet_cont_calc'] = 0

    if col_dry_cont_raw:
        df['dry_cont_calc'] = pd.to_numeric(df[col_dry_cont_raw], errors='coerce').fillna(0)
    else:
        df['dry_cont_calc'] = 0
    
    # --- START FILTERING ---
    st.sidebar.markdown("### 🔍 Filters") 
    df_filtered = df.copy()

    col_map_lower = {c.lower(): c for c in df.columns}
    col_zone = col_map_lower.get('zone')
    col_subzone = col_map_lower.get('subzone')
    col_street = col_map_lower.get('streetname')
    col_username = col_map_lower.get('username')
    col_premises = "Premises" if "Premises" in df.columns else col_map_lower.get('premises')
    
    # Attempt to find GPS columns robustly
    col_lat = next((c for c in df.columns if '_location_latitude' in c.lower()), None)
    if not col_lat: col_lat = next((c for c in df.columns if 'latitude' in c.lower()), None)

    col_lon = next((c for c in df.columns if '_location_longitude' in c.lower()), None)
    if not col_lon: col_lon = next((c for c in df.columns if 'longitude' in c.lower()), None)
    
    date_col = "Date" if "Date" in df.columns else col_map_lower.get('date')
    if not date_col:
        for c in ['today', 'start', '_submission_time']:
             if c in col_map_lower: date_col = col_map_lower[c]; break

    if date_col:
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        min_date, max_date = df_filtered[date_col].min().date(), df_filtered[date_col].max().date()
        st.sidebar.markdown("#### Date Range")
        d1, d2 = st.sidebar.columns(2)
        start_date = d1.date_input("Start", min_date, key=f"start_date_{selected_key}")
        end_date = d2.date_input("End", max_date, key=f"end_date_{selected_key}")
        mask = (df_filtered[date_col].dt.date >= start_date) & (df_filtered[date_col].dt.date <= end_date)
        df_filtered = df_filtered.loc[mask]
        
    # --- DEBUGGING FOR PERI AIRPORT COLUMNS (Can Remove Later) ---
    if selected_key == 'peri':
        with st.expander("🛠️ Debug: View Peri Data Columns"):
            st.write(df.columns.tolist())
            
    # --- FLIGHTS SCREENING SUMMARY ---
    if selected_key == 'flights':
        clean_cols = {c.strip().lower(): c for c in df.columns}
        staff1_col = clean_cols.get("flight_duty_personnel") 
        staff2_col = clean_cols.get("deputy")
        if not staff1_col: staff1_col = next((c for c in df.columns if "duty" in c.lower() and "personnel" in c.lower()), None)
        if not staff2_col: staff2_col = next((c for c in df.columns if "deputy" in c.lower()), None)

        if staff1_col and staff2_col:
            all_staff = pd.concat([df_filtered[staff1_col].dropna(), df_filtered[staff2_col].dropna()]).astype(str).unique().tolist()
            st.sidebar.markdown("#### Staff Filter")
            selected_personnel = st.sidebar.multiselect("Select Duty Personnel", sorted(all_staff), key=f"personnel_filter_{selected_key}")
            if selected_personnel:
                mask = (df_filtered[staff1_col].astype(str).isin(selected_personnel)) | (df_filtered[staff2_col].astype(str).isin(selected_personnel))
                df_filtered = df_filtered[mask]
        
        st.markdown("#### ✈️ Data Summary")
        summary_data = []
        total_entries = len(df_filtered)
        summary_data.append(["Total International Flights Screened", total_entries])
        total_days = df_filtered[date_col].dt.date.nunique() if date_col else 'N/A'
        summary_data.append(["Total Days of Screening", total_days])
        
        numeric_df = df_filtered.select_dtypes(include=['number']).fillna(0)
        exclude_cols = ['_index', 'latitude', 'longitude', 'accuracy', '_id', 'instanceid', 'start', 'end'] 
        for col in numeric_df.columns:
            if not col.startswith('_') and col.lower() not in exclude_cols:
                col_sum = numeric_df[col].sum()
                summary_data.append([col, f"{col_sum:,.0f}"])
                
        summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
        st.table(summary_df)
        st.download_button("Download Raw Flights Data", to_excel(df_filtered), "Flights_Raw_Data.xlsx", key="flights_raw_download")
        st.stop()

    # --- STANDARD DASHBOARD FILTERS ---
    if col_zone and col_zone in df_filtered.columns:
        st.sidebar.markdown("#### Location")
        opts = sorted(df_filtered[col_zone].dropna().unique().astype(str))
        st.sidebar.multiselect(f"Zone", opts, key=f"zone_filter_{selected_key}")
        if st.session_state.get(f"zone_filter_{selected_key}"):
             df_filtered = df_filtered[df_filtered[col_zone].astype(str).isin(st.session_state[f"zone_filter_{selected_key}"])]
             
    if col_subzone and col_subzone in df_filtered.columns:
        opts = sorted(df_filtered[col_subzone].dropna().unique().astype(str))
        st.sidebar.multiselect(f"SubZone", opts, key=f"subzone_filter_{selected_key}")
        if st.session_state.get(f"subzone_filter_{selected_key}"):
             df_filtered = df_filtered[df_filtered[col_subzone].astype(str).isin(st.session_state[f"subzone_filter_{selected_key}"])]

    display_count, positive_count, hi_val, ci_val, bi_val = 0, 0, 0, 0, 0
    if selected_key == 'intra':
        if col_premises and date_col:
            df_filtered['unique_premise_id'] = df_filtered[date_col].dt.date.astype(str) + "_" + df_filtered[col_premises].apply(normalize_string)
            agg_dict = {'pos_house_calc': 'max', 'pos_cont_calc': 'sum', 'wet_cont_calc': 'sum', 'dry_cont_calc': 'sum'}
            if date_col: agg_dict[date_col] = 'first'
            for c in [col_zone, col_lat, col_lon, col_premises, col_username]:
                if c and c in df_filtered.columns: agg_dict[c] = 'first'
            df_grouped = df_filtered.groupby('unique_premise_id', as_index=False).agg(agg_dict)
            
            total_unique_premises = df_grouped['unique_premise_id'].nunique()
            positive_premises_count = (df_grouped['pos_house_calc'] > 0).sum()
            hi_val = (positive_premises_count / total_unique_premises * 100) if total_unique_premises > 0 else 0
            ci_val = (df_grouped['pos_cont_calc'].sum() / df_grouped['wet_cont_calc'].sum() * 100) if df_grouped['wet_cont_calc'].sum() > 0 else 0
            bi_val = (df_grouped['pos_cont_calc'].sum() / total_unique_premises * 100) if total_unique_premises > 0 else 0
            df_for_graphs = df_grouped.copy()
            df_for_graphs['is_positive_premise'] = (df_grouped['pos_cont_calc'] > 0).astype(int)
            display_count, positive_count = total_unique_premises, positive_premises_count
        else: df_for_graphs = df_filtered.copy()
    else:
        display_count = len(df_filtered)
        # Recalculate based on filtered data
        positive_count = (df_filtered['pos_house_calc'] > 0).sum()
        if display_count > 0:
            hi_val = (positive_count / display_count) * 100
            ci_val = (df_filtered['pos_cont_calc'].sum() / df_filtered['wet_cont_calc'].sum() * 100) if df_filtered['wet_cont_calc'].sum() > 0 else 0
            bi_val = (df_filtered['pos_cont_calc'].sum() / display_count * 100)
        df_for_graphs = df_filtered.copy()

    label_hi = "Premises Index (PI)" if selected_key == 'intra' else "House Index (HI)"
    label_entries = "Unique Premises" if selected_key == 'intra' else "Total Entries"
    total_pos_containers = int(df_filtered['pos_cont_calc'].sum())
    
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(label_entries, display_count)
    m2.metric("Positive Found", positive_count)
    m3.metric("Total Positive Containers", total_pos_containers)
    m4.metric(label_hi, f"{hi_val:.2f}")
    m5.metric("Container Index (CI)", f"{ci_val:.2f}")
    m6.metric("Breteau Index (BI)", f"{bi_val:.2f}")

    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

    with st.expander("📊 Graphical Analysis", expanded=True):
        active_tab_labels = ["📈 Trend Analysis", "🌍 Zone Stats"]
        if selected_key == 'peri':
            active_tab_labels.extend(["🏘️ Subzone Stats", "🛣️ Street Stats"])
        elif selected_key == 'intra':
            active_tab_labels.append("🏢 Premises Stats")
            
        graph_tabs = st.tabs(active_tab_labels)
        current_tab_map = {label: i for i, label in enumerate(active_tab_labels)}

        with graph_tabs[current_tab_map['📈 Trend Analysis']]:
            if date_col and col_zone in df_filtered.columns:
                df_trend = df_filtered.copy()
                df_trend['Month'] = df_trend[date_col].dt.to_period('M').astype(str)
                trend_data = df_trend.groupby(['Month', col_zone]).agg(
                    pos=('pos_house_calc', lambda x: (x>0).sum()),
                    total=('pos_house_calc', 'count')
                ).reset_index()
                trend_data['HI'] = (trend_data['pos'] / trend_data['total'] * 100).fillna(0)
                fig_trend = px.line(trend_data, x='Month', y='HI', color=col_zone, markers=True, title=f"Trend of {label_hi} by Zone")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Insufficient data for Trend Analysis.")

        def render_standard_charts(group_col, title_prefix, tab_label):
            if tab_label not in current_tab_map: return
            with graph_tabs[current_tab_map[tab_label]]:
                if group_col not in df_for_graphs.columns:
                    st.warning(f"Column for {title_prefix} not found.")
                    return
                aggs = {
                    'pos_cont_calc': 'sum', 
                    'wet_cont_calc': 'sum',
                    'pos_house_calc': lambda x: (x > 0).sum(),
                    'dry_cont_calc': 'count'
                }
                g = df_for_graphs.groupby(group_col).agg(aggs).rename(columns={'dry_cont_calc': 'Total Entries'})
                g['HI'] = (g['pos_house_calc'] / g['Total Entries'] * 100).fillna(0)
                g['CI'] = (g['pos_cont_calc'] / g['wet_cont_calc'].replace(0, 1) * 100).fillna(0)
                g['BI'] = (g['pos_cont_calc'] / g['Total Entries'] * 100).fillna(0)
                g = g.reset_index().sort_values('HI', ascending=False)
                if len(g) > 20: g = g.head(20)

                c1, c2 = st.columns(2)
                c1.plotly_chart(plot_metric_bar(g, group_col, 'HI', f"{label_hi} by {title_prefix}", 'HI', 20), use_container_width=True)
                c2.plotly_chart(plot_metric_bar(g, group_col, 'Total Entries', f"Total Houses/Premises Visited by {title_prefix}", 'Total Entries', None), use_container_width=True)
                c3, c4 = st.columns(2)
                c3.plotly_chart(plot_metric_bar(g, group_col, 'CI', f"Container Index (CI) by {title_prefix}", 'CI', 20), use_container_width=True)
                c4.plotly_chart(plot_metric_bar(g, group_col, 'BI', f"Breteau Index (BI) by {title_prefix}", 'BI', 20), use_container_width=True)

        render_standard_charts(col_zone, "Zone", "🌍 Zone Stats")
        render_standard_charts(col_subzone, "Subzone", "🏘️ Subzone Stats")
        render_standard_charts(col_street, "Street", "🛣️ Street Stats")
        
        if "🏢 Premises Stats" in current_tab_map:
            with graph_tabs[current_tab_map['🏢 Premises Stats']]:
                if col_premises in df_for_graphs.columns:
                    render_standard_charts(col_premises, "Premise", "🏢 Premises Stats")
                else:
                    st.warning("Premises data not available for graphing.")

    with st.expander("🌍 Geo-Spatial Map", expanded=True):
        if col_lat in df_for_graphs.columns and col_lon in df_for_graphs.columns:
            map_df = df_for_graphs.dropna(subset=[col_lat, col_lon]).copy()
            if not map_df.empty:
                # --- FIX: Dynamic column selection for tooltips ---
                # Peri wants columns 7,8,9,10,14 which usually are Zone, Street, House No, etc.
                # Since column indices shift, we find them by name mostly.
                
                # Try to find House Number
                col_house = next((c for c in df_filtered.columns if 'house' in c.lower() and 'number' in c.lower()), None)
                if not col_house: col_house = next((c for c in df_filtered.columns if 'door' in c.lower() and 'number' in c.lower()), None)
                
                # Try to find Street Name
                col_street_map = col_street if col_street else next((c for c in df_filtered.columns if 'street' in c.lower()), None)

                m = folium.Map(location=[map_df[col_lat].mean(), map_df[col_lon].mean()], zoom_start=14)
                for _, row in map_df.iterrows():
                    color = '#00ff00' if row['pos_house_calc'] == 0 else '#ff0000'
                    
                    if selected_key == 'intra':
                        # Intra: Premises only, removed tooltip
                        pass
                    else:
                        # Peri-specific robust tooltip - removed for stability on request
                        pass
                    
                    # Tooltips removed per latest request to keep map simple and functional
                    folium.CircleMarker(
                        [row[col_lat], row[col_lon]], radius=7, color=color, fill=True, fill_color=color
                    ).add_to(m)
                
                st_folium(m, height=700, use_container_width=True, key=f"main_map_{selected_key}")
            else:
                 st.info("No GPS data found to render map.")

    if current_config.get('id_url'):
        with st.expander("🔬 Larvae Identification Data", expanded=False):
            df_id = load_kobo_data(current_config['id_url'])
            if not df_id.empty:
                COL_GENUS = "Select the Genus:".strip()
                COL_SPECIES = "Select the Species:".strip()
                COL_CONTAINER_LABEL = "Type of container in which the sample was collected from".strip() 
                COL_SUBMITTED = "_submitted_by".strip()

                clean_to_orig_map = {col.strip(): col for col in df_id.columns}
                col_genus = clean_to_orig_map.get(COL_GENUS)
                col_species = clean_to_orig_map.get(COL_SPECIES)
                col_container = clean_to_orig_map.get(COL_CONTAINER_LABEL)
                col_submitted = clean_to_orig_map.get(COL_SUBMITTED)
                
                if not col_container:
                    FALLBACK_KEY = "Type of container the sample was collected from".strip() 
                    col_container = clean_to_orig_map.get(FALLBACK_KEY)

                col_map_id = {c.lower(): c for c in df_id.columns}
                date_col_id = next((c for c in df_id.columns if c in ['Date', 'today', 'date']), None)
                addr_cols = ['address', 'location', 'premise', 'premises', 'streetname']
                col_address_id = next((col_map_id.get(k) for k in addr_cols if col_map_id.get(k)), 'N/A')
                img_search = ["Attach the microscopic image of the larva _URL", "Attach the microscopic image of the larva_URL", "image_url", "url"]
                col_img = next((c for c in img_search if c in df_id.columns), None)
                
                if date_col_id: df_id[date_col_id] = pd.to_datetime(df_id[date_col_id])
                df_display = pd.DataFrame()
                df_display['Date'] = df_id[date_col_id].dt.date if date_col_id else 'N/A'
                df_display['Address'] = df_id[col_address_id] if col_address_id != 'N/A' else 'N/A'
                df_display['Genus'] = df_id[col_genus] if col_genus else 'N/A'
                df_display['Species'] = df_id[col_species] if col_species else 'N/A'
                
                if col_img:
                    df_display['Thumbnail'] = df_id[col_img].apply(get_thumbnail_url)
                    df_id['Original_Image_URL'] = df_id[col_img]
                else:
                    df_display['Thumbnail'] = None
                    df_id['Original_Image_URL'] = None

                df_display = df_display.reset_index(drop=True)
                df_display.index += 1
                df_display.index.name = "S.No"
                df_display = df_display.reset_index()
                df_id['Calculated_Address'] = df_display['Address']
                
                st.info("💡 Click on a row to view full details and image.")
                
                event = st.dataframe(
                    df_display,
                    column_order=["S.No", "Date", "Address", "Thumbnail", "Genus", "Species"],
                    column_config={"Thumbnail": st.column_config.ImageColumn("Microscopic Image", width="small")},
                    hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
                )

                if len(event.selection.rows) > 0:
                    selected_index = event.selection.rows[0]
                    original_row = df_id.iloc[selected_index]
                    show_image_popup(original_row)

                st.divider()
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if col_genus:
                        st.write("#### Genus")
                        genus_counts = df_id[col_genus].value_counts().reset_index()
                        genus_counts.columns = ['Genus', 'Count']
                        fig_g = px.pie(genus_counts, values='Count', names='Genus', hole=0.4)
                        st.plotly_chart(fig_g, use_container_width=True)
                    else: st.info("Genus data missing")

                with c2:
                    if col_container:
                        st.write("#### Container")
                        cont_data = df_id[col_container].dropna()
                        cont_counts = cont_data.value_counts().reset_index()
                        cont_counts.columns = ['Container', 'Count']
                        fig_c = px.pie(cont_counts, values='Count', names='Container', hole=0.4)
                        st.plotly_chart(fig_c, use_container_width=True)
                    else: st.warning(f"Container data missing.")

                with c3:
                    if col_submitted:
                        st.write("#### Submitted By")
                        user_counts = df_id[col_submitted].value_counts().reset_index()
                        user_counts.columns = ['User', 'Count']
                        fig_u = px.pie(user_counts, values='Count', names='User', hole=0.4)
                        st.plotly_chart(fig_u, use_container_width=True)
                    else: st.info("User data missing")

            else:
                st.info("No identification data available.")

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
            
            staff_perf = staff_perf.reset_index()
            staff_perf.index += 1
            staff_perf.index.name = 'S.No'
            staff_perf = staff_perf.reset_index()
            
            final_cols_staff = ['S.No', 'Name', 'Days Worked', 'Total Entries', 'Positive Found', 'Positive Containers', 'Container Index']
            
            staff_final = staff_perf[[c for c in final_cols_staff if c in staff_perf.columns]]
            st.dataframe(staff_final, use_container_width=True)
            st.download_button("Download Staff Excel", to_excel(staff_final), "Staff_Performance.xlsx", key=f"staff_excel_download_{selected_key}")
        else: st.warning("Username column not found.")

    c_month, c_fort = st.columns(2)
    with c_month:
        with st.expander("📅 Monthly Report", expanded=False):
            if date_col:
                df_rep_raw = df.copy()
                df_rep_raw[date_col] = pd.to_datetime(df_rep_raw[date_col])
                for col, raw_col in [('pos_house_calc', col_pos_house_raw), ('pos_cont_calc', col_pos_cont_raw), ('wet_cont_calc', col_wet_cont_raw)]:
                    df_rep_raw[col] = pd.to_numeric(df_rep_raw[raw_col], errors='coerce').fillna(0) if raw_col in df_rep_raw.columns else 0
                df_rep_raw['dry_cont_calc'] = pd.to_numeric(df_rep_raw[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_rep_raw.columns else 0
                df_rep_raw['Month_Year'] = df_rep_raw[date_col].dt.strftime('%Y-%m')
                sel_mon = st.selectbox("Select Month:", sorted(df_rep_raw['Month_Year'].unique(), reverse=True), key=f"monthly_select_{selected_key}")
                if sel_mon:
                    df_m = df_rep_raw[df_rep_raw['Month_Year'] == sel_mon].copy()
                    rep_df = generate_report_df(df_m, date_col, col_username, selected_key, col_premises, col_subzone, col_street, current_config)
                    st.dataframe(rep_df, hide_index=True)
                    st.download_button("Download Excel", to_excel(rep_df), "Monthly.xlsx", key=f"monthly_download_{selected_key}")

    with c_fort:
        with st.expander("📆 Fortnight Report", expanded=False):
            if date_col:
                df_ft = df.copy()
                df_ft[date_col] = pd.to_datetime(df_ft[date_col])
                for col, raw_col in [('pos_house_calc', col_pos_house_raw), ('pos_cont_calc', col_pos_cont_raw), ('wet_cont_calc', col_wet_cont_raw)]:
                    df_ft[col] = pd.to_numeric(df_ft[raw_col], errors='coerce').fillna(0) if raw_col in df_ft.columns else 0
                df_ft['dry_cont_calc'] = pd.to_numeric(df_ft[col_dry_cont_raw], errors='coerce').fillna(0) if col_dry_cont_raw in df_ft.columns else 0

                df_ft['Month_Str'] = df_ft[date_col].dt.strftime('%B %Y')
                df_ft['Label'] = df_ft.apply(lambda x: f"First Half {x['Month_Str']}" if x[date_col].day <= 15 else f"Second Half {x['Month_Str']}", axis=1)
                df_ft = df_ft.sort_values(by=date_col, ascending=False)
                sel_ft = st.selectbox("Select Fortnight:", df_ft['Label'].unique(), key=f"fortnight_select_{selected_key}")
                if sel_ft:
                    df_sft = df_ft[df_ft['Label'] == sel_ft].copy()
                    ft_rep = generate_report_df(df_sft, date_col, col_username, selected_key, col_premises, col_subzone, col_street, current_config)
                    st.dataframe(ft_rep, hide_index=True)
                    st.download_button("Download Excel", to_excel(ft_rep), "Fortnightly.xlsx", key=f"fortnight_download_{selected_key}")

    st.divider()
    summary_text = generate_narrative_summary(df_filtered, selected_key, date_col, col_street, col_subzone, col_premises)
    st.markdown(summary_text)

def render_home_page():
    inject_custom_css()
    
    if not check_password():
        return

    st.markdown(f"""
        <div class="main-header">
            <h1 style="margin:0; font-size: 2.2rem;">AIRPORT HEALTH ORGANISATION</h1>
            <h3 style="margin:0; font-weight:400; font-size: 1.2rem;">TIRUCHIRAPPALLI INTERNATIONAL AIRPORT</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('page') not in ['peri', 'intra', 'flights', 'anti_larval', 'sanitary']:
        st.header("Select Activity Section")
        
        # Grid Layout for Home Page
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🦟 Outside Field Activities (Peri)", use_container_width=True, type="primary"):
                st.session_state['page'] = 'peri'
                st.rerun()
            st.write("") # Spacer
            if st.button("✈️ Inside Field Activities (Intra)", use_container_width=True, type="primary"):
                st.session_state['page'] = 'intra'
                st.rerun()
            st.write("") # Spacer
            if st.button("🧹 Sanitary & Toilet Reports", use_container_width=True, type="primary"):
                st.session_state['page'] = 'sanitary'
                st.rerun()

        with col2:
            if st.button("✈️ International Flights Screening", use_container_width=True, type="primary"):
                st.session_state['page'] = 'flights'
                st.rerun()
            st.write("") # Spacer
            if st.button("🛡️ Anti-Larval Action Reports", use_container_width=True, type="primary"):
                st.session_state['page'] = 'anti_larval'
                st.rerun()
                
    else:
        if st.sidebar.button("🏠 Back to Home", key="back_to_home_button"):
            st.session_state['page'] = 'home'
            st.rerun()
        render_dashboard(st.session_state['page'])

if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

render_home_page() 
