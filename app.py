import streamlit as st
import pandas as pd
import requests
import io

# --- 1. SETUP ---
st.set_page_config(page_title="Dashboard Diagnostic Mode", layout="wide")

# --- 2. DATA LOADING ---
@st.cache_data(ttl=300)
def load_kobo_data(url):
    try:
        # Assuming secrets exist; if running locally without secrets, replace with your string
        if "KOBO_TOKEN" in st.secrets:
            token = st.secrets["KOBO_TOKEN"]
        else:
            st.error("Missing KOBO_TOKEN in secrets.")
            return pd.DataFrame()
            
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

# --- 3. NAVIGATION ---
SECTION_CONFIG = {
    'peri': {
        'title': 'Peri-Airport Larvae Surveillance',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aXM5aSjVEJTgt6z5qMvNFe/export-settings/es9zUAYU5f8PqCokaZSuPmg/data.csv'
    },
    'intra': {
        'title': 'Intra-Airport Larvae Surveillance',
        'surv_url': 'https://kf.kobotoolbox.org/api/v2/assets/aEdcSxvmrBuXBmzXNECtjr/export-settings/esgYdEaEk79Y69k56abNGdW/data.csv'
    }
}

st.sidebar.header("Navigation")
selected_key = st.sidebar.radio("Select Report:", list(SECTION_CONFIG.keys()), format_func=lambda x: SECTION_CONFIG[x]['title'])
current_config = SECTION_CONFIG[selected_key]
st.title(f"🚨 DIAGNOSTIC MODE: {current_config['title']}")

# --- 4. DIAGNOSTIC LOGIC ---
st.warning("This mode is to find the correct column names to fix the KeyError.")

with st.spinner('Fetching data...'):
    df = load_kobo_data(current_config['surv_url'])

if not df.empty:
    st.success("Data Loaded Successfully!")
    
    # --- CHECK 1: RAW COLUMN LIST ---
    st.subheader("1. List of ALL Columns Found")
    st.write("Copy these exact names to fix the code:")
    
    # Get all columns and sort them
    all_cols = sorted(list(df.columns))
    st.code(all_cols)
    
    # --- CHECK 2: SEARCH FOR KEY COLUMNS ---
    st.subheader("2. Searching for Critical Columns")
    
    search_terms = {
        "Positive Container Count": ["among", "wet", "container", "how", "positive"],
        "Wet Container Count": ["number", "wet", "container", "found"],
        "Dry Container Count": ["dry", "hold", "water"],
        "Date": ["date", "today", "submission"],
        "Premises/Address": ["premise", "address", "location", "zone"]
    }
    
    for category, keywords in search_terms.items():
        st.write(f"**Potential matches for '{category}':**")
        found = []
        for col in df.columns:
            # Check if ANY keyword matches
            if any(k in col.lower() for k in keywords):
                found.append(col)
        
        if found:
            for f in found:
                st.code(f)
        else:
            st.error("No likely match found.")

    # --- CHECK 3: PREVIEW DATA ---
    with st.expander("View Raw Data Preview (First 5 rows)"):
        st.dataframe(df.head())

else:
    st.error("Could not load data. Check URL or Token.")
