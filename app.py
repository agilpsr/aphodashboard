import streamlit as st
import pandas as pd
import requests
import io

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="Kobo Dashboard", layout="wide")

# --- 2. PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # We look for the password in the "Secrets" (explained in Phase 4)
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        # Show input box for password
        password_input = st.text_input("Enter Password to Login:", type="password")
        if st.button("Login"):
            # Check against the secret password
            if password_input == st.secrets["APP_PASSWORD"]:
                st.session_state.password_correct = True
                st.rerun() # Reload the app to show the dashboard
            else:
                st.error("❌ Incorrect Password")
        return False
    else:
        return True

# If password is NOT correct, stop here.
if not check_password():
    st.stop()

# --- 3. DATA LOADING FUNCTION (UPDATED) ---
@st.cache_data(ttl=300) 
def load_kobo_data(url):
    try:
        token = st.secrets["KOBO_TOKEN"]
        headers = {"Authorization": f"Token {token}"}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        # FIX: Use 'sep=None' and 'engine=python' to auto-detect ; or ,
        # 'on_bad_lines' will skip that one broken line (565) so the app doesn't crash
        df = pd.read_csv(
            io.StringIO(response.text), 
            sep=None, 
            engine='python',
            on_bad_lines='skip' 
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Debugging: Print the first 200 characters to see if it's actually an error message
        if 'response' in locals():
            st.text("Raw data preview (first 200 chars):")
            st.code(response.text[:200])
        return pd.DataFrame()

# --- 4. YOUR DATA SOURCES ---
# Dictionary mapping the user-friendly name to your specific URL
DATA_SOURCES = {
    'Outside Premises': 'https://kf.kobotoolbox.org/api/v2/assets/aXM5aSjVEJTgt6z5qMvNFe/export-settings/es9zUAYU5f8PqCokaZSuPmg/data.csv',
    'Inside Premises': 'https://kf.kobotoolbox.org/api/v2/assets/aEdcSxvmrBuXBmzXNECtjr/export-settings/esgYdEaEk79Y69k56abNGdW/data.csv',
    'Out ID': 'https://kf.kobotoolbox.org/api/v2/assets/afU6pGvUzT8Ao4pAeX54QY/export-settings/esinGxnSujLzanzmAv6Mdb4/data.csv',
    'In ID': 'https://kf.kobotoolbox.org/api/v2/assets/anN9HTYvmLRTorb7ojXs5A/export-settings/esDnrutbSWfn8AbieZSqzdV/data.csv'
}

# --- 5. THE DASHBOARD LAYOUT ---
st.title("🏥 Medical Operations Dashboard")

# A. Sidebar Switch
st.sidebar.header("Navigation")
selected_section = st.sidebar.radio("Select Data Section:", list(DATA_SOURCES.keys()))

# Load the data for the selected section
current_url = DATA_SOURCES[selected_section]
st.write(f"### Viewing: {selected_section}")

with st.spinner('Fetching latest data from Kobo...'):
    df = load_kobo_data(current_url)

if not df.empty:
    # B. Date Filter Logic
    # We try to automatically find the 'submission_time' or 'start' column.
    # Kobo usually has '_submission_time' or 'start'.
    
    date_col = None
    possible_date_cols = ['_submission_time', 'start', 'end', 'today', 'Date']
    
    # Check which column exists in your data
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
            
    # If we found a date column, let's filter!
    if date_col:
        # Convert text to datetime objects
        df[date_col] = pd.to_datetime(df[date_col])
        
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)
            
        # Filter the data
        mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
        df_filtered = df.loc[mask]
    else:
        st.warning("⚠️ Could not automatically find a Date column. Showing all data.")
        df_filtered = df

    # C. Key Metrics
    total_entries = len(df_filtered)
    st.markdown(f"## Total Entries: **{total_entries}**")
    
    # D. Show Data Table
    with st.expander("View Raw Data Table"):
        st.dataframe(df_filtered)

    # E. Simple Analysis (Example)
    # If there is a column named 'deviceid' or similar, we can count it
    st.divider()
    st.write("#### Quick Summary")
    # This shows the column names so you know what to ask me to analyze next
    st.write("Columns available in this file:", list(df_filtered.columns))

else:
    st.info("No data found or unable to load.")
