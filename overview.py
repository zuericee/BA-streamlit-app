import streamlit as st

st.set_page_config(page_title="Bachelor Thesis", layout="wide")

# Define pages
balancetable_page = st.Page("umfrage/pages/balancetable.py", title="Balance Table")
powertest_page = st.Page("umfrage/pages/powertest.py", title="Power Test")

# Create navigation
pg = st.navigation([balancetable_page, powertest_page])

# Run selected page
pg.run()