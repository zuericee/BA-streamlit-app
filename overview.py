import streamlit as st
import pandas as pd
import openpyxl

st.set_page_config(page_title="Bachelor Thesis", layout="wide")

homepage = st.Page("pages/homepage.py", title="Fairness Judgments and the Roles of Opportunity, Risk-Taking, and Luck")

balancetable_page = st.Page("pages/balancetable.py", title="Balance Table")

powertest_page = st.Page("pages/powertest.py", title="Power Test")

user_pages = [homepage, balancetable_page, powertest_page]
