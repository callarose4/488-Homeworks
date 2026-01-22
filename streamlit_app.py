import streamlit as st

hw1_page = st.Page("HW1.py", title="Homework 1", default=True)

pg = st.navigation([hw1_page])
pg.run()