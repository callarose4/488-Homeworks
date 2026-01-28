import streamlit as st

hw1_page = st.Page("HW1.py", title="Homework 1")
hw2_page = st.Page("HW2.py", title="Homework 2", default = True) #default to Lab 2

pg = st.navigation([hw1_page, hw2_page])
pg.run()
