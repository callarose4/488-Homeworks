import streamlit as st

hw1_page = st.Page("HW1.py", title="Homework 1")
hw2_page = st.Page("HW2.py", title="Homework 2")
hw3_page = st.Page("HW3.py", title="Homework 3")
hw4_page = st.Page("HW4.py", title="Homework 4")
hw5_page = st.Page("HW5.py", title="Homework 5")
hw7_page = st.Page("HW7.py", title="Homework 7", default = True)

pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page, hw5_page, hw7_page])
pg.run()
