import streamlit as st

# Import your two project scripts
import project1
import project2

# Sidebar for navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Project 1", "Project 2"])

# Display the selected project
if choice == "Project 1":
    project1.main()
elif choice == "Project 2":
    project2.main()
