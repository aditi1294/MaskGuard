import streamlit as st
from auth_utils import login_user, signup_user

st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "email" not in st.session_state:
    st.session_state.email = ""

st.title("😷 Face Mask Detection Login")
auth_mode = st.radio("Choose Action", ["Login", "Signup"])

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if auth_mode == "Login":
    if st.button("Login"):
        user = login_user(email, password)
        if user:
            st.success("Login successful!")
            st.session_state.authenticated = True
            st.session_state.email = email
            st.switch_page("main.py")
        else:
            st.error("Invalid login credentials.")
else:
    if st.button("Create Account"):
        user = signup_user(email, password)
        if user:
            st.success("Account created. Please login.")
        else:
            st.error("Signup failed. Try a different email.")
