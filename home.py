import streamlit as st
from pages import home, page1, page2

# Dictionary of pages
PAGES = {
    "Home": home,
    "Page 1": page1,
    "Page 2": page2
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.write()

if __name__ == "__main__":
    main()
