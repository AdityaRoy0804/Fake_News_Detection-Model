import streamlit as st
import requests
import os

# The config object will hold all our environment variables
from app.config import Config
API_URL = Config.API_URL

# Initialize the result dictionary in Streamlit's session state
if 'result' not in st.session_state:
    st.session_state['result'] = {}


st.set_page_config(page_title="LLaMA News Verifier", page_icon="📰", layout="centered")
st.title("🛰️ LLaMA News Verifier")
st.write("Paste news text below, the model will classify it as REAL or FAKE and return sources if available.")


text = st.text_area("News text", height=200)
use_api = st.checkbox("Use remote API (otherwise run model locally)", value=True)


if st.button("Verify"):
    if not text.strip():
        st.warning("Please enter news text.")
    else:
        with st.spinner("Contacting verifier..."):
            if use_api:
                try:
                    res = requests.post(API_URL, json={"text": text})
                    res.raise_for_status()
                    data = res.json()
                    result = data["result"]
                    st.session_state['result'] = result
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.session_state['result'] = {"label": "ERROR", "confidence": 0.0, "explanation": str(e), "sources": []}
            else:
            # call local classifier import
                from app.classifier import get_verifier
                result = get_verifier().classify(text)
                st.session_state['result'] = result


# Display the result only if it exists in the session state
if st.session_state['result']:
    st.markdown("---")
    st.markdown(f"**Label:** {st.session_state['result'].get('label')}")
    st.markdown(f"**Confidence:** {st.session_state['result'].get('confidence')}")
    st.markdown(f"**Explanation:** {st.session_state['result'].get('explanation')}")
    sources = st.session_state['result'].get('sources') or []
    if sources:
        st.markdown("**Sources:**")
        for s in sources:
            st.markdown(f"- {s}")