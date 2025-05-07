import streamlit as st
import keras

from src.analyzer import Analyzer

keras.config.disable_interactive_logging()

a = Analyzer()

if "done" not in st.session_state:
    st.session_state.done = False

if "results" not in st.session_state:
    st.session_state.results = None

def to_csv(df):
    return df.to_csv(index=False)

def progress(p):
    st.progress(p)

@st.cache_resource
def load_model():
    return 'Emotion'

def analyze():
    with st.spinner("Analyzing....(Analyzed video will be saved to cwd and shown here when I'm done)"):
        model = load_model()
        video_bytes= a.analyze(file=file, confidence=confidence)
        st.session_state.done = True
        st.session_state.video = video_bytes

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(st.session_state.video)
st.title("Audience emotion analyzer")

st.header("What do I do?")
st.write("Upload a video file for analysis. Change the confidence if you want to, and press the Analyze button.")

file = st.file_uploader('Select file to upload...')

col = st.columns([1])[0]
with col:
    confidence = st.number_input('Emotion analyzer confidence (Show predictions above this probability)', 0.0, 1.0, 0.9, step=0.01)
    

st.button("Analyze", on_click=analyze)