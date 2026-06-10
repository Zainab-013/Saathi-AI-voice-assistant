import streamlit.components.v1 as components
import os

# Declare custom component pointing to this folder (where index.html sits)
_component_func = components.declare_component(
    "audio_recorder",
    path=os.path.dirname(os.path.abspath(__file__))
)

def audio_recorder(key=None):
    """
    Renders a browser-side microphone recorder.
    Returns the base64-encoded WAV data of the recorded audio when stopped.
    """
    return _component_func(key=key)
