import streamlit as st
import requests
import json
import os
from utils.env_helper import load_env_config # Reuse env loading

# --- Configuration ---
# Load .env file to get API port if defined
load_env_config()
API_HOST = os.environ.get("API_HOST", "localhost") # Allow overriding host
# Get port from config.py default or .env
try:
    from config import APP_PORT
    API_PORT = int(os.environ.get("APP_PORT", APP_PORT))
except (ImportError, ValueError):
    API_PORT = 8000 # Default if config/env loading fails

API_STREAM_URL = f"http://{API_HOST}:{API_PORT}/query/stream"
API_QUERY_URL = f"http://{API_HOST}:{API_PORT}/query" # For fetching sources later if needed

# --- Streamlit Page Setup ---
st.set_page_config(page_title="RAG Demo Chat", layout="wide")
st.title("💬 RAG Demo - Chat with Your Documents")
st.caption("Powered by FastAPI, Sentence Transformers, FAISS, and OpenAI")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Store chat history {role: "user/assistant", content: ""}

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Area ---
query = st.chat_input("Ask a question about your documents:")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Prepare request payload
    payload = {"query": query, "top_k": 3} # Keep top_k simple for now

    # Display thinking message and prepare placeholder for streamed response
    with st.chat_message("assistant"):
        status = st.status("Assistant is thinking...", expanded=False)
        answer_placeholder = st.empty()
        full_answer = ""
        error_occurred = False
        sources = [] # We'll try to get sources later if needed

        try:
            status.update(label="Connecting to RAG API...", state="running")
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            
            # Use requests to stream SSE
            with requests.post(API_STREAM_URL, json=payload, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                status.update(label="Receiving response...", state="running")
                
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("event: error"):
                        try:
                            error_data = json.loads(line.split("data: ", 1)[1])
                            st.error(f"An error occurred in the stream: {error_data.get('detail', 'Unknown stream error')}")
                        except:
                             st.error(f"An error occurred in the stream. Raw: {line}")
                        error_occurred = True
                        break # Stop processing on error event
                    elif line.startswith("event: end"):
                        status.update(label="Stream finished.", state="complete", expanded=False)
                        break # Stop processing on end event
                    elif line.startswith("data:"):
                        try:
                            chunk = json.loads(line.split("data: ", 1)[1])
                            if isinstance(chunk, str): # Ensure it's a string chunk
                                full_answer += chunk
                                answer_placeholder.markdown(full_answer + "▌") # Append chunk and show cursor
                            else:
                                # Handle potential non-string data if backend changes
                                pass 
                        except json.JSONDecodeError:
                            st.warning(f"Received non-JSON data: {line}")
                    # Ignore empty lines or other event types for now

            if not error_occurred:
                 answer_placeholder.markdown(full_answer) # Final answer without cursor
                 st.session_state.messages.append({"role": "assistant", "content": full_answer})

                 # Optional: Fetch sources after stream completes (using non-streaming endpoint)
                 # This avoids complicating the streaming logic
                 # try:
                 #     source_response = requests.post(API_QUERY_URL, json=payload, timeout=30)
                 #     source_response.raise_for_status()
                 #     source_data = source_response.json()
                 #     if source_data.get("success") and source_data.get("sources"):
                 #         sources = source_data["sources"]
                 #         with st.expander("Show Sources"):
                 #             st.json(sources) # Or format them nicely
                 # except Exception as source_e:
                 #     st.warning(f"Could not retrieve sources: {source_e}")

            else:
                 answer_placeholder.error("Failed to get full response due to stream error.")
                 # Add an error message to history if needed

        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: Could not connect to the RAG API at {API_STREAM_URL}. Details: {e}")
            answer_placeholder.empty() # Clear placeholder on connection error
            status.update(label="Connection failed.", state="error", expanded=True)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            answer_placeholder.empty()
            status.update(label="Processing error.", state="error", expanded=True)
            error_occurred = True # Mark as error to prevent saving partial response

        if error_occurred and not any(m["role"] == "assistant" and m["content"].startswith("Error") for m in st.session_state.messages[-2:]):
             # Avoid adding duplicate error messages if already shown
             pass 
             # Optionally add a generic error to history:
             # st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred."})


