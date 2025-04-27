import streamlit as st
import requests
from datetime import datetime

API_URL = "http://localhost:8000"


st.set_page_config(page_title="PDF Chat Assistant", layout="centered")
st.title("Chat with Your PDF")

# Initialize session state at the top
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

# Sidebar: Toggle and task history
with st.sidebar:
    # Toggle button
    if st.button("Close Sidebar" if st.session_state.sidebar_visible else "Open Sidebar"):
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible
        st.rerun()  # Force re-render to update sidebar visibility

    # Show task history only if sidebar is visible
    if st.session_state.sidebar_visible:
        st.markdown("### Task History")
        selected_pdf = st.session_state.selected_pdf
        if selected_pdf:
            # Combine questions and summaries
            tasks = []
            # Add questions from chat_history
            if selected_pdf in st.session_state.chat_history:
                tasks.extend([(q, "Question", timestamp) for q, _, timestamp in st.session_state.chat_history[selected_pdf]])
            # Add summaries
            if selected_pdf in st.session_state.summaries:
                tasks.extend([("Summary Generated", "Summary", timestamp) for _, timestamp in st.session_state.summaries.get(selected_pdf, [])])
            
            # Sort by timestamp (newest first)
            tasks.sort(key=lambda x: x[2], reverse=True)
            
            if tasks:
                for task_text, task_type, timestamp in tasks:
                    display_text = task_text[:50] + "..." if len(task_text) > 50 else task_text
                    st.markdown(f"**{task_type}:** {display_text} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                st.write("No tasks yet.")
        else:
            st.write("Select a PDF to view task history.")

# Upload section
st.subheader("Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    with st.spinner("Uploading and indexing PDFs..."):
        for uploaded_file in uploaded_files:
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(f"{API_URL}/upload_pdf", files=files)
            if response.status_code == 200:
                st.success(f"{uploaded_file.name} uploaded and indexed successfully.")
            else:
                st.error(f"Upload failed for {uploaded_file.name}: {response.text}")

# Fetch list of PDFs
st.subheader("Select a PDF to interact with")
try:
    response = requests.get(f"{API_URL}/list_pdfs")
    response.raise_for_status()
    pdf_list = response.json().get("pdfs", [])
except Exception as e:
    st.error(f"Failed to load PDFs: {e}")
    pdf_list = []

selected_pdf = st.selectbox("Available PDFs:", pdf_list, key="pdf_selector")
if selected_pdf:
    if selected_pdf not in st.session_state.chat_history:
        st.session_state.chat_history[selected_pdf] = []
    if selected_pdf not in st.session_state.summaries:
        st.session_state.summaries[selected_pdf] = []
st.session_state.selected_pdf = selected_pdf

# Summary button
if selected_pdf:
    if st.button("Summarize this PDF"):
        with st.spinner("Generating summary..."):
            res = requests.post(f"{API_URL}/summarize", json={"pdf_name": selected_pdf, "question": ""})
            if res.status_code == 200:
                summary = res.json().get("summary", "No summary returned.")
                timestamp = datetime.now()
                st.session_state.summaries[selected_pdf].append((summary, timestamp))
                st.success("Summary generated.")
                st.markdown("### PDF Summary")
                st.markdown(summary)
                st.rerun()  # Force re-render to update sidebar
            else:
                st.error(f"Error during summarization: {res.text}")

# Mode selection
if selected_pdf:
    mode = st.radio("Answer mode", ["detailed", "short"], horizontal=True)

# Ask a question
if selected_pdf:
    user_query = st.text_input("Ask a question:")
    if st.button("Ask"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                res = requests.post(f"{API_URL}/ask", json={
                    "pdf_name": selected_pdf,
                    "question": user_query,
                    "mode": mode
                })
                if res.status_code == 200:
                    answer = res.json().get("answer", "")
                    timestamp = datetime.now()
                    st.session_state.chat_history[selected_pdf].append((user_query, answer, timestamp))
                    st.markdown("### Answer")
                    st.write(answer)
                    # Show sources
                    sources = res.json().get("sources", [])
                    if sources:
                        with st.expander("Sources"):
                            for i, src in enumerate(sources):
                                st.markdown(f"**Source {i+1}:** {src}...")
                    st.rerun()  # Force re-render to update sidebar
                else:
                    st.error(f"Error during question answering: {res.text}")

    # Chat history
    if st.session_state.chat_history[selected_pdf]:
        st.markdown("### Chat History")
        for q, a, _ in reversed(st.session_state.chat_history[selected_pdf]):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")

    # Summaries
    if st.session_state.summaries.get(selected_pdf):
        st.markdown("### Summaries")
        for summary, timestamp in reversed(st.session_state.summaries[selected_pdf]):
            st.markdown(f"**Summary ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}):**")
            st.markdown(summary)

# Delete section
if selected_pdf:
    st.markdown("---")
    if st.button("Delete this PDF"):
        res = requests.delete(f"{API_URL}/delete_pdf", data={"pdf_name": selected_pdf})
        if res.status_code == 200:
            st.success("PDF and its index deleted.")
            st.session_state.chat_history.pop(selected_pdf, None)
            st.session_state.summaries.pop(selected_pdf, None)
            st.session_state.selected_pdf = None
            st.rerun()  # Force re-render to update sidebar
        else:
            st.error(f"Failed to delete PDF: {res.text}")
