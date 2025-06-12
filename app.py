# app.py
import streamlit as st
import pandas as pd
import platform
import asyncio

from process import chunk_and_embed, answer_query_from_documents
from langchain.schema.retriever import BaseRetriever

# Fix event loop issue on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Streamlit Page Config
st.set_page_config(page_title='Document Chatbot', layout='wide')
st.title("📚 Document Chatbot with Citation & Theme Extraction")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "doc_results" not in st.session_state:
    st.session_state["doc_results"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "topic_dict" not in st.session_state:
    st.session_state["topic_dict"] = {}

# Sidebar: File Upload
st.sidebar.header('📤 Upload Your Documents')
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple files (PDF, TXT, Images)",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

with st.sidebar.expander("⚙️ Optional Filters"):
    filter_author = st.text_input('Author name')
    filter_doc_type = st.selectbox("Document Type", ["All", "Report", "Article", "Scanned", "Other"])
    filter_date = st.date_input("Published After")

# Embed files and store retriever + topic dict
if uploaded_files:
    if not isinstance(st.session_state["retriever"], BaseRetriever):
        try:
            retriever, topic_dict = chunk_and_embed(uploaded_files)
            st.session_state["retriever"] = retriever
            st.session_state["topic_dict"] = topic_dict
            st.success("Files embedded and retriever created ✅")
        except Exception as e:
            st.error(f"Failed to embed files: {str(e)}")

# Chat input
user_query = st.chat_input("Ask something about your documents...")

if user_query and isinstance(st.session_state["retriever"], BaseRetriever):
    try:
        result = answer_query_from_documents(user_query, st.session_state['retriever'])
        st.session_state["chat_history"].append(("user", user_query))
        st.session_state["chat_history"].append(("bot", result["answer"]))
        st.session_state["doc_results"] = result["sources"]
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

# Show Chat History
st.subheader("💬 Chat History")
for sender, message in st.session_state["chat_history"]:
    role = "user" if sender == "user" else "assistant"
    st.chat_message(role).markdown(message)

# Show Source Citations
if st.session_state["doc_results"]:
    st.markdown("**📄 Source Citations:**")
    for i, source in enumerate(st.session_state["doc_results"], 1):
        st.markdown(f"**{i}. Document:** `{source['document']}`")
        st.code(source["content"], language="markdown")

# Sidebar: Uploaded Files
st.sidebar.subheader("📄 Documents Loaded")
if uploaded_files:
    st.sidebar.write(f"{len(uploaded_files)} file(s) uploaded")
    for f in uploaded_files:
        st.sidebar.caption(f.name)
else:
    st.sidebar.write("No files uploaded yet")

# Document-wise Topic Summary
st.markdown("---")
st.subheader("📄 Document-wise Topics (Summarized)")

topic_dict = st.session_state.get("topic_dict", {})
if topic_dict:
    for filename, topics in topic_dict.items():
        with st.expander(f"📁 {filename}", expanded=False):
            df = pd.DataFrame([{
                "Page": "-",
                "Point": f"#{i+1}",
                "Answer": topic
            } for i, topic in enumerate(topics)])
            st.dataframe(df, use_container_width=True)
else:
    st.info("Upload files to view extracted topics.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
