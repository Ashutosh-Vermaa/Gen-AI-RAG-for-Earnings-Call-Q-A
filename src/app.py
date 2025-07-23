import logging
import streamlit as st
import os
import re
import tempfile
import hashlib
import traceback

from ingestion import load_and_clean_pdf, split_docs
from Indexing import create_index, create_retriever, contextual_compression_retriever
from retriever import get_answer

# Custom styling for UI
st.markdown("""
    <style>
        .main > div:first-child { padding-top: 1rem; }
        .title h1 { font-size: 1.8rem !important; margin-bottom: 0.5rem; }
        .element-container:has(.stFileUploader) {
            max-width: 400px; margin-bottom: 0.5rem;
        }
        .stFileUploader { padding: 0.25rem 0.5rem !important; font-size: 0.85rem !important; }
        .stTextInput { max-width: 600px; }
        .stTextInput input { font-size: 1rem; padding: 0.5rem; }
        .answer-block {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            max-width: 700px;
        }
        .error-details {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin-top: 1rem;
            font-family: monospace;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# Setup logging
log_path = os.path.join("logs", "debug.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="AGM Transcript QA")
st.markdown("<div class='title'><h1>📄  Q&A on Earnings Call</h1></div>", unsafe_allow_html=True)

# Function to hash uploaded file contents
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.chunks = None
    st.session_state.last_file_hash = None

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_hash = get_file_hash(file_bytes)
    
    # Debug info
    st.write(f"**Debug Info:**")
    st.write(f"- File size: {len(file_bytes)} bytes")
    st.write(f"- File hash: {file_hash}")
    st.write(f"- Last processed hash: {st.session_state.last_file_hash}")

    # Index only if this is a new file
    if st.session_state.last_file_hash != file_hash:
        with st.spinner("🔄 Indexing uploaded PDF..."):
            try:
                logger.info(f"Starting PDF processing. File size: {len(file_bytes)} bytes")
                
                # Load and clean PDF using file_bytes
                st.write("Step 1: Loading and cleaning PDF...")
                complete_pdf = load_and_clean_pdf(file_bytes)
                logger.info("PDF loaded and cleaned successfully")
                st.write(f"✅ PDF loaded. Content length: {len(complete_pdf.page_content)} characters")

                # Split into chunks (using your custom speaker-based splitting)
                st.write("Step 2: Splitting into chunks by speaker...")
                chunks = split_docs(complete_pdf)
                logger.info(f"Document split into {len(chunks)} chunks")
                st.write(f"✅ Document split into {len(chunks)} chunks by speaker")

                # Create index
                st.write("Step 3: Creating index...")
                index_name = f"index-{file_hash}"[:45]
                logger.info(f"Creating index: {index_name}")
                index = create_index(chunks, index_name=index_name)
                logger.info("Index created successfully")
                st.write("✅ Index created")

                # Create retrievers
                st.write("Step 4: Creating retrievers...")
                retriever = create_retriever(index)
                contextual_retriever = contextual_compression_retriever(index, retriever)
                logger.info("Retrievers created successfully")
                st.write("✅ Retrievers created")

                # Save to session
                st.session_state.retriever = contextual_retriever
                st.session_state.chunks = chunks
                st.session_state.last_file_hash = file_hash

                logger.info("Indexing complete.")
                st.success("🎉 PDF processing complete!")

            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                
                logger.error(f"Error during processing: {error_msg}", exc_info=True)
                
                # Display detailed error information
                st.error(f"❌ Failed to process the PDF")
                
                # Show error details in an expandable section
                with st.expander("🔍 Error Details"):
                    st.markdown(f"""
                    <div class="error-details">
                    <strong>Error Message:</strong><br>
                    {error_msg if error_msg else "No error message available"}
                    <br><br>
                    <strong>Full Traceback:</strong><br>
                    {error_trace}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Also show in main interface if error message is empty
                if not error_msg:
                    st.warning("⚠️ Empty error message - check the error details above")
                
                st.stop()

    # Question Input
    question = st.text_input("💬 Ask a question about the document:")
    if question and st.session_state.retriever:
        try:
            logger.info(f"Answering question: {question}")
            response = get_answer(question, st.session_state.retriever, st.session_state.chunks)
            st.markdown("### ✅ Answer")
            st.write(response)
        except Exception as e:
            logger.error(f"Error during QA: {e}", exc_info=True)
            st.error("❌ Failed to answer the question. Check logs or try again.")
            with st.expander("🔍 QA Error Details"):
                st.text(str(e))
                st.text(traceback.format_exc())
else:
    st.info("👆 Please upload a PDF to begin.")