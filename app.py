import streamlit as st
from document_pipeline import main_pipeline

# Title and description
st.title("📄 Document Question Answering System")
st.markdown(
    """
    Upload a PDF document and ask questions about its content. The system will retrieve relevant sections of the document and generate an accurate answer using an AI model.
    """
)

# File uploader for the PDF document
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], help="Upload a PDF file for analysis")

# Text input for the query
query = st.text_input("Enter your question:", help="Type your question here")

# Display a button to process the input
if uploaded_file and query:
    with st.spinner("Processing your document and generating the answer..."):
        try:
            # Save uploaded file temporarily
            temp_file = "uploaded.pdf"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Call the main pipeline
            answer, retrieved_docs = main_pipeline(temp_file, query)
            
            # Display the answer
            st.success("Answer Generated!")
            st.markdown(f"### **Answer:**")
            st.write(answer)
            
            # Display the retrieved context
            st.markdown("### **Context Retrieved:**")
            for idx, doc in enumerate(retrieved_docs, start=1):
                st.markdown(f"**Chunk {idx}:**")
                st.write(f"{doc['text'][:200]}...")  # Display the first 200 characters
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF and enter a question to begin.")

# Footer for instructions or additional notes
st.markdown(
    """
    ---
    **Instructions:**
    - Upload PDF documents up to 100 pages.
    - Ask precise questions for better answers.
    - Ensure the document text is machine-readable (not an image-based PDF).
    """
)
