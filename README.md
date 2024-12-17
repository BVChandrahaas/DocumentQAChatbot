# **Document QA System**

This project implements a **Document QA System** that:
1. **Processes PDF documents** by extracting content and splitting it into manageable chunks.
2. **Generates embeddings** for efficient search and retrieval using FAISS.
3. **Retrieves relevant contexts** locally using similarity search.
4. **Generates answers** to user queries using an AI language model.

The application uses **Streamlit** for the UI, making it interactive and user-friendly.

---

## **Table of Contents**
1. [Setup Instructions](#setup-instructions)
2. [API Documentation](#api-documentation)
3. [Architecture Overview](#architecture-overview)
4. [Design Decisions](#design-decisions)
5. [Performance Metrics](#performance-metrics)
6. [Known Limitations](#known-limitations)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)
9. [License](#license)

---

## **Setup Instructions**

Follow these steps to set up and run the project:

### **1. Clone the Repository**
```bash
git clone <BVChandrahaas/DocumentQAChatbot>
cd project_root
```

### **2. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Install Poppler**
Poppler is required for PDF-to-image conversion:
- **For Ubuntu**:
   ```bash
   sudo apt-get install poppler-utils
   ```
- **For macOS**:
   ```bash
   brew install poppler
   ```
- **For Windows**:
   - Download Poppler from [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/).
   - Add the installation path to the system environment variables.

### **4. Set Up Configuration**
Create a file `config.py` in the root directory with the following content to hide API keys:
```python
# config.py
OPENAI_API_KEY = "your_openai_api_key"
```

### **5. Run the Application**
Launch the Streamlit app:
```bash
streamlit run app.py
```

---

## **API Documentation**

### **1. Document Processing Pipeline**
- **File**: `document_pipeline.py`
- **Function**: `main_pipeline(file_path, query)`
   - **Description**: Processes a PDF file, retrieves relevant chunks, and generates an answer.
   - **Input**:
     - `file_path` *(str)*: Path to the uploaded PDF.
     - `query` *(str)*: User's input question.
   - **Output**:
     - Generated answer.
     - Retrieved document chunks.

---

## **Architecture Overview**

### **Files and Modules**
1. **`document_pipeline.py`**:
   - Loads and splits PDF documents using `PyPDFLoader`.
   - Generates embeddings using **SentenceTransformers**.
   - Stores embeddings locally in FAISS for fast retrieval.
   - Combines retrieved chunks and generates answers via GPT-4o.

2. **`app.py`**:
   - Provides the user interface for document upload and query input.
   - Displays generated answers and relevant contexts using Streamlit.

3. **`config.py`**:
   - Stores sensitive API keys securely.

4. **`requirements.txt`**:
   - Lists all required dependencies for the project.

### **Data Flow**
1. User uploads a PDF → Document is split into chunks.
2. Embeddings are generated and stored locally using FAISS.
3. User inputs a query → Relevant contexts are retrieved.
4. GPT-4o-mini generates an answer → Answer and context are displayed.

---

## **Design Decisions**

1. **Embedding Model**:
   - Chose **SentenceTransformers** (`all-MiniLM-L6-v2`) for efficient and lightweight embeddings.
2. **Chunking Strategy**:
   - Used `RecursiveCharacterTextSplitter` for flexible document splitting with overlap.
3. **Retrieval Method**:
   - FAISS was selected for local similarity search due to its performance and simplicity.
4. **Answer Generation**:
   - Integrated GPT-4o-mini via OpenAI API for accurate and context-aware answers.
5. **Streamlit UI**:
   - Streamlit provides an intuitive interface for document upload and querying.

---

## **Performance Metrics**

| Metric                     | Result                |
|----------------------------|-----------------------|
| Chunking Time              | ~1-2s for 100 pages   |
| Embedding Generation Time  | ~5s for 100 chunks    |
| Retrieval Time             | ~100ms for top-5      |
| GPT-4o Answer Generation   | ~2s per query         |

---

## **Known Limitations**
1. **OCR Dependency**: Scanned PDFs rely on OCR accuracy, which may introduce errors.
2. **Context Limitations**: Large documents may exceed input size limits for the language model.
3. **API Latency**: OpenAI API response time may vary depending on server load.

---

## **Future Enhancements**
1. Optimize embeddings using GPU acceleration (CUDA).
2. Add support for querying multiple documents simultaneously.
3. Integrate a lightweight local language model for faster response times.
4. Improve OCR for scanned PDF quality issues.

---

## **Contributing**

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes and push to GitHub.
4. Submit a pull request for review.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.
