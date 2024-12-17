from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from config import OPENAI_API_KEY


def load_and_chunk_document(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Load and split the document into overlapping chunks.
    """
    print("Loading and chunking document...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")
    return chunks


def generate_embeddings_and_store(chunks):
    """
    Generate embeddings for chunks and store in FAISS.
    """
    print("Generating embeddings and storing in FAISS...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    metadata = [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    print("Embeddings generated and stored successfully.")
    return index, metadata, model


def retrieve_relevant_chunks(index, metadata, query, model, k=5):
    """
    Retrieve the most relevant chunks using FAISS and query embeddings.
    """
    print("Retrieving relevant chunks...")
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=k)

    # Fetch metadata for the top results
    retrieved_docs = [metadata[idx] for idx in indices[0]]

    # Debug: Print retrieved chunks
    for idx, doc in enumerate(retrieved_docs):
        print(f"DEBUG - Chunk {idx + 1}: {doc['text'][:200]}...")
    return retrieved_docs



def combine_chunks(retrieved_docs):
    """
    Combine chunks into a single string with proper structure.
    """
    combined_context = ""
    for idx, doc in enumerate(retrieved_docs):
        text = doc.get("text", "").strip()
        if text:
            combined_context += f"Chunk {idx + 1}:\n{text}\n\n"
    return combined_context



def generate_answer(context, query):
    """
    Generate an answer using OpenAI GPT-4 chat model.
    """
    # Define the chat-based prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a document assistant. Use only the information in the provided context to answer the question. 
        If the context is relevant but you don't know the answer, reply:
        "I'm sorry, I do not know the answer based on the provided context."
        
        If the question is unrelated to the context of the document, reply:
        "I can only answer questions that are related to the document's domain."

        Context: {context}

        Question: {question}

        Answer:
        """
    )

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=OPENAI_API_KEY)

    # Format the prompt
    formatted_prompt = prompt.format(context=context, question=query)
    print("DEBUG - Formatted Prompt:", formatted_prompt)

    # Generate response
    response = llm.invoke(formatted_prompt)
    print("DEBUG - LLM Response:", response.content)

    return response.content



def main_pipeline(file_path, query):
    """
    End-to-end pipeline for document-based question answering.
    """
    try:
        # Step 1: Load and Chunk Document
        chunks = load_and_chunk_document(file_path)

        # Step 2: Generate Embeddings and Store in FAISS
        index, metadata, model = generate_embeddings_and_store(chunks)

        # Step 3: Retrieve Relevant Chunks
        retrieved_docs = retrieve_relevant_chunks(index, metadata, query, model, k=5)

        # Step 4: Combine Retrieved Chunks into Context
        context = combine_chunks(retrieved_docs)
        print("DEBUG - Combined Context:", context)  # Ensure context is not empty

        # Step 5: Generate Answer Using LLM
        answer = generate_answer(context, query)

        return answer, retrieved_docs

    except Exception as e:
        print(f"Error in pipeline: {e}")
        return "An error occurred during processing.", []

