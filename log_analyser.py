import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

# Step 1: Load the logs
log_file = "bank_transactions.log"
loader = TextLoader(log_file)
documents = loader.load()

# Step 2: Split logs into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Step 3: Convert text chunks into embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

# Step 4: Set up a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve only top 3 relevant chunks

# Step 5: Load an LLM (Free Local Model)
llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=0,  # Force CPU execution
    n_threads=4  # Adjust based on CPU cores
)

# Step 6: Create a Retrieval-Augmented Generation (RAG) Chain
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
You are analyzing system logs to extract and summarize **relevant information** based on the user query.
The logs are of type ERROR, DEBUG, INFO and WARNING and based on the user query you have to extract the relevant logs only.

For example, if I need error logs for account 123, I should get logs like:
ERROR - Failed transaction for Account 123
ERROR - Invalid user tried to transfer money in account 123.

So based on this understanding, below are the logs and the query as well as some instructions. Please perform the necessary action

### **Logs Provided**:
{context}

### **User Query**:
{input}

### **Instructions**:
- Extract **only relevant messages** based on the query.
- Structure the response as follows:

### **Structured Response Format**:
1. **Summary of Findings:**  
   - Provide a **brief overview** of the key findings.
   - If no relevant logs are found, respond with: `"No relevant logs found for the given query."`

2. **List of Relevant Logs (if any):**  
   - **Timestamp:** [YYYY-MM-DD HH:MM:SS]  
   - **Message:** (Description of the relevant log entry)

3. **Patterns & Insights (if applicable):**  
   - Highlight **recurring issues** (e.g., repeated transaction failures, security breaches, etc.).
   - Identify **any abnormal patterns** in the occurrences.

Keep the response **concise, structured, and focused** on the user query.
"""
)
def chunk_text(text, chunk_size=1500, overlap=200):
    """Splits text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def llm_wrapper(input_text):
    if not isinstance(input_text, str):
        input_text = str(input_text)

    chunks = chunk_text(input_text, chunk_size=1500, overlap=200)  # Increase to 1500
    
    responses = []
    for chunk in chunks:
        trimmed_text = chunk[:400]  # Adjust to fit within 512 tokens (400 input + 100 output)
        response = llm(trimmed_text, max_tokens=100, temperature=0.2, top_p=0.9)
        responses.append(response["choices"][0]["text"].strip())

    return " ".join(responses)  

def extract_relevant_logs(logs, query):
    """
    Extracts log entries relevant to the given query.
    """
    query_lower = query.lower()
    relevant_logs = [log for log in logs if query_lower in log.lower()]

    return relevant_logs

def summarize_logs(logs, max_message_length=50):
    """Summarizes logs while preserving timestamp & log level."""
    summarized_logs = []
    
    for log in logs:
        parts = log.split(" - ", 2)  # Split into [timestamp, log level, message]
        
        if len(parts) == 3:
            timestamp, log_level, message = parts
            short_message = message[:max_message_length] + "..." if len(message) > max_message_length else message
            summarized_logs.append(f"{timestamp} - {log_level} - {short_message}")
        else:
            summarized_logs.append(log)  # In case of an unexpected format, keep original

    return "\n".join(summarized_logs)

combine_documents_chain = create_stuff_documents_chain(llm_wrapper, prompt)
qa_chain = create_retrieval_chain(retriever, combine_documents_chain)

# Continuous Query Mode
print("\n🔹 **Log Analysis Chatbot** 🔹")
print("Type your query below. Type 'exit' to quit.\n")

while True:
    query = input("🔎 Enter your query: ")
    if query.lower() == "exit":
        print("\n👋 Exiting Log Analysis Chatbot. Goodbye!")
        break
    
    # Retrieve relevant logs
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_logs = summarize_logs([doc.page_content for doc in retrieved_docs])

    # Filter logs based on query
    filtered_logs = extract_relevant_logs(retrieved_logs, query)
    formatted_logs = "\n".join(["- " + log.replace("\n", " ") for log in filtered_logs])

    # Run LLM-based QA
    response = qa_chain.invoke({"input": query, "context": formatted_logs})

    # Format and display output
    formatted_response = {
        "query": query,
        "response": {
            "summary": response.get("answer", "No relevant data found."),
            "relevant_logs": retrieved_logs  # Include raw logs related to the query
        }
    }
    
    print(json.dumps(formatted_response, indent=4))
    print("\n🔹 Ask another query or type 'exit' to quit.")