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
    n_threads=8  # Adjust based on CPU cores
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
def llm_wrapper(input_text):
    if not isinstance(input_text, str):
        input_text = str(input_text)

    # Truncate context to avoid exceeding LLM limit
    input_text = input_text[:800]  # Adjust based on the model's context limit

    response = llm(
        input_text,
        max_tokens=200,  # Limit response length
        temperature=0.5, # More consistent responses
        top_p=0.9
    )
    print(f"LLM RESPONSE: {response}")
    return response["choices"][0]["text"].strip()

def extract_relevant_logs(logs, query):
    """
    Extracts log entries relevant to the given query.
    """
    query_lower = query.lower()
    relevant_logs = [log for log in logs if query_lower in log.lower()]

    return relevant_logs

combine_documents_chain = create_stuff_documents_chain(llm_wrapper, prompt)
qa_chain = create_retrieval_chain(retriever, combine_documents_chain)

# Step 7: Ask a question
query = "What are the recent errors related to account 123?"
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_logs = [doc.page_content for doc in retrieved_docs]  # Extract raw log text

# Run LLM-based QA
filtered_logs = extract_relevant_logs(retrieved_logs, query)

formatted_logs = "\n".join(["- " + log.replace("\n", " ") for log in filtered_logs])
response = qa_chain.invoke({"input": query, "context": formatted_logs})

print(response)
# Step 10: Format the final output

formatted_response = {
    "query": query,
    "response": {
        "summary": response.get("answer", "No relevant data found."), 
        "relevant_logs": retrieved_logs  # Include raw logs related to the query
    }
}
# Print prettified JSON output
print(json.dumps(formatted_response, indent=4))