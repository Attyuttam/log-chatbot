from llama_cpp import Llama

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_gpu_layers=50)
response = llm("What is LangChain?")
print(response)
