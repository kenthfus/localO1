
import ollama

# generate a response
response = ollama.generate("nemotron-mini", "What is the meaning of life?")
print(response["response"])


