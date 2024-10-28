import requests
import json

def run_model(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "nematron",
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()["response"]

def chain_of_thought_prompting(prompt):
    steps = prompt.split(".")
    results = []
    for step in steps:
        result = run_model(step)
        results.append(result)
    return results

def split_cot_between_agents(prompt):
    agents = prompt.split("&")
    results = []
    for agent in agents:
        result = run_model(agent)
        results.append(result)
    return results

def optimize_for_fast_completion(prompt):
    return run_model(prompt)

def output_to_text_file(prompt, output):
    with open("output.txt", "w") as f:
        f.write(output)

prompt = "Step 1. Step 2. Step 3."
# output = chain_of_thought_prompting(prompt)
# output = split_cot_between_agents(prompt)
# output = optimize_for_fast_completion(prompt)
# output_to_text_file(prompt, output)



# Define the model and system message
model = "nemotron-mini"
system_message = "You are a helpful assistant."

# Create a generator for the conversation
generator = ollama.Generator(model)

# Start the conversation with the system message
generator.system(system_message)

# Continuously get user input and generate responses
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = generator.generate(user_input)
    print("Assistant:", response)
````