# multi-agent system to build our own o1 assistant

import ollama

def CEO_agent(idea):
    # chat style completion
    response = ollama.chat(
        model="nemotron-mini",
        messages=[
            {"role": "system", "content": "You are o1, an intelligent AI assistant focused on clear reasoning and step-by-step analysis. You break down complex problems methodically and explain your through process transparently. You aim to be precise, logical, and brief in your responses. Limit the response in 4 steps. Only responds the 4 steps in point form. No other explaination required for output."},
                {"role": "user", "content": f"Help me to design an actionable roadmap to build my AI startup in the next 30 days. I already have an idea is: {idea}"}
        ]
    )

    # Extract the steps from the response
    steps = response['message']['content'].split("\n")

    # Call the respective agent functions for each step and get the response
    response_01 = agent_01(steps[1].strip())
    response_02 = agent_02(steps[2].strip())
    response_03 = agent_03(steps[3].strip())
    response_04 = agent_04(steps[4].strip())

    # Combine the responses from all agents
    final_response = "response_01: \n" + response_01 + "\n\n" + "response_02: \n" + response_02 + "\n\n" + "response_03: \n" + response_03 + "\n\n" + "response_04: \n" + response_04

    # print the final response
    print(final_response)

    # Call the new agent_secretary function with the initial idea and final result
    summary_response = agent_secretary(idea, final_response)
    return summary_response

def agent_01(step):
    # Generate a response based on the extracted step
    response = ollama.chat(
        model="nemotron-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant focused on the first step of the process."},
            {"role": "user", "content": f"Please provide a detailed implementation of the following step: {step}"}
        ]
    )

    return response['message']['content']

def agent_02(step):
    # Generate a response based on the extracted step
    response = ollama.chat(
        model="nemotron-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant focused on the second step of the process."},
            {"role": "user", "content": f"Please provide a detailed implementation of the following step: {step}"}
        ]
    )

    return response['message']['content']

def agent_03(step):
    # Generate a response based on the extracted step
    response = ollama.chat(
        model="nemotron-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant focused on the third step of the process."},
            {"role": "user", "content": f"Please provide a detailed implementation of the following step: {step}"}
        ]
    )

    return response['message']['content']

def agent_04(step):
    # Generate a response based on the extracted step
    response = ollama.chat(
        model="nemotron-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant focused on the final step of the process."},
            {"role": "user", "content": f"Please provide a detailed implementation of the following step: {step}"}
        ]
    )

    return response['message']['content']

def agent_secretary(initial_idea, final_result):
    # Generate a summary of the idea and the final result
    summary = f"Initial idea: {initial_idea}\nFinal result: {final_result}"

    # Generate a response based on the summary
    response = ollama.chat(
        model="nemotron-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant focused on summarizing ideas and results."},
            {"role": "user", "content": f"Please provide a summary of the following: {summary}"}
        ]
    )

    return response['message']['content']

# example usage of the main function
if __name__ == "__main__":
    idea = "A clone of the Google meet with a virtual meeting platform that integrates AI note-taking and question-answering capabilities, enhancing productivity and collaboration."
    result = CEO_agent(idea)
    print(result)
