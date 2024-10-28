# localO1

## Local O1 is the local implementation of a new type of AI called a "reasoning model"
 - unlike GPT-4, it spends between 10 and 60 seconds thinking before it answers
 - it uses advanced prompting techniques like chain-of-thought and self-reflection to improve its answers
 - o1 is a massive model, which requires tons of compute to run (expensive)
 - o1 has low message cap when you use it inside of ChatGPT
 - o1 is not private, meaning your data is exposed and in OpenAI's hands

## Here is the solution
 1. llama-3.1-nemotron-70b-instruct (or llama3.2)
 - this is a powerful, new AI model from Nvida that's possible to run locally
 - it's build on the Lamma 3.1 archetecture, meaning it's 100% open-sourced
 - it's just as capable as GPT-4o and Gemini 1.5 Pro on various benchmarks

 2. Prompt engineering (Chain of thought)
 - this is a prompting technique that helps AI model think more like humans
 - it involves breaking down a complex problem into smaller, more manageable steps, reducing the chance of errors

 3. AI agents
 - by building a team of agents, we can specialize each agent on a certain step of the CoT process
 - this simple trick allows us to massively boost performance while using the same LLM

 ## The To-Do List
 1. Download the ollama
 2. Install nemotron via ollama (or ollama3.2)
 3. Get an ollama chat completion in python (using ollama library)
 4. Implement chain of thought prompting
 5. Split the CoT steps between multiple agents
 6. Optimalize for fast completion
 7. Output the answer into a text file
 8. Implement self-reflection (optional)

## Task
1. create a python script that uses the ollama API to run the nemotron model locally
2. implement chain of thought prompting in the script
3. split the CoT steps between multiple agents in the script
4. optimize the script for fast completion
5. output the answer into a text file

