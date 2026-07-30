from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

if load_dotenv():
    print("Successfully loaded api key")

# --- The Chat Completions API ---

# API Q1:
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("\nAPI Q1:")
print("\nAnswer:", response.choices[0].message.content)
print("\nModel:", response.model)
print("\nTotal tokens:", response.usage.total_tokens)

# API Q2:
temperatures = [0, 0.7, 1.5]

print("\nAPI Q2:")
for temperature in temperatures:
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Suggest a creative name for a data engineering consultancy.",
        }],
        temperature=temperature
    )
    print(f"\nResponse for temp={temperature}:", response2.choices[0].message.content)
    
# Comment:
# Temperature 0 produced the most consistent and predictable output. (Deterministic)
# Temperature 0.7 is a standard default; it is more creative but still makes sense.
# While temperature 1.5 generated the most varied and unpredictable responses. (Creative)
# If I needed a consistent, reproducible output, I would choose temperature=0.

# API Q3:
response3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
    }],
    n=3,
    temperature=1.0
)

print("\nAPI Q3:")

for i, response3.choice in enumerate(response3.choices):
    print(f"\nFun Fact #{i+1}",response3.choice.message.content)

# API Q4:
response4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Explain how neural networks work."
    }],
    max_tokens=15,
    temperature=0
)

print("\nAPI Q4:")
print(response4.choices[0].message.content)

# Comment:
# Max_tokens was set to 15, meaning only 15 words max.  Therefore, it seem to cut off after 15 words.
# It didn't finish what it had to say.  
# In real application, max_tokens helps control response length, reduce API costs, etc.

# --- System Messages and Personas ---
# System Q1:


# System Q2:
# --- Prompt Engineering ---
# Prompt Q1:
# Prompt Q2:
# Prompt Q3:
# Prompt Q4:
# Prompt Q5:
# Prompt Q6:
# --- Local Models with Ollama ---
# Ollama Q1:

