from dotenv import load_dotenv
from openai import OpenAI

# task1 - setup and system prompt:
if load_dotenv():
    print("Successfully connected to api key")

load_dotenv()
client = OpenAI()

messages =[{
    "role": "system",
    "content": """
    You are a friendly job application coach helping job seekers improve their resumes,
    cover letters, LinkedIn profiles, and other application materials.  
    
    Provide clear, constructive feedback and help candidates make their materials more effective 
    for their target roles.  Ask clarifying questions when important details are missing.  Stay focused
    on job application materials.
    
    Always remind the user to review and edit the outputs before submitting their job applications.  
    Acknowledge that you might not know the user's specific industry norms, and 
    that the user should use their own judgment.
    """
}]

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

response = get_completion(messages)
print(response)

# Comment:
# I chose to include instructions for the AI coach to ask clarifying questions because
# job application advice depends on the candidate's background, target role, and industry.
# This helps the model make its responses more relevant and personalized.