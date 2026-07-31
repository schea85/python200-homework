from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Task 1 - Setup and System Prompt: ---

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

# --- Task 2 - Bullet Point Rewriter: ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    IMPORTANT: Return ONLY a valid JSON, no other text.
    Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    
    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)
    print(response)
        
    # parse JSON safely
    try:
        result = json.loads(response.replace("```json", "").replace("```", "").strip())
            
        for item in result:
            print("Original:", item["original"])
            print("Improved:", item["improved"])
            print("\n")
            
        return result
        
    except json.JSONDecodeError:
        print("Error: response was not a valid JSON")
        return []

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
        
rewrite_bullets(bullets)

# Comment:
# The original bullets are weak because they are too general and do not show specific skills or impact that
# employers are looking for.
# The improved bullets sounds more professional by using stronger action
# verbs, adding more detail, and making the bullets sound more achievement-focused.

