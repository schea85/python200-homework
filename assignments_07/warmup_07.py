from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from datetime import datetime

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.\n')

# --- Lesson 02: Tool Definitions and the ReAct Loop ---

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Q1:
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

tools = [
    {
        "type": "function",
        "function":{
            "name": "celsius_to_fahrenheit",
            "description": "Returns a Celsius temp to Fahrenheit",
            "parameters": {
                "type": "object",
                "properties": {
                        "celsius": {"type": "number"}
                    },
                "required": ["celsius"],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]

celsius_temps = [0, 10, -40]

for temp in celsius_temps:
    print(celsius_to_fahrenheit(temp))
    
# Q2:
def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
                
            elif function_name == "celsius_to_fahrenheit":
                arguments = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(**arguments)
                
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Prediction Comments:
# 1.) No tool call. The only tool the model has is the get_current_time, which does not match
# the temperature conversion request. Therefore, I predict that the model will answer using its own 
# knowledge.
# 2.) I predict that there will be one API call because the model does not need
# to use the available tool.  Instead, it will answer directly.

print("\n")
answer_with_agent_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(answer_with_agent_q2)

# Q3:
print("\n")
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Comment:
# A matching tool does exist so the agent calls it.

print("\n")
response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Comment:
# No tool needed, so the agent answers using its own knowledge.

# --- Lesson 03: Multi-Tool Agent ---

# Q4:
