import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
import csv
from datetime import datetime

# Load environment variables from .env file
load_dotenv("/mnt/f/New Giza University/Courses/Spring 2024-2025/Data Mining/Week 5/Conversational Agents/.env")
API_KEY = os.environ.get("API_KEY", os.getenv('OPTOGPT_API_KEY'))
BASE_URL = os.environ.get("BASE_URL", os.getenv('BASE_URL'))
LLM_MODEL = os.environ.get("LLM_MODEL", os.getenv('OPTOGPT_MODEL'))
# Initialize the OpenAI client with custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Part 1: Implementing Basic Tool Calling
def get_current_weather(location):
    """Get the current weather for a location."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
    
    response = requests.get(url)
    data = response.json()
    
    if "error" in data:
        return f"Error: {data['error']['message']}"
    
    weather_info = data["current"]
    return json.dumps({
        "location": data["location"]["name"],
        "temperature_c": weather_info["temp_c"],
        "temperature_f": weather_info["temp_f"],
        "condition": weather_info["condition"]["text"],
        "humidity": weather_info["humidity"],
        "wind_kph": weather_info["wind_kph"]
    })

def get_weather_forecast(location, days=3):
    """Get a weather forecast for a location for a specified number of days."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days={days}&aqi=no"
    
    response = requests.get(url)
    data = response.json()
    try:
        days = int(days)
    except (ValueError, TypeError):
        return "Error: Days parameter must be an integer"
    if days < 1 or days > 10:
        return "Error: Days must be between 1 and 10"
    if "error" in data:
        return f"Error: {data['error']['message']}"
    
    forecast_days = data["forecast"]["forecastday"]
    forecast_data = []
    
    for day in forecast_days:
        forecast_data.append({
            "date": day["date"],
            "max_temp_c": day["day"]["maxtemp_c"],
            "min_temp_c": day["day"]["mintemp_c"],
            "condition": day["day"]["condition"]["text"],
            "chance_of_rain": day["day"]["daily_chance_of_rain"]
        })
    
    return json.dumps({
        "location": data["location"]["name"],
        "forecast": forecast_data
    })

# Define tools for the OpenAI API
weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA or country e.g., France",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get the weather forecast for a location for a specific number of days",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA or country e.g., France",
                    },
                    "days": {
                        "type": "integer",
                        "description": "The number of days to forecast (1-10)",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["location"],
            },
        },
    }
]

# Create a lookup dictionary
available_functions = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast
}

# Part 2: Enhancing with Chain of Thought Reasoning
def calculator(expression):
    """
    Evaluate a mathematical expression.
    Args:
        expression: A mathematical expression as a string
    Returns:
        The result of the evaluation
    """
    try:
        # Safely evaluate the expression
        # Note: This is not completely safe for production use
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define the calculator tool
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 2' or '5 * (3 + 2)'",
                }
            },
            "required": ["expression"],
        },
    }
}

# Add calculator to weather tools and available functions
cot_tools = weather_tools + [calculator_tool]
available_functions["calculator"] = calculator

# Chain of Thought system message
cot_system_message = """You are a helpful assistant that can answer questions about weather and perform calculations.
When responding to complex questions, please follow these steps:
1. Think step-by-step about what information you need
2. Break down the problem into smaller parts
3. Use the appropriate tools to gather information
4. Explain your reasoning clearly
5. Provide a clear final answer

For example, if someone asks about temperature conversions or comparisons between cities, first get the weather data, then use the calculator if needed, showing your work.
"""

# Part 3: Implementing ReAct Reasoning
def web_search(query):
    """
    Simulate a web search for information.
    Args:
        query: The search query
    Returns:
        Search results as JSON
    """
    # This is a simulated search function
    # In a real application, you would use an actual search API
    search_results = {
        "weather forecast": "Weather forecasts predict atmospheric conditions for a specific location and time period. They typically include temperature, precipitation, wind, and other variables.",
        "temperature conversion": "To convert Celsius to Fahrenheit: multiply by 9/5 and add 32. To convert Fahrenheit to Celsius: subtract 32 and multiply by 5/9.",
        "climate change": "Climate change refers to significant changes in global temperature, precipitation, wind patterns, and other measures of climate that occur over several decades or longer.",
        "severe weather": "Severe weather includes thunderstorms, tornadoes, hurricanes, blizzards, floods, and high winds that can cause damage, disruption, and loss of life."
    }
    
    # Find the closest matching key
    best_match = None
    best_match_score = 0
    
    for key in search_results:
        # Simple matching algorithm
        words_in_query = set(query.lower().split())
        words_in_key = set(key.lower().split())
        match_score = len(words_in_query.intersection(words_in_key))
        
        if match_score > best_match_score:
            best_match = key
            best_match_score = match_score
    
    if best_match_score > 0:
        return json.dumps({"query": query, "result": search_results[best_match]})
    else:
        return json.dumps({"query": query, "result": "No relevant information found."})

# Define the search tool
search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search for information on the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    }
}

# Create ReAct tools and available functions
react_tools = cot_tools + [search_tool]
available_functions["web_search"] = web_search

# ReAct system message
react_system_message = """You are a helpful weather and information assistant that uses the ReAct (Reasoning and Acting) approach to solve problems.

When responding to questions, follow this pattern:
1. Thought: Think about what you need to know and what steps to take
2. Action: Use a tool to gather information (weather data, search, calculator)
3. Observation: Review what you learned from the tool
4. ... (repeat the Thought, Action, Observation steps as needed)
5. Final Answer: Provide your response based on all observations

For example:
User: What's the temperature difference between New York and London today?
Thought: I need to find the current temperatures in both New York and London, then calculate the difference.
Action: [Use get_current_weather for New York]
Observation: [Results from weather tool]
Thought: Now I need London's temperature.
Action: [Use get_current_weather for London]
Observation: [Results from weather tool]
Thought: Now I can calculate the difference.
Action: [Use calculator to subtract]
Observation: [Result of calculation]
Final Answer: The temperature difference between New York and London today is X degrees.

Always make your reasoning explicit and show your work.
Important format rules:
- Assistant messages MUST EITHER contain content OR tool_calls
- Never include empty tool_calls in messages
"""

def process_messages(client, messages, tools=None, available_functions=None):
    tools = tools or []
    available_functions = available_functions or {}
    MAX_ITERATIONS = 3
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        # Convert messages to API-compatible format
        dict_messages = []
        for msg in messages:
            # Handle both dictionary and object formats
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", None)
                tool_call_id = msg.get("tool_call_id", None)
            else:
                role = msg.role
                content = msg.content
                tool_calls = getattr(msg, "tool_calls", None)
                tool_call_id = getattr(msg, "tool_call_id", None)
            
            entry = {"role": role, "content": content}
            
            if role == "assistant" and tool_calls:
                entry["tool_calls"] = tool_calls
            elif role == "tool":
                entry["tool_call_id"] = tool_call_id  # Critical fix
            
            dict_messages.append(entry)
        
        # Get model response
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=dict_messages,
            tools=tools,
        )
        
        # Store response properly
        response_message = response.choices[0].message
        assistant_message = {
            "role": "assistant",
            "content": response_message.content or ""
        }
        if response_message.tool_calls:
            assistant_message["tool_calls"] = response_message.tool_calls
        
        messages.append(assistant_message)
        
        if response_message.content:
            return messages
            
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                
                if not function_to_call:
                    # Include tool_call_id in error response
                    messages.append({
                        "role": "tool",
                        "content": f"Error: Function {function_name} not found",
                        "tool_call_id": tool_call.id  # <-- This was missing
                    })
                    continue
                    
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)
                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                        "tool_call_id": tool_call.id  # <-- Mandatory field
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "content": f"Error: {str(e)}",
                        "tool_call_id": tool_call.id  # <-- This was missing
                    })
        else:
            break
            
    if not messages[-1].get("content"):
        messages.append({
            "role": "assistant",
            "content": "I need more information to answer that. Could you please clarify?"
        })
        
    return messages

# Function to run the conversation
def run_conversation(client, system_message="You are a helpful weather assistant.", tools=None):
    """
    Run a conversation with the user, processing their messages and handling tool calls.
    
    Args:
        client: The OpenAI client
        system_message: The system message to initialize the conversation
        tools: The tools to make available to the agent
        
    Returns:
        The final conversation history
    """
    if tools is None:
        tools = weather_tools
        
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    
    print("Weather Assistant: Hello! I can help you with weather information. Ask me about the weather anywhere!")
    print("(Type 'exit' to end the conversation)\n")
    
    while True:
        # Request user input and append to messages
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nWeather Assistant: Goodbye! Have a great day!")
            break
            
        messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Process the messages and get tool calls if any
        messages = process_messages(client, messages, tools, available_functions)
        
        # After processing messages, search for the latest assistant message
        last_assistant_message = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant_message = msg["content"]
                break
        
        if last_assistant_message:
            print(f"\nWeather Assistant: {last_assistant_message}\n")
        else:
            print("\nWeather Assistant: Hmm, I'm having trouble with that. Could you rephrase?\n")

def comparative_evaluation(client, user_query):
    """
    Process a single query with all three agent types and compare the results
    
    Args:
        client: The OpenAI client
        user_query: The user's query to process
        
    Returns:
        None (saves results to CSV)
    """
    # Define the agent configurations
    agent_configs = [
        {"name": "Basic", "system_message": "You are a helpful weather assistant.", "tools": weather_tools},
        {"name": "Chain of Thought", "system_message": cot_system_message, "tools": cot_tools},
        {"name": "ReAct", "system_message": react_system_message, "tools": react_tools}
    ]
    
    results = []
    
    # Process the query with each agent type
    for config in agent_configs:
        print(f"\n===== Processing with {config['name']} Agent =====")
        
        # Initialize conversation with system message
        messages = [
            {
                "role": "system",
                "content": config["system_message"],
            },
            {
                "role": "user",
                "content": user_query,
            }
        ]
        
        # Process the message
        processed_messages = process_messages(client, messages.copy(), config["tools"], available_functions)
        
        # Get the response
        response = ""
        for message in processed_messages:
            # Handle both dictionary-type messages and ChatCompletionMessage objects
            if isinstance(message, dict):
                if message["role"] == "assistant" and message.get("content"):
                    response = message["content"]
                    break
            else:  # It's a ChatCompletionMessage object
                if message.role == "assistant" and message.content:
                    response = message.content
                    break
        
        # Print the response
        print(f"\n{config['name']} Agent Response:")
        print(response)
        
        # Collect user rating
        rating = 0
        while not (1 <= rating <= 5):
            try:
                rating = int(input(f"\nRate the {config['name']} agent's response (1-5): "))
                if not (1 <= rating <= 5):
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Store result
        results.append({
            "agent_type": config["name"],
            "query": user_query,
            "response": response,
            "rating": rating
        })
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_evaluation_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["agent_type", "query", "response", "rating"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nEvaluation results saved to {filename}")
    
    # Print a summary
    print("\nEvaluation Summary:")
    for result in results:
        print(f"{result['agent_type']} Agent: {result['rating']}/5")

# Main function
if __name__ == "__main__":
    print("CSAI 422: Conversational Agent with Tool Use and Reasoning")
    print("1: Basic Agent")
    print("2: Chain of Thought Agent")
    print("3: ReAct Agent")
    print("4: Comparative Evaluation")
    
    choice = input("\nChoose an option (1-4): ")
    
    if choice == "1":
        print("\nRunning Basic Agent:")
        run_conversation(client, "You are a helpful weather assistant.", weather_tools)
    elif choice == "2":
        print("\nRunning Chain of Thought Agent:")
        run_conversation(client, cot_system_message, cot_tools)
    elif choice == "3":
        print("\nRunning ReAct Agent:")
        run_conversation(client, react_system_message, react_tools)
    elif choice == "4":
        print("\nRunning Comparative Evaluation:")
        user_query = input("Enter a query to evaluate with all three agents: ")
        comparative_evaluation(client, user_query)
    else:
        print("Invalid choice. Defaulting to Basic agent.")
        run_conversation(client, "You are a helpful weather assistant.", weather_tools)