import openai
from typing import Optional, Dict, Any, List
from configs import Config, get_settings
import requests
from datetime import datetime
from tavily import TavilyClient

settings = get_settings()

class SearchTool:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key)
    
    def search(self, query: str, search_depth: str = "basic") -> List[Dict[str, Any]]:
        """
        Perform a web search using Tavily API
        search_depth: 'basic' or 'advanced' - basic is faster, advanced is more comprehensive
        """
        response = self.client.search(
            query=query,
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=False,
            max_results=5
        )
        
        return response

class OpenAIAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.search_tool = SearchTool(settings.tavily_api_key)
        
        # Define the function for OpenAI to use
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for real-time information about any topic. Only use this when you need up-to-date information or facts you're not confident about.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "search_depth": {
                                "type": "string",
                                "enum": ["basic", "advanced"],
                                "description": "The depth of search - basic is faster, advanced is more comprehensive"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant. You have access to a search tool, but only use it when you need current information or when you're not confident about facts. For general knowledge, common questions, or creative tasks like jokes or stories, respond directly without using the tool."
        }, {
            "role": "user",
            "content": prompt
        }]
        
        response = self.client.chat.completions.create(
            model=model if model is not None else self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",  # Let the model decide whether to use tools
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens
        )
        
        response_message = response.choices[0].message
        #print(f"Initial response: {response_message}")
        
        # Check if the model wants to call a function
        if tool_calls := response_message.tool_calls:
            print(f"Using search tool...")
            # Handle each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = eval(tool_call.function.arguments)
                
                if function_name == "search":
                    # Call the search API
                    search_results = self.search_tool.search(**function_args)
                    
                    # Add the function response to the messages
                    messages.append(response_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(search_results)
                    })
            
            # Get a new response from the model with the search results
            second_response = self.client.chat.completions.create(
                model=model if model is not None else self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
            
            return second_response.choices[0].message.content
        else:
            print("Responding without search tool...")
        
        return response_message.content

# Example usage
if __name__ == "__main__":
    agent = OpenAIAgent()
    
    # Example prompts to test different scenarios
    prompts = [
        "Give me a joke about AI",  # Should not use search
        "What are the latest developments in AI?",  # Should use search
        "Tell me a story about a cat",  # Should not use search
        "What are the current trending technologies in 2024?"  # Should use search
    ]
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        response = agent.generate_response(prompt)
        print(f"Response: {response}")