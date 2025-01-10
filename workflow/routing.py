from typing import Dict, List, Any, Union
from llm_factory.openai_agent import OpenAIAgent
from schemas import RouteConfig, RouteType

class WorkflowRouter:
    def __init__(
        self,
        agent: OpenAIAgent,
        route_type: RouteType = RouteType.SINGLE,
    ):
        self.agent = agent
        self.routes: Dict[str, RouteConfig] = {}
        self.route_type = route_type
        
    def add_route(
        self,
        name: str,
        description: str,
        system_prompt: str,
        response_template: str,
        confidence_threshold: float = 0.5,
        priority: int = 0
    ) -> None:
        """Add a new route configuration"""
        self.routes[name] = RouteConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            response_template=response_template,
            confidence_threshold=confidence_threshold,
            priority=priority
        )
        
    def _generate_routing_prompt(self, input_text: str) -> str:
        """Generate the routing prompt for the LLM"""
        routes_desc = "\n".join([
            f"- {name}: {route.description}" 
            for name, route in self.routes.items()
        ])
        
        return f"""Given the following input, determine the most appropriate routing(s).

                    Available routes:
                    {routes_desc}

                    Respond in the following JSON format:
                    {{
                        "routes": [
                            {{
                                "name": "route_name",
                                "confidence": 0.0 to 1.0,
                                "reasoning": "brief explanation"
                            }}
                        ]
                    }}

                    Input text: {input_text}

                    Provide route recommendations in order of confidence."""
        
    def _parse_route_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into structured route data"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            result = eval(json_str)
            return result.get('routes', [])
        except Exception as e:
            print(f"Error parsing route response: {e}")
            return []

    def _generate_final_response(self, input_text: str, route: RouteConfig) -> str:
        """Generate the final response using the route's configuration"""
        prompt = f"""{route.system_prompt}

                        User Input: {input_text}

                        {route.response_template}"""

        return self.agent.generate_response(prompt)

    def process_input(self, input_text: str) -> Union[str, List[str]]:
        """Process input through routing and response generation"""
        # Get routing recommendations
        routing_prompt = self._generate_routing_prompt(input_text)
        routing_response = self.agent.generate_response(routing_prompt)
        route_recommendations = self._parse_route_response(routing_response)

        if not route_recommendations:
            return "I apologize, but I'm unable to properly categorize your request. Could you please rephrase or provide more details?"

        # Handle different routing types
        if self.route_type == RouteType.SINGLE:
            # Take highest confidence route
            top_route = route_recommendations[0]
            route_config = self.routes.get(top_route['name'])
            if route_config and top_route['confidence'] >= route_config.confidence_threshold:
                return self._generate_final_response(input_text, route_config)
            
        elif self.route_type == RouteType.MULTI:
            # Process all routes above threshold
            responses = []
            for route_rec in route_recommendations:
                route_config = self.routes.get(route_rec['name'])
                if route_config and route_rec['confidence'] >= route_config.confidence_threshold:
                    response = self._generate_final_response(input_text, route_config)
                    responses.append(response)
            return responses if responses else ["No suitable routes found for your request."]
            
        elif self.route_type == RouteType.PRIORITY:
            # Try routes in priority order
            sorted_recommendations = sorted(
                route_recommendations,
                key=lambda x: self.routes.get(x['name'], RouteConfig("", "", "", "", priority=0)).priority,
                reverse=True
            )
            
            for route_rec in sorted_recommendations:
                route_config = self.routes.get(route_rec['name'])
                if route_config and route_rec['confidence'] >= route_config.confidence_threshold:
                    try:
                        return self._generate_final_response(input_text, route_config)
                    except Exception as e:
                        print(f"Handler {route_rec['name']} failed: {e}")
                        continue

        return "I apologize, but I'm unable to properly process your request at this time."

# Example usage
if __name__ == "__main__":
    agent = OpenAIAgent()
    workflow = WorkflowRouter(agent, route_type=RouteType.SINGLE)
    
    # Configure routes with appropriate prompts and templates
    # Technical Support Route
    workflow.add_route(
        name="technical_support",
        description="Technical issues, error messages, or software/hardware problems",
        system_prompt="""You are a technical support specialist. Your role is to:
                            1. Identify the technical issue
                            2. Provide step-by-step troubleshooting
                            3. Explain solutions in clear, technical but accessible language""",
        response_template="""Please provide a detailed technical support response including:
                                1. Problem identification
                                2. Step-by-step troubleshooting steps
                                3. Additional recommendations""",
        confidence_threshold=0.7
    )
    
    # Customer Service Route
    workflow.add_route(
        name="customer_service",
        description="General inquiries, account issues, or policy questions",
        system_prompt="""You are a customer service representative. Your role is to:
                            1. Address customer concerns empathetically
                            2. Provide clear policy information
                            3. Offer solutions that align with company policies""",
        response_template="""Please provide a customer-friendly response that:
                            1. Acknowledges the customer's concern
                            2. Explains relevant policies
                            3. Offers clear next steps""",
        confidence_threshold=0.6
    )
    
    # Sales Route
    workflow.add_route(
        name="sales",
        description="Product inquiries, pricing questions, or purchase intentions",
        system_prompt="""You are a sales representative. Your role is to:
                        1. Understand customer needs
                        2. Explain product benefits
                        3. Provide relevant pricing and purchasing information""",
        response_template="""Please provide a sales-focused response that:
                            1. Addresses the customer's interest
                            2. Highlights relevant benefits
                            3. Provides clear pricing/purchase information""",
        confidence_threshold=0.6
    )
    
    # Test the workflow
    test_inputs = [
        "My laptop won't turn on after the latest update",
        #"What are your prices for the premium plan?",
        #"I need to reset my account password",
        #"Can you explain how your product compares to competitors?",
    ]
    
    for test_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"Input: {test_input}")
        result = workflow.process_input(test_input)
        print(f"Response: {result}")