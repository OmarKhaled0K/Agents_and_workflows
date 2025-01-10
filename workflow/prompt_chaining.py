from typing import List, Dict, Any, Optional, Callable
import json
from llm_factory.openai_agent import OpenAIAgent  # Assuming your previous code is in openai_agent.py

class PromptChainer:
    def __init__(self, agent: OpenAIAgent):
        self.agent = agent
        
    def sequential_chain(self, prompts: List[str], context: str = "") -> List[str]:
        """
        Execute prompts in sequence, where each response becomes context for the next prompt
        """
        responses = []
        current_context = context
        
        for prompt in prompts:
            full_prompt = f"Context: {current_context}\n\nTask: {prompt}" if current_context else prompt
            response = self.agent.generate_response(full_prompt)
            responses.append(response)
            current_context = response
            
        return responses
    
    def parallel_chain(self, base_prompt: str, follow_up_prompts: List[str]) -> Dict[str, str]:
        """
        Execute a base prompt and then run multiple follow-up prompts in parallel using the base response
        """
        base_response = self.agent.generate_response(base_prompt)
        results = {"base_response": base_response}
        
        for prompt in follow_up_prompts:
            full_prompt = f"Based on this information:\n{base_response}\n\nTask: {prompt}"
            response = self.agent.generate_response(full_prompt)
            results[prompt] = response
            
        return results
    
    def conditional_chain(
        self, 
        initial_prompt: str,
        condition_check: Callable[[str], bool],
        success_prompt: str,
        failure_prompt: str
    ) -> Dict[str, str]:
        """
        Execute different prompts based on the condition of previous responses
        """
        initial_response = self.agent.generate_response(initial_prompt)
        
        if condition_check(initial_response):
            next_prompt = success_prompt
        else:
            next_prompt = failure_prompt
            
        full_prompt = f"Based on the previous response: {initial_response}\n\nTask: {next_prompt}"
        final_response = self.agent.generate_response(full_prompt)
        
        return {
            "initial_response": initial_response,
            "final_response": final_response,
            "path_taken": "success" if condition_check(initial_response) else "failure"
        }
    
    def iterative_refinement_chain(
        self, 
        initial_prompt: str,
        refinement_prompt: str,
        max_iterations: int = 3,
        stop_condition: Optional[Callable[[str, str], bool]] = None
    ) -> List[str]:
        """
        Iteratively refine responses until a condition is met or max iterations reached
        """
        responses = []
        current_response = self.agent.generate_response(initial_prompt)
        responses.append(current_response)
        
        for i in range(max_iterations - 1):
            full_prompt = f"Previous response: {current_response}\n\nTask: {refinement_prompt}"
            new_response = self.agent.generate_response(full_prompt)
            responses.append(new_response)
            
            if stop_condition and stop_condition(current_response, new_response):
                break
                
            current_response = new_response
            
        return responses
    
    def branching_chain(
        self, 
        initial_prompt: str,
        branches: Dict[str, List[str]],
        branch_selector: Callable[[str], str]
    ) -> Dict[str, List[str]]:
        """
        Execute different chains based on the initial response
        """
        initial_response = self.agent.generate_response(initial_prompt)
        selected_branch = branch_selector(initial_response)
        
        if selected_branch not in branches:
            raise ValueError(f"Branch '{selected_branch}' not found in available branches")
            
        branch_responses = []
        current_context = initial_response
        
        for prompt in branches[selected_branch]:
            full_prompt = f"Context: {current_context}\n\nTask: {prompt}"
            response = self.agent.generate_response(full_prompt)
            branch_responses.append(response)
            current_context = response
            
        return {
            "initial_response": initial_response,
            "selected_branch": selected_branch,
            "branch_responses": branch_responses
        }

# Example usage
if __name__ == "__main__":
    agent = OpenAIAgent()
    chainer = PromptChainer(agent)
    
    # Example 1: Sequential Chain
    print("\nSequential Chain Example:")
    sequential_prompts = [
        "Write a short story about a robot",
        "Transform the previous story into a poem",
        "Create a movie script outline based on the poem"
    ]
    sequential_results = chainer.sequential_chain(sequential_prompts)
    for i, result in enumerate(sequential_results):
        print(f"\nStep {i+1}:\n{result}")
    
    # Example 2: Parallel Chain
    print("\nParallel Chain Example:")
    base_prompt = "Explain the concept of machine learning"
    follow_up_prompts = [
        "What are the potential applications in healthcare?",
        "What are the ethical considerations?",
        "How might this impact employment?"
    ]
    parallel_results = chainer.parallel_chain(base_prompt, follow_up_prompts)
    for prompt, response in parallel_results.items():
        print(f"\nPrompt: {prompt}\nResponse: {response}\n")
    
    # Example 3: Conditional Chain
    print("\nConditional Chain Example:")
    def check_sentiment(text: str) -> bool:
        # Simple sentiment check (you might want to use a proper sentiment analyzer)
        positive_words = ['good', 'great', 'excellent', 'positive', 'wonderful']
        return any(word in text.lower() for word in positive_words)
    
    conditional_results = chainer.conditional_chain(
        "What do you think about the future of AI?",
        check_sentiment,
        "Elaborate on these positive aspects",
        "What solutions would you propose for these concerns?"
    )
    print(json.dumps(conditional_results, indent=2))
    
    # Example 4: Iterative Refinement
    print("\nIterative Refinement Example:")
    def check_improvement(prev: str, new: str) -> bool:
        # Stop if the new response is not significantly longer than the previous one
        return len(new) <= len(prev) * 1.1
    
    refinement_results = chainer.iterative_refinement_chain(
        "Explain quantum computing",
        "Make this explanation more detailed and precise",
        max_iterations=3,
        stop_condition=check_improvement
    )
    for i, result in enumerate(refinement_results):
        print(f"\nIteration {i+1}:\n{result}")
    
    # Example 5: Branching Chain
    print("\nBranching Chain Example:")
    def select_branch(response: str) -> str:
        # Select branch based on response content
        if 'technical' in response.lower():
            return 'technical'
        else:
            return 'general'
    
    branches = {
        'technical': [
            "Explain the technical architecture",
            "Discuss implementation challenges",
            "Provide code examples"
        ],
        'general': [
            "Explain in simple terms",
            "Give real-world analogies",
            "Provide practical applications"
        ]
    }
    
    branching_results = chainer.branching_chain(
        "Explain how neural networks work",
        branches,
        select_branch
    )
    print(json.dumps(branching_results, indent=2))