from llm_factory.openai_agent import OpenAIAgent
from schemas.parallelization_schema import ParallelizationType, Section, VotingConfig
from workflow.parallelization import ParallelProcessor
import json
from workflow.prompt_chaining import PromptChainer
from workflow.routing import WorkflowRouter
from schemas import RouteType
from workflow.orchestrator_workers import OrchestratorSystem
from workflow.evaluator_optimizer import EvaluatorOptimizer
from schemas import EvaluationCriteria,EvaluationType
class ExampleWorkflow:

    def prompt_chaining_example(self):
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

    def routing_example(self):
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

    def parallelization_example(self):
        agent = OpenAIAgent()
    
        # Example 1: Content Analysis using Sectioning
        sections = [
            Section(
                name="technical_analysis",
                system_prompt="You are a technical content analyst focusing on accuracy and technical depth.",
                task_prompt="Analyze the technical aspects of this content:",
                weight=1.0
            ),
            Section(
                name="readability_analysis",
                system_prompt="You are a readability expert focusing on clarity and accessibility.",
                task_prompt="Analyze the readability and clarity of this content:",
                weight=0.8
            ),
            Section(
                name="engagement_analysis",
                system_prompt="You are an engagement specialist focusing on user interest and appeal.",
                task_prompt="Analyze the engagement potential of this content:",
                weight=0.6
            )
        ]
        
        processor_sectioning = ParallelProcessor(agent, ParallelizationType.SECTIONING)
        
        # Example 2: Content Moderation using Voting
        voting_config = VotingConfig(
            prompt="You are a content moderator. Review the following content for appropriateness.",
            variations=[
                "Focus on hate speech and discriminatory content",
                "Focus on explicit adult content or inappropriate themes",
                "Focus on potentially harmful or dangerous information"
            ],
            threshold=0.7,
            aggregation_method="weighted"
        )
        
        processor_voting = ParallelProcessor(agent, ParallelizationType.VOTING)
        
        # Test content for analysis
        test_content = """
        Machine learning algorithms have revolutionized data analysis. 
        Neural networks can process complex patterns in datasets, 
        enabling applications from image recognition to natural language processing.
        """
        
        print("\nTesting Sectioning Parallelization:")
        section_results = processor_sectioning.process(
            test_content,
            sections=sections
        )
        print(json.dumps(section_results, indent=2))
        
        print("\nTesting Voting Parallelization:")
        voting_results = processor_voting.process(
            test_content,
            voting_config=voting_config
        )
        print(json.dumps(voting_results, indent=2))

    
    def orchestration_example(self):
        agent = OpenAIAgent()
        orchestrator = OrchestratorSystem(agent)
        
        # Example 1: Code Changes
        code_task = """
        Add input validation to the user registration form in our web app.
        The form should validate:
        - Email format
        - Password strength (min 8 chars, numbers, special chars)
        - Username (alphanumeric, 3-20 chars)
        Update both frontend validation and backend API validation.
        """
        
        print("\nProcessing Code Task:")
        code_result = orchestrator.process_task(code_task)
        print(f"Code Task Result: {code_result}")
        print(json.dumps(code_result, indent=2, ensure_ascii=False))

    def evaluator_optimizer_example(self):
        agent = OpenAIAgent()
    
        # Example 1: Translation Evaluation
        translation_criteria = [
            EvaluationCriteria(
                name="accuracy",
                description="Accuracy of meaning translation",
                weight=1.0
            ),
            EvaluationCriteria(
                name="fluency",
                description="Natural flow in target language",
                weight=0.8
            ),
            EvaluationCriteria(
                name="cultural_adaptation",
                description="Appropriate cultural context adaptation",
                weight=0.6
            )
        ]
        
        translation_evaluator = EvaluatorOptimizer(
            agent=agent,
            eval_type=EvaluationType.TRANSLATION,
            criteria=translation_criteria,
            max_iterations=3,
            target_score=0.9
        )
        
        

        # Test translation optimization
        translation_task = """
        Translate the following English text to French, maintaining the professional tone:
        'Our innovative approach to artificial intelligence combines cutting-edge technology 
        with ethical considerations, ensuring responsible development of AI solutions.'
        """
        
        print("\nOptimizing Translation:")
        translation_result = translation_evaluator.optimize(translation_task)
        print(json.dumps(translation_result, indent=2))