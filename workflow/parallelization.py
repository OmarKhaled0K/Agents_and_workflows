from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from llm_factory.openai_agent import OpenAIAgent
from schemas.parallelization_schema import ParallelizationType, Section, VotingConfig


class ParallelProcessor:
    def __init__(
        self,
        agent: OpenAIAgent,
        parallel_type: ParallelizationType,
        max_workers: int = 3
    ):
        self.agent = agent
        self.parallel_type = parallel_type
        
    def process_section(self, section: Section, input_text: str) -> Dict[str, Any]:
        """Process a single section"""
        prompt = f"""{section.system_prompt}

Input: {input_text}

{section.task_prompt}

Provide your response in JSON format with the following structure:
{{
    "analysis": "your detailed analysis",
    "key_points": ["list", "of", "key", "points"],
    "confidence": 0.0 to 1.0
}}"""
        
        response = self.agent.generate_response(prompt)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            result = json.loads(response[start:end])
            result['section_name'] = section.name
            result['weight'] = section.weight
            return result
        except Exception as e:
            print(f"Error processing section {section.name}: {e}")
            return {
                "section_name": section.name,
                "analysis": "Error processing section",
                "key_points": [],
                "confidence": 0.0,
                "weight": section.weight
            }

    def process_vote(
        self,
        base_prompt: str,
        variation: str,
        input_text: str
    ) -> Dict[str, Any]:
        """Process a single voting variation"""
        prompt = f"""{base_prompt}

Input to analyze: {input_text}

Specific focus: {variation}

Provide your response in JSON format with the following structure:
{{
    "vote": true/false,
    "confidence": 0.0 to 1.0,
    "reasoning": "explanation for your vote"
}}"""
        
        response = self.agent.generate_response(prompt)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            result = json.loads(response[start:end])
            result['variation'] = variation
            return result
        except Exception as e:
            print(f"Error processing vote for variation {variation}: {e}")
            return {
                "vote": False,
                "confidence": 0.0,
                "reasoning": f"Error processing vote: {str(e)}",
                "variation": variation
            }

    def aggregate_sections(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from sectioning parallelization"""
        total_weight = sum(result['weight'] for result in results)
        weighted_confidence = sum(
            result['confidence'] * result['weight'] 
            for result in results
        ) / total_weight
        
        all_key_points = []
        detailed_analysis = {}
        
        for result in results:
            all_key_points.extend(result['key_points'])
            detailed_analysis[result['section_name']] = {
                'analysis': result['analysis'],
                'confidence': result['confidence']
            }
        
        return {
            "overall_confidence": weighted_confidence,
            "key_points": list(set(all_key_points)),  # Remove duplicates
            "detailed_analysis": detailed_analysis,
            "summary": self._generate_summary(results)
        }

    def aggregate_votes(
        self,
        results: List[Dict[str, Any]],
        config: VotingConfig
    ) -> Dict[str, Any]:
        """Aggregate results from voting parallelization"""
        total_votes = len(results)
        positive_votes = sum(1 for r in results if r['vote'])
        
        # Calculate weighted confidence
        total_confidence = sum(r['confidence'] for r in results)
        average_confidence = total_confidence / total_votes
        
        # Determine final decision based on aggregation method
        decision = False
        if config.aggregation_method == "majority":
            decision = (positive_votes / total_votes) >= config.threshold
        elif config.aggregation_method == "unanimous":
            decision = positive_votes == total_votes
        elif config.aggregation_method == "weighted":
            weighted_positive = sum(
                r['confidence'] for r in results if r['vote']
            )
            decision = (weighted_positive / total_confidence) >= config.threshold
        
        return {
            "decision": decision,
            "confidence": average_confidence,
            "vote_ratio": positive_votes / total_votes,
            "detailed_votes": [
                {
                    "variation": r['variation'],
                    "vote": r['vote'],
                    "confidence": r['confidence'],
                    "reasoning": r['reasoning']
                }
                for r in results
            ]
        }

    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary prompt based on all section results"""
        summary_prompt = f"""Based on the following analyses, provide a concise summary:

{json.dumps(results, indent=2)}

Provide a coherent summary that integrates all perspectives."""
        
        return self.agent.generate_response(summary_prompt)

    def process(
        self,
        input_text: str,
        sections: Optional[List[Section]] = None,
        voting_config: Optional[VotingConfig] = None
    ) -> Dict[str, Any]:
        """Process input using either sectioning or voting parallelization"""
        if self.parallel_type == ParallelizationType.SECTIONING:
            if not sections:
                raise ValueError("Sections must be provided for sectioning parallelization")
            
            results = [
                self.process_section(section, input_text)
                for section in sections
            ]
            return self.aggregate_sections(results)
            
        else:  # VOTING
            if not voting_config:
                raise ValueError("Voting config must be provided for voting parallelization")
            
            results = [
                self.process_vote(voting_config.prompt, variation, input_text)
                for variation in voting_config.variations
            ]
            return self.aggregate_votes(results, voting_config)

# Example usage
if __name__ == "__main__":
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