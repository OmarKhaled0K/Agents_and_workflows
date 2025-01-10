from typing import List, Dict, Any, Optional
import json
from llm_factory.openai_agent import OpenAIAgent
from schemas import EvaluationCriteria, EvaluationResult, EvaluationType

class EvaluatorOptimizer:
    def __init__(
        self,
        agent: OpenAIAgent,
        eval_type: EvaluationType,
        criteria: List[EvaluationCriteria],
        max_iterations: int = 3,
        target_score: float = 0.9
    ):
        self.agent = agent
        self.eval_type = eval_type
        self.criteria = criteria
        self.max_iterations = max_iterations
        self.target_score = target_score
        
    def _generate_optimizer_prompt(
        self,
        task: str,
        previous_result: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate prompt for the optimizer"""
        base_prompt = f"""Task: {task}

Evaluation Type: {self.eval_type.value}

Criteria to consider:
{self._format_criteria()}"""

        if previous_result and feedback:
            base_prompt += f"""

Previous Result:
{previous_result}

Feedback Received:
{json.dumps(feedback, indent=2)}

Please improve the result based on the feedback while maintaining the original intent.
Focus especially on areas with lower scores.
"""
        
        return base_prompt

    def _generate_evaluator_prompt(self, task: str, result: str) -> str:
        """Generate prompt for the evaluator"""
        return f"""Evaluate the following result based on specified criteria.

Task: {task}
Evaluation Type: {self.eval_type.value}

Result to evaluate:
{result}

Criteria:
{self._format_criteria()}

Provide evaluation in JSON format:
{{
    "scores": {{
        "criteria_name": score (0.0 to 1.0)
    }},
    "feedback": {{
        "criteria_name": "detailed feedback"
    }},
    "overall_score": 0.0 to 1.0,
    "suggestions": [
        "specific improvement suggestions"
    ]
}}

Ensure feedback is specific and actionable."""

    def _format_criteria(self) -> str:
        """Format evaluation criteria for prompts"""
        return "\n".join([
            f"- {c.name} (weight: {c.weight}): {c.description}"
            for c in self.criteria
        ])

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_weight = sum(c.weight for c in self.criteria)
        weighted_sum = sum(
            scores[c.name] * c.weight
            for c in self.criteria
            if c.name in scores
        )
        return weighted_sum / total_weight

    def optimize(self, task: str, initial_result: Optional[str] = None) -> Dict[str, Any]:
        """Run the evaluation-optimization loop"""
        current_result = initial_result
        history = []
        
        for iteration in range(self.max_iterations):
            # Generate result if needed
            if not current_result:
                optimizer_prompt = self._generate_optimizer_prompt(task)
                current_result = self.agent.generate_response(optimizer_prompt)
            
            # Evaluate current result
            evaluator_prompt = self._generate_evaluator_prompt(task, current_result)
            evaluation_response = self.agent.generate_response(evaluator_prompt)
            
            try:
                evaluation = json.loads(
                    evaluation_response[
                        evaluation_response.find('{'):evaluation_response.rfind('}')+1
                    ]
                )
                
                evaluation_result = EvaluationResult(
                    scores=evaluation["scores"],
                    feedback=evaluation["feedback"],
                    overall_score=evaluation["overall_score"],
                    suggestions=evaluation["suggestions"],
                    iteration=iteration + 1
                )
                
                # Convert EvaluationResult to dict before adding to history
                history.append({
                    "iteration": iteration + 1,
                    "result": current_result,
                    "evaluation": evaluation_result.to_dict()  # Use to_dict() here
                })
                
                # Check if we've reached target score
                if evaluation_result.overall_score >= self.target_score:
                    break
                
                # Generate improved result
                optimizer_prompt = self._generate_optimizer_prompt(
                    task,
                    current_result,
                    evaluation
                )
                current_result = self.agent.generate_response(optimizer_prompt)
                
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                break
        
        return {
            "final_result": current_result,
            "iterations": history,
            "final_score": history[-1]["evaluation"]["overall_score"] if history else 0.0,  # Updated to access dict
            "improvement_summary": self._generate_improvement_summary(history)
        }

    def _generate_improvement_summary(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of improvements across iterations"""
        if not history:
            return {}
            
        first_eval = history[0]["evaluation"]  # Now accessing dict directly
        last_eval = history[-1]["evaluation"]  # Now accessing dict directly
        
        improvements = {}
        for criterion in self.criteria:
            name = criterion.name
            if name in first_eval["scores"] and name in last_eval["scores"]:  # Updated to access dict
                improvements[name] = {
                    "initial_score": first_eval["scores"][name],
                    "final_score": last_eval["scores"][name],
                    "improvement": last_eval["scores"][name] - first_eval["scores"][name]
                }
        
        return {
            "criteria_improvements": improvements,
            "overall_improvement": last_eval["overall_score"] - first_eval["overall_score"],  # Updated to access dict
            "iterations_required": len(history)
        }
    
    def _generate_improvement_summary(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of improvements across iterations"""
        if not history:
            return {}
            
        first_eval = history[0]["evaluation"]
        last_eval = history[-1]["evaluation"]
        
        improvements = {}
        for criterion in self.criteria:
            name = criterion.name
            if name in first_eval['scores'] and name in last_eval['scores']:
                improvements[name] = {
                    "initial_score": first_eval['scores'][name],
                    "final_score": last_eval['scores'][name],
                    "improvement": last_eval['scores'][name] - first_eval['scores'][name]
                }
        
        return {
            "criteria_improvements": improvements,
            "overall_improvement": last_eval['overall_score'] - first_eval['overall_score'],
            "iterations_required": len(history)
        }

# Example usage
if __name__ == "__main__":
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
    
    # Example 2: Writing Evaluation
    writing_criteria = [
        EvaluationCriteria(
            name="clarity",
            description="Clear and easy to understand",
            weight=1.0
        ),
        EvaluationCriteria(
            name="coherence",
            description="Logical flow and structure",
            weight=0.8
        ),
        EvaluationCriteria(
            name="engagement",
            description="Engaging and interesting content",
            weight=0.6
        ),
        EvaluationCriteria(
            name="technical_accuracy",
            description="Accurate technical information",
            weight=1.0
        )
    ]
    
    writing_evaluator = EvaluatorOptimizer(
        agent=agent,
        eval_type=EvaluationType.WRITING,
        criteria=writing_criteria,
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
    
    # # Test technical writing optimization
    # writing_task = """
    # Write a technical blog post explaining the concept of quantum computing
    # to software developers who are new to the field.
    # """
    
    # print("\nOptimizing Technical Writing:")
    # writing_result = await writing_evaluator.optimize(writing_task)
    # print(json.dumps(writing_result, indent=2))

    