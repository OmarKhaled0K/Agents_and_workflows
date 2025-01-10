import argparse
from examples.example_workflow import ExampleWorkflow

def main():
    parser = argparse.ArgumentParser(description="Run specific example workflows.")
    parser.add_argument(
        "example",
        choices=[
            "prompt_chaining",
            "routing",
            "parallelization",
            "orchestration",
            "evaluator_optimizer",
        ],
        nargs="?",
        default="prompt_chaining",  # Default value
        help="Specify the example workflow to run. Default is 'prompt_chaining'."
    )

    args = parser.parse_args()
    example_workflow = ExampleWorkflow()

    if args.example == "prompt_chaining":
        example_workflow.prompt_chaining_example()
    elif args.example == "routing":
        example_workflow.routing_example()
    elif args.example == "parallelization":
        example_workflow.parallelization_example()
    elif args.example == "orchestration":
        example_workflow.orchestration_example()
    elif args.example == "evaluator_optimizer":
        example_workflow.evaluator_optimizer_example()

if __name__ == "__main__":
    main()
