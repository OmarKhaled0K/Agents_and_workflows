# Project Title

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project explores the development of agentic systems using Large Language Models (LLMs), focusing on both **workflows** and **agents**. The implementation is guided by insights from Anthropic's research on building effective agents.

## Table of Contents

- [Introduction](#introduction)
- [Workflows vs. Agents](#workflows-vs-agents)
  - [Comparison: Workflow vs. Agent](#comparison-workflow-vs-agent)
- [Implemented Workflow Patterns](#implemented-workflow-patterns)
  - [Prompt Chaining](#prompt-chaining)
  - [Routing](#routing)
  - [Parallelization](#parallelization)
  - [Orchestrator-Workers](#orchestrator-workers)
  - [Evaluator-Optimizer](#evaluator-optimizer)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Agentic systems enhance the capabilities of LLMs by enabling them to perform complex tasks through structured processes. This project demonstrates various workflow patterns that orchestrate LLMs and tools to achieve specific objectives.

## Workflows vs. Agents

- **Workflows**: Systems where LLMs and tools are orchestrated through predefined code paths, offering predictability and consistency for well-defined tasks.
- **Agents**: Systems where LLMs dynamically direct their own processes and tool usage, suitable for tasks requiring flexibility and model-driven decision-making at scale.

For a detailed comparison, refer to Anthropic's research on [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents).

### Comparison: Workflow vs. Agent

| **Aspect**              | **Workflow**                                                                                   | **Agent**                                                                                   |
|-------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **Definition**          | Systems where LLMs and tools are orchestrated through predefined code paths.                  | Systems where LLMs dynamically direct their own processes and tool usage.                  |
| **Flexibility**         | Limited to predefined sequences; less adaptable to unexpected tasks.                          | Highly flexible; can adapt to a wide range of tasks without predefined paths.              |
| **Predictability**      | High; follows set patterns, leading to consistent outputs.                                    | Variable; outcomes can differ based on the agent's autonomous decisions.                   |
| **Complexity**          | Generally simpler; easier to implement and debug.                                             | More complex; requires extensive testing and validation.                                   |
| **Use Cases**           | Suitable for well-defined tasks with clear procedures.                                        | Ideal for open-ended problems requiring dynamic decision-making.                           |
| **Control**             | Developers maintain control over each step.                                                   | LLMs have autonomy over task execution, reducing direct developer control.                 |
| **Error Propagation**   | Lower; errors are easier to trace and fix due to predefined paths.                            | Higher; autonomous decisions can compound errors over multiple steps.                      |
| **Development Approach**| Often implemented using direct LLM API calls or simple frameworks.                            | May utilize advanced frameworks but can benefit from reduced abstraction layers.           |

## Implemented Workflow Patterns

This project includes implementations for the following workflow patterns:

### Prompt Chaining

Decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one.

- **Use Cases**: Multi-step tasks like document generation followed by translation.
- **Advantages**: Simplifies complex tasks into manageable steps; enhances clarity.
- **Considerations**: Requires careful design to ensure coherence between steps.

### Routing

An initial LLM call decides which model or process should be used next, directing tasks based on complexity or type.

- **Use Cases**: Directing simple tasks to basic models and complex tasks to advanced models.
- **Advantages**: Optimizes resource utilization; ensures tasks are handled by appropriate models.
- **Considerations**: Incorrect routing can lead to suboptimal performance; needs accurate initial assessment.

### Parallelization

Breaks a task into sub-tasks processed simultaneously, either by dividing data or using multiple models, and then combines the results.

- **Use Cases**: Processing multiple document pages concurrently; ensemble methods for improved accuracy.
- **Advantages**: Reduces processing time; can improve accuracy through ensemble approaches.
- **Considerations**: Requires mechanisms to handle and integrate parallel outputs; potential for increased complexity.

### Orchestrator-Workers

An orchestrator triggers multiple LLM calls (workers) that are then synthesized together, managing complex tasks by delegating components to specialized models.

- **Use Cases**: Aggregating information from various sources; complex data analysis tasks.
- **Advantages**: Enables handling of complex tasks by dividing them; promotes modularity.
- **Considerations**: Coordination overhead; requires effective synthesis of diverse outputs.

### Evaluator-Optimizer

One model generates outputs while another evaluates and optimizes them in a loop until a satisfactory result is achieved, enhancing quality through iterative refinement.

- **Use Cases**: Code generation with iterative debugging; content creation requiring quality assurance.
- **Advantages**: Improves output quality through iterative refinement; introduces feedback loops.
- **Considerations**: Can be resource-intensive; risk of infinite loops without proper stopping criteria.

## Getting Started

To explore these workflow patterns:

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/OmarKhaled0K/Agents_and_workflows.git)
   cd Agents_and_workflows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the examples**:
   ```bash
   python examples/example_workflow.py
   ```

For detailed usage instructions, refer to the [Usage](#usage) section.

## Usage

Each workflow pattern is implemented in the `workflows` directory. You can integrate these patterns into your projects by importing the relevant modules:

```python
from workflows.prompt_chaining import PromptChaining

# Initialize and use the PromptChaining workflow
workflow = PromptChaining()
result = workflow.execute(input_data)
```

## References

- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to suggest improvements or add new features.

## License

This project is licensed under the [MIT License](LICENSE).
