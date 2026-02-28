"""
---
name: Math Tutor
description: A math tutoring agent that uses reasoning_enabled and reasoning_effort for extended thinking when solving problems step by step.
tags: [agent, runner, tools, reasoning, extended-thinking, chain-of-thought]
---
---
This example demonstrates AFK's reasoning configuration — reasoning_enabled, reasoning_effort,
and reasoning_max_tokens — for enabling extended thinking (chain of thought) in agents. When
reasoning_enabled=True, the agent is instructed to "think through" problems before responding,
showing its step-by-step reasoning process. reasoning_effort controls how much thinking the
model does ("low" for quick answers, "medium" for balanced, "high" for deep multi-step reasoning).
reasoning_max_tokens caps the token budget for the thinking phase, preventing runaway reasoning
on simple questions. This is ideal for educational agents, problem-solving assistants, and any
scenario where showing work matters as much as the final answer. The math tutor agent solves
problems, provides progressive hints, validates student answers, and generates practice problems
— all while showing its extended thinking process.
---
"""

import random  # <- For generating random practice problems.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner orchestrates agent execution and applies reasoning configuration.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools — including reasoning settings.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Problem bank — categorized math problems with solutions and hints
# ===========================================================================

PROBLEM_BANK: dict[str, list[dict]] = {  # <- Organized by difficulty level. Each problem has a question, answer, step-by-step solution, and progressive hints.
    "algebra": [
        {
            "question": "Solve for x: 3x + 7 = 22",
            "answer": "5",
            "solution": [
                "Start with: 3x + 7 = 22",
                "Subtract 7 from both sides: 3x = 15",
                "Divide both sides by 3: x = 5",
            ],
            "hints": [
                "Try isolating the variable by moving constants to the other side.",
                "Subtract 7 from both sides first.",
                "After subtracting, you get 3x = 15. Now divide by 3.",
            ],
        },
        {
            "question": "Solve for x: 2(x - 3) = 10",
            "answer": "8",
            "solution": [
                "Start with: 2(x - 3) = 10",
                "Distribute the 2: 2x - 6 = 10",
                "Add 6 to both sides: 2x = 16",
                "Divide both sides by 2: x = 8",
            ],
            "hints": [
                "First, distribute the 2 into the parentheses.",
                "After distributing, you get 2x - 6 = 10.",
                "Add 6 to both sides, then divide by 2.",
            ],
        },
        {
            "question": "Solve for x: (x/4) + 5 = 12",
            "answer": "28",
            "solution": [
                "Start with: x/4 + 5 = 12",
                "Subtract 5 from both sides: x/4 = 7",
                "Multiply both sides by 4: x = 28",
            ],
            "hints": [
                "Isolate the fraction by removing the constant first.",
                "Subtract 5 from both sides to get x/4 = 7.",
                "Multiply both sides by 4 to solve for x.",
            ],
        },
    ],
    "geometry": [
        {
            "question": "Find the area of a triangle with base 12 cm and height 8 cm.",
            "answer": "48",
            "solution": [
                "Formula: Area = (1/2) * base * height",
                "Substitute: Area = (1/2) * 12 * 8",
                "Calculate: Area = (1/2) * 96 = 48 cm^2",
            ],
            "hints": [
                "The area formula for a triangle involves base and height.",
                "Area = (1/2) * base * height.",
                "Multiply 12 * 8, then divide by 2.",
            ],
        },
        {
            "question": "Find the circumference of a circle with radius 7 cm. (Use pi = 3.14)",
            "answer": "43.96",
            "solution": [
                "Formula: Circumference = 2 * pi * radius",
                "Substitute: C = 2 * 3.14 * 7",
                "Calculate: C = 6.28 * 7 = 43.96 cm",
            ],
            "hints": [
                "Circumference uses the formula involving 2, pi, and the radius.",
                "C = 2 * pi * r = 2 * 3.14 * 7.",
                "Multiply step by step: 2 * 3.14 = 6.28, then 6.28 * 7.",
            ],
        },
        {
            "question": "Find the hypotenuse of a right triangle with sides 3 and 4.",
            "answer": "5",
            "solution": [
                "Use the Pythagorean theorem: a^2 + b^2 = c^2",
                "Substitute: 3^2 + 4^2 = c^2",
                "Calculate: 9 + 16 = 25",
                "Take square root: c = sqrt(25) = 5",
            ],
            "hints": [
                "This is a classic Pythagorean theorem problem.",
                "Square both sides: 3^2 = 9, 4^2 = 16.",
                "Add the squares and take the square root: sqrt(9 + 16) = sqrt(25).",
            ],
        },
    ],
    "arithmetic": [
        {
            "question": "What is 15% of 240?",
            "answer": "36",
            "solution": [
                "Convert percentage to decimal: 15% = 0.15",
                "Multiply: 0.15 * 240 = 36",
            ],
            "hints": [
                "Convert the percentage to a decimal first by dividing by 100.",
                "15% = 0.15. Now multiply 0.15 by 240.",
            ],
        },
        {
            "question": "What is the result of 3^4 (3 to the power of 4)?",
            "answer": "81",
            "solution": [
                "3^4 = 3 * 3 * 3 * 3",
                "Step 1: 3 * 3 = 9",
                "Step 2: 9 * 3 = 27",
                "Step 3: 27 * 3 = 81",
            ],
            "hints": [
                "Exponentiation means multiplying the base by itself repeatedly.",
                "3^4 = 3 * 3 * 3 * 3. Start with 3 * 3 = 9.",
                "Continue: 9 * 3 = 27, then 27 * 3 = ?",
            ],
        },
        {
            "question": "Simplify the fraction 48/64.",
            "answer": "3/4",
            "solution": [
                "Find the GCD of 48 and 64: GCD = 16",
                "Divide numerator by GCD: 48 / 16 = 3",
                "Divide denominator by GCD: 64 / 16 = 4",
                "Simplified: 3/4",
            ],
            "hints": [
                "Find the greatest common divisor (GCD) of 48 and 64.",
                "Both 48 and 64 are divisible by 16.",
                "Divide both top and bottom by 16: 48/16 = 3, 64/16 = 4.",
            ],
        },
    ],
}

# --- Track current problem and hint progress ---
current_problem: dict = {}  # <- Tracks the currently active problem. Shared between tools.
hint_index: int = 0  # <- Tracks which hint to give next for progressive hinting.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class CheckAnswerArgs(BaseModel):  # <- Schema for the answer-checking tool.
    answer: str = Field(description="The student's answer to the current problem")


class GetHintArgs(BaseModel):  # <- Schema for the hint tool. No parameters needed — it reads from current_problem state.
    pass


class GenerateProblemArgs(BaseModel):  # <- Schema for generating a practice problem.
    category: str = Field(description="Math category: algebra, geometry, or arithmetic")


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(  # <- Answer validation tool. Compares student's answer to the stored correct answer and shows the step-by-step solution.
    args_model=CheckAnswerArgs,
    name="check_answer",
    description="Check the student's answer against the correct answer for the current problem. Shows the step-by-step solution after checking.",
)
def check_answer(args: CheckAnswerArgs) -> str:
    global current_problem  # <- Access the shared problem state.

    if not current_problem:
        return "No active problem! Use generate_problem first to get a question."

    student_answer = args.answer.strip().lower()  # <- Normalize for comparison.
    correct_answer = str(current_problem["answer"]).strip().lower()

    # --- Check for exact or approximate match ---
    is_correct = student_answer == correct_answer  # <- Exact match check.

    # --- Also check for numeric equivalence (e.g., "5.0" == "5") ---
    if not is_correct:
        try:
            if abs(float(student_answer) - float(correct_answer)) < 0.01:  # <- Fuzzy numeric comparison with tolerance.
                is_correct = True
        except ValueError:
            pass  # <- Not a number, stick with string comparison.

    # --- Build the response with solution walkthrough ---
    solution_steps = "\n".join(
        f"  Step {i}: {step}" for i, step in enumerate(current_problem["solution"], 1)
    )

    if is_correct:
        response = (
            f"Correct! Great job!\n\n"
            f"The answer is: {current_problem['answer']}\n\n"
            f"Solution walkthrough:\n{solution_steps}\n\n"
            f"Ready for another problem? Ask me to generate one!"
        )
    else:
        response = (
            f"Not quite. Your answer: {args.answer}\n"
            f"Correct answer: {current_problem['answer']}\n\n"
            f"Here's how to solve it:\n{solution_steps}\n\n"
            f"Don't worry — mistakes are how we learn! Want to try another problem?"
        )

    return response


@tool(  # <- Progressive hint tool. Each call reveals the next hint for the current problem. Hints get more specific as the student asks for more.
    args_model=GetHintArgs,
    name="get_hint",
    description="Get a hint for the current problem. Hints are progressive — each call reveals a more specific hint. Use when the student is stuck.",
)
def get_hint(args: GetHintArgs) -> str:
    global hint_index  # <- Track which hint to show next.

    if not current_problem:
        return "No active problem! Use generate_problem first to get a question."

    hints = current_problem.get("hints", [])

    if not hints:
        return "No hints available for this problem. Try working through it step by step!"

    if hint_index >= len(hints):
        return (
            f"You've used all {len(hints)} hints for this problem!\n"
            f"Last hint: {hints[-1]}\n"
            f"Try your best guess, or ask me to check your answer."
        )

    hint = hints[hint_index]
    hint_number = hint_index + 1
    total_hints = len(hints)
    hint_index += 1  # <- Advance to the next hint for the next call.

    remaining = total_hints - hint_index
    return (
        f"Hint {hint_number}/{total_hints}: {hint}\n"
        f"{'(' + str(remaining) + ' more hint(s) available)' if remaining > 0 else '(Last hint — give it your best shot!)'}"
    )


@tool(  # <- Problem generator. Picks a random problem from the chosen category and sets it as the current problem.
    args_model=GenerateProblemArgs,
    name="generate_problem",
    description="Generate a practice math problem from a specific category. Categories: algebra, geometry, arithmetic.",
)
def generate_problem(args: GenerateProblemArgs) -> str:
    global current_problem, hint_index  # <- Reset problem state for the new question.

    category = args.category.lower()
    if category not in PROBLEM_BANK:
        available = ", ".join(PROBLEM_BANK.keys())
        return f"Unknown category '{args.category}'. Available categories: {available}"

    problems = PROBLEM_BANK[category]
    problem = random.choice(problems)  # <- Pick a random problem from the category.

    current_problem = problem  # <- Store as the active problem.
    hint_index = 0  # <- Reset hint progress for the new problem.

    return (
        f"Here's your {category} problem:\n\n"
        f"  {problem['question']}\n\n"
        f"Take your time! You can:\n"
        f"  - Type your answer and I'll check it\n"
        f"  - Ask for a hint if you're stuck ({len(problem['hints'])} hints available)\n"
        f"  - Ask me to explain the solution when you're ready"
    )


# ===========================================================================
# Agent setup — with reasoning configuration
# ===========================================================================

math_tutor = Agent(
    name="math-tutor",  # <- The agent's display name.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model the agent will use.
    instructions="""
    You are an encouraging, patient math tutor. Your goal is to help students understand
    math concepts, not just get the right answer.

    Your approach:
    1. When a student asks a math question, THINK THROUGH IT step by step using your
       extended reasoning capabilities. Show your work!
    2. Use generate_problem to create practice problems for students.
    3. When a student submits an answer, use check_answer to verify it and show the solution.
    4. When a student is stuck, use get_hint for progressive hints (don't give the answer away!).
    5. After every problem, encourage the student and suggest the next step.

    Teaching principles:
    - Break complex problems into smaller, manageable steps
    - Use analogies and real-world examples when explaining concepts
    - Celebrate correct answers enthusiastically
    - When answers are wrong, be gentle — focus on the learning opportunity
    - Encourage struggle as a sign of growth, not failure

    You have extended thinking enabled, which means you can reason through problems deeply
    before responding. Use this to:
    - Work through multi-step problems methodically
    - Consider multiple approaches to a problem
    - Identify common misconceptions in student answers
    - Craft clear, pedagogically sound explanations
    """,  # <- Instructions emphasize step-by-step reasoning, matching the reasoning_enabled configuration.
    tools=[check_answer, get_hint, generate_problem],  # <- Three tools: check answers, give hints, generate problems.
    reasoning_enabled=True,  # <- ENABLE extended thinking. When True, the runner instructs the model to "think step by step" before generating its response. The thinking phase is separate from the final response — the model first reasons internally, then produces a user-facing answer.
    reasoning_effort="high",  # <- Set reasoning effort to "high". Options: "low" (quick, minimal thinking), "medium" (balanced), "high" (deep, thorough multi-step reasoning). For a math tutor, "high" ensures the model works through problems carefully.
    reasoning_max_tokens=2000,  # <- Cap the reasoning phase at 2000 tokens. This prevents the model from spending too many tokens on its internal thinking for simple questions. For complex proofs, you might increase this. For quick arithmetic, you could lower it.
)

runner = Runner()  # <- A single Runner handles all executions.


# ===========================================================================
# Main entry point — interactive tutoring conversation
# ===========================================================================

if __name__ == "__main__":
    print("Math Tutor Agent (type 'quit' to exit)")
    print("=" * 50)
    print("I can help you with algebra, geometry, and arithmetic!")
    print("\nTry:")
    print("  - 'Give me an algebra problem'")
    print("  - 'I need help with geometry'")
    print("  - 'What is the area of a circle with radius 5?'")
    print("  - Type an answer after getting a problem")
    print("  - 'Give me a hint' if you're stuck\n")

    while True:  # <- Conversation loop for the tutoring interaction.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Keep practicing! Math gets easier the more you do it. Goodbye!")
            break

        response = runner.run_sync(  # <- Synchronous run. The runner applies reasoning_enabled, reasoning_effort, and reasoning_max_tokens from the agent's configuration before sending the request to the LLM. The model thinks through the problem internally, then produces a response.
            math_tutor,
            user_message=user_input,
        )

        print(f"[math-tutor] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a math tutoring agent that uses reasoning_enabled=True,
reasoning_effort="high", and reasoning_max_tokens=2000 for extended thinking (chain of thought).
When reasoning is enabled, the runner instructs the model to think through problems step by step
before responding — the thinking phase is separate from the final user-facing answer. "high"
effort makes the model reason deeply on multi-step problems, while reasoning_max_tokens caps the
thinking budget. The agent has three tools: generate_problem (creates practice questions from a
categorized bank), check_answer (validates student answers and shows solutions), and get_hint
(provides progressive hints that get more specific with each call). This pattern is ideal for
educational agents, problem-solving assistants, and any scenario where step-by-step reasoning
and showing work matters.
---
---
What's next?
- Try changing reasoning_effort to "low" and compare the agent's problem-solving quality — "low" produces quicker but less thorough explanations.
- Increase reasoning_max_tokens to 4000 for more complex problems like multi-step algebraic proofs.
- Add a new category to PROBLEM_BANK (e.g., "calculus" or "statistics") with your own problems, solutions, and hints.
- Experiment with reasoning_enabled=False to see the difference — the agent will still work, but won't show its step-by-step thinking process.
- Add a "difficulty" parameter to generate_problem so students can choose easy, medium, or hard problems within each category.
- Combine reasoning with memory (see the Flashcard Tutor example) to track student progress across sessions and prioritize weak areas!
---
"""
