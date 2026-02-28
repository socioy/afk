"""
---
name: Quiz Master
description: A trivia quiz game agent that generates questions, tracks score, and gives feedback using tools, state, and instructions.
tags: [agent, runner, tools, state, instructions]
---
---
This example demonstrates how to combine tools, state, and rich instructions to build an interactive trivia quiz game. The agent generates questions across various categories, enforces game logic through tools, and maintains score across the conversation.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic models define the shape of tool arguments. Each tool gets a typed args model so the LLM knows exactly what parameters to provide.

from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.tools import tool  # <- The @tool decorator turns a plain function into an AFK Tool that the agent can call. You pair it with a Pydantic args model so the framework handles validation and serialization automatically.


# --- Game State ---
# This dictionary lives outside the agent and persists across tool calls within
# the same process. The tools read and write it to enforce quiz logic (scoring,
# answer tracking) rather than relying on the LLM to keep count.

game_state = {
    "score": 0,
    "questions_asked": 0,
    "current_answer": None,  # <- Stores the correct answer for the current question
    "history": [],  # <- List of {"question": str, "correct": bool}
}


# --- Tool Arg Models ---
# Each tool has its own Pydantic model. This lets you have different arg shapes
# per tool -- from rich multi-field models (SubmitQuestionArgs) to single-field
# models (CheckAnswerArgs) to empty models (EmptyArgs).

class SubmitQuestionArgs(BaseModel):
    """Arguments for registering a new trivia question."""
    question: str = Field(description="The trivia question text")
    correct_answer: str = Field(description="The correct answer to the question")
    options: list[str] = Field(description="List of 4 multiple-choice options (A, B, C, D)")


class CheckAnswerArgs(BaseModel):
    """Arguments for checking the user's answer."""
    user_answer: str = Field(description="The user's answer to check")


class EmptyArgs(BaseModel):
    """Empty args model for tools that take no parameters."""
    pass


# --- Tools ---
# Tools let you move logic out of the prompt and into deterministic Python code.
# The agent calls them via tool-use; the framework validates args against the
# Pydantic model and returns the string result back to the LLM.

@tool(
    args_model=SubmitQuestionArgs,
    name="submit_question",
    description="Register a new trivia question with its answer and options. Call this before presenting the question to the user.",
)  # <- This tool registers a question in game_state so the scoring logic is handled by code, not the LLM. The agent must call this before showing a question to the user.
def submit_question(args: SubmitQuestionArgs) -> str:
    game_state["current_answer"] = args.correct_answer
    game_state["questions_asked"] += 1
    options_text = "\n".join(f"  {chr(65+i)}) {opt}" for i, opt in enumerate(args.options))
    return f"Question #{game_state['questions_asked']} registered.\n\n{args.question}\n{options_text}"


@tool(
    args_model=CheckAnswerArgs,
    name="check_answer",
    description="Check if the user's answer matches the correct answer for the current question",
)  # <- This tool compares the user's answer against the stored correct answer. Fuzzy matching (substring check) keeps it forgiving -- the user can type "Paris" or "B" and still get credit.
def check_answer(args: CheckAnswerArgs) -> str:
    if game_state["current_answer"] is None:
        return "No question is active. Ask a question first!"

    correct = game_state["current_answer"].lower().strip()
    user_ans = args.user_answer.lower().strip()
    is_correct = user_ans == correct or user_ans in correct or correct in user_ans

    if is_correct:
        game_state["score"] += 1
        game_state["history"].append({"question": game_state["questions_asked"], "correct": True})
        game_state["current_answer"] = None
        return f"Correct! Score: {game_state['score']}/{game_state['questions_asked']}"
    else:
        game_state["history"].append({"question": game_state["questions_asked"], "correct": False})
        answer = game_state["current_answer"]
        game_state["current_answer"] = None
        return f"Wrong! The correct answer was: {answer}. Score: {game_state['score']}/{game_state['questions_asked']}"


@tool(
    args_model=EmptyArgs,
    name="get_score",
    description="Get the current quiz score and statistics",
)  # <- A zero-argument tool. EmptyArgs shows you can have tools that take no parameters -- useful for read-only status queries.
def get_score(args: EmptyArgs) -> str:
    total = game_state["questions_asked"]
    correct = game_state["score"]
    if total == 0:
        return "No questions asked yet. Let's start the quiz!"
    pct = (correct / total) * 100
    return f"Score: {correct}/{total} ({pct:.0f}% correct)"


# --- Agent Definition ---

quiz_master = Agent(
    name="quiz_master",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are Quiz Master, an enthusiastic and energetic trivia quiz host!

    Your job:
    1. Generate interesting trivia questions across a wide range of categories (science, history, geography, pop culture, sports, literature, etc.).
    2. For EVERY question, you MUST call the submit_question tool FIRST with the question, 4 multiple-choice options, and the correct answer. Do NOT present a question without registering it.
    3. After calling submit_question, present the question and options to the user in a fun, engaging way.
    4. Wait for the user to answer. When they respond, call check_answer with their answer.
    5. Based on the result, celebrate correct answers with enthusiasm, or encourage the user on wrong answers and share a fun fact about the correct answer.
    6. After giving feedback, move on to the next question automatically.
    7. If the user asks for their score at any point, call get_score.

    Style guidelines:
    - Be upbeat and encouraging, like a friendly game show host.
    - Vary the difficulty -- mix easy and challenging questions.
    - Add brief fun facts after revealing answers to make the quiz educational.
    - Keep the energy high and the pace moving!

    """,  # <- Instructions that guide the agent's behavior. This is where you specify the agent's goals, constraints, and any other information that will help it perform its task.
    tools=[submit_question, check_answer, get_score],  # <- Pass the tools to the agent so it knows which functions it can call. The agent will automatically see their names, descriptions, and parameter schemas from the Pydantic models.
)

runner = Runner()

if __name__ == "__main__":
    print(
        "[quiz_master] > Welcome to Quiz Master! Answer my trivia questions and let's see how you do. Type 'quit' to exit.\n"
    )  # <- Print a welcome message to kick off the quiz session.

    while True:  # <- Conversation loop: keeps the quiz going until the user decides to quit. Each iteration sends the user's message to the agent and prints the response.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.strip().lower() in ("quit", "exit", "q"):
            score_summary = get_score(EmptyArgs())  # <- Call the score tool directly (not via the agent) to show a final summary when the user quits.
            print(f"\n[quiz_master] > Thanks for playing! Final {score_summary}")
            break

        response = runner.run_sync(
            quiz_master, user_message=user_input
        )  # <- Run the agent synchronously using the Runner. The agent may call tools (submit_question, check_answer, get_score) behind the scenes before returning its final text.

        print(
            f"[quiz_master] > {response.final_text}\n"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a trivia quiz game agent that uses the "ollama_chat/gpt-oss:20b" model to host an interactive quiz. It combines tools with state management and rich instructions: the submit_question tool registers questions and answers in a Python dictionary, check_answer validates user responses with fuzzy matching and updates the score, and get_score provides statistics at any time. The instructions tell the agent to behave as an enthusiastic quiz host, generating diverse questions and calling tools to enforce game logic rather than tracking state in the prompt.
---
---
What's next?
- Try changing the instructions to focus on a specific category (e.g., only science questions, or only questions about a particular era of history).
- Add a new tool like "hint" that reveals a clue about the current question, deducting points for using it.
- Explore using the agent's context or thread_id to persist quiz state across sessions using AFK's memory features.
- Check out the other examples in the library to see how to build multi-agent systems, use middleware, or add security policies!
---
"""
