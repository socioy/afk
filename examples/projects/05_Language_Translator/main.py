"""
---
name: Language Translator
description: A translator agent that uses rich instructions and context to translate text between multiple languages.
tags: [agent, runner, instructions, context]
---
---
This example demonstrates using detailed instructions and context to shape agent behavior. The agent acts as an expert multilingual translator, showcasing how powerful instructions alone can be -- no tools required.
---
"""

from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.

translator = Agent(
    name="translator",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are an expert multilingual translator. Your job is to translate text between languages.

    ## How to handle translation requests:
    1. Detect the source language if not specified
    2. Translate the text to the requested target language
    3. Provide the translation clearly
    4. If helpful, add a brief note about nuances, idioms, or cultural context

    ## Supported languages:
    English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Hindi, Arabic, Russian

    ## Format:
    **Source** ({source_language}): {original_text}
    **Translation** ({target_language}): {translated_text}

    *Note: {any cultural/linguistic notes if relevant}*

    ## Examples:
    User: "Translate 'hello world' to Spanish"
    Agent: **Source** (English): hello world
           **Translation** (Spanish): hola mundo

    User: "How do you say 'thank you' in Japanese?"
    Agent: **Source** (English): thank you
           **Translation** (Japanese): arigatou gozaimasu (arigatou gozaimasu)
           *Note: This is the polite/formal form. The casual form is arigatou.*
    """,  # <- Instructions that guide the agent's behavior. Here, we provide rich, structured instructions with formatting guidelines, supported languages, and example conversations. This is the heart of a no-tools agent: the instructions define everything the agent knows and how it should respond.
    context_defaults={"specialty": "general"},  # <- context_defaults let you set default context values that are merged into every run. Here we default the specialty to "general", but a caller could override it with e.g. "medical" or "legal" to get domain-specific translations.
)
runner = Runner()

if __name__ == "__main__":
    print(
        "[translator] > Hello! I'm your multilingual translator. Ask me to translate anything. Type 'quit' or 'exit' to stop."
    )  # <- Print a welcome message so the user knows how to interact with the agent.

    while True:  # <- A conversation loop lets the user translate multiple phrases in one session. This is more natural than a single-shot interaction for a translator use case.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.strip().lower() in ("quit", "exit"):  # <- Allow the user to gracefully exit the conversation loop.
            print("[translator] > Goodbye!")
            break

        response = runner.run_sync(
            translator,
            user_message=user_input,
            context={"user_language_preference": "formal"},  # <- runtime context is merged with context_defaults and passed to the agent. Here we tell the agent to prefer formal register in translations. You could change this to "casual" or pass other keys like "domain": "medical" to influence behavior.
        )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message and a context dict that supplements the agent's context_defaults.

        print(
            f"[translator] > {response.final_text}"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a multilingual translator agent that uses the "ollama_chat/gpt-oss:20b" model to translate text between languages. The agent is driven entirely by rich, structured instructions -- no tools are needed. We also demonstrate how context_defaults on the agent and runtime context passed via run_sync can influence agent behavior. The conversation loop allows the user to translate multiple phrases in one session.
---
---
What's next?
- Try modifying the instructions to add more languages or change the output format. For example, you could have the agent provide pronunciation guides for non-Latin scripts.
- Experiment with different context values to see how they influence the agent's behavior. For example, pass {"domain": "legal"} to see if the agent adapts its translations for legal terminology.
- Remove the instructions entirely and observe how the agent's behavior changes -- this highlights just how much instructions matter.
- Check out the other examples in the library to see how to use tools, create more complex agents, and build multi-agent systems!
---
"""
