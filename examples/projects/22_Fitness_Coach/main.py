"""
---
name: Fitness Coach
description: A fitness coaching agent that uses tool prehooks and posthooks for input validation and output formatting.
tags: [agent, runner, tools, prehook, posthook, validation]
---
---
This example demonstrates how to use @prehook and @posthook decorators to add cross-cutting logic
around tool execution. Prehooks run BEFORE a tool executes — use them to validate, sanitize, or
transform input arguments. Posthooks run AFTER a tool returns — use them to format, enrich, or
log output. This separation keeps your core tool logic clean while adding layers of processing
around it. The fitness coach agent calculates workout plans and nutrition info, with hooks that
validate ranges and format results consistently.
---
"""

import json  # <- For formatting structured output.
from pydantic import BaseModel, Field  # <- Pydantic for typed argument schemas shared by tools, prehooks, and posthooks.
from afk.core import Runner  # <- Runner executes agents and manages their state.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, prehook, posthook  # <- @tool for tools, @prehook for pre-execution hooks, @posthook for post-execution hooks.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class CalorieArgs(BaseModel):  # <- Schema for calorie calculation. Used by the tool AND its prehook (they share the same args_model).
    weight_kg: float = Field(description="Body weight in kilograms")
    height_cm: float = Field(description="Height in centimeters")
    age: int = Field(description="Age in years")
    activity_level: str = Field(description="Activity level: sedentary, light, moderate, active, very_active")
    goal: str = Field(description="Fitness goal: lose, maintain, or gain")


class WorkoutArgs(BaseModel):  # <- Schema for workout plan generation.
    fitness_level: str = Field(description="Current fitness level: beginner, intermediate, advanced")
    goal: str = Field(description="Workout goal: strength, cardio, flexibility, or general")
    duration_minutes: int = Field(description="Available time for workout in minutes")


class BMIArgs(BaseModel):  # <- Schema for BMI calculation.
    weight_kg: float = Field(description="Body weight in kilograms")
    height_cm: float = Field(description="Height in centimeters")


# ===========================================================================
# Prehooks — run BEFORE the tool executes
# ===========================================================================

@prehook(args_model=CalorieArgs, name="validate_calorie_inputs", description="Validate that calorie calculation inputs are within reasonable ranges")
def validate_calorie_inputs(args: CalorieArgs) -> CalorieArgs:  # <- A prehook receives the same args as the tool. It can validate, transform, or reject them. Returning the args (modified or not) passes them to the tool. Raising an exception stops execution.
    if args.weight_kg < 20 or args.weight_kg > 300:
        raise ValueError(f"Weight {args.weight_kg}kg is outside reasonable range (20-300kg)")  # <- Raising an exception in a prehook prevents the tool from executing. The error message is returned to the agent.
    if args.height_cm < 50 or args.height_cm > 275:
        raise ValueError(f"Height {args.height_cm}cm is outside reasonable range (50-275cm)")
    if args.age < 5 or args.age > 120:
        raise ValueError(f"Age {args.age} is outside reasonable range (5-120)")
    if args.activity_level not in ("sedentary", "light", "moderate", "active", "very_active"):
        raise ValueError(f"Unknown activity level '{args.activity_level}'. Use: sedentary, light, moderate, active, very_active")
    if args.goal not in ("lose", "maintain", "gain"):
        raise ValueError(f"Unknown goal '{args.goal}'. Use: lose, maintain, gain")
    return args  # <- Return the validated args unchanged. You could also MODIFY args here (e.g., normalize units, fill defaults).


@prehook(args_model=WorkoutArgs, name="validate_workout_inputs", description="Validate workout plan inputs")
def validate_workout_inputs(args: WorkoutArgs) -> WorkoutArgs:
    if args.fitness_level not in ("beginner", "intermediate", "advanced"):
        raise ValueError(f"Unknown fitness level '{args.fitness_level}'. Use: beginner, intermediate, advanced")
    if args.goal not in ("strength", "cardio", "flexibility", "general"):
        raise ValueError(f"Unknown goal '{args.goal}'. Use: strength, cardio, flexibility, general")
    if args.duration_minutes < 10:
        raise ValueError("Workout duration must be at least 10 minutes")
    if args.duration_minutes > 180:
        raise ValueError("Workout duration capped at 180 minutes for safety")
    return args


@prehook(args_model=BMIArgs, name="validate_bmi_inputs", description="Validate BMI inputs")
def validate_bmi_inputs(args: BMIArgs) -> BMIArgs:
    if args.weight_kg <= 0 or args.height_cm <= 0:
        raise ValueError("Weight and height must be positive numbers")
    return args


# ===========================================================================
# Posthooks — run AFTER the tool executes
# ===========================================================================

@posthook(args_model=CalorieArgs, name="format_calorie_output", description="Add disclaimer and formatting to calorie results")
def format_calorie_output(args: CalorieArgs) -> str:  # <- A posthook receives the same args as the tool (not the tool's output). It returns a string that is APPENDED to the tool's output. Use this for consistent formatting, disclaimers, or enrichment.
    return (
        "\n\n---\n"
        "Disclaimer: These are estimates based on the Mifflin-St Jeor equation. "
        "Consult a healthcare professional for personalized nutrition advice.\n"
        f"Calculated for: {args.weight_kg}kg, {args.height_cm}cm, age {args.age}, "
        f"activity: {args.activity_level}, goal: {args.goal}"
    )  # <- The posthook's return value is appended to the tool's return value. This adds a consistent disclaimer to every calorie calculation.


@posthook(args_model=WorkoutArgs, name="format_workout_output", description="Add safety tips to workout plans")
def format_workout_output(args: WorkoutArgs) -> str:
    tips = {
        "beginner": "Start slow and focus on form over speed. Rest if you feel dizzy.",
        "intermediate": "Push yourself but listen to your body. Stay hydrated.",
        "advanced": "Challenge yourself but maintain proper form to avoid injury.",
    }
    return (
        f"\n\nSafety tip ({args.fitness_level}): {tips.get(args.fitness_level, 'Stay safe!')}\n"
        f"Plan tailored for: {args.duration_minutes} minute {args.goal} session"
    )


# ===========================================================================
# Tool definitions — with prehooks and posthooks attached
# ===========================================================================

@tool(
    args_model=CalorieArgs,
    name="calculate_calories",
    description="Calculate daily calorie needs based on body stats and goals",
    prehooks=[validate_calorie_inputs],  # <- Attach the prehook. It runs BEFORE this tool executes. Multiple prehooks run in order.
    posthooks=[format_calorie_output],  # <- Attach the posthook. It runs AFTER this tool returns. Its output is appended to the tool's result.
)
def calculate_calories(args: CalorieArgs) -> str:  # <- The core tool logic is clean — no validation or formatting clutter. Prehooks handle validation, posthooks handle formatting.
    # --- Mifflin-St Jeor equation for BMR ---
    bmr = 10 * args.weight_kg + 6.25 * args.height_cm - 5 * args.age + 5  # <- Simplified male formula. A real app would ask for gender.

    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    tdee = bmr * activity_multipliers[args.activity_level]  # <- TDEE = Total Daily Energy Expenditure.

    goal_adjustments = {"lose": -500, "maintain": 0, "gain": 300}
    target = tdee + goal_adjustments[args.goal]

    return (
        f"Your estimated daily calorie needs:\n"
        f"  BMR (Basal Metabolic Rate): {bmr:.0f} cal\n"
        f"  TDEE (with activity): {tdee:.0f} cal\n"
        f"  Target ({args.goal} weight): {target:.0f} cal/day\n\n"
        f"Macronutrient suggestion:\n"
        f"  Protein: {target * 0.30 / 4:.0f}g ({target * 0.30:.0f} cal)\n"
        f"  Carbs: {target * 0.40 / 4:.0f}g ({target * 0.40:.0f} cal)\n"
        f"  Fat: {target * 0.30 / 9:.0f}g ({target * 0.30:.0f} cal)"
    )


@tool(
    args_model=WorkoutArgs,
    name="generate_workout",
    description="Generate a workout plan based on fitness level, goal, and available time",
    prehooks=[validate_workout_inputs],
    posthooks=[format_workout_output],
)
def generate_workout(args: WorkoutArgs) -> str:
    # --- Workout exercise databases by goal ---
    exercises = {
        "strength": {
            "beginner": ["Bodyweight Squats", "Push-ups (knee)", "Dumbbell Rows", "Plank"],
            "intermediate": ["Barbell Squats", "Bench Press", "Deadlifts", "Pull-ups"],
            "advanced": ["Front Squats", "Overhead Press", "Romanian Deadlifts", "Muscle-ups"],
        },
        "cardio": {
            "beginner": ["Brisk Walking", "Jumping Jacks", "Step-ups", "March in Place"],
            "intermediate": ["Running Intervals", "Jump Rope", "Burpees", "Mountain Climbers"],
            "advanced": ["Sprint Intervals", "Box Jumps", "Battle Ropes", "Rowing Sprints"],
        },
        "flexibility": {
            "beginner": ["Cat-Cow Stretch", "Forward Fold", "Child's Pose", "Thread the Needle"],
            "intermediate": ["Pigeon Pose", "Lizard Pose", "Seated Twist", "Bridge"],
            "advanced": ["Full Splits", "Wheel Pose", "King Pigeon", "Firefly Pose"],
        },
        "general": {
            "beginner": ["Walking", "Bodyweight Squats", "Push-ups (knee)", "Stretching"],
            "intermediate": ["Jogging", "Dumbbell Lunges", "Push-ups", "Plank Variations"],
            "advanced": ["Running", "Weighted Squats", "Plyometric Push-ups", "L-sits"],
        },
    }

    selected = exercises.get(args.goal, exercises["general"]).get(args.fitness_level, exercises["general"]["beginner"])
    sets = 3 if args.fitness_level != "beginner" else 2
    time_per_exercise = args.duration_minutes // len(selected)

    lines = [f"Workout Plan: {args.goal.title()} ({args.fitness_level})", f"Total time: {args.duration_minutes} minutes", ""]
    for i, ex in enumerate(selected, 1):
        lines.append(f"  {i}. {ex} — {sets} sets x {time_per_exercise} min")
    lines.append(f"\nWarm-up: 5 min light movement")
    lines.append(f"Cool-down: 5 min stretching")

    return "\n".join(lines)


@tool(
    args_model=BMIArgs,
    name="calculate_bmi",
    description="Calculate Body Mass Index from weight and height",
    prehooks=[validate_bmi_inputs],
)
def calculate_bmi(args: BMIArgs) -> str:
    height_m = args.height_cm / 100
    bmi = args.weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return f"BMI: {bmi:.1f} — Category: {category}\n(Note: BMI is a rough guide and doesn't account for muscle mass, bone density, or body composition.)"


# ===========================================================================
# Agent and runner setup
# ===========================================================================

fitness_agent = Agent(
    name="fitness-coach",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a friendly fitness coach. You help users with:
    - Calculating daily calorie needs based on their stats and goals
    - Generating personalized workout plans
    - Calculating BMI
    - General fitness advice and motivation

    When calculating calories, ask for: weight (kg), height (cm), age, activity level, and goal.
    When generating workouts, ask for: fitness level, goal type, and available time.

    Be encouraging and supportive. Remind users that consistency beats perfection!

    **NOTE**: Always recommend consulting a healthcare professional for medical advice.
    """,
    tools=[calculate_calories, generate_workout, calculate_bmi],
)

runner = Runner()

if __name__ == "__main__":
    print("Fitness Coach Agent (type 'quit' to exit)")
    print("=" * 45)
    print("Try: 'Calculate my calories', 'Make me a workout plan', 'What's my BMI?'\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Stay active! Goodbye!")
            break

        response = runner.run_sync(fitness_agent, user_message=user_input)
        print(f"[fitness-coach] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a fitness coaching agent with tools that use @prehook for input validation
(range checking, enum validation) and @posthook for output enrichment (disclaimers, safety tips). Prehooks
run before the tool and can reject, validate, or transform arguments. Posthooks run after and append
formatted content to the result. This pattern keeps core tool logic clean while adding cross-cutting
concerns as composable layers.
---
---
What's next?
- Try sending invalid inputs (e.g., weight of -10 or age of 500) to see the prehook validation in action.
- Add multiple prehooks to a single tool to see them chain (they run in order).
- Create a posthook that logs every tool call to a file for analytics.
- Experiment with prehooks that MODIFY arguments (e.g., converting pounds to kg automatically).
- Combine prehooks/posthooks with @middleware for even more powerful cross-cutting patterns.
- Check out the Middleware example for registry-level middleware that applies to ALL tools!
---
"""
