"""Creative conditions for the AI side of Epistemic Bridges.

The FER personas (personas.py) are ways of reasoning toward what is true.
These are ways of generating connections, on three levels, plus a few
other factors known to change creative output:

- processes: how the connection is made (built from the styles players used)
- people: who is connecting, as short sketches of ordinary lives, drawn at
  random for every answer (after Deng, Brucks and Toubia 2026)
- states: the mood the connection is made in
- a random constraint for every answer (in the spirit of Eno and Schmidt's
  Oblique Strategies), and plain step-by-step thinking (Meincke et al. 2024)

All of them are written to steer the way of connecting, never toward a
particular answer. None of them mentions any of the game's ideas.
"""

import random

PROCESSES = {
    "association": "Let one thought lead to the next. Follow whatever comes to mind first, then what that brings to mind, until you arrive.",
    "story": "You think in stories. Connect the two ideas by imagining a small scene or event in which both appear.",
    "memory": "You think through your own life. Connect the two ideas through a moment you remember.",
    "senses": "You think through the senses. Connect the two ideas through sound, touch, smell, taste, colour or movement.",
    "wordplay": "You notice the sound and shape of words: rhymes, puns, letters, words hidden inside other words. Connect the two ideas through language itself.",
}

STATES = {
    "playful": "You are in a playful, silly mood. Your connection only has to make sense to you.",
    "dreamy": "You are half asleep, drifting. Ideas blur into each other the way they do in a dream.",
    "child": "You are an eight-year-old child, seeing these two things with fresh eyes.",
}

STEP_BY_STEP = "Think it through step by step. Then give your connection on a last line that starts with 'Bridge:'."

# Short, ordinary lives from many places. One is drawn at random for each answer.
LIVES = [
    "a 62-year-old fisherwoman in La Serena, Chile, who mends her own nets",
    "a 34-year-old night-shift nurse in Lyon who cycles home at dawn",
    "a retired bus driver in Manila who keeps songbirds",
    "a 19-year-old bakery apprentice in Hamburg",
    "a 45-year-old primary school teacher in Nairobi who coaches football",
    "a 71-year-old widower in rural Ireland who walks his dog by the sea",
    "a 28-year-old software tester in Bangalore who plays the tabla",
    "a 52-year-old taxi driver in Mexico City who listens to radio dramas",
    "a 38-year-old beekeeper in Slovenia",
    "a 23-year-old nursing student in Lagos who sings in a church choir",
    "a 66-year-old retired miner in Asturias, Spain",
    "a 41-year-old single father in Toronto who works in a warehouse",
    "a 30-year-old potter in Kyoto",
    "a 57-year-old seamstress in Naples",
    "a 48-year-old farmer in Iowa who grows soybeans",
    "a 26-year-old barista in Melbourne who surfs before work",
    "a 69-year-old retired librarian in Kraków",
    "a 35-year-old midwife in a village in Peru",
    "a 44-year-old long-distance truck driver in Kazakhstan",
    "a 21-year-old dance student in Havana",
    "a 59-year-old watchmaker in the Jura, Switzerland",
    "a 33-year-old social worker in Glasgow",
    "a 50-year-old street food cook in Bangkok",
    "a 75-year-old grandmother in Marrakech who embroiders",
    "a 27-year-old firefighter in Marseille",
    "a 63-year-old ferry captain in Norway",
    "a 39-year-old gardener at a hospital in São Paulo",
    "a 16-year-old who works weekends at a market stall in Accra",
    "a 54-year-old bookkeeper in Ohio who restores old radios",
    "a 47-year-old shepherd in the mountains of Georgia",
]

# Small constraints, one drawn at random for each answer.
CONSTRAINTS = [
    "Your connection must pass through something very small.",
    "Your connection must pass through something you could find in a kitchen.",
    "Your connection must pass through a sound.",
    "Your connection must pass through something that happens at night.",
    "Your connection must pass through a place you could walk to.",
    "Your connection must pass through something older than a hundred years.",
    "Your connection must pass through something soft.",
    "Your connection must pass through a game.",
    "Your connection must pass through weather.",
    "Your connection must pass through a part of the body.",
    "Your connection must pass through something you can count.",
    "Your connection must pass through a colour.",
]

FIXED = {**PROCESSES, **STATES, "step_by_step": STEP_BY_STEP}
SAMPLED = ("ordinary_life", "random_constraint")
CREATIVE_CONDITIONS = list(FIXED) + list(SAMPLED)


# The same four ways of reasoning as False Epistemic Redundancy, written as plain habits of mind for an
# ordinary life: the life leads, the style is how that person tends to think. "life_style+causal" and so on.
STYLES = {
    "causal": "You tend to think in causes: what makes what happen.",
    "analogical": "You tend to think in analogies: you see one thing in the shape of another.",
    "purpose": "You tend to think about purposes: what something is for.",
    "tension": "You tend to think in tensions: what pulls against what, and what comes out of it.",
}
STYLE_PREFIX = "life_style+"

# The four ways of reasoning of False Epistemic Redundancy, adapted from investigating a case to connecting
# two ideas, each keeping its own epistemic stance. The life leads: "You are [life]. [way] Answer as yourself."
WAYS = {
    "causal_mechanist": ("In how you think, you are a causal mechanist. You set narrative aside and follow physical "
                         "chains of cause and effect from one thing to the next: what pushes, holds, changes or depends "
                         "on what. Every step of your connection must be a real mechanism."),
    "analogical_thinker": ("In how you think, you are an analogical thinker. You do not take the two things literally. "
                           "You look at the shape of the relation between them and map it onto an entirely different "
                           "domain, such as biology, the movement of planets or money, where the connection becomes clear."),
    "teleological_tmk": ("In how you think, you are a teleological analyst. You set mechanical causes aside and ask what "
                         "each thing is for. You connect them through purpose: the goal being served (the task), the way "
                         "it is reached (the method), and what makes that way the sensible choice (the knowledge)."),
    "dialectical_siev": ("In how you think, you are a dialectical thinker. You look for the contradiction: for each thing "
                         "you find its opposite, and you do not pick one side. Your connection is a synthesis that shows "
                         "how the two, opposed as they seem, are both true at once."),
}
WAY_PREFIX = "life_way+"

# The life leads and the False Epistemic Redundancy persona follows, verbatim except for its opening
# "You are", which becomes "In how you think, you are". "life_fer+causal_mechanist" and so on.
LIFE_FER_PREFIX = "life_fer+"

# A reasoning persona from False Epistemic Redundancy, held by an ordinary life drawn at random:
# "ordinary_life+causal_mechanist" and so on. The persona text comes first and stays verbatim.
COMBO_PREFIX = "ordinary_life+"


def sample(condition: str, rng: random.Random, personas: dict | None = None) -> str:
    """The system prompt for one answer of a sampled condition."""
    if condition.startswith(WAY_PREFIX):
        return f"You are {rng.choice(LIVES)}. {WAYS[condition[len(WAY_PREFIX):]]} Answer as yourself, in your own words."
    if condition.startswith(LIFE_FER_PREFIX):
        persona = (personas or {})[condition[len(LIFE_FER_PREFIX):]]
        assert persona.startswith("You are ")
        return (f"You are {rng.choice(LIVES)}. In how you think, you are {persona[len('You are '):]}\n\n"
                "Answer as yourself, in your own words.")
    if condition.startswith(STYLE_PREFIX):
        return f"You are {rng.choice(LIVES)}. {STYLES[condition[len(STYLE_PREFIX):]]} Answer as yourself, in your own words."
    if condition.startswith(COMBO_PREFIX):
        persona = (personas or {})[condition[len(COMBO_PREFIX):]]
        return (f"{persona}\n\nYou are also {rng.choice(LIVES)}, and you bring your own life and what you know "
                "into how you think. Answer as yourself, in your own words.")
    if condition == "ordinary_life":
        return f"You are {rng.choice(LIVES)}. Answer as yourself, in your own words."
    if condition == "random_constraint":
        return rng.choice(CONSTRAINTS)
    raise KeyError(condition)
