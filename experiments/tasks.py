"""Shared task definitions for experiments.

Both the scaffold and baseline experiments import from here
to ensure they use identical inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TaskStep:
    input_text: str
    expected_answer: str | None = None  # None for statements, string for questions


def build_toy_task(verbose: bool = False) -> list[TaskStep]:
    """Build the entity tracking task.

    If verbose=True, facts are embedded in longer paragraphs that add
    context but no new information. This inflates the conversation
    token count without changing the core facts — testing whether the
    scaffold's compact graph storage outperforms the baseline's full
    conversation history under token pressure.
    """
    steps: list[TaskStep] = []

    if verbose:
        # --- Phase 1: Establish facts (verbose) ---
        steps.append(TaskStep(
            "I just finished counting the inventory in the north warehouse this morning. "
            "After going through every crate and checking each label carefully, I can confirm "
            "that Alice has 5 apples stored in the refrigerated section near the loading dock. "
            "The apples are all Fuji variety, medium sized, and in good condition."
        ))
        steps.append(TaskStep(
            "Moving on to the citrus area, I checked Bob's allocation as well. "
            "According to the latest manifest that was updated by the logistics team yesterday, "
            "Bob has 3 oranges in his designated storage bin. They are navel oranges "
            "sourced from the Valencia supplier we've been working with since last quarter."
        ))
        steps.append(TaskStep(
            "The tropical section took a while to inventory because of how the crates are stacked. "
            "After unpacking and recounting everything twice to make sure, Charlie has 10 bananas "
            "that were delivered last Tuesday. They are Cavendish bananas at varying stages of "
            "ripeness, with about half ready for immediate distribution."
        ))
        steps.append(TaskStep(
            "Down in the less-trafficked corner of the warehouse where we keep the specialty items, "
            "I found Diana's allocation. She has 2 pears sitting in the climate-controlled unit "
            "that we installed last month. They're Bartlett pears, quite firm, and should be "
            "perfect for distribution by the end of the week."
        ))
        steps.append(TaskStep(
            "Finally, I checked the vine fruit section, which is always the trickiest because of "
            "the delicate packaging requirements. Eve has 7 grapes — well, 7 bunches of grapes "
            "to be precise. They are Thompson seedless, properly wrapped in the new breathable "
            "packaging that the procurement team started using this season."
        ))

        # --- Phase 2: Verbose filler ---
        steps.append(TaskStep(
            "By the way, I noticed when I stepped outside for my break that the weather today "
            "is absolutely beautiful — sunny skies with barely a cloud in sight. The forecast "
            "said it might rain later this week but honestly looking at the sky right now "
            "I find that hard to believe. Perfect weather for the outdoor market tomorrow."
        ))
        steps.append(TaskStep(
            "Speaking of the market, I confirmed with the operations manager that the market "
            "opens at 9am sharp tomorrow as usual. They've been running the Saturday schedule "
            "for three months now and it seems to be working well for both the vendors and "
            "the customers. The early morning slot gets the most foot traffic apparently."
        ))
        steps.append(TaskStep(
            "I also had a chat with the pricing team during lunch. They mentioned that fruit "
            "prices have been remarkably stable this week, which is unusual for this time of "
            "year. Normally we see fluctuations due to seasonal transitions, but the supply "
            "chains have been running smoothly and demand has been consistent."
        ))
        steps.append(TaskStep(
            "On a related note, a new shipment of mangoes arrived yesterday afternoon around "
            "3pm. The delivery driver said there was some traffic on the highway but everything "
            "made it in good condition. We haven't sorted them into individual allocations yet "
            "but that should happen first thing tomorrow morning."
        ))
        steps.append(TaskStep(
            "One thing I wanted to flag for maintenance: the warehouse temperature is currently "
            "reading 4 degrees Celsius, which is right at the lower end of our acceptable range. "
            "The thermostat might need recalibration since we had that power outage last weekend. "
            "I've put in a ticket with facilities but haven't heard back yet."
        ))

        # --- Phase 3: Pre-correction recall ---
        steps.append(TaskStep(
            "Question: How many apples does Alice have?",
            expected_answer="5",
        ))
        steps.append(TaskStep(
            "Question: How many oranges does Bob have?",
            expected_answer="3",
        ))
        steps.append(TaskStep(
            "Question: How many bananas does Charlie have?",
            expected_answer="10",
        ))

        # --- Phase 4: Verbose filler ---
        steps.append(TaskStep(
            "Just got word from the transport coordinator that the delivery trucks are "
            "scheduled to arrive on Tuesdays and Fridays going forward. This is a change "
            "from the previous Monday-Wednesday-Friday schedule. The logistics team says "
            "this will reduce fuel costs by about 15% without impacting delivery times."
        ))
        steps.append(TaskStep(
            "I also learned that the inventory system was updated last month with a new "
            "module for tracking perishable goods. The old system only tracked quantities "
            "but the new one also monitors shelf life, storage temperature, and packaging "
            "integrity. It's been a learning curve for everyone but should pay off long term."
        ))
        steps.append(TaskStep(
            "Ran into Bob in the break room and he mentioned he might trade some of his "
            "oranges with one of the other warehouse tenants. Apparently there's been some "
            "interest from the juice vendor on the second floor. Nothing confirmed yet but "
            "Bob seemed keen on working something out before the oranges get too ripe."
        ))
        steps.append(TaskStep(
            "Alice visited the market this morning to check on her usual stall. She said "
            "the foot traffic was good and she's thinking about expanding her product range "
            "to include dried fruits in addition to the fresh inventory. She's been talking "
            "to a supplier in the neighboring county about dried mango and pineapple rings."
        ))

        # --- Phase 5: Corrections (verbose) ---
        steps.append(TaskStep(
            "Important update from the recount team: there was a counting error in the "
            "original north warehouse inventory. After a thorough recheck this afternoon, "
            "it turns out Alice now has 7 apples, not the 5 we originally recorded. "
            "Apparently two additional crates were delivered after the initial count and "
            "hadn't been logged in the system at that time."
        ))
        steps.append(TaskStep(
            "Another correction from the tropical section: Charlie's banana count needs "
            "to be updated as well. After reconciling with the delivery receipts, "
            "Charlie now has 4 bananas. It appears that 6 bunches were moved to a "
            "different storage area for a promotional event and shouldn't be counted "
            "in his regular inventory anymore."
        ))

        # --- Phase 6: Verbose filler ---
        steps.append(TaskStep(
            "Heads up to everyone: the store will close early on Friday this week due to "
            "the annual fire safety inspection. The inspection team needs access to all "
            "areas of the building including the storage zones, so we need everything "
            "cleared by 3pm. Please plan your distribution schedules accordingly."
        ))
        steps.append(TaskStep(
            "The pricing department sent out a memo saying that new pricing will take "
            "effect next week across all fruit categories. They've been analyzing market "
            "trends and competitor pricing for the past quarter and decided some adjustments "
            "are needed. Expect a 5-10% increase on tropical fruits and a slight decrease "
            "on stone fruits."
        ))
    else:
        # --- Concise version (original) ---
        steps.append(TaskStep("Alice has 5 apples."))
        steps.append(TaskStep("Bob has 3 oranges."))
        steps.append(TaskStep("Charlie has 10 bananas."))
        steps.append(TaskStep("Diana has 2 pears."))
        steps.append(TaskStep("Eve has 7 grapes."))

        steps.append(TaskStep("The weather today is sunny."))
        steps.append(TaskStep("The market opens at 9am."))
        steps.append(TaskStep("Fruit prices have been stable this week."))
        steps.append(TaskStep("A new shipment of mangoes arrived yesterday."))
        steps.append(TaskStep("The warehouse temperature is 4 degrees Celsius."))

        steps.append(TaskStep("Question: How many apples does Alice have?", expected_answer="5"))
        steps.append(TaskStep("Question: How many oranges does Bob have?", expected_answer="3"))
        steps.append(TaskStep("Question: How many bananas does Charlie have?", expected_answer="10"))

        steps.append(TaskStep("Transport trucks arrive on Tuesdays and Fridays."))
        steps.append(TaskStep("The inventory system was updated last month."))
        steps.append(TaskStep("Bob mentioned he might trade some oranges."))
        steps.append(TaskStep("Alice visited the market this morning."))

        steps.append(TaskStep("Correction: Alice now has 7 apples."))
        steps.append(TaskStep("Correction: Charlie now has 4 bananas."))

        steps.append(TaskStep("The store will close early on Friday."))
        steps.append(TaskStep("New pricing will take effect next week."))

    # --- Post-correction recall (same for both) ---
    steps.append(TaskStep("Question: How many apples does Alice have?", expected_answer="7"))
    steps.append(TaskStep("Question: How many bananas does Charlie have?", expected_answer="4"))
    steps.append(TaskStep("Question: How many oranges does Bob have?", expected_answer="3"))
    steps.append(TaskStep("Question: How many pears does Diana have?", expected_answer="2"))
    steps.append(TaskStep("Question: How many grapes does Eve have?", expected_answer="7"))

    return steps


# ---------------------------------------------------------------------------
# Scaled task: parameterized entity tracking with verbose descriptions
# ---------------------------------------------------------------------------

_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nathan", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]

_FRUITS = [
    "apples", "oranges", "bananas", "pears", "grapes", "mangoes",
    "peaches", "plums", "cherries", "strawberries", "blueberries",
    "kiwis", "lemons", "limes", "watermelons", "papayas",
    "figs", "dates", "coconuts", "pomegranates", "tangerines",
    "nectarines", "apricots", "guavas",
]

_LOCATIONS = [
    "the north warehouse near the loading dock",
    "the refrigerated section on aisle 3",
    "the south storage facility by the main entrance",
    "the climate-controlled unit in building B",
    "the overflow storage area behind the office",
    "the fresh produce cooler on the ground floor",
    "the temporary staging area near shipping",
    "the secure storage vault in the basement",
]

_VARIETIES = [
    "premium grade, hand-selected",
    "standard commercial grade",
    "organic certified, locally sourced",
    "imported, high quality",
    "mixed variety, various sizes",
    "farm-fresh, delivered this morning",
    "surplus stock from last week's order",
    "specialty variety, limited availability",
]

_FILLER_TEMPLATES = [
    "The maintenance team reported that {location} needs a new air filter replacement. "
    "They submitted a work order last {day} and are waiting for parts to arrive from "
    "the regional supply center. The estimated completion time is {n} business days, "
    "though expedited shipping could cut that down to {m} days if approved by management.",

    "According to the latest logistics report, delivery route {route} has been experiencing "
    "delays of approximately {n} minutes due to road construction on Highway {hw}. The "
    "transportation coordinator suggested rerouting through {alt_route} as a temporary "
    "measure until the construction project wraps up at the end of {month}.",

    "The quality assurance team completed their quarterly inspection of {location} and "
    "found that all storage conditions meet or exceed regulatory requirements. Temperature "
    "was measured at {temp} degrees Celsius, humidity at {humid}%, and ventilation rated "
    "as satisfactory. The next inspection is scheduled for {month}.",

    "A memo from corporate headquarters announced that the annual inventory reconciliation "
    "will take place during the week of {month} {n}th. All department heads are asked to "
    "ensure their records are up to date and that any discrepancies from the previous "
    "quarter have been resolved. The reconciliation team will need access to all facilities.",

    "The employee training session on the new barcode scanning system has been rescheduled "
    "from {day} to next {next_day}. The session will cover the basics of the updated "
    "handheld devices, the new software interface, and integration with the central "
    "inventory database. Attendance is mandatory for all warehouse staff.",

    "Market analysis from the research department indicates that consumer demand for "
    "fresh produce is expected to increase by {n}% over the next quarter. This is "
    "driven primarily by seasonal trends and the growing popularity of farm-to-table "
    "dining. The procurement team should plan accordingly for increased order volumes.",
]

_CORRECTION_TEMPLATES = [
    "Important correction from the recount team: after a thorough recheck of "
    "{location}, it turns out that {name} now has {new_count} {fruit}, not the "
    "{old_count} we originally recorded. {reason}",

    "Update to the inventory records: {name}'s {fruit} count has been revised. "
    "A reconciliation between the physical count and the database showed that "
    "{name} now has {new_count} {fruit}. The previous figure of {old_count} "
    "was incorrect due to {reason}",
]

_CORRECTION_REASONS = [
    "Additional crates were delivered after the initial count and hadn't been logged.",
    "some items were moved to a different storage area for a promotional event.",
    "a data entry error in the old system has been corrected.",
    "a batch was returned from a customer and added back to inventory.",
    "the original count included items that were already allocated elsewhere.",
]


def _make_filler(rng: random.Random) -> str:
    template = rng.choice(_FILLER_TEMPLATES)
    return template.format(
        location=rng.choice(_LOCATIONS),
        day=rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        next_day=rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        n=rng.randint(2, 15),
        m=rng.randint(1, 5),
        route=rng.choice(["Alpha", "Beta", "Gamma", "Delta"]),
        hw=rng.randint(1, 99),
        alt_route=rng.choice(["Oak Street bypass", "Riverside connector", "Industrial park loop"]),
        month=rng.choice(["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]),
        temp=rng.randint(2, 8),
        humid=rng.randint(40, 70),
    )


def build_scaled_task(
    num_entities: int = 15,
    filler_per_phase: int = 5,
    num_corrections: int = 5,
    seed: int = 42,
) -> list[TaskStep]:
    """Build a scaled entity tracking task with verbose descriptions.

    Args:
        num_entities: Number of person-fruit pairs to track
        filler_per_phase: Number of filler messages between each phase
        num_corrections: Number of entities to correct
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)

    num_entities = min(num_entities, len(_NAMES))
    num_corrections = min(num_corrections, num_entities)

    # Assign entities
    names = _NAMES[:num_entities]
    fruits = _FRUITS[:num_entities]
    locations = [rng.choice(_LOCATIONS) for _ in range(num_entities)]
    varieties = [rng.choice(_VARIETIES) for _ in range(num_entities)]
    counts = [rng.randint(2, 20) for _ in range(num_entities)]

    steps: list[TaskStep] = []

    # Phase 1: Establish facts
    for i in range(num_entities):
        steps.append(TaskStep(
            f"I just finished checking {locations[i]}. After carefully counting "
            f"and verifying against the manifest, I can confirm that {names[i]} "
            f"has {counts[i]} {fruits[i]} stored there. They are {varieties[i]}, "
            f"and all in good condition based on today's inspection."
        ))

    # Phase 2: Filler
    for _ in range(filler_per_phase):
        steps.append(TaskStep(_make_filler(rng)))

    # Phase 3: Pre-correction recall (ask about half the entities)
    recall_indices = rng.sample(range(num_entities), min(num_entities // 2, num_entities))
    for i in recall_indices:
        steps.append(TaskStep(
            f"Question: How many {fruits[i]} does {names[i]} have?",
            expected_answer=str(counts[i]),
        ))

    # Phase 4: More filler
    for _ in range(filler_per_phase):
        steps.append(TaskStep(_make_filler(rng)))

    # Phase 5: Corrections
    correction_indices = rng.sample(range(num_entities), num_corrections)
    new_counts = dict(zip(correction_indices, [rng.randint(1, 25) for _ in range(num_corrections)]))
    # Ensure new counts are different from old
    for idx in correction_indices:
        while new_counts[idx] == counts[idx]:
            new_counts[idx] = rng.randint(1, 25)

    for idx in correction_indices:
        template = rng.choice(_CORRECTION_TEMPLATES)
        reason = rng.choice(_CORRECTION_REASONS)
        steps.append(TaskStep(
            template.format(
                name=names[idx],
                fruit=fruits[idx],
                old_count=counts[idx],
                new_count=new_counts[idx],
                location=locations[idx],
                reason=reason,
            )
        ))

    # Phase 6: More filler
    for _ in range(filler_per_phase):
        steps.append(TaskStep(_make_filler(rng)))

    # Phase 7: Post-correction recall (ask about ALL entities)
    for i in range(num_entities):
        expected = str(new_counts[i]) if i in new_counts else str(counts[i])
        steps.append(TaskStep(
            f"Question: How many {fruits[i]} does {names[i]} have?",
            expected_answer=expected,
        ))

    return steps
