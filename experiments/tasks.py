"""Shared task definitions for experiments.

Both the scaffold and baseline experiments import from here
to ensure they use identical inputs.
"""

from __future__ import annotations

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
