"""
Unified prompt builder for ARS training, evaluation, and sampling.

Phase 0.1: Single prompt builder with harness hash validation.
"""

import hashlib
from typing import List, Dict, Optional

# Legacy exemplars (kept for reproducibility of earlier results)
LEGACY_GSM8K_EXEMPLARS = [
    {
        "q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "a": "Step 1: There are 15 trees originally.\nStep 2: Then there were 21 trees after some more were planted.\nStep 3: So there must have been 21 - 15 = 6 trees planted.\nFinal answer: 6",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "a": "Step 1: There are originally 3 cars.\nStep 2: 2 more cars arrive.\nStep 3: 3 + 2 = 5 cars.\nFinal answer: 5",
    },
    {
        "q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "a": "Step 1: Originally, Leah had 32 chocolates.\nStep 2: Her sister had 42.\nStep 3: So in total they had 32 + 42 = 74.\nStep 4: After eating 35, they had 74 - 35 = 39.\nFinal answer: 39",
    },
    {
        "q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "a": "Step 1: Jason started with 20 lollipops.\nStep 2: Then he had 12 after giving some to Denny.\nStep 3: So he gave Denny 20 - 12 = 8 lollipops.\nFinal answer: 8",
    },
    {
        "q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "a": "Step 1: Shawn started with 5 toys.\nStep 2: He got 2 toys from his mom.\nStep 3: He got 2 toys from his dad.\nStep 4: 5 + 2 + 2 = 9 toys.\nFinal answer: 9",
    },
    {
        "q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "a": "Step 1: There were originally 9 computers.\nStep 2: 5 more computers were added each day.\nStep 3: From Monday to Thursday is 4 days.\nStep 4: So 5 * 4 = 20 computers were added.\nStep 5: 9 + 20 = 29 computers.\nFinal answer: 29",
    },
    {
        "q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "a": "Step 1: Michael started with 58 golf balls.\nStep 2: After losing 23 on Tuesday, he had 58 - 23 = 35.\nStep 3: After losing 2 more on Wednesday, he had 35 - 2 = 33.\nFinal answer: 33",
    },
    {
        "q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "a": "Step 1: Olivia started with $23.\nStep 2: She bought 5 bagels for $3 each.\nStep 3: 5 bagels cost 5 * 3 = 15 dollars.\nStep 4: 23 - 15 = 8 dollars.\nFinal answer: 8",
    },
]

# New v5 exemplars (original problems, arithmetic checked)
V5_EXEMPLARS = [
    {
        "q": "A machine fills 45 bottles per minute. It runs for 8 minutes, and then 120 of the bottles are rejected. The rest are packed in cartons of 20. How many full cartons are there?",
        "a": "Step 1: In 8 minutes the machine fills 45 * 8 = 360 bottles.\nStep 2: After rejecting 120, 360 - 120 = 240 bottles remain.\nStep 3: The cartons hold 20 each, so 240 / 20 = 12 cartons.\nFinal answer: 12",
    },
    {
        "q": "A jacket costs $80. It is discounted by 25%, and then a 10% sales tax is added to the discounted price. What is the final price in dollars?",
        "a": "Step 1: The discount is 25% of 80, which is 80 * 0.25 = 20.\nStep 2: The discounted price is 80 - 20 = 60.\nStep 3: The tax is 10% of 60, which is 60 * 0.10 = 6.\nStep 4: The final price is 60 + 6 = 66.\nFinal answer: 66",
    },
    {
        "q": "Maria had $120. She spent 1/4 of it on books and then 1/3 of what was left on lunch. How much money does she have now?",
        "a": "Step 1: Books cost 120 / 4 = 30 dollars.\nStep 2: After the books she has 120 - 30 = 90 dollars.\nStep 3: Lunch costs 90 / 3 = 30 dollars.\nStep 4: She has 90 - 30 = 60 dollars left.\nFinal answer: 60",
    },
    {
        "q": "Sam has 12 marbles. Ben has 8 more than Sam. Tom has 3 times as many as Ben. Tom gives 10 marbles to Sam. How many marbles do Tom and Sam have together now?",
        "a": "Step 1: Ben has 12 + 8 = 20 marbles.\nStep 2: Tom starts with 3 * 20 = 60 marbles.\nStep 3: After giving 10 away, Tom has 60 - 10 = 50 marbles.\nStep 4: After receiving 10, Sam has 12 + 10 = 22 marbles.\nStep 5: Together they have 50 + 22 = 72 marbles.\nFinal answer: 72",
    },
    {
        "q": "A train travels for 3 hours and 45 minutes at 80 miles per hour. How many miles does it travel?",
        "a": "Step 1: 45 minutes is 45 / 60 = 0.75 hours.\nStep 2: The total time is 3 + 0.75 = 3.75 hours.\nStep 3: The distance is 80 * 3.75 = 300 miles.\nFinal answer: 300",
    },
    {
        "q": "A gym charges a $50 joining fee plus $30 per month. Lena has paid $290 in total. For how many months has she been a member?",
        "a": "Step 1: Remove the joining fee: 290 - 50 = 240 dollars of monthly fees.\nStep 2: Each month costs 30, so the number of months is 240 / 30 = 8.\nFinal answer: 8",
    },
    {
        "q": "The ratio of boys to girls in a class is 3:5, and there are 40 students. How many more girls than boys are there?",
        "a": "Step 1: The ratio has 3 + 5 = 8 parts.\nStep 2: Each part is 40 / 8 = 5 students.\nStep 3: There are 3 * 5 = 15 boys and 5 * 5 = 25 girls.\nStep 4: The difference is 25 - 15 = 10.\nFinal answer: 10",
    },
    {
        "q": "A bakery bakes 6 trays of 24 muffins. It sells 2/3 of the muffins at $2 each and the rest at $1 each. How much money does it make?",
        "a": "Step 1: There are 6 * 24 = 144 muffins in total.\nStep 2: It sells 144 * 2 / 3 = 96 muffins at $2, which brings in 96 * 2 = 192 dollars.\nStep 3: The remaining 144 - 96 = 48 muffins sell at $1, which brings in 48 dollars.\nStep 4: The total is 192 + 48 = 240 dollars.\nFinal answer: 240",
    },
]

PREFACE = "Solve each problem step by step. Write every step on its own line as \"Step N: ...\" and finish with \"Final answer: <number>\"."

# Exemplar set seed for reproducibility
EXEMPLAR_SEED = 42


def _get_harness_hash() -> str:
    """Compute hash of (preface + exemplars + builder source) for harness validation."""
    source = f"{PREFACE}{str(V5_EXEMPLARS)}{EXEMPLAR_SEED}"
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def build_prompt_unified(
    question: str,
    n_shot: int = 0,
    exemplars: Optional[List[Dict[str, str]]] = None,
    preface: Optional[str] = None,
    exemplar_set: str = "v5",
    include_preface: bool = True,
) -> str:
    """
    Unified prompt builder for training, evaluation, and sampling.
    
    Args:
        question: The test question
        n_shot: Number of exemplars (0 for 0-shot)
        exemplars: Optional custom exemplar list (uses default if None)
        preface: Optional custom preface (uses default if None)
        exemplar_set: "legacy" or "v5" (default "v5")
        include_preface: Whether to include preface (True for v5, False for legacy compatibility)
    
    Returns:
        Complete prompt string
    """
    if exemplars is None:
        exemplars = V5_EXEMPLARS if exemplar_set == "v5" else LEGACY_GSM8K_EXEMPLARS
    
    if preface is None:
        preface = PREFACE
    
    prompt = ""
    if include_preface:
        prompt = f"{preface}\n\n"
    
    # Add exemplars (shielded tokens - not counted in steering)
    for i in range(min(n_shot, len(exemplars))):
        ex = exemplars[i]
        prompt += f"Question: {ex['q']}\nAnswer: Let's think step by step.\n{ex['a']}\n\n"
    
    # Test item (unshielded)
    prompt += f"Question: {question}\nAnswer: Let's think step by step.\n"
    
    return prompt


def build_prompt_from_config(question: str, n_shot: int, cfg) -> str:
    """Build prompt using config settings."""
    exemplar_set = getattr(cfg, 'exemplar_set', 'legacy')
    include_preface = exemplar_set == 'v5'
    return build_prompt_unified(question, n_shot=n_shot, exemplar_set=exemplar_set, include_preface=include_preface)


def get_harness_hash() -> str:
    """Get the current harness hash for validation."""
    return _get_harness_hash()
