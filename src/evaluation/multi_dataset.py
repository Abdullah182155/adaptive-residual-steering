import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Benchmark Configuration
# ==============================================================================

@dataclass
class EvalBenchmarkConfig:
    """Configuration for multi-dataset reasoning benchmark harness."""
    datasets: List[str] = field(default_factory=lambda: ["gsm8k", "svamp", "arc", "math500", "gsm_plus"])
    samples_per_dataset: Dict[str, int] = field(default_factory=lambda: {
        "gsm8k": 100,
        "svamp": 100,
        "arc": 100,
        "math500": 50,
        "gsm_plus": 100,
    })
    shot_counts: List[int] = field(default_factory=lambda: [0, 2, 4, 8])
    prompt_shielding: bool = True
    seed: int = 42
    max_new_tokens: int = 512


# ==============================================================================
# 2. Few-Shot Exemplar Banks (0 to 8 Shots for Each Benchmark)
# ==============================================================================

EXEMPLARS_GSM8K: List[Dict[str, str]] = [
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

EXEMPLARS_SVAMP: List[Dict[str, str]] = [
    {
        "q": "Each pack of DVDs costs 6 dollars. If a customer buys 9 packs of DVDs, how much does the customer spend?",
        "a": "Step 1: The cost per pack is 6 dollars.\nStep 2: The customer buys 9 packs.\nStep 3: Total cost is 6 * 9 = 54 dollars.\nFinal answer: 54",
    },
    {
        "q": "Dan has 64 violet marbles. He gave 14 violet marbles to Mary and 22 to Jack. How many violet marbles does Dan have now?",
        "a": "Step 1: Dan started with 64 marbles.\nStep 2: He gave away 14 + 22 = 36 marbles.\nStep 3: He has 64 - 36 = 28 marbles left.\nFinal answer: 28",
    },
    {
        "q": "A bakery had 80 cakes. In the morning, they sold 25 cakes. In the afternoon, they sold another 30 cakes. How many cakes remained at the end of the day?",
        "a": "Step 1: Total cakes initially was 80.\nStep 2: Total sold is 25 + 30 = 55 cakes.\nStep 3: Remaining cakes is 80 - 55 = 25.\nFinal answer: 25",
    },
    {
        "q": "A farmer planted 4 rows of apple trees with 8 trees in each row. A storm destroyed 7 trees. How many trees are left?",
        "a": "Step 1: Total trees planted is 4 * 8 = 32 trees.\nStep 2: Storm destroyed 7 trees.\nStep 3: Trees left is 32 - 7 = 25 trees.\nFinal answer: 25",
    },
    {
        "q": "Lisa has 45 colored pencils. She gives 15 pencils to her brother and finds 8 more under her desk. How many colored pencils does Lisa have now?",
        "a": "Step 1: Lisa starts with 45 pencils.\nStep 2: After giving 15 away, she has 45 - 15 = 30 pencils.\nStep 3: After finding 8 more, she has 30 + 8 = 38 pencils.\nFinal answer: 38",
    },
    {
        "q": "There are 5 boxes with 12 oranges each. 10 oranges are rotten. How many good oranges are there?",
        "a": "Step 1: Total oranges is 5 * 12 = 60 oranges.\nStep 2: Rotten oranges count is 10.\nStep 3: Good oranges count is 60 - 10 = 50 oranges.\nFinal answer: 50",
    },
    {
        "q": "Mark scored 72 points in basketball. In the next game he scored 18 fewer points. How many points did he score in the second game?",
        "a": "Step 1: Score in first game was 72.\nStep 2: Score in second game is 72 - 18 = 54.\nFinal answer: 54",
    },
    {
        "q": "A library has 9 shelves with 14 books on each shelf. 26 books are checked out. How many books remain on the shelves?",
        "a": "Step 1: Total books originally is 9 * 14 = 126 books.\nStep 2: Books checked out is 26.\nStep 3: Remaining books is 126 - 26 = 100 books.\nFinal answer: 100",
    },
]

EXEMPLARS_ARC: List[Dict[str, str]] = [
    {
        "q": "Which energy transformation occurs when a flashlight is turned on?\n(A) chemical to electrical to light\n(B) light to electrical to chemical\n(C) thermal to mechanical to light\n(D) mechanical to chemical to light",
        "a": "Step 1: Flashlights use batteries storing chemical energy.\nStep 2: The chemical energy produces electrical current.\nStep 3: The filament or LED converts electrical energy into light.\nFinal answer: A",
    },
    {
        "q": "Which organ in the human body is primarily responsible for pumping blood?\n(A) Lungs\n(B) Heart\n(C) Liver\n(D) Kidney",
        "a": "Step 1: The heart contracts rhythmically to pump oxygenated blood through the circulatory system.\nFinal answer: B",
    },
    {
        "q": "Which characteristic best helps an animal survive in a cold arctic climate?\n(A) Thin skin\n(B) Large surface area\n(C) Thick layer of blubber\n(D) Brightly colored scales",
        "a": "Step 1: Arctic environments have freezing temperatures.\nStep 2: A thick layer of blubber provides thermal insulation against heat loss.\nFinal answer: C",
    },
    {
        "q": "What happens when water changes from liquid to gas during boiling?\n(A) Molecules move slower\n(B) Molecules move faster and spread apart\n(C) Mass decreases to zero\n(D) Chemical bonds are permanently broken",
        "a": "Step 1: Boiling adds thermal kinetic energy to water molecules.\nStep 2: Increased energy causes molecules to move faster and expand apart into vapor.\nFinal answer: B",
    },
    {
        "q": "Which object in our solar system produces its own visible light?\n(A) Moon\n(B) Mars\n(C) Jupiter\n(D) Sun",
        "a": "Step 1: The Sun is a star powered by nuclear fusion, emitting its own visible light.\nStep 2: Planets and moons reflect solar light.\nFinal answer: D",
    },
    {
        "q": "Which tool is best used to measure the mass of a rock sample?\n(A) Graduated cylinder\n(B) Triple beam balance\n(C) Metric ruler\n(D) Thermometer",
        "a": "Step 1: Mass is measured using a balance scale.\nStep 2: A triple beam balance directly determines mass in grams.\nFinal answer: B",
    },
    {
        "q": "Which type of rock is formed from cooling magma or lava?\n(A) Igneous\n(B) Sedimentary\n(C) Metamorphic\n(D) Fossil",
        "a": "Step 1: Molten rock cools and solidifies to form igneous rocks.\nFinal answer: A",
    },
    {
        "q": "What process do green plants use to convert sunlight into food?\n(A) Respiration\n(B) Photosynthesis\n(C) Fermentation\n(D) Transpiration",
        "a": "Step 1: Chloroplasts in plant cells capture solar photons to synthesize glucose from carbon dioxide and water.\nStep 2: This biochemical pathway is photosynthesis.\nFinal answer: B",
    },
]

EXEMPLARS_MATH500: List[Dict[str, str]] = [
    {
        "q": "Compute the sum of the positive roots of the equation x^2 - 7x + 12 = 0.",
        "a": "Step 1: Factor the quadratic equation: (x - 3)(x - 4) = 0.\nStep 2: The roots are x = 3 and x = 4.\nStep 3: Both roots are positive, so their sum is 3 + 4 = 7.\nFinal answer: 7",
    },
    {
        "q": "What is the degree measure of each interior angle in a regular hexagon?",
        "a": "Step 1: A hexagon has n = 6 sides.\nStep 2: The sum of interior angles is (n - 2) * 180 = 4 * 180 = 720 degrees.\nStep 3: Each angle measures 720 / 6 = 120 degrees.\nFinal answer: 120",
    },
    {
        "q": "Evaluate 5! / (3! * 2!).",
        "a": "Step 1: 5! = 120.\nStep 2: 3! = 6 and 2! = 2, so 3! * 2! = 12.\nStep 3: 120 / 12 = 10.\nFinal answer: 10",
    },
    {
        "q": "Find the slope of the line passing through points (2, 5) and (6, 17).",
        "a": "Step 1: Slope formula is m = (y2 - y1) / (x2 - x1).\nStep 2: m = (17 - 5) / (6 - 2) = 12 / 4 = 3.\nFinal answer: 3",
    },
    {
        "q": "If f(x) = 3x^2 - 4x + 1, what is f(3)?",
        "a": "Step 1: Substitute x = 3 into f(x): f(3) = 3*(3)^2 - 4*(3) + 1.\nStep 2: 3 * 9 = 27.\nStep 3: 27 - 12 + 1 = 16.\nFinal answer: 16",
    },
    {
        "q": "How many two-digit positive integers are divisible by 7?",
        "a": "Step 1: Smallest two-digit multiple is 14 = 7 * 2.\nStep 2: Largest two-digit multiple is 98 = 7 * 14.\nStep 3: Number of multiples is 14 - 2 + 1 = 13.\nFinal answer: 13",
    },
    {
        "q": "What is the area of a right triangle with legs of length 8 and 15?",
        "a": "Step 1: The area of a right triangle is (1/2) * leg1 * leg2.\nStep 2: Area = (1/2) * 8 * 15 = 4 * 15 = 60.\nFinal answer: 60",
    },
    {
        "q": "Solve for y in the system: 2x + y = 11 and x - y = 1.",
        "a": "Step 1: Add the two equations: (2x + y) + (x - y) = 11 + 1 => 3x = 12 => x = 4.\nStep 2: Substitute x = 4 into x - y = 1 => 4 - y = 1 => y = 3.\nFinal answer: 3",
    },
]

EXEMPLARS_GSMPLUS: List[Dict[str, str]] = [
    {
        "q": "A farmer starts with 24 sheep. He buys 18 more sheep, but 6 sheep wander away into the hills. How many sheep remain in the pen?",
        "a": "Step 1: The farmer starts with 24 sheep.\nStep 2: Adding 18 sheep gives 24 + 18 = 42 sheep.\nStep 3: Subtracting the 6 lost sheep gives 42 - 6 = 36 sheep.\nFinal answer: 36",
    },
    {
        "q": "A bookstore sells 15 novels on Monday and 3 times as many on Tuesday. On Wednesday they sell 10 fewer than Tuesday. How many novels were sold on Wednesday?",
        "a": "Step 1: Monday sales = 15.\nStep 2: Tuesday sales = 15 * 3 = 45.\nStep 3: Wednesday sales = 45 - 10 = 35.\nFinal answer: 35",
    },
    {
        "q": "A box contains 50 colored pencils. 12 are blue, 14 are red, and the remaining are green. How many pencils are green?",
        "a": "Step 1: Blue and red total is 12 + 14 = 26.\nStep 2: Green pencils count is 50 - 26 = 24.\nFinal answer: 24",
    },
    {
        "q": "A car travels at 60 mph for 3 hours, then increases speed to 70 mph for 2 hours. What is the total distance traveled in miles?",
        "a": "Step 1: Distance for first leg is 60 * 3 = 180 miles.\nStep 2: Distance for second leg is 70 * 2 = 140 miles.\nStep 3: Total distance is 180 + 140 = 320 miles.\nFinal answer: 320",
    },
    {
        "q": "A baker bakes 7 trays of cookies with 16 cookies on each tray. He packs them into boxes of 8 cookies each. How many boxes can he fill?",
        "a": "Step 1: Total cookies is 7 * 16 = 112 cookies.\nStep 2: Number of boxes is 112 / 8 = 14 boxes.\nFinal answer: 14",
    },
    {
        "q": "A train has 4 cars carrying 25 passengers each. At the first stop, 15 passengers get off and 20 get on. How many passengers are on the train now?",
        "a": "Step 1: Initial passengers = 4 * 25 = 100.\nStep 2: After 15 get off: 100 - 15 = 85.\nStep 3: After 20 get on: 85 + 20 = 105.\nFinal answer: 105",
    },
    {
        "q": "Maria earns $14 per hour. She works 35 hours a week. She saves $120 each week and spends the rest. How much does she spend each week?",
        "a": "Step 1: Total weekly earnings = 14 * 35 = 490 dollars.\nStep 2: Spending = 490 - 120 = 370 dollars.\nFinal answer: 370",
    },
    {
        "q": "An aquarium holds 90 gallons of water. It drains at a rate of 4 gallons per minute. How many gallons remain in the aquarium after 12 minutes?",
        "a": "Step 1: Amount drained = 4 * 12 = 48 gallons.\nStep 2: Amount remaining = 90 - 48 = 42 gallons.\nFinal answer: 42",
    },
]

BENCHMARK_EXEMPLARS: Dict[str, List[Dict[str, str]]] = {
    "gsm8k": EXEMPLARS_GSM8K,
    "svamp": EXEMPLARS_SVAMP,
    "arc": EXEMPLARS_ARC,
    "math500": EXEMPLARS_MATH500,
    "gsm_plus": EXEMPLARS_GSMPLUS,
}


# ==============================================================================
# 3. Prompt Construction
# ==============================================================================

def build_benchmark_prompt(dataset_name: str, question: str, n_shot: int = 0) -> str:
    """Build standardized reasoning prompt with n_shot exemplars and chain-of-thought prefix."""
    exemplars = BENCHMARK_EXEMPLARS.get(dataset_name.lower(), EXEMPLARS_GSM8K)
    prompt = ""
    for i in range(min(n_shot, len(exemplars))):
        ex = exemplars[i]
        prompt += f"Question: {ex['q']}\nAnswer: Let's think step by step.\n{ex['a']}\n\n"
    prompt += f"Question: {question}\nAnswer: Let's think step by step.\n"
    return prompt


# ==============================================================================
# 4. Answer Extraction & Equivalence Verifiers
# ==============================================================================

def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract numeric answer via strict priority hierarchy."""
    def safe_float(s: str) -> Optional[float]:
        try:
            s = s.replace(",", "").replace("$", "").replace("%", "").strip()
            if not s or not any(c.isdigit() for c in s):
                return None
            return float(s)
        except (ValueError, AttributeError):
            return None

    # Priority 1: #### <number>
    m = re.search(r"####\s*\$?(-?\d[\d,]*\.?\d*)", text)
    if m:
        return safe_float(m.group(1))

    # Priority 2: LaTeX \boxed{<num>}
    m = re.search(r"\\boxed\{\s*\$?(-?\d[\d,]*\.?\d*)\s*\}", text)
    if m:
        return safe_float(m.group(1))

    # Priority 3: Final answer [is] <number>
    m = re.search(r"[Ff]inal\s+answer(?:\s+is)?\s*[:\-]?\s*\$?(-?[\d,]+\.?\d*)", text)
    if m:
        return safe_float(m.group(1))

    # Priority 4: The answer is <number>
    m = re.search(r"[Tt]he\s+answer\s+is\s*[:\-]?\s*\$?(-?[\d,]+\.?\d*)", text)
    if m:
        return safe_float(m.group(1))

    # Priority 5: Therefore ... <number>.
    m = re.search(r"[Tt]herefore[^.]*?(-?\d[\d,]*\.?\d*)[^.]*\.", text)
    if m:
        sentence_match = re.search(r"[Tt]herefore[^.]*\.", text)
        if sentence_match:
            nums = re.findall(r"-?[\d,]+\.?\d*", sentence_match.group())
            if nums:
                return safe_float(nums[-1])

    # Priority 6: Last assignment = or is <number>
    matches = list(re.finditer(r"(?:=|is)\s*\$?(-?[\d,]+\.?\d*)", text))
    if matches:
        return safe_float(matches[-1].group(1))

    # Priority 7: Last standalone number
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        return safe_float(nums[-1])

    return None


def extract_choice_answer(text: str) -> Optional[str]:
    """Extract multiple-choice letter (A, B, C, D) or digit option."""
    # Priority 1: Final answer: (A) or Final answer: A
    m = re.search(r"[Ff]inal\s+answer(?:\s+is)?\s*[:\-]?\s*\(?([A-Ea-e1-5])\)?", text)
    if m:
        return m.group(1).upper()

    # Priority 2: The [correct] answer is (A)
    m = re.search(r"[Tt]he(?:\s+correct)?\s+answer\s+is\s*[:\-]?\s*\(?([A-Ea-e1-5])\)?", text)
    if m:
        return m.group(1).upper()

    # Priority 3: \boxed{A}
    m = re.search(r"\\boxed\{\s*([A-Ea-e1-5])\s*\}", text)
    if m:
        return m.group(1).upper()

    # Priority 4: Therefore, (A)
    m = re.search(r"[Tt]herefore[,\s]+\(?([A-Ea-e1-5])\)?", text)
    if m:
        return m.group(1).upper()

    # Priority 5: standalone (A) / (B) / (C) / (D) near the end
    matches = list(re.finditer(r"\(([A-Ea-e1-5])\)", text))
    if matches:
        return matches[-1].group(1).upper()

    return None


def evaluate_sample_answer(
    dataset_name: str,
    response: str,
    gold_answer: Union[float, str, int],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate response against gold answer based on benchmark dataset type."""
    is_mc = dataset_name.lower() in {"arc", "ai2_arc"}
    step_count = max(
        len(re.findall(r"Step \d+", response, re.IGNORECASE)),
        len(re.findall(r"\n\d+[\.\)]\s", response)),
        len(re.findall(r"=\s*\$?-?\d", response)),
    )

    if is_mc:
        extracted = extract_choice_answer(response)
        gold_str = str(gold_answer).strip().upper()
        # Handle digit to letter mapping e.g. '1' -> 'A'
        digit_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        norm_gold = digit_to_letter.get(gold_str, gold_str)
        norm_extracted = digit_to_letter.get(str(extracted), str(extracted)) if extracted else None
        correct = (norm_extracted is not None and norm_extracted == norm_gold)
        has_answer = extracted is not None
    else:
        extracted = extract_numeric_answer(response)
        try:
            gold_num = float(gold_answer) if gold_answer is not None else None
        except (ValueError, TypeError):
            gold_num = None

        correct = (
            extracted is not None
            and gold_num is not None
            and abs(extracted - gold_num) <= tolerance
        )
        has_answer = extracted is not None

    return {
        "correct": bool(correct),
        "extracted": extracted,
        "gold": gold_answer,
        "has_answer": has_answer,
        "steps": step_count,
    }


# ==============================================================================
# 5. Multi-Dataset Loader & Offline Fallbacks
# ==============================================================================

# Embedded offline backup samples for standalone / air-gapped test validation
EMBEDDED_BENCHMARK_SAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "gsm8k": [
        {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "gold": 72.0},
        {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "gold": 10.0},
        {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money he needs. Her parents gave her $15 and her grandparents gave her twice as much as her parents. How much more money does Betty need to buy the wallet?", "gold": 5.0},
        {"question": "Albert is wondering how much pizza he can eat in one month. He buys 2 large pizzas each week. Each pizza has 8 slices. In 4 weeks, how many slices of pizza will he eat?", "gold": 64.0},
        {"question": "Ken created a care package to send to his brother, who was away at school. Ken placed 4 boxes of cookies with 10 cookies in each box and 3 boxes of crackers with 12 crackers in each box. How many cookies and crackers did Ken send?", "gold": 76.0},
    ],
    "svamp": [
        {"question": "Each pack of DVDs costs 6 dollars. If there is a discount to buy 9 packs for 45 dollars, how much less does a customer pay compared to the regular price?", "gold": 9.0},
        {"question": "Dan has 64 violet marbles. He gave 14 to Mary and 22 to Jack. How many marbles does Dan have left?", "gold": 28.0},
        {"question": "A bakery had 80 cakes. In the morning they sold 25 cakes, and in the afternoon they sold 30. How many cakes remained?", "gold": 25.0},
        {"question": "There are 5 boxes with 12 oranges each. 10 oranges are bad. How many good oranges are there?", "gold": 50.0},
        {"question": "A farmer planted 4 rows of apple trees with 8 trees in each row. A storm destroyed 7 trees. How many trees are left?", "gold": 25.0},
    ],
    "arc": [
        {"question": "Which organ in the human body is primarily responsible for pumping blood?\n(A) Lungs\n(B) Heart\n(C) Liver\n(D) Kidney", "gold": "B"},
        {"question": "Which energy transformation occurs when a flashlight is turned on?\n(A) chemical to electrical to light\n(B) light to electrical to chemical\n(C) thermal to mechanical to light\n(D) mechanical to chemical to light", "gold": "A"},
        {"question": "Which tool is best used to measure the mass of a rock sample?\n(A) Graduated cylinder\n(B) Triple beam balance\n(C) Metric ruler\n(D) Thermometer", "gold": "B"},
        {"question": "Which object in our solar system produces its own visible light?\n(A) Moon\n(B) Mars\n(C) Jupiter\n(D) Sun", "gold": "D"},
        {"question": "What process do green plants use to convert sunlight into food?\n(A) Respiration\n(B) Photosynthesis\n(C) Fermentation\n(D) Transpiration", "gold": "B"},
    ],
    "math500": [
        {"question": "Compute the sum of the positive roots of the equation x^2 - 7x + 12 = 0.", "gold": 7.0},
        {"question": "What is the degree measure of each interior angle in a regular hexagon?", "gold": 120.0},
        {"question": "Evaluate 5! / (3! * 2!).", "gold": 10.0},
        {"question": "Find the slope of the line passing through points (2, 5) and (6, 17).", "gold": 3.0},
        {"question": "If f(x) = 3x^2 - 4x + 1, what is f(3)?", "gold": 16.0},
    ],
    "gsm_plus": [
        {"question": "A farmer starts with 24 sheep. He buys 18 more sheep, but 6 sheep wander away into the hills. How many sheep remain in the pen?", "gold": 36.0},
        {"question": "A bookstore sells 15 novels on Monday and 3 times as many on Tuesday. On Wednesday they sell 10 fewer than Tuesday. How many novels were sold on Wednesday?", "gold": 35.0},
        {"question": "A box contains 50 colored pencils. 12 are blue, 14 are red, and the remaining are green. How many pencils are green?", "gold": 24.0},
        {"question": "A car travels at 60 mph for 3 hours, then increases speed to 70 mph for 2 hours. What is the total distance traveled in miles?", "gold": 320.0},
        {"question": "A baker bakes 7 trays of cookies with 16 cookies on each tray. He packs them into boxes of 8 cookies each. How many boxes can he fill?", "gold": 14.0},
    ],
}


def load_benchmark_dataset(
    dataset_name: str,
    n_samples: int = 100,
    seed: int = 42,
    use_fallback_if_failed: bool = True,
) -> List[Dict[str, Any]]:
    """Load benchmark dataset from Hugging Face or fallback to embedded verified samples.

    Supported benchmark names:
    - 'gsm8k'    (`openai/gsm8k`)
    - 'svamp'    (`Chillee/SVAMP`)
    - 'arc'      (`allenai/ai2_arc`, ARC-Challenge)
    - 'math500'  (`HuggingFaceH4/MATH-500`)
    - 'gsm_plus' (`qintong/GSM-Plus`)
    """
    key = dataset_name.lower().replace("-", "_")
    samples: List[Dict[str, Any]] = []

    try:
        from datasets import load_dataset

        if key == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="test")
            ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
            for row in ds:
                ans_str = row["answer"]
                gold_val = None
                if "####" in ans_str:
                    try:
                        gold_val = float(ans_str.split("####")[1].replace(",", "").strip())
                    except ValueError:
                        gold_val = None
                samples.append({
                    "question": row["question"],
                    "gold": gold_val,
                    "dataset": "gsm8k",
                })

        elif key == "svamp":
            ds = load_dataset("Chillee/SVAMP", split="test")
            ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
            for row in ds:
                body = row.get("Body", "")
                question = row.get("Question", "")
                full_q = f"{body} {question}".strip()
                ans_raw = row.get("Answer", None)
                try:
                    gold_val = float(ans_raw) if ans_raw is not None else None
                except ValueError:
                    gold_val = None
                samples.append({
                    "question": full_q,
                    "gold": gold_val,
                    "dataset": "svamp",
                })

        elif key in {"arc", "ai2_arc"}:
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
            for row in ds:
                q_text = row["question"]
                choices = row["choices"]
                labels = choices["label"]
                texts = choices["text"]
                choice_str = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
                full_q = f"{q_text}\n{choice_str}"
                gold_key = str(row["answerKey"]).strip().upper()
                samples.append({
                    "question": full_q,
                    "gold": gold_key,
                    "dataset": "arc",
                })

        elif key in {"math500", "math_500"}:
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
            for row in ds:
                q_text = row.get("problem", "")
                ans_raw = row.get("answer", "")
                gold_val = extract_numeric_answer(str(ans_raw))
                samples.append({
                    "question": q_text,
                    "gold": gold_val,
                    "dataset": "math500",
                })

        elif key in {"gsm_plus", "gsmplus"}:
            ds = load_dataset("qintong/GSM-Plus", split="test")
            ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
            for row in ds:
                q_text = row.get("question", "")
                ans_str = row.get("answer", "")
                gold_val = extract_numeric_answer(str(ans_str))
                samples.append({
                    "question": q_text,
                    "gold": gold_val,
                    "dataset": "gsm_plus",
                })

        else:
            raise ValueError(f"Unknown benchmark dataset name: {dataset_name}")

    except Exception as e:
        logger.warning("Failed loading dataset '%s' from HuggingFace (%s). Using fallback: %s", key, e, use_fallback_if_failed)
        if not use_fallback_if_failed:
            raise

    # Fallback to embedded samples if loading was empty or failed
    if not samples:
        fallback_list = EMBEDDED_BENCHMARK_SAMPLES.get(key, EMBEDDED_BENCHMARK_SAMPLES["gsm8k"])
        # Cycle if requested n_samples > len(fallback_list)
        samples = []
        for i in range(n_samples):
            item = fallback_list[i % len(fallback_list)].copy()
            item["dataset"] = key
            samples.append(item)

    return samples
