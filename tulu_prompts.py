# These pompts are taken from: Tülu 3: Pushing Frontiers in Open Language Model Post-Training (https://arxiv.org/pdf/2411.15124)

# Precise Instuction Following (prompt)
prompt1 = """
    Create a verifiable instruction that the following persona might ask you to do:
    {persona}
    An example of verifiable instruction could be: {example}
    Note:
    1. The above example is not tied to any particular persona, but you should create one that
    is unique and specific to the given persona.
    2. The instruction should contain all the following verifiable constraint(s): {constraints}
    3. Your output should start with "User instruction:". Your output should not include an answer to
    the instruction.
"""

# Precise Instuction Following (response)
response1 = """
    Provide a response to the given instruction while satisfying the constraints.
    Instruction: {generated_instruction}
    Note that you should follow the instruction precisely and satisfy all the constraints.
"""

# Rewriting the Instruction Following Instance (Preference Data Construction)
prompt2 = """
    Rewrite the given instruction to remove one of the constraints.
    {Instruction}
    Note:
    1. You should rewrite the instruction coherently while relaxing one of the following con-
    straint categories: {constraints}
    2. Remember to entirely relax one of the constraint category
    3. Your output should start with "User instruction:". Your output should not include an answer to
    the instruction.
"""

# Hard Math Problems (prompt)
prompt3 = """
    Create a math problem related to the following persona:
    {persona}
    Note:
    1. The math problem should be challenging and involve advanced mathematical skills and
    knowledge. Only top talents can solve it correctly.
    2. You should make full use of the persona description to create the math problem to ensure that the
    math problem is unique and specific to the persona.
    3. Your response should always start with "Math problem:". Your response should not include a
    solution to the created math problem.
    4. Your created math problem should include no more than 2 sub-problems.
"""

# Hard Math Problems (response)
response3 = """
    Provide solution to the given math problem.
    Problem: {generated_math_problem}
    Note: Provide your solution step-by-step, and end your solution in a new line in the follow-
    ing format:
    Final Answer: The final answer is $final_answer$. I hope it is correct.
"""

# Code Completion (prompt)
prompt4 = """
    {persona}
    Assume you are the persona described above and you are asking a python programming
    question in stackoverflow.
    Note:
    1. Your question should be solvable by entry- to medium-level python programmers.
    2. Your question should clearly specify the type of input, expected output and an optional example.
    3. Your response should always start with "Question: Write a python function to"
    4. Your response should not include a solution to the created coding problem.
"""

# Code Completion (response)
response4 = """
    Provide solution to the given python programming question.
    Question: {generated_code_problem}
    Note:
    1. Your response should always start with the function definition and end with the final re-
    turn statement.
    2. Your response should only and only include python function.
"""

# System prompt for LLM-as-a-judge
prompt5 = """
    Your role is to evaluate text quality based on given criteria. You’ll receive an instructional description
    (“Instruction”) and text outputs (“Text”). Understand and interpret instructions to evaluate effectively.
    Provide annotations for each text with a rating and rationale. The texts given are independent, and
    should be evaluated separately.
"""

# Formatting a preference instance for LLM-as-a-judge
prompt6 = """
    { aspect_guideline }
    ## Format:
    ### Input
    Instruction: [Clearly specify the task goal and restrictions]
    Texts:
    {% for i in range(1, completions|length + 1) %}
    <text {{ i }}> [Text {{ i }}]
    {% endfor %}
    ### Output
    {% for i in range(1, completions|length + 1) %}
    #### Output for Text {{ i }}
    {% if identifier is defined %}
    Type: [List of numeric identifiers (or "None"), separatedby commas]
    Rationale: [Rationale for identification in short sentences]
    {% endif %}
    Rating: [Rating for text {{ i }}]
    Rational: [rational for the rating in short sentences]
    {% endfor %}
    —
    ## Annotation
    ### Input Instruction: {{ instruction }}
    Texts: {% for completion in completions %}
    <text {{ loop.index + 1 }}> {{ completion }}
    {% endfor %}
    ### Output
"""

# Instruction Following Aspect (prompt)
prompt7 = """
    # Instruction Following Assessment
    Evaluate alignment between output and intent. Assess understanding of task goal and restrictions.
    Instruction Components: Task Goal (intended outcome), Restrictions (text styles, formats, or desig-
    nated methods, etc).
    Scoring: Rate outputs 1 to 5:
    1. Irrelevant: No alignment.
    2. Partial Focus: Addresses one aspect poorly.
    3. Partial Compliance:
    - (1) Meets goal or restrictions, neglecting other.
    - (2) Acknowledges both but slight deviations.
    4. Almost There: Near alignment, minor deviations.
    5. Comprehensive Compliance: Fully aligns, meets all requirements.
"""

# Informativeness or Helpfulness Aspect (prompt)
prompt8 = """
    # Informativeness / Helpfulness Assessment
    Evaluate if model’s outputs fulfill task objectives and provide high-quality, correct, and, informative
    content.
    Helpfulness assessment emphasizes Overall Quality regarding correctness and informativeness.
    Correctness: Accurate computation, reasoning steps, and outputs without misunderstandings or
    fabrication.
    Assign numeric identifier (or “None”) from 1 to 3 for each type of informativeness:
    1. Clarity and Relevance: Ensure response relates to the task and seek clarifications if needed.
    2. Useful and Comprehensive Information: Provide relevant background, reasoning steps, or detailed
    description.
    3. Not Lengthy, No Repetition: Avoid verbosity or recycling content.
    Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
    1. Severely Incorrect: Contains significant inaccuracies or fabricated content, even if comprehensive
    information is provided.
    2. Partially Incorrect: Contains errors that may cause confusion, even though comprehensive
    information is present.
    3. Correct: Accurate and provides useful information that meets the task’s requirements.
    4. Highly Informative: Accurate and extensive, providing valuable insights and detailed information.
    5. Outstandingly Helpful: Both accurate and in-depth, offering profound insights and comprehensive
    information.
"""

# Honesty Aspect (prompt)
prompt9 = """
    # Honesty and Uncertainty Expression Assessment
    Assess how well the model conveys honesty and uncertainty. Evaluate if the model’s confidence aligns
    with response correctness.
    - Indicators of Uncertainty:
    1. Weakeners: e.g., ‘I guess,’ ‘probably.’
    2. Refusal to answer.
    3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning
    confident; (80, 100] high.
    - No uncertainty expression indicate confidence.
    - Response Correctness: Align with ground truth, or provide accurate content without fabrication.
    Scoring: Rate outputs 1 to 5 (or “N/A”):
    1. Confidently Incorrect: Confident but entirely wrong.
    2. Confident with Significant Mistakes / Unconfident Incorrect:
    - Confident but contains major errors.
    - Unconfident and entirely wrong.
    3. Uncertain / ‘I Don’t Know’ / Subtle Mistakes:
    - ‘I don’t know’ or declines.
    - Confident but contains minor errors.
    - Unconfident and contains significant mistakes.
    4. Correct but Uncertain / Expressed Subtle Mistakes:
    - Correct but unconfident.
    - Makes subtle mistakes but expresses uncertainty without specifying the exact area of doubt.
    5. Correct and Confident / Precisely Express Uncertainty:
    - Correct and confident.
    - Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential
    mistakes.
    N/A. Not Applicable: For creative writing tasks.
"""

# Truthfulness and Hallucination Aspect (prompt)
prompt10 = """
    # Truthfulness and Hallucination Assessment
    Evaluate the model’s accuracy in providing information without introducing misleading or fabricated
    details.
    Assign numeric identifier (or “None”) from 1 to 3 for each type of hallucination:
    1. Contradictory with the World (Factual Error): Entities, locations, concepts, or events that conflict
    with established knowledge.
    2. Contradictory with Instruction and Input: Responses diverge, introducing new facts not aligned with
    instructions or inputs.
    3. Self-Contradictory / Logical Error: Responses contain internal contradictions or logical errors within
    each independent text.
    Scoring: Rate outputs 1 to 5 based on extent of hallucination:
    1. Completely Hallucinated: Entirely unreliable due to hallucinations.
    2. Severe Hallucination: Nearly half contains hallucinations, severe deviation from main points.
    3. Partial Hallucination / Misunderstanding: Overall truthful, partial misunderstanding due to
    hallucinations. 4. Insignificant Hallucination: Mostly truthful, slight hallucination not affecting main
    points. 5. No Hallucination: Free of hallucinations.
"""

# 0-shot reasoning prompt for multiple-choice unseen tasks
prompt11 = """
    Answer the following multiple-choice question by giving the correct answer letter in parentheses.
    Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the
    answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
    Question: {question}
    (A) {choice_A}
    (B) {choice_B}
    (C)...
    Answer the above question and REMEMBER to finish your response with the exact phrase
    "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B),
    (C), (D), (E), etc.
"""

# 0-shot reasoning prompt for Deepmind Math unseen task
prompt12 = """
    Solve the following math problem efficiently:
    {math_problem}
    Show your work and conclude with the exact phrasing “Therefore, the final answer is [answer]. I hope
    it is correct.” where [answer] is just the final number, expression, or answer label representing the
    solution. Some example answers from this question category:
    - If the answer is {example_answer_1}, conclude with “Therefore, the final answer is {example_-
    answer_1}. I hope it is correct.”
    - If the answer is {example_answer_2}, conclude with “Therefore, the final answer is {example_-
    answer_2}. I hope it is correct.”
    - If the answer is {example_answer_3}, conclude with “Therefore, the final answer is {example_-
    answer_3}. I hope it is correct.”
    Note the formatting for the following answer types:
    - If the answer is a list (e.g., when there are two solutions to an equation), unless otherwise specified,
    present the solutions in a list separated by commas ordering them from the smallest to biggest e.g.: 2,
    10
    - Powers should be written with **, for instance x to the power of 2 should be written as x**2
    - Use * for multiplication, e.g.: 2*x
    - For fractions, separate the numerator and denominator with a slash (/) e.g.: -2/7
"""

