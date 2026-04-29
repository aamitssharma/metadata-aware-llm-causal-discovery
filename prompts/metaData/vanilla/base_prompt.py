
EDGE_PROMPT = """

### ROLE
	- You are an expert Causal Inference Engine with deep knowledge of Directed Acyclic Graphs (DAGs), structural equation modeling, and domain-specific mechanisms.
	- You are given the full set of variables and their definitions.
	- Analyze the provided domain and metadata to estimate the probability of a DIRECT causal relationship for every ordered pair A -> B.

### DEFINITION: "DIRECT CAUSE"
A directly causes B (A → B) if and only if:
	1. MANIPULATION: If an external agent intervenes to change the value of A, the distribution of B changes.
	2. ADJACENCY: This effect is not entirely mediated by any other variable provided in the list. If A affects B only through C, then A → B is FALSE (probability should be low).
	3. ASYMMETRY: Changes in B do not result in changes in A through the same mechanism.

### EVALUATION CRITERIA
	- **Avoid "Common Cause" bias:** If A and B are both caused by C (A ← C → B), the probability for A → B must be low despite high correlation.
	- **Avoid "Reverse Causality":** Penalize the score if B is more likely to be the parent of A, or if a strong feedback loop exists.
	- **Avoid "Proxy/Paper Tiger" variables:** If A is merely a lagging indicator or a symptom of an underlying trait (Z), and manipulating A alone doesn't change B, the probability is LOW.
	- **Avoid "Displacement" effects:** If A only shifts B to a different location or time without changing the net outcome at a system level, the causal link is weak.
	- **Identify "Plausible Mechanism":** A probability should be near zero if there is no physical, biological, or logical "How" connecting the two.
	- **Temporal Priority:** Causes must precede or be contemporaneous with their direct effects.

### INPUT
### METADATA

{input_json}

### TASK
- Evaluate every possible ordered pair of distinct variables in the metadata.
- For each ordered pair, provide only:
		- the probability that a direct edge A -> B exists
- Use the full variable set to reason about confounding, mediation, reverse causality, and proxy variables.


"""


NO_EDGE_PROMPT = """


### ROLE
- You are an expert Causal Inference Engine specializing in Causal Discovery and Structural Independence testing.
- Your goal is to estimate the probability that there is **NO DIRECT EDGE** for every ordered pair A -> B.

### TASK
Analyze the provided domain and metadata to estimate the probability of the statement: **"A does NOT directly cause B."** Do this for every possible ordered pair of distinct variables.
- High Probability (e.g., 0.9 - 1.0) = Strong evidence that A and B are independent, or that their relationship is purely indirect or spurious.
- Low Probability (e.g., 0.0 - 0.2) = Strong evidence that A is a direct parent of B.

### DEFINITION: "NO DIRECT CAUSE"
The relationship A → B is FALSE if any of the following are true:
1. **INDEPENDENCE:** Intervening on A has zero effect on the distribution of B.
2. **SPURIOUSNESS:** The correlation between A and B is entirely explained by a common cause C (A ← C → B).
3. **TOTAL MEDIATION:** A only affects B through an intermediate variable C (A → C → B). In this case, there is no *direct* edge between A and B.
4. **REVERSE ONLY:** B causes A, but A does not cause B.

### EVALUATION CRITERIA for "NO EDGE"
- **Identify Confounders:** If A and B are symptoms of the same root cause (e.g., Heat causing Ice Cream and Drowning), the probability of "No Edge" is **1.0**.
- **Identify Mediators:** If the metadata suggests a variable C exists that carries the entire causal load, the probability of "No Edge" between A and B is **HIGH**.
- **Check for Proxies:** If A is a "Paper Tiger" (a lagging indicator or label) that lacks a functional mechanism to change B, the probability of "No Edge" is **HIGH**.
- **Mechanism Failure:** If there is no logical, physical, or biological path for A to influence B, the probability of "No Edge" is **1.0**.

### INPUT
### METADATA
{input_json}

### TASK
- Evaluate every possible ordered pair of distinct variables in the metadata.
- For each ordered pair, provide only:
  - the probability that the statement "A does NOT directly cause B" is true
- Use the full variable set to reason about independence, confounding, mediation, reverse-only direction, and proxy variables.

"""
