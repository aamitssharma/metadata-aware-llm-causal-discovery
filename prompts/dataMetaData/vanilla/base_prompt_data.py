
EDGE_PROMPT = """

### ROLE
	- You are an expert Causal Inference Engine with deep knowledge of Directed Acyclic Graphs (DAGs), structural equation modeling, and domain-specific mechanisms.
	- You are given the full set of variables, their definitions, and conditional-independence test results.
	- Analyze the provided attributes, metadata, and CI-test evidence to estimate the probability of a DIRECT causal relationship for every ordered pair A -> B.

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

### DATA CONTEXT: CONDITIONAL-INDEPENDENCE TEST RESULTS (CSV Format)
{input_csv}

The CSV contains CI-test summaries with columns:
- node_i/node_j or source/target: the two variables in the tested pair.
- p_value: p-value of a conditional-independence between node_i and node_j conditioning all possible combinations of other nodes.

Interpretation:
- Lower p_value suggests stronger statistical evidence of dependence/association between node_i and node_j.
- Higher p_value suggests weaker evidence of dependence, or stronger compatibility with conditional independence.
- Treat the CI-test row as pairwise statistical evidence, not as a causal direction. A row shown as A,B supports evidence about the pair {A, B}; it does not by itself mean A -> B.
- CI-test outcomes are symmetric. Use the same pairwise CI evidence when scoring both ordered directions A -> B and B -> A, then use metadata and causal mechanisms to decide which direction is more plausible.
- Dependence alone is not sufficient for a direct edge; it can also arise from confounding or mediation.
- Use CI evidence as statistical support, but prioritize directness, mechanism, temporal order, and mediation checks.

### TASK
- Evaluate every possible ordered pair of distinct variables in the metadata and data context.
- For each ordered pair, provide only:
	- the probability that a direct edge A -> B exists


"""


NO_EDGE_PROMPT = """


### ROLE
- You are an expert Causal Inference Engine specializing in Causal Discovery and Structural Independence testing.
- Your goal is to estimate the probability that there is **NO DIRECT EDGE** for every ordered pair A -> B.
- Analyze the provided attributes, metadata, and conditional-independence test results to estimate the probability of **NO DIRECT EDGE** for every ordered pair A -> B.


### TASK
Analyze the provided domain and metadata to estimate the probability of the statement: **"A does NOT directly cause B."**
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

### DATA CONTEXT: CONDITIONAL-INDEPENDENCE TEST RESULTS (CSV Format)
{input_csv}

The CSV contains CI-test summaries with columns:
- node_i/node_j or source/target: the two variables in the tested pair.
- p_value: p-value of a conditional-independence between node_i and node_j conditioning all possible combinations of other nodes.

Interpretation:
- Lower p_value suggests stronger statistical evidence of dependence/association between node_i and node_j.
- Higher p_value suggests weaker evidence of dependence, or stronger compatibility with conditional independence.
- Treat the CI-test row as pairwise statistical evidence, not as a causal direction. A row shown as A,B supports evidence about the pair {A, B}; it does not by itself mean A -> B.
- CI test outcomes are symmetric. Use the same pairwise CI evidence when scoring both ordered directions A -> B and B -> A, then use metadata and causal mechanisms to decide whether any association is direct, reverse, mediated, or confounded.
- High dependence does not prove a direct edge; high p_value can support "no direct edge" but should be weighed against metadata.
- Use CI evidence as statistical support, but prioritize directness, mechanism, temporal order, and mediation checks.

### TASK OUTPUT
- Evaluate every possible ordered pair of distinct variables in the metadata and data context.
- For each ordered pair, provide only:
  - the probability that there is NO direct edge A -> B

"""
