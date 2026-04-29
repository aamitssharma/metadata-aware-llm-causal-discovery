EDGE_PROMPT_COT_FEW_SHOTS = """

### ROLE
- You are an expert Causal Inference Engine with deep knowledge of Directed Acyclic Graphs (DAGs) and structural equation modeling.
- Analyze the provided domain and metadata to estimate the probability of a DIRECT causal relationship for every ordered pair A -> B.

### TASK: STEP-BY-STEP LOGICAL DISCOVERY
Evaluate the probability of a direct edge using these specific logical steps:
1. **Correlation vs. Mechanism:** Is there a physical, biological, or logical "How"?
2. **Reverse Causality Check:** Could B be causing A, or is there a feedback loop?
3. **Common Cause Search:** Is there a third variable Z causing both (spurious correlation)?
4. **Proxy/Paper Tiger Test:** Is A just a lagging indicator or a "label" for an underlying trait?
5. **Adjacency Check:** Is the effect direct, or is it fully mediated by another variable (A -> C -> B)?

### FEW-SHOT EXAMPLES

Example 1
----------

Query: "Self-Confidence" -> "Success"

Reasoning: "Step-by-Step Logical Discovery 1. Establish Correlation Reasoning: Do successful people tend to be confident? Logic: Almost universally, yes. High achievers generally report high levels of self-esteem and confidence. Result: Very Strong Positive Correlation. 2. Identify the Theoretical Mechanism (The 'How') Reasoning: Why would a feeling create a result? Logic: The 'Self-Fulfilling Prophecy.' If you believe you can succeed, you are more likely to persist through failure, take necessary risks, and put in more effort. This 'grit' then leads to success. Result: Plausible Mechanism Identified. (This is why many people stop here and assume the claim is true). 3. Testing for Reverse Causality (The 'Mirror' Problem) Reasoning: Does Success (Y) cause Confidence (X)? Logic: Think about a student who gets an 'A' on a difficult math test. Does their confidence go up after the grade? Yes. If they win three more awards, their confidence grows further. Conclusion: It is highly likely that Success is the parent of Confidence, not just the other way around. This is a feedback loop, which makes isolating a single cause difficult. 4. Searching for the Confounder (The 'Innate Ability' Z) Reasoning: Is there a third factor that creates both the feeling and the result? Logic: Consider Actual Competence or Intelligence (Z). If a person is naturally gifted at a task (Z), they will perform well (Y). Because they perform well and know they are gifted, they feel confident (X). The 'Paper Tiger' Test: If you artificially inflate someone's confidence (e.g., through empty praise or participation trophies) without increasing their actual skill (Z), do they become more successful? Usually, no. In fact, overconfidence without competence often leads to worse results due to poor risk assessment. Verdict: Partially False (or 'Weakly Causal'). While confidence can act as a mediator (helping you work harder), it is rarely the primary cause. In most cases, confidence is a lagging indicator of previous success and underlying competence. If you remove the competence (Z), the confidence (X) fails to produce the success (Y)."

Probability: 0.25


Example 2
---------
Query: "Ice Cream Sales" -> "Drowning Incidents"

Reasoning: "Step-by-Step Logical Discovery 1. Establish Correlation Reasoning: Do these two things happen together? Logic: If we look at data from any coastal city, we will see a nearly perfect positive correlation. When the local ice cream shop has a record-breaking month, the local lifeguards also record a peak in rescues. Result: Strong Correlation Found. 2. Identify the Theoretical Mechanism (The 'How') Reasoning: Is there a physical or biological bridge? Logic: Does eating dairy products make a human body less buoyant? No. Does sugar intake cause a sudden inability to swim? No. Could the sight of an ice cream cone cause a person on a boat to fall overboard? Highly improbable. Result: No Plausible Mechanism. This is the first major red flag in our causal discovery. 3. The Counterfactual Test Reasoning: If we banned ice cream tomorrow, would people stop drowning? Logic: Imagine a city passes a law: 'No Ice Cream Sales.' People still go to the beach. The water is still deep. Currents are still strong. People who cannot swim will still struggle in the water. Result: The effect (Y) persists even when the 'cause' (X) is removed. Therefore, X is not a necessary cause. 4. Searching for the Confounder (The 'Lurking Variable') Reasoning: Is there a hidden variable (Z) that causes both X and Y? Logic: What changes when ice cream sales go up? The Weather or Temperature (Z). The Connection: 5. When it is hot (Z), more people buy ice cream (X). 6. When it is hot (Z), more people go swimming (Y). Conclusion: Z is the common cause or confounder. X and Y are spurious correlations. Decision: Is the Claim Correct? Verdict: False. The relationship is purely coincidental, driven by a third, unobserved variable (ambient temperature). In causal modeling, we would draw the graph like this: Temperature → Ice Cream Sales, Temperature → Drowning Incidents. Notice there is no direct connection between ice cream sales and drowning."

Probability: 0.0


Example 3
---------
Query: "Street Lights" -> "Crime Rate"

Reasoning: "Step-by-Step Logical Discovery 1. Identify the Theoretical Mechanism (The 'How') Reasoning: Why would light stop a crime? Logic: There are two main paths. First, deterrence: lighting increases the probability of detection (surveillance). A criminal is less likely to act if they can be seen by witnesses or cameras. Second, community pride: better lighting makes a street feel cared for, which encourages more foot traffic. More people on the street act as natural guardians. Result: The mechanism is logically sound and grounded in situational crime prevention theory. 2. The Counterfactual Test Reasoning: If a high-crime area is left in total darkness, does crime increase? Logic: If the claim X → Y is true, then removing X should increase Y. In many cases, darkness provides cover, making the counterfactual (no lights = more crime) logically consistent. Result: Passes the 'but-for' test in most urban settings. 3. Testing for Displacement (The Leakage Problem) Reasoning: Does the light reduce crime, or just move it? Logic: If one street is well lit, criminals may move to a darker nearby street. Result: If crime shifts locations, the claim is false at a macro level but true at a micro level. This distinction is important. 4. Ruling out Confounders (The 'Third Variable') Reasoning: Is there a variable Z driving both? Logic: Consider wealth or gentrification. Wealthier areas can afford better lighting and also tend to have lower crime due to other factors. Elimination: To prove causality, we need to observe cases where lighting is added without other changes. If crime drops, lighting is likely a true cause. Decision: Is the Claim Correct? Verdict: Conditionally True. The claim holds because lighting increases the risk of being caught. However, it may fail if lighting helps criminals see targets better or if crime simply shifts elsewhere."

Probability: 0.62


Example 4
---------
Query: "Employee Autonomy" -> "Workplace Innovation"

Reasoning: "Step-by-Step Logical Discovery 1. Identify the Theoretical Mechanism (The 'How') Reasoning: Why would freedom lead to new ideas? Logic: Innovation requires divergent thinking and risk-taking. If an employee is strictly managed (low autonomy), they focus on compliance and error avoidance. When autonomy is high, psychological safety increases, allowing individuals to experiment without fear of immediate reprimand for deviating from the norm. Result: There is a strong psychological bridge based on the Job Characteristics Model. 2. The Counterfactual Test Reasoning: If we have a creative team but remove their autonomy, does innovation still happen? Logic: Imagine a group of engineers forced to follow a rigid, step-by-step manual written years ago. Even if they are highly capable, the structure prevents the execution of new ideas. Result: Since innovation remains suppressed regardless of talent when autonomy is zero, autonomy is likely a necessary condition. 3. Testing for Reverse Causality Reasoning: Does innovation (Y) cause autonomy (X)? Logic: This is plausible. Managers tend to grant more freedom to employees who have already demonstrated innovative ability. In this case, innovation leads to autonomy. Result: This suggests a feedback loop rather than a simple one-way relationship. The claim is not false but incomplete. 4. Ruling out Confounders (The 'Third Variable') Reasoning: Is there a variable Z driving both? Logic: Consider company funding. A well-funded organization can afford to give employees more autonomy and also has the resources to implement innovative ideas. Elimination: To prove causality, we need to observe environments with limited resources. If autonomy still leads to increased innovation, then autonomy itself plays a causal role. Decision: Is the Claim Correct? Verdict: Likely True, but non-linear. The logic suggests that autonomy creates the environment necessary for innovation. However, there is a saturation point where too much autonomy without alignment can reduce effectiveness and execution."

Probability: 0.75

Example 5
---------
Query: "20-min afternoon nap" -> "Cognitive Performance"

Reasoning: "Step-by-Step Logical Discovery To decide if this claim is correct, we apply the causal criteria used in logic and philosophy. 1. Temporal Precedence (The Arrow of Time) Reasoning: For the nap to be the cause, the nap must happen before the increase in performance. Test: If performance is measured in the morning, it cannot be caused by a nap taken later in the day. Result: The timeline fits. X (nap) occurs before Y (performance). 2. Identifying the Biological Mechanism (The 'How') Reasoning: Is there a physical connection between sleep and brain function? Logic: Short naps help clear adenosine, a neurotransmitter that builds up during wakefulness and contributes to fatigue. A nap also provides brief sensory detachment, helping reset neural activity. Result: There is a known physiological mechanism, making the claim plausible. 3. The Counterfactual Test (The 'What If?') Reasoning: If someone is tired and does not take a nap, would performance improve anyway? Logic: In the absence of other interventions, cognitive performance typically declines during the afternoon due to the circadian dip. If performance improves only after the nap, the nap is the likely cause. Result: Improvement is unlikely without the nap. 4. Testing for Confounders (The 'Third Variable') Reasoning: Is there another variable influencing both the nap and performance? Logic: A good night’s sleep could influence both. Someone well-rested might perform better and also have time for a nap. Elimination: If a sleep-deprived person takes a nap and shows improvement, the nap itself is the causal factor. Decision: Is the Claim Correct? Verdict: True, with an intervention condition. Logic suggests that X leads to Y because actively taking a nap produces a measurable improvement in performance that cannot be explained by time alone or stable background factors."

Probability: 0.85

### INPUT
METADATA: {input_json}

### TASK OUTPUT
- Evaluate every possible ordered pair of distinct variables in the metadata.
- Use the examples only to guide your internal reasoning process.
- Do not output reasoning text.
- For each ordered pair, provide only:
  - the probability that a direct edge A -> B exists


"""


NO_EDGE_PROMPT_COT_FEW_SHOTS = """
### ROLE
- You are an expert Causal Inference Engine specializing in Causal Discovery and Structural Independence testing.
- Your goal is to logically prove why there might be **NO DIRECT EDGE** from A to B (A -/> B).

### TASK: STEP-BY-STEP INDEPENDENCE PROOF
Analyze the probability that A does NOT directly cause B by evaluating:
1. **Spuriousness:** Is the entire link explained by a common confounder?
2. **Total Mediation:** Is the relationship entirely indirect (A -> C -> B)?
3. **Independence:** Does an intervention on A fail to produce a change in B?
4. **Reverse Only:** Is the arrow exclusively flowing from B to A?

### FEW-SHOT EXAMPLES

Example 1
----------

Query: "Self-Confidence" -/> "Success"

Reasoning: "Step-by-Step Independence Proof 1. Establish Correlation Reasoning: Do successful people tend to be confident? Logic: Almost universally, yes. High achievers generally report high levels of self-esteem and confidence. Result: Strong correlation exists, so correlation alone cannot rule out a direct edge. 2. Test for Reverse Causality Reasoning: Does Success (Y) cause Confidence (X)? Logic: A student who earns repeated high grades often becomes more confident afterward. This means much of the observed relationship can run from Success to Confidence rather than from Confidence to Success. Result: Reverse causality is substantial. 3. Search for a Common Cause Reasoning: Is there a third factor that creates both? Logic: Actual competence or underlying ability can increase success directly and also produce confidence as a byproduct. If competence explains both, the direct edge from confidence to success weakens. Result: A strong confounder is plausible. 4. Proxy/Paper Tiger Test Reasoning: Does manipulating confidence alone reliably change success? Logic: Artificially boosting confidence without improving skill usually does not create lasting success and can even worsen decisions. Verdict: The evidence supports a high probability of no direct edge. Confidence is often a proxy or downstream indicator of competence and prior success rather than an independent direct cause of success."

No Edge Probability: 0.75


Example 2
---------
Query: "Ice Cream Sales" -/> "Drowning Incidents"

Reasoning: "Step-by-Step Logical Discovery 1. Establish Correlation Reasoning: Do these two things happen together? Logic: If we look at data from any coastal city, we will see a nearly perfect positive correlation. When the local ice cream shop has a record-breaking month, the local lifeguards also record a peak in rescues. Result: Strong Correlation Found. 2. Identify the Theoretical Mechanism (The 'How') Reasoning: Is there a physical or biological bridge? Logic: Does eating dairy products make a human body less buoyant? No. Does sugar intake cause a sudden inability to swim? No. Could the sight of an ice cream cone cause a person on a boat to fall overboard? Highly improbable. Result: No Plausible Mechanism. This is the first major red flag in our causal discovery. 3. The Counterfactual Test Reasoning: If we banned ice cream tomorrow, would people stop drowning? Logic: Imagine a city passes a law: 'No Ice Cream Sales.' People still go to the beach. The water is still deep. Currents are still strong. People who cannot swim will still struggle in the water. Result: The effect (Y) persists even when the 'cause' (X) is removed. Therefore, X is not a necessary cause. 4. Searching for the Confounder (The 'Lurking Variable') Reasoning: Is there a hidden variable (Z) that causes both X and Y? Logic: What changes when ice cream sales go up? The Weather or Temperature (Z). The Connection: 5. When it is hot (Z), more people buy ice cream (X). 6. When it is hot (Z), more people go swimming (Y). Conclusion: Z is the common cause or confounder. X and Y are spurious correlations. Decision: Is the Claim Correct? Verdict: False. The relationship is purely coincidental, driven by a third, unobserved variable (ambient temperature). In causal modeling, we would draw the graph like this: Temperature → Ice Cream Sales, Temperature → Drowning Incidents. Notice there is no direct connection between ice cream sales and drowning."

No Edge Probability: 1.0


Example 3
---------
Query: "Street Lights" -/> "Crime Rate"

Reasoning: "Step-by-Step Independence Proof 1. Identify the Mechanism Reasoning: Why might light affect crime? Logic: Better lighting can deter offenders and increase surveillance, so a direct mechanism is plausible. Result: This weakens the no-edge hypothesis. 2. Check for Total Mediation or Displacement Reasoning: Does lighting reduce total crime, or merely move it elsewhere? Logic: If crime is displaced to nearby darker areas, then the apparent local effect does not imply a clean direct reduction at the broader system level. Result: Some observed effects may be indirect or context-dependent. 3. Check for Confounders Reasoning: Is there a third variable driving both? Logic: Wealth, neighborhood investment, or gentrification can produce both better lighting and lower crime. Result: A confounded explanation is plausible in some settings. 4. Final Judgment Reasoning: Do these arguments prove the absence of a direct edge? Logic: Not fully. There is still a credible direct mechanism from lighting to local crime reduction. Verdict: The probability of no direct edge is below 0.5 because a direct causal pathway remains plausible, even though confounding and displacement prevent high confidence in a clean edge."

No Edge Probability: 0.38


Example 4
---------
Query: "Employee Autonomy" -/> "Workplace Innovation"

Reasoning: "Step-by-Step Independence Proof 1. Identify the Mechanism Reasoning: Why might autonomy affect innovation? Logic: Autonomy supports experimentation, risk-taking, and psychological safety, all of which can directly contribute to innovation. Result: A direct mechanism is plausible. 2. Check Reverse Causality Reasoning: Could innovation lead to autonomy instead? Logic: Managers often grant more freedom to employees who already show innovative performance. Result: Reverse causality is plausible and can explain part of the correlation. 3. Check for Confounders Reasoning: Is there a third factor driving both? Logic: Company resources, leadership quality, and culture can increase both autonomy and innovation. Result: Confounding is possible. 4. Final Judgment Reasoning: Do reverse causality and confounding eliminate the direct edge? Logic: Not entirely, because autonomy still has a strong functional pathway to innovation. Verdict: The probability of no direct edge should remain low. The relationship may be noisy or bidirectional, but the evidence does not support ruling out a direct effect from autonomy to innovation."

No Edge Probability: 0.25

Example 5
---------
Query: "20-min afternoon nap" -/> "Cognitive Performance"

Reasoning: "Step-by-Step Independence Proof 1. Temporal Order Reasoning: Could performance come before the nap? Logic: No. The nap occurs before the later performance measurement, so reverse causality is weak. Result: Temporal order does not support the no-edge hypothesis. 2. Check for a Mechanism Reasoning: Is there a physical pathway from nap to performance? Logic: Short naps can reduce fatigue and improve alertness through known sleep-related mechanisms. Result: A direct mechanism is plausible. 3. Check for Confounders Reasoning: Could another variable explain both? Logic: Baseline sleep quality can affect both nap-taking and later performance, but it does not fully eliminate the possibility that the nap itself has an effect. Result: Confounding may exist, but it is incomplete. 4. Final Judgment Reasoning: Does the evidence support no direct edge? Logic: No. The nap is an actionable intervention with a credible near-term physiological effect on cognition. Verdict: The probability of no direct edge is low because the evidence still supports a plausible direct causal effect from the nap to cognitive performance."

No Edge Probability: 0.15

### INPUT
METADATA: {input_json}

### TASK OUTPUT
- Evaluate every possible ordered pair of distinct variables in the metadata.
- Use the examples only to guide your internal reasoning process.
- Do not output reasoning text.
- For each ordered pair, provide only:
  - the probability that there is NO direct edge A -> B
"""
