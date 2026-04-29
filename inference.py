import os
import json
from itertools import permutations
from openai import OpenAI


def get_client():
    """Get OpenRouter client."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def build_prompt(description: dict, query_mode: str) -> str:
    """Build the prompt for causal edge queries."""
    field = description.get("field", "general")
    context = description.get("context", "")
    variables = description["variables"]

    var_descriptions = "\n".join(
        f"- {v['name']}: {v['description']}" for v in variables
    )
    var_names = [v["name"] for v in variables]

    # Build all ordered pairs
    pairs = list(permutations(var_names, 2))
    pairs_str = "\n".join(f"- {a} -> {b}" for a, b in pairs)

    if query_mode == "edge":
        question = "Does A directly cause B?"
        instruction = (
            "For every ordered pair below, estimate the probability of a DIRECT causal relationship "
            "from the first variable to the second."
        )
        decision_notes = (
            "- Use the variable descriptions to distinguish direct causation from correlation, reverse causality, "
            "common causes, and fully mediated effects.\n"
            "- For each pair, return the probability that the direct edge A -> B is present."
        )
    else:  # no_edge
        question = "Is there NO direct causal relationship from A to B?"
        instruction = (
            "For every ordered pair below, estimate the probability that there is NO DIRECT causal relationship "
            "from the first variable to the second."
        )
        decision_notes = (
            "- Use the variable descriptions to identify independence, spurious correlation, reverse-only direction, "
            "or total mediation through other variables.\n"
            "- For each pair, return the probability that there is NO direct edge A -> B."
        )

    prompt = f"""You are an expert in {field} and causal reasoning.

Context: {context}

Variables:
{var_descriptions}

{instruction}

For each of the following directed pairs, answer: {question}

Pairs to evaluate:
{pairs_str}

Guidelines:
{decision_notes}

Respond in JSON format with this structure:
{{
  "A->B": {{
    "probability": 0.0-1.0
  }},
  ...
}}

Where:
- "probability" is the probability that the queried statement is true for that mode

Return ONLY valid JSON, no other text."""

    return prompt


def parse_llm_response(response_text: str, variables: list, query_mode: str) -> dict:
    """Parse LLM JSON response into edge results."""
    # Try to extract JSON from response
    try:
        # Handle potential markdown code blocks
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1])
        results = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            results = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

    # Normalize results
    var_names = [v["name"] for v in variables]
    normalized = {}

    for source, target in permutations(var_names, 2):
        edge_key = f"{source}->{target}"

        if edge_key in results:
            entry = results[edge_key]
            confidence = entry.get("probability", 0.5)

            normalized[edge_key] = {
                "confidence": confidence,
            }
        else:
            # Default if missing
            normalized[edge_key] = {
                "confidence": 0.5,
            }

    return normalized


def query_llm_for_edges(
    model_id: str,
    description: dict,
    query_mode: str,
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> dict:
    """Query LLM for causal edge predictions."""
    client = get_client()
    prompt = build_prompt(description, query_mode)

    print(f"Querying {model_id}...")

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    response_text = response.choices[0].message.content
    print(f"Received response ({len(response_text)} chars)")

    results = parse_llm_response(response_text, description["variables"], query_mode)
    return results
