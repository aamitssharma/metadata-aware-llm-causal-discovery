from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


SYSTEM = """You are a causal-graph labeling assistant.
Use only the provided prompt instructions, metadata, and optional sampled data.
Return a valid structured response that matches the requested schema."""

PROMPT_ROOT = Path(__file__).with_name("prompts")

PROMPT_FILE_MAP = {
    ("metaData", "vanilla"): PROMPT_ROOT / "metaData" / "vanilla" / "base_prompt.py",
    ("metaData", "cot"): PROMPT_ROOT / "metaData" / "cot" / "prompt_reasoning.py",
    ("metaData", "cot5Shot"): PROMPT_ROOT / "metaData" / "cot5Shot" / "prompt_reasoning_cot_few_shots.py",
    ("dataMetaData", "vanilla"): PROMPT_ROOT / "dataMetaData" / "vanilla" / "base_prompt_data.py",
    ("dataMetaData", "cot"): PROMPT_ROOT / "dataMetaData" / "cot" / "prompt_reasoning_data.py",
    ("dataMetaData", "cot5Shot"): PROMPT_ROOT / "dataMetaData" / "cot5Shot" / "prompt_reasoning_cot_few_shots_data.py",
}


def available_prompt_styles(prompt_family: str) -> list[str]:
    return [
        prompt_style
        for family, prompt_style in PROMPT_FILE_MAP
        if family == prompt_family
    ]


def build_prompt(
    *,
    query_mode: str,
    schema: Dict[str, Any],
    prompt_family: str,
    prompt_style: str,
    attribute_a: str,
    attribute_b: str,
    input_csv: str = "",
) -> ChatPromptTemplate:
    prompt_module = _load_prompt_module(prompt_family, prompt_style)
    template = _pick_template(prompt_module, query_mode)
    rendered = _render_template(
        template,
        {
            "attribute_A": attribute_a,
            "attribute_B": attribute_b,
            "input_json": _schema_to_input_json(schema),
            "input_csv": input_csv,
        },
    )
    # Use concrete message objects so JSON braces inside the prompt examples
    # are treated as literal text rather than template variables.
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=rendered),
        ]
    )


def _load_prompt_module(prompt_family: str, prompt_style: str):
    prompt_path = PROMPT_FILE_MAP.get((prompt_family, prompt_style))
    if prompt_path is None:
        raise ValueError(f"Unsupported prompt selection: family={prompt_family}, style={prompt_style}")

    module_name = f"prompt_{prompt_family}_{prompt_style}"
    spec = spec_from_file_location(module_name, prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prompt file: {prompt_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_template(prompt_module, query_mode: str) -> str:
    if query_mode == "edge":
        for attr_name in ("EDGE_PROMPT", "EDGE_PROMPT_COT", "EDGE_PROMPT_COT_FEW_SHOTS"):
            if hasattr(prompt_module, attr_name):
                return getattr(prompt_module, attr_name)
    else:
        for attr_name in ("NO_EDGE_PROMPT", "NO_EDGE_PROMPT_COT", "NO_EDGE_PROMPT_COT_FEW_SHOTS"):
            if hasattr(prompt_module, attr_name):
                return getattr(prompt_module, attr_name)
    raise ValueError(f"Prompt file does not define a template for query_mode={query_mode}")


def _render_template(template: str, values: Dict[str, str]) -> str:
    # Use direct placeholder replacement so JSON braces inside prompt examples are preserved.
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _schema_to_input_json(schema: Dict[str, Any]) -> str:
    import json

    variables = [
        {"name": str(name), "description": str(desc)}
        for name, desc in (schema.get("variables", {}) or {}).items()
    ]
    obj = {
        "field": schema.get("dataset_field", ""),
        "context": schema.get("dataset_context", ""),
        "variables": variables,
    }
    return json.dumps(obj, indent=2)
