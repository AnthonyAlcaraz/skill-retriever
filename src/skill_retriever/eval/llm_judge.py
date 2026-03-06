"""LLM-as-judge evaluation for component quality (EVAL-01).

Uses Claude to evaluate components on 5 criteria:
1. Clarity - Are instructions clear and unambiguous?
2. Correctness - Does it encode accurate, up-to-date practices?
3. Specificity - Does it target a concrete domain or workflow? (not vague)
4. Completeness - Does it cover what it claims to address?
5. Purpose alignment - Does it improve weak-domain comprehension OR encode
   workflow preferences? Generic project context belongs in CLAUDE.md, not skills.

Based on SkillsBench research: curated skills +16.2pp, self-generated 0pp,
2-3 focused modules optimal.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentMetadata

logger = logging.getLogger(__name__)


# System prompt for skill evaluation
SKILL_JUDGE_PROMPT = """You are an expert evaluator for Claude Code skill components.

Your task: rate how likely this component is to provide MEASURABLE improvement
when installed as a Claude Code skill/agent/hook/command.

Context from SkillsBench research:
- Curated skills improve task performance by +16.2 percentage points.
- Self-generated skills show 0pp improvement (no better than baseline).
- 2-3 focused modules is the optimal number per task.
- Quality matters far more than quantity.

Rate the component on 5 criteria (each 0.0-1.0):

1. **Clarity** (0-1): Are instructions clear and unambiguous?
   - 1.0 = crystal clear, zero room for misinterpretation
   - 0.0 = vague, contradictory, or confusing

2. **Correctness** (0-1): Does it encode accurate, up-to-date practices?
   - 1.0 = all advice is technically correct and current
   - 0.0 = contains outdated or wrong information

3. **Specificity** (0-1): Does it target a concrete domain or workflow?
   - 1.0 = laser-focused on a specific task/domain
   - 0.0 = generic advice that could apply to anything

4. **Completeness** (0-1): Does it cover what it claims to address?
   - 1.0 = thorough coverage of its stated scope
   - 0.0 = major gaps in coverage

5. **Purpose alignment** (0-1): Does it improve weak-domain comprehension
   OR encode workflow preferences?
   - 1.0 = fills a clear capability gap or encodes expert workflow
   - 0.0 = generic project context that belongs in CLAUDE.md, not a skill

Output JSON only, no markdown formatting:
{
  "clarity": 0.0-1.0,
  "correctness": 0.0-1.0,
  "specificity": 0.0-1.0,
  "completeness": 0.0-1.0,
  "purpose_alignment": 0.0-1.0,
  "reasoning": "Brief explanation of scores"
}"""


class SkillEvalResult(BaseModel):
    """Result of LLM-as-judge evaluation for a component."""

    component_id: str
    overall_score: float = Field(ge=0.0, le=1.0, description="Average of 5 criteria")
    clarity: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    purpose_alignment: float = Field(ge=0.0, le=1.0)
    reasoning: str
    eval_model: str


@dataclass
class LLMSkillJudge:
    """Evaluates component quality using LLM-as-judge.

    Usage:
        judge = LLMSkillJudge()
        result = await judge.evaluate(component_metadata)
    """

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1500
    _client: "anthropic.Anthropic | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize Anthropic client if API key available."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed, LLM evaluation unavailable")
        else:
            logger.info("ANTHROPIC_API_KEY not set, LLM skill evaluation disabled")

    @property
    def is_available(self) -> bool:
        """Check if LLM evaluation is available."""
        return self._client is not None

    async def evaluate(self, component: ComponentMetadata) -> SkillEvalResult | None:
        """Evaluate a component using LLM-as-judge.

        Args:
            component: The component metadata to evaluate.

        Returns:
            SkillEvalResult with scores, or None if LLM unavailable.
        """
        if not self.is_available:
            logger.debug("LLM evaluation unavailable, returning None")
            return None

        user_prompt = f"""Evaluate this Claude Code component:

**Name:** {component.name}
**Type:** {component.component_type.value}
**Description:** {component.description}
**Tags:** {', '.join(component.tags) if component.tags else 'None'}
**Source repo:** {component.source_repo}

## Content

```
{component.raw_content[:8000]}
```

Rate this component on the 5 criteria and return JSON."""

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SKILL_JUDGE_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            response_text = response.content[0].text
            data = self._parse_response(response_text)

            clarity = float(data.get("clarity", 0.0))
            correctness = float(data.get("correctness", 0.0))
            specificity = float(data.get("specificity", 0.0))
            completeness = float(data.get("completeness", 0.0))
            purpose_alignment = float(data.get("purpose_alignment", 0.0))

            # Clamp values to 0-1
            clarity = max(0.0, min(1.0, clarity))
            correctness = max(0.0, min(1.0, correctness))
            specificity = max(0.0, min(1.0, specificity))
            completeness = max(0.0, min(1.0, completeness))
            purpose_alignment = max(0.0, min(1.0, purpose_alignment))

            overall = (clarity + correctness + specificity + completeness + purpose_alignment) / 5.0

            return SkillEvalResult(
                component_id=component.id,
                overall_score=round(overall, 3),
                clarity=clarity,
                correctness=correctness,
                specificity=specificity,
                completeness=completeness,
                purpose_alignment=purpose_alignment,
                reasoning=data.get("reasoning", ""),
                eval_model=self.model,
            )

        except Exception as e:
            logger.exception("LLM skill evaluation failed: %s", e)
            return None

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response."""
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM evaluation response as JSON")
            return {
                "clarity": 0.0,
                "correctness": 0.0,
                "specificity": 0.0,
                "completeness": 0.0,
                "purpose_alignment": 0.0,
                "reasoning": "Parse error",
            }
