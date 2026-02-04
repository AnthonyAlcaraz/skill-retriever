"""LLM-assisted security analysis for reducing false positives (SEC-02).

Uses Claude to analyze flagged findings and determine:
1. Whether the finding is a true positive or false positive
2. The actual intent behind the code pattern
3. Contextual risk assessment based on component purpose

This is an OPTIONAL layer on top of regex-based scanning (SEC-01).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from skill_retriever.security.scanner import SecurityFinding, SecurityScanResult

logger = logging.getLogger(__name__)


class FindingVerdict(StrEnum):
    """LLM verdict on a security finding."""

    TRUE_POSITIVE = "true_positive"  # Confirmed security concern
    FALSE_POSITIVE = "false_positive"  # Safe, not a real issue
    CONTEXT_DEPENDENT = "context_dependent"  # Depends on usage
    NEEDS_REVIEW = "needs_review"  # Cannot determine, human review needed


class LLMFindingAnalysis(BaseModel):
    """LLM analysis result for a single finding."""

    pattern_name: str
    verdict: FindingVerdict
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in verdict")
    reasoning: str = Field(description="Explanation of the verdict")
    mitigations: list[str] = Field(default_factory=list, description="Suggested mitigations if true positive")
    is_in_documentation: bool = Field(default=False, description="Pattern appears in docs/examples, not executable code")


class LLMSecurityAnalysis(BaseModel):
    """Complete LLM analysis for a component."""

    component_id: str
    original_risk_level: str
    adjusted_risk_level: str
    original_risk_score: float
    adjusted_risk_score: float
    finding_analyses: list[LLMFindingAnalysis]
    overall_assessment: str
    false_positive_count: int
    true_positive_count: int
    context_dependent_count: int


# System prompt for security analysis
SECURITY_ANALYZER_PROMPT = """You are a security analyst reviewing code patterns flagged by an automated scanner.

Your task is to analyze each finding and determine if it's a TRUE POSITIVE (real security concern) or FALSE POSITIVE (safe pattern incorrectly flagged).

Key considerations:
1. **Documentation vs Execution**: Code in markdown examples, comments, or documentation is typically safe - it's showing users how to use the tool, not executing malicious code.

2. **Intended Functionality**: A JWT library accessing JWT_SECRET from env is expected behavior, not a vulnerability. Context matters.

3. **Shell Variables in Bash Examples**: `$VAR` in markdown code blocks showing CLI usage (like `gh pr view $PR_NUMBER`) are documentation examples, not shell injection.

4. **Webhook Integrations**: Discord/Slack webhooks in a notification skill are legitimate integrations, not exfiltration.

5. **Legitimate Tool Usage**: Skills that help with git, deployment, or CI/CD will naturally interact with credentials and execute commands - that's their purpose.

For each finding, provide:
- verdict: true_positive, false_positive, context_dependent, or needs_review
- confidence: 0.0-1.0
- reasoning: Brief explanation
- is_in_documentation: true if the pattern is in markdown/comments, false if in executable code
- mitigations: If true_positive, suggest mitigations

Output JSON only, no markdown formatting."""


@dataclass
class LLMSecurityAnalyzer:
    """Analyzes security findings using LLM to reduce false positives.

    Usage:
        analyzer = LLMSecurityAnalyzer()
        result = await analyzer.analyze(scan_result, raw_content)
    """

    model: str = "claude-sonnet-4-20250514"  # Fast, cost-effective for analysis
    max_tokens: int = 2000
    _client: "anthropic.Anthropic | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize Anthropic client if API key available."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed, LLM analysis unavailable")
        else:
            logger.info("ANTHROPIC_API_KEY not set, LLM security analysis disabled")

    @property
    def is_available(self) -> bool:
        """Check if LLM analysis is available."""
        return self._client is not None

    async def analyze(
        self,
        scan_result: SecurityScanResult,
        raw_content: str,
        component_name: str = "",
        component_description: str = "",
    ) -> LLMSecurityAnalysis | None:
        """Analyze security findings using LLM.

        Args:
            scan_result: Result from regex-based security scanner
            raw_content: Full content of the component
            component_name: Name of the component for context
            component_description: Description for additional context

        Returns:
            LLMSecurityAnalysis with adjusted risk levels, or None if unavailable
        """
        if not self.is_available:
            logger.debug("LLM analysis unavailable, returning None")
            return None

        if not scan_result.findings:
            # No findings to analyze
            return LLMSecurityAnalysis(
                component_id=scan_result.component_id,
                original_risk_level=scan_result.risk_level.value,
                adjusted_risk_level=scan_result.risk_level.value,
                original_risk_score=scan_result.risk_score,
                adjusted_risk_score=scan_result.risk_score,
                finding_analyses=[],
                overall_assessment="No findings to analyze",
                false_positive_count=0,
                true_positive_count=0,
                context_dependent_count=0,
            )

        # Build the analysis prompt
        findings_text = self._format_findings(scan_result.findings)

        user_prompt = f"""Analyze these security findings for component: {component_name or scan_result.component_id}

Description: {component_description or "No description provided"}

## Findings to Analyze

{findings_text}

## Component Content (for context)

```
{raw_content[:8000]}  # Truncate to avoid token limits
```

Analyze each finding and return JSON with this structure:
{{
  "finding_analyses": [
    {{
      "pattern_name": "...",
      "verdict": "true_positive|false_positive|context_dependent|needs_review",
      "confidence": 0.0-1.0,
      "reasoning": "...",
      "is_in_documentation": true|false,
      "mitigations": ["..."] // only if true_positive
    }}
  ],
  "overall_assessment": "Brief summary of actual risk"
}}"""

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SECURITY_ANALYZER_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse response
            response_text = response.content[0].text
            analysis_data = self._parse_response(response_text)

            # Calculate adjusted scores
            finding_analyses = [
                LLMFindingAnalysis(**fa) for fa in analysis_data.get("finding_analyses", [])
            ]

            false_positive_count = sum(
                1 for fa in finding_analyses if fa.verdict == FindingVerdict.FALSE_POSITIVE
            )
            true_positive_count = sum(
                1 for fa in finding_analyses if fa.verdict == FindingVerdict.TRUE_POSITIVE
            )
            context_dependent_count = sum(
                1 for fa in finding_analyses if fa.verdict == FindingVerdict.CONTEXT_DEPENDENT
            )

            # Adjust risk score based on false positives
            adjusted_score = self._calculate_adjusted_score(
                scan_result.risk_score,
                len(scan_result.findings),
                false_positive_count,
                true_positive_count,
            )
            adjusted_level = self._score_to_level(adjusted_score)

            return LLMSecurityAnalysis(
                component_id=scan_result.component_id,
                original_risk_level=scan_result.risk_level.value,
                adjusted_risk_level=adjusted_level,
                original_risk_score=scan_result.risk_score,
                adjusted_risk_score=adjusted_score,
                finding_analyses=finding_analyses,
                overall_assessment=analysis_data.get("overall_assessment", ""),
                false_positive_count=false_positive_count,
                true_positive_count=true_positive_count,
                context_dependent_count=context_dependent_count,
            )

        except Exception as e:
            logger.exception("LLM security analysis failed: %s", e)
            return None

    def _format_findings(self, findings: list[SecurityFinding]) -> str:
        """Format findings for the LLM prompt."""
        lines = []
        for i, f in enumerate(findings, 1):
            lines.append(f"""### Finding {i}: {f.pattern_name}
- Category: {f.category}
- Risk Level: {f.risk_level}
- Description: {f.description}
- CWE: {f.cwe_id or "N/A"}
- Matched Text: `{f.matched_text[:200]}`
- Line Number: {f.line_number or "N/A"}
""")
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response."""
        import json

        # Try to extract JSON from response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {"finding_analyses": [], "overall_assessment": "Parse error"}

    def _calculate_adjusted_score(
        self,
        original_score: float,
        total_findings: int,
        false_positives: int,
        true_positives: int,
    ) -> float:
        """Calculate adjusted risk score based on LLM analysis.

        Formula:
        - Each false positive reduces score proportionally
        - True positives maintain their weight
        - Minimum score is 0 if all are false positives
        """
        if total_findings == 0:
            return original_score

        # Calculate the proportion that are actual issues
        actual_issues = true_positives + (total_findings - false_positives - true_positives) * 0.5
        issue_ratio = actual_issues / total_findings

        adjusted = original_score * issue_ratio
        return round(max(0.0, min(100.0, adjusted)), 1)

    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level string."""
        if score >= 60:
            return "critical"
        if score >= 40:
            return "high"
        if score >= 20:
            return "medium"
        if score >= 5:
            return "low"
        return "safe"


# Convenience function for one-off analysis
async def analyze_with_llm(
    scan_result: SecurityScanResult,
    raw_content: str,
    component_name: str = "",
    component_description: str = "",
) -> LLMSecurityAnalysis | None:
    """Convenience function for LLM security analysis.

    Args:
        scan_result: Result from SecurityScanner.scan()
        raw_content: Full content of the component
        component_name: Optional name for context
        component_description: Optional description for context

    Returns:
        LLMSecurityAnalysis or None if unavailable
    """
    analyzer = LLMSecurityAnalyzer()
    return await analyzer.analyze(
        scan_result, raw_content, component_name, component_description
    )
