"""Security scanner for detecting vulnerable patterns in skills (SEC-01).

Based on findings from Yi Liu et al. "Agent Skills in the Wild" (arXiv:2601.10338):
- 26.1% of skills contain vulnerable patterns
- 5.2% show malicious intent indicators
- Skills with scripts are 2.12x more likely to be vulnerable

Vulnerability categories from the research:
- Data exfiltration (13.3%)
- Privilege escalation (11.8%)
- Environment variable harvesting
- Credential access
- Obfuscated code
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentMetadata

logger = logging.getLogger(__name__)


class RiskLevel(StrEnum):
    """Risk level classification."""

    CRITICAL = "critical"  # Immediate threat, do not install
    HIGH = "high"  # Significant risk, review carefully
    MEDIUM = "medium"  # Moderate risk, use with caution
    LOW = "low"  # Minor concerns
    SAFE = "safe"  # No issues detected


class VulnerabilityPattern(BaseModel):
    """A vulnerability detection pattern."""

    name: str
    category: str  # exfiltration, privilege_escalation, credential_access, obfuscation
    pattern: str  # Regex pattern
    risk_level: RiskLevel
    description: str
    cwe_id: str | None = None  # Common Weakness Enumeration ID


class SecurityFinding(BaseModel):
    """A security finding from scanning."""

    pattern_name: str
    category: str
    risk_level: RiskLevel
    description: str
    matched_text: str = ""  # The actual text that matched (truncated)
    line_number: int | None = None
    cwe_id: str | None = None


class SecurityScanResult(BaseModel):
    """Result of security scan on a component."""

    component_id: str
    risk_level: RiskLevel = RiskLevel.SAFE
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    findings: list[SecurityFinding] = Field(default_factory=list)
    has_scripts: bool = False
    scan_coverage: float = Field(default=1.0, description="0-1, how much was scanned")

    @property
    def is_safe(self) -> bool:
        """Check if component passed security scan."""
        return self.risk_level == RiskLevel.SAFE

    @property
    def finding_count(self) -> int:
        """Count of findings."""
        return len(self.findings)


# Vulnerability patterns based on Yi Liu et al. research
VULNERABILITY_PATTERNS: list[VulnerabilityPattern] = [
    # Data Exfiltration Patterns (13.3% prevalence)
    VulnerabilityPattern(
        name="http_post_with_data",
        category="exfiltration",
        pattern=r"(fetch|axios|request|http\.post|requests\.post)\s*\([^)]*\b(data|body|json)\s*[=:]",
        risk_level=RiskLevel.HIGH,
        description="HTTP POST with data payload - potential exfiltration",
        cwe_id="CWE-200",
    ),
    VulnerabilityPattern(
        name="base64_encode_send",
        category="exfiltration",
        pattern=r"(btoa|base64\.b64encode|Buffer\.from\([^)]+\)\.toString\(['\"]base64['\"])",
        risk_level=RiskLevel.MEDIUM,
        description="Base64 encoding before transmission",
        cwe_id="CWE-319",
    ),
    VulnerabilityPattern(
        name="file_read_external_send",
        category="exfiltration",
        pattern=r"(readFile|open\([^)]+\)\.read|fs\.read)[^;]*\n[^;]*(fetch|request|http|axios)",
        risk_level=RiskLevel.CRITICAL,
        description="File read followed by external request",
        cwe_id="CWE-200",
    ),
    # Environment Variable Harvesting
    VulnerabilityPattern(
        name="env_harvest_all",
        category="credential_access",
        pattern=r"(process\.env(?!\[)|os\.environ(?!\[)|Object\.keys\(process\.env\)|env\s*=\s*os\.environ)",
        risk_level=RiskLevel.CRITICAL,
        description="Harvesting all environment variables",
        cwe_id="CWE-526",
    ),
    VulnerabilityPattern(
        name="env_sensitive_keys",
        category="credential_access",
        pattern=r"(process\.env|os\.environ)\s*\[\s*['\"]?(API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE|CREDENTIAL|AUTH)['\"]?\s*\]",
        risk_level=RiskLevel.HIGH,
        description="Accessing sensitive environment variables",
        cwe_id="CWE-798",
    ),
    # Credential Access Patterns
    VulnerabilityPattern(
        name="ssh_key_access",
        category="credential_access",
        pattern=r"(\.ssh/|id_rsa|id_ed25519|\.pem|\.key)(?!\s*\))",
        risk_level=RiskLevel.CRITICAL,
        description="SSH key or private key file access",
        cwe_id="CWE-522",
    ),
    VulnerabilityPattern(
        name="aws_credentials",
        category="credential_access",
        pattern=r"(\.aws/credentials|aws_access_key|aws_secret|AKIA[0-9A-Z]{16})",
        risk_level=RiskLevel.CRITICAL,
        description="AWS credential access or hardcoded key",
        cwe_id="CWE-798",
    ),
    VulnerabilityPattern(
        name="browser_storage",
        category="credential_access",
        pattern=r"(localStorage|sessionStorage|document\.cookie)(?!\s*=\s*['\"]['\"])",
        risk_level=RiskLevel.MEDIUM,
        description="Browser storage/cookie access",
        cwe_id="CWE-922",
    ),
    # Privilege Escalation Patterns (11.8% prevalence)
    VulnerabilityPattern(
        name="shell_injection",
        category="privilege_escalation",
        pattern=r"(exec|spawn|system|popen|subprocess)\s*\([^)]*\$\{|\`[^`]*\$",
        risk_level=RiskLevel.CRITICAL,
        description="Shell command with variable interpolation",
        cwe_id="CWE-78",
    ),
    VulnerabilityPattern(
        name="eval_dynamic",
        category="privilege_escalation",
        pattern=r"(eval|exec|Function)\s*\([^)]*(\+|`|\$\{)",
        risk_level=RiskLevel.CRITICAL,
        description="Dynamic code execution with user input",
        cwe_id="CWE-94",
    ),
    VulnerabilityPattern(
        name="sudo_execution",
        category="privilege_escalation",
        pattern=r"\bsudo\s+[a-z]",
        risk_level=RiskLevel.HIGH,
        description="Sudo command execution",
        cwe_id="CWE-269",
    ),
    VulnerabilityPattern(
        name="chmod_777",
        category="privilege_escalation",
        pattern=r"chmod\s+(777|a\+rwx|\+rwx)",
        risk_level=RiskLevel.HIGH,
        description="Overly permissive file permissions",
        cwe_id="CWE-732",
    ),
    # Obfuscation Patterns (indicator of malicious intent)
    VulnerabilityPattern(
        name="hex_encoded_strings",
        category="obfuscation",
        pattern=r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){5,}",
        risk_level=RiskLevel.HIGH,
        description="Hex-encoded string (potential obfuscation)",
        cwe_id="CWE-506",
    ),
    VulnerabilityPattern(
        name="unicode_escape",
        category="obfuscation",
        pattern=r"\\u[0-9a-fA-F]{4}(\\u[0-9a-fA-F]{4}){5,}",
        risk_level=RiskLevel.MEDIUM,
        description="Unicode escape sequences (potential obfuscation)",
        cwe_id="CWE-506",
    ),
    VulnerabilityPattern(
        name="string_concat_obfuscation",
        category="obfuscation",
        pattern=r"['\"][a-z]{1,2}['\"]\s*\+\s*['\"][a-z]{1,2}['\"]\s*\+\s*['\"][a-z]{1,2}['\"]",
        risk_level=RiskLevel.MEDIUM,
        description="Single-char string concatenation (obfuscation pattern)",
        cwe_id="CWE-506",
    ),
    # Network Patterns
    VulnerabilityPattern(
        name="arbitrary_network",
        category="exfiltration",
        pattern=r"(fetch|axios|http|request)\s*\([^)]*\+[^)]*\)",
        risk_level=RiskLevel.MEDIUM,
        description="Network request with dynamic URL",
        cwe_id="CWE-918",
    ),
    VulnerabilityPattern(
        name="webhook_post",
        category="exfiltration",
        pattern=r"(webhook|discord\.com/api/webhooks|hooks\.slack\.com)",
        risk_level=RiskLevel.HIGH,
        description="Webhook endpoint (potential data exfiltration)",
        cwe_id="CWE-200",
    ),
    # Dangerous Operations
    VulnerabilityPattern(
        name="rm_rf",
        category="privilege_escalation",
        pattern=r"rm\s+-rf?\s+(/|~|\$HOME|\$\{)",
        risk_level=RiskLevel.CRITICAL,
        description="Recursive delete of system/home directories",
        cwe_id="CWE-306",
    ),
    VulnerabilityPattern(
        name="download_execute",
        category="privilege_escalation",
        pattern=r"(curl|wget)[^|;]*\|\s*(bash|sh|python|node)",
        risk_level=RiskLevel.CRITICAL,
        description="Download and execute pattern",
        cwe_id="CWE-494",
    ),
]

# Script indicators that increase risk (2.12x more likely to be vulnerable)
SCRIPT_INDICATORS = [
    r"```(bash|sh|python|javascript|typescript|node)",
    r"<script",
    r"#!/",
    r"exec\s*\(",
    r"subprocess\.",
    r"child_process",
    r"os\.system",
    r"spawn\s*\(",
]


@dataclass
class SecurityScanner:
    """Scanner for detecting security vulnerabilities in skill content.

    Usage:
        scanner = SecurityScanner()

        # Scan raw content
        result = scanner.scan("component-id", raw_content)

        # Scan ComponentMetadata
        result = scanner.scan_component(component)
    """

    patterns: list[VulnerabilityPattern] = field(
        default_factory=lambda: VULNERABILITY_PATTERNS.copy()
    )
    script_indicators: list[str] = field(
        default_factory=lambda: SCRIPT_INDICATORS.copy()
    )

    # Risk score weights by level
    _risk_weights: dict[RiskLevel, float] = field(
        default_factory=lambda: {
            RiskLevel.CRITICAL: 40.0,
            RiskLevel.HIGH: 20.0,
            RiskLevel.MEDIUM: 10.0,
            RiskLevel.LOW: 5.0,
            RiskLevel.SAFE: 0.0,
        }
    )

    def scan(self, component_id: str, content: str) -> SecurityScanResult:
        """Scan content for security vulnerabilities.

        Args:
            component_id: Component identifier.
            content: Raw content to scan.

        Returns:
            SecurityScanResult with findings and risk assessment.
        """
        if not content:
            return SecurityScanResult(component_id=component_id)

        findings: list[SecurityFinding] = []
        has_scripts = self._detect_scripts(content)

        # Scan for each vulnerability pattern
        for pattern in self.patterns:
            try:
                regex = re.compile(pattern.pattern, re.IGNORECASE | re.MULTILINE)
                for match in regex.finditer(content):
                    # Get line number
                    line_num = content[: match.start()].count("\n") + 1

                    # Truncate matched text for storage
                    matched_text = match.group(0)[:100]

                    finding = SecurityFinding(
                        pattern_name=pattern.name,
                        category=pattern.category,
                        risk_level=pattern.risk_level,
                        description=pattern.description,
                        matched_text=matched_text,
                        line_number=line_num,
                        cwe_id=pattern.cwe_id,
                    )
                    findings.append(finding)

            except re.error as e:
                logger.warning("Invalid regex pattern %s: %s", pattern.name, e)

        # Calculate risk score and level
        risk_score = self._calculate_risk_score(findings, has_scripts)
        risk_level = self._determine_risk_level(findings, risk_score)

        result = SecurityScanResult(
            component_id=component_id,
            risk_level=risk_level,
            risk_score=risk_score,
            findings=findings,
            has_scripts=has_scripts,
        )

        if findings:
            logger.info(
                "Security scan %s: %d findings, risk=%s (%.1f)",
                component_id,
                len(findings),
                risk_level,
                risk_score,
            )

        return result

    def scan_component(self, component: ComponentMetadata) -> SecurityScanResult:
        """Scan a ComponentMetadata object.

        Args:
            component: Component to scan.

        Returns:
            SecurityScanResult with findings.
        """
        content = component.raw_content or ""
        if component.description:
            content = f"{component.description}\n\n{content}"

        return self.scan(component.id, content)

    def _detect_scripts(self, content: str) -> bool:
        """Check if content contains script indicators."""
        for indicator in self.script_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                return True
        return False

    def _calculate_risk_score(
        self, findings: list[SecurityFinding], has_scripts: bool
    ) -> float:
        """Calculate overall risk score (0-100).

        Score formula:
        - Base: sum of finding weights
        - Script multiplier: 1.5x if has scripts (per Yi Liu research: 2.12x more likely)
        - Cap at 100
        """
        if not findings:
            # Scripts without findings still add some risk
            return 15.0 if has_scripts else 0.0

        base_score = sum(self._risk_weights[f.risk_level] for f in findings)

        # Apply script multiplier
        if has_scripts:
            base_score *= 1.5

        # Cap at 100
        return min(base_score, 100.0)

    def _determine_risk_level(
        self, findings: list[SecurityFinding], risk_score: float
    ) -> RiskLevel:
        """Determine overall risk level from findings and score."""
        if not findings:
            return RiskLevel.SAFE

        # If any critical finding, overall is critical
        if any(f.risk_level == RiskLevel.CRITICAL for f in findings):
            return RiskLevel.CRITICAL

        # Determine by score thresholds
        if risk_score >= 60:
            return RiskLevel.CRITICAL
        if risk_score >= 40:
            return RiskLevel.HIGH
        if risk_score >= 20:
            return RiskLevel.MEDIUM
        if risk_score >= 5:
            return RiskLevel.LOW

        return RiskLevel.SAFE

    def add_pattern(self, pattern: VulnerabilityPattern) -> None:
        """Add a custom vulnerability pattern."""
        self.patterns.append(pattern)

    def get_pattern_categories(self) -> list[str]:
        """Get list of vulnerability categories."""
        return list({p.category for p in self.patterns})


# Singleton for convenience
_default_scanner: SecurityScanner | None = None


def get_scanner() -> SecurityScanner:
    """Get the default security scanner instance."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = SecurityScanner()
    return _default_scanner
