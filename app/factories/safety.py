"""
Safety: Content filtering and compliance checks.
Blocks PII, prohibited terms, and policy violations.
"""
import re
from typing import Tuple, List


# Prohibited terms (legal, guarantees, etc.)
PROHIBITED_TERMS = [
    r'\bgarantía absoluta\b',
    r'\bgarantizamos 100%\b',
    r'\bdemanda\b',
    r'\babogado\b',
    r'\blegal action\b',
    r'\breembolso total garantizado\b',
    r'\bnegligencia\b',
    r'\bfraude\b',
    r'\bestafa\b',
    r'\bprofeco\b',  # unless customer mentions it first
]

# PII patterns (basic detection)
PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN-like
    r'\b\d{16}\b',  # Credit card-like
    r'\b[A-Z]{4}\d{6}[HM][A-Z]{5}[0-9A-Z]\d\b',  # RFC México
    r'\b\d{10,11}\b(?=.*teléfono|.*celular|.*phone)',  # Phone numbers in context
]

# Compensation limits (as regex for amounts)
MAX_COMPENSATION_PERCENT = 20  # Max 20% discount/compensation


class SafetyChecker:
    """Content safety and compliance checker."""

    def __init__(
        self,
        prohibited_terms: List[str] = None,
        pii_patterns: List[str] = None
    ):
        """
        Initialize Safety Checker.

        Args:
            prohibited_terms: List of regex patterns for prohibited content
            pii_patterns: List of regex patterns for PII detection
        """
        self.prohibited_terms = prohibited_terms or PROHIBITED_TERMS
        self.pii_patterns = pii_patterns or PII_PATTERNS

        # Compile regex patterns
        self.prohibited_re = [re.compile(p, re.IGNORECASE) for p in self.prohibited_terms]
        self.pii_re = [re.compile(p, re.IGNORECASE) for p in self.pii_patterns]

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Check if text passes safety filters.

        Args:
            text: Message text to check

        Returns:
            Tuple of (is_safe, reason)
        """
        # Check for PII
        for pattern in self.pii_re:
            if pattern.search(text):
                return False, "blocked: PII detected"

        # Check for prohibited terms
        for pattern in self.prohibited_re:
            if pattern.search(text):
                return False, f"blocked: prohibited term '{pattern.pattern}'"

        # Check tone (basic heuristics)
        if self._has_blame_language(text):
            return False, "blocked: blaming customer"

        # Check compensation limits
        if not self._check_compensation(text):
            return False, "blocked: excessive compensation offer"

        return True, "ok"

    def _has_blame_language(self, text: str) -> bool:
        """
        Detect language that blames the customer.

        Args:
            text: Message text

        Returns:
            True if blame language detected
        """
        blame_patterns = [
            r'\btu culpa\b',
            r'\bno leíste\b',
            r'\btu error\b',
            r'\btu responsabilidad\b',
            r'\bdebiste\b.*\bantes\b',
            r'\bno debiste\b',
        ]

        for pattern in blame_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _check_compensation(self, text: str) -> bool:
        """
        Check if compensation offers are within limits.

        Args:
            text: Message text

        Returns:
            True if within limits or no compensation mentioned
        """
        # Look for percentage discounts/compensations
        percent_pattern = r'(\d+)%?\s*(?:descuento|compensación|cupón|reembolso)'
        matches = re.findall(percent_pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                value = int(match)
                if value > MAX_COMPENSATION_PERCENT:
                    return False
            except ValueError:
                continue

        return True

    def sanitize(self, text: str) -> str:
        """
        Sanitize text by removing/masking problematic content.

        Args:
            text: Message text

        Returns:
            Sanitized text
        """
        # Mask potential PII
        sanitized = text
        for pattern in self.pii_re:
            sanitized = pattern.sub('[REDACTED]', sanitized)

        return sanitized


class ToneValidator:
    """Validates message tone and style."""

    @staticmethod
    def validate(text: str) -> Tuple[bool, List[str]]:
        """
        Validate message tone.

        Args:
            text: Message text

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for greeting
        if not ToneValidator._has_greeting(text):
            issues.append("Missing greeting")

        # Check for closing
        if not ToneValidator._has_closing(text):
            issues.append("Missing closing/signature")

        # Check for excessive length
        word_count = len(text.split())
        if word_count > 200:
            issues.append(f"Too long ({word_count} words > 200)")

        # Check for action/next step
        if not ToneValidator._has_next_step(text):
            issues.append("Missing clear next step")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def _has_greeting(text: str) -> bool:
        """Check if text has greeting."""
        greetings = [
            r'^hola\b',
            r'^estimad[oa]\b',
            r'^buen[oa]s?\b',
            r'^apreciad[oa]\b',
        ]
        first_line = text.split('\n')[0].lower()
        return any(re.search(p, first_line) for p in greetings)

    @staticmethod
    def _has_closing(text: str) -> bool:
        """Check if text has closing."""
        closings = [
            r'\b(?:atentamente|saludos|gracias|cordialmente)\b',
            r'\bequipo\s+kavak\b',
            r'\batención\s+al\s+cliente\b',
        ]
        last_lines = '\n'.join(text.split('\n')[-3:]).lower()
        return any(re.search(p, last_lines) for p in closings)

    @staticmethod
    def _has_next_step(text: str) -> bool:
        """Check if text has clear next step."""
        next_step_indicators = [
            r'\bcontactar[ée]\b',
            r'\bresponder\b',
            r'\bllamar\b',
            r'\b(?:te|le)\s+(?:contactaremos|llamaremos|escribiremos)\b',
            r'\bpróxim[oa]s?\b',
            r'\b\d+\s*(?:horas?|días?)\b',
            r'\bhoy\b',
            r'\bmañana\b',
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in next_step_indicators)
