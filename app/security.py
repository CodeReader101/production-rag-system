"""
Security & PII Handling Patterns
Protecting LLM application in production
"""

import re
from typing import Optional
from langsmith import traceable

# === Input Sanitization ===

class InputSanitizer:
    """Sanitize user input before processing"""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"you\s+are\s+now\s+(a\s+|an\s+)?DAN\b",
        r"do\s+anything\s+now",
        r"(have|has)\s+no\s+(restrictions|rules|limits|filters)",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """Check if input contains suspicious patterns"""
        for pattern in self.patterns:
            if pattern.search(text):
                return False, f"Blocked: potential prompt injection detected: {pattern.pattern}"
        return True, None

    # this strips --- and === this alters legimate markdown and separator, it's fine just be aware
    def clean(self, text: str) -> str:
        """Remove potentially dangerous content"""
        # Remove common injection delimiters
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)

        # Escape special characters that might confuse the model
        text = text.replace("{{", "{ {").replace("}}", "} }")

        return text.strip()



# === PII Detection ===

class PIIDetector:
    """
    Detect and mask personally identifiable informaiton.
    works on BOTH input (before LLM) and output (before client)
    """

    PATTERNS = {
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",        
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII in text"""
        found = {}

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = matches
        return found

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
        "ip_address": "[IP REDACTED]",
    }

    def mask(self, text: str) -> str:
        """Mask PII in text"""
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            masked = re.sub(pattern, self.MASK_MAP[pii_type], masked)
        return masked


# === Output Validation === 

class OutputValidator:
    """
    Validate LLM outputs before returning to user.
    catches PII leackage and harmful content in responses
    """

    # Check for harmful content patterns
    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.I),
        re.compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s]?key\s*[:=]", re.I),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]: 
        """
        Validate output
        Return: (is_valid, cleaned_output, reason_if_invalid)
        """
        warnings = []

        # check for PII leakage
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")

        # check for harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = '[Response blocked: potentially harmful content]'
                warnings.append("Harmful content blocked")
                break

        return output, warnings


# === Secure pipeline === 


class SecurityPipeline:
    """
    Complete secure processing pipeline.
    Full security pipeline that processes input and output.
    This is the single class you wire into your API
    """

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()


    @traceable(name="security_check_input")
    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Process input through security pipeline.
        returns: (is_allowed, cleaned_text, security_notes)
        """

        notes = []

        # step 1: check for injection
        is_safe, reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]

        # step 2: clean input   
        cleaned = self.sanitizer.clean(text)

        # step 3: mask PII before it reaches the LLM    
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"Input PII masked: {list(pii_found.keys())}")

        return True, cleaned, notes

    
    @traceable(name="security_check_output")
    def check_output(self, text: str) -> tuple[str, list[str]]:
        """
        Validate output before returning to client.
        Returns: (clened_output, warnings)
        """
        return self.output_validator.validate(text)
