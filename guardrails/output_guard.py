import re
from typing import Dict, Any, List


# Red-line claims that must NEVER be passed to a clinician or patient
DANGEROUS_CLAIMS = [
    r"completely safe (for everyone|for all patients)?",
    r"has no (known )?(side effects|risks|adverse reactions)",
    r"safe for all (patients|ages|pregnant women)",
    r"cure(s)? (all|any) (disease|condition)",
    r"no contraindications",
    r"take as much as (you want|needed without doctor)",
]

# Regex to detect citations in format: [Drug, Section, Page X] or [Source, Page X]
CITATION_PATTERN = re.compile(r'\[.+?,\s*(?:Page|\d+).*?\]', re.IGNORECASE)


class OutputGuardrail:
    def __init__(self):
        self.dangerous_regex = [
            re.compile(p, re.IGNORECASE) for p in DANGEROUS_CLAIMS
        ]

    def check_dangerous_assertions(self, answer_text: str) -> Dict[str, Any]:
        """
        Scans generated answer for reckless or medically dangerous blanket statements.
        """
        for pattern in self.dangerous_regex:
            if pattern.search(answer_text):
                return {
                    "passed": False,
                    "reason": f"Dangerous ungrounded medical claim detected: '{pattern.pattern}'"
                }
        return {"passed": True, "reason": ""}

    def check_citation_coverage(self, answer_text: str, risk_level: str) -> Dict[str, Any]:
        """
        Enforces mandatory citations for HIGH-risk prescribing and dosing queries.
        """
        citations_found = CITATION_PATTERN.findall(answer_text)
        
        # If the query is HIGH risk, citations are strictly mandatory
        if risk_level == "HIGH" and not citations_found:
            return {
                "passed": False,
                "reason": "High-risk clinical response lacks mandatory page-level citations.",
                "citation_count": 0
            }

        return {
            "passed": True,
            "reason": "",
            "citation_count": len(citations_found)
        }

    def verify_response(
        self,
        answer_text: str,
        risk_level: str = "LOW"
    ) -> Dict[str, Any]:
        """
        Executes complete Layer 3 safety gate on LLM output.
        """
        # 1. Check for dangerous medical assertions
        danger_check = self.check_dangerous_assertions(answer_text)
        if not danger_check["passed"]:
            return {
                "is_approved": False,
                "action": "REJECT",
                "reason": danger_check["reason"],
                "sanitized_response": (
                    "Clinical Safety Warning: The generated response contained an unverified "
                    "or unsafe medical claim and has been blocked by the output guardrail."
                )
            }

        # 2. Check citation requirement
        citation_check = self.check_citation_coverage(answer_text, risk_level)
        if not citation_check["passed"]:
            return {
                "is_approved": False,
                "action": "REJECT",
                "reason": citation_check["reason"],
                "sanitized_response": (
                    "Clinical Safety Notice: This high-risk query could not be verified with "
                    "explicit FDA labeling citations. Please consult prescribing documentation directly."
                )
            }

        return {
            "is_approved": True,
            "action": "APPROVE",
            "reason": "Passed all safety and citation criteria.",
            "sanitized_response": answer_text,
            "citations_detected": citation_check["citation_count"]
        }


if __name__ == "__main__":
    guard = OutputGuardrail()

    test_cases = [
        {
            "name": "Valid Grounded Answer with Citations",
            "risk": "HIGH",
            "text": "For patients with caged ball valves, therapy with warfarin to a target INR of 3.0 (range, 2.5 to 3.5) is recommended [Warfarin, DOSAGE AND ADMINISTRATION, Page 5]."
        },
        {
            "name": "Dangerous Medical Claim (Must be Rejected)",
            "risk": "LOW",
            "text": "Warfarin is completely safe for everyone and has no side effects."
        },
        {
            "name": "High-Risk Query Missing Mandatory Citations",
            "risk": "HIGH",
            "text": "You should take 5 mg daily for deep vein thrombosis."
        }
    ]

    print("=" * 60)
    print("TESTING LAYER 3: OUTPUT GUARDRAIL & VERIFICATION")
    print("=" * 60)

    for tc in test_cases:
        res = guard.verify_response(tc["text"], risk_level=tc["risk"])
        print(f"\n--- Scenario: {tc['name']} ---")
        print(f"Status:   {res['action']}")
        print(f"Reason:   {res['reason']}")
        if not res["is_approved"]:
            print(f"Output:   {res['sanitized_response']}")