"""
Multi-Agent Insurance Claims Processing System
================================================
A LangGraph pipeline of 6 specialised AI agents that process insurance claims
through intake, policy validation, fraud detection, routing, decision-making,
and customer communication.

Uses Grok (xAI) via langchain-xai when XAI_API_KEY is set, with deterministic
rule-based fallbacks for fully offline operation.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Mock policy database
# ---------------------------------------------------------------------------
MOCK_POLICIES: dict[str, dict[str, Any]] = {
    "POL-10234": {
        "holder": "Jane Doe",
        "status": "active",
        "coverage_type": "auto",
        "coverage_limit": 10000,
        "deductible": 500,
    },
    "POL-20456": {
        "holder": "John Smith",
        "status": "active",
        "coverage_type": "home",
        "coverage_limit": 50000,
        "deductible": 1000,
    },
    "POL-30789": {
        "holder": "Alice Johnson",
        "status": "lapsed",
        "coverage_type": "auto",
        "coverage_limit": 15000,
        "deductible": 750,
    },
    "POL-40100": {
        "holder": "Bob Williams",
        "status": "active",
        "coverage_type": "health",
        "coverage_limit": 25000,
        "deductible": 200,
    },
    "POL-50321": {
        "holder": "Clara Davis",
        "status": "active",
        "coverage_type": "auto",
        "coverage_limit": 5000,
        "deductible": 300,
    },
}

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class ClaimState(TypedDict, total=False):
    # Raw input
    raw_text: str
    file_path: str

    # IntakeAgent output
    claimant_name: str
    policy_number: str
    claim_type: str          # auto / home / health
    incident_date: str
    claim_amount: float
    description: str

    # PolicyValidationAgent output
    policy_found: bool
    policy_status: str
    coverage_valid: bool
    coverage_limit: float
    policy_notes: str

    # FraudDetectionAgent output
    fraud_score: int
    red_flags: list[str]
    fraud_reasoning: str

    # Routing result
    route: str               # "auto_approve" | "escalate"

    # DecisionAgent / SeniorReviewAgent output
    decision_status: str     # "approved" | "escalated" | "denied"
    decision_reason: str

    # CommunicationAgent output
    communication_letter: str

    # Trace log (every agent appends here)
    trace: list[str]

    # Added for exclusion checking and app form state
    exclusion_check: dict
    treatment_date: str
    coverage_start_date: str
    benefit_type: str
    submission_deadline_exceeded: bool
    structured_claim: dict


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_llm():
    """Return a ChatXAI instance if XAI_API_KEY is available, else None."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_xai import ChatXAI
        return ChatXAI(
            model="grok-3-fast",
            temperature=0,
            xai_api_key=api_key,
        )
    except Exception:
        return None


def _llm_call(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Attempt an LLM call; return response text or None on any failure."""
    llm = _get_llm()
    if llm is None:
        return None
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = llm.invoke(msgs)
        return resp.content
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. IntakeAgent
# ---------------------------------------------------------------------------

def _fallback_intake(raw_text: str) -> dict[str, Any]:
    """Deterministic regex-based extraction from raw claim text."""
    text = raw_text

    # Claimant name
    name_match = re.search(
        r"[Cc]laimant\s*[:;-]\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)", text
    )
    claimant_name = name_match.group(1).strip() if name_match else "Unknown"

    # Policy number
    pol_match = re.search(r"(POL-\d{4,6})", text, re.IGNORECASE)
    policy_number = pol_match.group(1).upper() if pol_match else "UNKNOWN"

    # Claim type
    claim_type = "auto"
    lower = text.lower()
    if "home" in lower or "property" in lower or "house" in lower:
        claim_type = "home"
    elif "health" in lower or "medical" in lower or "hospital" in lower:
        claim_type = "health"

    # Incident date
    date_match = re.search(
        r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})", text
    )
    incident_date = date_match.group(1) if date_match else "Unknown"

    # Claim amount
    amount_match = re.search(r"\$\s?([\d,]+(?:\.\d{2})?)", text)
    claim_amount = 0.0
    if amount_match:
        claim_amount = float(amount_match.group(1).replace(",", ""))

    # Description: everything after common keywords, or last sentence
    desc_match = re.search(
        r"(?:description|details|incident|happened|damage|diagnosis)[:;\s-]*(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if desc_match:
        description = desc_match.group(1).strip()[:300]
    else:
        # Take the last sentence as description
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        description = sentences[-1] if sentences else text[:300]

    return {
        "claimant_name": claimant_name,
        "policy_number": policy_number,
        "claim_type": claim_type,
        "incident_date": incident_date,
        "claim_amount": claim_amount,
        "description": description,
    }


def intake_agent(state: ClaimState) -> dict:
    """Extract structured fields from raw claim text or a scanned receipt PDF."""
    raw = state.get("raw_text", "")
    file_path = state.get("file_path")
    trace = list(state.get("trace", []))
    
    result = None

    if file_path:
        try:
            from receipt_vision import is_scanned_pdf, pdf_to_images, extract_receipt_via_vision
            if is_scanned_pdf(file_path):
                trace.append(f"[IntakeAgent] Detected scanned PDF receipt: {file_path}")
                images = pdf_to_images(file_path)
                vision_result = extract_receipt_via_vision(images)
                
                # Map vision_result to standard structured_claim
                result = {
                    "claimant_name": vision_result.get("patient_name", "Unknown"),
                    "policy_number": "UNKNOWN", # Still need manual entry for this
                    "claim_type": "health",
                    "incident_date": vision_result.get("treatment_date", "Unknown"),
                    "claim_amount": vision_result.get("amount_charged", 0.0),
                    "description": vision_result.get("diagnosis_or_treatment_description", "Medical Receipt"),
                }
                
                method = vision_result.get("method", "vision")
                trace.append(f"[IntakeAgent] Used {method} extraction from receipt")
            else:
                trace.append(f"[IntakeAgent] PDF is text-based. Ignoring vision pipeline.")
        except Exception as e:
            trace.append(f"[IntakeAgent] Vision pipeline error: {e}")

    if result is None:
        trace.append(f"[IntakeAgent] Processing raw claim ({len(raw)} chars)")
        system_prompt = (
            "You are an insurance claim intake specialist. Extract the following fields "
            "from the raw claim text and return ONLY a valid JSON object with these keys: "
            "claimant_name, policy_number, claim_type (one of: auto, home, health), "
            "incident_date, claim_amount (number), description. "
            "Do NOT include any text outside the JSON."
        )

        llm_response = _llm_call(system_prompt, raw)
        if llm_response:
            try:
                # Strip markdown fences if present
                cleaned = re.sub(r"```(?:json)?\s*", "", llm_response).strip().rstrip("`")
                result = json.loads(cleaned)
                trace.append("[IntakeAgent] LLM extraction successful")
            except (json.JSONDecodeError, KeyError):
                trace.append("[IntakeAgent] LLM response parse failed, using fallback")
                result = None

        if result is None:
            result = _fallback_intake(raw)
            trace.append("[IntakeAgent] Used rule-based fallback extraction")

    # Merge with manual form fields if provided in state (from structured_claim)
    manual_fields = state.get("structured_claim", {})
    if manual_fields:
        for k, v in manual_fields.items():
            if v and (result.get(k) in [None, "Unknown", "UNKNOWN", 0, 0.0, ""]):
                result[k] = v

    # Normalise claim_amount to float
    try:
        if result["claim_amount"] is None:
            result["claim_amount"] = 0.0
        else:
            result["claim_amount"] = float(result["claim_amount"])
    except (ValueError, TypeError):
        result["claim_amount"] = 0.0

    trace.append(f"[IntakeAgent] Extracted: name={result.get('claimant_name')}, "
                 f"policy={result.get('policy_number')}, type={result.get('claim_type')}, "
                 f"amount=${result.get('claim_amount')}")

    return {**result, "trace": trace}


# ---------------------------------------------------------------------------
# 2. PolicyValidationAgent
# ---------------------------------------------------------------------------

_policy_collection = None

def set_chroma_collection(collection):
    """Store the ingested policy VectorCollection for the exclusion agent."""
    global _policy_collection
    _policy_collection = collection

def policy_validation_agent(state: ClaimState) -> dict:
    """Validate claim against mock policy database."""
    trace = list(state.get("trace", []))
    policy_number = state.get("policy_number", "UNKNOWN")
    claim_type = state.get("claim_type", "")
    claim_amount = state.get("claim_amount", 0.0)

    trace.append(f"[PolicyValidationAgent] Looking up policy {policy_number}")

    policy = MOCK_POLICIES.get(policy_number)

    if policy is None:
        trace.append(f"[PolicyValidationAgent] Policy {policy_number} NOT FOUND")
        return {
            "policy_found": False,
            "policy_status": "not_found",
            "coverage_valid": False,
            "coverage_limit": 0.0,
            "policy_notes": f"Policy {policy_number} does not exist in our records.",
            "trace": trace,
        }

    status = policy["status"]
    cov_type = policy["coverage_type"]
    cov_limit = policy["coverage_limit"]
    coverage_valid = (status == "active") and (cov_type == claim_type)

    notes_parts = []
    if status != "active":
        notes_parts.append(f"Policy is {status} (not active).")
    if cov_type != claim_type:
        notes_parts.append(
            f"Coverage type mismatch: policy covers '{cov_type}' but claim is '{claim_type}'."
        )
    if claim_amount > cov_limit:
        notes_parts.append(
            f"Claim amount ${claim_amount:,.2f} exceeds coverage limit ${cov_limit:,.2f}."
        )
    if not notes_parts:
        notes_parts.append("Policy is valid and covers this claim type.")

    notes = " ".join(notes_parts)
    trace.append(f"[PolicyValidationAgent] status={status}, coverage_valid={coverage_valid}, "
                 f"limit=${cov_limit:,.2f}")

    return {
        "policy_found": True,
        "policy_status": status,
        "coverage_valid": coverage_valid,
        "coverage_limit": float(cov_limit),
        "policy_notes": notes,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 3. FraudDetectionAgent
# ---------------------------------------------------------------------------

def _fallback_fraud(state: ClaimState) -> dict[str, Any]:
    """Deterministic heuristic fraud scoring."""
    score = 0
    flags: list[str] = []

    # Policy invalid
    if not state.get("coverage_valid", False):
        score += 30
        if state.get("policy_status") == "lapsed":
            flags.append("Policy is lapsed/inactive")
        elif not state.get("policy_found", False):
            flags.append("Policy not found in records")
        else:
            flags.append("Coverage type mismatch")

    # Amount near or over limit
    limit = state.get("coverage_limit", 0)
    amount = state.get("claim_amount", 0)
    if limit > 0:
        ratio = amount / limit
        if ratio > 1.0:
            score += 25
            flags.append("Claim amount exceeds coverage limit")
        elif ratio > 0.85:
            score += 15
            flags.append("Claim amount very close to coverage limit")

    # Urgency/pressure language
    raw = state.get("raw_text", "").lower()
    urgency_words = ["urgent", "immediately", "asap", "right away", "emergency",
                     "rush", "desperate", "need money", "as soon as possible"]
    for word in urgency_words:
        if word in raw:
            score += 10
            flags.append(f"Urgency language detected: '{word}'")
            break

    # Missing or vague dates
    if state.get("incident_date", "Unknown") == "Unknown":
        score += 15
        flags.append("Incident date missing or unclear")

    # Very short description
    desc = state.get("description", "")
    if len(desc) < 20:
        score += 10
        flags.append("Very short or missing incident description")

    # High amount absolute
    if amount > 20000:
        score += 10
        flags.append(f"High claim amount (${amount:,.2f})")

    score = min(score, 100)

    reasoning = "Heuristic analysis: "
    if flags:
        reasoning += "; ".join(flags) + "."
    else:
        reasoning += "No significant risk indicators detected."

    return {"fraud_score": score, "red_flags": flags, "fraud_reasoning": reasoning}


def fraud_detection_agent(state: ClaimState) -> dict:
    """Assess fraud risk using LLM or heuristic fallback."""
    trace = list(state.get("trace", []))
    trace.append("[FraudDetectionAgent] Analysing claim for fraud indicators")

    today_str = datetime.now().strftime("%Y-%m-%d")
    system_prompt = (
        "You are an insurance fraud detection specialist. Analyse the claim data below "
        "and return ONLY a JSON object with these keys:\n"
        "- fraud_score: integer 0-100 (0=no risk, 100=certain fraud)\n"
        "- red_flags: list of short strings describing risk indicators\n"
        "- fraud_reasoning: 1-2 sentence explanation\n\n"
        f"Today's date is {today_str}. Use this to correctly assess whether dates "
        "are in the past or future.\n\n"
        "Look for: policy invalid/lapsed, amount near or exceeding coverage limit, "
        "urgency or pressure language, missing documentation, unclear dates, "
        "inconsistencies between description and claim details."
    )

    user_prompt = (
        f"Claim data:\n"
        f"- Claimant: {state.get('claimant_name')}\n"
        f"- Policy: {state.get('policy_number')}\n"
        f"- Claim type: {state.get('claim_type')}\n"
        f"- Incident date: {state.get('incident_date')}\n"
        f"- Claim amount: ${state.get('claim_amount', 0):,.2f}\n"
        f"- Policy found: {state.get('policy_found')}\n"
        f"- Policy status: {state.get('policy_status')}\n"
        f"- Coverage valid: {state.get('coverage_valid')}\n"
        f"- Coverage limit: ${state.get('coverage_limit', 0):,.2f}\n"
        f"- Policy notes: {state.get('policy_notes')}\n\n"
        f"Raw claim text:\n{state.get('raw_text', '')}"
    )

    result = None
    llm_response = _llm_call(system_prompt, user_prompt)
    if llm_response:
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", llm_response).strip().rstrip("`")
            result = json.loads(cleaned)
            result["fraud_score"] = int(result["fraud_score"])
            result["red_flags"] = list(result["red_flags"])
            trace.append(f"[FraudDetectionAgent] LLM scoring: {result['fraud_score']}")
        except (json.JSONDecodeError, KeyError, ValueError):
            trace.append("[FraudDetectionAgent] LLM parse failed, using fallback")
            result = None

    if result is None:
        result = _fallback_fraud(state)
        trace.append(f"[FraudDetectionAgent] Heuristic scoring: {result['fraud_score']}")

    trace.append(f"[FraudDetectionAgent] Score={result['fraud_score']}, "
                 f"Flags={result['red_flags']}")

    return {
        "fraud_score": result["fraud_score"],
        "red_flags": result["red_flags"],
        "fraud_reasoning": result["fraud_reasoning"],
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 4. Conditional Router
# ---------------------------------------------------------------------------

def route_claim(state: ClaimState) -> str:
    """Route to DecisionAgent or SeniorReviewAgent based on risk factors."""
    fraud_score = state.get("fraud_score", 0)
    coverage_valid = state.get("coverage_valid", True)
    claim_amount = state.get("claim_amount", 0)
    coverage_limit = state.get("coverage_limit", float("inf"))
    policy_found = state.get("policy_found", True)
    excluded = state.get("exclusion_check", {}).get("excluded", False)
    deadline_exceeded = state.get("submission_deadline_exceeded", False)

    if fraud_score >= 60:
        return "senior_review"
    if not policy_found or not coverage_valid:
        return "senior_review"
    if claim_amount > coverage_limit:
        return "senior_review"
    if excluded:
        return "senior_review"
    if deadline_exceeded:
        return "senior_review"
    return "auto_approve"


# ---------------------------------------------------------------------------
# 5. ExclusionMatchAgent
# ---------------------------------------------------------------------------

def _fallback_exclusion_match(claim_desc: str, retrieved_clauses: list[dict]) -> dict:
    """TF-IDF fallback for exclusion matching."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        texts = [claim_desc] + [c["text"] for c in retrieved_clauses]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Similarities of claim_desc (index 0) with clauses (index 1 to N)
        cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        max_idx = cosine_sims.argmax()
        max_sim = cosine_sims[max_idx]
        
        if max_sim > 0.25:
            matched = retrieved_clauses[max_idx]
            # Try to extract clause number
            clause_num = None
            import re
            m = re.match(r'^(\d+)\.', matched["text"])
            if m:
                clause_num = int(m.group(1))
                
            return {
                "excluded": True,
                "matched_clause_number": clause_num,
                "matched_clause_text": matched["text"],
                "reasoning": f"Heuristic match (similarity: {max_sim:.2f}) with exclusion clause.",
                "method": "rule_based"
            }
    except Exception:
        pass
        
    return {
        "excluded": False,
        "matched_clause_number": None,
        "matched_clause_text": None,
        "reasoning": "No strong heuristic match found.",
        "method": "rule_based"
    }

def exclusion_match_agent(state: ClaimState) -> dict:
    trace = list(state.get("trace", []))
    trace.append("[ExclusionMatchAgent] Checking for policy exclusions")
    
    if _policy_collection is None:
        trace.append("[ExclusionMatchAgent] No policy PDF ingested, skipping exclusion check")
        return {
            "exclusion_check": {
                "excluded": False,
                "matched_clause_number": None,
                "matched_clause_text": None,
                "reasoning": "No policy document loaded",
                "method": "skipped"
            },
            "trace": trace
        }
        
    desc = state.get("description", "")
    if not desc:
        desc = state.get("structured_claim", {}).get("description", "")
        
    try:
        from retriever import retrieve_relevant_clauses
        clauses = retrieve_relevant_clauses(_policy_collection, desc, section_type="exclusion", k=5)
    except Exception as e:
        trace.append(f"[ExclusionMatchAgent] Retrieval failed: {e}")
        return {"trace": trace}
        
    if not clauses:
        trace.append("[ExclusionMatchAgent] No exclusions retrieved")
        return {
            "exclusion_check": {
                "excluded": False,
                "matched_clause_number": None,
                "matched_clause_text": None,
                "reasoning": "No exclusions found in document.",
                "method": "rule_based"
            },
            "trace": trace
        }
        
    clauses_text = "\n\n".join([f"Clause: {c['text']}" for c in clauses])
    
    system_prompt = (
        "You are an insurance claims exclusion checker. "
        "Given the retrieved policy exclusion clauses and the claim description, "
        "determine if any exclusion applies. "
        "Respond with ONLY a JSON object: excluded (bool), matched_clause_number (int or null), "
        "matched_clause_text (string or null), reasoning (1-2 sentences)."
    )
    user_prompt = f"Retrieved Clauses:\n{clauses_text}\n\nClaim Description:\n{desc}"
    
    result = None
    llm_response = _llm_call(system_prompt, user_prompt)
    if llm_response:
        try:
            import re
            cleaned = re.sub(r"```(?:json)?\s*", "", llm_response).strip().rstrip("`")
            result = json.loads(cleaned)
            result["method"] = "grok"
            trace.append("[ExclusionMatchAgent] LLM exclusion check successful")
        except Exception:
            trace.append("[ExclusionMatchAgent] LLM parse failed, using fallback")
            result = None
            
    if result is None:
        result = _fallback_exclusion_match(desc, clauses)
        trace.append("[ExclusionMatchAgent] Used rule-based fallback")
        
    if result.get("excluded"):
        trace.append(f"[ExclusionMatchAgent] Claim EXCLUDED by clause: {result.get('matched_clause_number')}")
    else:
        trace.append("[ExclusionMatchAgent] Claim not excluded by policy.")
        
    return {
        "exclusion_check": result,
        "trace": trace
    }

def route_after_exclusion(state: ClaimState) -> str:
    """Route based on exclusion result."""
    if state.get("exclusion_check", {}).get("excluded", False):
        return "senior_review"
    if state.get("submission_deadline_exceeded", False):
        return "senior_review"
    return "fraud_detection"

# ---------------------------------------------------------------------------
# 6. DecisionAgent
# ---------------------------------------------------------------------------

def decision_agent(state: ClaimState) -> dict:
    """Auto-approve low-risk claims."""
    trace = list(state.get("trace", []))
    trace.append("[DecisionAgent] Auto-approving claim")

    reason = (
        f"Claim approved. Policy {state.get('policy_number')} is "
        f"{state.get('policy_status', 'active')} with valid coverage. "
        f"Fraud score is low ({state.get('fraud_score', 0)}/100). "
        f"Claim amount ${state.get('claim_amount', 0):,.2f} is within the "
        f"coverage limit of ${state.get('coverage_limit', 0):,.2f}."
    )

    trace.append(f"[DecisionAgent] Decision: approved  --  {reason}")

    return {
        "route": "auto_approve",
        "decision_status": "approved",
        "decision_reason": reason,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 6. SeniorReviewAgent
# ---------------------------------------------------------------------------

def senior_review_agent(state: ClaimState) -> dict:
    """Escalate high-risk claims for senior review."""
    trace = list(state.get("trace", []))
    trace.append("[SeniorReviewAgent] Reviewing escalated claim")

    triggers: list[str] = []
    if state.get("fraud_score", 0) >= 60:
        triggers.append(f"High fraud score ({state.get('fraud_score')}/100)")
    if not state.get("policy_found", True):
        triggers.append("Policy not found in records")
    if not state.get("coverage_valid", True):
        triggers.append(f"Invalid coverage (policy status: {state.get('policy_status')})")
    if state.get("claim_amount", 0) > state.get("coverage_limit", float("inf")):
        triggers.append(
            f"Claim amount ${state.get('claim_amount', 0):,.2f} exceeds "
            f"coverage limit ${state.get('coverage_limit', 0):,.2f}"
        )
    if state.get("exclusion_check", {}).get("excluded"):
        triggers.append(f"Policy exclusion applies: Clause {state['exclusion_check'].get('matched_clause_number')}")
    if state.get("submission_deadline_exceeded", False):
        triggers.append("Submission deadline (90 days) exceeded")
    if not triggers:
        triggers.append("Escalated for manual review")

    reason = (
        f"Claim escalated for senior review. Trigger(s): {'; '.join(triggers)}. "
        f"Policy: {state.get('policy_number')}, Fraud score: {state.get('fraud_score', 0)}/100."
    )

    trace.append(f"[SeniorReviewAgent] Decision: escalated  --  triggers: {triggers}")

    return {
        "route": "escalate",
        "decision_status": "escalated",
        "decision_reason": reason,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 7. CommunicationAgent
# ---------------------------------------------------------------------------

_LETTER_APPROVED = """Dear {name},

Re: Insurance Claim  --  Policy {policy}

We are pleased to inform you that your insurance claim has been reviewed and approved.

Decision: APPROVED
{reason}

Your claim of ${amount} will be processed shortly, subject to your policy deductible. Please allow 5-7 business days for the payment to be issued.

If you have any questions, please don't hesitate to contact our claims department.

Warm regards,
Claims Processing Team"""

_LETTER_ESCALATED = """Dear {name},

Re: Insurance Claim  --  Policy {policy}

Thank you for filing your insurance claim. After our initial review, your claim has been escalated to our senior review team for further evaluation.

Decision: ESCALATED FOR REVIEW
{reason}

A senior claims specialist will be assigned to your case and will contact you within 2-3 business days to discuss next steps. We appreciate your patience during this process.

If you have any urgent questions, please call our priority claims line.

Sincerely,
Claims Processing Team"""

_LETTER_DENIED = """Dear {name},

Re: Insurance Claim  --  Policy {policy}

Thank you for submitting your insurance claim. After careful review, we regret to inform you that your claim could not be approved at this time.

Decision: DENIED
{reason}

You have the right to appeal this decision within 30 days. Please contact our claims department if you would like to provide additional documentation or discuss this further.

We understand this may be disappointing, and we are here to help you through the process.

Respectfully,
Claims Processing Team"""


def _fallback_communication(state: ClaimState) -> str:
    """Generate letter from templates."""
    name = state.get("claimant_name", "Valued Customer")
    policy = state.get("policy_number", "N/A")
    status = state.get("decision_status", "escalated")
    reason = state.get("decision_reason", "")
    amount = f"{state.get('claim_amount', 0):,.2f}"

    if status == "approved":
        template = _LETTER_APPROVED
    elif status == "denied":
        template = _LETTER_DENIED
    else:
        template = _LETTER_ESCALATED

    return template.format(name=name, policy=policy, reason=reason, amount=amount)


def communication_agent(state: ClaimState) -> dict:
    """Draft a customer-facing letter about the claim decision."""
    trace = list(state.get("trace", []))
    status = state.get("decision_status", "escalated")
    trace.append(f"[CommunicationAgent] Drafting letter for '{status}' decision")

    system_prompt = (
        "You are an empathetic insurance communications specialist. "
        "Draft a short, professional letter to the claimant about their claim decision. "
        "Use a warm but professional tone. Include the claimant's name, policy number, "
        "decision status, and reason. Return ONLY the letter text, no JSON wrapping."
    )

    user_prompt = (
        f"Claimant: {state.get('claimant_name')}\n"
        f"Policy: {state.get('policy_number')}\n"
        f"Claim amount: ${state.get('claim_amount', 0):,.2f}\n"
        f"Decision: {status}\n"
        f"Reason: {state.get('decision_reason')}\n\n"
        f"Please draft the letter."
    )

    letter = None
    llm_response = _llm_call(system_prompt, user_prompt)
    if llm_response and len(llm_response.strip()) > 50:
        letter = llm_response.strip()
        trace.append("[CommunicationAgent] LLM-generated letter")

    if letter is None:
        letter = _fallback_communication(state)
        trace.append("[CommunicationAgent] Used template-based letter")

    trace.append(f"[CommunicationAgent] Letter drafted ({len(letter)} chars)")

    return {
        "communication_letter": letter,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph state graph."""
    graph = StateGraph(ClaimState)

    # Add nodes
    graph.add_node("intake", intake_agent)
    graph.add_node("policy_validation", policy_validation_agent)
    graph.add_node("exclusion_check", exclusion_match_agent)
    graph.add_node("fraud_detection", fraud_detection_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("senior_review", senior_review_agent)
    graph.add_node("communication", communication_agent)

    # Linear edges
    graph.set_entry_point("intake")
    graph.add_edge("intake", "policy_validation")
    graph.add_edge("policy_validation", "exclusion_check")
    
    # Conditional routing after exclusion
    graph.add_conditional_edges(
        "exclusion_check",
        route_after_exclusion,
        {
            "fraud_detection": "fraud_detection",
            "senior_review": "senior_review",
        },
    )

    # Conditional routing after fraud detection
    graph.add_conditional_edges(
        "fraud_detection",
        route_claim,
        {
            "auto_approve": "decision",
            "senior_review": "senior_review",
        },
    )

    # Both decision paths feed into communication
    graph.add_edge("decision", "communication")
    graph.add_edge("senior_review", "communication")

    # Communication is the terminal node
    graph.add_edge("communication", END)

    return graph


def compile_graph():
    """Build and compile the graph, ready for invocation."""
    return build_graph().compile()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def process_claim(raw_text: str, initial_data: dict = None) -> ClaimState:
    """
    Process a raw claim text through the full agent pipeline.

    Returns the final ClaimState with all fields populated.
    """
    app = compile_graph()
    initial_state: ClaimState = {
        "raw_text": raw_text,
        "trace": [f"[System] Claim received at {datetime.now().isoformat()}"],
    }
    if initial_data:
        initial_state.update(initial_data)
        
    result = app.invoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# Sample claims for testing
# ---------------------------------------------------------------------------

SAMPLE_CLAIMS = {
    "approved": (
        "Claimant: Jane Doe. Policy: POL-10234. Auto claim. "
        "Incident date 2026-08-01. Amount: $4,500. "
        "My car was rear-ended at a stoplight on Highway 101. "
        "The other driver admitted fault. Police report #PR-2026-5567 filed. "
        "Damage to rear bumper, trunk, and tail lights."
    ),
    "fraud_escalated": (
        "Claimant: Alice Johnson. Policy: POL-30789. Auto claim. "
        "Incident date 2026-07-15. Amount: $14,000. "
        "I need this processed URGENTLY  --  my car was totaled in a hit-and-run. "
        "No witnesses. I need the money as soon as possible. "
        "Please rush this claim immediately."
    ),
    "over_limit_escalated": (
        "Claimant: Clara Davis. Policy: POL-50321. Auto claim. "
        "Incident date 2026-06-20. Amount: $7,500. "
        "A tree fell on my car during a storm. "
        "Damage to roof, windshield, and hood. "
        "Fire department report available."
    ),
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Multi-Agent Insurance Claims Processing System")
    print("=" * 70)

    api_key = os.environ.get("XAI_API_KEY")
    print(f"\nXAI_API_KEY: {'SET' if api_key else 'NOT SET (using fallbacks)'}\n")

    for scenario, raw in SAMPLE_CLAIMS.items():
        print(f"\n{'-' * 70}")
        print(f"  SCENARIO: {scenario.upper()}")
        print(f"{'-' * 70}")
        print(f"Input: {raw[:100]}...")

        result = process_claim(raw)

        print(f"\n  Claimant:    {result.get('claimant_name')}")
        print(f"  Policy:      {result.get('policy_number')}")
        print(f"  Claim Type:  {result.get('claim_type')}")
        print(f"  Amount:      ${result.get('claim_amount', 0):,.2f}")
        print(f"  Policy OK:   {result.get('coverage_valid')}")
        print(f"  Fraud Score: {result.get('fraud_score')}/100")
        print(f"  Red Flags:   {result.get('red_flags')}")
        print(f"  Decision:    {result.get('decision_status')}")
        print(f"  Reason:      {result.get('decision_reason', '')[:150]}")
        print(f"\n  Letter (first 200 chars):")
        print(f"  {result.get('communication_letter', '')[:200]}")
        print(f"\n  Trace log:")
        for entry in result.get("trace", []):
            print(f"    {entry}")
