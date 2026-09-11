"""
EduAI – IBM Granite Integration
=================================
Steps covered: 11 (credentials/fallback), 12 (role separation),
               13 (prompt design), 14 (response validation)

IBM Granite (via IBM watsonx.ai) is the GENERATIVE AI component.
  – ML model  → predicts risk level
  – Risk factors module → identifies observable indicators
  – Granite   → explains the situation and generates recommendations

Credentials are loaded from environment variables ONLY.
Never hard-code API keys, project IDs, or passwords.

If credentials are missing, a clearly labelled DEMO/FALLBACK mode
is used so the rest of the prototype can still be tested.
"""

import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


# ── Credential keys (set these as environment variables) ─────────────────────
ENV_API_KEY     = "WATSONX_API_KEY"
ENV_PROJECT_ID  = "WATSONX_PROJECT_ID"
ENV_REGION      = "WATSONX_REGION"          # e.g. "us-south"

DEFAULT_REGION  = "us-south"
GRANITE_MODEL   = "ibm/granite-4-h-small"

# IBM watsonx.ai inference URL pattern
WATSONX_URL_TEMPLATE = "https://{region}.ml.cloud.ibm.com"


# ─── Credentials loader ───────────────────────────────────────────────────────
def _load_credentials() -> Tuple[Optional[str], Optional[str], str]:
    """
    Load IBM watsonx.ai credentials from environment variables.

    Returns
    -------
    (api_key, project_id, region)  – any value may be None if not set.
    """
    api_key    = os.environ.get(ENV_API_KEY)
    project_id = os.environ.get(ENV_PROJECT_ID)
    region     = os.environ.get(ENV_REGION, DEFAULT_REGION)
    return api_key, project_id, region


def credentials_available() -> bool:
    """Return True if both WATSONX_API_KEY and WATSONX_PROJECT_ID are set."""
    api_key, project_id, _ = _load_credentials()
    return bool(api_key and project_id)


# ─── Prompt builder ───────────────────────────────────────────────────────────
def build_granite_prompt(student_id: str,
                         marks: float,
                         attendance: float,
                         quiz_score: float,
                         assignment_completion: float,
                         previous_marks: float,
                         risk_level: str,
                         risk_factors: List[str]) -> str:
    """
    Build a structured prompt for IBM Granite.
    Instructs Granite to generate:
      A. Explanation (2–3 sentences)
      B. Exactly three recommendations
      C. One monitoring suggestion

    The prompt explicitly instructs Granite to:
      - Not invent information
      - Avoid sensitive assumptions
      - Use supportive, non-judgmental language
      - Not make medical/psychological diagnoses
    """
    factors_text = (
        "\n".join(f"  - {f}" for f in risk_factors)
        if risk_factors
        else "  - No significant academic risk indicators detected."
    )

    performance_delta = marks - previous_marks
    delta_str = (
        f"{performance_delta:+.1f} (improvement)"
        if performance_delta >= 0
        else f"{performance_delta:+.1f} (decline)"
    )

    prompt = textwrap.dedent(f"""
    You are an academic support assistant helping educators identify students
    who may need academic attention.  Use only the information provided below.
    Do NOT invent information.  Do NOT make assumptions about the student's
    personal, financial, family, health, or social circumstances.
    Use supportive, encouraging, and non-judgmental language.
    Do NOT make medical or psychological diagnoses.
    Do NOT treat the risk category as a permanent label.
    Do NOT claim certainty about future academic performance.

    ── Student Academic Indicators (ALL DATA IS SYNTHETIC) ──────────────────
    Student ID              : {student_id}
    Current Marks           : {marks}
    Previous Marks          : {previous_marks}
    Performance Change      : {delta_str}
    Attendance              : {attendance}%
    Quiz Score              : {quiz_score}
    Assignment Completion   : {assignment_completion}%
    ML-Predicted Risk Level : {risk_level}

    Observable Academic Indicators (rule-based):
    {factors_text}

    ── Required Output Format ───────────────────────────────────────────────
    Respond using EXACTLY the following structure.
    Do not add extra sections, headers, or preamble.

    Explanation:
    [Write 2 to 3 sentences explaining why this student may need academic
    attention based only on the indicators above.]

    Recommendations:
    1. [Practical, specific academic support recommendation]
    2. [Practical, specific academic support recommendation]
    3. [Practical, specific academic support recommendation]

    Monitoring Suggestion:
    [One practical suggestion for how a teacher or mentor can monitor this
    student's academic progress going forward.]
    """).strip()

    return prompt


# ─── IBM Granite API call ─────────────────────────────────────────────────────
def _call_granite_api(prompt: str,
                      api_key: str,
                      project_id: str,
                      region: str) -> str:
    """
    Call the IBM watsonx.ai text generation API using ibm-watsonx-ai SDK.

    Returns the raw text response from Granite.
    Raises RuntimeError on any API error.
    """
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    except ImportError as e:
        raise RuntimeError(
            "ibm-watsonx-ai package is not installed. "
            "Run: pip install ibm-watsonx-ai"
        ) from e

    url = WATSONX_URL_TEMPLATE.format(region=region)

    credentials = Credentials(url=url, api_key=api_key)
    client      = APIClient(credentials=credentials)

    params = {
        GenParams.TEMPERATURE:    0.4,
        GenParams.TOP_P:          0.9,
        GenParams.REPETITION_PENALTY: 1.1,
    }

    model = ModelInference(
        model_id=GRANITE_MODEL,
        api_client=client,
        project_id=project_id,
        params=params,
    )

    response = model.chat(
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# ─── Response parser ──────────────────────────────────────────────────────────
def parse_granite_response(raw: str) -> Dict:
    """
    Parse the structured Granite response into its components.
    Steps covered: 14 (response validation)

    Expected sections:
      Explanation:
      Recommendations:
        1. …
        2. …
        3. …
      Monitoring Suggestion:

    Returns
    -------
    dict with keys:
      explanation, recommendations (list of 3), monitoring_suggestion,
      parse_success (bool), parse_errors (list)
    """
    result = {
        "explanation":          "",
        "recommendations":      [],
        "monitoring_suggestion": "",
        "parse_success":        False,
        "parse_errors":         [],
    }

    # ── Extract Explanation ───────────────────────────────────────────────
    exp_match = re.search(
        r"Explanation:\s*(.*?)(?=Recommendations:|$)", raw,
        re.DOTALL | re.IGNORECASE
    )
    if exp_match:
        result["explanation"] = exp_match.group(1).strip()
    else:
        result["parse_errors"].append("Explanation section not found.")

    # ── Extract Recommendations ───────────────────────────────────────────
    rec_match = re.search(
        r"Recommendations:\s*(.*?)(?=Monitoring Suggestion:|$)", raw,
        re.DOTALL | re.IGNORECASE
    )
    if rec_match:
        rec_block = rec_match.group(1).strip()
        # Match numbered items 1. / 2. / 3.
        items = re.findall(
            r"^\s*[1-3][.)]\s*(.+?)(?=^\s*[1-3][.)]|\Z)",
            rec_block, re.MULTILINE | re.DOTALL
        )
        result["recommendations"] = [item.strip().replace("\n", " ")
                                      for item in items][:3]

        if len(result["recommendations"]) != 3:
            result["parse_errors"].append(
                f"Expected exactly 3 recommendations; "
                f"found {len(result['recommendations'])}."
            )
    else:
        result["parse_errors"].append("Recommendations section not found.")

    # ── Extract Monitoring Suggestion ─────────────────────────────────────
    mon_match = re.search(
        r"Monitoring Suggestion:\s*(.*?)$", raw,
        re.DOTALL | re.IGNORECASE
    )
    if mon_match:
        result["monitoring_suggestion"] = mon_match.group(1).strip()
    else:
        result["parse_errors"].append("Monitoring Suggestion section not found.")

    result["parse_success"] = len(result["parse_errors"]) == 0
    return result


# ─── Fallback (demo) response ─────────────────────────────────────────────────
def _demo_response(student_id: str,
                   risk_level: str,
                   risk_factors: List[str]) -> Dict:
    """
    Return a clearly labelled DEMO/FALLBACK response.

    IMPORTANT: This response is NOT generated by IBM Granite.
    It is used only when IBM watsonx.ai credentials are unavailable.
    It is clearly marked as a demo response throughout the UI.
    """
    factors_str = (
        "; ".join(risk_factors)
        if risk_factors
        else "no significant academic risk indicators"
    )

    if risk_level == "At Risk":
        explanation = (
            f"[DEMO – Not IBM Granite] Student {student_id} has been classified as "
            f"'{risk_level}' based on observable academic indicators: {factors_str}. "
            "Multiple areas of academic concern are present and early educator "
            "engagement is suggested to support the student's learning."
        )
        recs = [
            "Work with the student to establish a structured weekly study plan "
            "that prioritises subjects with the lowest current marks.",
            "Improve regular attendance and explore barriers to consistent "
            "class participation with the educator's support.",
            "Complete outstanding assignments and use quiz results to identify "
            "specific topics that require additional review or practice.",
        ]
        monitoring = (
            "Review the student's attendance record, assignment submissions, "
            "and marks every two weeks and schedule brief check-in meetings "
            "with the student to track progress."
        )
    elif risk_level == "Needs Attention":
        explanation = (
            f"[DEMO – Not IBM Granite] Student {student_id} shows early signs of "
            f"academic concern: {factors_str}. "
            "Targeted support in specific areas could help the student return "
            "to a stronger academic trajectory."
        )
        recs = [
            "Focus revision sessions on topics linked to low quiz scores and "
            "aim to consolidate understanding before the next assessment.",
            "Maintain consistent attendance to avoid missing key instructional "
            "content that contributes to assessment performance.",
            "Prioritise timely submission of remaining assignments to strengthen "
            "the overall academic record.",
        ]
        monitoring = (
            "Check in with the student every three weeks to review marks, "
            "quiz performance, and assignment completion trends."
        )
    else:  # On Track
        explanation = (
            f"[DEMO – Not IBM Granite] Student {student_id} is currently "
            f"performing well across academic indicators. "
            "Continued consistency and engagement will help maintain this "
            "positive academic trajectory."
        )
        recs = [
            "Continue with the current study routine and aim to challenge "
            "learning in areas of strength.",
            "Maintain high attendance and active participation in class "
            "activities and discussions.",
            "Review previous quiz and assignment feedback to identify any "
            "areas for further improvement.",
        ]
        monitoring = (
            "Continue periodic review every four to six weeks to ensure "
            "consistent academic performance is maintained."
        )

    return {
        "explanation":           explanation,
        "recommendations":       recs,
        "monitoring_suggestion": monitoring,
        "parse_success":         True,
        "parse_errors":          [],
        "is_demo":               True,
    }


# ─── Main public function ─────────────────────────────────────────────────────
def generate_granite_response(student_id: str,
                               marks: float,
                               attendance: float,
                               quiz_score: float,
                               assignment_completion: float,
                               previous_marks: float,
                               risk_level: str,
                               risk_factors: List[str]) -> Dict:
    """
    Generate an IBM Granite response for a student.

    If credentials are available, calls the watsonx.ai API.
    If not, returns a clearly labelled DEMO/FALLBACK response.

    Returns
    -------
    dict with keys:
      explanation, recommendations (list of 3), monitoring_suggestion,
      parse_success, parse_errors, is_demo (bool)
    """
    api_key, project_id, region = _load_credentials()

    if not (api_key and project_id):
        # ── DEMO MODE ─────────────────────────────────────────────────────
        return _demo_response(student_id, risk_level, risk_factors)

    # ── LIVE IBM GRANITE MODE ─────────────────────────────────────────────
    prompt = build_granite_prompt(
        student_id=student_id,
        marks=marks,
        attendance=attendance,
        quiz_score=quiz_score,
        assignment_completion=assignment_completion,
        previous_marks=previous_marks,
        risk_level=risk_level,
        risk_factors=risk_factors,
    )

    try:
        raw = _call_granite_api(prompt, api_key, project_id, region)
    except RuntimeError as e:
        # Package missing or network error
        fallback = _demo_response(student_id, risk_level, risk_factors)
        fallback["parse_errors"] = [
            f"IBM Granite API error: {e}. DEMO response shown instead."
        ]
        return fallback
    except Exception as e:  # noqa: BLE001
        fallback = _demo_response(student_id, risk_level, risk_factors)
        fallback["parse_errors"] = [
            f"Unexpected error calling IBM Granite: {e}. "
            "DEMO response shown instead."
        ]
        return fallback

    parsed = parse_granite_response(raw)
    parsed["is_demo"] = False

    # ── Validate: exactly 3 recommendations ──────────────────────────────
    if len(parsed["recommendations"]) != 3:
        parsed["parse_errors"].append(
            "Response validation failed: could not extract exactly 3 "
            "recommendations from IBM Granite output."
        )

    return parsed
