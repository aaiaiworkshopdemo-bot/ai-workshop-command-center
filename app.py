import base64
import hashlib
import json
import re
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =============================
# App Config
# =============================
st.set_page_config(
    page_title="AI Workshop Command Center",
    page_icon="🤖",
    layout="wide",
)

FORM_SHEET_NAME = "Form Responses 1"
AI_OUTPUT_SHEET_NAME = "AI_Output"
ROOM_SUMMARY_SHEET_NAME = "Room_Summary"

AI_OUTPUT_HEADERS = [
    "response_id",
    "timestamp",
    "name",
    "email",
    "audience_type",
    "industry",
    "ai_level",
    "interested_topics",
    "ai_expectation",
    "ai_question",
    "consent_followup",
    "topic_tags",
    "main_need",
    "recommended_sections",
    "email_subject",
    "email_body",
    "status",
    "reviewed",
    "sent_at",
    "error_message",
    "last_updated",
]

ROOM_SUMMARY_HEADERS = [
    "summary_id",
    "generated_at",
    "total_responses",
    "audience_summary",
    "top_needs",
    "top_questions",
    "audience_dynamics",
    "recommended_workshop_focus",
    "suggested_examples",
    "what_not_to_focus_on",
    "opening_line",
    "raw_ai_output",
]

WORKSHOP_TOPICS = [
    "How to Talk to AI Like a Professional",
    "Custom AI Agents",
    "AI Content Creation",
    "AI Agents Working Together / Automation",
    "AI Product Development",
    "Future of Work with AI",
]

# =============================
# Utility Functions
# =============================
def clean_col_name(col: str) -> str:
    """Normalize Google Form column names by trimming extra spaces."""
    col = str(col).replace("\n", " ").replace("\t", " ")
    col = re.sub(r"\s+", " ", col).strip()
    return col


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_response_id(row: Dict[str, Any]) -> str:
    raw = "|".join([
        str(row.get("timestamp", "")),
        str(row.get("name", "")),
        str(row.get("email", "")),
        str(row.get("ai_expectation", "")),
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_json_safely(text: str) -> Dict[str, Any]:
    """Extract JSON object from model output, even if wrapped in markdown."""
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def df_to_sheet_values(df: pd.DataFrame, headers: List[str]) -> List[List[str]]:
    out = df.copy()
    for h in headers:
        if h not in out.columns:
            out[h] = ""
    out = out[headers]
    out = out.fillna("").astype(str)
    return [headers] + out.values.tolist()


def basic_html_email(body: str) -> str:
    """Convert plain text body into simple readable HTML."""
    escaped = (
        body.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.5; font-size: 14px; color: #222;">
        {escaped}
      </body>
    </html>
    """

# =============================
# Secrets / Connections
# =============================
def require_secret(key: str) -> str:
    value = st.secrets.get(key, "")
    if not value:
        st.error(f"Missing secret: {key}")
        st.stop()
    return value


@st.cache_resource(show_spinner=False)
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    service_account_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def get_spreadsheet():
    sheet_id = require_secret("GOOGLE_SHEET_ID")
    client = get_gspread_client()
    return client.open_by_key(sheet_id)


def get_or_create_worksheet(name: str, headers: List[str], rows: int = 1000, cols: int = 30):
    spreadsheet = get_spreadsheet()
    try:
        ws = spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=rows, cols=cols)
        ws.update([headers])
    current_headers = ws.row_values(1)
    if not current_headers:
        ws.update([headers])
    return ws

# =============================
# Google Sheets Data
# =============================
def read_form_responses() -> pd.DataFrame:
    ws = get_spreadsheet().worksheet(FORM_SHEET_NAME)
    records = ws.get_all_records(default_blank="")
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    df.columns = [clean_col_name(c) for c in df.columns]

    # Map exact cleaned Google Form questions to standard app names.
    col_map = {
        "Timestamp": "timestamp",
        "Name / Nickname": "name",
        "Email Address": "email",
        "Which best describes you?": "audience_type",
        "What field or industry are you connected to?": "industry",
        "How familiar are you with AI tools?": "ai_level",
        "Which workshop topics are you most interested in?": "interested_topics",
        "What is one thing you want AI to help you with?": "ai_expectation",
        "What is one question or concern you have about AI?": "ai_question",
        "Would you like to receive a personalized AI action plan after the workshop?": "consent_followup",
    }

    renamed = {}
    for c in df.columns:
        cleaned = clean_col_name(c)
        if cleaned in col_map:
            renamed[c] = col_map[cleaned]
    df = df.rename(columns=renamed)

    standard_cols = [
        "timestamp",
        "name",
        "email",
        "audience_type",
        "industry",
        "ai_level",
        "interested_topics",
        "ai_expectation",
        "ai_question",
        "consent_followup",
    ]
    for c in standard_cols:
        if c not in df.columns:
            df[c] = ""

    if df["consent_followup"].replace("", pd.NA).isna().all():
        df["consent_followup"] = "Yes, send me a personalized follow-up email"

    df = df[standard_cols].fillna("")
    df["response_id"] = df.apply(lambda r: make_response_id(r.to_dict()), axis=1)
    return df


def read_ai_output() -> pd.DataFrame:
    ws = get_or_create_worksheet(AI_OUTPUT_SHEET_NAME, AI_OUTPUT_HEADERS)
    records = ws.get_all_records(default_blank="")
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=AI_OUTPUT_HEADERS)
    df.columns = [clean_col_name(c) for c in df.columns]
    for c in AI_OUTPUT_HEADERS:
        if c not in df.columns:
            df[c] = ""
    return df[AI_OUTPUT_HEADERS].fillna("")


def write_ai_output(df: pd.DataFrame):
    ws = get_or_create_worksheet(AI_OUTPUT_SHEET_NAME, AI_OUTPUT_HEADERS)
    values = df_to_sheet_values(df, AI_OUTPUT_HEADERS)
    ws.clear()
    ws.update(values)


def append_room_summary(summary: Dict[str, Any], total_responses: int, raw_text: str):
    ws = get_or_create_worksheet(ROOM_SUMMARY_SHEET_NAME, ROOM_SUMMARY_HEADERS)
    row = {
        "summary_id": hashlib.md5((now_str() + raw_text[:100]).encode("utf-8")).hexdigest()[:12],
        "generated_at": now_str(),
        "total_responses": total_responses,
        "audience_summary": summary.get("audience_summary", ""),
        "top_needs": "\n".join(summary.get("top_needs", [])) if isinstance(summary.get("top_needs", []), list) else summary.get("top_needs", ""),
        "top_questions": "\n".join(summary.get("top_questions", [])) if isinstance(summary.get("top_questions", []), list) else summary.get("top_questions", ""),
        "audience_dynamics": summary.get("audience_dynamics", ""),
        "recommended_workshop_focus": summary.get("recommended_workshop_focus", ""),
        "suggested_examples": "\n".join(summary.get("suggested_examples", [])) if isinstance(summary.get("suggested_examples", []), list) else summary.get("suggested_examples", ""),
        "what_not_to_focus_on": summary.get("what_not_to_focus_on", ""),
        "opening_line": summary.get("opening_line", ""),
        "raw_ai_output": raw_text,
    }
    ws.append_row([str(row.get(h, "")) for h in ROOM_SUMMARY_HEADERS])
    return row


def get_latest_room_summary() -> pd.DataFrame:
    ws = get_or_create_worksheet(ROOM_SUMMARY_SHEET_NAME, ROOM_SUMMARY_HEADERS)
    records = ws.get_all_records(default_blank="")
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=ROOM_SUMMARY_HEADERS)
    for c in ROOM_SUMMARY_HEADERS:
        if c not in df.columns:
            df[c] = ""
    return df[ROOM_SUMMARY_HEADERS]

# =============================
# Gemini API
# =============================
def call_gemini(prompt: str, temperature: float = 0.4) -> str:
    api_key = require_secret("GEMINI_API_KEY")
    model = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    }
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Could not parse Gemini response: {data}") from e


def generate_room_summary(df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
    rows = df[[
        "name",
        "audience_type",
        "industry",
        "ai_level",
        "interested_topics",
        "ai_expectation",
        "ai_question",
    ]].fillna("").to_dict(orient="records")

    prompt = f"""
You are an AI workshop strategist helping Rishad run a 3-hour practical workshop titled:
"AI That Works For You: Practical AI for Career Growth, Business Productivity & Everyday Work".

Workshop topics:
{json.dumps(WORKSHOP_TOPICS, indent=2)}

Audience responses are below as JSON. Analyze them and return ONLY valid JSON with these keys:
- audience_summary: short paragraph, simple language
- top_needs: list of 5 concise needs
- top_questions: list of 5 concise questions/concerns
- audience_dynamics: short paragraph explaining who is in the room and maturity level
- recommended_workshop_focus: short practical recommendation for how Rishad should adjust the session
- suggested_examples: list of 5 examples/demos Rishad should use
- what_not_to_focus_on: short paragraph on what to avoid for this audience
- opening_line: one strong sentence Rishad can say on stage

Important style:
- Practical, inspiring, non-technical
- Optimize for audience engagement, not theory
- Do not mention private data
- Do not overstate certainty

Audience responses:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""
    raw = call_gemini(prompt, temperature=0.35)
    parsed = parse_json_safely(raw)
    return parsed, raw


def generate_personal_email(row: Dict[str, Any]) -> Dict[str, Any]:
    name = safe_str(row.get("name")) or "there"
    prompt = f"""
You are helping Rishad create a personalized follow-up email for a participant of his workshop:
"AI That Works For You: Practical AI for Career Growth, Business Productivity & Everyday Work".

Workshop topics:
{json.dumps(WORKSHOP_TOPICS, indent=2)}

Participant details:
Name: {safe_str(row.get('name'))}
Audience type: {safe_str(row.get('audience_type'))}
Industry/field: {safe_str(row.get('industry'))}
AI level: {safe_str(row.get('ai_level'))}
Interested topics: {safe_str(row.get('interested_topics'))}
What they want AI to help with: {safe_str(row.get('ai_expectation'))}
Their AI question/concern: {safe_str(row.get('ai_question'))}

Return ONLY valid JSON with these keys:
- topic_tags: comma-separated tags, maximum 5
- main_need: one concise sentence
- recommended_sections: comma-separated list of 2-3 relevant workshop sections
- email_subject: concise subject line
- email_body: a warm, useful plain-text email body signed "Best regards,\nRishad"

Email body requirements:
- Start with "Hi {name},"
- Thank them for joining the workshop
- Mention what their main AI interest seems to be
- Recommend 2-3 relevant workshop sections and explain why
- Give 3 practical AI use cases they can try tomorrow
- Give 2 starter prompts they can copy
- Give one first action after the workshop
- Keep it concise and readable
- Avoid technical jargon
- Do not claim that AI knows more than what the participant provided
- Do not include sensitive personal advice
"""
    raw = call_gemini(prompt, temperature=0.45)
    parsed = parse_json_safely(raw)
    return {
        "topic_tags": parsed.get("topic_tags", ""),
        "main_need": parsed.get("main_need", ""),
        "recommended_sections": parsed.get("recommended_sections", ""),
        "email_subject": parsed.get("email_subject", "Your Personal AI Action Plan"),
        "email_body": parsed.get("email_body", ""),
    }

# =============================
# Email Sending
# =============================
def send_email(to_email: str, subject: str, body: str) -> Tuple[bool, str]:
    gmail_user = require_secret("GMAIL_USER")
    gmail_password = require_secret("GMAIL_APP_PASSWORD")
    sender_name = st.secrets.get("SENDER_NAME", "AI Workshop Team")

    if not to_email or "@" not in to_email:
        return False, "Invalid recipient email"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{gmail_user}>"
    msg["To"] = to_email

    text_part = MIMEText(body, "plain", "utf-8")
    html_part = MIMEText(basic_html_email(body), "html", "utf-8")
    msg.attach(text_part)
    msg.attach(html_part)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, [to_email], msg.as_string())
        return True, "Sent"
    except Exception as e:
        return False, str(e)

# =============================
# Admin Login
# =============================
def admin_gate():
    admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    if not admin_password:
        st.warning("ADMIN_PASSWORD is not set in secrets. Add it before deployment.")
        return

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("AI Workshop Command Center")
        st.caption("Admin access")
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if pwd == admin_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

# =============================
# App UI
# =============================
admin_gate()

st.title("🤖 AI Workshop Command Center")
st.caption("Google Form → Google Sheet → Gemini → Personalized Emails → Gmail Send")

with st.sidebar:
    st.subheader("Controls")
    if st.button("🔄 Clear cache / refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.write("**Sheet tabs expected:**")
    st.code(f"{FORM_SHEET_NAME}\n{AI_OUTPUT_SHEET_NAME}\n{ROOM_SUMMARY_SHEET_NAME}")
    st.markdown("---")
    st.warning("Do not show raw personal email addresses on projector unless needed.")

try:
    form_df = read_form_responses()
    output_df = read_ai_output()
except Exception as e:
    st.error("Could not connect to Google Sheets. Check Sheet ID, service account JSON, APIs, and sharing permissions.")
    st.exception(e)
    st.stop()

# Merge missing responses into output with New status placeholders
def sync_output_with_form(form_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    existing_ids = set(output_df["response_id"].astype(str).tolist()) if not output_df.empty else set()
    new_rows = []
    for _, r in form_df.iterrows():
        rid = r["response_id"]
        if rid not in existing_ids:
            new_rows.append({
                "response_id": rid,
                "timestamp": r.get("timestamp", ""),
                "name": r.get("name", ""),
                "email": r.get("email", ""),
                "audience_type": r.get("audience_type", ""),
                "industry": r.get("industry", ""),
                "ai_level": r.get("ai_level", ""),
                "interested_topics": r.get("interested_topics", ""),
                "ai_expectation": r.get("ai_expectation", ""),
                "ai_question": r.get("ai_question", ""),
                "consent_followup": r.get("consent_followup", ""),
                "topic_tags": "",
                "main_need": "",
                "recommended_sections": "",
                "email_subject": "",
                "email_body": "",
                "status": "New",
                "reviewed": "No",
                "sent_at": "",
                "error_message": "",
                "last_updated": now_str(),
            })
    if new_rows:
        output_df = pd.concat([output_df, pd.DataFrame(new_rows)], ignore_index=True)
        write_ai_output(output_df)
    return output_df

output_df = sync_output_with_form(form_df, output_df)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Audience Pulse",
    "🧠 AI Room Summary",
    "✉️ Personalized Emails",
    "🚀 Send Center",
])

# =============================
# Tab 1: Live Audience Pulse
# =============================
with tab1:
    st.header("Live Audience Pulse")
    if form_df.empty:
        st.info("No form responses yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total responses", len(form_df))
        c2.metric("Unique emails", form_df["email"].nunique())
        c3.metric("Audience types", form_df["audience_type"].replace("", pd.NA).dropna().nunique())
        c4.metric("AI levels", form_df["ai_level"].replace("", pd.NA).dropna().nunique())

        st.subheader("Audience Type Split")
        audience_counts = form_df["audience_type"].replace("", "Not specified").value_counts().reset_index()
        audience_counts.columns = ["Audience Type", "Count"]
        st.bar_chart(audience_counts.set_index("Audience Type"))

        st.subheader("AI Familiarity Split")
        level_counts = form_df["ai_level"].replace("", "Not specified").value_counts().reset_index()
        level_counts.columns = ["AI Level", "Count"]
        st.bar_chart(level_counts.set_index("AI Level"))

        st.subheader("Latest Responses")
        display_cols = ["timestamp", "name", "audience_type", "industry", "ai_level", "interested_topics", "ai_expectation", "ai_question"]
        st.dataframe(form_df[display_cols].tail(20), use_container_width=True, hide_index=True)

# =============================
# Tab 2: AI Room Summary
# =============================
with tab2:
    st.header("AI Room Summary")
    st.write("Generate the top-level audience dynamics and workshop recommendation from live form responses.")

    if form_df.empty:
        st.info("No responses available yet.")
    else:
        if st.button("🧠 Generate / Refresh Room Summary", type="primary"):
            with st.spinner("Gemini is reading the room..."):
                try:
                    summary, raw = generate_room_summary(form_df)
                    saved = append_room_summary(summary, len(form_df), raw)
                    st.success("Room summary generated and saved.")
                    st.session_state.latest_summary = saved
                except Exception as e:
                    st.error("Failed to generate room summary.")
                    st.exception(e)

        latest_summary_df = get_latest_room_summary()
        if not latest_summary_df.empty:
            latest = latest_summary_df.tail(1).iloc[0].to_dict()
            st.subheader("Latest Summary")
            st.caption(f"Generated at: {latest.get('generated_at', '')} | Responses: {latest.get('total_responses', '')}")

            st.markdown("### Audience Summary")
            st.write(latest.get("audience_summary", ""))

            st.markdown("### Audience Dynamics")
            st.write(latest.get("audience_dynamics", ""))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Top Needs")
                st.write(latest.get("top_needs", ""))

                st.markdown("### Suggested Examples")
                st.write(latest.get("suggested_examples", ""))
            with col_b:
                st.markdown("### Top Questions")
                st.write(latest.get("top_questions", ""))

                st.markdown("### What Not To Focus On")
                st.write(latest.get("what_not_to_focus_on", ""))

            st.markdown("### Recommended Workshop Focus")
            st.success(latest.get("recommended_workshop_focus", ""))

            st.markdown("### Opening Line")
            st.info(latest.get("opening_line", ""))
        else:
            st.info("No room summary generated yet.")

# =============================
# Tab 3: Personalized Emails
# =============================
with tab3:
    st.header("Personalized Emails")
    st.write("Generate, review, edit, and send personalized action-plan emails.")

    if output_df.empty:
        st.info("No participant records available yet.")
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            generate_missing = st.button("✨ Generate Missing Emails", type="primary")
        with col2:
            regenerate_all = st.button("♻️ Regenerate All Emails")
        with col3:
            status_filter = st.selectbox("Filter by status", ["All", "New", "Generated", "Reviewed", "Sent", "Failed", "Skipped"])

        if generate_missing or regenerate_all:
            working_df = output_df.copy()
            targets = working_df.index if regenerate_all else working_df[working_df["email_body"].astype(str).str.strip() == ""].index
            progress = st.progress(0)
            for i, idx in enumerate(targets):
                row = working_df.loc[idx].to_dict()
                consent = safe_str(row.get("consent_followup")).lower()
                if consent.startswith("no"):
                    working_df.at[idx, "status"] = "Skipped"
                    working_df.at[idx, "error_message"] = "Participant opted out of follow-up"
                    continue
                try:
                    result = generate_personal_email(row)
                    for k, v in result.items():
                        working_df.at[idx, k] = v
                    working_df.at[idx, "status"] = "Generated"
                    working_df.at[idx, "reviewed"] = "No"
                    working_df.at[idx, "error_message"] = ""
                    working_df.at[idx, "last_updated"] = now_str()
                except Exception as e:
                    working_df.at[idx, "status"] = "Failed"
                    working_df.at[idx, "error_message"] = str(e)
                    working_df.at[idx, "last_updated"] = now_str()
                progress.progress((i + 1) / max(len(targets), 1))
            write_ai_output(working_df)
            st.success("Email generation complete.")
            st.rerun()

        filtered_df = output_df.copy()
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        st.write(f"Showing {len(filtered_df)} participant(s)")

        for idx, row in filtered_df.iterrows():
            title = f"{row.get('name', 'No name')} | {row.get('audience_type', '')} | {row.get('status', 'New')}"
            with st.expander(title):
                st.write(f"**Email:** {row.get('email', '')}")
                st.write(f"**AI Level:** {row.get('ai_level', '')}")
                st.write(f"**Interested topics:** {row.get('interested_topics', '')}")
                st.write(f"**Main need:** {row.get('main_need', '')}")
                st.write(f"**Recommended sections:** {row.get('recommended_sections', '')}")
                st.write(f"**Question/Concern:** {row.get('ai_question', '')}")

                # Streamlit widget values persist in session_state after first render.
                # If an email was generated after the widgets first appeared, the old empty
                # session_state value can override the sheet value. This block hydrates the
                # editable fields from Google Sheet whenever the widget is blank or the
                # generated sheet content has changed.
                subject_key = f"subject_{row['response_id']}"
                body_key = f"body_{row['response_id']}"
                source_subject = safe_str(row.get("email_subject", ""))
                source_body = safe_str(row.get("email_body", ""))
                subject_source_key = f"subject_source_{row['response_id']}"
                body_source_key = f"body_source_{row['response_id']}"

                if (
                    subject_key not in st.session_state
                    or safe_str(st.session_state.get(subject_key, "")).strip() == ""
                    or st.session_state.get(subject_source_key) != source_subject
                ):
                    st.session_state[subject_key] = source_subject
                    st.session_state[subject_source_key] = source_subject

                if (
                    body_key not in st.session_state
                    or safe_str(st.session_state.get(body_key, "")).strip() == ""
                    or st.session_state.get(body_source_key) != source_body
                ):
                    st.session_state[body_key] = source_body
                    st.session_state[body_source_key] = source_body

                subject = st.text_input("Subject", key=subject_key)
                body = st.text_area("Email body", height=360, key=body_key)

                b1, b2, b3, b4 = st.columns(4)
                if b1.button("💾 Save Edit", key=f"save_{row['response_id']}"):
                    output_df.loc[output_df["response_id"] == row["response_id"], "email_subject"] = subject
                    output_df.loc[output_df["response_id"] == row["response_id"], "email_body"] = body
                    output_df.loc[output_df["response_id"] == row["response_id"], "last_updated"] = now_str()
                    write_ai_output(output_df)
                    st.success("Saved.")
                    st.rerun()

                if b2.button("✅ Mark Reviewed", key=f"review_{row['response_id']}"):
                    output_df.loc[output_df["response_id"] == row["response_id"], "email_subject"] = subject
                    output_df.loc[output_df["response_id"] == row["response_id"], "email_body"] = body
                    output_df.loc[output_df["response_id"] == row["response_id"], "reviewed"] = "Yes"
                    output_df.loc[output_df["response_id"] == row["response_id"], "status"] = "Reviewed"
                    output_df.loc[output_df["response_id"] == row["response_id"], "last_updated"] = now_str()
                    write_ai_output(output_df)
                    st.success("Marked reviewed.")
                    st.rerun()

                if b3.button("📨 Send Now", key=f"send_{row['response_id']}"):
                    if not safe_str(subject).strip() or not safe_str(body).strip():
                        st.error("Subject and email body cannot be empty. Please generate or edit the email before sending.")
                    else:
                        ok, msg = send_email(row.get("email", ""), subject, body)
                        mask = output_df["response_id"] == row["response_id"]
                        output_df.loc[mask, "email_subject"] = subject
                        output_df.loc[mask, "email_body"] = body
                        output_df.loc[mask, "status"] = "Sent" if ok else "Failed"
                        output_df.loc[mask, "reviewed"] = "Yes" if ok else output_df.loc[mask, "reviewed"]
                        output_df.loc[mask, "sent_at"] = now_str() if ok else ""
                        output_df.loc[mask, "error_message"] = "" if ok else msg
                        output_df.loc[mask, "last_updated"] = now_str()
                        write_ai_output(output_df)
                        if ok:
                            st.success("Email sent.")
                        else:
                            st.error(f"Send failed: {msg}")
                        st.rerun()

                if b4.button("⏭️ Skip", key=f"skip_{row['response_id']}"):
                    output_df.loc[output_df["response_id"] == row["response_id"], "status"] = "Skipped"
                    output_df.loc[output_df["response_id"] == row["response_id"], "last_updated"] = now_str()
                    write_ai_output(output_df)
                    st.success("Skipped.")
                    st.rerun()

# =============================
# Tab 4: Send Center
# =============================
with tab4:
    st.header("Send Center")
    st.write("Controlled sending. Recommended: send test first, then send only reviewed emails.")

    if output_df.empty:
        st.info("No emails available yet.")
    else:
        status_counts = output_df["status"].replace("", "New").value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.dataframe(status_counts, use_container_width=True, hide_index=True)

        st.markdown("### Send test email")
        test_to = st.text_input("Test recipient email", value=st.secrets.get("GMAIL_USER", ""))
        sample_rows = output_df[output_df["email_body"].astype(str).str.strip() != ""]
        if st.button("📨 Send first generated email as test"):
            if sample_rows.empty:
                st.warning("No generated email available to test.")
            else:
                sample = sample_rows.iloc[0]
                ok, msg = send_email(test_to, "TEST - " + sample.get("email_subject", "AI Workshop Follow-up"), sample.get("email_body", ""))
                if ok:
                    st.success(f"Test email sent to {test_to}.")
                else:
                    st.error(f"Test send failed: {msg}")

        st.markdown("### Send all reviewed emails")
        reviewed_df = output_df[(output_df["status"] == "Reviewed") & (output_df["reviewed"] == "Yes")].copy()
        ready_df = reviewed_df[
            (reviewed_df["email_subject"].astype(str).str.strip() != "")
            & (reviewed_df["email_body"].astype(str).str.strip() != "")
            & (reviewed_df["email"].astype(str).str.strip() != "")
        ]
        blocked_df = reviewed_df.drop(ready_df.index)
        st.write(f"Reviewed: **{len(reviewed_df)}** | Ready to send: **{len(ready_df)}** | Blocked because email/subject/body is empty: **{len(blocked_df)}**")

        confirm = st.text_input("Type SEND to confirm bulk sending", value="")
        if st.button("🚀 Send All Reviewed", type="primary"):
            if confirm != "SEND":
                st.error("Type SEND exactly to confirm.")
            elif ready_df.empty:
                st.warning("No reviewed emails with non-empty recipient, subject, and body are ready to send.")
            else:
                working_df = output_df.copy()
                progress = st.progress(0)
                sent_count = 0
                failed_count = 0
                for i, (_, row) in enumerate(ready_df.iterrows()):
                    ok, msg = send_email(row.get("email", ""), row.get("email_subject", ""), row.get("email_body", ""))
                    mask = working_df["response_id"] == row["response_id"]
                    working_df.loc[mask, "status"] = "Sent" if ok else "Failed"
                    working_df.loc[mask, "sent_at"] = now_str() if ok else ""
                    working_df.loc[mask, "error_message"] = "" if ok else msg
                    working_df.loc[mask, "last_updated"] = now_str()
                    sent_count += 1 if ok else 0
                    failed_count += 0 if ok else 1
                    progress.progress((i + 1) / max(len(ready_df), 1))
                write_ai_output(working_df)
                st.success(f"Bulk send complete. Sent: {sent_count}, Failed: {failed_count}")
                st.rerun()

        st.markdown("### Email Log")
        log_cols = ["name", "email", "status", "reviewed", "sent_at", "error_message", "last_updated"]
        st.dataframe(output_df[log_cols], use_container_width=True, hide_index=True)

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download AI Output CSV",
            data=csv,
            file_name="ai_workshop_email_log.csv",
            mime="text/csv",
        )
