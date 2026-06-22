"""
AI Project Risk Monitor - Streamlit Application
================================================
A prototype that monitors project tasks and flags risks, dependencies,
and blockers using rule-based AI logic with sample GPT/Gemini prompts.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import re

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Project Risk Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130 0%, #252836 100%);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #8b9dc3 !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e8eaf6;
        padding: 10px 0 6px 0;
        border-bottom: 2px solid #3d4166;
        margin-bottom: 16px;
    }

    /* Risk badges */
    .badge-red    { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.8rem; }
    .badge-yellow { background:#ffa50022; color:#ffcc44; border:1px solid #ffa50055; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.8rem; }
    .badge-green  { background:#21c35422; color:#21c354; border:1px solid #21c35455; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.8rem; }
    .badge-blue   { background:#4b8bff22; color:#4b8bff; border:1px solid #4b8bff55; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.8rem; }

    /* Alert boxes */
    .alert-red    { background:#ff4b4b15; border-left:4px solid #ff4b4b; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
    .alert-yellow { background:#ffa50015; border-left:4px solid #ffcc44;  border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
    .alert-green  { background:#21c35415; border-left:4px solid #21c354; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }

    /* AI response box */
    .ai-response {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2540 100%);
        border: 1px solid #3d5a8a;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #c8d8f0;
    }
    .ai-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #7b9fd4;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* Prompt code block */
    .prompt-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #79c0ff;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* Summary card */
    .summary-card {
        background: linear-gradient(135deg, #1a1f35 0%, #252040 100%);
        border: 1px solid #5a3d9a;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 10px 0;
        color: #d4c8f0;
        line-height: 1.8;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("project_data.csv")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["due_date"]   = pd.to_datetime(df["due_date"])
    df["last_update"]= pd.to_datetime(df["last_update"])
    df["dependencies"] = df["dependencies"].fillna("")
    return df


# ── Risk analysis engine ────────────────────────────────────────────────────────
def analyze_risks(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """
    Rule-based risk classification.
    Returns the dataframe with new columns:
        risk_level    : 'BLOCKED' | 'AT RISK' | 'ON TRACK' | 'COMPLETE'
        risk_reasons  : list[str]
        days_to_due   : int
    """
    today_dt = pd.Timestamp(today)
    completed_ids = set(df.loc[df["status"] == "Complete", "task_id"])

    risk_levels  = []
    risk_reasons = []

    for _, row in df.iterrows():
        reasons = []

        # Already complete
        if row["status"] == "Complete":
            risk_levels.append("COMPLETE")
            risk_reasons.append(reasons)
            continue

        # Dependency check
        if row["dependencies"]:
            dep_ids = [d.strip() for d in row["dependencies"].split(";") if d.strip()]
            incomplete_deps = [d for d in dep_ids if d not in completed_ids]
            if incomplete_deps:
                reasons.append(f"Waiting on incomplete deps: {', '.join(incomplete_deps)}")

        # Overdue check
        if row["due_date"] < today_dt and row["status"] != "Complete":
            overdue_days = (today_dt - row["due_date"]).days
            reasons.append(f"Overdue by {overdue_days} day(s)")

        # Near deadline with low completion
        days_left = (row["due_date"] - today_dt).days
        if 0 <= days_left <= 7 and row["completion_pct"] < 80:
            reasons.append(
                f"Due in {days_left} day(s) but only {row['completion_pct']}% complete"
            )

        # Stale update (no update in 7+ days, not complete)
        stale_days = (today_dt - row["last_update"]).days
        if stale_days >= 7 and row["status"] not in ("Complete", "Not Started"):
            reasons.append(f"No update in {stale_days} day(s)")

        # Explicitly blocked
        if row["status"] == "Blocked":
            if not any("dep" in r.lower() for r in reasons):
                reasons.append("Task marked as Blocked by assignee")

        # Classify
        if row["status"] == "Blocked" or any("dep" in r.lower() or "blocked" in r.lower() for r in reasons):
            risk_levels.append("BLOCKED")
        elif reasons:
            risk_levels.append("AT RISK")
        else:
            risk_levels.append("ON TRACK")

        risk_reasons.append(reasons)

    df = df.copy()
    df["risk_level"]  = risk_levels
    df["risk_reasons"] = risk_reasons
    df["days_to_due"]  = (pd.to_datetime(df["due_date"]) - today_dt).dt.days
    return df


def find_dependency_chains(df: pd.DataFrame) -> dict:
    """Identify tasks that are blocking 2+ downstream tasks."""
    blocker_count: dict[str, int] = {}
    for _, row in df.iterrows():
        if row["dependencies"]:
            for dep in row["dependencies"].split(";"):
                dep = dep.strip()
                if dep:
                    blocker_count[dep] = blocker_count.get(dep, 0) + 1
    return {k: v for k, v in blocker_count.items() if v >= 2}


# ── ChatGPT / Gemini prompt templates ──────────────────────────────────────────
PROMPT_RISK_ANALYSIS = """You are a senior project manager AI assistant.

Analyze the following project task data and identify:
1. Tasks that are at risk of missing their deadline
2. Hidden dependencies that could cause cascade delays
3. Resource bottlenecks (one person owning too many critical tasks)
4. Overall project health score (0-100)

Project tasks (JSON):
{tasks_json}

Today's date: {today}

Respond with:
- Executive summary (2-3 sentences)
- Top 3 risks with severity (HIGH/MEDIUM/LOW)
- Recommended immediate actions
- Predicted project completion date based on current velocity
"""

PROMPT_BLOCKER_RESOLUTION = """You are an expert in Agile project management.

The following task is BLOCKED and preventing {downstream_count} other tasks from starting:

Task: {task_name}
Assignee: {assignee}
Blocked since: {last_update}
Blocking downstream tasks: {downstream_tasks}
Block reason: {block_reason}

Generate:
1. A concise Slack message to the assignee requesting an update
2. Three potential solutions to unblock this task
3. Suggested escalation path if not resolved in 24 hours
4. Impact assessment if this remains blocked for 3 more days
"""

PROMPT_WEEKLY_SUMMARY = """Generate a weekly project status update email for stakeholders.

Project data:
- Total tasks: {total_tasks}
- Completed: {completed}
- In Progress: {in_progress}
- Blocked: {blocked}
- At Risk: {at_risk}
- On Track: {on_track}
- Overall completion: {avg_completion:.1f}%

Critical blocked tasks: {critical_blocked}
Upcoming deadlines (next 7 days): {upcoming_deadlines}

Write a professional, concise status update (150 words max) suitable for
executive stakeholders. Use clear language, highlight urgency where needed,
and end with a clear next steps section.
"""


# ── Simulated AI responses ──────────────────────────────────────────────────────
def generate_simulated_gpt_response(df: pd.DataFrame, today: date) -> str:
    blocked  = df[df["risk_level"] == "BLOCKED"]
    at_risk  = df[df["risk_level"] == "AT RISK"]
    on_track = df[df["risk_level"] == "ON TRACK"]
    complete = df[df["risk_level"] == "COMPLETE"]
    avg_pct  = df[df["status"] != "Complete"]["completion_pct"].mean()

    top_blocker = blocked["task_name"].iloc[0] if len(blocked) > 0 else "None"

    return f"""**Executive Summary:**
The project shows moderate risk with {len(blocked)} blocked task(s) and {len(at_risk)} task(s)
at risk of missing deadlines. Overall in-progress completion stands at {avg_pct:.0f}%,
which is below the expected threshold for this project phase.

**Top 3 Risks:**

1. 🔴 HIGH — "{top_blocker}" is blocked and creating a cascade delay affecting downstream
   tasks including Security Audit and Payment Integration. Immediate escalation recommended.

2. 🟡 MEDIUM — Backend API Development ({df[df['task_id']=='T004']['completion_pct'].values[0]}% complete)
   is a dependency bottleneck: 4 other tasks cannot start until it completes. Current velocity
   suggests a 5–8 day delay.

3. 🟡 MEDIUM — Frontend UI and Admin Dashboard share a dependency on an incomplete Backend API,
   creating synchronized risk. A delay in either propagates to UAT and Go-Live.

**Recommended Immediate Actions:**
- Assign a senior developer to unblock "{top_blocker}" within 24 hours
- Schedule a dependency review meeting for all CRITICAL-priority tasks
- Reassign Documentation (T016) to free up capacity for testing tasks (T011, T012)

**Predicted Completion:** Based on current velocity, the project is tracking approximately
10–14 days behind the planned Go-Live date of 2026-07-20.
"""


def generate_simulated_gemini_response(df: pd.DataFrame, today: date) -> str:
    blocked  = df[df["risk_level"] == "BLOCKED"]
    at_risk  = df[df["risk_level"] == "AT RISK"]
    complete = df[df["risk_level"] == "COMPLETE"]
    today_dt = pd.Timestamp(today)

    upcoming = df[
        (df["days_to_due"] >= 0) & (df["days_to_due"] <= 7) &
        (df["status"] != "Complete")
    ]["task_name"].tolist()

    return f"""**Gemini Project Intelligence Report**

**Project Health Score: 58/100** ⚠️

**Pattern Analysis:**
I've identified a critical dependency chain: T003 → T004 → [T007, T008, T009, T010, T011]
This single chain represents 8 of your 20 tasks (40% of project scope). Any further delay
in Backend API Development (T004, currently {df[df['task_id']=='T004']['completion_pct'].values[0]}%)
will compress the testing timeline and jeopardize your July 20 launch.

**Anomaly Detected:**
t006_overdue = max(0, (today_dt - t006_due).days)   # computed BEFORE f-string
t006_overdue_str = f"{t006_overdue} day(s)"
# then in f-string:
Authentication Module (T006) was due {t006_overdue_str} ago
and sits at 80% — a final 20% that commonly hides the hardest integration work. Security Audit
(T015) and Payment Integration (T007) are directly blocked by this incomplete task.

**Risk Heatmap:**
- CRITICAL path tasks at risk: T004, T006, T007
- Downstream exposure: T013, T015, T017, T018, T019, T020 (6 tasks)
- Assignees under pressure: David Kim (T004), Grace Kim (T007)

**Upcoming Deadlines (next 7 days):**
{', '.join(upcoming) if upcoming else 'No immediate deadlines — but multiple tasks are already overdue.'}

**Gemini Recommendation:**
Consider a scope review for the July 20 launch. Based on the dependency graph, a realistic
completion date with current velocity is August 1–5, 2026. Running T011 and T012 testing
in parallel earlier (before full API completion) could recover 3–4 days.
"""


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("AI Risk Monitor")
    st.caption("Project Intelligence Dashboard")
    st.divider()

    today_input = st.date_input(
        "Simulation date (Today)",
        value=date(2026, 6, 22),
        help="Adjust this to simulate risk analysis at different points in time.",
    )
    st.divider()

    st.markdown("**Filter tasks**")
    show_statuses = st.multiselect(
        "Status filter",
        options=["Complete", "In Progress", "Not Started", "Blocked"],
        default=["In Progress", "Not Started", "Blocked"],
    )
    show_priorities = st.multiselect(
        "Priority filter",
        options=["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"],
    )
    st.divider()
    st.markdown("**About**")
    st.caption(
        "This prototype uses rule-based AI logic to flag project risks. "
        "Prompt templates show how real ChatGPT/Gemini API calls would work."
    )


# ── Main content ────────────────────────────────────────────────────────────────
st.markdown("# 🤖 AI Project Risk Monitor")
st.markdown(
    "Real-time project intelligence — flagging blockers, dependencies, "
    "and deadline risks before they derail your sprint."
)

df_raw  = load_data()
df      = analyze_risks(df_raw, today_input)
chains  = find_dependency_chains(df)

# Apply sidebar filters
df_view = df[
    df["status"].isin(show_statuses) &
    df["priority"].isin(show_priorities)
]

# ── 1. METRICS CARDS ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Project Overview</div>', unsafe_allow_html=True)

total     = len(df)
complete  = len(df[df["risk_level"] == "COMPLETE"])
blocked   = len(df[df["risk_level"] == "BLOCKED"])
at_risk   = len(df[df["risk_level"] == "AT RISK"])
on_track  = len(df[df["risk_level"] == "ON TRACK"])
avg_pct   = df["completion_pct"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Tasks",    total,    help="All tasks in the project")
col2.metric("Complete",       complete, help="Tasks marked Complete")
col3.metric("In Progress",    len(df[df["status"] == "In Progress"]),
            help="Currently active tasks")
col4.metric("🔴 Blocked",      blocked,  delta=f"-{blocked} critical",
            delta_color="inverse", help="Tasks that cannot proceed")
col5.metric("🟡 At Risk",       at_risk,  delta=f"-{at_risk} warnings",
            delta_color="inverse", help="Tasks approaching deadline or stale")
col6.metric("Avg Completion", f"{avg_pct:.0f}%", help="Mean completion across all tasks")

st.divider()


# ── 2. RISK SUMMARY PANEL ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚦 Risk Summary</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])

with c1:
    # Health gauge
    health_score = max(0, min(100, int(
        (complete / total * 40) +
        (on_track  / total * 35) +
        (at_risk   / total * 15) +
        (blocked   / total * 0)
    )))
    color = "#21c354" if health_score >= 70 else "#ffcc44" if health_score >= 45 else "#ff4b4b"
    st.markdown(f"""
    <div style="text-align:center; padding:20px; background:#1e2130; border-radius:12px;
                border:1px solid #3d4166;">
        <div style="font-size:0.8rem; color:#8b9dc3; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:8px;">Project Health Score</div>
        <div style="font-size:3.5rem; font-weight:800; color:{color};">{health_score}</div>
        <div style="font-size:0.85rem; color:{color}; margin-top:4px;">
            {"Healthy" if health_score >= 70 else "Needs Attention" if health_score >= 45 else "Critical"}
        </div>
        <div style="margin-top:12px; background:#2a2e42; border-radius:100px; height:8px;">
            <div style="width:{health_score}%; height:8px; border-radius:100px;
                        background:linear-gradient(90deg,{color}88,{color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    # Risk breakdown
    if blocked > 0:
        blocked_names = df[df["risk_level"] == "BLOCKED"]["task_name"].tolist()
        st.markdown(
            f'<div class="alert-red">🔴 <strong>BLOCKED ({blocked})</strong> — '
            f'{", ".join(blocked_names)}</div>',
            unsafe_allow_html=True,
        )
    if at_risk > 0:
        risk_names = df[df["risk_level"] == "AT RISK"]["task_name"].tolist()
        st.markdown(
            f'<div class="alert-yellow">🟡 <strong>AT RISK ({at_risk})</strong> — '
            f'{", ".join(risk_names)}</div>',
            unsafe_allow_html=True,
        )
    if on_track > 0:
        track_names = df[df["risk_level"] == "ON TRACK"]["task_name"].tolist()
        st.markdown(
            f'<div class="alert-green">🟢 <strong>ON TRACK ({on_track})</strong> — '
            f'{", ".join(track_names[:4])}{"..." if len(track_names) > 4 else ""}</div>',
            unsafe_allow_html=True,
        )
    if chains:
        chain_list = []
        for tid, cnt in sorted(chains.items(), key=lambda x: -x[1]):
            name_rows = df[df["task_id"] == tid]["task_name"]
            name = name_rows.values[0] if len(name_rows) else tid
            chain_list.append(f"{name} ({cnt} dependents)")
        st.markdown(
            f'<div class="alert-yellow">🔗 <strong>DEPENDENCY CHAINS</strong> — '
            f'{"; ".join(chain_list)}</div>',
            unsafe_allow_html=True,
        )

st.divider()


# ── 3. TASK TABLE ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Task Dashboard</div>', unsafe_allow_html=True)

RISK_EMOJI = {"BLOCKED": "🔴", "AT RISK": "🟡", "ON TRACK": "🟢", "COMPLETE": "✅"}

display_cols = [
    "task_id", "task_name", "assignee", "status", "priority",
    "due_date", "completion_pct", "risk_level", "days_to_due"
]

df_display = df_view[display_cols].copy()
df_display["risk_flag"] = df_display["risk_level"].map(RISK_EMOJI)
df_display["due_date"]  = df_display["due_date"].dt.strftime("%b %d, %Y")
df_display["completion_pct"] = df_display["completion_pct"].astype(str) + "%"
df_display["days_to_due"] = df_display["days_to_due"].apply(
    lambda x: f"{x}d" if x >= 0 else f"Overdue {abs(x)}d"
)

df_display = df_display.rename(columns={
    "task_id": "ID", "task_name": "Task", "assignee": "Assignee",
    "status": "Status", "priority": "Priority", "due_date": "Due Date",
    "completion_pct": "Complete %", "risk_level": "Risk Level",
    "days_to_due": "Days Left", "risk_flag": "Flag",
})

def style_risk(val):
    color_map = {
        "BLOCKED": "background-color: #ff4b4b22; color: #ff6b6b; font-weight: bold;",
        "AT RISK": "background-color: #ffa50022; color: #ffcc44; font-weight: bold;",
        "ON TRACK": "background-color: #21c35422; color: #21c354; font-weight: bold;",
        "COMPLETE": "background-color: #4b8bff22; color: #6ba3ff; font-weight: bold;",
    }
    return color_map.get(val, "")

styled = df_display.style.map(style_risk, subset=["Risk Level"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

st.divider()


# ── 4. AI RISK ANALYSIS ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 AI Risk Analysis</div>', unsafe_allow_html=True)
st.markdown(
    "Rule-based analysis runs automatically. The prompt templates below show "
    "exactly how **ChatGPT (GPT-4o)** or **Gemini 1.5 Pro** would be called "
    "with real API keys."
)

tab1, tab2, tab3 = st.tabs(["🔴 Blocked Tasks", "🟡 At-Risk Tasks", "🔗 Dependency Chains"])

# ─── Tab 1: Blocked Tasks ────────────────────────────────────────────────────
with tab1:
    df_blocked = df[df["risk_level"] == "BLOCKED"]
    if df_blocked.empty:
        st.success("No blocked tasks detected.")
    else:
        for _, row in df_blocked.iterrows():
            reasons = row["risk_reasons"]
            with st.expander(
                f"🔴 **{row['task_id']}** — {row['task_name']}  "
                f"| Assignee: {row['assignee']}  | Priority: {row['priority']}"
            ):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Status", row["status"])
                col_b.metric("Completion", f"{row['completion_pct']}%")
                col_c.metric("Days to Due", str(row["days_to_due"]) + "d")

                st.markdown("**Risk Reasons:**")
                for r in reasons:
                    st.markdown(f"- ⚠️ {r}")

                # Downstream count
                downstream = [
                    r["task_name"] for _, r in df.iterrows()
                    if row["task_id"] in r["dependencies"].split(";")
                ]
                if downstream:
                    st.markdown(f"**Blocking downstream tasks:** {', '.join(downstream)}")

                # Prompt template
                st.markdown("**ChatGPT Prompt Template:**")
                prompt_filled = PROMPT_BLOCKER_RESOLUTION.format(
                    downstream_count=len(downstream),
                    task_name=row["task_name"],
                    assignee=row["assignee"],
                    last_update=row["last_update"].strftime("%Y-%m-%d"),
                    downstream_tasks=", ".join(downstream) if downstream else "None",
                    block_reason="; ".join(reasons) if reasons else "Unknown",
                )
                st.markdown(
                    f'<div class="prompt-box">{prompt_filled}</div>',
                    unsafe_allow_html=True,
                )


# ─── Tab 2: At-Risk Tasks ────────────────────────────────────────────────────
with tab2:
    df_at_risk = df[df["risk_level"] == "AT RISK"]
    if df_at_risk.empty:
        st.success("No at-risk tasks detected.")
    else:
        for _, row in df_at_risk.iterrows():
            reasons = row["risk_reasons"]
            color = "#ff4b4b" if row["priority"] == "Critical" else "#ffcc44"
            with st.expander(
                f"🟡 **{row['task_id']}** — {row['task_name']}  "
                f"| Assignee: {row['assignee']}  | Priority: {row['priority']}"
            ):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Status", row["status"])
                col_b.metric("Completion", f"{row['completion_pct']}%")
                col_c.metric("Days Left", f"{row['days_to_due']}d")
                col_d.metric("Priority", row["priority"])

                st.markdown("**Risk Reasons:**")
                for r in reasons:
                    st.markdown(f"- ⚠️ {r}")


# ─── Tab 3: Dependency Chains ────────────────────────────────────────────────
with tab3:
    if not chains:
        st.success("No critical dependency chains detected.")
    else:
        st.markdown("Tasks acting as **critical hubs** — blocking 2+ downstream tasks.")
        for tid, count in sorted(chains.items(), key=lambda x: -x[1]):
            name_rows = df[df["task_id"] == tid]
            if name_rows.empty:
                continue
            row = name_rows.iloc[0]
            downstream_names = [
                r["task_name"] for _, r in df.iterrows()
                if tid in r["dependencies"].split(";")
            ]
            with st.expander(
                f"🔗 **{tid}** — {row['task_name']}  "
                f"| Blocking **{count}** task(s)  | Status: {row['status']}"
            ):
                col_a, col_b = st.columns(2)
                col_a.metric("Completion", f"{row['completion_pct']}%")
                col_b.metric("Risk Level", row["risk_level"])
                st.markdown(f"**Downstream tasks waiting:** {', '.join(downstream_names)}")
                st.markdown(
                    f"**Impact:** If {row['task_name']} is delayed, "
                    f"{count} task(s) cannot start, potentially affecting "
                    f"{count * 1} assignee(s) and cascading through the project timeline."
                )

st.divider()


# ── 5. WHAT AI WOULD SAY ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">💬 What AI Would Say</div>', unsafe_allow_html=True)
st.markdown(
    "Below are **simulated AI responses** matching what GPT-4o and Gemini 1.5 Pro "
    "would generate when given the project data and prompts above."
)

ai_tab1, ai_tab2 = st.tabs(["🤖 ChatGPT (GPT-4o) Response", "✨ Gemini 1.5 Pro Response"])

with ai_tab1:
    st.markdown('<div class="ai-label">⚡ Simulated GPT-4o Output (rule-based)</div>',
                unsafe_allow_html=True)
    gpt_response = generate_simulated_gpt_response(df, today_input)
    st.markdown(
        f'<div class="ai-response">{gpt_response}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📋 View Full GPT-4o Prompt Template"):
        tasks_sample = df[["task_id","task_name","status","completion_pct","due_date"]].head(5).to_json(orient="records", indent=2)
        prompt_preview = PROMPT_RISK_ANALYSIS.format(
            tasks_json=tasks_sample + "\n  ... (all 20 tasks)",
            today=today_input.strftime("%Y-%m-%d"),
        )
        st.markdown(f'<div class="prompt-box">{prompt_preview}</div>', unsafe_allow_html=True)

with ai_tab2:
    st.markdown('<div class="ai-label">✨ Simulated Gemini 1.5 Pro Output (rule-based)</div>',
                unsafe_allow_html=True)
    gemini_response = generate_simulated_gemini_response(df, today_input)
    st.markdown(
        f'<div class="ai-response">{gemini_response}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📋 View Weekly Summary Prompt Template"):
        critical_blocked = df[
            (df["risk_level"] == "BLOCKED") & (df["priority"] == "Critical")
        ]["task_name"].tolist()
        upcoming_deadlines = df[
            (df["days_to_due"] >= 0) & (df["days_to_due"] <= 7) &
            (df["status"] != "Complete")
        ]["task_name"].tolist()
        prompt_weekly = PROMPT_WEEKLY_SUMMARY.format(
            total_tasks=total,
            completed=complete,
            in_progress=len(df[df["status"] == "In Progress"]),
            blocked=blocked,
            at_risk=at_risk,
            on_track=on_track,
            avg_completion=avg_pct,
            critical_blocked=", ".join(critical_blocked) if critical_blocked else "None",
            upcoming_deadlines=", ".join(upcoming_deadlines) if upcoming_deadlines else "None",
        )
        st.markdown(f'<div class="prompt-box">{prompt_weekly}</div>', unsafe_allow_html=True)

st.divider()


# ── 6. HOW TO CONNECT REAL AI APIS ────────────────────────────────────────────
with st.expander("🔌 How to Connect Real AI APIs (Code Snippets)"):
    st.markdown("### ChatGPT API Integration (Python)")
    st.code("""
import openai

client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

def analyze_project_with_gpt(tasks_json: str, today: str) -> str:
    prompt = PROMPT_RISK_ANALYSIS.format(
        tasks_json=tasks_json,
        today=today,
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior project management AI."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content
""", language="python")

    st.markdown("### Gemini API Integration (Python)")
    st.code("""
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

def analyze_project_with_gemini(tasks_json: str, today: str) -> str:
    prompt = PROMPT_RISK_ANALYSIS.format(
        tasks_json=tasks_json,
        today=today,
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1000,
        ),
    )
    return response.text
""", language="python")

st.divider()


# ── 7. 100-WORD SUMMARY ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 100-Word Summary</div>', unsafe_allow_html=True)

summary_text = """
This AI Project Risk Monitor prototype demonstrates how artificial intelligence can proactively
surface project risks before they escalate. The system ingests structured task data — including
statuses, dependencies, deadlines, and completion percentages — and applies intelligent rules to
classify each task as Blocked, At Risk, or On Track. It identifies critical dependency chains
where a single delayed task cascades across multiple workstreams. The dashboard pairs this analysis
with ready-to-use ChatGPT and Gemini prompt templates, showing exactly how teams could integrate
large language models for automated blocker resolution messages, stakeholder summaries, and velocity
predictions. Built on Google Sheets, Python, and Streamlit — entirely free, shareable, and deployable
in minutes.
"""

word_count = len(summary_text.split())
st.markdown(
    f'<div class="summary-card">{summary_text.strip()}'
    f'<div style="margin-top:12px; font-size:0.75rem; color:#7b9fd4;">Word count: {word_count}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; color:#555; font-size:0.8rem; padding:20px;">'
    "AI Project Risk Monitor | Built with Streamlit + Python | "
    "Uses simulated rule-based AI with ChatGPT/Gemini prompt templates | "
    "No confidential data"
    "</div>",
    unsafe_allow_html=True,
)
