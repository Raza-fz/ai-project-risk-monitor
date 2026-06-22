"""
AI Project Risk Monitor - Streamlit Application
"""
import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="AI Project Risk Monitor", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2130 0%, #252836 100%);
    border: 1px solid #3d4166; border-radius: 12px; padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #8b9dc3 !important; font-size:0.85rem; font-weight:600; text-transform:uppercase; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { color:#fff !important; font-size:2rem !important; font-weight:700; }
.section-header { font-size:1.4rem; font-weight:700; color:#e8eaf6; padding:10px 0 6px 0; border-bottom:2px solid #3d4166; margin-bottom:16px; }
.alert-red    { background:#ff4b4b15; border-left:4px solid #ff4b4b; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
.alert-yellow { background:#ffa50015; border-left:4px solid #ffcc44; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
.alert-green  { background:#21c35415; border-left:4px solid #21c354; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; }
.ai-response  { background:linear-gradient(135deg,#1a1f35,#1e2540); border:1px solid #3d5a8a; border-radius:10px; padding:16px 20px; margin:10px 0; font-size:0.92rem; line-height:1.7; color:#c8d8f0; }
.ai-label     { font-size:0.75rem; font-weight:700; color:#7b9fd4; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; }
.prompt-box   { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:14px; font-family:monospace; font-size:0.82rem; color:#79c0ff; white-space:pre-wrap; line-height:1.6; }
.summary-card { background:linear-gradient(135deg,#1a1f35,#252040); border:1px solid #5a3d9a; border-radius:12px; padding:20px 24px; margin:10px 0; color:#d4c8f0; line-height:1.8; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("project_data.csv")
    df["start_date"]   = pd.to_datetime(df["start_date"])
    df["due_date"]     = pd.to_datetime(df["due_date"])
    df["last_update"]  = pd.to_datetime(df["last_update"])
    df["dependencies"] = df["dependencies"].fillna("")
    return df


def analyze_risks(df, today):
    today_dt      = pd.Timestamp(today)
    completed_ids = set(df.loc[df["status"] == "Complete", "task_id"])
    risk_levels, risk_reasons = [], []

    for _, row in df.iterrows():
        reasons = []
        if row["status"] == "Complete":
            risk_levels.append("COMPLETE")
            risk_reasons.append(reasons)
            continue

        if row["dependencies"]:
            deps = [d.strip() for d in row["dependencies"].split(";") if d.strip()]
            missing = [d for d in deps if d not in completed_ids]
            if missing:
                reasons.append(f"Waiting on: {', '.join(missing)}")

        if row["due_date"] < today_dt:
            reasons.append(f"Overdue by {(today_dt - row['due_date']).days} day(s)")

        days_left = (row["due_date"] - today_dt).days
        if 0 <= days_left <= 7 and row["completion_pct"] < 80:
            reasons.append(f"Due in {days_left}d but only {row['completion_pct']}% done")

        stale = (today_dt - row["last_update"]).days
        if stale >= 7 and row["status"] not in ("Complete", "Not Started"):
            reasons.append(f"No update in {stale} day(s)")

        if row["status"] == "Blocked" and not any("waiting" in r.lower() for r in reasons):
            reasons.append("Marked Blocked by assignee")

        if row["status"] == "Blocked" or any("waiting" in r.lower() or "blocked" in r.lower() for r in reasons):
            risk_levels.append("BLOCKED")
        elif reasons:
            risk_levels.append("AT RISK")
        else:
            risk_levels.append("ON TRACK")

        risk_reasons.append(reasons)

    df = df.copy()
    df["risk_level"]   = risk_levels
    df["risk_reasons"] = risk_reasons
    df["days_to_due"]  = (pd.to_datetime(df["due_date"]) - pd.Timestamp(today)).dt.days
    return df


def find_chains(df):
    count = {}
    for _, row in df.iterrows():
        if row["dependencies"]:
            for d in row["dependencies"].split(";"):
                d = d.strip()
                if d:
                    count[d] = count.get(d, 0) + 1
    return {k: v for k, v in count.items() if v >= 2}


def safe_int(series, default):
    """Return int value from a pandas Series safely."""
    if len(series) > 0:
        return int(series.values[0])
    return default


def days_overdue(df, task_id, today_dt):
    """Return 'X day(s)' string for how overdue a task is. Never raises."""
    rows = df[df["task_id"] == task_id]
    if len(rows) == 0:
        return "several days"
    due = pd.Timestamp(rows["due_date"].values[0])
    n = max(0, (today_dt - due).days)
    return f"{n} day(s)"


PROMPT_RISK = """You are a senior project manager AI assistant.

Analyze the following project task data and identify:
1. Tasks at risk of missing their deadline
2. Hidden dependencies causing cascade delays
3. Resource bottlenecks
4. Overall project health score (0-100)

Project tasks (JSON):
{tasks_json}

Today's date: {today}

Respond with:
- Executive summary (2-3 sentences)
- Top 3 risks with severity (HIGH/MEDIUM/LOW)
- Recommended immediate actions
- Predicted completion date based on current velocity
"""

PROMPT_BLOCKER = """You are an expert in Agile project management.

The following task is BLOCKED and preventing {downstream_count} other tasks from starting:

Task: {task_name}
Assignee: {assignee}
Blocked since: {last_update}
Blocking downstream: {downstream_tasks}
Block reason: {block_reason}

Generate:
1. A concise Slack message to the assignee requesting an update
2. Three potential solutions to unblock this task
3. Suggested escalation path if not resolved in 24 hours
4. Impact if this remains blocked for 3 more days
"""

PROMPT_WEEKLY = """Generate a weekly project status update email for stakeholders.

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

Write a professional, concise update (150 words max) for executive stakeholders.
"""


def gpt_response(df, today):
    blocked   = df[df["risk_level"] == "BLOCKED"]
    at_risk   = df[df["risk_level"] == "AT RISK"]
    avg_pct   = df[df["status"] != "Complete"]["completion_pct"].mean()
    top       = blocked["task_name"].iloc[0] if len(blocked) > 0 else "None"
    t004_val  = safe_int(df[df["task_id"] == "T004"]["completion_pct"], 65)

    return (
        f"**Executive Summary:**\n"
        f"The project shows moderate risk with {len(blocked)} blocked task(s) and "
        f"{len(at_risk)} task(s) at risk of missing deadlines. Overall in-progress "
        f"completion stands at {avg_pct:.0f}%, below the expected threshold.\n\n"
        f"**Top 3 Risks:**\n\n"
        f"1. 🔴 HIGH — \"{top}\" is blocked, creating cascade delays affecting "
        f"Security Audit and Payment Integration. Immediate escalation recommended.\n\n"
        f"2. 🟡 MEDIUM — Backend API Development ({t004_val}% complete) is a dependency "
        f"bottleneck: 4 tasks cannot start until it completes. Expect 5–8 day delay.\n\n"
        f"3. 🟡 MEDIUM — Frontend UI and Admin Dashboard share a dependency on the "
        f"incomplete Backend API — a delay propagates to UAT and Go-Live.\n\n"
        f"**Recommended Actions:**\n"
        f"- Assign a senior developer to unblock \"{top}\" within 24 hours\n"
        f"- Schedule a dependency review for all CRITICAL-priority tasks\n"
        f"- Reassign Documentation (T016) to free capacity for testing\n\n"
        f"**Predicted Completion:** ~10–14 days behind planned Go-Live (2026-07-20)."
    )


def gemini_response(df, today):
    today_dt  = pd.Timestamp(today)
    t004_val  = safe_int(df[df["task_id"] == "T004"]["completion_pct"], 65)
    t006_str  = days_overdue(df, "T006", today_dt)

    upcoming = df[
        (df["days_to_due"] >= 0) & (df["days_to_due"] <= 7) &
        (df["status"] != "Complete")
    ]["task_name"].tolist()
    upcoming_str = ", ".join(upcoming) if upcoming else "No immediate deadlines — but multiple tasks are already overdue."

    return (
        f"**Gemini Project Intelligence Report**\n\n"
        f"**Project Health Score: 58/100** ⚠️\n\n"
        f"**Pattern Analysis:**\n"
        f"I've identified a critical dependency chain: T003 → T004 → [T007, T008, T009, T010, T011]. "
        f"This chain represents 8 of 20 tasks (40% of project scope). Any further delay in "
        f"Backend API Development (T004, currently {t004_val}%) will compress the testing "
        f"timeline and jeopardize the July 20 launch.\n\n"
        f"**Anomaly Detected:**\n"
        f"Authentication Module (T006) was due {t006_str} ago and sits at 80% — "
        f"the final 20% commonly hides the hardest integration work. Security Audit (T015) "
        f"and Payment Integration (T007) are directly blocked by this incomplete task.\n\n"
        f"**Risk Heatmap:**\n"
        f"- CRITICAL path tasks at risk: T004, T006, T007\n"
        f"- Downstream exposure: T013, T015, T017, T018, T019, T020 (6 tasks)\n"
        f"- Assignees under pressure: David Kim (T004), Grace Kim (T007)\n\n"
        f"**Upcoming Deadlines (next 7 days):**\n"
        f"{upcoming_str}\n\n"
        f"**Gemini Recommendation:**\n"
        f"A realistic completion date with current velocity is August 1–5, 2026. "
        f"Running T011 and T012 testing in parallel earlier could recover 3–4 days."
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("AI Risk Monitor")
    st.caption("Project Intelligence Dashboard")
    st.divider()
    today_input = st.date_input("Simulation date (Today)", value=date(2026, 6, 22))
    st.divider()
    st.markdown("**Filter tasks**")
    show_statuses = st.multiselect(
        "Status", ["Complete", "In Progress", "Not Started", "Blocked"],
        default=["In Progress", "Not Started", "Blocked"]
    )
    show_priorities = st.multiselect(
        "Priority", ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )
    st.divider()
    st.caption("Rule-based AI logic flags risks. Prompt templates show real ChatGPT/Gemini integration.")


# ── Load & analyse ───────────────────────────────────────────────────────────────
df_raw  = load_data()
df      = analyze_risks(df_raw, today_input)
chains  = find_chains(df)
df_view = df[df["status"].isin(show_statuses) & df["priority"].isin(show_priorities)]

total    = len(df)
complete = len(df[df["risk_level"] == "COMPLETE"])
blocked  = len(df[df["risk_level"] == "BLOCKED"])
at_risk  = len(df[df["risk_level"] == "AT RISK"])
on_track = len(df[df["risk_level"] == "ON TRACK"])
avg_pct  = df["completion_pct"].mean()

# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 AI Project Risk Monitor")
st.markdown("Real-time project intelligence — flagging blockers, dependencies, and deadline risks early.")

# ── Metrics ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Project Overview</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Tasks",    total)
c2.metric("Complete",       complete)
c3.metric("In Progress",    len(df[df["status"] == "In Progress"]))
c4.metric("🔴 Blocked",     blocked,  delta=f"-{blocked} critical",  delta_color="inverse")
c5.metric("🟡 At Risk",     at_risk,  delta=f"-{at_risk} warnings",  delta_color="inverse")
c6.metric("Avg Completion", f"{avg_pct:.0f}%")
st.divider()

# ── Risk summary ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚦 Risk Summary</div>', unsafe_allow_html=True)
col_g, col_r = st.columns([1, 2])

with col_g:
    hs    = max(0, min(100, int((complete/total*40) + (on_track/total*35) + (at_risk/total*15))))
    color = "#21c354" if hs >= 70 else "#ffcc44" if hs >= 45 else "#ff4b4b"
    label = "Healthy" if hs >= 70 else "Needs Attention" if hs >= 45 else "Critical"
    st.markdown(f"""
    <div style="text-align:center;padding:20px;background:#1e2130;border-radius:12px;border:1px solid #3d4166;">
      <div style="font-size:.8rem;color:#8b9dc3;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">Health Score</div>
      <div style="font-size:3.5rem;font-weight:800;color:{color};">{hs}</div>
      <div style="font-size:.85rem;color:{color};margin-top:4px;">{label}</div>
      <div style="margin-top:12px;background:#2a2e42;border-radius:100px;height:8px;">
        <div style="width:{hs}%;height:8px;border-radius:100px;background:linear-gradient(90deg,{color}88,{color});"></div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_r:
    if blocked > 0:
        names = df[df["risk_level"] == "BLOCKED"]["task_name"].tolist()
        st.markdown(f'<div class="alert-red">🔴 <strong>BLOCKED ({blocked})</strong> — {", ".join(names)}</div>', unsafe_allow_html=True)
    if at_risk > 0:
        names = df[df["risk_level"] == "AT RISK"]["task_name"].tolist()
        st.markdown(f'<div class="alert-yellow">🟡 <strong>AT RISK ({at_risk})</strong> — {", ".join(names)}</div>', unsafe_allow_html=True)
    if on_track > 0:
        names = df[df["risk_level"] == "ON TRACK"]["task_name"].tolist()
        st.markdown(f'<div class="alert-green">🟢 <strong>ON TRACK ({on_track})</strong> — {", ".join(names[:4])}{"..." if len(names)>4 else ""}</div>', unsafe_allow_html=True)
    if chains:
        items = []
        for tid, cnt in sorted(chains.items(), key=lambda x: -x[1]):
            nr = df[df["task_id"] == tid]["task_name"]
            items.append(f"{nr.values[0] if len(nr) else tid} ({cnt} dependents)")
        st.markdown(f'<div class="alert-yellow">🔗 <strong>DEPENDENCY CHAINS</strong> — {"; ".join(items)}</div>', unsafe_allow_html=True)

st.divider()

# ── Task table ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Task Dashboard</div>', unsafe_allow_html=True)
EMOJI = {"BLOCKED":"🔴","AT RISK":"🟡","ON TRACK":"🟢","COMPLETE":"✅"}
cols  = ["task_id","task_name","assignee","status","priority","due_date","completion_pct","risk_level","days_to_due"]
dfd   = df_view[cols].copy()
dfd["Flag"]        = dfd["risk_level"].map(EMOJI)
dfd["due_date"]    = dfd["due_date"].dt.strftime("%b %d, %Y")
dfd["completion_pct"] = dfd["completion_pct"].astype(str) + "%"
dfd["days_to_due"] = dfd["days_to_due"].apply(lambda x: f"{x}d" if x >= 0 else f"Overdue {abs(x)}d")
dfd = dfd.rename(columns={"task_id":"ID","task_name":"Task","assignee":"Assignee",
    "status":"Status","priority":"Priority","due_date":"Due Date",
    "completion_pct":"Complete %","risk_level":"Risk Level","days_to_due":"Days Left"})

def style_risk(val):
    m = {"BLOCKED":"background-color:#ff4b4b22;color:#ff6b6b;font-weight:bold;",
         "AT RISK": "background-color:#ffa50022;color:#ffcc44;font-weight:bold;",
         "ON TRACK":"background-color:#21c35422;color:#21c354;font-weight:bold;",
         "COMPLETE":"background-color:#4b8bff22;color:#6ba3ff;font-weight:bold;"}
    return m.get(val, "")

st.dataframe(dfd.style.map(style_risk, subset=["Risk Level"]), use_container_width=True, hide_index=True, height=420)
st.divider()

# ── AI Analysis tabs ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 AI Risk Analysis</div>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["🔴 Blocked Tasks", "🟡 At-Risk Tasks", "🔗 Dependency Chains"])

with t1:
    df_b = df[df["risk_level"] == "BLOCKED"]
    if df_b.empty:
        st.success("No blocked tasks detected.")
    else:
        for _, row in df_b.iterrows():
            reasons = row["risk_reasons"]
            with st.expander(f"🔴 **{row['task_id']}** — {row['task_name']} | {row['assignee']} | {row['priority']}"):
                a,b,c = st.columns(3)
                a.metric("Status",      row["status"])
                b.metric("Completion",  f"{row['completion_pct']}%")
                c.metric("Days to Due", f"{row['days_to_due']}d")
                st.markdown("**Risk Reasons:**")
                for r in reasons:
                    st.markdown(f"- ⚠️ {r}")
                downstream = [r2["task_name"] for _, r2 in df.iterrows() if row["task_id"] in r2["dependencies"].split(";")]
                if downstream:
                    st.markdown(f"**Blocking:** {', '.join(downstream)}")
                prompt = PROMPT_BLOCKER.format(
                    downstream_count=len(downstream), task_name=row["task_name"],
                    assignee=row["assignee"], last_update=row["last_update"].strftime("%Y-%m-%d"),
                    downstream_tasks=", ".join(downstream) or "None",
                    block_reason="; ".join(reasons) or "Unknown",
                )
                st.markdown("**ChatGPT Prompt Template:**")
                st.markdown(f'<div class="prompt-box">{prompt}</div>', unsafe_allow_html=True)

with t2:
    df_a = df[df["risk_level"] == "AT RISK"]
    if df_a.empty:
        st.success("No at-risk tasks detected.")
    else:
        for _, row in df_a.iterrows():
            with st.expander(f"🟡 **{row['task_id']}** — {row['task_name']} | {row['assignee']} | {row['priority']}"):
                a,b,c,d = st.columns(4)
                a.metric("Status",     row["status"])
                b.metric("Completion", f"{row['completion_pct']}%")
                c.metric("Days Left",  f"{row['days_to_due']}d")
                d.metric("Priority",   row["priority"])
                st.markdown("**Risk Reasons:**")
                for r in row["risk_reasons"]:
                    st.markdown(f"- ⚠️ {r}")

with t3:
    if not chains:
        st.success("No critical dependency chains detected.")
    else:
        st.markdown("Tasks blocking **2+ downstream** tasks.")
        for tid, cnt in sorted(chains.items(), key=lambda x: -x[1]):
            nr = df[df["task_id"] == tid]
            if nr.empty:
                continue
            row = nr.iloc[0]
            dn  = [r2["task_name"] for _, r2 in df.iterrows() if tid in r2["dependencies"].split(";")]
            with st.expander(f"🔗 **{tid}** — {row['task_name']} | Blocking {cnt} task(s) | {row['status']}"):
                a,b = st.columns(2)
                a.metric("Completion", f"{row['completion_pct']}%")
                b.metric("Risk Level", row["risk_level"])
                st.markdown(f"**Downstream tasks:** {', '.join(dn)}")

st.divider()

# ── What AI Would Say ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💬 What AI Would Say</div>', unsafe_allow_html=True)
ai1, ai2 = st.tabs(["🤖 ChatGPT (GPT-4o)", "✨ Gemini 1.5 Pro"])

with ai1:
    st.markdown('<div class="ai-label">⚡ Simulated GPT-4o Output</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-response">{gpt_response(df, today_input)}</div>', unsafe_allow_html=True)
    with st.expander("📋 View GPT-4o Prompt Template"):
        sample = df[["task_id","task_name","status","completion_pct","due_date"]].head(5).to_json(orient="records", indent=2)
        st.markdown(f'<div class="prompt-box">{PROMPT_RISK.format(tasks_json=sample + chr(10)+"  ...(all 20 tasks)", today=today_input.strftime("%Y-%m-%d"))}</div>', unsafe_allow_html=True)

with ai2:
    st.markdown('<div class="ai-label">✨ Simulated Gemini 1.5 Pro Output</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-response">{gemini_response(df, today_input)}</div>', unsafe_allow_html=True)
    with st.expander("📋 View Weekly Summary Prompt Template"):
        cb  = df[(df["risk_level"]=="BLOCKED") & (df["priority"]=="Critical")]["task_name"].tolist()
        udl = df[(df["days_to_due"]>=0) & (df["days_to_due"]<=7) & (df["status"]!="Complete")]["task_name"].tolist()
        st.markdown(f'<div class="prompt-box">{PROMPT_WEEKLY.format(total_tasks=total,completed=complete,in_progress=len(df[df["status"]=="In Progress"]),blocked=blocked,at_risk=at_risk,on_track=on_track,avg_completion=avg_pct,critical_blocked=", ".join(cb) or "None",upcoming_deadlines=", ".join(udl) or "None")}</div>', unsafe_allow_html=True)

st.divider()

# ── API Code Snippets ────────────────────────────────────────────────────────────
with st.expander("🔌 How to Connect Real AI APIs"):
    st.markdown("**ChatGPT (OpenAI)**")
    st.code('import openai\nclient = openai.OpenAI(api_key="YOUR_KEY")\nresponse = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content": your_prompt}])\nprint(response.choices[0].message.content)', language="python")
    st.markdown("**Gemini (Google)**")
    st.code('import google.generativeai as genai\ngenai.configure(api_key="YOUR_KEY")\nmodel = genai.GenerativeModel("gemini-1.5-pro")\nprint(model.generate_content(your_prompt).text)', language="python")

st.divider()
st.markdown('<div style="text-align:center;color:#555;font-size:.8rem;padding:20px;">AI Project Risk Monitor | Streamlit + Python | No confidential data</div>', unsafe_allow_html=True)
