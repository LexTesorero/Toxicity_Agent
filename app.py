import streamlit as st
from pathlib import Path
import re
import html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToxicityAgent",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load & inject CSS ─────────────────────────────────────────────────────────
def load_css(path: str) -> None:
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ── Try importing real agent; fall back to mock ───────────────────────────────
try:
    from agentai.agent import ToxicityAgent
    @st.cache_resource
    def get_agent():
        return ToxicityAgent()
    agent = get_agent()
    MOCK = False
except Exception:
    MOCK = True

def mock_analyze(text: str) -> dict:
    lower = text.lower()
    if any(w in lower for w in ["hate", "kill", "idiot", "stupid"]):
        cls, sub = "TOXIC", "Hate Speech"
    elif any(w in lower for w in ["okay", "fine", "whatever"]):
        cls, sub = "NEUTRAL", "Indifferent"
    else:
        cls, sub = "GOOD", "Positive"

    sarcasm = "no"
    meaning = ""
    if "totally" in lower or "sure" in lower:
        sarcasm = "ambiguous"
        meaning = "Possible ironic tone detected."

    return {
        "classification":    cls,
        "sub_label":         sub,
        "explanation":       "This is a mock response — connect the real ToxicityAgent to enable AI analysis.",
        "is_sarcasm":        sarcasm,
        "meaning":           meaning,
        "original":          text,
        "detected_language": "en",
        "translated":        None,
    }

def analyze(text: str) -> dict:
    return mock_analyze(text) if MOCK else agent.detect_and_respond(text)

# ── HTML helpers ──────────────────────────────────────────────────────────────

def compact(s: str) -> str:
    """Remove blank lines so Streamlit's Markdown parser never exits an HTML block early."""
    return "\n".join(line for line in s.splitlines() if line.strip())

# ── HTML builders ─────────────────────────────────────────────────────────────

def build_classifier_section(result: dict) -> str:
    cls     = result.get("classification", "NEUTRAL").strip().upper()
    sub     = html.escape(result.get("sub_label", "—"))
    css_cls = cls.lower()

    toxic_mod = "toxic-tint" if css_cls == "toxic" else ""

    return compact(f"""
    <div class="agent-section {toxic_mod}">
        <span class="agent-tag tag-classifier">Classifier</span>
        <div class="classifier-bubble {css_cls}">
            <span class="tox-level">{cls}</span>
            <span class="tox-sublabel">{sub}</span>
        </div>
    </div>
    """)


def build_sarcasm_section(result: dict) -> str:
    sarcasm = result.get("is_sarcasm", "no").strip().lower()
    meaning = html.escape(result.get("meaning", ""))

    label_map = {
        "sarcastic": "SARCASM DETECTED",
        "ambiguous": "AMBIGUOUS TONE",
        "no":        "NO SARCASM",
    }
    label = label_map.get(sarcasm, "UNKNOWN")

    meaning_html = (
        f'<div class="sarcasm-meaning">&#8627; {meaning}</div>'
        if sarcasm == "YES" and meaning else ""
    )

    return compact(f"""
    <div class="agent-section">
        <span class="agent-tag tag-sarcasm">Sarcasm Detector</span>
        <div class="sarcasm-bubble {sarcasm}">
            <div>
                <span class="sarcasm-status">{label}</span>
                {meaning_html}
            </div>
        </div>
    </div>
    """)


def build_responder_section(result: dict) -> str:
    explanation = result.get("explanation", "No response generated.").strip()
    cls         = result.get("classification", "NEUTRAL").strip().lower()
    toxic_mod   = "toxic-tint" if cls == "toxic" else ""

    lines      = explanation.splitlines()
    bullet_re  = re.compile(r'^-\s+(.+)')
    html_lines = []
    in_list    = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        m = bullet_re.match(stripped)
        if m:
            if not in_list:
                html_lines.append('<ul class="responder-list">')
                in_list = True
            html_lines.append(f"<li>{html.escape(m.group(1))}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f'<p class="responder-para">{html.escape(stripped)}</p>')

    if in_list:
        html_lines.append("</ul>")

    body = "\n".join(html_lines)

    return compact(f"""
    <div class="agent-section {toxic_mod}">
        <span class="agent-tag tag-responder">Responder</span>
        <div class="responder-text">{body}</div>
    </div>
    """)


def build_mother_container(result: dict) -> str:
    original  = result.get("original", "")
    cls       = result.get("classification", "NEUTRAL").strip().lower()
    toxic_mod = "toxic-tint" if cls == "toxic" else ""

    flat    = " ".join(original.splitlines()).strip()
    preview = html.escape((flat[:72] + "…") if len(flat) > 72 else flat)

    original_html = "".join(
        f'<p class="input-line">{html.escape(line)}</p>' if line.strip()
        else '<div class="input-break"></div>'
        for line in original.splitlines()
    ) or f'<p class="input-line">{html.escape(original)}</p>'

    return compact(f"""
    <div class="mother-container {cls}">
        <div class="mother-header {cls}">
            <span class="mother-header-label">INPUT</span>
            <span class="mother-header-input">{preview}</span>
        </div>
        <div class="agent-section input-full {toxic_mod}">
            {original_html}
        </div>
        {build_classifier_section(result)}
        {build_sarcasm_section(result)}
        {build_responder_section(result)}
    </div>
    """)

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown(compact("""
<div class="page-header">
    <div class="page-title">TOXICITY AGENT</div>
    <div class="page-subtitle">multi-agent content analysis system</div>
</div>
"""), unsafe_allow_html=True)

if MOCK:
    st.markdown(compact("""
    <div style="font-family:var(--font-pixel);font-size:0.38rem;color:#ffaa00;
                border:2px solid #ffaa00;padding:0.5rem 0.8rem;margin-bottom:1.2rem;">
        DEMO MODE &nbsp;|&nbsp; Real agent not found — showing mock responses
    </div>
    """), unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
with st.form(key="analyze_form", clear_on_submit=False):
    user_input = st.text_area(
        "CONTENT TO ANALYZE",
        placeholder="Type or paste content here…",
        height=110,
    )
    submitted = st.form_submit_button("ANALYZE")

st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

# ── Result ────────────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    with st.spinner("Running pipeline…"):
        result = analyze(user_input.strip())
    st.markdown(build_mother_container(result), unsafe_allow_html=True)
else:
    st.markdown(compact("""
    <div class="empty-state">
        <span class="empty-state-icon">[]</span>
        <p class="empty-state-text">
            NO ANALYSIS YET<br>
            ENTER TEXT ABOVE AND HIT ANALYZE
        </p>
    </div>
    """), unsafe_allow_html=True)