"""
Theme module for the Veto dashboard.
Color palette pulled from Bitget's own brand imagery (bright cyan/
turquoise accent on dark background, clean futuristic feel) so the
dashboard visually matches the platform it's built for.

Streamlit itself can't do true video/3D motion -- this module gets as
close as the framework allows: animated gradients, glow effects,
fade-in-on-load transitions, and animated counting numbers via a
small injected <script>. Everything here is CSS/JS injected through
st.markdown(unsafe_allow_html=True), which is the standard, supported
way to theme a Streamlit app beyond its built-in options.
"""

import streamlit as st

# --- Bitget-inspired palette ---
CYAN = "#00E5FF"
CYAN_DEEP = "#00B8C4"
CYAN_GLOW = "rgba(0, 229, 255, 0.35)"
BG_DARK = "#0A0E14"
BG_CARD = "#12171F"
BG_CARD_HOVER = "#161C26"
TEXT_PRIMARY = "#F2F5F8"
TEXT_MUTED = "#8B95A5"
GREEN = "#00E5A0"
RED = "#FF5C7A"
BORDER = "rgba(0, 229, 255, 0.15)"


def apply_theme():
    """Call once, near the top of the app, before any other st.* calls."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: {BG_DARK};
        }}

        /* Animated gradient hero banner */
        .veto-hero {{
            position: relative;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            background: linear-gradient(120deg, #0A0E14 0%, #0D2830 45%, #0A0E14 100%);
            background-size: 200% 200%;
            animation: heroShift 8s ease-in-out infinite;
            border: 1px solid {BORDER};
            overflow: hidden;
        }}
        .veto-hero::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -20%;
            width: 60%;
            height: 200%;
            background: radial-gradient(circle, {CYAN_GLOW} 0%, transparent 70%);
            animation: sweep 6s ease-in-out infinite;
            pointer-events: none;
        }}
        @keyframes heroShift {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}
        @keyframes sweep {{
            0%, 100% {{ transform: translateX(0) translateY(0); }}
            50% {{ transform: translateX(80%) translateY(20%); }}
        }}
        .veto-hero h1 {{
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0 0 0.4rem 0;
            color: {TEXT_PRIMARY};
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }}
        .veto-hero h1 .accent {{
            color: {CYAN};
            text-shadow: 0 0 20px {CYAN_GLOW};
        }}
        .veto-hero p {{
            font-size: 1.05rem;
            color: {TEXT_MUTED};
            max-width: 800px;
            margin: 0;
            position: relative;
            z-index: 1;
            line-height: 1.5;
        }}

        /* Fade-in-up entrance animation applied to main content blocks */
        [data-testid="stVerticalBlock"] > div {{
            animation: fadeInUp 0.5s ease-out backwards;
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            transition: all 0.25s ease;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {CYAN};
            box-shadow: 0 0 24px {CYAN_GLOW};
            transform: translateY(-2px);
        }}
        [data-testid="stMetricValue"] {{
            color: {CYAN};
            font-weight: 800;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED};
        }}

        /* Section headers */
        h2, h3 {{
            color: {TEXT_PRIMARY} !important;
            font-weight: 700 !important;
        }}
        h2 {{
            border-left: 3px solid {CYAN};
            padding-left: 0.7rem;
        }}

        /* Expanders (rejected trades) */
        [data-testid="stExpander"] {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            transition: all 0.2s ease;
        }}
        [data-testid="stExpander"]:hover {{
            border-color: {CYAN_DEEP};
        }}

        /* Dataframes */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(90deg, {CYAN_DEEP}, {CYAN});
            color: {BG_DARK};
            font-weight: 700;
            border: none;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            box-shadow: 0 0 20px {CYAN_GLOW};
            transform: translateY(-1px);
        }}

        /* Divider */
        hr {{
            border-color: {BORDER} !important;
        }}

        /* Badge pills for direction labels */
        .veto-badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 0.6rem;
        }}
        .veto-badge-buy {{ background: rgba(0,229,160,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
        .veto-badge-sell {{ background: rgba(255,92,122,0.15); color: {RED}; border: 1px solid {RED}; }}
        .veto-badge-hold {{ background: rgba(139,149,165,0.15); color: {TEXT_MUTED}; border: 1px solid {TEXT_MUTED}; }}

        /* Footer caption */
        .veto-footer {{
            text-align: center;
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            padding: 1.5rem 0 0.5rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_banner(title_prefix: str, title_accent: str, subtitle: str):
    """Renders the animated gradient hero header."""
    accent_span = f'<span class="accent">{title_accent}</span>' if title_accent else ""
    st.markdown(
        f"""
        <div class="veto-hero">
            <h1>{title_prefix} {accent_span}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def animated_counter(label: str, value, suffix: str = "", duration_ms: int = 900):
    """
    Renders a metric-style number that counts up from 0 to `value` on
    load, using a small injected script -- Streamlit's own st.metric
    has no built-in animation, so this is layered on top via
    components for the motion effect specifically requested.
    Works for both int and float values.
    """
    import streamlit.components.v1 as components

    is_float = isinstance(value, float)
    safe_id = "counter-" + "".join(ch for ch in label if ch.isalnum())

    html = f"""
    <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px;
                padding:1rem 1.2rem; font-family:Inter,sans-serif;">
        <div style="color:{TEXT_MUTED}; font-size:0.8rem; margin-bottom:0.3rem;">{label}</div>
        <div id="{safe_id}" style="color:{CYAN}; font-size:1.8rem; font-weight:800;">0{suffix}</div>
    </div>
    <script>
        (function() {{
            const el = document.getElementById("{safe_id}");
            const target = {value};
            const isFloat = {str(is_float).lower()};
            const duration = {duration_ms};
            const start = performance.now();
            function tick(now) {{
                const progress = Math.min((now - start) / duration, 1);
                const current = isFloat ? (progress * target).toFixed(2) : Math.floor(progress * target);
                el.textContent = current + "{suffix}";
                if (progress < 1) requestAnimationFrame(tick);
            }}
            requestAnimationFrame(tick);
        }})();
    </script>
    """
    components.html(html, height=90)


def direction_badge(direction: str) -> str:
    """Returns an HTML badge pill for a trade direction, for use inside st.markdown."""
    cls = f"veto-badge-{direction.lower()}"
    return f'<span class="veto-badge {cls}">{direction.upper()}</span>'


def footer(text: str):
    st.markdown(f'<div class="veto-footer">{text}</div>', unsafe_allow_html=True)