import streamlit as st
import numpy as np
import cv2
import io
import os

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from restoration import analyze_image, restore_image


def make_page_icon():

    size = 128
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)

    ink = (30, 42, 120, 255)
    magenta = (200, 30, 106, 255)
    c = size // 2

    draw.ellipse((c - 34, c - 34, c + 34, c + 34), outline=ink, width=9)
    draw.line((c, 4, c, c - 34), fill=ink, width=9)
    draw.line((c, c + 34, c, size - 4), fill=ink, width=9)
    draw.line((4, c, c - 34, c), fill=ink, width=9)
    draw.line((c + 34, c, size - 4, c), fill=ink, width=9)
    draw.ellipse((c - 12, c - 12, c + 12, c + 12), fill=magenta)

    return icon


st.set_page_config(
    page_title="BrandRestore AI",
    page_icon=make_page_icon(),
    layout="wide"
)


# -------------------- LOGO AND ICONS --------------------

def logo_svg(css_class):

    return (
        f'<svg class="{css_class}" viewBox="0 0 64 64" aria-hidden="true">'
        '<circle cx="32" cy="32" r="18" fill="none" stroke="currentColor" stroke-width="4"/>'
        '<path d="M32 4v14M32 46v14M4 32h14M46 32h14" fill="none" stroke="currentColor" '
        'stroke-width="4" stroke-linecap="round"/>'
        '<circle cx="32" cy="32" r="6" style="fill:var(--magenta)"/>'
        '</svg>'
    )


def icon_svg(paths):

    return (
        '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{paths}</svg>'
    )


ICON_RESTORE = icon_svg(
    '<path d="M4 9V4h5M20 15v5h-5M4 4l6 6M20 20l-6-6"/>'
    '<rect x="9" y="9" width="6" height="6" rx="1"/>'
)

ICON_ANALYSIS = icon_svg(
    '<path d="M4 20V11M10 20V4M16 20v-6M2 20h20"/>'
)

ICON_REUSE = icon_svg(
    '<path d="M20 12a8 8 0 0 1-14 5.3M4 12a8 8 0 0 1 14-5.3"/>'
    '<path d="M18 3v4h-4M6 21v-4h4"/>'
)


# -------------------- THEME --------------------

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700;800&display=swap');

:root {
  --paper: #ECEEF2; --sheet: #FAFBFC; --table: #DDE1E8; --ink: #1E2A78;
  --text: #262B36; --muted: #5A6475; --rule: #C6CCD6; --magenta: #C81E6A;
  --btn: #1E2A78; --btn-hover: #C81E6A; --hover: #EEF0F5; --active: #E6EAF6; --dash: #8D96A6;
  --info-bg: #E8EEFF; --success-bg: #E3F4E9;
  --font: 'Schibsted Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp button, .stApp input, .stApp textarea, .stApp small, .stApp a,
[data-testid="stCaptionContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-family: var(--font);
}

header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stAppDeployButton"] { display: none; }

.block-container { max-width: 1180px; padding: 2.4rem 2rem 4rem; }

.stApp p, .stApp li, .stApp label { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3 { color: var(--ink); letter-spacing: -.01em; font-weight: 700; }
.stApp h2 { font-size: 1.4rem; }
.stApp h3 { font-size: 1.1rem; }
.stApp hr { border-color: var(--rule); }
[data-testid="stCaptionContainer"], .small-text { color: var(--muted); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--sheet); border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] h1 { font-size: 2rem; padding: 0; margin: 0; }
[data-testid="stSidebar"] h2 { font-size: 1.25rem; padding: .2rem 0 0; margin: 0; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: .85rem; color: var(--muted); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; padding: .55rem .8rem; border-radius: 3px;
  border-left: 3px solid transparent; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 600; font-size: .98rem; color: var(--text); }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--hover); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--active); border-left-color: var(--magenta); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: var(--ink); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) { outline: 3px solid var(--ink); outline-offset: 1px; }

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden; background: #121A5A;
  border-radius: 4px; padding: 2.4rem 2.4rem 2.2rem; margin-bottom: 1.6rem;
}
.hero > * { position: relative; z-index: 1; }
.stApp .hero h1 {
  color: #fff; font-size: clamp(2rem, 3.4vw, 2.9rem); font-weight: 800;
  line-height: 1.08; letter-spacing: -.02em; padding: 0; margin: 0 0 .7rem;
}
.stApp .hero p { color: #C9D1F6; max-width: 50ch; font-size: 1.05rem; line-height: 1.5; margin: 0 0 .5rem; }
.hero-art { min-height: 330px; }
.hero-art > * { max-width: 46%; }
.hero-art::after {
  content: ""; position: absolute; top: 0; right: 0; bottom: 0; width: 58%; z-index: 0;
  background:
    linear-gradient(90deg, #121A5A 0%, rgba(18,26,90,.6) 26%, rgba(18,26,90,0) 62%),
    url("__ART__") 6% 50% / cover no-repeat;
}
@media (max-width: 800px) {
  .hero { padding: 1.6rem 1.3rem; }
  .hero-art > * { max-width: 100%; }
  .hero-art::after { width: 100%; opacity: .3; }
}

/* ---------- Cards, metrics, images ---------- */
.custom-card {
  background: var(--sheet); border: 1px solid var(--rule); border-left: 3px solid var(--magenta);
  border-radius: 4px; padding: 20px 22px; margin-bottom: 15px;
}
.custom-card h3 { margin: 0 0 .35rem; padding: 0; color: var(--ink); font-size: 1.1rem; }
.custom-card [data-testid="stHeadingWithActionElements"]:not(:first-child) { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
.custom-card { line-height: 1.5; }
.custom-card p { color: var(--muted); margin: 0; line-height: 1.5; }

div[data-testid="stMetric"] {
  background: var(--sheet); padding: 16px 18px; border: 1px solid var(--rule);
  border-top: 3px solid var(--ink); border-radius: 4px;
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 500; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 700; font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.2; }
[data-testid="stMetricValue"] { font-size: 1.7rem; }

[data-testid="stImage"] img {
  background: var(--table); padding: 14px; border: 1px solid var(--rule);
  border-radius: 2px; box-sizing: border-box;
}
[data-testid="stImageCaption"] { color: var(--muted); font-size: .88rem; }
[data-testid="stAlert"] { border-radius: 4px; }

/* ---------- Inputs and buttons ---------- */
div.stButton > button, div.stDownloadButton > button {
  background: var(--btn); color: #fff; border: 1px solid var(--btn);
  border-radius: 4px; padding: .55rem 1.2rem; font-weight: 600; min-height: 2.75rem;
}
.stApp .stButton > button p, .stApp .stDownloadButton > button p { color: #fff; font-weight: 600; }
div.stButton > button:hover, div.stDownloadButton > button:hover {
  background: var(--btn-hover); border-color: var(--btn-hover); color: #fff;
}
.stApp button:focus-visible { outline: 3px solid var(--ink); outline-offset: 2px; }

div[data-testid="stFileUploader"] {
  background: var(--sheet); padding: 14px; border: 1px solid var(--rule); border-radius: 4px;
}
[data-testid="stFileUploaderDropzone"] { background: transparent; border: 1px dashed var(--dash); border-radius: 4px; }
[data-testid="stFileUploaderDropzone"] button { background: var(--sheet); color: var(--ink); border: 1px solid var(--rule); }

.logo-side { display: block; width: 56px; height: 56px; margin: 0 auto .4rem; color: var(--ink); }
.logo-inline { display: inline-block; width: .8em; height: .8em; vertical-align: -.04em; margin-right: .4em; color: inherit; }
.ico { display: inline-block; width: 1.25rem; height: 1.25rem; vertical-align: -4px; margin-right: .5rem; color: var(--magenta); }

button[data-baseweb="tab"] p { font-weight: 600; font-size: 1.05rem; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); }
div[data-baseweb="tab-highlight"] { background-color: var(--magenta); height: 3px; }
div[data-baseweb="tab-border"] { background-color: var(--rule); }

.stApp .stButton > button [data-testid="stIconMaterial"] { color: #fff; }

/* ---------- Native widgets follow the chosen theme ---------- */
.stApp { background: var(--paper); color: var(--text); }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] svg, [data-testid="stMainMenuButton"] { color: var(--text); }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] svg { color: var(--muted); }

[data-testid="stAlertContainer"] { background: var(--info-bg); color: var(--text); border: 1px solid var(--rule); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background: var(--success-bg); }
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] svg { color: var(--text); }

/* checkboxes */
[data-testid="stCheckbox"] label:not(:has(input[role="switch"])):not([data-selected="true"]) > div:first-of-type {
  background: var(--sheet) !important; border-color: var(--muted) !important;
}
/* dark mode switch */
[data-testid="stCheckbox"] label:has(input[role="switch"]):not([data-selected="true"]) > div:first-of-type {
  background: var(--dash) !important; border-color: var(--dash) !important;
}
/* sidebar radio dots */
[data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:first-child {
  background: var(--sheet) !important; border-color: var(--muted) !important;
}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {
  background: var(--magenta) !important; border-color: var(--magenta) !important;
}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div { background: #fff !important; }

/* uploaded file chip */
[data-testid="stFileChip"] { background: var(--hover) !important; border-radius: 4px; }
[data-testid="stFileChipName"] { color: var(--text) !important; }
[data-testid="stFileChip"] * { color: var(--muted); }
[data-testid="stFileChipDeleteBtn"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted); }
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] { color: var(--muted); }

button[data-baseweb="tab"] p { color: var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); }
.stApp code { background: var(--active); color: var(--ink); }
[data-testid="stSpinner"] * { color: var(--text); }
.stApp a { color: var(--magenta); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""

DARK_CSS = """
:root {
  --paper: #0E1222; --sheet: #151A2E; --table: #1B2137; --ink: #B4C0FF;
  --text: #E7EAF6; --muted: #9BA4BF; --rule: #2C3452; --magenta: #FF5C9E;
  --btn: #4054D6; --btn-hover: #D42B79; --hover: #1D2440; --active: #252E52; --dash: #4A5578;
  --info-bg: #1B2740; --success-bg: #12301F;
}
"""


def hero_art_uri():

    import base64

    if not os.path.exists("brandrestore_bg.jpg"):
        return ""

    image = Image.open("brandrestore_bg.jpg").convert("RGB")
    image.thumbnail((1100, 1100))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=78)

    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()


def detect_system_dark():

    try:
        return st.context.theme.get("type") == "dark"
    except Exception:
        return False


SYSTEM_DARK = detect_system_dark()


def apply_theme():

    dark = st.session_state.get("dark_mode", SYSTEM_DARK)

    css = THEME_CSS.replace("__ART__", hero_art_uri())

    if dark:
        css += DARK_CSS

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


apply_theme()


# -------------------- SAMPLE IMAGE --------------------

def generate_sample_image():

    sample_path = "brandrestore_sample.jpg"

    if not os.path.exists(sample_path):
        raise FileNotFoundError(
            "brandrestore_sample.jpg was not found in the project folder. "
            "Place the sample image beside app.py."
        )

    image = Image.open(sample_path).convert("RGB")

    return image


def create_low_resolution_sample():

    return generate_sample_image()


# -------------------- SESSION STATE --------------------

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "restored_image" not in st.session_state:
    st.session_state.restored_image = None

if "asset_library" not in st.session_state:
    st.session_state.asset_library = []

if "image_analysis" not in st.session_state:
    st.session_state.image_analysis = None

if "studio_phase" not in st.session_state:
    st.session_state.studio_phase = 1

if "about_phase" not in st.session_state:
    st.session_state.about_phase = 1

if "restore_opts" not in st.session_state:
    st.session_state.restore_opts = {
        "denoise": False,
        "sharpen": False,
        "contrast": False,
        "color": False
    }


def set_studio_phase(phase):

    st.session_state.studio_phase = phase


def set_about_phase(phase):

    st.session_state.about_phase = phase


# -------------------- SIDEBAR --------------------

st.sidebar.markdown(
    f"""
    <div style="text-align:center">

    {logo_svg("logo-side")}

    <h2 style="color:var(--ink);">
    BrandRestore AI
    </h2>

    <p style="color:var(--muted);">
    Legacy Creative Asset Restorer
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Restore Studio",
        "Brand Integrity Lab",
        "Campaign Studio",
        "Asset Library",
        "About"
    ]
)

st.sidebar.toggle(
    "Dark mode",
    value=SYSTEM_DARK,
    key="dark_mode"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-powered creative asset restoration"
)

st.sidebar.caption(
    "Swin2SR Real-World X4"
)


# -------------------- OVERVIEW --------------------

if page == "Overview":

    st.markdown(
        f"""
        <div class="hero hero-art">

        <h1>{logo_svg("logo-inline")}BrandRestore AI</h1>

        <p>
        Restore low-resolution brand assets into clearer,
        high-resolution creative images using AI.
        </p>

        <p>
        Revive logos, product visuals, packaging images
        and legacy creative assets.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "AI Upscaling",
        "4×"
    )

    c2.metric(
        "Restoration Model",
        "Swin2SR"
    )

    c3.metric(
        "Supported Formats",
        "PNG / JPG / WEBP"
    )

    st.markdown("## What can BrandRestore AI do?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="custom-card">

            <h3>{ICON_RESTORE}AI Restoration</h3>

            <p>
            Reconstruct visual details from low-resolution
            images using a pretrained super-resolution model.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="custom-card">

            <h3>{ICON_ANALYSIS}Asset Analysis</h3>

            <p>
            Examine image dimensions, sharpness,
            brightness and image quality.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="custom-card">

            <h3>{ICON_REUSE}Creative Reuse</h3>

            <p>
            Prepare older visual assets for modern
            creative and digital workflows.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Restoration Workflow")

    st.markdown(
        """
        <div class="custom-card">

        <h3>01 — Upload</h3>
        Upload an image or use the built-in sample.

        <h3>02 — Analyze</h3>
        Examine the input image's resolution and quality.

        <h3>03 — Restore</h3>
        Apply Swin2SR AI super-resolution.

        <h3>04 — Compare</h3>
        Compare the original and restored images.

        <h3>05 — Export</h3>
        Download the restored image.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Navigate to Restore Studio to start restoring images."
    )


# -------------------- RESTORE STUDIO --------------------

elif page == "Restore Studio":

    st.markdown(
        """
        <div class="hero">

        <h1>Restore Studio</h1>

        <p>
        Transform low-resolution images into
        high-resolution visuals using Swin2SR.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    phase = st.session_state.studio_phase

    if phase >= 2 and st.session_state.original_image is None:
        phase = 1
        st.session_state.studio_phase = 1

    if phase == 3 and st.session_state.restored_image is None:
        phase = 2
        st.session_state.studio_phase = 2

    # ---------- Phase 1: upload and analysis ----------

    if phase == 1:

        st.markdown("## Upload your image")

        uploaded_file = st.file_uploader(
            "Upload a low-resolution image",
            type=["png", "jpg", "jpeg", "webp"]
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Load Sample Image",
                use_container_width=True
            ):

                sample = create_low_resolution_sample()

                st.session_state.original_image = sample

                st.session_state.restored_image = None

                st.session_state.image_analysis = analyze_image(
                    sample
                )

        with col2:

            if st.button(
                "Clear Workspace",
                use_container_width=True
            ):

                st.session_state.original_image = None

                st.session_state.restored_image = None

                st.session_state.image_analysis = None

        if uploaded_file is not None:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.session_state.original_image = uploaded_image

            st.session_state.restored_image = None

            st.session_state.image_analysis = analyze_image(
                uploaded_image
            )

        if st.session_state.original_image is not None:

            original = st.session_state.original_image

            analysis = analyze_image(original)

            st.markdown("## Image Analysis")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Width",
                f"{analysis['width']} px"
            )

            c2.metric(
                "Height",
                f"{analysis['height']} px"
            )

            c3.metric(
                "Sharpness",
                analysis["sharpness"]
            )

            c4.metric(
                "Brightness",
                analysis["brightness"]
            )

            st.caption(
                f"Quality: {analysis['quality']} | "
                f"Lighting: {analysis['lighting']}"
            )

            st.button(
                "Next",
                icon=":material/arrow_forward:",
                on_click=set_studio_phase,
                args=(2,),
                use_container_width=True
            )

        else:

            st.info(
                "Upload an image or click Load Sample Image "
                "to begin restoration."
            )

    # ---------- Phase 2: original image and options ----------

    elif phase == 2:

        original = st.session_state.original_image

        st.markdown("## Original Image")

        st.image(
            original,
            caption="Original Input",
            use_container_width=True
        )

        st.markdown("## Restoration Options")

        saved = st.session_state.restore_opts

        for name in saved:

            if f"opt_{name}" not in st.session_state:
                st.session_state[f"opt_{name}"] = saved[name]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            denoise = st.checkbox(
                "Noise Reduction",
                key="opt_denoise"
            )

        with col2:

            sharpen = st.checkbox(
                "Extra Sharpening",
                key="opt_sharpen"
            )

        with col3:

            contrast = st.checkbox(
                "Enhance Contrast",
                key="opt_contrast"
            )

        with col4:

            color = st.checkbox(
                "Enhance Color",
                key="opt_color"
            )

        st.session_state.restore_opts = {
            "denoise": denoise,
            "sharpen": sharpen,
            "contrast": contrast,
            "color": color
        }

        st.info(
            "The AI model performs the primary 4× "
            "restoration. Optional enhancements are applied "
            "after the AI reconstruction."
        )

        back_col, restore_col = st.columns(2)

        with back_col:

            st.button(
                "Back",
                icon=":material/arrow_back:",
                on_click=set_studio_phase,
                args=(1,),
                use_container_width=True
            )

        with restore_col:

            restore_clicked = st.button(
                "Restore Image with AI",
                icon=":material/auto_fix_high:",
                use_container_width=True
            )

        if restore_clicked:

            with st.spinner(
                "Loading model and restoring image... "
                "This may take a while on CPU."
            ):

                restored = restore_image(
                    original,
                    denoise=denoise,
                    sharpen=sharpen,
                    contrast=contrast,
                    color=color,
                    upscale=4
                )

                st.session_state.restored_image = restored

                st.session_state.asset_library.append(
                    restored.copy()
                )

            st.session_state.studio_phase = 3

            st.rerun()

    # ---------- Phase 3: result ----------

    else:

        original = st.session_state.original_image

        restored = st.session_state.restored_image

        st.success(
            "AI restoration completed!"
        )

        st.markdown("## Before & After Comparison")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                original,
                caption="Original Image",
                use_container_width=True
            )

        with col2:

            st.image(
                restored,
                caption="Swin2SR AI Restored Image",
                use_container_width=True
            )

        st.markdown("## Restored Image Information")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Original Size",
            f"{original.width} × {original.height}"
        )

        c2.metric(
            "Restored Size",
            f"{restored.width} × {restored.height}"
        )

        c3.metric(
            "Scale",
            "4×"
        )

        st.markdown("## Download")

        buffer = io.BytesIO()

        restored.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="Download Restored Image",
            data=buffer.getvalue(),
            file_name="brandrestore_restored.png",
            mime="image/png",
            use_container_width=True
        )

        st.button(
            "Back",
            icon=":material/arrow_back:",
            on_click=set_studio_phase,
            args=(2,),
            use_container_width=True
        )


# -------------------- BRAND INTEGRITY LAB --------------------

elif page == "Brand Integrity Lab":

    st.markdown(
        """
        <div class="hero">

        <h1>Brand Integrity Lab</h1>

        <p>
        Inspect image quality before and after
        restoration.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    tab_original, tab_restored = st.tabs(
        [
            "Original Asset Analysis",
            "Restored Asset Analysis"
        ]
    )

    with tab_original:

        if st.session_state.original_image is not None:

            original = st.session_state.original_image

            analysis = analyze_image(original)

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    original,
                    caption="Original Asset",
                    use_container_width=True
                )

            with col2:

                st.markdown(
                    """
                    <div class="custom-card">
                    <h3>Quality Report</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    "Resolution:",
                    f"{analysis['width']} × {analysis['height']}"
                )

                st.write(
                    "Sharpness:",
                    analysis["sharpness"]
                )

                st.write(
                    "Brightness:",
                    analysis["brightness"]
                )

                st.write(
                    "Sharpness Category:",
                    analysis["quality"]
                )

                st.write(
                    "Lighting:",
                    analysis["lighting"]
                )

        else:

            st.info(
                "Load an image in Restore Studio to analyze it here."
            )

    with tab_restored:

        if st.session_state.restored_image is not None:

            restored = st.session_state.restored_image

            restored_analysis = analyze_image(
                restored
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Restored Width",
                restored_analysis["width"]
            )

            c2.metric(
                "Restored Height",
                restored_analysis["height"]
            )

            c3.metric(
                "Restored Sharpness",
                restored_analysis["sharpness"]
            )

            st.image(
                restored,
                caption="Restored Asset",
                use_container_width=True
            )

            st.caption(
                "Sharpness scores are image-analysis indicators, "
                "not proof of original detail recovery or brand authenticity."
            )

        else:

            st.info(
                "Restore an image in Restore Studio to analyze it here."
            )


# -------------------- CAMPAIGN STUDIO --------------------

elif page == "Campaign Studio":

    st.markdown(
        """
        <div class="hero">

        <h1>Campaign Studio</h1>

        <p>
        Prepare restored assets for creative reuse.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Creative Workspace")

    if st.session_state.restored_image is not None:

        image = st.session_state.restored_image

        st.image(
            image,
            caption="Restored Creative Asset",
            use_container_width=True
        )

        st.markdown("### Asset Dimensions")

        st.write(
            f"{image.width} × {image.height} pixels"
        )

        st.success(
            "Your restored asset is available "
            "for creative reuse."
        )

    else:

        st.info(
            "Restore an image in Restore Studio "
            "to bring it into Campaign Studio."
        )

    st.markdown("## Creative Preparation")

    st.markdown(
        """
        - Restore low-resolution visuals.
        - Review the restored asset.
        - Export the image for further editing.
        - Use it in digital creative workflows.
        """
    )

    st.caption(
        "BrandRestore AI restores images; it does not "
        "automatically generate complete marketing campaigns."
    )


# -------------------- ASSET LIBRARY --------------------

elif page == "Asset Library":

    st.markdown(
        """
        <div class="hero">

        <h1>Asset Library</h1>

        <p>
        View restored images from your current session.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.asset_library:

        st.markdown(
            f"## Saved Assets: {len(st.session_state.asset_library)}"
        )

        numbered_assets = list(
            enumerate(
                st.session_state.asset_library,
                start=1
            )
        )[::-1]

        for start in range(0, len(numbered_assets), 4):

            columns = st.columns(4)

            for column, (number, asset) in zip(
                columns,
                numbered_assets[start:start + 4]
            ):

                with column:

                    st.markdown(
                        f"**Restored Asset {number}**"
                    )

                    st.image(
                        asset,
                        width=200
                    )

                    buffer = io.BytesIO()

                    asset.save(
                        buffer,
                        format="PNG"
                    )

                    st.download_button(
                        label=f"Download Asset {number}",
                        data=buffer.getvalue(),
                        file_name=f"restored_asset_{number}.png",
                        mime="image/png",
                        key=f"download_asset_{number}"
                    )

    else:

        st.info(
            "No restored assets saved in this session yet. "
            "Restore an image first."
        )

    st.caption(
        "This library is session-based and does not "
        "permanently store files on a database."
    )


# -------------------- ABOUT --------------------

elif page == "About":

    st.markdown(
        """
        <div class="hero">

        <h1>About BrandRestore AI</h1>

        <p>
        AI-powered restoration for legacy creative assets.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.about_phase == 1:

        st.markdown(
            """
            ## Our Goal

            BrandRestore AI aims to improve the usability of
            low-resolution creative assets by reconstructing
            high-resolution images using deep learning.

            ## Technology Stack

            - Python
            - Streamlit
            - PyTorch
            - Hugging Face Transformers
            - Swin2SR
            - OpenCV
            - Pillow
            - NumPy
            """
        )

        st.button(
            "Next",
            icon=":material/arrow_forward:",
            on_click=set_about_phase,
            args=(2,),
            use_container_width=True,
            key="about_next"
        )

    else:

        st.markdown(
            """
            ## AI Model

            **Swin2SR Real-World X4**

            The pretrained super-resolution model is used to
            reconstruct a 4× resolution output from an
            input image.

            ## Features

            - Image uploading
            - Sample image workflow
            - Image quality analysis
            - AI super-resolution
            - Optional image enhancements
            - Before-and-after comparison
            - PNG downloads
            - Session-based asset library

            ## Project Purpose

            Restore older visual assets for reuse in modern
            digital creative workflows.
            """
        )

        st.markdown("---")

        st.markdown(
            f"""
            <div style="text-align:center">

            <h3 style="color:var(--ink);">
            {logo_svg("logo-inline")}BrandRestore AI
            </h3>

            <p>
            Reviving creative assets, one image at a time.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.button(
            "Back",
            icon=":material/arrow_back:",
            on_click=set_about_phase,
            args=(1,),
            use_container_width=True,
            key="about_back"
        )
