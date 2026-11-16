"""CoughCare — Streamlit app (normal vs atypical cough research prototype)."""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from coughcare.audio import AudioError  # noqa: E402
from coughcare.components import waveform_player  # noqa: E402
from coughcare.config import (  # noqa: E402
    ASSETS_DIR,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    configure_logging,
    demo_mode,
)
from coughcare.inference import Prediction, analyze_audio  # noqa: E402

configure_logging()

st.set_page_config(
    page_title="CoughCare — normal vs atypical cough",
    page_icon=str(ASSETS_DIR / "logo.png"),
    layout="centered",
    initial_sidebar_state="collapsed",
)

CSS = (ASSETS_DIR / "theme.css").read_text(encoding="utf-8")
LOGO = base64.b64encode((ASSETS_DIR / "logo.svg").read_bytes()).decode("ascii")
DEMO_ACTIVE = demo_mode() or not MODEL_PATH.exists()

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def cached_analysis(audio_bytes: bytes, demo: bool) -> Prediction:
    return analyze_audio(audio_bytes, demo=demo)


@st.cache_data(show_spinner=False)
def load_metrics_cached(mtime: float) -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_metrics() -> dict | None:
    try:
        mtime = METRICS_PATH.stat().st_mtime if METRICS_PATH.exists() else 0.0
    except OSError:
        mtime = 0.0
    return load_metrics_cached(mtime)


def pct(value: float, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def ci_text(ci: dict | None) -> str:
    if not ci or not isinstance(ci, dict):
        return "—"
    lower, upper = ci.get("lower"), ci.get("upper")
    if lower is None or upper is None:
        return "—"
    return f"{lower * 100:.1f}–{upper * 100:.1f}%"


def brand_header() -> None:
    st.markdown(
        f"""
        <div class="cc-brand">
          <img class="cc-mark" src="data:image/svg+xml;base64,{LOGO}" alt="CoughCare logo"/>
          <div class="cc-brand-text">
            <span class="cc-wordmark">CoughCare</span>
            <span class="cc-tagline">a cough check for Delhi air</span>
          </div>
          <span class="cc-beta">Research prototype</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div class="cc-footer">
          <p><strong>CoughCare is not a medical device.</strong> It does not diagnose,
          rule out, or screen for any disease. It compares one cough recording with a
          small research dataset and returns a coarse “normal / atypical” signal.</p>
          <p>If you have severe breathlessness, chest pain, blue lips, confusion, or a
          cough with blood, seek urgent medical care. For persistent symptoms, speak to
          a qualified clinician.</p>
          <p>Built as a student research prototype with the public COUGHVID dataset
          (CC BY 4.0). No recording is stored by this app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Check page
# --------------------------------------------------------------------------
def hero() -> None:
    st.markdown(
        """
        <section class="cc-hero">
          <p class="cc-eyebrow">Cough acoustics · Delhi</p>
          <h1>Listen first.<br/>Then decide.</h1>
          <p class="cc-lede">Record one cough. CoughCare compares its sound with
          research recordings and gives a simple signal — <strong>normal</strong> or
          <strong>monitor this cough</strong>. It cannot tell you what is wrong, and it
          is not a diagnosis.</p>
          <div class="cc-pill-row">
            <span class="cc-pill"><span class="cc-dot"></span> About 20 seconds</span>
            <span class="cc-pill"><span class="cc-dot"></span> No account needed</span>
            <span class="cc-pill"><span class="cc-dot cc-dot--rust"></span> Research use only</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def how_to_record() -> None:
    st.markdown(
        """
        <div class="cc-steps">
          <div class="cc-step">
            <div class="cc-step-num">1</div>
            <h4>Find a quiet spot</h4>
            <p>Background noise makes the cough harder to read.</p>
          </div>
          <div class="cc-step">
            <div class="cc-step-num">2</div>
            <h4>Phone a hand away</h4>
            <p>Hold it near your mouth, not touching your lips.</p>
          </div>
          <div class="cc-step">
            <div class="cc-step-num">3</div>
            <h4>Cough once, gently</h4>
            <p>A natural cough is better than a forced one.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pick_audio() -> tuple[bytes | None, str]:
    """Render record/upload inputs and return the chosen clip."""
    round_id = st.session_state.get("input_round", 0)
    tabs = st.tabs(["Record a cough", "Upload a clip"])

    with tabs[0]:
        recorded = st.audio_input(
            "Record a cough",
            key=f"rec_{round_id}",
            help="Allow microphone access in your browser, then cough once.",
        )
        st.caption("Place your phone a hand’s width from your mouth. Cough once, then stop the recording.")

    with tabs[1]:
        uploaded = st.file_uploader(
            "Upload a cough clip",
            type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
            key=f"up_{round_id}",
            help="WAV, MP3, M4A, OGG, WebM or FLAC. Short clips work best.",
        )
        st.caption("A single cough in a quiet room. Clips under 1 second are too short.")

    st.caption("No cough handy? Try a labelled sample from the COUGHVID evaluation subset.")
    sample_cols = st.columns([1, 1, 1], gap="small")
    with sample_cols[0]:
        if st.button("Normal sample", key=f"sample_normal_{round_id}", use_container_width=True):
            st.session_state["sample"] = "normal"
    with sample_cols[1]:
        if st.button("Atypical sample", key=f"sample_atypical_{round_id}", use_container_width=True):
            st.session_state["sample"] = "atypical"
    if st.session_state.get("sample"):
        with sample_cols[2]:
            if st.button("Clear sample", key=f"sample_clear_{round_id}", use_container_width=True):
                st.session_state["sample"] = None

    sources: list[tuple[str, bytes]] = []
    sample = st.session_state.get("sample")
    if sample:
        sample_path = ASSETS_DIR / "samples" / f"{sample}.webm"
        if sample_path.exists():
            sources.append(("sample", sample_path.read_bytes()))
    if recorded is not None:
        sources.append(("recording", recorded.getvalue()))
    if uploaded is not None:
        sources.append(("upload", uploaded.getvalue()))

    if not sources:
        return None, ""

    if len(sources) == 1:
        return sources[0][1], sources[0][0]

    choice = st.radio(
        "Which clip should we analyse?",
        [name for name, _ in sources],
        horizontal=True,
        key=f"choose_{round_id}",
    )
    selected = dict(sources)[choice]
    return selected, choice


def render_result(pred: Prediction, metrics: dict | None) -> None:
    confidence = pred.confidence
    tone = "alert" if pred.is_atypical else "normal"
    icon = "!" if pred.is_atypical else "✓"
    title = "Monitor this cough" if pred.is_atypical else "Normal cough"
    kicker = "Model signal · not a diagnosis"

    if pred.is_atypical:
        first = "The model noticed acoustic features that resemble the “symptomatic” examples it was trained on."
        second = "That is a prompt to pay attention — <strong>not a diagnosis</strong>."
        advice = (
            "Keep an eye on it. If the cough lasts more than three weeks, gets worse, or "
            "comes with breathlessness, fever, wheezing, or chest pain, speak to a clinician. "
            "On high-pollution days, limit outdoor exertion and keep prescribed medication nearby."
        )
    else:
        first = "The model did not flag strong atypical features in this clip."
        second = "Most coughs are ordinary, including coughs caused by dust and pollution."
        advice = (
            "Keep monitoring how you feel. If a cough persists, or you develop breathlessness, "
            "fever, or chest pain, get medical advice regardless of this result."
        )

    low_confidence = confidence < 0.60
    low_note = ""
    if low_confidence:
        low_note = (
            '<div class="cc-low"><strong>Close call.</strong> This clip sits near the '
            "model’s decision boundary, so the label is best treated as inconclusive. "
            "Notice your symptoms rather than the score.</div>"
        )

    weak_model_note = ""
    auc = None
    if metrics:
        auc = metrics.get("primary_metrics", {}).get("roc_auc")
    if auc is not None and auc < 0.65 and pred.mode == "real":
        weak_model_note = (
            f'<div class="cc-low"><strong>Weak model warning.</strong> In held-out testing, '
            f"this model separated normal and atypical coughs only slightly better than chance "
            f"(ROC-AUC {auc:.2f}). Treat the result as a demo, not a screening test.</div>"
        )

    st.markdown(
        f"""
        <div class="cc-result cc-result--{tone}">
          <div class="cc-result-top">
            <div class="cc-result-icon" aria-hidden="true">{icon}</div>
            <div>
              <p class="cc-result-kicker">{kicker}</p>
              <p class="cc-result-title">{title}</p>
            </div>
          </div>
          <p class="cc-result-copy">{first} {second}</p>
          <p class="cc-result-copy">{advice}</p>
          <div class="cc-confidence">
            <span class="cc-confidence-num">{confidence * 100:.0f}%</span>
            <span class="cc-confidence-label">model confidence<br/>not probability of illness</span>
          </div>
          <div class="cc-meter" role="img" aria-label="Model confidence {confidence * 100:.0f} percent">
            <div class="cc-meter-fill" style="width:{confidence * 100:.1f}%"></div>
          </div>
          <div class="cc-meter-caption"><span>uncertain</span><span>confident</span></div>
          {low_note}
          {weak_model_note}
          <p class="cc-caveat">Confidence describes how close the clip is to one side of the
          model’s boundary. It is not the chance that you are unwell.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1], gap="small")
    with cols[0]:
        st.download_button(
            "Save clip (WAV)",
            data=pred.playback_wav,
            file_name="coughcare_clip.wav",
            mime="audio/wav",
            use_container_width=True,
        )
    with cols[1]:
        if st.button("Check another cough", use_container_width=True):
            st.session_state["input_round"] = st.session_state.get("input_round", 0) + 1
            st.session_state["sample"] = None
            st.rerun()

    with st.expander("Technical details"):
        st.markdown(
            f"""
            - **Model score, P(atypical):** `{pred.probability_atypical:.3f}`
            - **Decision threshold:** `{pred.threshold:.2f}`
            - **Clip duration:** `{pred.duration_s:.2f} s` (analysed window `{pred.window_duration_s:.2f} s`)
            - **Peak level:** `{pred.quality.peak_dbfs:.1f} dBFS` · **RMS:** `{pred.quality.rms_dbfs:.1f} dBFS`
            - **Clipping:** `{pred.quality.clipping_ratio * 100:.2f}%` of samples
            - **Model:** {html.escape(pred.model_note)}
            - **Mode:** `{pred.mode}`
            """
        )
        st.caption(
            "Audio is processed in memory. The app keeps only the derived waveform and "
            "metrics for this browser session."
        )


def check_page(metrics: dict | None) -> None:
    hero()

    if DEMO_ACTIVE:
        reason = (
            "COUGHCARE_DEMO is set."
            if demo_mode()
            else "No trained model file was found."
        )
        st.markdown(
            f"""
            <div class="cc-demo">
              <strong>Demo mode.</strong>&nbsp;{reason} Results come from a transparent
              heuristic, not the trained COUGHVID model. They are illustrative only and
              must not be used for health decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    how_to_record()
    audio_bytes, source = pick_audio()

    if audio_bytes is None:
        st.markdown(
            """
            <div class="cc-card cc-card--quiet">
              <h3>What happens next</h3>
              <p>The clip is decoded, trimmed to the loudest cough, and turned into
              spectrogram features. A logistic regression model returns one score.
              No audio leaves the app.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if len(audio_bytes) > 30 * 1024 * 1024:
        st.error("That file is larger than 30 MB. Please upload a shorter clip.")
        return

    with st.spinner("Listening for the cough…"):
        try:
            pred = cached_analysis(audio_bytes, DEMO_ACTIVE)
        except AudioError as exc:
            st.error(f"We couldn’t analyse that clip: {exc}")
            st.caption(
                "Try a quiet room, move the phone a little closer, and record for 2–5 seconds. "
                "WAV, MP3, M4A, OGG and WebM are supported."
            )
            return
        except FileNotFoundError:
            st.error(
                "The trained model isn’t available in this deployment. "
                "Set COUGHCARE_DEMO=1 to use the labelled demo mode, or train the model "
                "with `python scripts/train_model.py`."
            )
            return
        except Exception as exc:  # noqa: BLE001 - never dump a traceback at the user
            st.error("Something went wrong while analysing this clip. Please try another recording.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
            return

    st.caption(f"Analysed your {source}. Waveform below — press play.")

    for warning in pred.quality.warnings:
        st.warning(warning.split(":", 1)[-1].strip())

    components.html(
        waveform_player(pred.playback_wav, pred.peaks, pred.window_duration_s),
        height=196,
        scrolling=False,
    )

    render_result(pred, metrics)


# --------------------------------------------------------------------------
# How it works
# --------------------------------------------------------------------------
def how_page(metrics: dict | None) -> None:
    st.markdown(
        """
        <section class="cc-hero">
          <p class="cc-eyebrow">How it works</p>
          <h1>A narrow question,<br/>answered cautiously.</h1>
          <p class="cc-lede">CoughCare does one small thing: it compares the sound of
          your cough with examples labelled “healthy” and “symptomatic” in a public
          research dataset. It does not look for a named disease.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## From cough to signal")
    st.markdown(
        """
        <div class="cc-steps">
          <div class="cc-step">
            <div class="cc-step-num">1</div>
            <h4>Find the cough</h4>
            <p>The loudest 1.5 seconds of the clip are kept. Silence and background noise are trimmed.</p>
          </div>
          <div class="cc-step">
            <div class="cc-step-num">2</div>
            <h4>Describe the sound</h4>
            <p>MFCCs, log-mel energies and spectral statistics summarise pitch, texture and noisiness.</p>
          </div>
          <div class="cc-step">
            <div class="cc-step-num">3</div>
            <h4>Compare, don’t diagnose</h4>
            <p>A logistic regression returns a score. A threshold turns it into “normal” or “monitor”.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cc-card">
          <h3>What the two labels mean</h3>
          <p><strong>Normal cough</strong> — the model did not find strong atypical
          features. This does <em>not</em> mean you are healthy.</p>
          <p><strong>Monitor this cough</strong> — the model found features closer to the
          “symptomatic” examples. This does <em>not</em> mean you have a condition.</p>
          <p>Both labels are about one recording, not about you.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Built for a Delhi context")
    st.write(
        "Delhi’s air can irritate the airways, and coughs are common — especially in winter. "
        "A tool that helps someone notice a cough that sounds different from their baseline "
        "could support earlier conversations with a clinician. That is the whole ambition here. "
        "CoughCare does not measure air quality, exposure, infection, or lung function."
    )

    st.markdown(
        """
        <div class="cc-note">
          <p><strong>Important:</strong> a cough can be caused by pollution, a cold, allergies,
          asthma, reflux, smoking, or many other things. A short audio clip cannot separate
          those causes. Only a clinician can assess your symptoms and history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Privacy and limits")
    st.markdown(
        """
        <ul class="cc-about-list">
          <li><strong>In memory only.</strong> This prototype does not save recordings, names, or accounts.</li>
          <li><strong>No disease detection.</strong> The model is trained on “healthy vs symptomatic” labels, not on a disease.</li>
          <li><strong>Phone microphones vary.</strong> A different device or room can change the sound and the score.</li>
          <li><strong>Weak evidence.</strong> On held-out data the model performs only modestly; see About &amp; stats for the real numbers.</li>
          <li><strong>Not for emergencies.</strong> Severe breathlessness or chest pain needs urgent care, not an app.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("When should I see a doctor?"):
        st.write(
            "Get medical advice if a cough lasts more than three weeks, keeps getting worse, "
            "brings up blood, or comes with breathlessness, chest pain, fever, unexplained "
            "weight loss, or night sweats. Seek emergency care for severe breathing difficulty, "
            "blue lips, confusion, or fainting."
        )

    with st.expander("Why does the app ask for one cough?"):
        st.write(
            "The COUGHVID recordings are short, single-cough clips. Multiple coughs or a long "
            "speech recording shift the features away from that training distribution. One "
            "natural cough in a quiet room is the closest match."
        )

    with st.expander("Can it tell me if I have an infection?"):
        st.write(
            "No. The model has never been validated to detect infection or any specific "
            "condition. It only ranks a clip against broad healthy/symptomatic examples, "
            "and its held-out performance is modest."
        )


# --------------------------------------------------------------------------
# About & stats
# --------------------------------------------------------------------------
def stat_card(label: str, value: str, ci: str = "", context: str = "") -> str:
    ci_html = f'<div class="cc-stat-ci">{ci}</div>' if ci else ""
    context_html = f'<div class="cc-stat-context">{context}</div>' if context else ""
    return (
        f'<div class="cc-stat"><p class="cc-stat-label">{label}</p>'
        f'<div class="cc-stat-value">{value}</div>{ci_html}{context_html}</div>'
    )


def about_page(metrics: dict | None) -> None:
    st.markdown(
        """
        <section class="cc-hero">
          <p class="cc-eyebrow">About &amp; stats</p>
          <h1>The numbers,<br/>including the awkward ones.</h1>
          <p class="cc-lede">CoughCare is a student research prototype. Its model is trained
          on a small, noisy, crowdsourced dataset. The honest summary: the signal is weak,
          so the app should be read as a demonstration — not a screening tool.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not metrics:
        st.info(
            "No metrics report found yet. Run `python scripts/train_model.py` to generate "
            "`reports/metrics.json`, then refresh this page."
        )
        dataset_section()
        return

    primary = metrics["primary_metrics"]
    ci = metrics.get("bootstrap_ci_95", {})
    dataset = metrics.get("dataset", {})
    split = metrics.get("split", {})
    model = metrics.get("model", {})
    baselines = metrics.get("baselines", {})
    balance = dataset.get("class_balance", {})
    auc = primary.get("roc_auc", 0.0)

    if auc < 0.65:
        st.markdown(
            f"""
            <div class="cc-note cc-note--alert">
              <p><strong>Read this first.</strong> On held-out data, ROC-AUC is
              <strong>{auc:.2f}</strong> (0.50 is chance). Sensitivity and specificity are
              close to a coin flip. The model does not reliably separate normal from
              atypical coughs, and no result from this app should inform a health decision.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Held-out performance")
    st.caption(
        "Five-fold grouped cross-validation. The numbers below come from out-of-fold "
        "predictions at a fixed 0.50 threshold, not from the final model’s training data."
    )

    stats = [
        ("ROC-AUC", f"{auc:.2f}", f"95% CI {ci_text(ci.get('roc_auc'))}", "Chance = 0.50"),
        ("PR-AUC", f"{primary.get('pr_auc', 0):.2f}", f"95% CI {ci_text(ci.get('pr_auc'))}", f"Baseline = {balance.get('normal_fraction', 0.5):.2f}"),
        ("Accuracy", pct(primary.get("accuracy", 0)), f"95% CI {ci_text(ci.get('accuracy'))}", "Balanced classes"),
        ("Sensitivity", pct(primary.get("sensitivity", 0)), f"95% CI {ci_text(ci.get('sensitivity'))}", "Atypical caught"),
        ("Specificity", pct(primary.get("specificity", 0)), f"95% CI {ci_text(ci.get('specificity'))}", "Normal recognised"),
        ("F1 score", f"{primary.get('f1', 0):.2f}", "", "Atypical class"),
        ("Class balance", f"{balance.get('normal', 500)} / {balance.get('atypical', 500)}", "", "Normal / atypical"),
        ("Samples", f"{primary.get('n', 1000):,}", "", "Held-out predictions"),
    ]
    cards = "".join(stat_card(*row) for row in stats)
    st.markdown(f'<div class="cc-stat-grid">{cards}</div>', unsafe_allow_html=True)

    cm = primary.get("confusion_matrix", {})
    st.markdown(
        f"""
        <table class="cc-table" aria-label="Confusion matrix">
          <thead><tr><th>Held-out prediction</th><th>Actually normal</th><th>Actually atypical</th></tr></thead>
          <tbody>
            <tr><td>Predicted normal</td><td>{cm.get('tn', 0)}</td><td>{cm.get('fn', 0)}</td></tr>
            <tr><td>Predicted atypical</td><td>{cm.get('fp', 0)}</td><td>{cm.get('tp', 0)}</td></tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Bootstrap 95% confidence intervals from {ci.get('_meta', {}).get('rounds', 1000)} "
        f"resamples of the {primary.get('n', 0)} held-out predictions."
    )

    st.markdown("## Context: what would a trivial model do?")
    majority = baselines.get("majority_class", {})
    st.markdown(
        f"""
        <table class="cc-table" aria-label="Baseline comparison">
          <thead><tr><th>Model</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th></tr></thead>
          <tbody>
            <tr><td>CoughCare (logistic regression)</td><td>{pct(primary.get('accuracy', 0))}</td>
                <td>{pct(primary.get('sensitivity', 0))}</td><td>{pct(primary.get('specificity', 0))}</td>
                <td>{primary.get('roc_auc', 0):.2f}</td></tr>
            <tr><td>Majority class ({majority.get('predicted_class', 'normal')})</td>
                <td>{pct(majority.get('accuracy', 0))}</td><td>{pct(majority.get('sensitivity', 0))}</td>
                <td>{pct(majority.get('specificity', 0))}</td><td>0.50</td></tr>
            <tr><td>Random guess</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>0.50</td></tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Because the evaluation set is balanced, a model that always says “normal” gets 50% "
        "accuracy while catching zero atypical coughs. That is why sensitivity and specificity "
        "are reported alongside accuracy."
    )

    st.markdown("## Dataset")
    included = dataset.get("selection", {})
    st.markdown(
        f"""
        <div class="cc-card">
          <h3>COUGHVID · Zenodo 4048312</h3>
          <p>A crowdsourced corpus of cough recordings collected during the COVID-19
          pandemic, released under CC BY 4.0. Metadata contains <strong>20,072</strong>
          recordings; <strong>10,746</strong> pass the dataset’s cough detector at
          p ≥ 0.8.</p>
          <p><strong>This model uses {included.get('selected_total', 1000):,} clips</strong>
          — {balance.get('normal', 500)} normal and {balance.get('atypical', 500)} atypical —
          all at least 1 second long, selected deterministically with a fixed seed.</p>
          <p><strong>Label rule:</strong> expert diagnosis where available, otherwise
          self-reported status. “Healthy” maps to normal; “symptomatic” and expert-noted
          pathology map to atypical. Self-reported COVID-19 rows were excluded entirely.
          No other label was mapped to a disease.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <ul class="cc-about-list">
          <li><strong>Features:</strong> {model.get('n_features', 395)} values from log-mel energies,
          MFCCs + deltas, spectral contrast, flatness, centroid, roll-off and zero-crossing rate.</li>
          <li><strong>Split:</strong> {split.get('n_splits', 5)}-fold {split.get('type', 'StratifiedGroupKFold')},
          grouped by <em>{html.escape(str(split.get('grouping_key', 'recording uuid')))}</em>.</li>
          <li><strong>Model:</strong> {html.escape(str(model.get('name', 'StandardScaler + LogisticRegression')))},
          class-weight balanced, C = {model.get('selected_C', '—')}; threshold
          {model.get('threshold', 0.5):.2f}.</li>
          <li><strong>Per-fold ROC-AUC:</strong> {metrics.get('fold_auc_mean', float('nan')):.2f}
          ± {metrics.get('fold_auc_std', float('nan')):.2f} (mean ± SD).</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    panel_path = FIGURES_DIR / "evaluation_panel.png"
    if panel_path.exists():
        st.image(str(panel_path), caption="Held-out confusion matrix, ROC curve and precision–recall curve", use_container_width=True)
    else:
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        roc_path = FIGURES_DIR / "roc_pr_curves.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Held-out confusion matrix", use_container_width=True)
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC and precision–recall curves", use_container_width=True)

    st.markdown("## Limitations, stated plainly")
    st.markdown(
        """
        <ul class="cc-about-list">
          <li><strong>No participant IDs.</strong> This COUGHVID release has no persistent user
          identifier, so recordings — not people — are grouped during cross-validation.
          Participant-level leakage cannot be ruled out; metrics may be optimistic.</li>
          <li><strong>Noisy labels.</strong> “Healthy” and “symptomatic” are self-reported, and
          expert labels cover only a subset. Neither is a clinical diagnosis.</li>
          <li><strong>Domain gap.</strong> Crowdsourced recordings from many countries are not a
          Delhi sample. Phone hardware, rooms and cough technique all vary.</li>
          <li><strong>Small sample.</strong> 1,000 clips is a demo-sized subset, not a
          representative epidemiological sample.</li>
          <li><strong>No external validation.</strong> These are cross-validated COUGHVID numbers,
          not performance on local users.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    citations = metrics.get("dataset", {}).get("name", "COUGHVID")
    with st.expander("Citation and data access"):
        st.markdown(
            f"""
            **Dataset:** {citations} (CC BY 4.0).\
            Orlandic, L., Arora, T. & Idelchik, M. *The COUGHVID crowdsourcing dataset,
            a corpus for the study of large-scale cough analysis algorithms.*
            Scientific Data **8**, 156 (2021).

            **Data file:** `public_dataset.zip` from Zenodo record 4048312,
            MD5 `5c30a8b00c8bb7783a2c15a48cb8ea9e`.

            **CoughCare:** not affiliated with EPFL or the COUGHVID authors.
            """
        )


def dataset_section() -> None:
    st.markdown(
        """
        ## Dataset
        COUGHVID (Zenodo 4048312) is a public crowdsourced cough corpus released under
        CC BY 4.0. This prototype uses a deterministic, balanced 1,000-clip subset:
        500 normal and 500 atypical, all with cough detector confidence ≥ 0.8 and a
        duration of at least 1 second. Self-reported COVID-19 recordings are excluded.
        Expert labels are used where available; otherwise self-reported health status
        provides the label.
        """
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    brand_header()

    page_options = ["Check a cough", "How it works", "About & stats"]
    if "nav" not in st.session_state:
        requested = str(st.query_params.get("page", "")).strip().lower()
        aliases = {"check": 0, "how": 1, "about": 2, "stats": 2}
        st.session_state["nav"] = page_options[aliases.get(requested, 0)]

    if "sample" in st.query_params and not st.session_state.get("_url_sample_handled"):
        requested_sample = str(st.query_params.get("sample", "")).strip().lower()
        if requested_sample in {"normal", "atypical"}:
            st.session_state["sample"] = requested_sample
        st.session_state["_url_sample_handled"] = True

    page = st.radio(
        "Navigate",
        page_options,
        horizontal=True,
        label_visibility="collapsed",
        key="nav",
    )
    metrics = get_metrics()

    if page == "Check a cough":
        check_page(metrics)
    elif page == "How it works":
        how_page(metrics)
    else:
        about_page(metrics)

    footer()


if __name__ == "__main__":
    if "input_round" not in st.session_state:
        st.session_state["input_round"] = 0
    if "sample" not in st.session_state:
        st.session_state["sample"] = None
    main()
