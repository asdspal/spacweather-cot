"""Generate the SpaceWeather-CoT pitch deck (python-pptx).

Run: python3 build_deck.py
Output: SpaceWeather-CoT-Pitch.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---- palette (matches the landing page) ----
INK = RGBColor(0x0A, 0x0D, 0x11)
PANEL = RGBColor(0x16, 0x1C, 0x23)
PAPER = RGBColor(0xEA, 0xE6, 0xDC)
PAPER_DIM = RGBColor(0x9A, 0xA3, 0xAF)
AMBER = RGBColor(0xE3, 0xA9, 0x4B)
AMBER_DIM = RGBColor(0x8A, 0x6A, 0x33)
COOL = RGBColor(0x9F, 0xC6, 0xE0)
BAD = RGBColor(0xC9, 0x6A, 0x56)
LINE = RGBColor(0x2A, 0x30, 0x38)

SERIF = "Georgia"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide():
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = INK
    bg.line.fill.background()
    bg.shadow.inherit = False
    bg._element.spPr.insert(0, bg._element.spPr.find(qn("a:solidFill")))
    return slide


def add_text(slide, left, top, width, height, text, size, color, font=SERIF, bold=False,
             italic=False, align=PP_ALIGN.LEFT, line_spacing=1.0, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = font
        run.font.bold = bold
        run.font.italic = italic
    return box


def add_rule(slide, left, top, width, color=LINE, weight=0.75):
    line = slide.shapes.add_connector(1, left, top, left + width, top)
    line.line.color.rgb = color
    line.line.width = Pt(weight)


def kicker_free_heading(slide, title, sub=None, title_size=40):
    add_text(slide, Inches(0.7), Inches(0.55), Inches(11.9), Inches(1.3), title, title_size,
              PAPER, font=SERIF, bold=False)
    if sub:
        add_text(slide, Inches(0.7), Inches(1.5), Inches(11.5), Inches(0.6), sub, 16, PAPER_DIM,
                  font=SERIF, italic=True)


def chip(slide, left, top, w, h, text, fg, bg=None):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    box.adjustments[0] = 0.5
    box.fill.solid()
    box.fill.fore_color.rgb = bg or PANEL
    box.line.color.rgb = LINE
    box.line.width = Pt(0.75)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(13)
    run.font.name = MONO
    run.font.color.rgb = fg


def bullet_block(slide, left, top, width, height, items, size=18, color=PAPER, font=SERIF,
                  gap=10, bullet_color=AMBER):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.15
        r1 = p.add_run(); r1.text = "— "
        r1.font.size = Pt(size); r1.font.color.rgb = bullet_color; r1.font.name = font
        r2 = p.add_run(); r2.text = item
        r2.font.size = Pt(size); r2.font.color.rgb = color; r2.font.name = font
    return box


# ============================================================ Slide 1: Title
s = add_slide()
add_text(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(0.5), "TimeNet Online Hackathon — Round 1 submission",
          16, AMBER, font=MONO)
add_text(s, Inches(0.9), Inches(3.0), Inches(11.5), Inches(1.6),
          "SpaceWeather-CoT", 60, PAPER, font=SERIF, bold=False)
add_text(s, Inches(0.9), Inches(4.3), Inches(10.5), Inches(1.0),
          "A forecaster explains the Sun by hand, twice a day.\nWe taught a model to try — and measured exactly where it fails.",
          20, PAPER_DIM, font=SERIF, italic=True, line_spacing=1.3)
add_rule(s, Inches(0.9), Inches(5.55), Inches(4.5))

# ============================================================ Slide 2: The problem
s = add_slide()
kicker_free_heading(s, "Space weather is forecast by hand", "Grids, GPS, satellites, and aviation all depend on that judgment call.")
bullet_block(s, Inches(0.9), Inches(2.2), Inches(7.2), Inches(4.5), [
    "NOAA's Space Weather Prediction Center writes a Forecast Discussion twice a day, from raw GOES X-ray and solar-wind instrument data.",
    "It's expert prose, not a dashboard: a human reads the charts and writes the reasoning.",
    "No existing TimeNet dataset covers this domain. And the field's known weakness is rationale quality — TimeNet's own SleepEDF-CoT card documents GPT-generated rationale defects.",
    "We asked: can a fine-tuned time-series model learn to reason like a real forecaster, trained on the forecaster's own words — and can we be honest about exactly where that breaks down?",
], size=19)
chip(s, Inches(8.6), Inches(2.3), Inches(3.8), Inches(0.55), "Grids  ·  GPS  ·  Satellites  ·  Aviation", PAPER_DIM)

# ============================================================ Slide 3: What we built
s = add_slide()
kicker_free_heading(s, "What we built", "A new TimeNet dataset, and two models trained on it.")
# dataset card
card1 = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(2.2), Inches(11.5), Inches(2.1))
card1.fill.solid(); card1.fill.fore_color.rgb = PANEL; card1.line.color.rgb = LINE; card1.line.width = Pt(0.75)
card1.shadow.inherit = False
add_text(s, Inches(1.15), Inches(2.35), Inches(6.5), Inches(0.4), "noaa/spaceweather-cot", 20, AMBER, font=MONO, bold=True)
add_text(s, Inches(1.15), Inches(2.8), Inches(10.9), Inches(1.4),
          "656 records · March 2024 – April 2025\n"
          "24h GOES-16 X-ray flux + OMNI solar-wind & Kp per record\n"
          "NOAA's 24h summary, 3-day forecast text, 3 numeric probabilities — TimeNet PR #842",
          14, PAPER_DIM, font=MONO, line_spacing=1.35)

# two model cards
card_w = Inches(5.6)
for i, (title, role, desc, color) in enumerate([
    ("Fine-tuned OpenTSLM", "WRITES THE NARRATIVE",
     "Llama-3.2-1B + LoRA, from OpenTSLM's ECG-reasoning checkpoint. Reads 5 input channels, writes a NOAA-formatted discussion in the forecaster's voice.", COOL),
    ("Gradient-boosted decision model", "PREDICTS THE NUMBERS",
     "Physics features (flare class, Bz, wind speed/shocks, Kp) plus persistence from the prior 1-2 issuances. Mirrors NOAA's own Model Output Statistics.", AMBER),
]):
    left = Inches(0.9) + i * Inches(5.85)
    card = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(4.6), card_w, Inches(2.35))
    card.fill.solid(); card.fill.fore_color.rgb = PANEL; card.line.color.rgb = LINE; card.line.width = Pt(0.75)
    card.shadow.inherit = False
    add_text(s, left + Inches(0.25), Inches(4.78), card_w - Inches(0.5), Inches(0.35), role, 12, color, font=MONO, bold=True)
    add_text(s, left + Inches(0.25), Inches(5.12), card_w - Inches(0.5), Inches(0.5), title, 22, PAPER, font=SERIF)
    add_text(s, left + Inches(0.25), Inches(5.68), card_w - Inches(0.5), Inches(1.2), desc, 13.5, PAPER_DIM, font=SERIF, line_spacing=1.2)

# ============================================================ Slide 4: How it helps
s = add_slide()
kicker_free_heading(s, "How it helps", "Explainable forecasting, built the way the real operation works.")
bullet_block(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(4.5), [
    "The narrative model writes in the forecaster's own voice and structure — not a black-box score, a readable discussion a grid or satellite operator already knows how to use.",
    "The decision model gives calibrated numbers backed by a measured R\u00b2, not a language model's guess — exactly the kind of number an operator would act on.",
    "The split itself is the finding: it mirrors NOAA's real internal practice (Model Output Statistics calibrating raw guidance, separate from the human narrative) rather than forcing one model to do both badly.",
    "The dataset is reusable: it's TimeNet's first space-weather domain and first genuinely expert-written chain-of-thought — a foundation other TimeNet teams can build on in later rounds.",
], size=19)

# ============================================================ Slide 5: What we learned (part 1 - data/process)
s = add_slide()
kicker_free_heading(s, "What we learned building it", "Three things the data forced us to confront.")
bullet_block(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(4.7), [
    "Real archives are messy across time: widening the date range into 2025 surfaced two NOAA formatting variants we hadn't seen (a missing \u201cSolar Activity\u201d header, a missing \u201cGeospace\u201d header) — both are genuine archive quirks, now handled as real parser cases, not skipped.",
    "Upstream data gaps hide silently: OMNI2's \u201cProton flux >10 MeV\u201d field — exactly what NOAA's S1 radiation-storm scale is defined on — is 100% missing in the 2024-2025 files. We verified this directly against the raw archive rather than assuming our connector was broken.",
    "Cloud training isn't guaranteed to run uninterrupted: a Modal GPU container was preempted mid-run with zero visibility, because our own logging buffered output until exit. We fixed the pipeline to stream logs live before relaunching.",
], size=18)

# ============================================================ Slide 6: What we learned (part 2 - modeling)
s = add_slide()
kicker_free_heading(s, "What we learned modeling it", "The core technical finding of this submission.")
bullet_block(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(2.2), [
    "A fine-tuned LLM asked to free-generate NOAA's numeric forecasts as text learns the format (0% \u2192 73-81% of outputs became parseable) but not the calibration — it lost to a trivial constant-mean baseline on 2 of 3 targets.",
    "The fix wasn't a bigger model. It was persistence: the previous 1-2 issuances\u2019 own NOAA forecasts, since 3-day outlooks barely change every 12 hours. That one feature carries 73-89% of the decision model\u2019s predictive weight.",
], size=18)
chip(s, Inches(0.9), Inches(5.0), Inches(5.4), Inches(0.7), "LLM next-token generation: no notion\nthat \u201834\u2019 is closer to \u201835\u2019 than to \u201899\u2019", BAD)
chip(s, Inches(6.6), Inches(5.0), Inches(5.8), Inches(0.7), "Persistence + physics features: real,\nmeasured R\u00b2 improvement, same data", AMBER)

# ============================================================ Slide 7: Results across every run
s = add_slide()
kicker_free_heading(s, "Results, across every run", "What actually changed, run over run — not just the final number.")

rows = [
    ("Run", "Setup", "Outcome"),
    ("LLM v1", "465 records, no numeric guidance in target text", "0% of generations contain a parseable number"),
    ("LLM v2", "465 records, NOAA numeric guidance appended to target", "73-81% parseable; MAE worse than baseline on 2/3 targets"),
    ("LLM v3/v4", "656 records (expanded archive), 2 format bugs fixed", "Narrative quality improves; numeric grounding limitation persists"),
    ("Decision v1", "Physics features only (flux, Bz, wind, Kp)", "R\u00b2 = 0.173 / 0.160 / \u22120.108"),
    ("Decision v2", "+ persistence (lag-1, lag-2) + flare-minute counts", "R\u00b2 = 0.537 / 0.842 / 0.436 \u2014 all beat the R\u00b2=0 baseline"),
]
top = Inches(2.15)
col_x = [Inches(0.9), Inches(2.5), Inches(7.0)]
col_w = [Inches(1.55), Inches(4.4), Inches(5.3)]
row_h = Inches(0.78)
for r, row in enumerate(rows):
    is_header = r == 0
    rowtop = top + r * row_h
    if not is_header and r % 2 == 0:
        band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), rowtop, Inches(11.4), row_h)
        band.fill.solid(); band.fill.fore_color.rgb = PANEL; band.line.fill.background(); band.shadow.inherit = False
    for c, text in enumerate(row):
        color = AMBER if is_header else (PAPER if c < 2 else PAPER)
        size = 13 if is_header else 13.5
        add_text(s, col_x[c] + Inches(0.1), rowtop + Inches(0.08), col_w[c] - Inches(0.2), row_h - Inches(0.1),
                  text, size, color, font=MONO, bold=is_header, line_spacing=1.1, anchor=MSO_ANCHOR.MIDDLE)
add_rule(s, Inches(0.9), top + Inches(0.78), Inches(11.4), color=AMBER_DIM, weight=1.25)
add_text(s, Inches(0.9), Inches(6.85), Inches(11.4), Inches(0.4),
          "Training dynamics, consistent across all LLM runs: best validation loss always lands at epoch 2; every run after that overfits (<450 training examples).",
          13, PAPER_DIM, font=MONO)

# ============================================================ Slide 8: Honest limitations
s = add_slide()
kicker_free_heading(s, "What doesn't work yet", "Stated plainly, not buried in a footnote.")
bullet_block(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(3.6), [
    "S1 radiation-storm % (R\u00b2 = 0.436) falls short of the other two targets, and we know exactly why: its defining physical driver, >10 MeV proton flux, is missing from the OMNI2 files we used \u2014 an upstream gap, not a connector defect.",
    "The fix is scoped, not attempted here: GOES's own SEISS proton sensor (sgps-l2-avg5m) carries the real signal, but ships as per-day files rather than one pinned annual archive \u2014 a clear next step for a later round.",
    "The fine-tuned language model still cannot be trusted for specific facts (it invents sunspot region numbers) \u2014 which is exactly why the numeric forecast comes from a separate, calculated model instead.",
], size=18)

# ============================================================ Slide 9: Close / links
s = add_slide()
kicker_free_heading(s, "Links", "Everything here is real and reproducible.")
bullet_block(s, Inches(0.9), Inches(2.3), Inches(11.5), Inches(3.0), [
    "Dataset PR: github.com/OpenTSLM/TimeNet/pull/842",
    "Project repo: github.com/asdspal/spacweather-cot",
    "Live page: site-eta-one-73.vercel.app",
    "Dataset license: CC0-1.0 \u2014 NOAA/NASA source data are U.S. government public domain",
], size=19, font=MONO)
add_text(s, Inches(0.9), Inches(5.6), Inches(10), Inches(0.6), "Demo \u2192", 24, AMBER, font=SERIF, italic=True)

prs.save("SpaceWeather-CoT-Pitch.pptx")
print("saved SpaceWeather-CoT-Pitch.pptx,", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
