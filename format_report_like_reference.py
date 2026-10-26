"""Restyle the current report in the supplied Amity academic-report format.

All project prose, model comparisons and result tables are reused from the
existing report generator; reference-project content is not imported.
"""

import contextlib
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
    Flowable,
    KeepTogether,
)
import generate_academic_report as original

BASE = Path(__file__).resolve().parent
OUT = BASE / "output/pdf/ADA_Project_Report_Reference_Format.pdf"
W = A4[0] - 144
FONTS = Path("/System/Library/Fonts/Supplemental")
for name, file in [
    ("TNR", "Times New Roman.ttf"),
    ("TNR-Bold", "Times New Roman Bold.ttf"),
    ("TNR-Italic", "Times New Roman Italic.ttf"),
    ("TNR-BoldItalic", "Times New Roman Bold Italic.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily(
    "TNR",
    normal="TNR",
    bold="TNR-Bold",
    italic="TNR-Italic",
    boldItalic="TNR-BoldItalic",
)
BODY = ParagraphStyle(
    "body", fontName="TNR", fontSize=12, leading=15, alignment=TA_JUSTIFY, spaceAfter=8
)
H1 = ParagraphStyle(
    "h1",
    parent=BODY,
    fontName="TNR-Bold",
    fontSize=13,
    leading=16,
    spaceBefore=0,
    spaceAfter=9,
    alignment=TA_LEFT,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "h2",
    parent=BODY,
    fontName="TNR-Bold",
    fontSize=12,
    leading=15,
    spaceBefore=7,
    spaceAfter=7,
    alignment=TA_LEFT,
    keepWithNext=True,
)
CELL = ParagraphStyle(
    "cell", parent=BODY, fontSize=11, leading=13, spaceAfter=0, alignment=TA_LEFT
)
CAP = ParagraphStyle(
    "caption",
    parent=BODY,
    fontName="TNR-Italic",
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    spaceBefore=5,
    spaceAfter=10,
)
CENTER = ParagraphStyle(
    "center", parent=BODY, alignment=TA_CENTER, spaceAfter=4, leading=16
)
BOLD_CENTER = ParagraphStyle("bold-center", parent=CENTER, fontName="TNR-Bold")


class Capture:
    story = None

    def __init__(self, *args, **kwargs):
        pass

    def build(self, story, **kwargs):
        Capture.story = story


def capture_content():
    real = original.SimpleDocTemplate
    original.SimpleDocTemplate = Capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            original.main()
    finally:
        original.SimpleDocTemplate = real
    sections = {}
    current = None
    for f in Capture.story:
        if isinstance(f, Paragraph) and f.style.name == "Heading1":
            current = f.getPlainText()
            sections[current] = []
        elif current is not None and not isinstance(f, PageBreak):
            sections[current].append(f)
    return sections


def split_chunks(flows):
    result = {}
    name = "intro"
    result[name] = []
    for f in flows:
        if isinstance(f, Paragraph) and f.style.name == "Heading2":
            name = f.getPlainText()
            result[name] = []
        else:
            result[name].append(f)
    return result


class MonochromePipeline(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width = W
        self.height = 200

    def draw(self):
        c = self.canv
        boxes = [
            (2, 122, 112, 48, ["Source code", "+ language"]),
            (166, 122, 118, 48, ["TF-IDF and", "structural features"]),
            (333, 122, 112, 48, ["Three-branch", "classifier ensemble"]),
            (2, 15, 112, 48, ["Static parsing", "with Lizard"]),
            (166, 15, 118, 48, ["Per-function", "complexity metrics"]),
            (333, 15, 112, 48, ["Scores, prediction", "and complexity plot"]),
        ]
        for x, y, w, h, lines in boxes:
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.8)
            c.rect(x, y, w, h, stroke=1, fill=0)
            c.setFont("TNR-Bold", 10)
            for i, line in enumerate(lines):
                c.drawCentredString(x + w / 2, y + h / 2 + 5 - i * 12, line)

        def arrow(x1, y1, x2, y2):
            c.line(x1, y1, x2, y2)
            import math

            a = math.atan2(y2 - y1, x2 - x1)
            for d in [-0.45, 0.45]:
                c.line(x2, y2, x2 - 5 * math.cos(a + d), y2 - 5 * math.sin(a + d))

        for args in [
            (114, 146, 166, 146),
            (284, 146, 333, 146),
            (58, 122, 58, 63),
            (114, 39, 166, 39),
            (284, 39, 333, 39),
            (389, 122, 389, 63),
        ]:
            arrow(*args)


def main():
    src = capture_content()
    cover = json.loads((BASE / "report_cover_details.json").read_text())
    primary = json.loads(
        (BASE / "results/final-holdout/evaluation_metrics.json").read_text()
    )
    public = json.loads(
        (BASE / "results/improved-public/evaluation_metrics.json").read_text()
    )
    story = []
    table_no = 0
    figure_no = 0

    def p(text, style=BODY):
        story.append(Paragraph(text, style))

    def section(title):
        story.append(PageBreak())
        p(title, H1)

    def h(text):
        p(text, H2)

    def table(rows, widths, caption):
        nonlocal table_no
        formatted = []
        for i, row in enumerate(rows):
            style = (
                ParagraphStyle(
                    "header-cell",
                    parent=CELL,
                    fontName="TNR-Bold",
                    textColor=colors.white,
                )
                if i == 0
                else CELL
            )
            formatted.append(
                [
                    Paragraph(
                        c.text if isinstance(c, Paragraph) else escape(str(c)), style
                    )
                    for c in row
                ]
            )
        widths = [w / sum(widths) * W for w in widths]
        t = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#777777")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        table_no += 1
        p(f"Table {table_no}. {caption}", CAP)

    def existing_table(f, caption):
        table(f._cellvalues, f._colWidths, caption)

    def flows(items):
        for f in items:
            if isinstance(f, Paragraph):
                p(f.text, H2 if f.style.name == "Heading2" else BODY)
            elif isinstance(f, Table):
                existing_table(f, "Recorded project information.")
            elif isinstance(f, Image):
                nonlocal figure_no
                story.append(
                    Image(
                        f.filename, width=min(W, 420), height=280, kind="proportional"
                    )
                )
                figure_no += 1
                p(
                    f"Figure {figure_no}. Confusion matrix for the evaluation described above.",
                    CAP,
                )
            elif isinstance(f, Spacer):
                if f.height < 15:
                    story.append(Spacer(1, min(f.height, 5)))

    def figure(file, caption, height=285):
        nonlocal figure_no
        story.append(Image(str(file), width=W, height=height, kind="proportional"))
        figure_no += 1
        p(f"Figure {figure_no}. {caption}", CAP)

    # Cover: border, centered Times New Roman, logo and user-provided details.
    story.append(Spacer(1, 34))
    p("Project report", CENTER)
    p("on", CENTER)
    p(
        "Machine-Generated Code Detection and Cyclomatic Complexity Analysis",
        BOLD_CENTER,
    )
    p(f"Course code: {cover['course']}", CENTER)
    p("Submitted to", CENTER)
    p(cover["university"], CENTER)
    story.append(Spacer(1, 8))
    story.append(
        Image(
            str(BASE / "assets/reference-cover-logo-0.jpg"),
            width=118,
            height=139,
            kind="proportional",
        )
    )
    story.append(Spacer(1, 11))
    p(
        "In partial fulfillment of the requirements for the award of the degree of",
        CENTER,
    )
    p(cover["programme"], CENTER)
    p("In", CENTER)
    p(cover["discipline"], CENTER)
    story.append(Spacer(1, 4))
    p("by", CENTER)
    p("Name: " + cover["name"], BOLD_CENTER)
    p("Enrollment number: " + cover["enrollment"], CENTER)
    story.append(Spacer(1, 4))
    p("Under the guidance of", CENTER)
    p(cover["guide"], BOLD_CENTER)
    p(cover["designation"], CENTER)
    story.append(Spacer(1, 5))
    p("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", CENTER)
    p("AMITY SCHOOL OF ENGINEERING &amp; TECHNOLOGY", CENTER)
    p("AMITY UNIVERSITY UTTAR PRADESH, NOIDA", CENTER)
    p("BATCH: " + cover["batch"], BOLD_CENTER)
    p("26 October 2026", CENTER)

    # 1. Preserve the abstract/overview as the formal introduction.
    section("1. INTRODUCTION")
    intro = split_chunks(src["Abstract and overview"])
    flows(intro["intro"])
    h("Key Features of the Proposed System")
    flows(intro["Implemented deliverables"])
    h("Project Workflow")
    story.append(MonochromePipeline())
    figure_no += 1
    p(
        f"Figure {figure_no}. Code input, authorship classification and parallel complexity workflow.",
        CAP,
    )

    # 2. Same content, numbered subsections as in the reference.
    section("2. PROBLEM STATEMENT AND OBJECTIVES")
    problem = split_chunks(src["1. Problem statement and objectives"])
    h("2.1 Problem Statement")
    flows(problem["intro"])
    h("2.2 Objectives")
    flows(problem["Objectives"])
    h("2.3 Scope")
    flows(problem["Scope"])
    h("Connection to Analysis and Design of Algorithms")
    flows(problem["Connection to analysis and design of algorithms"])

    # 3. Existing architecture and module responsibilities.
    section("3. PROPOSED SYSTEM DESIGN AND ARCHITECTURE")
    p(
        "The system separates authorship inference from static complexity analysis. Both paths operate on the submitted code without executing it."
    )
    story.append(MonochromePipeline())
    figure_no += 1
    p(
        f"Figure {figure_no}. Proposed classifier and complexity-analysis architecture.",
        CAP,
    )
    arch = split_chunks(src["3. System architecture"])
    h("3.1 Major Modules")
    for f in arch["Module responsibilities"]:
        if isinstance(f, Table):
            existing_table(f, "Major modules and their responsibilities.")
    h("3.2 System Data Flow")
    flows(arch["Parallel complexity path"])

    # 4. Data and methodology, unchanged measured counts.
    section("4. DATA SOURCES AND PREPROCESSING")
    data = split_chunks(src["2. Data and experimental methodology"])
    h("4.1 Dataset Details")
    for f in data["intro"]:
        if isinstance(f, Table):
            existing_table(f, "Data sources, roles and provenance.")
        elif isinstance(f, Paragraph):
            p(f.text)
    h("4.2 Preparation and Duplicate Grouping")
    flows(data["Preparation and grouping"])
    h("4.3 Separation of Training, Validation and Evaluation")
    flows(data["Separation of roles"])

    # 5. Core model, parameters, computational considerations.
    section("5. MACHINE LEARNING MODEL AND METHODOLOGY")
    methods = split_chunks(src["4. Algorithms and implementation"])
    h("5.1 Selected Authorship Classifier")
    flows(methods["Selected authorship classifier"])
    h("5.2 Recorded Model Settings")
    table(
        [
            ["Setting", "Value"],
            ["Word/token TF-IDF", "1-2 grams; maximum 80,000 features"],
            ["Character TF-IDF", "3-5 grams; maximum 100,000 features"],
            ["Lexical classifier", "Logistic regression, C = 1.0"],
            ["Structural branches", "Histogram gradient boosting and ExtraTrees"],
            ["Ensemble rule", "Mean of three machine-class scores"],
            ["Decision threshold", "0.55; selected using validation data"],
            ["Model selection", "Shared-validation macro F1"],
        ],
        [165, W - 165],
        "Model configuration used by the academic demo.",
    )
    h("5.3 Computational Considerations")
    for f in methods["Computational considerations"]:
        if isinstance(f, Table):
            existing_table(f, "Algorithmic cost and interpretation.")
        elif isinstance(f, Paragraph):
            p(f.text)

    section("6. EXPLAINABILITY AND EVIDENCE GENERATION")
    h("6.1 Decision Scores and Threshold")
    flows(methods["Decision threshold"])
    h("6.2 Cyclomatic Complexity Evidence")
    flows(arch["Parallel complexity path"])
    p(
        "Per-function complexity, lines of code and token counts provide structural information alongside the authorship estimate. They do not explain the causal basis of the classifier decision or prove who wrote a program."
    )
    h("6.3 Responsible Interpretation")
    p(
        "Predictions and uncalibrated confidence scores are estimates, not proof of authorship. Cyclomatic complexity measures control-flow structure, not asymptotic runtime. A human reviewer must interpret both outputs in context."
    )

    section("7. IMPLEMENTATION AND ENVIRONMENT")
    h("7.1 Software and Hardware")
    table(
        [
            ["Component", "Recorded implementation"],
            ["Language", "Python 3.11"],
            ["Application", "Gradio local web interface"],
            [
                "Machine learning",
                "scikit-learn; PyTorch and Hugging Face Transformers for CodeBERT experiments",
            ],
            ["Data and numerical work", "pandas, NumPy, SciPy and joblib"],
            ["Static analysis", "Lizard"],
            ["Visualization and report", "Matplotlib and ReportLab"],
            ["Hardware", "Local CPU; no usable GPU in the execution environment"],
        ],
        [140, W - 140],
        "Implementation environment.",
    )
    h("7.2 Training and Inference Procedure")
    p(
        "Prepare and validate labeled data; separate fitting and evaluation groups; generate permitted training views; fit candidate models; choose the model and decision threshold using validation data; save metadata and checkpoints; evaluate the locked model; expose inference and complexity analysis in Gradio."
    )
    h("7.3 Shared Inference and Checkpoint Handling")
    p(
        "The shared detector interface supports both the CPU ensemble and trained CodeBERT checkpoints. Missing or incompatible classification weights are rejected. The original untrained-base-model fallback is removed. CodeBERT transfer uses first-and-last token handling; ensemble inference uses lexical and structural features from the full snippet."
    )

    # 8. Preserve all actual experiment and evaluation tables, with captions.
    section("8. RESULTS AND PERFORMANCE ANALYSIS")
    experiments = split_chunks(src["5. Experiments and model selection"])
    h("8.1 Exploratory Experiments")
    flows(experiments["intro"])
    h("8.2 Completed Shared-Validation Comparison")
    flows(experiments["Completed shared-validation comparison"])
    story.append(PageBreak())
    h("8.3 Final Reserved-Holdout Results")
    result = src["6. Final reserved-holdout results"]
    for f in result:
        if isinstance(f, Table):
            existing_table(
                f,
                "Class-wise precision, recall, F1-score and support on the reserved holdout.",
            )
        elif isinstance(f, Image):
            figure(
                BASE / "results/final-holdout/confusion_matrix.png",
                "Confusion matrix on 1,450 reserved snippets/views from 454 parent files.",
                height=280,
            )
        elif isinstance(f, Paragraph):
            p(f.text)
    story.append(PageBreak())
    h("8.4 Original SemEval Public Sample: Separate Result")
    for f in src["7. Original SemEval sample: separate result"]:
        if isinstance(f, Table):
            existing_table(
                f,
                "Class-wise results on the reused 1,000-example SemEval public sample.",
            )
        elif isinstance(f, Image):
            figure(
                BASE / "results/improved-public/confusion_matrix.png",
                "Confusion matrix for the SemEval public-sample development recheck.",
                height=280,
            )
        elif isinstance(f, Paragraph):
            p(f.text)

    last = split_chunks(src["8. Verification, limitations and future work"])
    section("9. ETHICAL CONSIDERATIONS")
    p(
        "Authorship detection can produce false positives and false negatives. A human-written submission can be wrongly flagged, while AI-written code can be missed. The application is an academic demonstration and should not be used as a standalone basis for an accusation, grade penalty or consequential decision."
    )
    p(
        "Source-level labels may be imperfect, and function views inherit their parent labels. Differences in language, domain, formatting, code length and generator family affect generalization. Confidence values are uncalibrated decision scores; high confidence is not equivalent to established authorship."
    )
    p(
        "Downloaded datasets, local weights and raw prediction files remain separate from tracked code and summary metrics. The broader demo uses external CodeMirage training data, whose source and license are recorded. This configuration is not an official competition submission."
    )
    p(
        "The report retains weak and unsuccessful experiments and distinguishes the reserved holdout from the previously inspected public sample. Thresholds must not be tuned on final test labels merely to exceed an accuracy target."
    )

    section("10. TESTING AND VALIDATION")
    h("10.1 Validation Plan")
    table(
        [
            ["Test", "Verified behavior"],
            [
                "CSV labels and missing data",
                "Invalid, fractional or missing labels are rejected; both classes are required",
            ],
            [
                "Checkpoint handling",
                "Missing or incompatible trained classifiers are rejected",
            ],
            [
                "Leakage guard",
                "Evaluation rejects snippets used in fitting or validation",
            ],
            [
                "Preprocessing and features",
                "Normalization is consistent; structural features remain finite",
            ],
            [
                "Fragment generation",
                "Extraction is deterministic and does not execute code",
            ],
            [
                "Decision scores",
                "Threshold mapping preserves the intended class boundary",
            ],
            [
                "CodeBERT tokens",
                "First-and-last token handling retains both ends within the token budget",
            ],
        ],
        [150, W - 150],
        "Functional validation checks implemented in the test suite.",
    )
    h("10.2 Model Validation")
    flows(last["Verification completed"])
    h("10.3 Grouped Final Evaluation")
    p(
        "The final reserved evaluation is performed after shared-validation selection is locked. Because several views may come from one parent file, the report includes both view-level metrics and a parent-level check based on averaged scores. Parent and view counts must not be treated as independent sample counts."
    )

    section("11. ADVANTAGES, LIMITATIONS AND FUTURE SCOPE")
    h("11.1 Advantages")
    p(
        "The project provides a reproducible local workflow, trained-model guards, a CPU-efficient classifier, independent cyclomatic-complexity analysis, explicit data provenance, and recorded model-selection evidence. CodeBERT and unsuccessful variants remain available for comparison instead of being hidden."
    )
    h("11.2 Limitations and 11.3 Future Scope")
    flows(last["Limitations and next steps"])
    p(
        "The current greater-than-75% result applies to the separately described reserved CodeMirage-based holdout. The original SemEval sample still scores 66.3%. This remaining distribution gap is a limitation, not an improvement that can be assumed from results on another dataset."
    )

    section("12. CONCLUSION AND REPRODUCIBILITY")
    h("12.1 Overall Results")
    p(
        f"The final deployed ensemble obtained {primary['accuracy']:.2%} accuracy across {primary['test_samples']:,} reserved views, with macro precision {primary['macro_precision']:.4f}, macro recall {primary['macro_recall']:.4f} and macro F1 {primary['macro_f1']:.4f}. Parent-file accuracy was {primary['parent_accuracy']:.2%} across {primary['parent_files']} files. The reused SemEval public sample achieved {public['accuracy']:.2%}, compared with the original 43.1% baseline."
    )
    h("12.2 Conclusion")
    p(
        "The project now supplies a working authorship-detection and complexity-analysis application with reproducible evaluation. Broader coverage and fragment-aware preparation improved the demonstration, while the independent public-sample gap shows why dataset-specific results must be interpreted carefully. The detector remains experimental and supports human review rather than replacing it."
    )
    h("12.3 Reproducibility Checklist")
    flows(last["Reproduction and local use"])
    table(
        [
            ["Artifact", "Purpose"],
            [
                "README.md / IMPROVEMENTS.md",
                "Environment, experiment commands, provenance and limitations",
            ],
            ["active_model.json", "Selected local checkpoint"],
            ["model_selection.json", "Shared-validation selection and file hash"],
            [
                "Training metadata and manifests",
                "Data revisions, row counts, hashes and settings",
            ],
            [
                "Evaluation outputs",
                "Metrics, per-example predictions and confusion matrices",
            ],
            ["Tests", "Data, inference, grouping, threshold and tokenization checks"],
        ],
        [170, W - 170],
        "Project artifacts supporting reproducibility.",
    )

    section("13. REFERENCES")
    reference_style = ParagraphStyle(
        "reference",
        parent=BODY,
        alignment=TA_LEFT,
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=10,
    )
    for item in last["References and software sources"]:
        if isinstance(item, Paragraph):
            p(item.text, reference_style)

    def draw_cover(c, doc):
        c.saveState()
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.rect(24, 24, A4[0] - 48, A4[1] - 48, stroke=1, fill=0)
        c.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=60,
        title="Machine-Generated Code Detection and Cyclomatic Complexity Analysis",
        author=cover["name"],
    ).build(story, onFirstPage=draw_cover)
    print(OUT)


if __name__ == "__main__":
    main()
