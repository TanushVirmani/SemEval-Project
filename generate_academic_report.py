"""Build the academic report using only recorded, reproducible experiment results."""

import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
    Flowable,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "output/pdf/ADA_Project_Report.pdf"
INK = colors.HexColor("#172c43")
MUTED = colors.HexColor("#536476")
LIGHT = colors.HexColor("#eaf0f5")
WIDTH = A4[0] - 96


class Pipeline(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width = WIDTH
        self.height = 270

    def draw(self):
        c = self.canv
        boxes = [
            ("Code + language", "Input validation; no execution of submitted code"),
            (
                "Shared preprocessing",
                "Literal/comment normalization and structural statistics",
            ),
            (
                "Trained authorship ensemble",
                "Lexical logistic regression + boosting + randomized trees",
            ),
            (
                "Decision scores and explanation",
                "Validation-selected threshold; experimental prediction",
            ),
        ]
        y = 235
        for i, (title, detail) in enumerate(boxes):
            c.setFillColor(LIGHT)
            c.setStrokeColor(colors.HexColor("#cad5df"))
            c.roundRect(20, y - 35, WIDTH - 40, 49, 5, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(32, y - 3, title)
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8.5)
            c.drawString(32, y - 18, detail)
            if i < 3:
                c.setStrokeColor(MUTED)
                c.line(WIDTH / 2, y - 35, WIDTH / 2, y - 48)
                p = c.beginPath()
                p.moveTo(WIDTH / 2 - 3, y - 44)
                p.lineTo(WIDTH / 2, y - 48)
                p.lineTo(WIDTH / 2 + 3, y - 44)
                c.drawPath(p)
            y -= 65


def main():
    primary = json.loads(
        (BASE / "results/final-holdout/evaluation_metrics.json").read_text()
    )
    public = json.loads(
        (BASE / "results/improved-public/evaluation_metrics.json").read_text()
    )
    selection = json.loads((BASE / "model_selection.json").read_text())
    meta = primary["training"]
    manifest = json.loads(
        (BASE / "data/transfer/fresh_holdout_manifest.json").read_text()
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=31,
            textColor=INK,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            fontName="Helvetica",
            fontSize=13,
            leading=19,
            textColor=MUTED,
            spaceAfter=12,
        )
    )
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 10.5
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 9
    styles["Heading1"].fontName = "Helvetica-Bold"
    styles["Heading1"].fontSize = 18
    styles["Heading1"].leading = 23
    styles["Heading1"].textColor = INK
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 17
    styles["Heading2"].textColor = INK
    styles.add(
        ParagraphStyle(
            name="Small",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=9, leading=12)
    )
    story = []

    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    def h(text):
        p(text, "Heading2")

    def page(title):
        story.append(PageBreak())
        p(title, "Heading1")
        story.append(Spacer(1, 8))

    def table(rows, widths):
        converted = [
            [Paragraph(escape(str(cell)), styles["Cell"]) for cell in row]
            for row in rows
        ]
        t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#b9c8d6")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#dce3e9")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([t, Spacer(1, 12)])

    def class_table(m):
        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for name in ["Human-written", "Machine-generated"]:
            r = m["classification_report"][name]
            rows.append(
                [
                    name,
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['f1-score']:.4f}",
                    int(r["support"]),
                ]
            )
        table(rows, [155, 80, 80, 80, WIDTH - 395])

    def figure(path, width=440, height=320):
        story.extend(
            [
                Image(str(path), width=width, height=height, kind="proportional"),
                Spacer(1, 9),
            ]
        )

    # 1 - Cover. Unknown institutional fields are intentionally omitted.
    story.append(Spacer(1, 50))
    p("ACADEMIC PROJECT REPORT", "Subtitle")
    p(
        "Machine-Generated Code Detection<br/>and Cyclomatic Complexity Analysis",
        "ReportTitle",
    )
    p("A SemEval-inspired academic demonstration", "Subtitle")
    story.append(Spacer(1, 18))
    p("ADA Project", "Heading2")
    p("Submitted by <b>Tanush Virmani</b>")
    p("Bachelor of Technology<br/>Computer Science and Engineering")
    p("Amity School of Engineering &amp; Technology<br/>Amity University Uttar Pradesh")
    p("Batch 2024-2028<br/>26 October 2026")
    story.append(Spacer(1, 28))
    table(
        [
            ["Report scope", "Recorded outcome"],
            [
                "Final reserved holdout",
                f"{primary['accuracy']:.2%} view accuracy on {primary['test_samples']:,} snippets/views",
            ],
            [
                "Independent parent-file check",
                f"{primary.get('parent_accuracy', 0):.2%} accuracy across {primary.get('parent_files', 0)} parent files",
            ],
            [
                "Reused SemEval public sample",
                f"{public['accuracy']:.2%} accuracy on 1,000 examples; exploratory recheck",
            ],
        ],
        [190, WIDTH - 190],
    )
    p(
        "This report distinguishes the broader academic holdout from the original SemEval public sample. "
        "External training data is used in the deployed demo; this is not an official competition submission.",
        "Small",
    )

    # 2 - Abstract and reading map.
    page("Abstract and overview")
    p(
        "This project implements a local web application that estimates whether source code is human-written "
        "or machine-generated and independently measures cyclomatic complexity. The initial implementation "
        "used CodeBERT as a pretrained encoder, but lacked trained classification weights and a complete "
        "evaluation workflow. A runnable training and evaluation pipeline was added, followed by systematic "
        "experiments to address poor generalization."
    )
    p(
        "The original 2,000-example CodeBERT experiment obtained 95.8% validation accuracy but only 43.1% "
        "accuracy on the 1,000-example public SemEval sample. This gap motivated language/class balancing, "
        "broader production-code coverage, function-fragment views and validation-based model selection. "
        "The selected deployed model combines lexical TF-IDF classification with structural boosting and "
        "randomized-tree classifiers. Its decision threshold is chosen using validation data."
    )
    p(
        f"The final reserved holdout contains {primary['test_samples']:,} views from {primary['parent_files']} "
        f"previously uninspected parent files. The selected model obtained {primary['accuracy']:.2%} "
        f"view accuracy and {primary['parent_accuracy']:.2%} parent-file accuracy. The reused SemEval "
        f"sample obtained {public['accuracy']:.2%}. These results describe different distributions and are reported separately."
    )
    h("Implemented deliverables")
    p(
        "A Gradio interface; validated local checkpoint loading; CPU-efficient and CodeBERT inference; "
        "data preparation, training, threshold selection and evaluation scripts; per-example predictions; "
        "confusion matrices; provenance metadata; and automated tests."
    )
    h("Report map")
    table(
        [
            ["Page", "Section"],
            ["3", "Problem statement, objectives and ADA relevance"],
            ["4", "Datasets and evaluation design"],
            ["5", "System architecture and workflow"],
            ["6", "Algorithms and implementation"],
            ["7", "Experiments and model selection"],
            ["8", "Final reserved-holdout results"],
            ["9", "SemEval public-sample comparison"],
            ["10", "Verification, limitations and references"],
        ],
        [55, WIDTH - 55],
    )

    # 3 - Problem and objectives.
    page("1. Problem statement and objectives")
    p(
        "Given a source-code snippet and its programming language, the detector assigns one of two labels: "
        "<b>0 - Human-written</b> or <b>1 - Machine-generated</b>. The label is a statistical prediction, "
        "not a determination of authorship. Code copied, edited, generated by templates, or jointly authored "
        "by people and AI may not fit a strict binary interpretation."
    )
    h("Objectives")
    p(
        "1. Make the original project reproducible and prevent inference with randomly initialized classifier weights.<br/>"
        "2. Calculate accuracy, precision, recall, F1 and confusion matrices from labeled, held-out examples.<br/>"
        "3. Improve detection using broader training coverage and validation-selected models.<br/>"
        "4. Expose predictions and cyclomatic-complexity analysis through a local application.<br/>"
        "5. Preserve evidence, failed experiments and limitations rather than report only favorable numbers."
    )
    h("Connection to analysis and design of algorithms")
    p(
        "The application compares two computational approaches: a transformer encoder and sparse lexical/structural "
        "classifiers. The project therefore considers preprocessing, feature extraction, model inference and "
        "memory/runtime trade-offs. The CPU ensemble provides a practical alternative when transformer "
        "fine-tuning is expensive on the available machine."
    )
    p(
        "Cyclomatic complexity is a separate structural measure of control-flow paths. For a control-flow graph, "
        "the standard expression is V(G) = E - N + 2P, where E is the edge count, N the node count and P the "
        "number of connected components. The application uses Lizard to identify functions and count "
        "language-specific branch constructs. This measure is not the same as asymptotic runtime complexity."
    )
    h("Scope")
    p(
        "The academic demo supports the languages exposed by its current classifier and uses a binary "
        "authorship task inspired by SemEval Subtask A. It does not implement generator attribution, "
        "plagiarism detection or proof of misconduct. HTML can be classified in the broad demo, but "
        "cyclomatic-complexity analysis is unavailable for HTML."
    )

    # 4 - Data.
    page("2. Data and experimental methodology")
    table(
        [
            ["Dataset", "Use and provenance"],
            [
                "SemEval-2026 Task 13, configuration A",
                "Official train/validation data and the 1,000-example public test sample. Training emphasizes Python, C++ and Java.",
            ],
            [
                "CodeMirage",
                "Broader academic training coverage, including production-style code and normal/paraphrased AI examples. Original source: HanxiGuo/CodeMirage; license CC BY-NC-ND 4.0.",
            ],
            [
                "Final reserved files",
                f"{manifest['represented_parents']} remaining CodeMirage test parent files, not used in fitting or shared-validation selection.",
            ],
        ],
        [155, WIDTH - 155],
    )
    h("Preparation and grouping")
    p(
        "Empty or missing code/labels are rejected or removed at preparation time. Labels remain exactly "
        "0 and 1. Exact code hashes prevent direct reuse across fitting and evaluation. A conservative "
        "normalized fingerprint also removes identifier names, literal values, comments and whitespace "
        "to group obvious renamed or reformatted code. Grouping does not establish that all semantically "
        "related programs or repository families are independent."
    )
    p(
        "Function views are extracted using Lizard; a deterministic contiguous window is used when suitable "
        "functions are unavailable. Each view inherits its parent label. This broadens the input-length "
        "distribution, but inherited labels can be noisy when one file contains mixed or generated components."
    )
    h("Separation of roles")
    p(
        "Training data fits model parameters. Validation data selects models and thresholds. A shared "
        "1,000-example validation subset compares the completed final candidates. Reserved parent files "
        "are evaluated only after this choice is locked. Repeated code patterns matching fitting, tuning "
        "or previously inspected views are removed before the final holdout evaluation."
    )
    p(
        "The SemEval public sample was inspected during the original failure analysis. Subsequent results "
        "on that sample are development rechecks, not fresh estimates. The broader track uses external "
        "training data and is not competition-compliant. Dataset revisions, file hashes and selection "
        "records are retained in manifests and training metadata."
    )

    # 5 - Architecture.
    page("3. System architecture")
    story.append(Pipeline())
    h("Parallel complexity path")
    p(
        "The same input is sent separately to Lizard for function-level complexity, lines of code and token "
        "counts. A horizontal chart shows complexity per function. Complexity analysis remains available "
        "when classifier weights are missing; the interface reports the detector as unavailable instead "
        "of inventing a prediction."
    )
    h("Module responsibilities")
    table(
        [
            ["Module", "Responsibility"],
            [
                "app.py",
                "Local Gradio UI, input handling, scores and complexity display",
            ],
            [
                "detector.py / model_utils.py",
                "Shared inference, tokenization, checkpoint validation and decision scores",
            ],
            [
                "improve_model.py / fragment_views.py",
                "CPU models, language/class balancing and fragment preparation",
            ],
            [
                "train_model.py",
                "CodeBERT training, resumable checkpoints and optional layer freezing",
            ],
            [
                "evaluate_model.py",
                "Metrics, leakage guards, predictions, parent aggregation and confusion matrix",
            ],
            [
                "model_selection.json",
                "Recorded validation-only choice of the deployed model",
            ],
        ],
        [185, WIDTH - 185],
    )

    # 6 - Algorithms.
    page("4. Algorithms and implementation")
    h("Selected authorship classifier")
    p(
        "The selected model is <b>lexical_C1.0_ensemble</b>. Word/token TF-IDF uses 1-2 grams, with an "
        "80,000-feature cap. Character TF-IDF uses 3-5 grams, with a 100,000-feature cap. Comment and "
        "literal normalization reduces some superficial differences. Structural statistics include "
        "line and identifier lengths, punctuation ratios, indentation and control-flow keyword frequencies."
    )
    p(
        "The lexical branch uses logistic regression with C = 1.0. Two structural branches use histogram "
        "gradient boosting and an ExtraTrees ensemble. Their three machine-class scores are averaged. "
        "Candidate branches and ensembles are selected using validation macro F1. These feature caps "
        "are configuration limits, not claims about the final learned vocabulary size."
    )
    h("Decision threshold")
    p(
        f"The deployed threshold is t = {meta.get('decision_threshold', 0.5):.2f}, chosen on validation data. "
        "For consistent UI decisions, raw machine score p is mapped to p* = p(1-t) / "
        "[p(1-t) + (1-p)t]. A displayed machine score above 0.5 is equivalent to a raw score above t. "
        "These values are uncalibrated decision scores, not verified probabilities of authorship."
    )
    h("Computational considerations")
    table(
        [
            ["Operation", "Main cost / interpretation"],
            [
                "TF-IDF transformation",
                "Character/token scans for fixed n-gram lengths; sparse output",
            ],
            [
                "Logistic inference",
                "Sparse dot product, proportional to active feature count",
            ],
            [
                "Tree inference",
                "One traversal per tree; depends on tree count and depth",
            ],
            [
                "Transformer inference",
                "Per-layer attention has a quadratic dependence on sequence length; feed-forward layers also contribute",
            ],
            [
                "Cyclomatic analysis",
                "Static parsing and branch counting; no source code is executed",
            ],
        ],
        [150, WIDTH - 150],
    )
    p(
        "No controlled latency benchmark was conducted. Layer freezing reduces backpropagation work but "
        "does not remove the transformer forward pass. The local CPU was used for the recorded experiments.",
        "Small",
    )

    # 7 - Experiments.
    page("5. Experiments and model selection")
    p(
        "Experiments were preserved even when their public-sample results were worse. High scores on "
        "similar validation data did not guarantee performance across languages, domains or snippet types."
    )
    table(
        [
            ["Exploratory experiment", "Public SemEval recheck"],
            ["Original CodeBERT; 2,000 official training examples", "43.1%"],
            ["Balanced 48,000-example CPU ensemble", "29.7%"],
            ["Broader full-file CPU ensemble", "49.0%"],
            ["Fragment-aware ensemble; validation-selected threshold", "66.3%"],
        ],
        [340, WIDTH - 340],
    )
    h("Completed shared-validation comparison")
    rows = [["Candidate", "Accuracy", "Macro F1"]]
    labels = {
        "checkpoints-fragments": "Fragment-aware CPU ensemble",
        "checkpoints-transfer": "CodeBERT transfer; 8 lower layers frozen",
        "checkpoints-external": "Imported external detector",
    }
    for name, r in selection["candidates"].items():
        rows.append(
            [labels.get(name, name), f"{r['accuracy']:.1%}", f"{r['macro_f1']:.4f}"]
        )
    table(rows, [300, 95, WIDTH - 395])
    p(
        "The fragment-aware ensemble was selected using the shared validation macro F1. CodeBERT transfer "
        "used 4,000 examples, 128 first-and-last tokens, one epoch, and a 5e-5 learning rate. The external "
        "detector was a comparison only; its precise training dataset is not documented by its authors, "
        "so its training/evaluation overlap could not be independently checked. It was not deployed."
    )
    p(
        "The final reserved parent files were not used in this candidate selection. The model-selection "
        "record includes the validation-file SHA-256 hash and scores for all completed candidates."
    )

    # 8 - Primary held-out results.
    page("6. Final reserved-holdout results")
    p(
        f"Evaluation used <b>{primary['test_samples']:,} snippets/views</b> from <b>{primary['parent_files']} parent files</b>. "
        f"The majority-class accuracy baseline is {primary['majority_class_accuracy']:.2%}. "
        f"View accuracy is <b>{primary['accuracy']:.2%}</b>, macro precision {primary['macro_precision']:.4f}, "
        f"macro recall {primary['macro_recall']:.4f} and macro F1 <b>{primary['macro_f1']:.4f}</b>."
    )
    class_table(primary)
    figure(BASE / "results/final-holdout/confusion_matrix.png", width=420, height=300)
    p(
        f"Averaging scores across each parent's views gives <b>{primary['parent_accuracy']:.2%} parent-file accuracy</b> "
        f"and {primary['parent_macro_f1']:.4f} macro F1. Views from one parent are correlated; these two "
        "levels must not be treated as independent sample counts."
    )
    p(
        "This result measures the specified reserved CodeMirage-based task. It does not establish the same "
        "accuracy on all source code, generators, domains or future submissions.",
        "Small",
    )

    # 9 - Separate public benchmark.
    page("7. Original SemEval sample: separate result")
    p(
        f"The same 1,000-example public sample used in the original project now gives <b>{public['accuracy']:.2%}</b> "
        f"accuracy, macro precision {public['macro_precision']:.4f}, macro recall {public['macro_recall']:.4f} "
        f"and macro F1 {public['macro_f1']:.4f}. This is an exploratory recheck after development."
    )
    class_table(public)
    figure(BASE / "results/improved-public/confusion_matrix.png", width=420, height=300)
    p(
        "Accuracy improved from 43.1% to 66.3%, but remains below both 75% and the sample's 77.7% "
        "majority-class accuracy baseline. The greater-than-75% objective is met on the "
        "separately described broader holdout (79.66%); it is "
        "<b>not achieved on this original SemEval sample</b>. The two benchmarks must not be merged "
        "or presented as directly interchangeable results."
    )

    # 10 - Verification, limitations and references.
    page("8. Verification, limitations and future work")
    h("Verification completed")
    p(
        "Automated checks cover invalid/fractional/missing labels, both-class requirements, missing "
        "checkpoints, exact training/evaluation overlap, normalization, finite structural features, "
        "deterministic fragment extraction, threshold mapping, and first-and-last token handling. "
        "Saved predictions are checked against accuracy, macro F1 and confusion matrices. The local "
        "application was opened and tested through its browser interface."
    )
    h("Limitations and next steps")
    p(
        "Source-level labels can be noisy; function views may inherit imperfect labels. Normalized "
        "fingerprints cannot detect all semantic clones or shared repositories. Confidence is "
        "uncalibrated, and no detector output should be treated as proof of misconduct. Larger and "
        "better-grouped datasets, generator/repository-held-out validation, stronger code encoders, "
        "and carefully controlled GPU experiments are appropriate next steps. Thresholds must not "
        "be tuned on final test labels simply to exceed a target accuracy."
    )
    h("Reproduction and local use")
    p(
        "README.md and IMPROVEMENTS.md describe the environment and experiment commands. Run app.py "
        "from the project environment to use http://127.0.0.1:7860. active_model.json identifies the "
        "selected local checkpoint. The repository retains the original CodeBERT model and results "
        "as separate local artifacts. Raw datasets, caches and large local weights are ignored by Git."
    )
    h("References and software sources")
    refs = [
        "[1] SemEval-2026 Task 13 organizers. Task definition and restrictions. https://github.com/mbzuai-nlp/SemEval-2026-Task13",
        "[2] DaniilOr/SemEval-2026-Task13. Official dataset. https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13",
        "[3] HanxiGuo/CodeMirage. Dataset card and provenance. https://huggingface.co/datasets/HanxiGuo/CodeMirage",
        "[4] Microsoft CodeBERT. Pretrained code encoder. https://github.com/microsoft/CodeBERT",
        "[5] Lizard static complexity analyzer. https://github.com/terryyin/lizard",
        "[6] scikit-learn. Feature extraction and classifier implementations. https://scikit-learn.org/stable/",
        "[7] azherali/CodeGenDetect-CodeBert. External comparison model card. https://huggingface.co/azherali/CodeGenDetect-CodeBert",
    ]
    for ref in refs:
        p(escape(ref), "Small")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d6dfe7"))
        canvas.line(48, 43, A4[0] - 48, 43)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(48, 29, "Tanush Virmani | ADA Project | Academic demo")
        canvas.drawRightString(A4[0] - 48, 29, str(doc.page))
        canvas.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=55,
        title="Machine-Generated Code Detection and Cyclomatic Complexity Analysis",
        author="Tanush Virmani",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
