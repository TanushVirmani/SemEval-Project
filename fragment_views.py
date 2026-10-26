"""Label-preserving full-file/function views; never execute supplied code."""

import hashlib
import random
import lizard

FILENAMES = {
    "Python": "x.py",
    "C++": "x.cpp",
    "CPP": "x.cpp",
    "Java": "x.java",
    "C#": "x.cs",
    "CSharp": "x.cs",
    "JavaScript": "x.js",
    "C": "x.c",
    "Go": "x.go",
    "Ruby": "x.rb",
    "PHP": "x.php",
}


def fragments(code, language, limit=3):
    lines = code.splitlines()
    if len(lines) < 5:
        return []
    rng = random.Random(int(hashlib.sha256(code.encode()).hexdigest()[:16], 16))
    spans = []
    if language in FILENAMES:
        try:
            analysis = lizard.analyze_file.analyze_source_code(
                FILENAMES[language], code
            )
            spans = [(f.start_line - 1, f.end_line) for f in analysis.function_list]
        except (ValueError, IndexError, RecursionError):
            spans = []
    rng.shuffle(spans)
    result = []
    for start, end in spans:
        snippet = "\n".join(lines[start:end]).strip()
        if len(snippet) >= 40 and snippet != code.strip() and snippet not in result:
            result.append(snippet)
        if len(result) >= limit:
            break
    if not result and len(lines) >= 16:
        length = min(40, max(8, len(lines) // 3))
        for _ in range(min(limit, 2)):
            start = rng.randrange(max(1, len(lines) - length + 1))
            snippet = "\n".join(lines[start : start + length]).strip()
            if len(snippet) >= 40 and snippet not in result:
                result.append(snippet)
    return result
