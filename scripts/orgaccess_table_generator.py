import json
import sys
from datetime import datetime
from pathlib import Path
import re

# ---------------------------------------------------
# OrgAccess Official Baseline (HARD SPLIT ONLY)
# ---------------------------------------------------
ORGACCESS_HARD_BASELINE = {
    "Gemma-3-4B":                {"accuracy": 0.13, "f1": 0.09},
    "Qwen-2.5-7B":               {"accuracy": 0.19, "f1": 0.15},
    "Mistral-7B":                {"accuracy": 0.17, "f1": 0.14},
    "Llama-3.1-8B":              {"accuracy": 0.16, "f1": 0.11},
    "Aya-Expanse-8B":            {"accuracy": 0.18, "f1": 0.12},
    "Falcon-3-10B":              {"accuracy": 0.17, "f1": 0.13},
    "Gemma-3-12B":               {"accuracy": 0.20, "f1": 0.16},
    "Qwen-2.5-14B":              {"accuracy": 0.19, "f1": 0.10},
    "Phi-4-14B":                 {"accuracy": 0.20, "f1": 0.10},
    "Mistral-Small-3.1-24B":     {"accuracy": 0.22, "f1": 0.18},
}

# ---------------------------------------------------
# UNIVERSAL METRIC EXTRACTION LOGIC
# ---------------------------------------------------
def extract_metrics(json_path):
    """Extract accuracy + f1_macro from ANY OrgAccess results file format."""

    with open(json_path, "r") as f:
        data = json.load(f)

    model = (
        data.get("model_name")
        or data.get("model")
        or Path(json_path).stem
    )

    # ------------- FORMAT A: top-level metrics -------------
    if "accuracy" in data and "f1_macro" in data:
        return model, data["accuracy"], data["f1_macro"]

    # ------------- FORMAT B: results → known difficulty keys -------------
    if "results" in data:
        results = data["results"]

        # Direct formats: hard, medium
        for key in ["hard", "hard_test", "medium", "medium_test"]:
            if key in results:
                metrics = results[key].get("metrics", results[key])
                return model, metrics.get("accuracy", 0.0), metrics.get("f1_macro", 0.0)

        # ------------- FORMAT C: sharded keys like hard-00000-of-00001 -------------
        shard_key = next(
            (k for k in results.keys() if re.search(r"hard", k, re.IGNORECASE)),
            None
        )
        if shard_key:
            metrics = results[shard_key].get("metrics", results[shard_key])
            return model, metrics.get("accuracy", 0.0), metrics.get("f1_macro", 0.0)

        # ------------- FORMAT D: grouped_metrics -------------
        if "grouped_metrics" in data:
            gm = data["grouped_metrics"]
            if "HARD" in gm:
                m = gm["HARD"].get("metrics", gm["HARD"])
                return model, m.get("accuracy", 0.0), m.get("f1_macro", 0.0)

    raise KeyError(
        f"❌ Could not find accuracy/f1_macro in any recognized OrgAccess format for file: {json_path}"
    )


# ---------------------------------------------------
# Format LaTeX table row
# ---------------------------------------------------
def format_latex_row(model, acc, f1):
    baseline = ORGACCESS_HARD_BASELINE.get(model, {"accuracy": 0.0, "f1": 0.0})
    delta_acc = acc - baseline["accuracy"]
    delta_f1  = f1 - baseline["f1"]

    return (
        f"{model} & "
        f"{acc:.3f} & "
        f"{f1:.3f} & "
        f"{delta_acc:+.3f} & "
        f"{delta_f1:+.3f} \\\\"
    )


# ---------------------------------------------------
# Append to LaTeX output
# ---------------------------------------------------
def append_to_table(latex_line, output_file="orgaccess_latex_table.tex"):
    header = (
        "% Auto-generated OrgAccess Hard-Split Results Table\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Accuracy & F1$_{macro}$ & $\\Delta$Acc & $\\Delta$F1 \\\\\n"
        "\\midrule\n"
    )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{OrgAccess Hard-Split Fine-Tuned Performance vs. Original Baseline}\n"
        "\\label{tab:orgaccess-hard}\n"
        "\\end{table}\n"
    )

    path = Path(output_file)

    if not path.exists():
        with open(path, "w") as f:
            f.write(f"% Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(header)
            f.write(latex_line + "\n")
            f.write(footer + "\n\n")
    else:
        with open(path, "r") as f:
            content = f.readlines()

        idx = next((i for i, line in enumerate(content) if "\\bottomrule" in line), len(content))
        content.insert(idx, latex_line + "\n")

        with open(path, "w") as f:
            f.writelines(content)

    print(f"✅ Added: {latex_line.split('&')[0].strip()}")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orgaccess_table_generator.py <results.json> [outputfile]")
        sys.exit(1)

    json_path = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "orgaccess_latex_table.tex"

    model, acc, f1 = extract_metrics(json_path)
    row = format_latex_row(model, acc, f1)
    append_to_table(row, out_file)
