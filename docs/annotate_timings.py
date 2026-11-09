"""Stamp each code cell of executed notebooks with its wall time and record per-notebook totals.

    python docs/annotate_timings.py docs/tutorials/*.ipynb

nbclient stores execution timestamps in ``cell.metadata.execution``; this script appends a
``[wall time: ...]`` line to every executed code cell's output (replacing a previous stamp) and
updates the table between the ``<!-- timings:start -->`` / ``<!-- timings:end -->`` markers in the
``README.md`` next to the notebooks.  Notebooks not passed on the command line keep their old row.
"""
import datetime as dt
import json
import os
import re
import sys

STAMP = "[wall time:"


def cell_seconds(cell):
    ex = cell.get("metadata", {}).get("execution", {})
    try:
        t0 = dt.datetime.fromisoformat(ex["iopub.execute_input"].replace("Z", "+00:00"))
        t1 = dt.datetime.fromisoformat(ex["shell.execute_reply"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    return (t1 - t0).total_seconds()


def fmt(seconds):
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def annotate(path):
    """Add the wall-time stamps in place; return (total seconds, device string or None)."""
    with open(path) as f:
        nb = json.load(f)
    total, device = 0.0, None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        seconds = cell_seconds(cell)
        if seconds is None:
            continue
        total += seconds
        outputs = [o for o in cell.get("outputs", [])
                   if not (o.get("output_type") == "stream" and "".join(o.get("text", "")).startswith(STAMP))]
        outputs.append({"output_type": "stream", "name": "stdout", "text": [f"{STAMP} {fmt(seconds)}]\n"]})
        cell["outputs"] = outputs
        for o in outputs:                                   # the setup cell prints "torch <ver> on <device> (<dtype>)"
            m = re.search(r"torch \S+ on (\w+) \((\S+)\)", "".join(o.get("text", "")))
            if m:
                device = f"{m.group(1)} / {m.group(2)}"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return total, device


def update_readme(readme_path, totals, device):
    with open(readme_path) as f:
        readme = f.read()
    rows = dict(re.findall(r"^\| (\d\d_\S+\.ipynb) \| (.+?) \|$", readme, flags=re.M))
    rows.update({nb: fmt(t) for nb, t in totals.items()})
    table = "\n".join(f"| {nb} | {t} |" for nb, t in sorted(rows.items()))
    block = (f"<!-- timings:start -->\nLast executed on **{device or 'unknown device'}** "
             f"({dt.date.today().isoformat()}); every code cell shows its own wall time at the end of its output.\n\n"
             f"| notebook | total wall time |\n|---|---|\n{table}\n<!-- timings:end -->")
    if "<!-- timings:start -->" in readme:
        readme = re.sub(r"<!-- timings:start -->.*?<!-- timings:end -->", lambda _: block, readme, flags=re.S)
    else:
        readme = readme.rstrip("\n") + "\n\n## Run times\n\n" + block + "\n"
    with open(readme_path, "w") as f:
        f.write(readme)


def main(paths):
    totals, device = {}, None
    for path in paths:
        total, dev = annotate(path)
        totals[os.path.basename(path)] = total
        device = dev or device
        print(f"{os.path.basename(path)}: {fmt(total)}")
    if paths:
        update_readme(os.path.join(os.path.dirname(os.path.abspath(paths[0])), "README.md"), totals, device)


if __name__ == "__main__":
    main(sys.argv[1:])
