import re, sys
def bad_lines(path):
    out, case = [], None
    for line in open(path):
        line = line.rstrip()
        if line.startswith("==="):
            case = line[4:]; continue
        if "secs" in line or "levels" in line or "freq" in line or "pi" in line.split(":")[0]:
            continue
        m = re.search(r"maxabsdiff=([0-9.e+-]+|nan)", line)
        if m:
            if m.group(1) == "nan" or float(m.group(1)) > 1e-8: out.append(f"{case} |{line.strip()[:110]}")
            continue
        m = re.match(r"\s+(\S+): R=(.*) PY=(.*)", line)
        if m and m.group(2).strip() != m.group(3).strip():
            out.append(f"{case} |{line.strip()[:110]}")
    return out
for f in sys.argv[1:]:
    b = bad_lines("before_" + f); a = bad_lines("after_" + f)
    print(f"{f}: mismatching lines before {len(b)}, after {len(a)}")
    for x in a: print("   still:", x)
    for x in sorted(set(b) - set(a)): print("   fixed:", x)
