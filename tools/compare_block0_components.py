import re
from pathlib import Path


CPP_PATTERN = re.compile(
    r"\[moss_layer\] call=1 layer=0 stage=(comp_[a-z0-9_]+) "
    r"sum=([-\deE\.\+]+) l2=([-\deE\.\+]+) min=([-\deE\.\+]+) max=([-\deE\.\+]+)"
)
PY_PATTERN = re.compile(
    r"\[py_comp\] (comp_[a-z0-9_]+) "
    r"sum=([-\deE\.\+]+) l2=([-\deE\.\+]+) min=([-\deE\.\+]+) max=([-\deE\.\+]+)"
)


def parse_cpp(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = CPP_PATTERN.search(line)
        if not m:
            continue
        out[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    return out


def parse_py(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PY_PATTERN.search(line)
        if not m:
            continue
        out[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    return out


def main():
    cpp = parse_cpp(Path("/tmp/moss_cpp_comp.log"))
    py = parse_py(Path("/tmp/moss_py_comp.log"))
    keys = sorted(set(cpp) | set(py))
    print("stage\tpresent_cpp\tpresent_py\td_sum\td_l2\td_min\td_max")
    ranking = []
    for k in keys:
        c = cpp.get(k)
        p = py.get(k)
        if c is None or p is None:
            print(f"{k}\t{c is not None}\t{p is not None}\tNA\tNA\tNA\tNA")
            continue
        d = tuple(abs(c[i] - p[i]) for i in range(4))
        print(f"{k}\tTrue\tTrue\t{d[0]:.6e}\t{d[1]:.6e}\t{d[2]:.6e}\t{d[3]:.6e}")
        ranking.append((max(d), k, d, c, p))

    ranking.sort(reverse=True)
    print("\nTOP_MISMATCH")
    for _, k, d, c, p in ranking[:10]:
        print(f"{k} dmax={max(d):.6e} cpp={c} py={p}")


if __name__ == "__main__":
    main()
