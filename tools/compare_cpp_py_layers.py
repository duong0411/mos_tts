import re
from pathlib import Path


CPP_PAT = re.compile(
    r"\[moss_layer\] call=(\d+) layer=([-\d]+) stage=([a-zA-Z_]+) "
    r"sum=([-\deE\.\+]+) l2=([-\deE\.\+]+) min=([-\deE\.\+]+) max=([-\deE\.\+]+)"
)
PY_PAT = re.compile(
    r"\[py_layer\] stack=([a-zA-Z_]+) call=(\d+) layer=([-\d]+) stage=([a-zA-Z_]+) "
    r"sum=([-\deE\.\+]+) l2=([-\deE\.\+]+) min=([-\deE\.\+]+) max=([-\deE\.\+]+)"
)


def parse_cpp(path: Path):
    m = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        mm = CPP_PAT.search(line)
        if not mm:
            continue
        call = int(mm.group(1))
        layer = int(mm.group(2))
        stage = mm.group(3)
        stats = tuple(float(mm.group(i)) for i in range(4, 8))
        m[(call, layer, stage)] = stats
    return m


def parse_py(path: Path):
    m = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        mm = PY_PAT.search(line)
        if not mm:
            continue
        stack = mm.group(1)
        call = int(mm.group(2))
        layer = int(mm.group(3))
        stage = mm.group(4)
        stats = tuple(float(mm.group(i)) for i in range(5, 9))
        # map to cpp call numbering: global call=1, local call=2
        cpp_call = 1 if stack == "global" else 2
        m[(cpp_call, layer, stage)] = stats
    return m


def main():
    cpp = parse_cpp(Path("/tmp/moss_cpp_layer.log"))
    py = parse_py(Path("/tmp/moss_py_layer.log"))
    keys = sorted(set(cpp) & set(py))
    print("key\td_sum\td_l2\td_min\td_max")
    worst = []
    for k in keys:
        c = cpp[k]
        p = py[k]
        d = tuple(abs(c[i] - p[i]) for i in range(4))
        print(f"{k}\t{d[0]:.6e}\t{d[1]:.6e}\t{d[2]:.6e}\t{d[3]:.6e}")
        worst.append((max(d), k, d, c, p))
    worst.sort(reverse=True)
    print("\nTOP_DIFF")
    for _, k, d, c, p in worst[:12]:
        print(f"{k} dmax={max(d):.6e} cpp={c} py={p}")


if __name__ == "__main__":
    main()
