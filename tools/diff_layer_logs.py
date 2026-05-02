import re
from pathlib import Path


PATTERN = re.compile(
    r"\[moss_layer\] call=(\d+) layer=([-\d]+) stage=([a-zA-Z_]+) "
    r"sum=([-\deE\.\+]+) l2=([-\deE\.\+]+) min=([-\deE\.\+]+) max=([-\deE\.\+]+)"
)


def parse(path: Path):
    m = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        mm = PATTERN.search(line)
        if not mm:
            continue
        k = (int(mm.group(1)), int(mm.group(2)), mm.group(3))
        m[k] = tuple(float(mm.group(i)) for i in range(4, 8))
    return m


def main():
    avx = parse(Path("/tmp/moss_layer_avx2.log"))
    generic = parse(Path("/tmp/moss_layer_generic.log"))
    keys = sorted(set(avx) & set(generic))
    print("key\td_sum\td_l2\td_min\td_max")
    worst = []
    for k in keys:
        a = avx[k]
        g = generic[k]
        d = tuple(abs(a[i] - g[i]) for i in range(4))
        print(f"{k}\t{d[0]:.6e}\t{d[1]:.6e}\t{d[2]:.6e}\t{d[3]:.6e}")
        worst.append((max(d), k, d, a, g))
    worst.sort(reverse=True)
    print("\nTOP_DIFF")
    for _, k, d, a, g in worst[:10]:
        print(f"{k} dmax={max(d):.6e} avx={a} generic={g}")


if __name__ == "__main__":
    main()
