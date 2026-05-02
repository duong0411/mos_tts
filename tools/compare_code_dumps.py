import argparse
from pathlib import Path


def load_dump(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise RuntimeError(f"Empty dump file: {path}")
    frames = int(lines[0].strip())
    rows = []
    for i, line in enumerate(lines[1:], start=1):
        toks = [int(x) for x in line.strip().split()]
        rows.append(toks)
    if frames != len(rows):
        raise RuntimeError(f"Frame count mismatch in {path}: header={frames} rows={len(rows)}")
    if rows and len(rows[0]) != 16:
        raise RuntimeError(f"Expected 16 channels in {path}, got {len(rows[0])}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare C++ and Python token dumps.")
    parser.add_argument("--cpp", required=True, help="C++ dump path.")
    parser.add_argument("--py", required=True, help="Python dump path.")
    args = parser.parse_args()

    cpp = load_dump(Path(args.cpp))
    py = load_dump(Path(args.py))
    n = min(len(cpp), len(py))

    first = None
    same = 0
    total = n * 16
    for t in range(n):
        for q in range(16):
            if cpp[t][q] == py[t][q]:
                same += 1
            elif first is None:
                first = (t, q, cpp[t][q], py[t][q])

    print(f"cpp_frames={len(cpp)} py_frames={len(py)} overlap_frames={n}")
    print(f"token_match_ratio_overlap={same}/{total} ({(same / total) if total else 0.0:.6f})")
    if first is None:
        print("first_mismatch=NONE (all overlap tokens equal)")
    else:
        t, q, c, p = first
        print(f"first_mismatch frame={t} channel={q} cpp={c} py={p}")
    if len(cpp) != len(py):
        print(f"frame_count_delta cpp_minus_py={len(cpp) - len(py)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
