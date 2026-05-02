import argparse
from pathlib import Path
import sys


def dump_codes(path: Path, ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        frames = int(ids.shape[0])
        n_q = int(ids.shape[1])
        f.write(f"{frames}\n")
        for t in range(frames):
            row = [str(int(ids[t, q])) for q in range(n_q)]
            f.write(" ".join(row))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run infer.py and dump audio_token_ids to text.")
    parser.add_argument("--repo", required=True, help="Absolute path to MOSS-TTS-Nano directory.")
    parser.add_argument("--out-codes", required=True, help="Output token dump file.")
    parser.add_argument("--out-wav", required=True, help="Output wav path.")
    parser.add_argument("--text", required=True, help="Synthesis text.")
    parser.add_argument("--prompt-audio-path", required=True, help="Prompt wav path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-frames", type=int, default=96)
    parser.add_argument("--do-sample", type=int, default=0, choices=[0, 1])
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import infer as infer_mod  # pylint: disable=import-error

    argv = [
        "--device", "cpu",
        "--seed", str(args.seed),
        "--max-new-frames", str(args.max_new_frames),
        "--do-sample", str(args.do_sample),
        "--text-temperature", str(args.text_temperature),
        "--audio-temperature", str(args.audio_temperature),
        "--audio-top-p", str(args.audio_top_p),
        "--audio-top-k", str(args.audio_top_k),
        "--audio-repetition-penalty", str(args.audio_repetition_penalty),
        "--prompt-audio-path", args.prompt_audio_path,
        "--text", args.text,
        "--output-audio-path", args.out_wav,
    ]
    result = infer_mod.main(argv)
    ids = result["audio_token_ids"].detach().cpu()
    dump_codes(Path(args.out_codes), ids)
    print(f"python_frames={ids.shape[0]} n_q={ids.shape[1]} dump={args.out_codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
