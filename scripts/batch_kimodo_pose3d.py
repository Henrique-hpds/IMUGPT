"""Batch-process all Kimodo NPZ files through run_pose3d_from_kimodo.

Reads kimodo_generation_manifest.jsonl, runs metric_normalizer + root_estimator
on each entry, and writes a new manifest with pose3d_npz_path artifacts.

Usage:
    python scripts/batch_kimodo_pose3d.py \
        --manifest output/robot_emotions_kimodo_generated_hands_all/kimodo_generation_manifest.jsonl \
        --output-dir output/robot_emotions_kimodo_pose3d
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_module.pipeline import run_pose3d_from_kimodo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok_entries = [e for e in entries if e.get("status") == "ok"]
    print(f"Total entries: {len(entries)} | ok: {len(ok_entries)}", flush=True)

    result_entries: list[dict] = []
    num_ok = num_skip = num_fail = 0
    t_start = time.time()

    for i, entry in enumerate(ok_entries):
        prompt_id = entry["prompt_id"]
        kimodo_npz = entry["artifacts"].get("kimodo_npz_path", "")
        gen_cfg = entry["artifacts"].get("generation_config_json_path", "")
        clip_id = entry.get("sample_id") or prompt_id
        pose3d_dir = output_root / prompt_id

        if args.skip_existing and (pose3d_dir / "pose3d.npz").exists():
            result_entries.append({**entry, "pose3d_artifacts": {"pose3d_npz_path": str((pose3d_dir / "pose3d.npz").resolve())}})
            num_skip += 1
            continue

        try:
            result = run_pose3d_from_kimodo(
                clip_id=clip_id,
                kimodo_npz_path=kimodo_npz,
                output_dir=pose3d_dir,
                generation_config_path=gen_cfg or None,
                export_bvh=False,
            )
            result_entries.append({**entry, "pose3d_artifacts": result["artifacts"]})
            num_ok += 1
        except Exception as exc:
            result_entries.append({**entry, "pose3d_artifacts": {}, "pose3d_error": f"{type(exc).__name__}: {exc}"})
            num_fail += 1
            print(f"  FAIL [{i+1}] {prompt_id}: {exc}", flush=True)
            (pose3d_dir / "error_trace.txt").parent.mkdir(parents=True, exist_ok=True)
            (pose3d_dir / "error_trace.txt").write_text(traceback.format_exc())

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(ok_entries) - i - 1) / rate
            print(
                f"  [{i+1}/{len(ok_entries)}] ok={num_ok} skip={num_skip} fail={num_fail} "
                f"| {rate:.1f} it/s | ~{remaining/60:.1f} min left",
                flush=True,
            )

    out_manifest = output_root / "kimodo_pose3d_manifest.jsonl"
    out_manifest.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in result_entries) + "\n",
        encoding="utf-8",
    )

    summary = {
        "manifest_path": str(out_manifest.resolve()),
        "num_total": len(ok_entries),
        "num_ok": num_ok,
        "num_skip": num_skip,
        "num_fail": num_fail,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (output_root / "kimodo_pose3d_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {summary}", flush=True)


if __name__ == "__main__":
    main()
