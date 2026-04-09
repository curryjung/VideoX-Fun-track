#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


DEFAULT_FINAL_MP4 = "final_832x480_16fps_81f.mp4"
DEFAULT_DECODED_MP4 = "decoded_from_vae_832x480_16fps_81f.mp4"
DEFAULT_TEXT_NPZ = "text_feature_wan_t5.npz"
DEFAULT_LATENT_PT = "vae_latents.pt"
DEFAULT_FIRST_FRAME_VAE_LATENT_PT = "first_frame_vae_latent.pt"


def _to_rel_or_abs(path: str, root: str, use_relative: bool) -> str:
    if not use_relative:
        return os.path.abspath(path)
    return os.path.relpath(path, root)


def _read_caption(text_npz_path: str) -> str:
    if not os.path.isfile(text_npz_path):
        return ""
    try:
        data = np.load(text_npz_path, allow_pickle=True)
    except Exception:
        return ""
    if "caption" not in data:
        return ""
    caption = data["caption"]
    try:
        if isinstance(caption, np.ndarray):
            if caption.ndim == 0:
                return str(caption.item())
            if caption.size > 0:
                return str(caption.reshape(-1)[0])
        return str(caption)
    except Exception:
        return ""


def _read_track_points(track_npz_path: str) -> int:
    try:
        data = np.load(track_npz_path)
    except Exception:
        return -1
    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    elif "tracks" in data:
        tracks = data["tracks"]
    else:
        return -1
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if tracks.ndim != 3:
        return -1
    return int(tracks.shape[1])


def _is_recently_updated(paths: List[str], min_age_seconds: int) -> bool:
    if min_age_seconds <= 0:
        return False
    now = time.time()
    newest_mtime = None
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        newest_mtime = mtime if newest_mtime is None else max(newest_mtime, mtime)
    if newest_mtime is None:
        return False
    return (now - newest_mtime) < float(min_age_seconds)


def _discover_samples_walk(
    preprocess_root: str,
    primary_filename: str,
    track_prefix: str,
    show_progress: bool,
    progress_every: int,
) -> List[Tuple[str, str, str, str, str]]:
    samples: List[Tuple[str, str, str, str, str]] = []
    discovered_dirs = 0
    iterator = os.walk(preprocess_root)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=f"scan:{os.path.basename(preprocess_root)}",
            unit="dir",
            leave=False,
        )
    for root, _, files in iterator:
        discovered_dirs += 1
        if (not show_progress) and progress_every > 0 and discovered_dirs % progress_every == 0:
            print(
                f"[scan] root={preprocess_root} visited_dirs={discovered_dirs} "
                f"matched_samples={len(samples)}"
            )
        if primary_filename not in files:
            continue
        track_candidates = [f for f in files if f.startswith(track_prefix) and f.endswith(".npz")]
        if len(track_candidates) == 0:
            continue

        primary_path = os.path.join(root, primary_filename)
        # Pick lexicographically first track file for determinism.
        track_path = os.path.join(root, sorted(track_candidates)[0])
        text_path = os.path.join(root, DEFAULT_TEXT_NPZ)

        # sample_id is parent directory of processed_* folder.
        # expected: <sample_id>/processed_832x480_fps16/<files>
        parent = os.path.dirname(root)
        sample_id = os.path.basename(parent) if parent else os.path.basename(root)
        samples.append((preprocess_root, sample_id, primary_path, track_path, text_path))
    print(
        f"[scan-done] root={preprocess_root} visited_dirs={discovered_dirs} "
        f"discovered_samples={len(samples)}"
    )
    return samples


def _discover_samples_fixed_layout(
    preprocess_root: str,
    processed_dir_name: str,
    primary_filename: str,
    track_prefix: str,
    show_progress: bool,
    progress_every: int,
) -> List[Tuple[str, str, str, str, str]]:
    # Fixed layout:
    # <preprocess_root>/<sample_id>/<processed_dir_name>/{primary_file, transformed_tracks_*.npz, text_feature_*.npz}
    samples: List[Tuple[str, str, str, str, str]] = []
    visited_sample_dirs = 0

    iterator = os.scandir(preprocess_root)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=f"scan-fixed:{os.path.basename(preprocess_root)}",
            unit="sample_dir",
            leave=False,
        )

    for entry in iterator:
        if not entry.is_dir(follow_symlinks=False):
            continue

        visited_sample_dirs += 1
        if (not show_progress) and progress_every > 0 and visited_sample_dirs % progress_every == 0:
            print(
                f"[scan-fixed] root={preprocess_root} visited_sample_dirs={visited_sample_dirs} "
                f"matched_samples={len(samples)}"
            )

        processed_dir = os.path.join(entry.path, processed_dir_name)
        if not os.path.isdir(processed_dir):
            continue

        primary_path = os.path.join(processed_dir, primary_filename)
        if not os.path.isfile(primary_path):
            continue

        track_candidates: List[str] = []
        for f in os.scandir(processed_dir):
            if not f.is_file(follow_symlinks=False):
                continue
            name = f.name
            if name.startswith(track_prefix) and name.endswith(".npz"):
                track_candidates.append(name)
        if len(track_candidates) == 0:
            continue

        track_path = os.path.join(processed_dir, sorted(track_candidates)[0])
        text_path = os.path.join(processed_dir, DEFAULT_TEXT_NPZ)
        sample_id = entry.name
        samples.append((preprocess_root, sample_id, primary_path, track_path, text_path))

    print(
        f"[scan-fixed-done] root={preprocess_root} visited_sample_dirs={visited_sample_dirs} "
        f"discovered_samples={len(samples)}"
    )
    return samples


def _load_existing_records(output_meta: str) -> List[Dict[str, str]]:
    if not os.path.isfile(output_meta):
        return []
    with open(output_meta, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Existing metadata must be a list: {output_meta}")
    return data


def _record_key(record: Dict[str, str], data_root: str) -> str:
    track_path = record.get("track_file_path", "")
    if track_path == "":
        file_path = record.get("file_path", "")
        if file_path == "":
            return ""
        if os.path.isabs(file_path):
            return os.path.abspath(file_path)
        return os.path.abspath(os.path.join(data_root, file_path))
    if os.path.isabs(track_path):
        return os.path.abspath(track_path)
    return os.path.abspath(os.path.join(data_root, track_path))


def _resolve_preprocess_roots(args: argparse.Namespace) -> List[str]:
    roots: List[str] = []
    for root in args.preprocess_root:
        roots.append(os.path.abspath(root))
    if args.preprocess_root_list is not None:
        with open(args.preprocess_root_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                roots.append(os.path.abspath(line))
    deduped = sorted(set(roots))
    if len(deduped) == 0:
        raise ValueError("No preprocess roots provided.")
    return deduped


def _parse_root_aliases(args: argparse.Namespace) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if args.root_alias is None:
        return alias_map
    for item in args.root_alias:
        if "=" not in item:
            raise ValueError(f"Invalid --root_alias '{item}'. Expected /abs/root=alias")
        root_path, alias = item.split("=", 1)
        root_abs = os.path.abspath(root_path.strip())
        alias = alias.strip()
        if root_path.strip() == "" or alias == "":
            raise ValueError(f"Invalid --root_alias '{item}'. Root and alias must be non-empty.")
        alias_map[root_abs] = alias
    return alias_map


def _default_root_id(root_abs: str) -> str:
    base = os.path.basename(root_abs.rstrip("/"))
    if base == "":
        return "root"
    return base


def build_metadata(args: argparse.Namespace) -> List[Dict[str, str]]:
    preprocess_roots = _resolve_preprocess_roots(args)
    for root in preprocess_roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"preprocess_root not found: {root}")

    if args.data_root is not None:
        data_root = os.path.abspath(args.data_root)
    else:
        data_root = os.path.commonpath(preprocess_roots)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    if args.sample_media == "video":
        primary_filename = args.video_filename
    else:
        primary_filename = args.latent_filename

    discovered: List[Tuple[str, str, str, str, str]] = []
    root_aliases = _parse_root_aliases(args)
    used_root_map: Dict[str, str] = {}
    for root in preprocess_roots:
        print(f"[scan-start] root={root}")
        if args.discovery_mode == "fixed":
            discovered.extend(
                _discover_samples_fixed_layout(
                    preprocess_root=root,
                    processed_dir_name=args.processed_dir_name,
                    primary_filename=primary_filename,
                    track_prefix=args.track_prefix,
                    show_progress=not args.no_progress_bar,
                    progress_every=args.progress_every,
                )
            )
        else:
            discovered.extend(
                _discover_samples_walk(
                    preprocess_root=root,
                    primary_filename=primary_filename,
                    track_prefix=args.track_prefix,
                    show_progress=not args.no_progress_bar,
                    progress_every=args.progress_every,
                )
            )
    if len(discovered) == 0:
        raise RuntimeError(
            "No samples found under given preprocess roots with "
            f"sample_media='{args.sample_media}', primary_filename='{primary_filename}', "
            f"track_prefix='{args.track_prefix}'."
        )

    new_record_map: Dict[str, Dict[str, str]] = {}
    skipped_no_text = 0
    skipped_low_points = 0
    duplicate_collisions = 0
    skipped_incomplete = 0
    skipped_recent = 0
    discovered_iter = discovered
    if not args.no_progress_bar:
        discovered_iter = tqdm(discovered, desc="filter+build", unit="sample")
    for idx, (source_root, sample_id, primary_path, track_path, text_path) in enumerate(discovered_iter, start=1):
        if args.skip_incomplete:
            if not os.path.isfile(primary_path) or not os.path.isfile(track_path):
                skipped_incomplete += 1
                continue
            if args.require_text and (not os.path.isfile(text_path)):
                skipped_incomplete += 1
                continue

        check_paths = [primary_path, track_path]
        if os.path.isfile(text_path):
            check_paths.append(text_path)
        if _is_recently_updated(check_paths, min_age_seconds=args.skip_recent_seconds):
            skipped_recent += 1
            continue

        key_abs = os.path.abspath(track_path)
        if key_abs in new_record_map:
            duplicate_collisions += 1
            continue

        if args.min_track_points > 1:
            n_points = _read_track_points(track_path)
            if n_points < args.min_track_points:
                skipped_low_points += 1
                continue

        caption = _read_caption(text_path)
        if args.require_text and caption == "":
            skipped_no_text += 1
            continue
        if caption == "":
            caption = sample_id

        record = {
            "file_path": _to_rel_or_abs(primary_path, data_root, args.relative_paths),
            "text": caption,
            "type": "video",
            "track_file_path": _to_rel_or_abs(track_path, data_root, args.relative_paths),
        }
        if args.include_root_id:
            root_id = root_aliases.get(source_root, _default_root_id(source_root))
            record[args.root_id_key] = root_id
            used_root_map[root_id] = source_root
        if args.sample_media == "latent":
            record["latent_file_path"] = record["file_path"]
            record["source_media"] = "latent"
            first_frame_vae_latent_path = os.path.join(
                os.path.dirname(primary_path),
                args.first_frame_vae_latent_filename,
            )
            if os.path.isfile(first_frame_vae_latent_path):
                record["first_frame_vae_latent_file_path"] = _to_rel_or_abs(
                    first_frame_vae_latent_path,
                    data_root,
                    args.relative_paths,
                )
        new_record_map[key_abs] = record
        if args.no_progress_bar and args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"[build] processed={idx}/{len(discovered)} unique_new={len(new_record_map)} "
                f"skipped_low_points={skipped_low_points} skipped_no_text={skipped_no_text} "
                f"duplicate_collisions={duplicate_collisions} skipped_incomplete={skipped_incomplete} "
                f"skipped_recent={skipped_recent}"
            )

    if args.update_existing:
        existing_records = _load_existing_records(args.output_meta)
        merged_map: Dict[str, Dict[str, str]] = {}
        for record in existing_records:
            key = _record_key(record, data_root)
            if key == "":
                continue
            merged_map[key] = record

        merged_map.update(new_record_map)

        if args.prune_missing:
            discovered_keys: Set[str] = {os.path.abspath(item[2]) for item in discovered}
            merged_map = {k: v for k, v in merged_map.items() if k in discovered_keys}
        record_map = merged_map
    else:
        record_map = new_record_map

    records = sorted(record_map.values(), key=lambda x: x.get("file_path", ""))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_meta)), exist_ok=True)
    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {len(records)} samples -> {args.output_meta}")
    print(f"[info] preprocess_roots={len(preprocess_roots)}")
    for idx, root in enumerate(preprocess_roots):
        print(f"[info] root[{idx}]={root}")
    print(f"[info] data_root={data_root}")
    print(f"[info] discovered={len(discovered)}")
    print(f"[info] unique_new={len(new_record_map)}")
    print(f"[info] skipped_low_points={skipped_low_points}")
    print(f"[info] skipped_incomplete={skipped_incomplete}")
    print(f"[info] skipped_recent={skipped_recent} (min_age={args.skip_recent_seconds}s)")
    print(f"[info] duplicate_collisions={duplicate_collisions}")
    if args.require_text:
        print(f"[info] skipped_no_text={skipped_no_text}")
    if args.update_existing:
        print(f"[info] update_existing=True prune_missing={args.prune_missing}")
    if args.include_root_id:
        print(f"[info] include_root_id=True root_id_key={args.root_id_key}")
        print(f"[info] unique_root_ids={len(used_root_map)}")
        for rid, root in sorted(used_root_map.items(), key=lambda x: x[0]):
            print(f"[info] root_map[{rid}]={root}")
    if args.output_root_map_json is not None:
        if not args.include_root_id:
            raise ValueError("--output_root_map_json requires --include_root_id.")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_root_map_json)), exist_ok=True)
        with open(args.output_root_map_json, "w", encoding="utf-8") as f:
            json.dump(used_root_map, f, ensure_ascii=False, indent=2)
        print(f"[done] wrote root map -> {args.output_root_map_json}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata_track.json from OpenVid+CoTracker preprocess outputs."
    )
    parser.add_argument(
        "--preprocess_root",
        type=str,
        action="append",
        required=True,
        help="Repeatable preprocess root. Use multiple times for multi-server outputs.",
    )
    parser.add_argument(
        "--preprocess_root_list",
        type=str,
        default=None,
        help="Optional txt file with one preprocess root path per line.",
    )
    parser.add_argument("--output_meta", type=str, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root path used by train_data_dir. Defaults to preprocess_root.",
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default=DEFAULT_FINAL_MP4,
        choices=[DEFAULT_FINAL_MP4, DEFAULT_DECODED_MP4],
    )
    parser.add_argument(
        "--sample_media",
        type=str,
        choices=["video", "latent"],
        default="latent",
        help="Primary training media path written to file_path. Use latent for vae_latents.pt-based metadata.",
    )
    parser.add_argument(
        "--latent_filename",
        type=str,
        default=DEFAULT_LATENT_PT,
        help="Primary filename used when --sample_media=latent.",
    )
    parser.add_argument(
        "--first_frame_vae_latent_filename",
        type=str,
        default=DEFAULT_FIRST_FRAME_VAE_LATENT_PT,
        help=(
            "Optional precomputed first-frame VAE latent filename under each processed dir. "
            "If found, metadata includes first_frame_vae_latent_file_path."
        ),
    )
    parser.add_argument(
        "--track_prefix",
        type=str,
        default="transformed_tracks_grid",
        help="Track npz filename prefix under processed_832x480_fps16.",
    )
    parser.add_argument(
        "--discovery_mode",
        type=str,
        choices=["fixed", "walk"],
        default="fixed",
        help="Discovery mode. `fixed` is faster for known layout; `walk` is recursive fallback.",
    )
    parser.add_argument(
        "--processed_dir_name",
        type=str,
        default="processed_832x480_fps16",
        help="Only used when --discovery_mode=fixed.",
    )
    parser.add_argument("--min_track_points", type=int, default=1)
    parser.add_argument("--require_text", action="store_true")
    parser.add_argument(
        "--skip_incomplete",
        action="store_true",
        default=True,
        help="Skip samples missing required files (video/track and text if --require_text). Default: enabled.",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Allow incomplete samples (overrides --skip_incomplete).",
    )
    parser.add_argument(
        "--skip_recent_seconds",
        type=int,
        default=120,
        help="Skip samples with files updated within this many seconds (to avoid in-progress writes).",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable tqdm progress bars and print periodic text logs only.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5000,
        help="When --no_progress_bar is enabled, print progress every N items/dirs.",
    )
    parser.add_argument(
        "--update_existing",
        action="store_true",
        help="Merge with existing output_meta and upsert by track_file_path.",
    )
    parser.add_argument(
        "--prune_missing",
        action="store_true",
        help="With --update_existing, remove records not discovered in current scan.",
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        default=True,
        help="Store file paths relative to data_root (recommended).",
    )
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Store absolute file paths instead of relative paths.",
    )
    parser.add_argument(
        "--include_root_id",
        action="store_true",
        help="Include root id field in each metadata record for multi-root training.",
    )
    parser.add_argument(
        "--root_id_key",
        type=str,
        default="root_id",
        help="Record key used for root id when --include_root_id is enabled.",
    )
    parser.add_argument(
        "--root_alias",
        action="append",
        default=None,
        help="Optional alias mapping in the form /abs/root=alias. Repeatable.",
    )
    parser.add_argument(
        "--output_root_map_json",
        type=str,
        default=None,
        help="Optional output path to write root_id->abs_root JSON map.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.absolute_paths:
        args.relative_paths = False
    if args.allow_incomplete:
        args.skip_incomplete = False
    build_metadata(args)
