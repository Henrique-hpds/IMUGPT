"""Single-person track selection for stage 5.2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pose_module.interfaces import PoseFrameInstance

TRACK_MAX_GAP_FRAMES = 5
TRACK_MIN_IOU = 0.1
TRACK_MAX_CENTER_DISTANCE = 1.75


@dataclass
class TrackState:
    track_id: int
    detections: Dict[int, PoseFrameInstance] = field(default_factory=dict)
    first_frame_id: Optional[int] = None
    last_frame_id: Optional[int] = None
    gap_total: int = 0
    transition_count: int = 0
    iou_history: List[float] = field(default_factory=list)
    center_distance_history: List[float] = field(default_factory=list)
    bbox_scores: List[float] = field(default_factory=list)

    @property
    def detection_count(self) -> int:
        return int(len(self.detections))

    @property
    def span_frames(self) -> int:
        if self.first_frame_id is None or self.last_frame_id is None:
            return 0
        return int(self.last_frame_id - self.first_frame_id + 1)

    @property
    def continuity_ratio(self) -> float:
        span = self.span_frames
        if span <= 0:
            return 0.0
        return float(self.detection_count) / float(span)

    @property
    def mean_bbox_score(self) -> float:
        if len(self.bbox_scores) == 0:
            return 0.0
        return float(np.mean(np.asarray(self.bbox_scores, dtype=np.float64)))

    @property
    def mean_iou(self) -> float:
        if len(self.iou_history) == 0:
            return 0.0
        return float(np.mean(np.asarray(self.iou_history, dtype=np.float64)))

    def add_detection(
        self,
        detection: PoseFrameInstance,
        *,
        iou: Optional[float] = None,
        center_distance: Optional[float] = None,
    ) -> None:
        if self.last_frame_id is not None:
            gap = int(detection.frame_id - self.last_frame_id - 1)
            if gap > 0:
                self.gap_total += gap
            self.transition_count += 1
        if iou is not None:
            self.iou_history.append(float(iou))
        if center_distance is not None:
            self.center_distance_history.append(float(center_distance))
        self.detections[int(detection.frame_id)] = detection
        if self.first_frame_id is None:
            self.first_frame_id = int(detection.frame_id)
        self.last_frame_id = int(detection.frame_id)
        self.bbox_scores.append(float(detection.bbox_score))


def link_person_tracks(
    frame_predictions: Sequence[Mapping[str, Any]],
    *,
    max_gap_frames: int = TRACK_MAX_GAP_FRAMES,
    min_iou: float = TRACK_MIN_IOU,
    max_center_distance: float = TRACK_MAX_CENTER_DISTANCE,
) -> List[TrackState]:
    tracks: List[TrackState] = []
    next_track_id = 0

    for frame_payload in frame_predictions:
        frame_id = int(frame_payload["frame_id"])
        detections = list(frame_payload.get("instances", []))
        unmatched_detection_indices = set(range(len(detections)))
        active_tracks = [
            track
            for track in tracks
            if track.last_frame_id is not None
            and 0 < frame_id - track.last_frame_id <= int(max_gap_frames) + 1
        ]

        candidate_matches = []
        for track in active_tracks:
            previous_bbox = track.detections[int(track.last_frame_id)].bbox_xyxy
            for detection_index, detection in enumerate(detections):
                iou = _bbox_iou(previous_bbox, detection.bbox_xyxy)
                center_distance = _bbox_center_distance_ratio(previous_bbox, detection.bbox_xyxy)
                if iou < float(min_iou) and center_distance > float(max_center_distance):
                    continue
                frame_gap = int(frame_id - int(track.last_frame_id) - 1)
                score = float(iou) - (0.05 * float(center_distance)) - (0.01 * float(frame_gap))
                candidate_matches.append(
                    (
                        score,
                        float(iou),
                        float(center_distance),
                        int(track.track_id),
                        int(detection_index),
                    )
                )

        matched_tracks = set()
        candidate_matches.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        for _, iou, center_distance, track_id, detection_index in candidate_matches:
            if track_id in matched_tracks or detection_index not in unmatched_detection_indices:
                continue
            track = next(track for track in tracks if track.track_id == track_id)
            detection = detections[detection_index]
            track.add_detection(detection, iou=iou, center_distance=center_distance)
            matched_tracks.add(track_id)
            unmatched_detection_indices.remove(detection_index)

        for detection_index in sorted(unmatched_detection_indices):
            detection = detections[detection_index]
            track = TrackState(track_id=next_track_id)
            track.add_detection(detection)
            tracks.append(track)
            next_track_id += 1

    return sorted(tracks, key=_track_sort_key, reverse=True)


def build_person_track_report(
    tracks: Sequence[TrackState],
    *,
    selected_track: Optional[TrackState],
    total_frames: int,
) -> Dict[str, Any]:
    warnings: List[str] = []
    if selected_track is None:
        warnings.append("no_valid_track_found")
    elif selected_track.gap_total > 0:
        warnings.append("selected_track_has_gaps")

    if len(tracks) > 1 and selected_track is not None:
        runner_up = tracks[1]
        if runner_up.detection_count >= max(5, int(math.floor(0.75 * selected_track.detection_count))):
            warnings.append("competing_secondary_track_detected")

    overview = []
    for track in tracks[:10]:
        overview.append(
            {
                "track_id": int(track.track_id),
                "detection_count": int(track.detection_count),
                "first_frame_id": None if track.first_frame_id is None else int(track.first_frame_id),
                "last_frame_id": None if track.last_frame_id is None else int(track.last_frame_id),
                "span_frames": int(track.span_frames),
                "gap_total": int(track.gap_total),
                "continuity_ratio": float(track.continuity_ratio),
                "mean_bbox_score": float(track.mean_bbox_score),
                "mean_iou": float(track.mean_iou),
            }
        )

    selected_track_stability = 0.0
    if selected_track is not None and total_frames > 0:
        coverage = float(selected_track.detection_count) / float(total_frames)
        selected_track_stability = (0.6 * float(selected_track.continuity_ratio)) + (0.4 * coverage)

    return {
        "status": "ok" if selected_track is not None else "fail",
        "selected_track_id": None if selected_track is None else int(selected_track.track_id),
        "selected_track_stability": float(selected_track_stability),
        "frames_total": int(total_frames),
        "num_tracks": int(len(tracks)),
        "frames_with_detections": int(len({frame_id for track in tracks for frame_id in track.detections.keys()})),
        "track_overview": overview,
        "warnings": warnings,
    }


def _bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    if not np.isfinite(bbox_a[:4]).all() or not np.isfinite(bbox_b[:4]).all():
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in bbox_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in bbox_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _bbox_center_distance_ratio(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    if not np.isfinite(bbox_a[:4]).all() or not np.isfinite(bbox_b[:4]).all():
        return float("inf")
    a_center = np.asarray(
        [
            (float(bbox_a[0]) + float(bbox_a[2])) * 0.5,
            (float(bbox_a[1]) + float(bbox_a[3])) * 0.5,
        ],
        dtype=np.float32,
    )
    b_center = np.asarray(
        [
            (float(bbox_b[0]) + float(bbox_b[2])) * 0.5,
            (float(bbox_b[1]) + float(bbox_b[3])) * 0.5,
        ],
        dtype=np.float32,
    )
    distance = float(np.linalg.norm(a_center - b_center))
    a_diag = math.hypot(float(bbox_a[2] - bbox_a[0]), float(bbox_a[3] - bbox_a[1]))
    b_diag = math.hypot(float(bbox_b[2] - bbox_b[0]), float(bbox_b[3] - bbox_b[1]))
    scale = max(1.0, a_diag, b_diag)
    return float(distance / scale)


def _track_sort_key(track: TrackState) -> Tuple[float, float, float]:
    return (
        float(track.detection_count),
        float(track.continuity_ratio),
        float(track.mean_bbox_score),
    )
