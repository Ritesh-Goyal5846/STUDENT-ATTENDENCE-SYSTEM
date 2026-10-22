"""
classroom_monitor.py
====================
Behavioral Monitoring & Smart Features for an Image-Based Class Attendance System.

Components
----------
ClassroomPipeline       : camera / video file / image-folder reader (OpenCV), resize + colour handling,
                          interval sampling (heavy inference runs only every N seconds).
PhoneDetectionEngine    : YOLOv8n (COCO) person + cell-phone detection and phone->person linking.
IdentityResolver        : maps person boxes to student IDs (face_recognition + position tracking fallback).
AttendanceTracker       : interval-based presence store and LEFT_EARLY logic.
EventLogger             : JSON-lines event log.
ClassroomMonitor        : orchestrator. `process_frame()` is the integration point for Streamlit / FastAPI.

Run:  python classroom_monitor.py --source 0                       (webcam)
      python classroom_monitor.py --source class.mp4 --faces-dir known_faces
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:  # optional: only needed for face-based student identification
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None

log = logging.getLogger("classroom_monitor")

PERSON_CLS, PHONE_CLS = 0, 67          # COCO class ids: 0 = person, 67 = cell phone
Box = Tuple[float, float, float, float]  # x1, y1, x2, y2
RED, GREEN, YELLOW, DARK = (0, 0, 255), (0, 170, 0), (0, 215, 255), (40, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class MonitorConfig:
    # --- detection ---
    model_path: str = "yolov8n.pt"
    device: Optional[str] = None            # None = auto, "cpu", "cuda:0", ...
    imgsz: int = 960                        # larger than 640 helps with small phones
    person_conf: float = 0.40
    phone_conf: float = 0.25                # phones are small -> lower threshold
    # --- pipeline ---
    resize_width: int = 960
    sample_every_sec: float = 5.0           # run inference once per N seconds (use 30-60 in real classes)
    # --- phone <-> student linking ---
    min_containment: float = 0.50           # >=50% of phone box inside person box
    max_edge_dist_ratio: float = 0.15       # ...or phone within 15% of person height from the box
    phone_confirm_samples: int = 1          # consecutive samples required (use 2-3 to cut false alarms)
    # --- identity ---
    faces_dir: Optional[str] = None         # folder with ST102.jpg, ST105.jpg, ...
    face_tolerance: float = 0.50
    max_match_dist_px: float = 90.0         # position-based re-identification radius
    # --- attendance ---
    interval_minutes: float = 15.0
    num_intervals: int = 3
    # --- output ---
    log_path: str = "events.jsonl"
    attendance_csv: str = "attendance.csv"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner`'s area that lies inside `outer` (asymmetric, unlike IoU)."""
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def center(b: Box) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def point_to_box_distance(pt: Tuple[float, float], b: Box) -> float:
    """Euclidean distance from a point to the nearest point of a box (0 if inside)."""
    dx = max(b[0] - pt[0], 0.0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0.0, pt[1] - b[3])
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
# 1. Camera / file pipeline
# --------------------------------------------------------------------------- #
class ClassroomPipeline:
    """
    Yields (frame_bgr, elapsed_seconds) sampled every `sample_every_sec`.

    source: webcam index ("0"), video path, RTSP/HTTP URL, or a folder of snapshots
            (each image = one sample, spaced `sample_every_sec` apart).
    Frames between samples are only `grab()`-ed (cheap), never decoded/processed.
    """

    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, source, resize_width: int = 960, sample_every_sec: float = 5.0):
        self.resize_width = resize_width
        self.sample_every_sec = sample_every_sec
        self.image_paths: Optional[List[Path]] = None
        self.cap: Optional[cv2.VideoCapture] = None

        src = str(source)
        if Path(src).is_dir():
            self.image_paths = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in self.IMG_EXT)
            if not self.image_paths:
                raise RuntimeError(f"No images found in {src}")
            self.is_live = False
            return

        self.source = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.is_live = isinstance(self.source, int) or src.startswith(("rtsp://", "http://", "https://"))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 and not math.isnan(fps) else 25.0

    # -- helpers --
    def resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.resize_width and w > self.resize_width:
            s = self.resize_width / w
            frame = cv2.resize(frame, (self.resize_width, int(h * s)), interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
        """OpenCV is BGR; face_recognition/dlib expects RGB. (YOLO via ultralytics takes BGR arrays.)"""
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # -- main generator --
    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        if self.image_paths is not None:
            for i, path in enumerate(self.image_paths):
                img = cv2.imread(str(path))
                if img is not None:
                    yield self.resize(img), i * self.sample_every_sec
            return

        start, next_t, idx = time.monotonic(), 0.0, 0
        while True:
            if not self.cap.grab():
                break
            elapsed = (time.monotonic() - start) if self.is_live else idx / self.fps
            idx += 1
            if elapsed + 1e-9 < next_t:
                continue                      # skip: no decode, no inference
            ok, frame = self.cap.retrieve()
            if not ok:
                continue
            next_t = elapsed + self.sample_every_sec
            yield self.resize(frame), elapsed

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# --------------------------------------------------------------------------- #
# 2. Phone detection engine
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    box: Box
    conf: float


@dataclass
class PhoneLink:
    person_idx: int
    phone: Detection
    containment: float
    edge_dist_ratio: float
    anchor_dist_ratio: float


class PhoneDetectionEngine:
    """YOLOv8 person + cell-phone detector with spatial phone->person association."""

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)      # downloads yolov8n.pt on first run

    def detect_batch(self, frames: List[np.ndarray]) -> List[Tuple[List[Detection], List[Detection]]]:
        """Batched inference (useful for snapshot folders). Returns [(persons, phones), ...]."""
        results = self.model.predict(
            frames, classes=[PERSON_CLS, PHONE_CLS],
            conf=min(self.cfg.person_conf, self.cfg.phone_conf),
            imgsz=self.cfg.imgsz, device=self.cfg.device, verbose=False,
        )
        out = []
        for r in results:
            persons, phones = [], []
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                det = Detection(tuple(float(v) for v in b), float(c))
                if k == PERSON_CLS and c >= self.cfg.person_conf:
                    persons.append(det)
                elif k == PHONE_CLS and c >= self.cfg.phone_conf:
                    phones.append(det)
            out.append((persons, phones))
        return out

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        return self.detect_batch([frame_bgr])[0]

    def link_phones_to_persons(self, persons: List[Detection], phones: List[Detection]) -> List[PhoneLink]:
        """
        For every phone pick the single most plausible owner.

        A person is a *valid candidate* if
              containment(phone in person) >= min_containment
           OR point-to-box distance / person_height <= max_edge_dist_ratio
        Among valid candidates, the owner is the one whose "hand anchor"
        (horizontal centre, 60% down the box) is closest to the phone centre,
        measured in units of person height (scale-invariant).
        """
        links: List[PhoneLink] = []
        for ph in phones:
            pc = center(ph.box)
            best: Optional[PhoneLink] = None
            for i, p in enumerate(persons):
                h = max(p.box[3] - p.box[1], 1.0)
                cont = containment(ph.box, p.box)
                edge = point_to_box_distance(pc, p.box) / h
                if cont < self.cfg.min_containment and edge > self.cfg.max_edge_dist_ratio:
                    continue
                anchor = ((p.box[0] + p.box[2]) / 2.0, p.box[1] + 0.6 * h)
                a_dist = math.dist(pc, anchor) / h
                if best is None or a_dist < best.anchor_dist_ratio:
                    best = PhoneLink(i, ph, cont, edge, a_dist)
            if best:
                links.append(best)
        return links


# --------------------------------------------------------------------------- #
# Identity resolution
# --------------------------------------------------------------------------- #
class IdentityResolver:
    """
    Assigns a stable ID to every person box.
      1. Face match against enrolled students (face_recognition), if available.
      2. Otherwise inherit the ID of the nearest previously-seen position (students are seated,
         so position is a strong cue; this is a centroid tracker seeded by face matches).
      3. Otherwise create a new anonymous ID (TRK_001, ...).
    Enrolment: put one photo per student in `faces_dir`, named <STUDENT_ID>.jpg (e.g. ST102.jpg).
    """

    def __init__(self, faces_dir: Optional[str], tolerance: float, max_match_dist_px: float):
        self.tolerance = tolerance
        self.max_dist = max_match_dist_px
        self.known_ids: List[str] = []
        self.known_enc: List[np.ndarray] = []
        self.last_pos: Dict[str, Tuple[float, float]] = {}
        self._next_trk = 1
        if faces_dir:
            self._load_faces(faces_dir)

    @property
    def roster(self) -> List[str]:
        return sorted(set(self.known_ids))

    def _load_faces(self, faces_dir: str):
        if face_recognition is None:
            log.warning("face_recognition not installed -> falling back to anonymous tracking only.")
            return
        for p in sorted(Path(faces_dir).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            enc = face_recognition.face_encodings(face_recognition.load_image_file(str(p)))
            if not enc:
                log.warning("No face found in %s", p.name)
                continue
            self.known_ids.append(p.stem.split("_")[0])   # ST102_2.jpg -> ST102
            self.known_enc.append(enc[0])
        log.info("Enrolled %d face(s) for %d student(s)", len(self.known_ids), len(self.roster))

    def _match_face(self, rgb: Optional[np.ndarray], box: Box) -> Optional[str]:
        if rgb is None or not self.known_enc:
            return None
        x1, y1, x2, y2 = (max(int(v), 0) for v in box)
        crop = rgb[y1:y1 + int((y2 - y1) * 0.5), x1:x2]     # head/shoulders = top half of the box
        if crop.size == 0:
            return None
        crop = np.ascontiguousarray(crop)
        locs = face_recognition.face_locations(crop, number_of_times_to_upsample=2, model="hog")
        if not locs:
            return None
        encs = face_recognition.face_encodings(crop, locs)
        if not encs:
            return None
        d = face_recognition.face_distance(self.known_enc, encs[0])
        j = int(np.argmin(d))
        return self.known_ids[j] if d[j] <= self.tolerance else None

    def resolve(self, rgb: Optional[np.ndarray], boxes: List[Box]) -> List[str]:
        n = len(boxes)
        ids: List[Optional[str]] = [None] * n
        used: Set[str] = set()

        for i, b in enumerate(boxes):                         # pass 1: faces
            sid = self._match_face(rgb, b)
            if sid and sid not in used:
                ids[i] = sid
                used.add(sid)

        pairs = sorted(                                       # pass 2: nearest known position
            (math.dist(center(boxes[i]), pos), i, sid)
            for i in range(n) if ids[i] is None
            for sid, pos in self.last_pos.items() if sid not in used
        )
        for d, i, sid in pairs:
            if d > self.max_dist:
                break
            if ids[i] is None and sid not in used:
                ids[i] = sid
                used.add(sid)

        for i in range(n):                                    # pass 3: new anonymous track
            if ids[i] is None:
                ids[i] = f"TRK_{self._next_trk:03d}"
                self._next_trk += 1
            self.last_pos[ids[i]] = center(boxes[i])
        return ids  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 3. Leave-early logic
# --------------------------------------------------------------------------- #
class AttendanceTracker:
    """
    Stores which students were seen in each interval:
        presence[0] -> Interval 1 (0-15 min), presence[1] -> Interval 2, ...
    Rule: present in Interval 1 but absent from any *completed* later interval -> LEFT_EARLY / ABSENT.
    Only completed intervals are judged, so nobody is flagged for an interval still in progress.
    """

    PRESENT, ABSENT, PENDING = "PRESENT", "ABSENT", "PENDING"

    def __init__(self, interval_sec: float, num_intervals: int, roster=()):
        self.interval_sec = interval_sec
        self.num_intervals = num_intervals
        self.roster: Set[str] = set(roster)
        self.presence: List[Set[str]] = [set() for _ in range(num_intervals)]
        self.status: Dict[str, str] = {s: self.PENDING for s in self.roster}
        self.left_early: Set[str] = set()

    def interval_index(self, elapsed: float) -> int:
        return min(int(elapsed // self.interval_sec), self.num_intervals - 1)

    def interval_label(self, k: int) -> str:
        m = self.interval_sec / 60.0
        return f"Interval {k + 1} ({k * m:g}-{(k + 1) * m:g} min)"

    def update(self, present_ids: List[str], elapsed: float) -> List[str]:
        """Record one sample. Returns IDs newly flagged LEFT_EARLY by this call."""
        k = self.interval_index(elapsed)
        for sid in present_ids:
            if self.roster and sid not in self.roster:
                continue                                      # ignore unknown people when a roster exists
            self.presence[k].add(sid)
            self.status.setdefault(sid, self.PENDING)
            if k == 0 and sid not in self.left_early:
                self.status[sid] = self.PRESENT
        return self._evaluate(completed=k)

    def finalize(self, elapsed: float) -> List[str]:
        """Call when the stream ends. Judges every interval that actually finished."""
        completed = min(self.num_intervals, int(elapsed // self.interval_sec))
        return self._evaluate(completed)

    def _evaluate(self, completed: int) -> List[str]:
        newly: List[str] = []
        if completed >= 1:                                    # Interval 1 is over: no-shows are ABSENT
            for sid, st in self.status.items():
                if st == self.PENDING:
                    self.status[sid] = self.ABSENT
        for sid in sorted(self.presence[0]):
            if sid in self.left_early:
                continue
            if any(sid not in self.presence[k] for k in range(1, completed)):
                self.left_early.add(sid)
                self.status[sid] = self.ABSENT
                newly.append(sid)
        return newly

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for sid in sorted(self.status):
            row = {"student_id": sid,
                   "status": "LEFT_EARLY" if sid in self.left_early else self.status[sid],
                   "attendance": self.status[sid]}
            for k in range(self.num_intervals):
                row[self.interval_label(k)] = sid in self.presence[k]
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Logging + orchestration
# --------------------------------------------------------------------------- #
class EventLogger:
    """Appends one JSON object per line (JSONL) - easy to tail, ship, or load with pandas.read_json(lines=True)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


@dataclass
class FrameResult:
    frame: np.ndarray       # annotated BGR frame
    event: dict             # structured event (JSON-serialisable)
    elapsed: float


def _put_label(img, text, org, color, scale=0.5):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), max(int(org[1]), th + bl + 2)
    cv2.rectangle(img, (x, y - th - bl - 2), (x + tw + 4, y + bl - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


class ClassroomMonitor:
    """High-level API. Typical integration:

        monitor = ClassroomMonitor(MonitorConfig(faces_dir="known_faces"))
        result  = monitor.process_frame(frame_bgr, elapsed_sec)   # Streamlit / FastAPI handler
        result.frame  -> annotated image      result.event -> dict for the teacher dashboard
        ...
        df = monitor.finish(last_elapsed)                          # final attendance table
    """

    def __init__(self, cfg: Optional[MonitorConfig] = None, class_start: Optional[datetime] = None):
        self.cfg = cfg or MonitorConfig()
        self.class_start = class_start or datetime.now()
        self.detector = PhoneDetectionEngine(self.cfg)
        self.identity = IdentityResolver(self.cfg.faces_dir, self.cfg.face_tolerance, self.cfg.max_match_dist_px)
        self.attendance = AttendanceTracker(self.cfg.interval_minutes * 60, self.cfg.num_intervals,
                                            self.identity.roster)
        self.logger = EventLogger(self.cfg.log_path)
        self._phone_streak: Dict[str, int] = {}
        self.phone_alert_counts: Dict[str, int] = {}
        self._finished = False

    # ---- core ----
    def process_frame(self, frame_bgr: np.ndarray, elapsed: float) -> FrameResult:
        persons, phones = self.detector.detect(frame_bgr)
        rgb = ClassroomPipeline.to_rgb(frame_bgr) if self.identity.known_enc else None
        ids = self.identity.resolve(rgb, [p.box for p in persons])

        # phone -> student, keep the most confident phone per student
        best: Dict[int, PhoneLink] = {}
        for link in self.detector.link_phones_to_persons(persons, phones):
            if link.person_idx not in best or link.phone.conf > best[link.person_idx].phone.conf:
                best[link.person_idx] = link

        alerts, confirmed, active = [], [], set()
        for idx, link in best.items():
            sid = ids[idx]
            active.add(sid)
            self._phone_streak[sid] = self._phone_streak.get(sid, 0) + 1
            if self._phone_streak[sid] >= self.cfg.phone_confirm_samples:   # temporal debounce
                alerts.append({"student_id": sid, "confidence": round(link.phone.conf, 2)})
                self.phone_alert_counts[sid] = self.phone_alert_counts.get(sid, 0) + 1
                confirmed.append((idx, link))
        for sid in self._phone_streak:
            if sid not in active:
                self._phone_streak[sid] = 0

        newly_left = self.attendance.update(ids, elapsed)

        event = self._make_event(elapsed, alerts, newly_left)
        event["people_detected"] = len(persons)
        self.logger.log(event)
        return FrameResult(self._draw(frame_bgr, persons, ids, phones, confirmed, event), event, elapsed)

    def run_stream(self, source) -> Iterator[FrameResult]:
        """Generator over a webcam / video / snapshot folder. Call `finish()` yourself afterwards."""
        total = self.cfg.interval_minutes * 60 * self.cfg.num_intervals
        with ClassroomPipeline(source, self.cfg.resize_width, self.cfg.sample_every_sec) as pipe:
            for frame, elapsed in pipe.frames():
                yield self.process_frame(frame, elapsed)
                if elapsed >= total:
                    break

    def finish(self, elapsed: float) -> pd.DataFrame:
        """Finalize attendance, write the CSV, log a closing event, return the attendance DataFrame."""
        if self._finished:
            return self.attendance.to_dataframe()
        self._finished = True
        flagged = self.attendance.finalize(elapsed)
        event = self._make_event(elapsed, [], flagged)
        event["final"] = True
        self.logger.log(event)
        df = self.attendance.to_dataframe()
        if not df.empty:
            df["phone_alerts"] = df["student_id"].map(self.phone_alert_counts).fillna(0).astype(int)
        df.to_csv(self.cfg.attendance_csv, index=False)
        return df

    # ---- internals ----
    def _make_event(self, elapsed: float, alerts: list, flags: list) -> dict:
        ts = self.class_start + timedelta(seconds=elapsed)
        k = self.attendance.interval_index(elapsed)
        return {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 1),
            "interval": self.attendance.interval_label(k),
            "phone_alerts": alerts,
            "leave_early_flags": flags,
        }

    def _draw(self, frame, persons, ids, phones, confirmed, event) -> np.ndarray:
        out = frame.copy()
        for ph in phones:                                           # all phones: thin yellow
            cv2.rectangle(out, *[tuple(map(int, p)) for p in (ph.box[:2], ph.box[2:])], YELLOW, 1)
        alerted = {idx: link for idx, link in confirmed}
        for i, (p, sid) in enumerate(zip(persons, ids)):
            x1, y1, x2, y2 = map(int, p.box)
            if i in alerted:
                link = alerted[i]
                cv2.rectangle(out, (x1, y1), (x2, y2), RED, 3)
                px1, py1, px2, py2 = map(int, link.phone.box)
                cv2.rectangle(out, (px1, py1), (px2, py2), RED, 3)
                pc = tuple(map(int, center(link.phone.box)))
                cv2.line(out, pc, (int((x1 + x2) / 2), int(y1 + 0.6 * (y2 - y1))), RED, 1, cv2.LINE_AA)
                _put_label(out, f"PHONE ALERT {sid} {link.phone.conf:.2f}", (x1, y1), RED)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
                _put_label(out, sid, (x1, y1), GREEN)

        left = sorted(self.attendance.left_early)
        mm, ss = divmod(int(event["elapsed_sec"]), 60)
        lines = [f"t={mm:02d}:{ss:02d}  {event['interval']}",
                 f"People: {len(persons)}   Phone alerts: {len(confirmed)}",
                 f"Left early: {', '.join(left) if left else '-'}"]
        for n, line in enumerate(lines):
            _put_label(out, line, (8, 22 + n * 22), RED if (n == 2 and left) else DARK, 0.55)
        return out


# --------------------------------------------------------------------------- #
# 5. Example usage
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Classroom behavioural monitor")
    ap.add_argument("--source", default="0", help="webcam index, video path, RTSP URL, or folder of snapshots")
    ap.add_argument("--faces-dir", default=None, help="folder of enrolled faces, e.g. ST102.jpg")
    ap.add_argument("--sample-sec", type=float, default=5.0, help="run inference every N seconds")
    ap.add_argument("--interval-minutes", type=float, default=15.0,
                    help="length of each attendance interval (use e.g. 0.5 to test on a short clip)")
    ap.add_argument("--intervals", type=int, default=3)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = MonitorConfig(faces_dir=args.faces_dir, sample_every_sec=args.sample_sec,
                        interval_minutes=args.interval_minutes, num_intervals=args.intervals)
    monitor = ClassroomMonitor(cfg)

    last = 0.0
    stream = monitor.run_stream(args.source)
    try:
        for res in stream:
            last = res.elapsed
            print(json.dumps(res.event))
            if not args.no_display:
                cv2.imshow("Classroom Monitor (q to quit)", res.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stream.close()
        df = monitor.finish(last)
        cv2.destroyAllWindows()

    print("\n=== Final attendance ===")
    print(df.to_string(index=False) if not df.empty else "(no students tracked)")
    print(f"\nEvents -> {cfg.log_path} | Attendance -> {cfg.attendance_csv}")


if __name__ == "__main__":
    main()
