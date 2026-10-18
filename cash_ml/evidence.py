"""Three real frames around a hypothesis; timestamps always come from the decoder."""

from backend.services.video import extract_evidence_frame


def context_frames(path, timestamp, duration, *, reader=extract_evidence_frame):
    primary = reader(path, timestamp)
    frames = [primary]
    # Stay inside the video even for a short clip or its final analysis window.
    for target in (max(0.0, timestamp - 2.0), min(max(0.0, duration - 0.1), timestamp + 2.0)):
        if abs(target - timestamp) < 0.04:
            continue
        try:
            frame = reader(path, target)
        except ValueError:
            continue
        if all(abs(frame.timestamp_sec - other.timestamp_sec) >= 0.04 for other in frames):
            frames.append(frame)
    return primary.timestamp_sec, sorted(frames, key=lambda x: x.timestamp_sec)
