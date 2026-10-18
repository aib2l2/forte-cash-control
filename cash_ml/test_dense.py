"""Checks of temporal semantics, training boundaries and numeric model exports."""

import json

import numpy as np
import pytest

from cash_ml.dense_features import DenseSpec, aggregate, clip_schedule, dense_views
from cash_ml.experiment import fit_head, head_logits, load_model, save_model


@pytest.mark.parametrize("fps", [8, 15])
def test_sampling_rate_overlap_tail_and_short_clip(fps):
    times = clip_schedule(10.23, 30, fps)
    np.testing.assert_allclose(np.diff(times, axis=1), 1/fps, atol=1e-8)
    assert times[0, 0] == 0
    assert times[-1, -1] == pytest.approx(10.23-1/30)
    assert np.all(times[1:, 0] < times[:-1, -1])
    short = clip_schedule(.2, 30, fps)
    assert short.shape == (1, 16)
    assert short.min() == 0 and short.max() < .2
    assert np.all(np.diff(short[0]) >= 0)


@pytest.mark.parametrize("duration,rate", [(0, 30), (float('nan'), 30), (1, 0)])
def test_invalid_video_time_rejected(duration, rate):
    with pytest.raises(ValueError):
        clip_schedule(duration, rate, 8)


def synthetic_raw():
    centers = np.arange(.5, 18., 1.)
    return {"features": np.repeat(centers[:, None], 1536, axis=1), "centers": centers,
            "starts": np.maximum(0., centers-.5), "ends": centers+.5,
            "duration": np.asarray(18.), "fps": np.asarray(8), "views": np.asarray("details"),
            "frame_times": np.repeat(centers[:, None], 16, axis=1)}


def test_ordered_context_retains_before_center_after_without_moving_anchor():
    raw = synthetic_raw()
    center = aggregate(raw, DenseSpec(8, "full", 0, "center"))
    mean = aggregate(raw, DenseSpec(8, "full", 8, "mean"))
    ordered = aggregate(raw, DenseSpec(8, "full", 8, "ordered"))
    np.testing.assert_array_equal(center['centers'], mean['centers'])
    np.testing.assert_array_equal(center['centers'], ordered['centers'])
    i = int(np.argmin(abs(ordered['centers']-8)))
    assert ordered['features'][i, 0] < ordered['features'][i, 512] < ordered['features'][i, 1024]
    assert np.all(ordered['starts'] <= ordered['centers'])
    assert np.all(ordered['ends'] >= ordered['centers'])
    assert ordered['edge_context'][0] and ordered['edge_context'][-1]
    assert np.all(np.isfinite(ordered['features']))
    assert ordered['context_starts'][0] == 0 and ordered['context_ends'][-1] == 18
    assert np.max(ordered['ends']-ordered['starts']) <= 2


def test_single_frame_context_and_missing_bins_are_finite():
    raw = synthetic_raw()
    raw = {k: v[:1] if isinstance(v, np.ndarray) and v.ndim else v for k, v in raw.items()}
    raw['duration'] = np.asarray(.6)
    out = aggregate(raw, DenseSpec(8, "full", 16, "ordered"))
    assert out['features'].shape == (1, 1536)
    assert np.all(np.isfinite(out['features']))
    assert out['centers'][0] == .5


def test_full_and_multiview_share_exact_full_pixels_and_features():
    frame = np.random.default_rng(7).integers(0, 256, (848, 464, 3), dtype=np.uint8)
    np.testing.assert_array_equal(dense_views(frame, 'full')[0], dense_views(frame, 'details')[0])
    raw = synthetic_raw()
    full = aggregate(raw, DenseSpec(8, 'full', 8, 'mean'))
    details = aggregate(raw, DenseSpec(8, 'details', 8, 'mean'))
    np.testing.assert_allclose(full['features'], details['features'][:, :512])
    assert details['features'].shape[1] == 1536


def test_sample_rate_and_missing_view_mismatch_are_rejected():
    raw = synthetic_raw()
    with pytest.raises(ValueError, match='rate'):
        aggregate(raw, DenseSpec(15, 'full', 8, 'mean'))
    raw['views'] = np.asarray('full')
    with pytest.raises(ValueError, match='detail'):
        aggregate(raw, DenseSpec(8, 'details', 8, 'mean'))


@pytest.mark.parametrize('family', ['logistic', 'svm'])
def test_dense_model_export_preserves_ordered_spec_and_logits(tmp_path, family):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 6))
    binary = fit_head(x, [0]*6+[1]*6, np.ones(12), family)
    typed = fit_head(x, ['a']*4+['b']*4+['c']*4, np.ones(12), family)
    path = tmp_path/'model.npz'
    save_model(path, {'binary': binary, 'typed': typed}, DenseSpec().to_json(), .25, {'allowed_types': ['a', 'b', 'c']})
    model, spec, threshold, metadata = load_model(path)
    assert DenseSpec(**json.loads(spec)) == DenseSpec()
    assert threshold == .25 and metadata['allowed_types'] == ['a', 'b', 'c']
    np.testing.assert_array_equal(head_logits(model['binary'], x), head_logits(binary, x))
    np.testing.assert_array_equal(head_logits(model['typed'], x), head_logits(typed, x))


def test_dense_corrupt_video_fails_instead_of_returning_normal(tmp_path):
    from cash_ml.dense_features import DenseExtractor

    path = tmp_path/'broken.mov'
    path.write_bytes(b'not a video')
    extractor = object.__new__(DenseExtractor)
    extractor.spec = DenseSpec()
    with pytest.raises(ValueError, match='decoded'):
        extractor.extract_raw(path)


def test_hard_negatives_cannot_use_validation_or_positive_video(monkeypatch):
    from cash_ml import dense_experiment

    monkeypatch.setattr(dense_experiment, 'train_heads', lambda *a: {})
    train = [{'id': 'normal_train', 'label': 0}]
    with pytest.raises(ValueError, match='TRAIN'):
        dense_experiment.train_dense(train, {}, 'logistic', {'normal_validation': [1.]})


def test_dense_inference_uses_real_evidence_time_and_whitelist(monkeypatch):
    from pathlib import Path

    from pydantic import ValidationError

    from backend.schemas import EvidenceFrame
    from cash_ml import dense_service
    from cash_ml.service import violation_contract

    analyzer = object.__new__(dense_service.DenseCandidateAnalyzer)
    analyzer.model = {}
    analyzer.threshold = .5
    analyzer.event_model = violation_contract(['safe_left_open'])
    item = {'features': np.ones((1, 1)), 'centers': np.array([1.]), 'starts': np.array([0.]),
            'ends': np.array([2.]), 'context_starts': np.array([0.]), 'context_ends': np.array([5.])}
    monkeypatch.setattr(dense_service, 'predict', lambda *a: (np.array([.8]), np.array(['safe_left_open'])))
    monkeypatch.setattr(dense_service, 'extract_evidence_frame', lambda *a: EvidenceFrame(
        timestamp_sec=1.033, image_base64='/9j/2Q=='))
    result = analyzer.from_features(Path('anonymous.mov'), item)
    assert result.violations[0].timecode_sec == 1.033
    assert result.violations[0].evidence_frames[0].timestamp_sec == 1.033
    monkeypatch.setattr(dense_service, 'predict', lambda *a: (np.array([.8]), np.array(['new_type'])))
    with pytest.raises(ValidationError):
        analyzer.from_features(Path('anonymous.mov'), item)


def test_candidate_tampering_is_rejected_before_model_loading(tmp_path):
    from cash_ml.dense_service import DenseCandidateAnalyzer

    (tmp_path/'selected.npz').write_bytes(b'modified')
    (tmp_path/'selection_frozen.json').write_text(json.dumps({'selected': 'selected', 'model_sha256': 'wrong'}))
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact hash'):
        DenseCandidateAnalyzer(tmp_path)


def test_real_short_video_decode_retains_actual_frame_times(tmp_path):
    import cv2
    import torch

    from cash_ml.dense_features import DenseExtractor

    path = tmp_path/'short.mov'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (48, 80))
    assert writer.isOpened()
    for value in range(6):
        writer.write(np.full((80, 48, 3), value*30, np.uint8))
    writer.release()

    class RecordingNet(torch.nn.Module):
        def forward(self, tensor):
            assert tensor.shape[1:] == (3, 16, 112, 112)
            return tensor.mean(dim=(1, 2, 3, 4)).unsqueeze(1)

    extractor = object.__new__(DenseExtractor)
    extractor.spec = DenseSpec(15, 'full', 16, 'ordered')
    extractor.net = RecordingNet()
    raw = extractor.extract_raw(path)
    assert raw['frame_times'].shape == (1, 16)
    assert raw['duration'] == pytest.approx(.2)
    assert raw['frame_times'][0, -1] == pytest.approx(5/30)
    item = aggregate(raw, extractor.spec)
    assert item['centers'][0] == raw['frame_times'][0, 8]
    assert item['context_starts'][0] == 0 and item['context_ends'][0] == .2
    assert np.all(np.isfinite(item['features']))


def test_evidence_verification_handles_fractional_seek_rounding(monkeypatch):
    import base64

    import cv2

    from cash_ml import dense_verify

    frames = [np.full((10, 10, 3), i*40, np.uint8) for i in range(5)]

    class Capture:
        def __init__(self, path):
            self.index = -1

        def set(self, prop, value):
            self.index = 1  # Re-seek erroneously returns frame 2, not frame 3.

        def read(self):
            self.index += 1
            return True, frames[self.index]

        def get(self, prop):
            return self.index/30*1000

        def release(self):
            pass

    _, jpeg = cv2.imencode('.jpg', frames[3], [cv2.IMWRITE_JPEG_QUALITY, 85])
    event = {'timecode_sec': .1, 'evidence_frames': [{'timestamp_sec': .1,
             'image_base64': base64.b64encode(jpeg).decode()}]}
    monkeypatch.setattr(dense_verify.cv2, 'VideoCapture', Capture)
    checked = dense_verify.verify_evidence('anonymous.mov', event)
    assert checked['exact_jpeg_match'] and checked['verification_method'] == 'sequential_decode'
