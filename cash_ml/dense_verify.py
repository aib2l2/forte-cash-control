"""Verify actual candidate inference, timing and local production transport."""

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from backend.config import Settings
from backend.services.video import extract_evidence_frame
from cash_ml.dense_experiment import cache_path, data_for
from cash_ml.dense_service import DenseCandidateAnalyzer
from cash_ml.experiment import load_model, load_npz, predict, write_json
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, Extractor, sha256
from cash_ml.service import event_peaks


def verify_evidence(source, event):
    assert event["timecode_sec"] == event["evidence_frames"][0]["timestamp_sec"]
    evidence = event["evidence_frames"][0]
    cap = cv2.VideoCapture(str(source))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, evidence["timestamp_sec"]*1000)
        ok, frame = cap.read()
        assert ok
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        actual = base64.b64decode(evidence["image_base64"])
        method = 'seek'
        actual_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if actual != encoded.tobytes() or abs(actual_time-evidence['timestamp_sec']) > 1e-6:
            # Fractional MOV seeks can round down on a second seek. Establish
            # the evidence's source frame by sequential timestamps instead.
            cap.release()
            cap = cv2.VideoCapture(str(source))
            method = 'sequential_decode'
            while True:
                ok, frame = cap.read()
                assert ok, 'Evidence timestamp not present in source'
                actual_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if actual_time >= evidence['timestamp_sec']-1e-6:
                    break
            assert abs(actual_time-evidence['timestamp_sec']) <= 1e-6
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert actual == encoded.tobytes()
        return {"shape": list(frame.shape), "timestamp": evidence["timestamp_sec"],
                "exact_jpeg_match": True, "verification_method": method}
    finally:
        cap.release()


def benchmark(run, records):
    candidate = DenseCandidateAnalyzer(run)
    spec = candidate.extractor.spec
    tests = [r for r in records if r["split"] == 'test']
    cached = data_for(run, tests, spec)
    baseline, _, threshold, _ = load_model(run/'baseline_refit.npz')
    extractor = Extractor()
    rows = []
    for r in tests:
        source = Path(r["path"])
        for name in ('baseline', 'candidate'):
            start = time.monotonic()
            if name == 'baseline':
                item = extractor.extract(source)
                expected = load_npz(MODEL_DIR/'features'/f"{r['sha256']}-{PREPROCESS_VERSION}.npz")
                scores, types = predict(baseline, item)
                indices = event_peaks(item['starts'], item['ends'], scores, threshold, types)
                events = []
                for i in indices:
                    evidence = extract_evidence_frame(source, float(item['centers'][i]))
                    events.append({'type': str(types[i]), 'timecode_sec': evidence.timestamp_sec,
                                   'evidence_frames': [evidence.model_dump()]})
            else:
                item = candidate.extractor.extract(source)
                expected = cached[r['id']]
                scores, types = predict(candidate.model, item)
                result = candidate.from_features(source, item)
                events = [e.model_dump() for e in result.violations]
                write_json(run/'candidate_real_result.json', result.model_dump())
            elapsed = time.monotonic()-start
            delta = float(np.max(abs(item['features']-expected['features'])))
            np.testing.assert_allclose(item['features'], expected['features'], atol=2e-5, rtol=1e-5)
            np.testing.assert_array_equal(item['centers'], expected['centers'])
            model = baseline if name == 'baseline' else candidate.model
            limit = threshold if name == 'baseline' else candidate.threshold
            expected_scores, expected_types = predict(model, expected)
            np.testing.assert_array_equal(types, expected_types)
            np.testing.assert_array_equal(scores >= limit, expected_scores >= limit)
            evidence_checks = [verify_evidence(source, e) for e in events]
            rows.append({'video': r['id'], 'model': name, 'duration_seconds': float(item['duration']),
                         'elapsed_seconds': elapsed, 'features_max_abs_difference': delta,
                         'event_count': len(events), 'evidence_checks': evidence_checks})
            write_json(run/'benchmark_partial.json', rows)
            print('benchmark', name, r['id'], round(elapsed, 2), 'seconds', flush=True)
    report = {'device': 'CPU, 4 torch threads', 'includes': 'decode, feature extraction, heads, merge, JPEG evidence',
              'excludes': 'model load, upload, queue and extra verification decode', 'rows': rows,
              'summary': {n: {'mean_seconds': float(np.mean([r['elapsed_seconds'] for r in rows if r['model'] == n])),
                              'total_seconds': sum(r['elapsed_seconds'] for r in rows if r['model'] == n),
                              'video_seconds': sum(r['duration_seconds'] for r in rows if r['model'] == n)}
                          for n in ('baseline', 'candidate')}}
    write_json(run/'benchmark.json', report)


def temporal_audit(run, records):
    rows = []
    for record in records:
        for fps, views in [(8, 'details'), (15, 'full')]:
            path = cache_path(run, record, fps, views)
            if not path.exists():
                continue
            d = load_npz(path)
            actual, targets, duration = d['frame_times'], d['target_times'], float(d['duration'])
            assert actual.shape == targets.shape and actual.shape[1] == 16
            assert actual.min() >= 0 and actual.max() < duration
            assert np.all(np.diff(actual, axis=1) >= 0)
            assert targets[0, 0] == 0
            assert duration-targets[-1, -1] <= 1/float(d['source_fps'])+1e-6
            assert np.max(abs(actual-targets)) <= max(.25, 2/float(d['source_fps']))
            rows.append({'video': record['id'], 'fps': fps, 'views': views, 'clips': len(actual),
                         'max_frame_error_seconds': float(np.max(abs(actual-targets))),
                         'elapsed_seconds': float(d['elapsed_seconds']), 'path': str(path), 'sha256': sha256(path)})
    write_json(run/'temporal_audit.json', {'all_checks_passed': True, 'pairs': len(rows),
                                          'clips': sum(r['clips'] for r in rows), 'rows': rows})


def live_api(run, records):
    settings = Settings.from_env()
    headers = {'X-API-Key': settings.api_key}
    rows = []
    allowed = json.loads((run/'protocol.json').read_text())['allowed_types']
    with httpx.Client(base_url=settings.api_base_url, headers=headers, timeout=60, trust_env=False) as client:
        status = client.get('/api/v1/model').json()
        assert status['active_version'] == 'typed_v2'
        assert status['allowed_types'] == allowed
        for name in ['Х-20', 'Х-21', 'corrupt']:
            source = next((Path(r['path']) for r in records if r['id'] == name), None)
            content = source.read_bytes() if source else b'invalid MOV'
            start = time.monotonic()
            response = client.post('/api/v1/analyze', files={'file': ('anonymous.mov', content, 'video/quicktime')})
            assert response.status_code == 202
            job_id = response.json()['job_id']
            states = []
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                job = client.get('/api/v1/jobs/'+job_id).json()
                states.append(job['status'])
                if job['status'] != 'processing':
                    break
                time.sleep(.25)
            assert job['status'] == ('failed' if name == 'corrupt' else 'done')
            checks = []
            for event in job.get('violations', []):
                assert event['type'] in allowed
                checks.append(verify_evidence(source, event))
            rows.append({'source': name, 'states': sorted(set(states)), 'job_id': job_id,
                         'elapsed_seconds_including_queue': time.monotonic()-start,
                         'verdict': job.get('verdict'), 'events': [{k: e[k] for k in ['type', 'timecode_sec']}
                                                               for e in job.get('violations', [])],
                         'evidence_checks': checks})
        invalid = [client.post('/api/v1/analyze', files={'file': (name, content, 'video/quicktime')}).status_code
                   for name, content in [('invalid.mp4', b'a'), ('empty.mov', b'')]]
        assert invalid == [415, 400]
        write_json(run/'api_e2e.json', {'active_version': status['active_version'], 'allowed_types': allowed,
                                       'invalid_upload_statuses': invalid, 'rows': rows})
    print('live API queue, JSON, types, actual JPEGs and invalid uploads verified', flush=True)


def candidate_api(run, records):
    from fastapi.testclient import TestClient

    from backend.main import create_app

    analyzer = DenseCandidateAnalyzer(run)
    comparison = json.loads((run/'comparison.json').read_text())
    flagged = {r['video'] for r in comparison['candidate']['test']['videos'] if r['events']}
    eligible = [r for r in records if r['split'] == 'test' and r['id'] in flagged]
    record = eligible[0] if eligible else next(r for r in records if r['split'] == 'test')
    settings = Settings(api_key='local-candidate-check', storage_dir=run/'candidate_queue', max_pending_jobs=1)
    rows = []
    with TestClient(create_app(settings, analyzer)) as client:
        headers = {'X-API-Key': settings.api_key}
        for corrupt in (False, True):
            content = b'broken MOV' if corrupt else Path(record['path']).read_bytes()
            response = client.post('/api/v1/analyze', headers=headers,
                                   files={'file': ('anonymous.mov', content, 'video/quicktime')})
            assert response.status_code == 202
            job_id = response.json()['job_id']
            states = []
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                job = client.get('/api/v1/jobs/'+job_id, headers=headers).json()
                states.append(job['status'])
                if job['status'] != 'processing':
                    break
                time.sleep(.25)
            assert job['status'] == ('failed' if corrupt else 'done')
            for event in job.get('violations', []):
                verify_evidence(Path(record['path']), event)
            rows.append({'corrupt': corrupt, 'states': sorted(set(states)), 'job_id': job_id,
                         'verdict': job.get('verdict'), 'events': [{k: e[k] for k in ('type', 'timecode_sec')}
                                                               for e in job.get('violations', [])]})
    assert not list((settings.storage_dir/'uploads').glob('*'))
    write_json(run/'candidate_api_e2e.json', {'isolated_test_client': True, 'production_changed': False,
                                             'source': record['id'], 'rows': rows})
    print('candidate queue, real video, JPEG evidence and corrupt-input failure verified', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=['benchmark', 'api', 'temporal', 'candidate-api'], required=True)
    args = parser.parse_args()
    records = json.loads((args.run/'grouped_records.json').read_text())
    {'benchmark': benchmark, 'api': live_api, 'temporal': temporal_audit,
     'candidate-api': candidate_api}[args.mode](args.run, records)


if __name__ == '__main__':
    main()
