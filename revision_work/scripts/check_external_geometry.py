"""CPU check of external box scaling, empty frames and false-alarm denominators."""
import json

import numpy as np
import torch

from revision_work.new_protocol.common import SERIES, atomic_json, now
from revision_work.new_protocol.data import ClipDataset
from revision_work.new_protocol.evaluate import evaluate_model
from revision_work.new_protocol.external import assert_external


def main():
    torch.set_num_threads(2)
    lock = assert_external()
    dataset = ClipDataset(lock['manifest'])
    chosen = []
    for lo, hi in [(0, 3), (3, 7), (7, float('inf'))]:
        chosen.append(next(i for i, row in enumerate(dataset.rows)
                           if len(row['boxes']) == 1 and lo < row['boxes'][0]['max_side_px'] <= hi))
    chosen.append(next(i for i, row in enumerate(dataset.rows) if len(row['boxes']) == 1 and i not in chosen))
    chosen += [i for i, row in enumerate(dataset.rows) if not row['boxes']][:4]
    assert len(chosen) == len(set(chosen)) == 8
    dataset.indices = chosen
    dataset.labels = [{'cls': np.array([[b['class']] for b in dataset.rows[i]['boxes']], dtype=np.float32)} for i in chosen]
    samples = [dataset[i] for i in range(len(dataset))]

    class CoordinateProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.names = {0: 'target'}
            self.cursor = 0

        def forward(self, images):
            output = torch.zeros(len(images), 5, 10)
            for j in range(len(images)):
                index = self.cursor + j
                box = samples[index]['bboxes_t5']
                if len(box):
                    output[j, :4, 0] = box[0] * 640
                    output[j, 4, 0] = .9
                elif index == 4:
                    output[j, :4, 0] = torch.tensor([320., 320., 20., 20.])
                    output[j, 4, 0] = .9
            self.cursor += len(images)
            return output

    out = SERIES / 'preflight/external_geometry'
    result = evaluate_model(CoordinateProbe(), dataset, out, 'synthetic_coordinate_probe',
                            lock['evaluation'], formal=False, workers=0)
    assert result['samples'] == result['unique_keyframes'] == 8
    assert result['false_positives'] == 1 and result['false_positives_per_frame'] == .125
    assert result['empty_prediction_frames'] == 3
    assert result['metrics']['metrics/recall(B)'] == 1.
    predictions = [json.loads(line) for line in (out / 'per_frame_predictions.jsonl').read_text().splitlines()]
    for i in range(4):
        np.testing.assert_allclose(predictions[i]['boxes_xyxy'][0], dataset.rows[chosen[i]]['boxes'][0]['xyxy'], atol=1e-3, rtol=0)
    audit = {'passed': True, 'checked_at': now(), 'device': 'CPU', 'formal_model_metrics': False,
             'description': 'Artificial GT-coordinate predictions plus one known false positive; no trained model or target scoring used',
             'source_samples': [[dataset.rows[i]['seq_id'], dataset.rows[i]['frame_id']] for i in chosen],
             'native_box_roundtrip_tolerance_px': .001, 'positive_frames': 4, 'negative_GT_frames': 4,
             'expected_false_positives': 1, 'expected_FP_per_frame': .125, 'empty_predictions_preserved': 3}
    atomic_json(out / 'checks.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
