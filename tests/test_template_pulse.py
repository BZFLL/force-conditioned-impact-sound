"""Template-pulse control: the dataset substitution.

(a) With force_input="template_peak", force_waveform is template x
    max|measured window| and force_features (and the mel) are unchanged.
(b) The pooled template curve peaks in frame 240 // 32 = 7.
(a) needs the dataset and is skipped without it; (b) needs only the template
files in stats/.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import dataset as D
from model_ddsp import ForceConditionedFilterbankDDSP

ROOT = Path(__file__).resolve().parents[1]
N_INDICES = 20


def template_file(split):
    return ROOT / "stats" / f"template_pulse_{split}.pt"


needs_data = pytest.mark.skipif(
    not Path(D.DATA_ROOT).is_dir() or not Path(D.OBJECTS_CSV).exists(),
    reason="ObjectFolder-Real is not available on this machine")


def spread_indices(n):
    return np.linspace(0, n - 1, N_INDICES).astype(int).tolist()


@pytest.fixture(scope="module")
def measured():
    return D.ObjectFolderRealDataset()


def test_unknown_force_input_raises():
    with pytest.raises(AssertionError):
        D.ObjectFolderRealDataset(force_input="shuffled")


def test_template_peak_needs_an_existing_template():
    with pytest.raises(AssertionError):
        D.ObjectFolderRealDataset(force_input="template_peak", template_path="/nonexistent.pt")


@needs_data
@pytest.mark.skipif(not template_file("within").exists(), reason="template not fitted")
def test_template_peak_substitutes_only_the_force_window(measured):
    template = torch.load(template_file("within"), map_location="cpu",
                          weights_only=False)["mean_pulse"].numpy().astype(np.float64)
    templ = D.ObjectFolderRealDataset(force_input="template_peak",
                                      template_path=str(template_file("within")))
    for i in spread_indices(len(measured)):
        m, t = measured[i], templ[i]
        expect = torch.tensor(template * float(m["force_waveform"].abs().max()), dtype=torch.float32)
        assert torch.equal(t["force_waveform"], expect), f"index {i}: force_waveform"
        assert not torch.equal(t["force_waveform"], m["force_waveform"]), f"index {i}: unchanged window"
        assert torch.equal(t["force_features"], m["force_features"]), f"index {i}: force_features"
        assert torch.equal(t["mel"], m["mel"]), f"index {i}: mel"


@pytest.mark.parametrize("split", ["within", "heldout"])
def test_pooled_template_peaks_in_frame_seven(split):
    if not template_file(split).exists():
        pytest.skip("template not fitted")
    t = torch.load(template_file(split), map_location="cpu", weights_only=False)["mean_pulse"]
    curve = ForceConditionedFilterbankDDSP.pool_force_curve(t.float().view(1, -1), 300, 32)
    assert int(curve[0, :, 1].argmax()) == 240 // 32


def _c(delta, lo, hi, det):
    return dict(delta=delta, ci95=[lo, hi], detected=det)


@pytest.mark.parametrize("split, t_l4, t_l3, row, informative_null", [
    ("within", _c(+0.15, +0.05, +0.25, True), _c(-0.10, -0.20, -0.05, True), 1, None),
    ("within", _c(-0.15, -0.25, -0.05, True), _c(-0.30, -0.40, -0.20, True), 4, None),
    ("within", _c(+0.02, -0.05, +0.09, False), _c(-0.20, -0.30, -0.10, True), 2, True),
    ("within", _c(+0.00, -0.11, +0.11, False), _c(-0.20, -0.30, -0.10, True), 2, True),
    ("within", _c(+0.05, -0.02, +0.13, False), _c(-0.20, -0.30, -0.10, True), 2, False),
    ("heldout", _c(+0.02, -0.05, +0.09, False), _c(-0.20, -0.30, -0.10, True), 2, False),
    ("within", _c(+0.02, -0.05, +0.09, False), _c(-0.02, -0.10, +0.05, False), 3, None),
    ("within", _c(-0.02, -0.09, +0.05, False), _c(+0.20, +0.10, +0.30, True), None, None),
])
def test_template_pulse_reading_table(split, t_l4, t_l3, row, informative_null):
    import eval_template_pulse as TP
    r = TP.reading_row(t_l4, t_l3, split)
    assert r["row"] == row
    if row == 2:
        assert "is not detected" in r["sentence"] and "same gain" not in r["sentence"]
        assert r["informative_null"] is informative_null
