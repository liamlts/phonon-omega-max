import matplotlib
matplotlib.use("Agg")

import numpy as np

from phonon_omegamax.eval.parity import headline_figure


def test_headline_figure_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 1500, 200)
    gbdt = y + rng.normal(0, 50, 200)
    cgcnn = y + rng.normal(0, 30, 200)
    out = tmp_path / "headline.png"
    fig = headline_figure(y_true=y, gbdt_pred=gbdt, cgcnn_pred=cgcnn, out_path=out)
    assert out.exists()
    # Two panels (parity + residual KDE).
    assert len(fig.axes) >= 2
