from phonon_omegamax.eval.leaderboard import build_leaderboard


def test_leaderboard_includes_ours_and_published():
    table = build_leaderboard(
        ours_gbdt={"mae_mean": 78.0, "mae_std": 4.0, "r2_mean": 0.71, "r2_std": 0.03},
        ours_cgcnn={"mae_mean": 60.0, "mae_std": 3.0, "r2_mean": 0.82, "r2_std": 0.02},
    )
    assert "Magpie + GBDT (ours)" in table
    assert "CGCNN (ours)" in table
    assert "Roost (published)" in table
    assert "75.6" in table  # Roost MAE
    assert "29.5" in table  # ALIGNN MAE
    # Δ vs Magpie column for CGCNN (78 - 60 = 18 cm⁻¹)
    assert "+18.0" in table or "18.0" in table


def test_leaderboard_handles_missing_cgcnn():
    table = build_leaderboard(
        ours_gbdt={"mae_mean": 78.0, "mae_std": 4.0, "r2_mean": 0.71, "r2_std": 0.03},
        ours_cgcnn=None,
    )
    assert "Magpie + GBDT (ours)" in table
    assert "CGCNN (ours)" not in table
