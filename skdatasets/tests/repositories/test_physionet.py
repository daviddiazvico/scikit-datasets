from __future__ import annotations

from skdatasets.repositories.physionet import fetch


def test_fetch_ctu_uhb_ctgdb() -> None:
    """Tests LIBSVM australian dataset."""
    X, y = fetch(
        'ctu-uhb-ctgdb',
        return_X_y=True,
        target_column=["pH", "BDecf", "pCO2", "BE", "Apgar1", "Apgar5"],
    )
    assert X.shape == (552, 30)
    assert y.shape == (552, 6)


def test_fetch_ctu_uhb_ctgdb_single_target() -> None:
    """Tests LIBSVM australian dataset."""
    X, y = fetch(
        'ctu-uhb-ctgdb',
        return_X_y=True,
        target_column="pH",
    )
    assert X.shape == (552, 35)
    assert y.shape == (552,)


def test_fetch_ctu_uhb_ctgdb_bunch() -> None:
    """Tests LIBSVM australian dataset."""
    bunch = fetch(
        'ctu-uhb-ctgdb',
        as_frame=True,
        target_column=["pH", "BDecf", "pCO2", "BE", "Apgar1", "Apgar5"],
    )
    assert bunch.data.shape == (552, 30)
    assert bunch.target.shape == (552, 6)
    assert bunch.frame.shape == (552, 36)
