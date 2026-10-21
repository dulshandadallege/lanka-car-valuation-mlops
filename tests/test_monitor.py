from src.monitor import drift_report


def test_no_drift_on_same_data(datasets):
    data = datasets[0]
    rep = drift_report(data, data.sample(frac=0.5, random_state=0))
    assert not rep["retrain_recommended"]
    assert max(rep["psi"].values()) < 0.1


def test_drift_detected_on_shifted_data(datasets):
    data = datasets[0]
    shifted = data.copy()
    shifted["age"] = shifted["age"] + 8
    shifted["mileage_km"] = shifted["mileage_km"] * 1.8
    rep = drift_report(data, shifted)
    assert rep["status"]["age"] == "alert" and rep["retrain_recommended"]
