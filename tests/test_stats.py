import pandas as pd

from vbcv.stages.stats import rally_table


def test_rally_table_reports_ballistic_serve_speed():
    rallies = pd.DataFrame({"rally_id": [0], "start_s": [0.0], "end_s": [10.0]})
    contacts = pd.DataFrame({"rally_id": [0], "side": ["near"], "label": ["serve"]})
    serves = pd.DataFrame({"rally_id": [0], "avg_ground_speed_kmh": [999.0], "ballistic_avg_kmh": [62.5]})
    out = rally_table(rallies, contacts, serves)
    assert out.serve_speed_kmh.iloc[0] == 62.5
