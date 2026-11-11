import json
import re

from micro_insurance import static_dashboard


def test_static_dashboard_embeds_data(tmp_path):
    data = static_dashboard.build_data()
    assert len(data["fleets"]) == len(data["ebct2"]["premium"]) == len(data["frequency"])
    assert data["stop_loss"]["per_claim"][0] > data["stop_loss"]["per_claim"][-1] >= 0

    html = static_dashboard.render(data)
    assert html.startswith("<!doctype html>") and static_dashboard.PLACEHOLDER not in html
    assert "<!--BODY-->" not in html
    embedded = re.search(r"const D = (\{.*?\});\n", html, re.S).group(1)
    assert json.loads(embedded)["meta"]["n_fleets"] == len(data["fleets"])

    fragment = static_dashboard.render(data, full_document=False)
    assert "<!doctype" not in fragment and fragment.lstrip().startswith("<title>")
