import json
from pathlib import Path
from types import SimpleNamespace

from atlas_posthoc_plus.config import load_config
from atlas_posthoc_plus.suite import run_suite


def test_suite_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "blocks.shp").touch()
    (project / "precincts.shp").touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[inputs]
blocks = "blocks.shp"
precincts = "precincts.shp"

[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "atlas_posthoc_plus.suite.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    config = load_config(config_path)
    code = run_suite("test", ["build-crosswalk"], config)

    assert code == 0
    manifest_path = project / "results/suite-test-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["success"] is True
    assert manifest["results"][0]["workflow"] == "build-crosswalk"
    assert "-m atlas_posthoc_plus.workflows.setup.crosswalk" in (
        manifest["results"][0]["command"]
    )

