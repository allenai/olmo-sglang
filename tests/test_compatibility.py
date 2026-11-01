from pathlib import Path

import tomllib


def test_sglang_compatibility_source_is_frozen() -> None:
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    compatibility = metadata["tool"]["olmo-sglang"]["compatibility"]

    assert compatibility == {
        "sglang-version": "0.0.0.dev1+gb84bec576",
        "sglang-source-commit": "b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02",
    }
