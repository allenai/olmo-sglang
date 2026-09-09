from pathlib import Path

import tomllib


def test_sglang_compatibility_source_is_frozen() -> None:
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    compatibility = metadata["tool"]["olmo-sglang"]["compatibility"]

    assert compatibility == {
        "sglang-version": "0.5.19.dev49+g3145136",
        "sglang-source-commit": "3145136dcd1238754e0ea2b2ffd546532119c71c",
        "sglang-tag-object": "ff4c6e641d9f9bb174d34ff651c01c114aea8e40",
        "sglang-miles-audit-commit": "3bbb2812e2ca1defdee76f6ec09dbb2456c21c69",
    }
