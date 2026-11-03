"""The package imports, under the name the directory actually has.

This directory was called resnet/ before it grew a second model family and was renamed to
other_models/. The rename was done on disk but not in the code, so every module still said
`from resnet.config import ...` and the whole package -- training, tests, everything -- failed with
ModuleNotFoundError. Nothing else in the suite could catch that, because the suite could not load.
"""
from __future__ import annotations

import importlib
import re

import pytest

from other_models.datasets import ROOT

PACKAGE = ROOT / "other_models"
MODULES = ["artifacts", "augment", "config", "datasets", "engine", "metrics", "models",
           "prepare_data", "train", "visualise_masks"]

# rfdetr_adapter is deliberately not in MODULES: importing it pulls in rfdetr, which costs
# ~10 s because of transformers and would roughly double the fast suite. It gets its own
# slow test below instead. The source scan further down still covers it, since that reads
# files rather than importing them.
LAZY_MODULES = ["rfdetr_adapter"]

# "resnet50"/"resnet18" are torchvision model names and entirely legitimate; what must not appear
# is the old package name, as an import (`resnet.config`) or a path (`resnet/train.py`).
OLD_NAME = re.compile(r"\bresnet[./]")


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    importlib.import_module(f"other_models.{name}")


def test_no_module_imports_rfdetr_eagerly():
    """Training a Mask R-CNN must not pay rfdetr's ~10 s import.

    models.py names a variant by string and imports rfdetr only inside build(); artifacts.py
    holds RFDETR_LOSS_COLUMNS as plain data for the same reason. This is what notices if one
    of those becomes a module-level `import rfdetr` again.

    Run in a subprocess rather than by clearing sys.modules in-process. Dropping a parent
    package from sys.modules while its submodules stay cached leaves `rfdetr.config`
    unreachable by attribute for the rest of the session, which broke four unrelated tests
    when this was written the quick way.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        + "".join(f"import other_models.{name};" for name in MODULES)
        + "print('rfdetr' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=ROOT, check=True)
    assert result.stdout.strip() == "False", (
        "some module imports rfdetr at module scope, which adds ~10 s to every run and to "
        "test collection"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", LAZY_MODULES)
def test_every_lazy_module_imports(name):
    pytest.importorskip("rfdetr")
    importlib.import_module(f"other_models.{name}")


SOURCES = [p for p in sorted(PACKAGE.rglob("*.py")) + sorted(PACKAGE.glob("*.md"))
           if p.name != "test_imports.py"]      # this file names the old package to explain it


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_file_still_refers_to_the_old_package_name(path):
    """Docstrings, README commands and error messages too -- they tell the user what to run."""
    offenders = [f"{path.name}:{n}: {line.strip()}"
                 for n, line in enumerate(path.read_text().splitlines(), 1)
                 if OLD_NAME.search(line)]
    assert not offenders, "\n".join(offenders)
