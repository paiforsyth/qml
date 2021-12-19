from dataclasses import dataclass

from pants.backend.python.target_types import PythonSourceField
from pants.engine.rules import collect_rules
from pants.engine.target import Dependencies, FieldSet


@dataclass(frozen=True)
class ImportLinterFieldSet(FieldSet):
    """Used to let pants know what kind of targets importlinter cares about."""

    required_fields = (PythonSourceField,)
    source: PythonSourceField
    dependencies: Dependencies


def rules():
    return (*collect_rules(),)
