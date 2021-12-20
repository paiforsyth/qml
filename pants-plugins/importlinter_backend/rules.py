from importlinter_backend.subsystem import ImportLinterFieldSet
from pants.backend.python.util_rules import pex_from_targets
from pants.core.goals.lint import LintRequest, LintResult, LintResults
from pants.engine.rules import collect_rules, rule
from pants.engine.unions import UnionRule


class ImportLinterRequest(LintRequest):
    """Will be pants by the pants engine to indicate that we have received a
    request to lint with import linter."""

    field_set_type = ImportLinterFieldSet


def rules():
    return [
        *collect_rules(),
        UnionRule(LintRequest, ImportLinterRequest),
        *pex_from_targets.rules(),
    ]


@rule
async def run_importlinter(
    request: ImportLinterRequest,
) -> LintResults:
    return LintResults(
        [LintResult(exit_code=0, stdout="", stderr="")], linter_name="importlinter"
    )
