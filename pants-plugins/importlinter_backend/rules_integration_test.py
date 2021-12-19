import pytest
from importlinter_backend.rules import ImportLinterRequest
from importlinter_backend.subsystem import ImportLinterFieldSet
from pants.backend.python.target_types import PythonSourceTarget
from pants.build_graph.address import Address
from pants.core.goals.lint import LintResult, LintResults
from pants.engine.rules import QueryRule
from pants.engine.target import Target
from pants.testutil.rule_runner import RuleRunner


@pytest.fixture
def rule_runner() -> RuleRunner:
    return RuleRunner(
        target_types=[PythonSourceTarget],
        rules=[QueryRule(LintResults, [ImportLinterRequest])],
    )


def run_import_linter(
    rule_runner: RuleRunner, targets: list[Target]
) -> tuple[LintResult, ...]:
    request = ImportLinterRequest(ImportLinterFieldSet.create(tgt) for tgt in targets)
    result = rule_runner.request(LintResults, [request])
    return result.results


def test_run_one_file(rule_runner: RuleRunner) -> None:
    """test that we do not get an error when running import linter in a
    directory with a single file."""
    # Set up the files and targets.
    file_name = "f1"
    rule_runner.write_files(
        {
            f"project/{file_name}.py": "",
            "project/BUILD": "python_sources()",
        }
    )
    tgt1 = rule_runner.get_target(Address("project", target_name=file_name))
    results = run_import_linter(rule_runner=rule_runner, targets=[tgt1])
    assert results[0].exit_code == 0
