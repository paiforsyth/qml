import pytest
from importlinter_backend.rules import ImportLinterRequest
from importlinter_backend.rules import rules as importlinter_rules
from importlinter_backend.subsystem import ImportLinterFieldSet
from pants.backend.python import target_types_rules
from pants.backend.python.target_types import (
    PythonRequirementTarget,
    PythonSourcesGeneratorTarget,
    PythonSourceTarget,
)
from pants.build_graph.address import Address
from pants.core.goals.lint import LintResult, LintResults
from pants.core.util_rules import config_files
from pants.engine.rules import QueryRule
from pants.engine.target import Target
from pants.testutil.rule_runner import RuleRunner


@pytest.fixture
def rule_runner() -> RuleRunner:
    return RuleRunner(
        target_types=[
            PythonSourceTarget,
            PythonSourcesGeneratorTarget,
            PythonRequirementTarget,
        ],
        rules=[
            *importlinter_rules(),
            *target_types_rules.rules(),
            *config_files.rules(),
            QueryRule(LintResults, [ImportLinterRequest]),
        ],
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
    project_name = "project"
    rule_runner.write_files(
        {
            f"{project_name}/{file_name}.py": "print('hello')",
            f"{project_name}/BUILD": "python_sources()",
        }
    )
    tgt1 = rule_runner.get_target(
        Address(project_name, relative_file_path=file_name + ".py")
    )
    results = run_import_linter(rule_runner=rule_runner, targets=[tgt1])
    assert results[0].exit_code == 0
