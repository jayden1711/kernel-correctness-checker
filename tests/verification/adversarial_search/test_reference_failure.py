"""Tests for the reference-failure classification split (2026-08-27).

The contract under test: only curated domain checks may classify a reference
failure as "domain"; execution errors are "domain"; ANY other failing check --
including ones that do not exist yet -- classifies as "invariant"
(reference-suspect). The July flash_attention signature must classify as
"invariant" both from a live result object and from a stored pre-split
failure summary.
"""

from types import SimpleNamespace

from verification.adversarial_search.reference_failure import (
    DOMAIN_CHECKS,
    classify_failed_checks,
    classify_reference_failure,
    failed_checks_from_summary,
    invariant_failures,
)


def _res(passed_checker=False, error=None, failed=(), passed=()):
    return SimpleNamespace(
        passed_checker=passed_checker,
        error=error,
        check_results=(
            [{"check_name": n, "passed": False} for n in failed]
            + [{"check_name": n, "passed": True} for n in passed]),
    )


def test_reference_passed_is_none():
    assert classify_reference_failure(_res(passed_checker=True)) is None


def test_execution_error_is_domain():
    r = _res(error=SimpleNamespace(error_type="TimeoutError"))
    assert classify_reference_failure(r) == "domain"


def test_pure_domain_checks_are_domain():
    assert classify_failed_checks(["nan_inf"]) == "domain"
    assert classify_failed_checks(["nan_inf", "dtype_preserved"]) == "domain"


def test_invariant_check_is_invariant():
    # the exact July 2026 signature
    assert classify_failed_checks(["attention_weights_sum_to_one"]) == "invariant"


def test_mixed_failures_are_invariant():
    got = classify_failed_checks(["nan_inf", "attention_weights_sum_to_one"])
    assert got == "invariant"
    assert invariant_failures(["nan_inf", "attention_weights_sum_to_one"]) == \
        ["attention_weights_sum_to_one"]


def test_unknown_future_check_defaults_to_invariant():
    # The loud direction: a check added later must NOT be silently absorbed
    # into "invalid input". This is the regression guard for the failure mode
    # that hid the masking bug.
    assert classify_failed_checks(["some_check_added_in_2027"]) == "invariant"


def test_no_records_is_domain():
    assert classify_failed_checks([]) == "domain"


def test_live_result_object_invariant():
    r = _res(failed=["attention_weights_sum_to_one"], passed=["nan_inf"])
    assert classify_reference_failure(r) == "invariant"


def test_summary_parse_july_record():
    # verbatim from the 2026-07-23 flash run's stored verdicts
    s = ("Reference failed: ['attention_weights_sum_to_one'] | "
         "Missed: ['approx_denom', 'drop_last_tile', 'skip_rescaling', "
         "'wrong_mask']")
    failed = failed_checks_from_summary(s)
    assert failed == ["attention_weights_sum_to_one"]
    assert classify_failed_checks(failed) == "invariant"


def test_summary_parse_no_reference_failure():
    assert failed_checks_from_summary("All checks passed.") is None


def test_domain_list_is_curated_and_small():
    # If this grows, it must be a deliberate decision -- see the module
    # docstring. The test forces the diff to touch this file too.
    assert DOMAIN_CHECKS == {"nan_inf", "dtype_preserved", "output_shape",
                             "kernel_executed"}
