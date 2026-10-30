from repeated_errors import repeated_stage_error


def test_repeated_errors_are_per_session_and_stage():
    first = {}
    second = {}
    assert not repeated_stage_error(first, 's1', 'missing clock')
    assert not repeated_stage_error(second, 's1', 'missing clock')
    assert not repeated_stage_error(first, 's2', 'missing clock')
    assert repeated_stage_error(first, 's1', 'missing clock')
