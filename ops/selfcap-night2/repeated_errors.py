"""Per-session, exact-message retry accounting."""
def repeated_stage_error(info: dict, stage: str, message: str) -> bool:
    counts = info.setdefault(stage + '_error_counts', {})
    counts[message] = counts.get(message, 0) + 1
    return counts[message] >= 2


def is_soma_input_limitation(returncode: int, output: str) -> bool:
    return returncode == 20 and any(line.startswith('SOMA_INPUT_LIMITATION ') for line in output.splitlines())
