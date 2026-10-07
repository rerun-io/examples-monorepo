"""Conversion policy outcomes shared by parsers and command entry points."""


class SkipSession(ValueError):
    """A source excluded by the parser's supported-history policy."""
