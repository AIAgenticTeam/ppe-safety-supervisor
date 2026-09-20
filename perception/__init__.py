"""Lane A: the deterministic perception core -- zones, assessment, tracking, events.

Deliberately imports nothing. Every module here is also a CLI (`python -m perception.zones`),
and an `__init__` that pulled its submodules in would make runpy warn about them.
"""
