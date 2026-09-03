"""Test package.

Exists so shared helpers — notably `tests.normalize`, which `docs/04-SCHEMAS.md` §3
designates as the sole owner of the non-deterministic-field strip list — can be
imported from more than one test module. Library code never imports from here.
"""
