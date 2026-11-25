"""Allow `python -m svdt ...` in addition to the installed `svdt` entry point."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
