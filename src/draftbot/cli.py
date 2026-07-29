"""Command-line entry point. Subcommands are registered lazily to keep startup fast."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="draftbot", description="MTG draft transformer toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download 17lands draft data")
    fetch.set_defaults(entry="draftbot.data.fetch")

    args, rest = parser.parse_known_args(argv)
    if args.command == "fetch":
        from draftbot.data import fetch as mod
        return mod.main(rest)
    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
