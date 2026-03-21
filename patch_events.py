#!/usr/bin/env python3
"""Interactive utility to fix addresses in a scraped events.json file."""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set or correct the venue address for all events in an events JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python patch_events.py events.json
  python patch_events.py events.json -o events_fixed.json
        """,
    )
    parser.add_argument("input", metavar="FILE", help="events.json file to patch")
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="Write patched JSON to this file instead of overwriting input",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    if not events:
        print("No events found in file.", file=sys.stderr)
        sys.exit(0)

    current_addr = events[0].get("address") or "(none)"
    print(f"Current address: {current_addr}")

    try:
        reply = input("New address (Enter to keep current): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        sys.exit(0)

    if not reply:
        print("No changes made.", file=sys.stderr)
        sys.exit(0)

    for event in events:
        event["address"] = reply

    output_path = args.output or args.input
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated address for {len(events)} event(s) to: {reply}", file=sys.stderr)
    print(f"Saved to {output_path}.", file=sys.stderr)


if __name__ == "__main__":
    main()
