#!/usr/bin/env python3
"""Batch venue scraper -- reads an Excel file (venue name, URL) and scrapes each venue."""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

import openpyxl

import venue_scraper as vs

logger = logging.getLogger("batch_scraper")


# ---------------------------------------------------------------------------
# Excel reader
# ---------------------------------------------------------------------------

def read_venue_list(path: str) -> list[tuple[str, str]]:
    """
    Read (venue_name, url) pairs from an Excel file.

    Expects column A = venue name, column B = website URL.
    Rows where the URL column doesn't start with 'http' are skipped
    automatically, so a header row is harmlessly ignored.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    venues: list[tuple[str, str]] = []

    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if len(row) < 2:
            logger.warning("Row %d: fewer than 2 columns, skipping", i)
            continue

        name, url = row[0], row[1]

        # Skip completely blank rows
        if not name and not url:
            continue

        url_str = str(url).strip() if url else ""
        name_str = str(name).strip() if name else ""

        if not url_str or not url_str.startswith("http"):
            logger.warning("Row %d (%s): no valid URL (%r), skipping", i, name_str, url_str)
            continue

        venues.append((name_str, url_str))

    wb.close()
    return venues


# ---------------------------------------------------------------------------
# Per-venue scraping
# ---------------------------------------------------------------------------

async def scrape_venue(
    fetcher: vs.PageFetcher,
    venue_name: str,
    url: str,
    days: Optional[int],
    use_llm: bool,
    ask_address: bool = False,
) -> list[dict]:
    """Run the full scraping pipeline for a single venue. Returns formatted events."""
    logger.info("Scraping: %s  <%s>", venue_name, url)
    try:
        finder = vs.EventPageFinder()
        html, events_url, homepage_html = await finder.find_events_page(fetcher, url, use_llm=use_llm)

        extractor = vs.EventExtractor()
        result = await extractor.extract(html, events_url, use_llm=use_llm, homepage_html=homepage_html)

        events = result.events
        if days is not None:
            events = vs.filter_events_by_date(events, days)

        # If the page didn't expose a venue name, use the one from the spreadsheet
        if venue_name and not result.venue.name:
            result.venue.name = venue_name

        # Prompt for address if not found on the page
        if not result.venue.address:
            if ask_address:
                print(f"\n  Address not found for: {venue_name or url}", file=sys.stderr)
                user_input = input("  Enter venue address (or press Enter to skip): ").strip()
                if user_input:
                    result.venue.address = user_input
            else:
                logger.warning("Address not found for %s -- rerun with --ask-address to provide it interactively", venue_name or url)

        # Geocode once per venue
        lat, lon = None, None
        if result.venue.address:
            lat, lon = vs.geocode_address(result.venue.address)

        formatted: list[dict] = []
        for e in events:
            ev = vs.format_output_event(e, result.venue, url)
            ev["latitude"] = lat
            ev["longitude"] = lon
            formatted.append(ev)

        logger.info("  -> %d event(s) found", len(formatted))
        return formatted

    except Exception as exc:
        logger.error("Failed to scrape %s (%s): %s", venue_name, url, exc)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-scrape venues listed in an Excel file (columns: venue name, URL).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python batch_scraper.py venues.xlsx
  python batch_scraper.py venues.xlsx -d 14 -o events.json -v
  python batch_scraper.py venues.xlsx --no-llm -o events.json
        """,
    )
    parser.add_argument("excel", metavar="FILE",
                        help="Excel file (.xlsx) with venue name in column A and URL in column B")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write JSON output to file (default: stdout)")
    parser.add_argument("-d", "--days", type=int, default=None, metavar="N",
                        help="Only include events within the next N days (default: all)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM fallback (rule-based extraction only)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress and debug info to stderr")
    parser.add_argument("--ask-address", action="store_true",
                        help="Interactively prompt for venue address when it cannot be found on the page")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, stream=sys.stderr, format="%(levelname)s: %(message)s")

    use_llm = not args.no_llm
    if use_llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning(
                "ANTHROPIC_API_KEY not set. LLM fallback will fail if triggered. "
                "Use --no-llm to disable."
            )

    venues = read_venue_list(args.excel)
    if not venues:
        print("No valid venue rows found in the Excel file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(venues)} venue(s) to scrape.", file=sys.stderr)

    fetcher = vs.PageFetcher()
    await fetcher.start()

    all_events: list[dict] = []
    try:
        for i, (venue_name, url) in enumerate(venues, start=1):
            print(f"[{i}/{len(venues)}] {venue_name or url}", file=sys.stderr)
            events = await scrape_venue(fetcher, venue_name, url, args.days, use_llm, ask_address=args.ask_address)
            all_events.extend(events)
    finally:
        await fetcher.stop()

    json_str = json.dumps({"events": all_events}, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"Wrote {len(all_events)} event(s) to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    if not all_events:
        print("WARNING: No events found across any venue. Try -v for details.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
