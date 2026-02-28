# Venue Event Scraper

Scrape upcoming events from venue websites and export them as structured JSON. Supports live music, trivia nights, DJ sets, comedy shows, and any other scheduled happenings.

## Features

- **Multi-strategy extraction** — tries JSON-LD, microdata, WordPress event plugins, generic HTML pattern detection, and an LLM fallback in order
- **Smart event-page navigation** — automatically follows links to find a venue's calendar page
- **Structured descriptions** — builds clean 2–4 sentence descriptions via an intermediate enrichment step (pricing, doors/show times, links deduplicated)
- **Event classification** — labels events as live music, trivia, DJ night, dance, open mic, karaoke, comedy, food, or sport
- **Geocoding** — resolves venue addresses to lat/lon via OpenStreetMap Nominatim
- **Date filtering** — optional window to only return events within the next N days
- **Batch mode** — feed an Excel file of venue names + URLs and scrape them all in one run

## Requirements

Python 3.10+

```bash
pip install playwright beautifulsoup4 lxml python-dateutil anthropic openpyxl
playwright install chromium
```

The LLM fallback and navigation assistant use the Anthropic API. Set your key before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Pass `--no-llm` to skip all API calls and use rule-based extraction only.

## Usage

### Single venue — `venue_scraper.py`

```bash
python venue_scraper.py <URL> [options]
```

| Flag | Description |
|------|-------------|
| `-d N` / `--days N` | Only return events within the next N days |
| `-o FILE` / `--output FILE` | Write JSON to a file instead of stdout |
| `--no-llm` | Disable LLM fallback (rule-based extraction only) |
| `-v` / `--verbose` | Print progress and debug info to stderr |

**Examples**

```bash
# Print all upcoming events to stdout
python venue_scraper.py https://www.emptybottle.com

# Events in the next 10 days, saved to a file
python venue_scraper.py https://www.emptybottle.com -d 10 -o events.json

# No API calls, verbose output
python venue_scraper.py https://www.catscradle.com --no-llm -v
```

### Batch scraping — `batch_scraper.py`

Reads an Excel file (`.xlsx`) where **column A** is the venue name and **column B** is the website URL. Header rows are automatically skipped.

```bash
python batch_scraper.py <FILE> [options]
```

| Flag | Description |
|------|-------------|
| `-d N` / `--days N` | Only return events within the next N days |
| `-o FILE` / `--output FILE` | Write JSON to a file instead of stdout |
| `--no-llm` | Disable LLM fallback |
| `-v` / `--verbose` | Print per-venue progress to stderr |

**Excel format**

| A (Venue Name) | B (URL) |
|----------------|---------|
| Empty Bottle | https://www.emptybottle.com |
| Cat's Cradle | https://www.catscradle.com |

**Examples**

```bash
# Scrape all venues and print combined JSON
python batch_scraper.py venues.xlsx

# Events in the next 2 weeks, saved to a file, with progress output
python batch_scraper.py venues.xlsx -d 14 -o events.json -v
```

## Output format

Both scripts output a JSON object with an `events` array:

```json
{
  "events": [
    {
      "title": "Live Show: Horsegirl",
      "description": "Indie rock trio Horsegirl performs original material. Doors open at 19:00, show starts at 20:00. Admission: $15 advance / $18 at the door. Tickets: https://...",
      "category": "music",
      "address": "1035 N Western Ave, Chicago, IL 60622",
      "latitude": 41.9007,
      "longitude": -87.6876,
      "date": "2026-03-15",
      "time": "20:00",
      "isPublic": true
    }
  ]
}
```

**`category`** is one of: `music`, `trivia`, `food`, `sport`, `other`

## How it works

For each venue URL the scraper:

1. **Finds the events page** — checks if the landing page already has events; otherwise scores all links by keyword relevance and follows the best candidate. Falls back to an LLM link-picker if heuristics fail.
2. **Extracts events** using five strategies in priority order:
   1. JSON-LD / Schema.org structured data
   2. HTML microdata (`itemtype="schema.org/Event"`)
   3. WordPress event plugin selectors (The Events Calendar, Events Manager)
   4. Generic repeated-card detection keyed on date patterns
   5. LLM fallback — sends page text to Claude and parses the response
3. **Enriches descriptions** — raw scraped data is normalised into an intermediate JSON (pricing split into advance/day-of, links deduplicated, times separated) before a clean 2–4 sentence description is composed.
4. **Geocodes** the venue address once per venue via Nominatim.
