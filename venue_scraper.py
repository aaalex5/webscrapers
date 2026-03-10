#!/usr/bin/env python3
"""Venue event scraper -- extracts events from venue websites."""

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
from urllib.parse import urljoin, urlparse, quote_plus
from urllib.request import urlopen, Request

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup, Tag
from dateutil import parser as dateparser

from event_classifier import classify_event, FINAL_CATEGORY_MAP

logger = logging.getLogger("venue_scraper")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Event:
    event_name: Optional[str] = None
    event_type: Optional[str] = None  # "live_music", "trivia", "dj_night", "dance", "open_mic", "karaoke", "comedy", "other"
    date: Optional[str] = None
    time: Optional[str] = None
    doors_time: Optional[str] = None
    description: Optional[str] = None
    ticket_url: Optional[str] = None
    price: Optional[str] = None


@dataclass
class Venue:
    name: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ExtractionResult:
    venue: Venue
    events: list[Event]
    extraction_method: Optional[str] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALENDAR_KEYWORDS = [
    "events", "calendar", "shows", "schedule", "lineup",
    "upcoming", "concerts", "gigs", "whats-on", "what-s-on",
    "live-music", "performances", "tour", "tickets",
]

DATE_PATTERNS = [
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:[,\s]+\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),
    re.compile(
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[,\s]+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b", re.IGNORECASE),
]

TIME_PATTERN = re.compile(
    r"\b(\d{1,2}:\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?|\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.))\b",
    re.IGNORECASE,
)
DOORS_PATTERN = re.compile(r"doors?\s*(?:@|at|:)?\s*(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)", re.IGNORECASE)

# Prefixes applied to event titles based on event_type
TITLE_PREFIX_MAP = {
    "live_music": "Live Show: ",
    "dj_night": "DJ Night: ",
    "open_mic": "Open Mic: ",
    "dance": "Dance Night: ",
    "karaoke": "Karaoke Night: ",
    "trivia": "Trivia Night: ",
    "food": "Food Event: ",
    "sport": "Sports Night: ",
    "comedy": "Comedy Night: ",
    "other": "",
}


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

EXTRACTION_PROMPT = """You are extracting event data from a venue website. This includes ALL types of events: live music/concerts, trivia nights, DJ sets, dance nights, open mic, karaoke, comedy shows, and any other scheduled happenings.

URL: {url}

Below is the text content of the page. Extract ALL upcoming events and return them as a JSON array.

Each event object must have these fields (use null if not found):
- "event_name": string (name of the event, band, performer, or activity — e.g. "Trivia Night", "DJ Spinmaster", "Horsegirl")
- "event_type": string (one of: "live_music", "trivia", "dj_night", "dance", "open_mic", "karaoke", "comedy", "other")
- "date": string (YYYY-MM-DD format)
- "time": string (HH:MM in 24h format, or null)
- "doors_time": string (HH:MM in 24h format, or null)
- "description": string (brief description, max 200 chars, or null)
- "ticket_url": string (full URL, or null)
- "price": string (e.g. "$15", "Free", or null)

Also extract venue info and return it as a top-level object:
- "venue_name": string
- "venue_address": string (full street address, or null)

Return ONLY valid JSON in this exact format, no other text:
{{"venue_name": "...", "venue_address": "...", "events": [...]}}

Page content:
{page_text}"""

NAV_PROMPT = """Below is a list of links found on a venue's website. Which link most likely leads to a page listing upcoming events (live music, trivia, DJ nights, dance nights, or any other scheduled happenings)?

Links:
{links_text}

Return ONLY the URL of the best matching link, nothing else."""


# ---------------------------------------------------------------------------
# PageFetcher
# ---------------------------------------------------------------------------

class PageFetcher:
    """Manages Playwright browser lifecycle and page loading."""

    def __init__(self):
        self._playwright = None
        self._browser = None

    async def start(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def fetch(self, url: str) -> tuple[str, str]:
        """Fetch a URL and return (rendered_html, final_url)."""
        context = await self._browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()
        try:
            try:
                await page.goto(url, wait_until="networkidle", timeout=15000)
            except PlaywrightTimeout:
                logger.warning("networkidle timed out for %s, falling back to domcontentloaded", url)
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            # Extra wait for any lazy-loaded content
            await page.wait_for_timeout(2000)
            html = await page.content()
            final_url = page.url
            return html, final_url
        finally:
            await context.close()


# ---------------------------------------------------------------------------
# EventPageFinder
# ---------------------------------------------------------------------------

class EventPageFinder:
    """Given a starting URL, find the page that lists events."""

    def _page_has_events(self, soup: BeautifulSoup) -> bool:
        """Quick check: does this page appear to contain event listings?"""
        # Check JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if self._jsonld_has_events(data):
                    return True
            except (json.JSONDecodeError, TypeError):
                continue

        # Check for known event plugin selectors
        event_selectors = [
            ".tribe-events", ".tribe-events-calendar",
            ".em-events-list", ".eventlist",
            "[itemtype*='schema.org/Event']",
            "[itemtype*='schema.org/MusicEvent']",
            "a.fc-event[aria-label]",        # FullCalendar (TicketWeb, etc.)
            ".tw-calendar-event-title",       # TicketWeb specific
        ]
        for sel in event_selectors:
            if soup.select_one(sel):
                return True

        return False

    def _jsonld_has_events(self, data) -> bool:
        if isinstance(data, list):
            return any(self._jsonld_has_events(item) for item in data)
        if isinstance(data, dict):
            t = data.get("@type", "")
            if isinstance(t, list):
                types = t
            else:
                types = [t]
            if any(tp in ("Event", "MusicEvent", "DanceEvent", "Festival") for tp in types):
                return True
            if "@graph" in data:
                return self._jsonld_has_events(data["@graph"])
        return False

    def _score_link(self, text: str, href: str, base_domain: str) -> int:
        """Score a link by likelihood of leading to an events page."""
        text_lower = (text or "").lower().strip()
        href_lower = (href or "").lower()
        score = 0

        parsed = urlparse(href)
        link_domain = parsed.netloc.replace("www.", "")

        # Penalize off-site links
        if link_domain and link_domain != base_domain:
            score -= 3
        # Penalize social media
        if any(s in href_lower for s in ["facebook", "instagram", "twitter", "youtube", "spotify", "tiktok"]):
            score -= 3

        path_segments = parsed.path.lower().strip("/").split("/")

        for keyword in CALENDAR_KEYWORDS:
            if text_lower == keyword:
                score += 4
            elif keyword in text_lower.split():
                score += 3
            elif keyword in text_lower:
                score += 1
            if keyword in path_segments:
                score += 3
            elif keyword in href_lower:
                score += 1

        return score

    async def find_events_page(self, fetcher: PageFetcher, start_url: str, use_llm: bool = True) -> tuple[str, str, Optional[str]]:
        """Return (events_html, events_url, homepage_html) of the events page.

        homepage_html is the original start-page HTML when we navigated to a
        different events sub-page, so callers can still extract venue info from
        it.  It is None when the start page itself already contained events.
        """
        logger.info("Fetching starting page: %s", start_url)
        html, final_url = await fetcher.fetch(start_url)
        soup = BeautifulSoup(html, "lxml")

        # Check if current page already has events
        if self._page_has_events(soup):
            logger.info("Current page already contains events")
            return html, final_url, None

        # Preserve homepage HTML for venue info extraction
        homepage_html = html

        # Score all links
        base_domain = urlparse(final_url).netloc.replace("www.", "")
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(final_url, a["href"])
            text = a.get_text(strip=True)
            score = self._score_link(text, href, base_domain)
            links.append((score, text, href))

        links.sort(key=lambda x: x[0], reverse=True)

        # Try top-scoring links (up to 2 hops)
        tried_urls = {final_url}
        for score, text, href in links[:5]:
            if score < 2:
                break
            if href in tried_urls:
                continue
            tried_urls.add(href)

            logger.info("Following link (score=%d): %s -> %s", score, text, href)
            try:
                hop_html, hop_url = await fetcher.fetch(href)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", href, e)
                continue

            hop_soup = BeautifulSoup(hop_html, "lxml")
            if self._page_has_events(hop_soup):
                logger.info("Found events page at: %s", hop_url)
                return hop_html, hop_url, homepage_html

            # Even if no structured event data, a high-scoring link is likely right
            if score >= 3:
                logger.info("Using high-scoring link as events page: %s", hop_url)
                return hop_html, hop_url, homepage_html

        # LLM fallback for finding events page
        if use_llm and links:
            logger.info("Using LLM to identify events page link")
            try:
                llm_url = await self._llm_find_link(links[:20], final_url)
                if llm_url and llm_url not in tried_urls:
                    hop_html, hop_url = await fetcher.fetch(llm_url)
                    logger.info("LLM suggested events page: %s", hop_url)
                    return hop_html, hop_url, homepage_html
            except Exception as e:
                logger.warning("LLM nav fallback failed: %s", e)

        # Fall back to original page (homepage_html is the same page, no separate homepage)
        logger.warning("Could not find a dedicated events page, using starting page")
        return html, final_url, None

    async def _llm_find_link(self, links: list, base_url: str) -> Optional[str]:
        import anthropic
        links_text = "\n".join(f"- [{text}]({href})" for _, text, href in links)
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": NAV_PROMPT.format(links_text=links_text)}],
        )
        url = message.content[0].text.strip()
        # Validate it looks like a URL
        if url.startswith("http"):
            return url
        return None


# ---------------------------------------------------------------------------
# Extraction Strategies
# ---------------------------------------------------------------------------

PST = ZoneInfo("America/Los_Angeles")


def _to_pst(dt):
    """Convert a datetime to PST/PDT if it has timezone info."""
    if dt.tzinfo is not None:
        return dt.astimezone(PST)
    return dt


def _parse_date(text: str) -> Optional[str]:
    """Try to parse a date string into YYYY-MM-DD format (in PST)."""
    try:
        dt = dateparser.parse(text, fuzzy=True)
        if dt:
            dt = _to_pst(dt)
            return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        pass
    return None


def _parse_time(text: str) -> Optional[str]:
    """Try to parse a time string into HH:MM 24h format (in PST).

    Only returns a value when the text contains an explicit time indicator
    (e.g. '7pm', '19:30', 'T19:30') to avoid false positives from date-only
    strings that dateparser would resolve to midnight.
    """
    if not re.search(
        r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|T\d{2}:\d{2}|\d{2}:\d{2}",
        text,
        re.IGNORECASE,
    ):
        return None
    try:
        dt = dateparser.parse(text, fuzzy=True)
        if dt:
            dt = _to_pst(dt)
            return dt.strftime("%H:%M")
    except (ValueError, OverflowError):
        pass
    return None


def _parse_iso_datetime(text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse an ISO 8601 datetime string directly (no fuzzy matching).

    Returns (YYYY-MM-DD, HH:MM) for full datetimes, or (YYYY-MM-DD, None)
    for date-only strings.  Returns (None, None) if parsing fails.
    """
    try:
        normalized = text.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        dt = _to_pst(dt)
        # No 'T' separator means date-only; don't emit a spurious 00:00 time
        if "T" not in text:
            return dt.strftime("%Y-%m-%d"), None
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except (ValueError, AttributeError):
        return None, None


def _clean_text(text: str) -> str:
    """Clean whitespace from extracted text."""
    return re.sub(r"\s+", " ", text).strip()


# --- Strategy 1: JSON-LD / Schema.org ---

def extract_jsonld(soup: BeautifulSoup) -> tuple[list[Event], Venue]:
    """Extract events from JSON-LD structured data."""
    events = []
    venue = Venue()

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        _process_jsonld(data, events, venue)

    return events, venue


def _extract_venue_from_jsonld_schema(data, venue: Venue):
    """Extract venue name/address from LocalBusiness/Place/Organization JSON-LD schemas."""
    if isinstance(data, list):
        for item in data:
            _extract_venue_from_jsonld_schema(item, venue)
        return
    if not isinstance(data, dict):
        return

    if "@graph" in data:
        _extract_venue_from_jsonld_schema(data["@graph"], venue)

    t = data.get("@type", "")
    types = t if isinstance(t, list) else [t]

    business_types = {
        "LocalBusiness", "FoodEstablishment", "BarOrPub", "NightClub",
        "MusicVenue", "EntertainmentBusiness", "Organization", "Place",
        "CivicStructure", "Restaurant",
    }
    if any(tp in business_types for tp in types):
        if not venue.name:
            venue.name = data.get("name")
        if not venue.address:
            address = data.get("address")
            if isinstance(address, dict):
                parts = [
                    address.get("streetAddress", ""),
                    address.get("addressLocality", ""),
                    address.get("addressRegion", ""),
                    address.get("postalCode", ""),
                ]
                joined = ", ".join(p for p in parts if p)
                if joined:
                    venue.address = joined
            elif isinstance(address, str) and address:
                venue.address = address


def _process_jsonld(data, events: list[Event], venue: Venue):
    if isinstance(data, list):
        for item in data:
            _process_jsonld(item, events, venue)
        return

    if not isinstance(data, dict):
        return

    # Handle @graph
    if "@graph" in data:
        _process_jsonld(data["@graph"], events, venue)

    t = data.get("@type", "")
    types = t if isinstance(t, list) else [t]

    if any(tp in ("Event", "MusicEvent", "DanceEvent", "Festival", "TheaterEvent") for tp in types):
        event = Event()

        # Artist/name
        performer = data.get("performer") or data.get("performers")
        if performer:
            if isinstance(performer, list):
                names = [p.get("name", "") if isinstance(p, dict) else str(p) for p in performer]
                event.event_name = ", ".join(n for n in names if n)
            elif isinstance(performer, dict):
                event.event_name = performer.get("name")
            else:
                event.event_name = str(performer)
        if not event.event_name:
            event.event_name = data.get("name")

        # Date/time -- prefer direct ISO parsing; fall back to fuzzy for non-standard strings
        start = data.get("startDate", "")
        if start:
            iso_date, iso_time = _parse_iso_datetime(start)
            if iso_date:
                event.date = iso_date
                event.time = iso_time
            else:
                event.date = _parse_date(start)
                event.time = _parse_time(start)

        door_time = data.get("doorTime", "")
        if door_time:
            _, iso_doors = _parse_iso_datetime(door_time)
            event.doors_time = iso_doors or _parse_time(door_time)

        # Description
        desc = data.get("description", "")
        if desc:
            event.description = _clean_text(desc)[:500]

        # Tickets
        offers = data.get("offers")
        if isinstance(offers, dict):
            event.ticket_url = offers.get("url")
            price = offers.get("price")
            currency = offers.get("priceCurrency", "")
            if price is not None:
                event.price = f"${price}" if currency == "USD" else str(price)
        elif isinstance(offers, list) and offers:
            offer = offers[0]
            if isinstance(offer, dict):
                event.ticket_url = offer.get("url")
                price = offer.get("price")
                if price is not None:
                    event.price = f"${price}" if offer.get("priceCurrency") == "USD" else str(price)

        # Venue info from location
        location = data.get("location")
        if isinstance(location, dict):
            if not venue.name:
                venue.name = location.get("name")
            address = location.get("address")
            if address and not venue.address:
                if isinstance(address, dict):
                    parts = [
                        address.get("streetAddress", ""),
                        address.get("addressLocality", ""),
                        address.get("addressRegion", ""),
                        address.get("postalCode", ""),
                    ]
                    venue.address = ", ".join(p for p in parts if p)
                elif isinstance(address, str):
                    venue.address = address

        if event.event_name and event.date:
            event.event_type = classify_event(event.event_name or "", event.description or "").internal_category
            events.append(event)


# --- Strategy 2: Microdata ---

def extract_microdata(soup: BeautifulSoup) -> tuple[list[Event], Venue]:
    """Extract events from HTML microdata."""
    events = []
    venue = Venue()

    event_elements = soup.find_all(attrs={"itemtype": re.compile(r"schema\.org/(Music)?Event", re.IGNORECASE)})

    for el in event_elements:
        event = Event()

        name_el = el.find(attrs={"itemprop": "name"})
        performer_el = el.find(attrs={"itemprop": "performer"})
        event.event_name = (performer_el or name_el or Tag(name="span")).get_text(strip=True) or None

        start_el = el.find(attrs={"itemprop": "startDate"})
        if start_el:
            dt_str = start_el.get("datetime") or start_el.get("content") or start_el.get_text(strip=True)
            iso_date, iso_time = _parse_iso_datetime(dt_str)
            if iso_date:
                event.date = iso_date
                event.time = iso_time
            else:
                event.date = _parse_date(dt_str)
                event.time = _parse_time(dt_str)

        door_el = el.find(attrs={"itemprop": "doorTime"})
        if door_el:
            dt_str = door_el.get("datetime") or door_el.get_text(strip=True)
            _, iso_doors = _parse_iso_datetime(dt_str)
            event.doors_time = iso_doors or _parse_time(dt_str)

        desc_el = el.find(attrs={"itemprop": "description"})
        if desc_el:
            event.description = _clean_text(desc_el.get_text(strip=True))[:500]

        url_el = el.find(attrs={"itemprop": "url"})
        if url_el:
            event.ticket_url = url_el.get("href") or url_el.get_text(strip=True)

        price_el = el.find(attrs={"itemprop": "price"})
        if price_el:
            event.price = price_el.get("content") or price_el.get_text(strip=True)

        # Venue from location
        loc_el = el.find(attrs={"itemprop": "location"})
        if loc_el and not venue.name:
            vname = loc_el.find(attrs={"itemprop": "name"})
            if vname:
                venue.name = vname.get_text(strip=True)
            addr = loc_el.find(attrs={"itemprop": "address"})
            if addr:
                venue.address = _clean_text(addr.get_text(strip=True))

        if event.event_name and event.date:
            event.event_type = classify_event(event.event_name or "", event.description or "").internal_category
            events.append(event)

    return events, venue


# --- Strategy 3: WordPress Plugin Heuristics ---

def extract_wp_plugins(soup: BeautifulSoup) -> list[Event]:
    """Extract events from known WordPress event plugin HTML structures."""
    events = []

    # The Events Calendar (Tribe)
    tribe_container = soup.select_one(".tribe-events, .tribe-common")
    if tribe_container:
        for item in tribe_container.select(
            ".tribe-events-calendar-list__event-row, "
            ".tribe-common-g-row, "
            ".type-tribe_events, "
            ".tribe-events-list .tribe-events-loop .type-tribe_events"
        ):
            event = Event()

            title_el = item.select_one(
                ".tribe-events-calendar-list__event-title a, "
                ".tribe-events-list-event-title a, "
                ".tribe-event-url a, "
                "h2 a, h3 a"
            )
            if title_el:
                event.event_name = _clean_text(title_el.get_text(strip=True))

            time_el = item.select_one("time[datetime], .tribe-event-date-start")
            if time_el:
                dt_str = time_el.get("datetime") or time_el.get_text(strip=True)
                event.date = _parse_date(dt_str)
                event.time = _parse_time(dt_str)

            desc_el = item.select_one(".tribe-events-calendar-list__event-description, .tribe-events-list-event-description")
            if desc_el:
                event.description = _clean_text(desc_el.get_text(strip=True))[:500]

            if event.event_name and event.date:
                event.event_type = classify_event(event.event_name or "", event.description or "").internal_category
                events.append(event)

        if events:
            return events

    # Events Manager plugin
    em_container = soup.select_one(".em-events-list, #em-events-list")
    if em_container:
        for item in em_container.select(".event, .em-item"):
            event = Event()
            title_el = item.select_one(".event-title a, h2 a, h3 a")
            if title_el:
                event.event_name = _clean_text(title_el.get_text(strip=True))
            date_el = item.select_one(".event-date, time[datetime]")
            if date_el:
                dt_str = date_el.get("datetime") or date_el.get_text(strip=True)
                event.date = _parse_date(dt_str)
                event.time = _parse_time(dt_str)
            if event.event_name and event.date:
                event.event_type = classify_event(event.event_name or "", event.description or "").internal_category
                events.append(event)

    return events


# --- Strategy 3.5: FullCalendar / TicketWeb ---

def extract_fullcalendar(soup: BeautifulSoup) -> list[Event]:
    """Extract events from FullCalendar-based widgets (TicketWeb, etc.).

    Handles the pattern:
      <a class="fc-event" aria-label="EventName|YYYY-MM-DD|HH:MM AM/PM">
        <div class="tw-calendar-event-title">...</div>
        <span class="tw-calendar-event-doors">Doors: 7:00 PM</span>
        <span class="tw-calendar-event-time">Show: 8:00 PM</span>
      </a>
    The parent <td data-date="YYYY-MM-DD"> is used as a date fallback.
    """
    events = []

    fc_event_links = soup.select("a.fc-event[aria-label]")
    if not fc_event_links:
        return events

    for a_el in fc_event_links:
        event = Event()

        # Primary: parse the pipe-delimited aria-label "Name|YYYY-MM-DD|HH:MM AM/PM"
        aria = a_el.get("aria-label", "")
        parts = [p.strip() for p in aria.split("|")]
        if parts:
            event.event_name = parts[0] or None
        if len(parts) >= 2:
            iso_date, _ = _parse_iso_datetime(parts[1])
            event.date = iso_date or _parse_date(parts[1])
        if len(parts) >= 3:
            event.time = _parse_time(parts[2])

        # Refine name from explicit title element if present
        title_el = a_el.select_one(".tw-calendar-event-title, .fc-event-title")
        if title_el:
            event.event_name = _clean_text(title_el.get_text(strip=True)) or event.event_name

        # Doors/show times from dedicated spans (more reliable than aria-label time)
        doors_el = a_el.select_one(".tw-calendar-event-doors")
        if doors_el:
            event.doors_time = _parse_time(doors_el.get_text(strip=True))

        show_el = a_el.select_one(".tw-calendar-event-time")
        if show_el:
            event.time = _parse_time(show_el.get_text(strip=True))

        # Fallback date from parent <td data-date="...">
        if not event.date:
            parent_td = a_el.find_parent("td", attrs={"data-date": True})
            if parent_td:
                data_date = parent_td.get("data-date", "")
                iso_date, _ = _parse_iso_datetime(data_date)
                event.date = iso_date or _parse_date(data_date)

        if event.event_name and event.date:
            event.event_type = classify_event(event.event_name, event.description or "").internal_category
            events.append(event)

    return events


# --- Strategy 4: Generic HTML Pattern Detection ---

def extract_generic_html(soup: BeautifulSoup) -> list[Event]:
    """Detect repeated HTML structures containing date patterns."""
    events = []

    # Remove nav, header, footer to reduce noise
    for tag_name in ["nav", "header", "footer", "aside"]:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Find all text nodes matching date patterns
    body = soup.find("body") or soup
    date_elements = []

    for text_node in body.find_all(string=True):
        text = str(text_node).strip()
        if not text:
            continue
        for pattern in DATE_PATTERNS:
            if pattern.search(text):
                date_elements.append((text_node, text))
                break

    if not date_elements:
        return events

    # For each date match, walk up to find the event "card" container
    card_candidates = []
    for text_node, _ in date_elements:
        el = text_node.parent
        # Walk up to find a block-level container that has siblings of same type
        for _ in range(8):
            if el is None or el.name in ("body", "html", "[document]"):
                break
            if el.name in ("div", "li", "article", "section", "tr"):
                siblings = [s for s in (el.parent or el).children if isinstance(s, Tag) and s.name == el.name]
                if len(siblings) >= 3:
                    card_candidates.append(el)
                    break
            el = el.parent

    # Deduplicate cards
    seen = set()
    unique_cards = []
    for card in card_candidates:
        card_id = id(card)
        if card_id not in seen:
            seen.add(card_id)
            unique_cards.append(card)

    for card in unique_cards:
        event = Event()
        card_text = card.get_text(separator=" ", strip=True)

        # Extract date
        for pattern in DATE_PATTERNS:
            match = pattern.search(card_text)
            if match:
                event.date = _parse_date(match.group())
                break

        # Extract time
        time_matches = TIME_PATTERN.findall(card_text)
        if time_matches:
            event.time = _parse_time(time_matches[0])
            if len(time_matches) > 1:
                event.doors_time = _parse_time(time_matches[0])
                event.time = _parse_time(time_matches[1])

        # Extract doors time specifically
        doors_match = DOORS_PATTERN.search(card_text)
        if doors_match:
            event.doors_time = _parse_time(doors_match.group(1))

        # Extract artist: prefer headings or prominent links
        artist_el = card.find(["h1", "h2", "h3", "h4"])
        if not artist_el:
            # Try the first link that isn't a "buy tickets" type link
            for a in card.find_all("a"):
                a_text = a.get_text(strip=True)
                if a_text and not re.search(r"ticket|buy|rsvp|more\s+info", a_text, re.IGNORECASE):
                    artist_el = a
                    break

        if artist_el:
            event.event_name = _clean_text(artist_el.get_text(strip=True))

        # Find ticket link
        for a in card.find_all("a", href=True):
            a_text = (a.get_text(strip=True) + " " + (a.get("href") or "")).lower()
            if any(kw in a_text for kw in ["ticket", "buy", "rsvp", "eventbrite", "dice.fm"]):
                event.ticket_url = a.get("href")
                break

        # Description: remaining text, minus the artist name
        desc_text = _clean_text(card_text)
        if event.event_name and event.event_name in desc_text:
            desc_text = desc_text.replace(event.event_name, "", 1).strip()
        if len(desc_text) > 20:
            event.description = desc_text[:500]

        if event.event_name and event.date:
            event.event_type = classify_event(event.event_name or "", event.description or "").internal_category
            events.append(event)

    return events


# --- Strategy 5: LLM Fallback ---

async def extract_llm(html: str, url: str) -> tuple[list[Event], Venue]:
    """Send page text to Claude API for extraction."""
    import anthropic

    soup = BeautifulSoup(html, "lxml")

    # Strip noise elements
    for tag_name in ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]:
        for el in soup.find_all(tag_name):
            el.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Truncate to ~80k chars to stay within token limits
    if len(text) > 80000:
        text = text[:80000]

    logger.info("Sending %d chars to LLM for extraction", len(text))

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(page_text=text, url=url)}],
    )

    response_text = message.content[0].text.strip()

    # Parse the JSON response
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            logger.warning("Could not parse LLM response as JSON")
            return [], Venue()

    venue = Venue(
        name=data.get("venue_name"),
        address=data.get("venue_address"),
    )

    events = []
    for item in data.get("events", []):
        event = Event(
            event_name=item.get("event_name"),
            event_type=item.get("event_type", "other"),
            date=item.get("date"),
            time=item.get("time"),
            doors_time=item.get("doors_time"),
            description=item.get("description"),
            ticket_url=item.get("ticket_url"),
            price=item.get("price"),
        )
        if event.event_name and event.date:
            events.append(event)

    return events, venue


# ---------------------------------------------------------------------------
# Venue Info Extraction
# ---------------------------------------------------------------------------

def extract_venue_info(soup: BeautifulSoup, url: str) -> Venue:
    """Extract venue name and address from the page."""
    venue = Venue(url=url)

    # Try JSON-LD for LocalBusiness/Organization schemas first (most reliable)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        _extract_venue_from_jsonld_schema(data, venue)
        if venue.name and venue.address:
            return venue

    # Try og:site_name
    og_name = soup.find("meta", property="og:site_name")
    if og_name:
        venue.name = og_name.get("content")

    # Try title tag
    if not venue.name:
        title = soup.find("title")
        if title:
            name = title.get_text(strip=True)
            # Strip common suffixes
            name = re.split(r"\s*[|\-–—]\s*(?:Events|Calendar|Shows|Schedule|Home|Welcome)", name, flags=re.IGNORECASE)[0]
            venue.name = _clean_text(name)

    # Try to find address
    addr_el = soup.find(attrs={"itemprop": "address"})
    if addr_el:
        venue.address = _clean_text(addr_el.get_text(strip=True))

    if not venue.address:
        # Look for common address patterns in the page
        addr_el = soup.find(class_=re.compile(r"address", re.IGNORECASE))
        if addr_el:
            venue.address = _clean_text(addr_el.get_text(strip=True))

    if not venue.address:
        # Try the semantic <address> HTML tag
        addr_tag = soup.find("address")
        if addr_tag:
            addr_text = _clean_text(addr_tag.get_text(strip=True))
            # Basic sanity check: should start with a street number
            if re.search(r"\d+\s+\w", addr_text):
                venue.address = addr_text

    if not venue.address:
        # Check footer for address
        footer = soup.find("footer")
        if footer:
            footer_text = footer.get_text(separator=" ", strip=True)
            # Look for street address pattern
            addr_match = re.search(
                r"\d+\s+[\w\s]+(?:St(?:reet)?|Ave(?:nue)?|Blvd|Rd|Dr|Way|Ln|Pl|Ct|Pike|Hwy)"
                r"[.,]?\s*[\w\s]*,?\s*[A-Z]{2}\s*\d{5}",
                footer_text,
            )
            if addr_match:
                venue.address = _clean_text(addr_match.group())

    return venue


# ---------------------------------------------------------------------------
# EventExtractor (Orchestrator)
# ---------------------------------------------------------------------------

class EventExtractor:
    """Runs extraction strategies in priority order."""

    async def extract(self, html: str, url: str, use_llm: bool = True, homepage_html: Optional[str] = None) -> ExtractionResult:
        soup = BeautifulSoup(html, "lxml")

        # Get venue info from page metadata
        venue = extract_venue_info(soup, url)

        # If the address is still missing, try the original homepage (which is more
        # likely to carry a LocalBusiness/Organization JSON-LD with the full address)
        if not venue.address and homepage_html:
            home_soup = BeautifulSoup(homepage_html, "lxml")
            home_venue = extract_venue_info(home_soup, url)
            venue = self._merge_venue(venue, home_venue)

        # Strategy 1: JSON-LD
        logger.info("Trying JSON-LD extraction...")
        events, jld_venue = extract_jsonld(soup)
        if events:
            logger.info("JSON-LD: found %d events", len(events))
            venue = self._merge_venue(venue, jld_venue)
            return ExtractionResult(venue=venue, events=self._dedup(events), extraction_method="json-ld")

        # Strategy 2: Microdata
        logger.info("Trying microdata extraction...")
        events, md_venue = extract_microdata(soup)
        if events:
            logger.info("Microdata: found %d events", len(events))
            venue = self._merge_venue(venue, md_venue)
            return ExtractionResult(venue=venue, events=self._dedup(events), extraction_method="microdata")

        # Strategy 3: WordPress plugins
        logger.info("Trying WordPress plugin extraction...")
        events = extract_wp_plugins(soup)
        if events:
            logger.info("WP plugins: found %d events", len(events))
            return ExtractionResult(venue=venue, events=self._dedup(events), extraction_method="html-heuristic")

        # Strategy 3.5: FullCalendar / TicketWeb
        logger.info("Trying FullCalendar extraction...")
        events = extract_fullcalendar(soup)
        if events:
            logger.info("FullCalendar: found %d events", len(events))
            return ExtractionResult(venue=venue, events=self._dedup(events), extraction_method="fullcalendar")

        # Strategy 4: Generic HTML
        logger.info("Trying generic HTML extraction...")
        events = extract_generic_html(BeautifulSoup(html, "lxml"))  # Fresh soup since Strategy 4 decomposes elements
        if len(events) >= 3:
            logger.info("Generic HTML: found %d events", len(events))
            return ExtractionResult(venue=venue, events=self._dedup(events), extraction_method="html-heuristic")
        elif events:
            logger.info("Generic HTML: found only %d events (below threshold), continuing...", len(events))

        # Strategy 5: LLM fallback
        if use_llm:
            logger.info("Trying LLM fallback extraction...")
            try:
                events, llm_venue = await extract_llm(html, url)
                if events:
                    logger.info("LLM: found %d events", len(events))
                    venue = self._merge_venue(venue, llm_venue)
                    return ExtractionResult(venue=venue, events=self._dedup(events), extraction_method="llm-fallback")
            except Exception as e:
                logger.warning("LLM extraction failed: %s", e)
        else:
            logger.info("LLM fallback disabled")

        # Nothing worked
        logger.warning("All extraction strategies failed")
        return ExtractionResult(venue=venue, events=[], extraction_method=None)

    def _merge_venue(self, base: Venue, other: Venue) -> Venue:
        """Merge venue info, preferring non-None values from other."""
        return Venue(
            name=other.name or base.name,
            address=other.address or base.address,
            url=base.url or other.url,
        )

    def _dedup(self, events: list[Event]) -> list[Event]:
        """Deduplicate events by (artist, date)."""
        seen = set()
        unique = []
        for e in events:
            key = ((e.event_name or "").lower().strip(), e.date)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def geocode_address(address: str) -> tuple[Optional[float], Optional[float]]:
    """Geocode an address using OpenStreetMap Nominatim. Returns (lat, lon) or (None, None)."""
    try:
        encoded = quote_plus(address)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", address, e)
    return None, None


def _parse_pricing(price_str: Optional[str]) -> dict:
    """Parse a raw price string into structured advance / day-of / notes components."""
    if not price_str:
        return {"advance": None, "day_of": None, "notes": None}

    price_lower = price_str.lower()

    if "free" in price_lower:
        return {"advance": "Free", "day_of": None, "notes": None}

    has_fees = bool(re.search(r"fee", price_lower))

    # Match advance price
    adv_match = re.search(r"\$?\s*([\d.]+)\s*(?:adv(?:ance)?|in\s*adv)", price_lower)
    # Match day-of / door price
    dos_match = re.search(r"\$?\s*([\d.]+)\s*(?:dos|door|day[- ]of)", price_lower)

    notes = "after fees" if has_fees else None

    if adv_match and dos_match:
        return {
            "advance": f"${adv_match.group(1)}",
            "day_of": f"${dos_match.group(1)}",
            "notes": notes,
        }
    if adv_match:
        return {"advance": f"${adv_match.group(1)}", "day_of": None, "notes": notes}
    if dos_match:
        return {"advance": None, "day_of": f"${dos_match.group(1)}", "notes": notes}

    # Single price — normalise to $X
    single = re.search(r"\$?\s*([\d.]+)", price_str)
    normalised = f"${single.group(1)}" if single else price_str
    return {"advance": normalised, "day_of": None, "notes": notes}


def _build_event_intermediate(event: Event, venue: Venue, venue_url: str) -> dict:
    """
    Build the intermediate enrichment JSON from raw scraped data.

    Schema:
      summary    – 1 sentence, plain language
      details    – 2-4 sentences, cleaned up
      pricing    – { advance, day_of, notes }
      doors_time – string | null
      show_time  – string | null
      links      – { tickets, venue, more_info }
      warnings   – string[]
    """
    warnings: list[str] = []

    # ── Pricing ──────────────────────────────────────────────────────────────
    pricing = _parse_pricing(event.price)
    if not event.price:
        warnings.append("no pricing info")

    # ── Links (deduplicate; prefer official ticketing link) ───────────────────
    ticket_link: Optional[str] = event.ticket_url or None
    venue_link: Optional[str] = venue_url or None
    # If ticket URL is the same as the venue URL, don't list it twice
    if ticket_link and venue_link and ticket_link.rstrip("/") == venue_link.rstrip("/"):
        venue_link = None
    if not ticket_link:
        warnings.append("no ticket link")

    links = {"tickets": ticket_link, "venue": venue_link, "more_info": None}

    # ── Summary (1 sentence, plain language) ─────────────────────────────────
    type_labels = {
        "live_music": "live music show",
        "dj_night": "DJ night",
        "open_mic": "open mic night",
        "dance": "dance night",
        "karaoke": "karaoke night",
        "trivia": "trivia night",
        "food": "food event",
        "sport": "sports event",
        "comedy": "comedy show",
        "other": "event",
    }
    type_label = type_labels.get(event.event_type or "other", "event")
    venue_name = venue.name or "the venue"

    if event.event_name and event.event_type == "live_music":
        summary = f"{event.event_name} performs at {venue_name}."
    elif event.event_name:
        summary = f"{event.event_name} is a {type_label} at {venue_name}."
    else:
        summary = f"A {type_label} at {venue_name}."

    # ── Details (2–4 sentences) ───────────────────────────────────────────────
    sentences: list[str] = []

    # Sentence 1 – what it is / who is performing (from raw description)
    if event.description:
        desc = _clean_text(event.description)
        if len(desc) > 400:
            desc = desc[:400].rsplit(" ", 1)[0] + "..."
            warnings.append("description looks truncated")
        sentences.append(desc)
    else:
        warnings.append("no description available")
        sentences.append(summary)  # fall back to summary as first sentence

    # Sentence 2 – doors / show time
    if event.doors_time and event.time:
        sentences.append(f"Doors open at {event.doors_time}, show starts at {event.time}.")
    elif event.doors_time:
        sentences.append(f"Doors open at {event.doors_time}.")
        warnings.append("missing show time")
    elif event.time:
        sentences.append(f"Show starts at {event.time}.")
    else:
        warnings.append("missing show time")

    # Sentence 3 – admission pricing (human-readable; URL goes in links)
    price_parts: list[str] = []
    if pricing["advance"]:
        price_parts.append(f"{pricing['advance']} advance")
    if pricing["day_of"]:
        price_parts.append(f"{pricing['day_of']} at the door")
    if price_parts:
        price_sentence = "Admission: " + " / ".join(price_parts)
        if pricing["notes"]:
            price_sentence += f" ({pricing['notes']})"
        price_sentence += "."
        sentences.append(price_sentence)

    details = " ".join(sentences[:4]) if sentences else None

    return {
        "summary": summary,
        "details": details,
        "pricing": pricing,
        "doors_time": event.doors_time,
        "show_time": event.time,
        "links": links,
        "warnings": warnings,
    }


def _description_from_intermediate(intermediate: dict) -> str:
    """Compose the final output description string from the intermediate JSON."""
    parts: list[str] = []

    if intermediate.get("details"):
        parts.append(intermediate["details"])
    elif intermediate.get("summary"):
        parts.append(intermediate["summary"])

    links = intermediate.get("links", {})
    if links.get("tickets"):
        parts.append(f"Tickets: {links['tickets']}")
    if links.get("venue"):
        parts.append(links["venue"])

    return " ".join(parts)


def format_output_event(event: Event, venue: Venue, venue_url: str) -> dict:
    """Transform an internal Event into the target output format."""
    intermediate = _build_event_intermediate(event, venue, venue_url)
    if intermediate["warnings"]:
        logger.debug("Event '%s' intermediate warnings: %s", event.event_name, intermediate["warnings"])
    description = _description_from_intermediate(intermediate)

    prefix = TITLE_PREFIX_MAP.get(event.event_type or "", "")
    title = f"{prefix}{event.event_name}" if event.event_name else event.event_name

    return {
        "title": title,
        "description": description,
        "category": FINAL_CATEGORY_MAP.get(event.event_type or "", "other"),
        "address": venue.address,
        "latitude": None,
        "longitude": None,
        "date": event.date,
        "time": event.time,
        "isPublic": True,
    }


def filter_events_by_date(events: list[Event], days: int) -> list[Event]:
    """Keep only events within the next N days from today."""
    today = date.today()
    cutoff = today + timedelta(days=days)
    filtered = []
    for e in events:
        if not e.date:
            continue
        try:
            event_date = date.fromisoformat(e.date)
        except ValueError:
            # Keep events with unparseable dates rather than silently dropping them
            filtered.append(e)
            continue
        if today <= event_date <= cutoff:
            filtered.append(e)
    return filtered


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape events (live music, trivia, DJ nights, etc.) from a venue website.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python venue_scraper.py https://www.emptybottle.com
  python venue_scraper.py https://www.emptybottle.com -d 10
  python venue_scraper.py https://www.thebottleneck.com -d 7 -o events.json -v
  python venue_scraper.py https://www.catscradle.com --no-llm
        """,
    )
    parser.add_argument("url", metavar="URL", help="Venue website URL (homepage or events page)")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write JSON output to file (default: stdout)")
    parser.add_argument("-d", "--days", type=int, default=None, metavar="N",
                        help="Only show events within the next N days (default: show all)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback (rule-based only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress and debug info to stderr")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(levelname)s: %(message)s")

    use_llm = not args.no_llm

    # Check for API key if LLM is enabled
    if use_llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set. LLM fallback will fail if needed. Use --no-llm to disable.")

    fetcher = PageFetcher()
    await fetcher.start()

    try:
        # Step 1: Find the events page
        finder = EventPageFinder()
        html, events_url, homepage_html = await finder.find_events_page(fetcher, args.url, use_llm=use_llm)

        # Step 2: Extract events
        extractor = EventExtractor()
        result = await extractor.extract(html, events_url, use_llm=use_llm, homepage_html=homepage_html)

        # Step 3: Filter by date range if requested
        events = result.events
        if args.days is not None:
            total_before = len(events)
            events = filter_events_by_date(events, args.days)
            logger.info("Date filter (next %d days): %d/%d events kept", args.days, len(events), total_before)

        # Step 4: Prompt for address if not found on the page
        if not result.venue.address:
            print("Could not find the venue address on the page.", file=sys.stderr)
            user_input = input("Please enter the venue address (or press Enter to skip): ").strip()
            if user_input:
                result.venue.address = user_input

        # Step 4b: Geocode the venue address
        lat, lon = None, None
        if result.venue.address:
            logger.info("Geocoding address: %s", result.venue.address)
            lat, lon = geocode_address(result.venue.address)
            if lat:
                logger.info("Geocoded to: %f, %f", lat, lon)

        # Step 5: Format events to target output structure
        venue_url = args.url
        output_events = []
        for e in events:
            formatted = format_output_event(e, result.venue, venue_url)
            formatted["latitude"] = lat
            formatted["longitude"] = lon
            output_events.append(formatted)

        json_str = json.dumps({"events": output_events}, indent=2, ensure_ascii=False)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str + "\n")
            print(f"Wrote {len(output_events)} events to {args.output}", file=sys.stderr)
        else:
            print(json_str)

        if not output_events:
            print("WARNING: No events found. Try with --verbose to see extraction details.", file=sys.stderr)

    finally:
        await fetcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
