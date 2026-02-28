##Event ranking and classifier:

I have a web scraping script that collects event descriptions from venue websites.  
I categorize events into one of five types:

- music
- food
- trivia
- sport
- other

Right now I use keyword matching, but I’m seeing incorrect classifications — especially music vs food — because some events mention dinner or happy hour even when the main event is live music.

I want to improve the categorization logic to be more accurate and robust.

Constraints:
- The script is written in Python.
- I want to keep this rule-based (not ML) for now.
- I want to implement scoring/weights instead of simple keyword presence.
- I want thresholds and priority rules to resolve conflicts.
- I want negative keyword handling where appropriate.
- I want to map internal subcategories into the 5 final categories above.

Please help me:

1. Design a scoring-based categorization system.
2. Provide Python code to implement it.
3. Suggest thresholds and precedence logic.
4. Improve the keyword structure.
5. Reduce false positives between music and food.
6. Handle ambiguous phrases like "game night".
7. Return both:
   - internal category
   - final mapped category

Here is my improved keyword structure idea with weighted strengths:

RULES = {
  "live_music": {
    "strong": ["doors", "tickets", "ticket", "show at", "set time", "opening act", "support", "w/"],
    "medium": ["live music", "concert", "gig", "show", "performance", "touring"],
    "weak": ["band", "tour"]
  },
  "food": {
    "strong": ["prix fixe", "tasting menu", "chef's table", "wine pairing", "reservation required", "coursed dinner", "omakase"],
    "medium": ["wine tasting", "beer tasting", "cocktail class", "pop-up dinner", "supper club"],
    "weak": ["brunch", "dinner", "happy hour"]
  },
  "trivia": {
    "strong": ["pub quiz", "quiz night"],
    "medium": ["trivia"],
    "weak": ["brain game"]
  },
  "sport": {
    "strong": ["watch party", "fight night", "ufc", "pay-per-view"],
    "medium": ["nfl", "nba", "mlb", "nhl", "soccer", "playoffs", "super bowl", "world cup"],
    "weak": ["game night"]
  },
  "comedy": {
    "strong": ["stand-up", "stand up", "improv"],
    "medium": ["comedy", "comic"],
    "weak": ["laugh"]
  },
  "karaoke": {
    "strong": ["karaoke"],
    "medium": [],
    "weak": []
  },
  "open_mic": {
    "strong": ["open mic", "open mike"],
    "medium": ["songwriter night"],
    "weak": ["acoustic night"]
  },
  "dj_night": {
    "strong": ["dj set", "dance party"],
    "medium": ["club night"],
    "weak": ["edm night"]
  },
  "dance": {
    "strong": ["swing dance", "salsa night", "line dance"],
    "medium": ["bachata", "cumbia"],
    "weak": ["two-step"]
  }
}

Final category mapping:

music = live_music, dj_night, dance, open_mic, karaoke
trivia = trivia
sport = sport
food = food
other = everything else (including comedy)

I would like the algorithm to:

- Normalize text
- Score matches by strength (strong > medium > weak)
- Apply boosts for strong indicators like "tickets", "doors", etc.
- Apply penalties or ignore weak matches when strong matches from another category exist
- Use thresholds to determine confidence
- Choose the best category
- Fall back to "other" if confidence is low

Please provide:

- A clean Python implementation
- Suggested scoring weights
- Threshold logic
- Comments explaining reasoning
- Suggestions for future improvements