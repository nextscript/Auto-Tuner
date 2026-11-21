# AutoTuner language packs

AutoTuner ships nine validated built-in interface packs:

- English (UK), the fallback and default
- Deutsch
- Nederlands
- Svenska
- 日本語
- Français
- Ελληνικά
- Polski
- Русский

Open the **⋯** toolbar and use **Language** to switch live. Missing strings
always fall back to English (UK), so older custom packs remain usable when new
controls are added.

## What is translated

Every built-in pack covers:

- toolbar, launch-options, server, and settings labels;
- two-level hover help (**In short:** / **Technical details:**) on all controls;
- Expert panel sections/labels, performance-test, OCR, performance-report, and
  path-manager dialogs;
- message boxes, model context menu, **★ Favourites**, and model-row tooltips;
- performance-target descriptions and metric explanations;
- `profile_notes` explanations for every bundled `settings/*.yaml` profile.

Hover help is stored as plain text; `localization.setting_tooltip_html()` builds
the Qt HTML and `LanguageManager.translate_tooltip()` translates it live while
preserving dynamic suffixes (`Active build:`, `Resolved path:`, etc.).
`test_localization.py` verifies with `ast` that every GUI help constant and
every shipped profile note is present and translated across all nine packs.

## Model profile explanations

Each `settings/*.yaml` file carries an English canonical `notes` field.
Language packs translate it via an optional `profile_notes` object keyed by
profile filename:

```json
{
  "schema_version": 1,
  "id": "de-DE",
  "name": "Deutsch",
  "locale": "de-DE",
  "strings": { "⚙ Settings": "⚙ Einstellungen" },
  "profile_notes": {
    "gemma-4.yaml": "Gemma ist empfindlich gegenüber repeat_penalty > 1.0. …"
  }
}
```

- Lookup order: active pack → English (UK) pack → YAML `notes`.
- Keys must be valid profile filenames (`name.yaml`); values are non-empty
  strings up to 20,000 characters (max 500 entries).
- The CLI/TUI prints the English YAML text.

## Custom packs

Click **Open language folder** to open `~/.autotuner/languages` and create
`custom-language-template.json` (a complete copy of the English catalogue,
never overwritten once created).

Rules for custom UTF-8 JSON packs (`schema_version: 1`, max 2 MiB, max 2,000
strings):

- `id`: 1–64 chars, letters/digits/`.`/`_`/`-`, starting with a letter or digit.
- `name`: native display name shown in the dropdown.
- `locale`: intended locale tag.
- `strings`: maps exact English (UK) source text to its translation (partial
  packs fall back to English).
- `profile_notes`: optional map of profile filename → translated note.

Use **Custom language pack…** to import a file. AutoTuner validates a staged
copy before replacing anything in `~/.autotuner/languages`. Custom packs survive
both source updates and compiled binary swaps.
