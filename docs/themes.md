# AutoTuner themes

AutoTuner ships **System**, **Dark**, **Dark Gray**, **Light**, **High Contrast**, and the pink-accented **Midnight Rose** themes. **System** keeps the operating system's native palette and controls; the other themes apply AutoTuner's complete color palette. Open **⚙ Settings** and select a theme. The change is previewed immediately; **Cancel** restores the previous appearance and **OK** remembers it. Font size remains the separate toolbar **A− / A+** accessibility setting.

## Create a theme

Use **⚙ Settings → Customize…** to edit the selected theme's colors and UI/monospace fonts. Built-ins stay read-only: their editor draft automatically appends `-user` to the ID and name and saves as a new user theme. Selecting an existing user theme lets you edit and overwrite it with the same ID; changing its ID creates a separate theme instead. **Open folder** opens the place for manually installed themes; click **Reload** after copying a file there.

User themes live under AutoTuner's shared data folder:

- default: `~/.autotuner/themes/` (Windows: `%USERPROFILE%\.autotuner\themes\`)
- portable override: `<AUTOTUNER_DATA_DIR>/themes/`

They are stored outside the packaged application resources, so source updates and binary swaps retain them.

## JSON format (schema v1)

Themes are UTF-8 JSON files smaller than 64 KiB. They may contain **only** the documented fields below. Colors are opaque `#RRGGBB`; arbitrary QSS, selectors, URLs and paths are deliberately unsupported.

```json
{
  "schema_version": 1,
  "id": "my-ocean",
  "name": "My Ocean",
  "description": "A readable blue theme",
  "colors": {
    "window_bg": "#101820", "panel_bg": "#16232e", "control_bg": "#203443",
    "control_hover": "#29475b", "control_pressed": "#35627d", "text": "#f2f7fa",
    "muted_text": "#b7c5ce", "accent": "#55bdeb", "accent_text": "#071218",
    "border": "#466474", "selection_bg": "#247da5", "selection_text": "#ffffff",
    "disabled_text": "#75858e", "success": "#66cc88", "warning": "#f0c75e",
    "error": "#f07878", "favorite_active": "#ffd54f", "favorite_inactive": "#777777",
    "sysbar_bg": "#161625", "sysbar_text": "#88bbee", "section_text": "#88bbee"
  },
  "font": { "ui_family": "", "mono_family": "" }
}
```

`id` must use lowercase letters, digits, `_` or `-`, begin with a letter/digit, and be at most 64 characters. Every color role is required:

| Role | Used for |
|---|---|
| `window_bg`, `panel_bg` | main window and panels |
| `control_bg`, `control_hover`, `control_pressed` | buttons/controls and states |
| `text`, `muted_text`, `disabled_text` | normal, secondary and disabled text |
| `accent`, `accent_text`, `border` | accent and outlines |
| `selection_bg`, `selection_text` | selected list/combo entries |
| `success`, `warning`, `error` | semantic status colors |
| `favorite_active`, `favorite_inactive` | model star states; the configured colors are rendered exactly |
| `sysbar_bg`, `sysbar_text`, `section_text` | hardware bar and section headings |

Empty font families use the platform UI font and Qt's fixed-width font. A named font must be installed on the current OS; unavailable fonts simply fall back through Qt.

## Errors and fallback

Invalid JSON, wrong schema, missing/extra roles, invalid colors, duplicate IDs, oversized files, or a deleted active user theme are ignored. AutoTuner continues with **System** and records a readable warning during discovery. Fix the file and click **Reload**.

## Why no QSS or emoji editor?

v1 intentionally accepts declarative colors and fonts only. Free QSS could hide controls, load external resources, or make the UI unusable. Emoji/icon customization is also excluded: color emoji rendering, font fallback, glyph widths, and native symbols differ across Windows, Linux and macOS, while symbols occur throughout the UI and terminal output. A future icon system should use tested semantic SVG/QIcon assets instead.
