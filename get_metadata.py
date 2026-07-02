import os
from gguf import GGUFReader

def scan_models(start_path, output_md):
    print(f"Starte Scan in: {os.path.abspath(start_path)}...")
    
    with open(output_md, 'w', encoding='utf-8') as md:
        md.write("# GGUF Modelle - Metadaten Übersicht\n\n")

        for root, _, files in os.walk(start_path):
            for file in files:
                if file.endswith('.gguf'):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, start_path)
                    print(f"Lese: {file}")

                    md.write(f"## {file}\n")
                    md.write(f"**Pfad:** `{rel_path}`\n\n")

                    long_fields = {}
                    md.write("| Metadaten-Schlüssel | Wert |\n")
                    md.write("|---|---|\n")

                    try:
                        reader = GGUFReader(filepath)
                        for key, field in reader.fields.items():
                            # Den tatsächlichen Wert aus dem GGUF-Feld extrahieren
                            val_str = str(field.parts[-1] if field.parts else "")

                            # Lange Texte (wie Jinja Templates) separat behandeln
                            if "template" in key or len(val_str) > 80:
                                long_fields[key] = val_str
                                md.write(f"| `{key}` | *Siehe Code-Block unten* |\n")
                            else:
                                # Zeichen escapen, die die Markdown-Tabelle brechen könnten
                                val_str = val_str.replace("|", "\\|").replace("\n", " ")
                                md.write(f"| `{key}` | `{val_str}` |\n")

                        md.write("\n")

                        # Chat-Templates und andere lange Strings als Block anfügen
                        for l_key, l_val in long_fields.items():
                            md.write(f"**{l_key}:**\n```jinja\n{l_val}\n```\n\n")

                    except Exception as e:
                        md.write(f"| Fehler | Konnte Metadaten nicht lesen: {e} |\n\n")

                    md.write("---\n\n")
                    
    print(f"\nFertig! Die Datei '{output_md}' wurde erstellt.")

if __name__ == "__main__":
    # Startet im aktuellen Verzeichnis und erstellt die Markdown-Datei
    scan_models(".", "models_metadata.md")
