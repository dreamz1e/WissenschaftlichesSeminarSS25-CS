import random

# --- Konfiguration ---
SOURCE_FILE = 'Corpus/TEST_DATA/TEST_DATA.de'
TARGET_FILE = 'Corpus/TEST_DATA/TEST_DATA.en'
GOLD_SOURCE_FILE = 'gold.de'  # Ausgabedatei für die 200 deutschen Sätze
GOLD_TARGET_FILE = 'gold.en'  # Ausgabedatei für die 200 englischen Sätze
SAMPLE_SIZE = 200

# --- Skript ---
print("Lese Quelldatei...")
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    source_lines = f.readlines()

print("Lese Zieldatei...")
with open(TARGET_FILE, 'r', encoding='utf-8') as f:
    target_lines = f.readlines()

# Sicherstellen, dass die Anzahl der Zeilen übereinstimmt
if len(source_lines) != len(target_lines):
    raise ValueError("Die Anzahl der Zeilen in den Quell- und Zieldateien stimmt nicht überein!")

# Ein Tupel für jedes Satzpaar erstellen
sentence_pairs = list(zip(source_lines, target_lines))

# Eine zufällige Stichprobe von 200 Paaren ziehen
print(f"Ziehe eine zufällige Stichprobe von {SAMPLE_SIZE} Sätzen...")
gold_sample = random.sample(sentence_pairs, SAMPLE_SIZE)

# Die ausgewählten Sätze in neue Dateien schreiben
with open(GOLD_SOURCE_FILE, 'w', encoding='utf-8') as f_src, \
     open(GOLD_TARGET_FILE, 'w', encoding='utf-8') as f_trg:
    for src_line, trg_line in gold_sample:
        f_src.write(src_line)
        f_trg.write(trg_line)

print(f"Fertig! Die 200 Sätze wurden in '{GOLD_SOURCE_FILE}' und '{GOLD_TARGET_FILE}' gespeichert.")