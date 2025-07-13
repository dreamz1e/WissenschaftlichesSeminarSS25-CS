import torch
from transformers import BertModel, BertTokenizerFast
import numpy as np
import argparse
import os
from tqdm import tqdm

def get_word_alignments(source_sent, target_sent, model, tokenizer, layer_index=-2):
    """
    Extrahiert Wort-Alignments aus den Attention-Gewichten eines BERT-Modells.

    Args:
        source_sent (str): Der Quellsatz.
        target_sent (str): Der Zielsatz.
        model (BertModel): Das geladene Transformer-Modell.
        tokenizer (BertTokenizerFast): Der zum Modell passende Fast-Tokenizer.
        layer_index (int): Der Layer, aus dem die Attention-Gewichte extrahiert werden.
                           -1 ist der letzte, -2 der vorletzte (oft eine gute Wahl).

    Returns:
        list: Eine Liste von Tupeln, die die alignierten Wort-Indizes repräsentieren,
              z.B. [(0, 0), (1, 2)]
    """
    # 1. Sätze tokenisieren und für das Modell vorbereiten
    # Format: [CLS] source_tokens [SEP] target_tokens [SEP]
    source_tokens = tokenizer.tokenize(source_sent)
    target_tokens = tokenizer.tokenize(target_sent)

    # Erstellen der Subword-zu-Wort-Zuordnung. Dies ist der entscheidende Schritt,
    # um von Subword-Alignments auf Wort-Alignments zu kommen.
    source_word_ids = tokenizer(source_sent, add_special_tokens=False).word_ids()
    target_word_ids = tokenizer(target_sent, add_special_tokens=False).word_ids()

    # Wir brauchen die Anzahl der echten Wörter für später
    num_source_words = max(filter(None, source_word_ids)) + 1 if any(source_word_ids) else 0
    num_target_words = max(filter(None, target_word_ids)) + 1 if any(target_word_ids) else 0
    
    # Modell-Input vorbereiten
    input_tokens = ['[CLS]'] + source_tokens + ['[SEP]'] + target_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_tensor = torch.tensor([input_ids])

    # 2. Modell-Forward-Pass durchführen, um Attention zu erhalten
    with torch.no_grad():
        outputs = model(input_tensor, output_attentions=True)
        attentions = outputs.attentions

    # 3. Attention-Gewichte extrahieren und aggregieren
    # Wir nehmen die Gewichte aus dem gewählten Layer
    attention_layer = attentions[layer_index]
    # Durchschnitt über alle Attention-Heads bilden, um eine "Konsens-Attention" zu erhalten
    attention_matrix = attention_layer.mean(dim=1)[0] # Shape: (seq_len, seq_len)

    # 4. Relevante Attention-Matrix extrahieren (Ziel -> Quelle)
    # Die Positionen der Quell- und Ziel-Tokens im Gesamt-Input bestimmen
    source_start = 1
    source_end = source_start + len(source_tokens)
    target_start = source_end + 1
    target_end = target_start + len(target_tokens)
    
    # Wir interessieren uns dafür, worauf jedes Ziel-Token seine Aufmerksamkeit richtet
    # Daher: Zeilen = Ziel-Tokens, Spalten = Quell-Tokens
    target_to_source_attention = attention_matrix[target_start:target_end, source_start:source_end]

    # 5. Subword-Attention zu Wort-Attention aggregieren
    # Erstelle eine leere Matrix mit den Dimensionen (Anzahl Zielwörter x Anzahl Quellwörter)
    word_level_attention = np.zeros((num_target_words, num_source_words))
    subword_counts = np.zeros((num_target_words, num_source_words))

    for i in range(len(target_tokens)): # Iteriere über Ziel-Subwords
        for j in range(len(source_tokens)): # Iteriere über Quell-Subwords
            target_word_idx = target_word_ids[i]
            source_word_idx = source_word_ids[j]
            
            if target_word_idx is not None and source_word_idx is not None:
                # Addiere die Attention-Werte für alle Subword-Paare, die zu einem Wort-Paar gehören
                word_level_attention[target_word_idx, source_word_idx] += target_to_source_attention[i, j].item()
                subword_counts[target_word_idx, source_word_idx] += 1
    
    # Durchschnitt bilden, um eine faire Verteilung zu gewährleisten
    # Division durch Null vermeiden, falls keine Alignments gefunden wurden
    word_level_attention /= np.where(subword_counts > 0, subword_counts, 1)

    # 6. Finale Alignments extrahieren
    # Für jedes Zielwort das Quellwort mit der höchsten Attention finden (argmax)
    alignments = []
    for target_word_idx in range(num_target_words):
        source_word_idx = np.argmax(word_level_attention[target_word_idx, :])
        alignments.append((int(source_word_idx), target_word_idx))
        
    return alignments


def process_files(source_file, target_file, output_file, model_name='bert-base-multilingual-cased', max_lines=None):
    """
    Verarbeitet parallele Textdateien und erstellt Wort-Alignments.
    
    Args:
        source_file (str): Pfad zur Quelldatei (Deutsch)
        target_file (str): Pfad zur Zieldatei (Englisch)
        output_file (str): Pfad für die Ausgabedatei mit Alignments
        model_name (str): Name des BERT-Modells
        max_lines (int): Maximale Anzahl zu verarbeitender Zeilen (None für alle)
    """
    print(f"Lade Modell '{model_name}' und Tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    print("Modell geladen.\n")

    # Dateien öffnen
    with open(source_file, 'r', encoding='utf-8') as src_f, \
         open(target_file, 'r', encoding='utf-8') as tgt_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        # Header für die Ausgabedatei schreiben
        out_f.write("# Word alignments generated by BERT\n")
        out_f.write("# Format: target_word_index-source_word_index (0-indexed)\n")
        out_f.write("# Each line corresponds to one sentence pair\n\n")
        
        # Zeilen zählen für Fortschrittsanzeige
        if max_lines is None:
            print("Zähle Zeilen...")
            with open(source_file, 'r', encoding='utf-8') as temp_f:
                total_lines = sum(1 for _ in temp_f)
        else:
            total_lines = max_lines
            
        print(f"Verarbeite {total_lines} Satzpaare...")
        
        # Verarbeitung mit Fortschrittsanzeige
        processed_lines = 0
        with tqdm(total=total_lines, desc="Processing") as pbar:
            for line_num, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                if max_lines and line_num >= max_lines:
                    break
                
                # Zeilen säubern
                src_sent = src_line.strip()
                tgt_sent = tgt_line.strip()
                
                # Leere Zeilen überspringen
                if not src_sent or not tgt_sent:
                    out_f.write("\n")
                    pbar.update(1)
                    continue
                
                try:
                    # Alignments generieren
                    alignments = get_word_alignments(src_sent, tgt_sent, model, tokenizer)
                    
                    # Alignments im Pharaoh-Format ausgeben (target-source)
                    alignment_str = " ".join([f"{tgt_idx}-{src_idx}" for src_idx, tgt_idx in alignments])
                    out_f.write(f"{alignment_str}\n")
                    
                    processed_lines += 1
                    
                except Exception as e:
                    print(f"\nFehler bei Zeile {line_num + 1}: {e}")
                    out_f.write("# ERROR IN THIS LINE\n")
                
                pbar.update(1)
                
                # Zwischenspeichern alle 1000 Zeilen
                if (line_num + 1) % 1000 == 0:
                    out_f.flush()
    
    print(f"\nVerarbeitung abgeschlossen!")
    print(f"Erfolgreich verarbeitete Zeilen: {processed_lines}")
    print(f"Alignments gespeichert in: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT-basierte Wort-Alignment-Generierung')
    parser.add_argument('--source_file', '-s', default='../../Corpus/TEST_DATA/TEST_DATA.de', 
                        help='Pfad zur Quelldatei (Deutsch)')
    parser.add_argument('--target_file', '-t', default='../../Corpus/TEST_DATA/TEST_DATA.en', 
                        help='Pfad zur Zieldatei (Englisch)')
    parser.add_argument('--output_file', '-o', default='bert_alignments.txt', 
                        help='Pfad für die Ausgabedatei')
    parser.add_argument('--model_name', '-m', default='bert-base-multilingual-cased', 
                        help='Name des BERT-Modells')
    parser.add_argument('--max_lines', type=int, default=None, 
                        help='Maximale Anzahl zu verarbeitender Zeilen (für Tests)')
    
    args = parser.parse_args()
    
    # Überprüfung der Eingabedateien
    if not os.path.exists(args.source_file):
        print(f"Fehler: Quelldatei '{args.source_file}' nicht gefunden!")
        exit(1)
    if not os.path.exists(args.target_file):
        print(f"Fehler: Zieldatei '{args.target_file}' nicht gefunden!")
        exit(1)
    
    print("=== BERT Word Alignment Generator ===")
    print(f"Quelldatei (DE): {args.source_file}")
    print(f"Zieldatei (EN): {args.target_file}")
    print(f"Ausgabedatei: {args.output_file}")
    print(f"Modell: {args.model_name}")
    if args.max_lines:
        print(f"Maximale Zeilen: {args.max_lines}")
    print()
    
    # Verarbeitung starten
    process_files(args.source_file, args.target_file, args.output_file, 
                  args.model_name, args.max_lines)