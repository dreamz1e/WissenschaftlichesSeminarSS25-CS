import spacy
from collections import Counter, defaultdict
import pandas as pd
from functools import lru_cache

# --- Konfiguration ---
MIN_FREQUENCY = 2  # Minimum frequency for a term to be kept
MAX_PHRASE_LEN = 5 # Maximum length of extracted terms
DEVELOPMENT_MODE = False  # Set to True to process only first 5000 sentences for testing
MAX_SENTENCES_DEV = 5000  # Number of sentences to process in development mode

# Laden der spaCy Modelle für Lemmatisierung
try:
    nlp_de = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
    nlp_en = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except OSError:
    print("Fehler: spaCy-Modelle nicht gefunden. Bitte führen Sie aus:")
    print("python -m spacy download de_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    exit()

# --- Optimierte Hilfsfunktionen ---

@lru_cache(maxsize=10000)
def lemmatize_cached(text, lang='de'):
    """Cached lemmatization to avoid repeated processing of the same text."""
    nlp = nlp_de if lang == 'de' else nlp_en
    return " ".join([token.lemma_.lower() for token in nlp(text.strip())])

def lemmatize_batch(texts, lang='de'):
    """Batch lemmatization for better performance."""
    if not texts:
        return []
    
    nlp = nlp_de if lang == 'de' else nlp_en
    # Process multiple texts in a single spaCy call
    docs = list(nlp.pipe(texts))
    return [" ".join([token.lemma_.lower() for token in doc]) for doc in docs]


def find_gold_sentences_in_testdata(gold_src_file, gold_trg_file, test_src_file, test_trg_file):
    """
    Findet die Positionen der Gold-Standard-Sätze in den Test-Daten.
    
    Returns:
        dict: Mapping von Gold-Index zu Test-Index (0-based)
    """
    print("Suche Gold-Standard-Sätze in Test-Daten...")
    
    # Lade Gold-Standard-Sätze
    print("  Lade Gold-Standard-Sätze...")
    with open(gold_src_file, 'r', encoding='utf-8') as f:
        gold_src_lines = [line.strip() for line in f.readlines()]
    
    with open(gold_trg_file, 'r', encoding='utf-8') as f:
        gold_trg_lines = [line.strip() for line in f.readlines()]
    
    # Lade Test-Daten
    print("  Lade Test-Daten...")
    with open(test_src_file, 'r', encoding='utf-8') as f:
        test_src_lines = [line.strip() for line in f.readlines()]
    
    with open(test_trg_file, 'r', encoding='utf-8') as f:
        test_trg_lines = [line.strip() for line in f.readlines()]
    
    print(f"  Durchsuche {len(test_src_lines)} Test-Sätze...")
    
    # Erstelle Mapping
    gold_to_test_mapping = {}
    
    for gold_idx, (gold_src, gold_trg) in enumerate(zip(gold_src_lines, gold_trg_lines)):
        if gold_idx % 10 == 0:
            print(f"    Verarbeite Gold-Satz {gold_idx + 1}/{len(gold_src_lines)}")
        
        # Suche nach exakter Übereinstimmung in Test-Daten
        for test_idx, (test_src, test_trg) in enumerate(zip(test_src_lines, test_trg_lines)):
            if gold_src == test_src and gold_trg == test_trg:
                gold_to_test_mapping[gold_idx] = test_idx
                break
    
    print(f"Gefunden: {len(gold_to_test_mapping)} von {len(gold_src_lines)} Gold-Standard-Sätzen in Test-Daten")
    
    if len(gold_to_test_mapping) != len(gold_src_lines):
        missing_count = len(gold_src_lines) - len(gold_to_test_mapping)
        print(f"WARNUNG: {missing_count} Gold-Standard-Sätze wurden nicht in den Test-Daten gefunden!")
        
        # Zeige erste 5 fehlende Sätze
        for gold_idx in range(min(5, len(gold_src_lines))):
            if gold_idx not in gold_to_test_mapping:
                print(f"  Fehlend (Gold {gold_idx}): {gold_src_lines[gold_idx][:50]}...")
    
    return gold_to_test_mapping


def extract_alignments_for_gold_sentences(model_align_file, gold_to_test_mapping, output_file):
    """
    Extrahiert die Alignments für die Gold-Standard-Sätze aus den Modell-Alignment-Dateien.
    
    Args:
        model_align_file: Pfad zur Modell-Alignment-Datei (basierend auf Test-Daten)
        gold_to_test_mapping: Mapping von Gold-Index zu Test-Index
        output_file: Pfad zur Ausgabedatei für Gold-Alignments
    
    Returns:
        str: Pfad zur erstellten Ausgabedatei
    """
    print(f"Extrahiere Alignments für Gold-Standard-Sätze aus {model_align_file}...")
    
    # Lade alle Alignments
    print("  Lade Alignment-Datei...")
    with open(model_align_file, 'r', encoding='utf-8') as f:
        all_alignments = [line.strip() for line in f.readlines()]
    
    print(f"  Alignment-Datei enthält {len(all_alignments)} Zeilen")
    
    # Extrahiere Alignments für Gold-Standard-Sätze
    gold_alignments = []
    
    for gold_idx in sorted(gold_to_test_mapping.keys()):
        test_idx = gold_to_test_mapping[gold_idx]
        if test_idx < len(all_alignments):
            gold_alignments.append(all_alignments[test_idx])
        else:
            print(f"WARNUNG: Test-Index {test_idx} außerhalb des Bereichs für Gold-Index {gold_idx}")
            gold_alignments.append("")  # Leeres Alignment als Fallback
    
    # Schreibe extrahierte Alignments
    with open(output_file, 'w', encoding='utf-8') as f:
        for alignment in gold_alignments:
            f.write(alignment + '\n')
    
    print(f"Alignments für {len(gold_alignments)} Gold-Standard-Sätze nach {output_file} geschrieben")
    return output_file


def parse_alignment(alignment_line):
    """Konvertiert eine Pharaoh-Alignment-Zeile in ein Set von (src_idx, trg_idx) Tupeln."""
    alignments = set()
    if alignment_line.strip():
        try:
            for pair in alignment_line.strip().split():
                src_idx, trg_idx = map(int, pair.split('-'))
                alignments.add((src_idx, trg_idx))
        except ValueError:
            pass # Ignoriere fehlerhafte Alignments
    return alignments

def load_gold_terminology(filepath):
    """Optimierte Terminologie-Ladung mit Caching."""
    gold_terms = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    src_texts = []
    trg_texts = []
    
    for line in lines:
        if '|||' in line:
            src, trg = line.split('|||')
            src_texts.append(src.strip())
            trg_texts.append(trg.strip())
    
    # Batch-Lemmatisierung
    if src_texts and trg_texts:
        lemmatized_src = lemmatize_batch(src_texts, 'de')
        lemmatized_trg = lemmatize_batch(trg_texts, 'en')
        
        for lemma_src, lemma_trg in zip(lemmatized_src, lemmatized_trg):
            if lemma_src and lemma_trg:
                gold_terms.add((lemma_src, lemma_trg))
    
    return gold_terms

# --- Evaluationsfunktionen ---

def calculate_aer(model_align_file, gold_align_file):
    """Berechnet die Alignment Error Rate (AER). Annahme: Alle Gold-Alignments sind 'Sure'."""
    
    model_A = []
    gold_S = []

    with open(model_align_file, 'r', encoding='utf-8') as f_model, \
         open(gold_align_file, 'r', encoding='utf-8') as f_gold:
        
        for line_model, line_gold in zip(f_model, f_gold):
            model_A.append(parse_alignment(line_model))
            gold_S.append(parse_alignment(line_gold))

    if len(model_A) != len(gold_S):
        print(f"Warnung: {model_align_file} und {gold_align_file} haben unterschiedliche Zeilenanzahlen.")
        return None

    total_A = 0
    total_S = 0
    total_intersection = 0

    for A, S in zip(model_A, gold_S):
        intersection = len(A.intersection(S))
        total_intersection += intersection
        total_A += len(A)
        total_S += len(S)

    # AER-Formel (vereinfacht für S=P): 1 - (2 * |A ∩ S|) / (|A| + |S|)
    if total_A + total_S == 0:
        return 0.0
    
    aer = 1.0 - (2.0 * total_intersection) / (total_A + total_S)
    return aer


def evaluate_terminology(extracted_terms_set, gold_terms_set):
    """Berechnet Precision, Recall und F1-Score."""
    
    TP = len(extracted_terms_set.intersection(gold_terms_set))
    FP = len(extracted_terms_set - gold_terms_set)
    FN = len(gold_terms_set - extracted_terms_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall, "F1-Score": f1
    }

# --- Phrasenextraktions-Pipeline ---

def extract_consistent_phrases(src_tokens, trg_tokens, alignment_set):
    """Optimierte Kern-Heuristik zur Extraktion konsistenter Phrasenpaare."""
    if not alignment_set:
        return []
    
    extracted = []
    src_len = len(src_tokens)
    trg_len = len(trg_tokens)
    
    # Frühe Terminierung bei sehr langen Sätzen
    if src_len > 50 or trg_len > 50:
        return []
    
    # Erstelle Lookup-Sets für bessere Performance
    alignment_dict = defaultdict(set)
    for src_idx, trg_idx in alignment_set:
        alignment_dict[src_idx].add(trg_idx)
    
    for i in range(src_len):
        for j in range(i, min(i + MAX_PHRASE_LEN, src_len)):
            # Source Phrase Span: [i, j]
            
            # Finde zugehörige Target Indices - optimiert
            target_indices = set()
            for src_idx in range(i, j + 1):
                target_indices.update(alignment_dict[src_idx])
            
            if not target_indices:
                continue

            min_trg = min(target_indices)
            max_trg = max(target_indices)

            if max_trg - min_trg + 1 > MAX_PHRASE_LEN:
                continue

            # --- Optimierter Konsistenz-Check ---
            is_consistent = True
            for src_idx, trg_idx in alignment_set:
                is_source_in_phrase = i <= src_idx <= j
                is_target_in_phrase = min_trg <= trg_idx <= max_trg
                
                if is_source_in_phrase != is_target_in_phrase:
                    is_consistent = False
                    break
            
            if is_consistent:
                source_phrase = " ".join(src_tokens[i:j+1])
                target_phrase = " ".join(trg_tokens[min_trg:max_trg+1])
                
                # Sammle für Batch-Lemmatisierung
                extracted.append((source_phrase, target_phrase))

    return extracted


def run_extraction_pipeline(source_file, target_file, alignment_file):
    """Optimierte Extraktion und Filterung."""
    print("  Starte Phrasenextraktion...")
    all_extracted_pairs = []
    
    # Zähle Zeilen für Fortschrittsanzeige
    with open(source_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Development mode: process only subset
    if DEVELOPMENT_MODE and total_lines > MAX_SENTENCES_DEV:
        total_lines = MAX_SENTENCES_DEV
        print(f"  DEVELOPMENT MODE: Verarbeite nur erste {total_lines} Satzpaare")
    
    print(f"  Verarbeite {total_lines} Satzpaare...")
    
    # Batch-Verarbeitung für bessere Performance
    batch_size = 500
    current_batch = []
    processed = 0
    
    with open(source_file, 'r', encoding='utf-8') as f_src, \
         open(target_file, 'r', encoding='utf-8') as f_trg, \
         open(alignment_file, 'r', encoding='utf-8') as f_align:

        for line_idx, (src_line, trg_line, align_line) in enumerate(zip(f_src, f_trg, f_align)):
            # Development mode: stop after MAX_SENTENCES_DEV
            if DEVELOPMENT_MODE and processed >= MAX_SENTENCES_DEV:
                break
                
            src_tokens = src_line.strip().split()
            trg_tokens = trg_line.strip().split()
            align_set = parse_alignment(align_line)

            phrases = extract_consistent_phrases(src_tokens, trg_tokens, align_set)
            current_batch.extend(phrases)
            processed += 1
            
            # Verarbeite Batch oder zeige Fortschritt
            if processed % batch_size == 0 or processed == total_lines:
                # Batch-Lemmatisierung
                if current_batch:
                    src_phrases = [phrase[0] for phrase in current_batch]
                    trg_phrases = [phrase[1] for phrase in current_batch]
                    
                    # Lemmatisiere in Batches
                    lemmatized_src = lemmatize_batch(src_phrases, 'de')
                    lemmatized_trg = lemmatize_batch(trg_phrases, 'en')
                    
                    # Kombiniere lemmatisierte Paare
                    for lemma_src, lemma_trg in zip(lemmatized_src, lemmatized_trg):
                        if lemma_src and lemma_trg:
                            all_extracted_pairs.append((lemma_src, lemma_trg))
                    
                    current_batch = []
                
                # Fortschrittsanzeige weniger häufig
                if processed % 2500 == 0 or processed == total_lines:
                    progress = processed / total_lines * 100
                    print(f"    Fortschritt: {processed}/{total_lines} ({progress:.1f}%)")

    print(f"  Extraktion abgeschlossen. Gefundene Phrasenpaare: {len(all_extracted_pairs)}")
    
    # Frequenz-Filterung
    print("  Führe Frequenz-Filterung durch...")
    pair_counts = Counter(all_extracted_pairs)
    final_term_set = {pair for pair, count in pair_counts.items() if count >= MIN_FREQUENCY}
    
    print(f"  Nach Filterung (>= {MIN_FREQUENCY}): {len(final_term_set)} einzigartige Terme")
    
    return final_term_set, pair_counts


# --- Hauptanalyse-Funktion ---

def analyze_model(model_name, source_file, target_file, model_align_file, gold_align_file, gold_terms_set, evaluation_type="Gold"):
    """
    Analysiert ein Modell entweder auf Gold-Standard-Daten oder auf den vollständigen Test-Daten.
    
    Args:
        evaluation_type: "Gold" für Gold-Standard-Evaluation, "Full" für vollständige Test-Daten
    """
    print(f"\n--- Analysiere Modell: {model_name} ({evaluation_type}) ---")
    
    # Zeige Dateigröße für bessere Einschätzung
    import os
    if os.path.exists(source_file):
        file_size = os.path.getsize(source_file) / (1024 * 1024)  # MB
        print(f"  Quelldatei: {source_file} ({file_size:.1f} MB)")
    
    if os.path.exists(model_align_file):
        align_size = os.path.getsize(model_align_file) / (1024 * 1024)  # MB
        print(f"  Alignment-Datei: {model_align_file} ({align_size:.1f} MB)")

    # 1. Intrinsische Evaluation (AER) - nur für Gold-Standard verfügbar
    if evaluation_type == "Gold":
        print("  Berechne AER...")
        aer = calculate_aer(model_align_file, gold_align_file)
        print(f"  AER: {aer:.4f}")
    else:
        aer = None
        print("  AER: Nicht verfügbar (nur für Gold-Standard)")

    # 2. Terminologieextraktion
    print("  Starte Terminologieextraktion...")
    extracted_terms, term_counts = run_extraction_pipeline(source_file, target_file, model_align_file)
    print(f"  Anzahl extrahierter Terme (nach Filterung >= {MIN_FREQUENCY}): {len(extracted_terms)}")

    # 3. Extrinsische Evaluation (P/R/F1) - nur wenn Gold-Terminologie verfügbar
    if gold_terms_set:
        print("  Berechne Terminologie-Metriken...")
        metrics = evaluate_terminology(extracted_terms, gold_terms_set)
        print(f"  Terminologie-Evaluation: P={metrics['Precision']:.4f}, R={metrics['Recall']:.4f}, F1={metrics['F1-Score']:.4f}")
        
        # Qualitative Analyse (Beispiele)
        false_positives = list(extracted_terms - gold_terms_set)[:5]
        false_negatives = list(gold_terms_set - extracted_terms)[:5]
        
        print("\n  Beispiele für False Positives (Extrahiert, aber nicht Gold):")
        for fp in false_positives:
            print(f"    - {fp[0]} ||| {fp[1]}")

        print("\n  Beispiele für False Negatives (Gold, aber nicht gefunden):")
        for fn in false_negatives:
            print(f"    - {fn[0]} ||| {fn[1]}")
    else:
        metrics = {"Precision": 0, "Recall": 0, "F1-Score": 0, "TP": 0, "FP": 0, "FN": 0}
        print("  Terminologie-Evaluation: Nicht verfügbar (Gold-Terminologie nicht geladen)")

    # Sammeln der Ergebnisse für die finale Tabelle
    results = {
        'Model': f"{model_name} ({evaluation_type})",
        'AER': aer if aer is not None else 0.0,
        'Extracted_Count': len(extracted_terms),
        'Evaluation_Type': evaluation_type,
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score'],
        'TP': metrics['TP'],
        'FP': metrics['FP'],
        'FN': metrics['FN']
    }
    
    print(f"  Modell-Analyse abgeschlossen: {model_name} ({evaluation_type})")
    return results

# --- Main Execution ---
if __name__ == "__main__":
    
    print("="*60)
    print("PERFORMANCE-OPTIMIERTE PIPELINE ANALYZER")
    print("="*60)
    
    if DEVELOPMENT_MODE:
        print(f"⚠️  DEVELOPMENT MODE AKTIV: Verarbeite nur erste {MAX_SENTENCES_DEV} Sätze")
        print("   Setze DEVELOPMENT_MODE = False für vollständige Verarbeitung")
        print("")
    
    print("🚀 Optimierungen aktiv:")
    print("   • Batch-Lemmatisierung für bessere Performance")
    print("   • Cached Lemmatization für wiederholte Phrasen")
    print("   • Optimierte Phrase-Extraktion mit früher Terminierung")
    print("   • Reduzierte Fortschrittsanzeige (alle 2500 Sätze)")
    print("   • Effiziente Datenstrukturen für Alignment-Lookups")
    print("")
    
    # Dateipfade
    # Test-Daten (50,000 Sätze)
    TEST_SRC = '../Corpus/TEST_DATA/TEST_DATA.de'
    TEST_TRG = '../Corpus/TEST_DATA/TEST_DATA.en'
    
    # Gold-Standard-Daten (100 Sätze)
    GOLD_SRC = '../Corpus/GOLD_MANUAL/gold.de'
    GOLD_TRG = '../Corpus/GOLD_MANUAL/gold.en'
    GOLD_ALIGN = '../Corpus/GOLD_MANUAL/gold.align'
    GOLD_TERMS_FILE = '../Corpus/GOLD_MANUAL/gold.terminology.txt'

    # 1. Lade Goldstandard-Terminologie
    print("Lade Gold-Terminologie...")
    gold_terms = load_gold_terminology(GOLD_TERMS_FILE)
    print(f"Anzahl einzigartiger Gold-Terme: {len(gold_terms)}")

    # 2. Finde Gold-Standard-Sätze in Test-Daten
    gold_to_test_mapping = find_gold_sentences_in_testdata(GOLD_SRC, GOLD_TRG, TEST_SRC, TEST_TRG)
    
    # 3. Definiere die Modelle, die evaluiert werden sollen
    models_to_evaluate = {
        "Fast_Align": "fast_align/test.final.align",
        "SimAlign": "SimAlign/sim_alignments.txt", 
        "Attention": "bert-base-multilingual-cased/bert_alignments.txt"
    }

    # 4. Führe die Analyse durch
    all_results = []
    
    for model_name, test_align_file in models_to_evaluate.items():
        try:
            # Extrahiere Gold-Standard-Alignments aus den Test-Alignments
            gold_align_extracted = f"temp_{model_name.lower()}_gold.align"
            extract_alignments_for_gold_sentences(test_align_file, gold_to_test_mapping, gold_align_extracted)
            
            # Evaluiere auf Gold-Standard-Daten (mit AER)
            results_gold = analyze_model(
                model_name, 
                GOLD_SRC, 
                GOLD_TRG, 
                gold_align_extracted, 
                GOLD_ALIGN, 
                gold_terms,
                evaluation_type="Gold"
            )
            all_results.append(results_gold)
            
            # Evaluiere auf vollständigen Test-Daten (ohne AER)
            results_full = analyze_model(
                model_name,
                TEST_SRC,
                TEST_TRG,
                test_align_file,
                None,  # Kein Gold-Alignment für vollständige Test-Daten
                gold_terms,  # Verwende Gold-Terminologie trotzdem für P/R/F1
                evaluation_type="Full"
            )
            all_results.append(results_full)
            
        except FileNotFoundError as e:
            print(f"\nFEHLER: Datei nicht gefunden für Modell {model_name}: {e}")
        except Exception as e:
            print(f"\nFEHLER bei Modell {model_name}: {e}")

    # 5. Ergebnisse präsentieren
    if all_results:
        print("\n" + "="*60)
        print("      ZUSAMMENFASSUNG DER ERGEBNISSE")
        print("="*60)
        
        df = pd.DataFrame(all_results)
        # Wähle die relevanten Spalten für die finale Tabelle aus
        summary_df = df[['Model', 'Evaluation_Type', 'AER', 'Precision', 'Recall', 'F1-Score', 'Extracted_Count', 'TP', 'FP', 'FN']]
        summary_df = summary_df.round(4)
        
        print(summary_df.to_markdown(index=False))
        
        # Zusätzliche Analysen
        print("\n" + "="*60)
        print("      VERGLEICHSANALYSE")
        print("="*60)
        
        # Gruppiere nach Modell für Vergleich
        gold_results = df[df['Evaluation_Type'] == 'Gold'].copy()
        full_results = df[df['Evaluation_Type'] == 'Full'].copy()
        
        if len(gold_results) > 0:
            print("\nGold-Standard-Evaluation (mit AER):")
            gold_summary = gold_results.loc[:, ['Model', 'AER', 'Precision', 'Recall', 'F1-Score', 'Extracted_Count']].round(4)
            print(gold_summary.to_markdown(index=False))
        
        if len(full_results) > 0:
            print("\nVollständige Test-Daten-Evaluation:")
            full_summary = full_results.loc[:, ['Model', 'Precision', 'Recall', 'F1-Score', 'Extracted_Count']].round(4)
            print(full_summary.to_markdown(index=False))
    
    # 6. Aufräumen der temporären Dateien
    import os
    for model_name in models_to_evaluate.keys():
        temp_file = f"temp_{model_name.lower()}_gold.align"
        try:
            os.remove(temp_file)
            print(f"Temporäre Datei entfernt: {temp_file}")
        except FileNotFoundError:
            pass