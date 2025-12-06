# FlowTransformer Experimente zur Netzwerkeinbruchserkennung

Dieses Projekt untersucht die Anwendung von Transformer-Modellen zur Erkennung von Netzwerkeinbrüchen (Network Intrusion Detection System - NIDS). Basierend auf dem **FlowTransformer-Framework** werden verschiedene Architekturen auf dem **UNSW-NB15-Datensatz** trainiert und evaluiert.

> **Hinweis:** Die hier vorliegenden Notebooks basieren auf den originalen Demonstrationen (`demonstration.ipynb` und `FlowTransformer_demo.ipynb`) aus dem [FlowTransformer GitHub Repository](https://github.com/liamdm/FlowTransformer). Sie wurden für diesen spezifischen Anwendungsfall und Datensatz angepasst und erweitert.

## 📂 Projektstruktur

Der Ordner ist wie folgt aufgebaut. Die Verzeichnisse `framework` und `implementations` enthalten dabei den Quellcode des FlowTransformer-Frameworks selbst.

```text
FlowTransformer/
├── framework/                                 # Kern-Quellcode des FlowTransformer Frameworks
├── implementations/                           # Spezifische Implementierungen von Modellkomponenten
├── demonstration/                             # Zielordner für den UNSW-NB15 Datensatz
├── basictransformer.keras                     # Gespeichertes Modell (Bestes Ergebnis)
├── content/cache_folder/                      # Cache für Zwischenspeicherungen
├── tiny_bert.py                               # Wrapper-Klasse für das TinyBERT-Experiment
├── FlowTransformer_Pretraining_BasicTransformer.ipynb  # Experiment 1 (Erfolgreich)
├── FlowTransformer_Pretraining_BERT.ipynb              # Experiment 2 (Underfitting)
├── FlowTransformer_Finetuning_TinyBERT.ipynb           # Experiment 3 (Fehlerhaft)
└── README.md                                  # Projektdokumentation
```

## Installation

**Repository klonen und Abhängigkeiten installieren**: Da die Quellcodes (framework, implementations) bereits im Ordner enthalten sind, stellen Sie sicher, dass die notwendigen Python-Bibliotheken (TensorFlow/Keras, Pandas, NumPy, etc.) installiert sind. Wenn Sie das Framework frisch installieren wollen:

Bash

```bash
git clone https://github.com/liamdm/FlowTransformer.git
cd FlowTransformer
pip install .
```

Laden sie ebenfalls den NF-UNSW-NB16-v3 [hier](https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f/) herunter und packen sie ihn in den Ordner `/demonstration`

## Notebooks

### 1. `FlowTransformer_Pretraining_BasicTransformer.ipynb`

Dieses Notebook dokumentiert das erste und erfolgreichste Experiment.

- **Modell:** Ein `BasicTransformer` mit einer einfachen Architektur.
- **Ergebnis:** Das Modell erzielt eine hervorragende Leistung mit einer **ausgewogenen Genauigkeit von 99,81 %** und einem **F1-Score von 0,992**. Dies zeigt die Wirksamkeit des FlowTransformer-Frameworks und des `BasicTransformer`-Modells für diese Aufgabe.

### 2. `FlowTransformer_Pretraining_BERT.ipynb`

Dieses Notebook experimentiert mit einer komplexeren, vordefinierten BERT-ähnlichen Architektur.

- **Modell:** Ein `BERTSmallTransformer`.
- **Ergebnis:** Das Modell lernt nicht effektiv und erreicht nur eine ausgewogene Genauigkeit von **50 %** und einen F1-Score von **0**. Dies deutet darauf hin, dass die `BERTSmallTransformer`-Architektur in der konfigurierten Form für diesen Datensatz nicht gut geeignet ist.

### 3. `FlowTransformer_Finetuning_TinyBERT.ipynb`

Dieses Notebook versucht, ein vorab trainiertes `TinyBERT`-Modell zu optimieren.

- **Modell:** Ein `TinyBERT`-Modell, das den `prajjwal1/bert-tiny`-Checkpoint von Hugging Face verwendet.
- **Ergebnis:** Das Experiment schlägt während der Trainingsphase aufgrund eines `ValueError` fehl. Dies deutet auf ein Problem mit der Implementierung des Modells oder seiner Kompatibilität mit der Trainingspipeline hin.

## Unterstützende Dateien

### `tiny_bert.py`

Diese Datei enthält die Implementierung der `TinyBERT`-Klasse, die ein Wrapper um das `bert-tiny`-Modell von Hugging Face ist. Sie ermöglicht die Verwendung des vorab trainierten Modells im FlowTransformer-Framework und ermöglicht die Feinabstimmung, indem die Gewichte des Modells trainierbar gemacht werden.

## Fazit

Die Experimente zeigen, dass ein speziell erstelltes, kleineres Transformer-Modell (`BasicTransformer`) ein komplexeres, allgemeineres BERT-Modell (`BERTSmallTransformer`) bei der Netzwerkeinbruchserkennung mit dem UNSW-NB15-Datensatz übertreffen kann. Der Versuch, ein `TinyBERT`-Modell zu optimieren, war aufgrund eines Laufzeitfehlers erfolglos.

Der erfolgreichste Ansatz ist in `FlowTransformer_Pretraining_SmallTransformer.ipynb` dokumentiert.
