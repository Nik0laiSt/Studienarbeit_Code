# TinyBERT für Netzwerkeinbruchserkennung

Dieses Projekt demonstriert die Verwendung eines TinyBERT-Modells zur Netzwerkeinbruchserkennung (NIDS). Das Ziel ist es, Netzwerkverkehrsflüsse in verschiedene Angriffskategorien oder als gutartig zu klassifizieren.

## Projektstruktur

Das Projekt ist in die folgenden Hauptverzeichnisse und -dateien gegliedert:

- `data/`: Dieses Verzeichnis sollte die Datensatzdatei enthalten.
- `src/`: Dieses Verzeichnis enthält den Quellcode, einschließlich Jupyter-Notebooks für die verschiedenen Phasen des Projekts.
- `*.pkl`: Gespeicherte Modelle oder Skalierer.
- `attack_category_mapping.npy`: Eine Zuordnung von Angriffskategorienamen zu ihrer numerischen Darstellung.

### Quelldateien

- `01_Explorative_Datenanalyse.ipynb`: Führt eine erste explorative Datenanalyse des Datensatzes durch.
- `02_Daten-Preprocessing.ipynb`: Bereinigt und verarbeitet die Daten für das Modelltraining.
- `03_TinyBERT_FineTuning.ipynb`: Feinabstimmung des TinyBERT-Modells auf den vorverarbeiteten Daten.
- `04_TinyBERT_Demonstration_Evaluation.ipynb`: Demonstriert die Verwendung des trainierten Modells zur Inferenz und bewertet seine Leistung.
- `05_TinyBERT_Explainability_Research.ipynb`: Untersucht die Erklärbarkeit der Modellvorhersagen.
- `requirements.txt`: Eine Liste der Python-Abhängigkeiten, die zum Ausführen des Projekts erforderlich sind.
- `tinybert_with_gap.py`: Ein benutzerdefiniertes Python-Skript, das die Modellarchitektur enthält.

## Datensatz

Dieses Projekt verwendet den NF-UNSW-NB15-v3-Datensatz. Sie können den Datensatz unter folgendem Link herunterladen:

[https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f](https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f)

Legen Sie die Datei `NF-UNSW-NB15-v3.csv` nach dem Herunterladen im Verzeichnis `data/` ab.

## Erste Schritte

### Voraussetzungen

Um dieses Projekt auszuführen, müssen Sie Python 3 und die in `requirements.txt` aufgeführten Abhängigkeiten installiert haben.

Sie können die Abhängigkeiten mit pip installieren:

```bash
pip install -r src/requirements.txt
```

### Einrichtung

1.  Klonen Sie dieses Repository auf Ihren lokalen Rechner.
2.  Laden Sie den NF-UNSW-NB15-v3-Datensatz herunter und legen Sie ihn im Verzeichnis `data/` ab.

## Arbeitsablauf

Das Projekt ist in eine Reihe von Jupyter-Notebooks unterteilt, die in der folgenden Reihenfolge ausgeführt werden sollten:

1.  **`01_Explorative_Datenanalyse.ipynb`**: Dieses Notebook bietet einen ersten Einblick in den Datensatz, einschließlich seiner Struktur, Merkmale und der Verteilung der Zielvariable.
2.  **`02_Daten-Preprocessing.ipynb`**: Dieses Notebook befasst sich mit der Datenbereinigung, dem Feature-Engineering und der Umwandlung der Daten in ein für das TinyBERT-Modell geeignetes Format.
3.  **`03_TinyBERT_FineTuning.ipynb`**: Dieses Notebook stimmt das TinyBERT-Modell auf den vorverarbeiteten Daten ab. Es beinhaltet eine Hyperparameter-Optimierung und speichert das beste Modell.
4.  **`04_TinyBERT_Demonstration_Evaluation.ipynb`**: Dieses Notebook zeigt, wie das feinabgestimmte Modell geladen und zur Vorhersage auf neuen Daten verwendet wird. Es enthält auch eine detaillierte Bewertung der Modellleistung.
5.  **`05_TinyBERT_Explainability_Research.ipynb`**: Dieses Notebook untersucht Methoden zum Verständnis und zur Interpretation der Modellvorhersagen.

## Ausführen der Anwendung

Um die Anwendung auszuführen, müssen Sie die Jupyter-Notebooks im Verzeichnis `src/` in der oben beschriebenen Reihenfolge ausführen. Sie können Jupyter Notebook oder JupyterLab verwenden, um die Notebooks zu öffnen und auszuführen.

## Bewertung

Die Leistung des Modells wird anhand verschiedener Metriken bewertet, darunter:

-   Genauigkeit
-   Präzision
-   Recall
-   F1-Score
-   Klassifizierungsbericht
-   Konfusionsmatrix

Die Bewertung wird im Notebook `04_TinyBERT_Demonstration_Evaluation.ipynb` durchgeführt.

## Modell

Das endgültige trainierte Modell wird in dem im Notebook `03_TinyBERT_FineTuning.ipynb` angegebenen Verzeichnis gespeichert. Dies umfasst die Modellgewichte, die Konfiguration und den Tokenizer.
