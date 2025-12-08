# LLM Extractor & RAG App

Ein modularer **Retrieval-Augmented-Generation (RAG)**-Stack mit zwei Index-Pipelines (OCR-basiert und LLM-Extraktion) sowie einem zweistufigen Retrieval-Prozess mit LLM-Kompression. Die Lösung ist auf schnelle Experimente ausgelegt und setzt auf **LangGraph/LangChain** und einen **Chroma**-Vektorstore. Alle Modell- und Storage-Provider sind austauschbar.

## Schnellüberblick
- **Zwei Indexer**: OCR-Extraktion mit Metadaten-Anreicherung und LLM-basierte Extraktion/Splitting.
- **Retrieval**: 2-Stage-Retriever mit LLM-Kompression für fokussierte Antworten.
- **LangGraph Dev/Studio**: Graphen lassen sich lokal per `langgraph dev` ausführen und visuell steuern.
- **Beispieldaten**: CANCOM-Dokumente sind bereits in `.chroma_directory` indiziert und sofort nutzbar.

## Architektur
### Index-Prozesse
| Pipeline | Diagramm | Studio-Ansicht |
| --- | --- | --- |
| **OCR-Indexer**: Extrahiert Text aus PDF-Seiten (PNG-Bilder), reichert ihn mit LLM-Metadaten an und persistiert in Chroma. | ![UML: OCR-Indexer](/docs/index_ocr_uml.png) | ![Studio: OCR-Indexer](/docs/ocr_indexer_studio.png) |
| **LLM-Indexer**: Nutzt LLM für Text-Extraktion und Chunking, ergänzt Metadaten und schreibt in Chroma. | ![UML: LLM-Indexer](/docs/index_llm_uml.png) | ![Studio: LLM-Indexer](/docs/llm_indexer_studio.png) |

### Retrieval-Prozess (2-Stage mit Kompression)
- Erstes Ranking via Vektor-Ähnlichkeit, gefolgt von einer LLM-Kompression auf relevante Passagen.
- Unterstützt mehrsprachige Queries und kann sowohl OCR- als auch LLM-Indexe ansprechen.

| Diagramm | Studio-Ansicht |
| --- | --- |
| ![UML: Retriever](/docs/retriever_uml.png) | ![Studio: Retriever](/docs/retriever_studio.png) |

## Voraussetzungen
- **Python 3.12** (empfohlen) und `virtualenv` oder `venv`.
- **poppler** für `pdf2image` (Pfad ggf. in `.env` als `POPLER_PATH` setzen; Windows-Build: https://github.com/oschwartz10612/poppler-windows/releases/).
- API-Keys über `.env` bereitstellen (mindestens `OPENAI_API_KEY`; optional `LANGSMITH_API_KEY`).

## Installation & Start
```bash
python -m venv .venv          # alternativ: py -3.12 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .[dev]
langgraph dev                 # startet lokale LangGraph-Dev/Studio-Oberfläche
```

## Graph-Ausführung in LangGraph Studio
1. **Graph wählen:** In der UI zwischen `index_graph` (OCR/LLM-Index) und `retrieval_graph` wechseln.
2. **Parameter setzen:** `collection_id`, `doc_id` und optional `chunking_config` an eure Daten anpassen.
3. **Run starten:** In der Studio-Ansicht den gewünschten Node ausführen oder den gesamten Graphen triggern.

### Beispielaufrufe
**Retrieval-Graph** – vorhandene Collection `./data/DB_ZB25.pdf` nutzen:
- `collection_id`: `DB_ZB`
- `doc_id`: `DB_ZB_S55` (`95776bb8-7907-486c-b1fb-04e456c44e2c`)
- `query` Beispiele:
  - „Was macht die DB in Bezug auf Nachhaltigkeit?“
  - „Wie viele Arbeitskräfte hat der DB Konzern auf der ganzen Welt?“

**Indexer** – bestehende Collections (bereits in `.chroma_directory` gespeichert):
- **LLM-Indexer** `collection_id: Cancom_LLM`
  - `doc_id: Cancom_240514`, `path: ./data/Cancom/240514_CANCOM_Zwischenmitteilung.pdf`
  - `doc_id: Cancom_20241112`, `path: ./data/Cancom/20241112_CANCOM_Zwischenmitteilung.pdf`
- **OCR-Indexer** `collection_id: Cancom_OCR`
  - `doc_id: Cancom_240514`, `path: ./data/Cancom/240514_CANCOM_Zwischenmitteilung.pdf`
  - `doc_id: Cancom_20241112`, `path: ./data/Cancom/20241112_CANCOM_Zwischenmitteilung.pdf`

**Retrieval-Queries**
- Ganze Collection abfragen: `collection_id = Cancom_LLM` oder `Cancom_OCR`, `doc_id = None`
  - Beispiel-Frage: „Wie beschreibt CANCOM in den Zwischenmitteilungen Q1 2024 und Q3 2024 die Auswirkungen der Übernahme der KBC- / CANCOM-Austria-Gruppe auf Umsatz, Rohertrag und EBITDA?“
- Einzelnes Dokument abfragen: `collection_id = Cancom_LLM/OCR`, `doc_id = Cancom_240514`
  - Beispiel-Fragen:
    - „Wo hat die Cancom SE ihren Sitz, und unter welcher Telefonnummer ist sie erreichbar?“ (Evidenz Seite 13)
    - „Wie hoch ist der prozentuale und absolute Unterschied der Vorräte zwischen dem 31.12.2023 und dem 31.03.2024?“ (Evidenz Seite 18)

## Betriebs- und Troubleshooting-Hinweise
- **Sprache setzen:** Suchsprache passend zur Dokumentensprache einstellen, auch für multimodale Embeddings.
- **Extraktion:** Jeder Segment-Metadaten-Eintrag enthält eine Sprachkennung; Modelle sollen in dieser Sprache antworten.
- **Rate Limits:** Unterschiedliche Modelle können eigene Limits haben; optional Rate-Limiter pro Provider setzen (OpenAI-Limits: <https://platform.openai.com/settings/organization/limits>).
- **Logging:** Exceptions aus dem Extractor werden geloggt und im State zurückgegeben.

### Bekannte Einschränkungen
- Große, dicht beschriebene Seiten können zu `LengthFinishReasonError` führen, weil PNG + Prompt die Modell-Länge überschreiten.
- Der OCR-Indexer `extract_text_node` verarbeitet aktuell auch Tabellentext; in Kombination mit `extract_tables` kann es zu doppelten Segmenten kommen.
- Prompts sind generisch gehalten und sollten für neue Domänen angepasst werden.

### Weiterentwicklungsideen
- Optionales LLM-Enrichment im OCR-Graph.
- Alternative 1-Stage-Retriever.
- Ähnlichkeitsscore in Dokument-Metadaten zurückschreiben.
- Prompt-Optimierung und Benchmarking.
- Evaluation auf verschiedenen Testdatensätzen
