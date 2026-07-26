# Where things are

A quick map of the app. The tabs run along the top; the **Selected corpus**
dropdown (top-right) is app-wide: it sets which knowledge base every tab works
on.

Jump to: [Search](#search) · [Library](#library) · [Evaluation](#evaluation) ·
[Installs](#installs) · [Keys](#keys) · [Settings](#settings) ·
[Info icons](#info-icons)

<a id="search"></a>
## Search

Ask questions of the selected corpus. The **chat** is on the left; the right
column has its own sub-tabs:

- **View:** the answer (with citations) or an opened result file, plus a pager to
  step back through recent results.
- **Retrieval:** per-query search knobs (how many results, hybrid vs
  vector-only, MMR, direct Cypher).
- **LLMs:** which model answers, and its settings.
- **Info:** search tips, including Cypher.

<a id="library"></a>
## Library

Build and manage corpora.

- **Create New:** start a new corpus (its own knowledge base).
- **Ingest:** add documents; the app chunks + embeds them and builds the graph.
- **Metadata:** browse and edit what's indexed.
- **Info:** how the Library works.

<a id="evaluation"></a>
## Evaluation

Measure retrieval and answer quality.

- **Run evaluation** and **Create test cases** author and run evals; **Info**
  explains how to use the harness.
- The dashboards (**Run Summary**, **Run Charts**, **Compare Datasets**,
  **Trends**) visualise the results.
- The **Metrics Guide** defines what each metric means.

<a id="installs"></a>
## Installs

Install and manage the pieces retrieval needs: LLM providers, embedders, entity
extractors, document parsers, and ontologies.

<a id="keys"></a>
## Keys

Store API keys for the providers you use (for example Anthropic or Voyage). Keys
are kept in your operating system's keyring, not in the project folder.

<a id="settings"></a>
## Settings

App behaviour, database connections (Neo4j + LanceDB), and where saved results and
chats are written.

<a id="info-icons"></a>
## Info icons (i)

Throughout the app, small **(i)** icons give contextual help next to a control,
useful for *what a specific setting does*. Turn them on or off with the
**info-icons toggle in Settings**.
