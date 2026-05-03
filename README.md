# Fair Credit Scoring & Bias Analysis under the EU AI Act

This project demonstrates how a machine learning model used in credit scoring can be audited for bias, and how detected bias patterns can be actively mitigated, following the requirements of the EU AI Act.

**Dataset:** Statlog German Credit Data (UCI ML Repository, ID 144), 1,000 credit applications, 20 features, collected in Germany in 1994. A widely used benchmark in fairness research in the financial sector.

**Why this matters:** Credit scoring systems fall under the **high-risk category** of the EU AI Act (Annex III). This means: mandatory risk analysis, documentation of training data, and proof of fairness measures, before such a system can go live.

---

## Key Findings

**Baseline model (before mitigation) achieves 81.5% overall accuracy, and still systematically discriminates.**

### Bias by Age (Baseline, before Reweighing)

| Group | Accuracy | Selection Rate | False Positive Rate |
|---|---|---|---|
| 25-45 | 84.0% | 13.6% | 2.2% |
| under 25 | 80.5% | 41.5% | **21.4%** |
| over 45 | 73.5% | 17.6% | 4.8% |

Young people (under 25) are classified as credit risk **3× more often** than the 25-45 group. Their false positive rate is **10× higher**, meaning they are wrongly rejected far more frequently despite being creditworthy.

**After Reweighing (MA-02):** FPR-Differenz reduced from 19% to 7.1%, within the target interval of max. 10%. Overall accuracy: 76.5%.

### Bias by Marital Status & Gender (personal_status)

| Group | Accuracy | False Positive Rate |
|---|---|---|
| male, divorced | **68.7%** | **11.1%** |
| female | 82.1% | 5.1% |
| male, single | 82.4% | 5.2% |
| male, married | 85.0% | 12.5% |

Counterintuitive finding: bias does not run along the expected male vs. female line, but along marital status. Single people and women are treated more fairly than married or divorced men.

Overall accuracy alone would never reveal either of these patterns.

---

## Project Structure

```
fair-credit-scoring-eu-ai-act/
├── fair_credit_scoring_v2.ipynb # Main analysis notebook (full Fraunhofer roadmap)
├── requirements.txt            # Python dependencies
├── setup.md                    # Setup documentation & code explanations
└── README.md
```

---

## What the Notebook Does

**1. Data Loading & Preparation**
Loads the German Credit Dataset via the official UCI API. Identifies the four demographically marked features: age, marital status/gender, employment, and foreign worker status.

**2. Model Training**
Trains a Random Forest Classifier (100 trees, 80/20 train-test split). Achieves 81.5% accuracy on the test set.

**3. Explainability with SHAP**
Uses SHAP (SHapley Additive exPlanations) to identify which features drive individual predictions. `checking_account` is the strongest predictor; `age` and `personal_status` have measurable influence despite being sensitive attributes.

**4. Fairness Analysis with Microsoft Fairlearn**
Measures accuracy, selection rate, and false positive rate across demographic groups (age and marital status). Quantifies the extent of disparate treatment per group.

**5. Bias Mitigation**
Applies `ExponentiatedGradient` with `EqualizedOdds` as a fairness constraint. Trains the model with adjusted sample weights to balance performance across age groups. Documents the accuracy-fairness trade-off.

**6. EU AI Act Mapping**
Maps each notebook step to the corresponding EU AI Act article requirement.

---

## EU AI Act Compliance Mapping

| Step | EU AI Act Article | Requirement |
|---|---|---|
| Data loading & metadata review | Art. 10 (2) | Data quality and representativeness |
| Identifying demographic features | Art. 10 (2)(f) | Detecting bias from sensitive attributes |
| SHAP explainability | Art. 13 | Transparency and traceability |
| Fairness analysis (MetricFrame) | Art. 10 (2)(f) | Measuring and documenting bias per group |
| Bias mitigation (ExponentiatedGradient) | Art. 9 | Risk management: implementing countermeasures |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `ucimlrepo` | Load German Credit Dataset from UCI |
| `scikit-learn` | Random Forest model, preprocessing, metrics |
| `shap` | Feature importance & explainability |
| `fairlearn` | Fairness metrics and bias mitigation |
| `matplotlib` + `seaborn` | Visualizations |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/liudmila-litger/fair-credit-scoring-eu-ai-act.git
cd fair-credit-scoring-eu-ai-act

# 2. Create and activate conda environment
conda create -n fair-credit python=3.10
conda activate fair-credit

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open notebook in VS Code
# Select "fair-credit" as the kernel
```

---

## Fraunhofer KI-Prüfkatalog: Dimension Fairness (FN)

### Schutzbedarf-Analyse

| Stufe | Beschreibung | Beispiele |
|---|---|---|
| **Hoch** | KI entscheidet über etwas Wesentliches. Persönlichkeitsrechte direkt betroffen. | Kredit, Visum, Studienplatz, Behandlung |
| **Mittel** | KI verarbeitet Personendaten, aber die Ausgabe hat keine großen Konsequenzen. | Spracherkennung, Altersschätzung auf Fotos |
| **Gering** | Keine sensiblen Personendaten. | Maschinenausfall-Prognose, Werbung |

**Dieses Projekt: Schutzbedarf HOCH** (Kreditvergabe, EU AI Act Anhang III)

---

### Risikoanalyse

**Bias** ist das Problem in den Daten oder im Modell. Es entsteht bevor das Modell eine Entscheidung trifft. Beispiel: Im Trainingsdatensatz wurden junge Menschen historisch häufiger abgelehnt. Das Modell lernt dieses Muster, nicht weil junge Menschen schlechtere Zahler sind, sondern weil die Daten das so zeigen.

**Fairness** ist die Frage: Wie stark wirkt sich dieser Bias auf die Entscheidungen aus? FPR 21% bei unter 25 vs. 2% bei 25-45 ist Unfairness im Output.

---

### Planung (KR)

**KR-01: Was messen wir im Modell-Output?**
- Gruppen: Altersgruppen (unter 25, 25-45, über 45), Geschlecht
- Fairness-Definition: Equalized Odds, weil fälschliche Ablehnung und fälschliche Genehmigung beide relevant sind
- Metrik: FPR pro Gruppe (`false_positive_rate`)
- Zielintervall: FPR-Differenz zwischen Gruppen max. 10%

**KR-02: Was messen wir in den Trainingsdaten?**
- Bias-Maß: Disparate Impact Ratio, Klassenverteilung pro Gruppe
- Zielintervall: Disparate Impact Ratio mind. 0.8, keine Gruppe unter 10% Repräsentation

---

### Ausführung (MA)

**MA-01: Daten auf Bias prüfen**
- EDA mit pandas: Datenpunkte pro Altersgruppe, Klassenverteilung, fehlende Werte
- Bias-Maße berechnen (aus KR-02): Disparate Impact Ratio pro Gruppe
- Statistischer Test: Chi-Quadrat-Test

**MA-02: Faire Datenvorverarbeitung**
- Reweighing: junge Menschen mit gutem Kredit bekommen höheres Gewicht beim Training
- Begründung: passt zu Equalized Odds aus KR-01, adressiert FPR-Differenz direkt

**MA-03: Modell trainieren und erklären**
- Modell: Random Forest, 100 Bäume, scikit-learn
- SHAP nach dem Training: Kontostand ist wichtigstes Feature (sachlich gerechtfertigt), Alter auf Platz 9 und Familienstand auf Platz 10 haben messbaren Einfluss trotz fehlendem sachlichem Zusammenhang mit Kreditwürdigkeit

**MA-04: Bias aktiv bekämpfen**
- In-Processing: ExponentiatedGradient mit EqualizedOdds-Constraint
- Post-Processing: ThresholdOptimizer, Entscheidungsschwelle pro Gruppe anpassen
- Trade-off: FPR-Differenz nach Reweighing bereits auf 7.1% (Zielintervall erreicht), Accuracy 76.5%. ExponentiatedGradient als zusätzliche Absicherung dokumentiert.

**MA-05: Fairness auf Testdaten messen**
- Fairlearn MetricFrame auf 200 ungesehenen Testdaten
- Ergebnis gegen Zielintervall aus KR-01 prüfen und dokumentieren

**MA-06: Faire Weiterverarbeitung**
- Sachbearbeiter trifft finale Entscheidung. Risiko: manuelle Overrides gruppenspezifisch
- Maßnahme: Audit-Log pro Entscheidung, quartalsweise Auswertung

**MA-07: Gesamtsystem testen**
- Gesamter Prozess: Dateneingabe, Score, Sachbearbeiter, finale Entscheidung
- Testdaten: mind. 50 Fälle pro Altersgruppe

**MA-08: Monitoring im Betrieb**
- Monatlich FPR pro Gruppe auf neuen Produktivdaten berechnen
- Bei FPR-Differenz über 15%: automatischer Alert, Modell einfrieren bis Review

---

### Beherrschung der Dynamik (BD)

Fairness hört nicht nach dem Deployment auf. Das Modell kann durch neue Daten oder veränderte Gesetze wieder unfair werden.

**Model Drift:** neue Trainingsdaten bringen neuen Bias. Beispiel: Crowdsourcing-Labels spiegeln gesellschaftliche Trends wider.

**Concept Drift:** Gesetze ändern sich. Beispiel: Geschlecht darf nicht mehr für Versicherungstarife genutzt werden.

| BD-Schritt | Maßnahme |
|---|---|
| BD-MA-01 | Neue Trainingsdaten regelmäßig auf Bias prüfen, bevor sie ins Modell fließen |
| BD-MA-02 | Modell-Output in festgelegten Intervallen auf Fairness-Metriken messen |
| BD-MA-03 | Prozess bei Unfairness: Modell verbessern ohne neue Diskriminierung zu erzeugen |
| BD-MA-04 | Externe Faktoren beobachten: neue Gesetze, gesellschaftliche Entwicklungen |

---

### Fairness-Definitionen

| Definition | Frage | Metrik | Wann sinnvoll |
|---|---|---|---|
| Demographic Parity | Bekommt jede Gruppe gleich oft ein positives Ergebnis? | Selection Rate, Disparate Impact Ratio | Stellenanzeigen, kein sachlicher Grund für Unterschied |
| Equalized Odds | Macht das Modell bei jeder Gruppe gleich oft Fehler in beide Richtungen? | FPR + FNR pro Gruppe | Kreditvergabe, beide Fehlerrichtungen relevant |
| Equal Opportunity | Haben kreditwürdige Personen in jeder Gruppe gleiche Chance? | TPR pro Gruppe | Fokus nur auf fälschliche Ablehnung |
| Predictive Parity | Wenn das Modell "positiv" sagt, stimmt das für alle Gruppen gleich oft? | Precision pro Gruppe | Score-basierte Systeme wie SCHUFA |
| Calibration | Stimmt die vorhergesagte Wahrscheinlichkeit mit der Realität überein? | Calibration Score pro Gruppe | Wenn Modell Wahrscheinlichkeit ausgibt |

---

### Fairness-Metriken im Output

| Metrik | Frage | Python |
|---|---|---|
| False Positive Rate (FPR) | Wie oft wird jemand fälschlicherweise abgelehnt, obwohl er kreditwürdig ist? | `from fairlearn.metrics import false_positive_rate` |
| False Negative Rate (FNR) | Wie oft wird jemand fälschlicherweise genehmigt, obwohl er nicht kreditwürdig ist? | `from fairlearn.metrics import false_negative_rate` |
| Accuracy pro Gruppe | Wie oft liegt das Modell bei jeder Gruppe richtig? | `from sklearn.metrics import accuracy_score` |
| Selection Rate | Wie oft bekommt eine Gruppe ein positives Ergebnis? | `from fairlearn.metrics import selection_rate` |
| Disparate Impact Ratio | Selection Rate Gruppe A geteilt durch Gruppe B, Zielwert mind. 0.8 | `from fairlearn.metrics import demographic_parity_ratio` |

### Bias-Masse in Trainingsdaten

| Mass | Frage | Python |
|---|---|---|
| Klassenverteilung pro Gruppe | Wie oft kommt jede Klasse pro Gruppe vor? | `df.groupby('age_group')['credit'].value_counts(normalize=True)` |
| Disparate Impact Ratio | Anteil positiver Labels Gruppe A geteilt durch Gruppe B | `from fairlearn.metrics import demographic_parity_ratio` |
| Pearson-Korrelation | Korreliert ein sensitives Merkmal mit dem Label? | `df['age'].corr(df['credit'])` |
| Repräsentationsrate | Wie viel Prozent macht jede Gruppe aus? | `df['age_group'].value_counts(normalize=True)` |

### Statistische Tests

| Test | Frage | Python |
|---|---|---|
| Chi-Quadrat-Test | Sind zwei kategoriale Variablen unabhängig? | `from scipy.stats import chi2_contingency` |
| t-Test | Unterscheiden sich Mittelwerte zwischen zwei Gruppen? | `from scipy.stats import ttest_ind` |
| ANOVA | Wie t-Test, aber für mehr als zwei Gruppen | `from scipy.stats import f_oneway` |
| Mann-Whitney U | Wie t-Test, aber wenn Daten nicht normalverteilt sind | `from scipy.stats import mannwhitneyu` |

---

## Core Message

A credit scoring model can achieve 81.5% overall accuracy and still wrongly reject young people ten times more often than middle-aged applicants. **Overall accuracy is not a fairness guarantee.**

The EU AI Act mandates the measurement, documentation, and remediation of such patterns for high-risk AI systems. This notebook demonstrates what that looks like in practice, from raw data to audit-ready fairness evidence.

---

*Dataset: Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77*
