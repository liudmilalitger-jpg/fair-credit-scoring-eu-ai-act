# Setup – fair-credit-scoring-eu-ai-act

---

## 1. GitHub Repository einrichten

> Einmalig, nur beim ersten Mal nötig.

```bash
mkdir ~/Developer/fair-credit-scoring-eu-ai-act
cd ~/Developer/fair-credit-scoring-eu-ai-act
brew install gh
gh auth login
git init
```

| Befehl | Was er tut |
|---|---|
| `mkdir` | Erstellt den Projektordner |
| `cd` | Wechselt in den Ordner |
| `brew install gh` | Installiert GitHub CLI |
| `gh auth login` | Öffnet den Browser zur Authentifizierung |
| `git init` | Erstellt ein leeres Git-Repository |

---

## 2. Erste Datei anlegen und hochladen

```bash
echo "# fair-credit-scoring-eu-ai-act" > README.md
git add .
git commit -m "initial commit"
gh repo create fair-credit-scoring-eu-ai-act --public --source=. --remote=origin --push
```

**Schritt für Schritt:**

- `echo "..." > README.md` schreibt den Text in eine neue Datei. Das `>` bedeutet "schreib das in diese Datei". Ohne `>` würde der Text nur im Terminal angezeigt.
- `git add .` merkt alle Dateien für den nächsten Commit vor. Stell dir vor: du packst einen Koffer. `git add` legt die Sachen rein, `git commit` schließt den Koffer und klebt ein Etikett drauf.
- `git commit -m "..."` speichert den aktuellen Stand als Snapshot mit einer Nachricht.
- `gh repo create ...` erstellt das Repo auf GitHub, verbindet es mit dem lokalen Ordner und lädt alles hoch.

---

## 3. Python-Umgebung erstellen

```bash
conda create -n fair-credit python=3.10
conda activate fair-credit
```

Eine eigene Umgebung verhindert Konflikte mit anderen Projekten. `-n fair-credit` ist der Name der Umgebung. In VS Code diese Umgebung als Kernel auswählen (oben rechts im Notebook).

---

## 4. Pakete installieren

```bash
touch requirements.txt
pip install -r requirements.txt
```

Inhalt von `requirements.txt`:

```
ucimlrepo
scikit-learn
shap
fairlearn
matplotlib
seaborn
```

| Paket | Wozu |
|---|---|
| `ucimlrepo` | Lädt das German Credit Dataset direkt von UCI |
| `scikit-learn` | Baut das ML-Modell (Random Forest) |
| `shap` | Erklärt warum das Modell eine Entscheidung getroffen hat |
| `fairlearn` | Misst Bias und Fairness-Metriken |
| `matplotlib` + `seaborn` | Erstellt Grafiken und Visualisierungen |

`-r` steht für "read from file", also "lies die Pakete aus dieser Datei".

---

## 5. Jupyter Notebook erstellen

```bash
touch fair_credit_scoring.ipynb
```

Jupyter-Extension in VS Code installieren, dann die Datei öffnen und den fair-credit Kernel auswählen.

---

## Code-Erklärungen

### Dataset laden

```python
X = dataset.data.features
y = dataset.data.targets
```

- `X` sind die Eingabedaten: alles was das Modell zum Entscheiden benutzt (Alter, Kreditbetrag, Beschäftigung usw.) – 20 Spalten, 1000 Zeilen.
- `y` ist die Zielvariable: gut oder schlechter Kredit – 1000 Werte, eine Spalte.
- Kurzform: X = "was wissen wir über die Person", y = "was ist passiert".

---

### Zielvariable umwandeln

Das Original speichert y so: `1 = guter Kredit`, `2 = schlechter Kredit`. ML-Modelle erwarten Standard `0/1`.

```python
y = dataset.data.targets.values.ravel()
y = (y == 2).astype(int)
```

`ravel()` macht aus einem 2D-Array (`[[1],[2],[1]]`) ein flaches 1D-Array (`[1, 2, 1]`).

`== 2` fragt für jeden Wert: "Ist das eine 2?" → Ergebnis: `[False, True, False]`

`.astype(int)` wandelt Boolean in Zahlen um: `False → 0`, `True → 1`

Endergebnis: `0 = guter Kredit`, `1 = schlechter Kredit`.

---

### Modell trainieren

```python
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

- `LabelEncoder` wandelt Text-Werte (A91, A92...) in Zahlen um, weil Modelle nicht mit Text rechnen können.
- `train_test_split` teilt die Daten auf: 80% Training (800 Zeilen), 20% Test (200 Zeilen). `test_size=0.2` bedeutet 20% für den Test.
- `random_state=42` sorgt dafür dass die Aufteilung immer gleich ist. Die Zahl 42 ist ein Witz aus "Per Anhalter durch die Galaxis" – jede andere Zahl funktioniert genauso. Die ML-Community hat das irgendwann übernommen und seitdem nutzt es fast jeder.
- `RandomForestClassifier` baut 100 Entscheidungsbäume. Jeder lernt aus leicht anderen Daten, am Ende stimmen alle ab. Mehrheit gewinnt.
- `model.fit(...)` ist das eigentliche Lernen. `model.score(...)` zeigt wie oft das Modell auf den 200 Testdaten richtig lag.

---

### Jetzt noch auf GitHub pushen:


```
`echo "setup.md" >> .gitignore
git add .
git commit -m "Add full analysis: SHAP, Fairlearn bias detection, ExponentiatedGradient mitigation"
git push`

```
