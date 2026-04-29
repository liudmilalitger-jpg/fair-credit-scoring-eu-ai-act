# Fair Credit Scoring & Bias Analysis under the EU AI Act

This project demonstrates how a machine learning model used in credit scoring can be audited for bias, and how detected bias patterns can be actively mitigated — following the requirements of the EU AI Act.

**Dataset:** Statlog German Credit Data (UCI ML Repository, ID 144) — 1,000 credit applications, 20 features, collected in Germany in 1994. A widely used benchmark in fairness research in the financial sector.

**Why this matters:** Credit scoring systems fall under the **high-risk category** of the EU AI Act (Annex III). This means: mandatory risk analysis, documentation of training data, and proof of fairness measures — before such a system can go live.

---

## Key Findings

**The model achieves 81.5% overall accuracy — and still systematically discriminates.**

### Bias by Age

| Group | Accuracy | Selection Rate | False Positive Rate |
|---|---|---|---|
| 25–45 | 84.0% | 13.6% | 2.2% |
| under 25 | 80.5% | 41.5% | **21.4%** |
| over 45 | 73.5% | 17.6% | 4.8% |

Young people (under 25) are classified as credit risk **3× more often** than the 25–45 group. Their false positive rate is **10× higher** — meaning they are wrongly rejected far more frequently despite being creditworthy.

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
├── fair_credit_scoring.ipynb   # Main analysis notebook
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

## Core Message

A credit scoring model can achieve 81.5% overall accuracy and still wrongly reject young people ten times more often than middle-aged applicants. **Overall accuracy is not a fairness guarantee.**

The EU AI Act mandates the measurement, documentation, and remediation of such patterns for high-risk AI systems. This notebook demonstrates what that looks like in practice — from raw data to audit-ready fairness evidence.

---

*Dataset: Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77*
