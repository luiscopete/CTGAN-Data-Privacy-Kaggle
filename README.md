## 🛡️ CTGAN Synthetic Data Generator: Meta Kaggle User Stats

This repository implements a **Conditional Tabular Generative Adversarial Network (CTGAN)** to create high-fidelity synthetic data based on aggregated user statistics from the official Meta Kaggle dataset.

-----

## 💡 What is CTGAN?

The **CTGAN** is a deep learning model specifically designed for generating realistic **tabular data**. It addresses the limitations of standard Generative Adversarial Networks (GANs) when dealing with real-world tables, which often contain:

  * **Mixed Data Types:** A combination of continuous (e.g., `Followers`) and discrete/categorical (e.g., `Country`) columns.
  * **Irregular Distributions:** Numerical columns that don't follow a simple Gaussian distribution (e.g., income might have two peaks).

The CTGAN's core innovation is using **Conditional Generation** to learn complex relationships and handle imbalanced categories, ensuring that the generated synthetic data maintains the statistical integrity of the original.

-----

## 🔑 Importance in Data Engineering

Using synthetic data offers immense benefits, particularly in environments handling sensitive information:

1.  **Privacy and Compliance (GDPR, HIPAA):** The synthetic dataset contains **no real Personal Identifiable Information (PII)**. It allows developers and analysts to work with production-level data characteristics without the risk of exposing real users, drastically simplifying compliance requirements.
2.  **Agile Development:** Synthetic data can be generated **on-demand** and scaled to any size. This is crucial for:
      * **Pipeline Testing:** Validating ETL/ELT pipelines before deployment.
      * **Bug Replication:** Simulating specific edge cases or rare events that might not exist in small test sets.
3.  **Cost Reduction:** It eliminates the need for expensive, highly-secured infrastructure required for housing and accessing real PII in non-production environments.

-----

## 📊 Dataset Overview

The synthetic model is trained on **Aggregated User Statistics** sourced from the official Meta Kaggle dataset.

  * **Source:** Official Meta Kaggle Dataset (Aggregated User Stats).
  * **Sample Size Used for Training:** 100,000 rows.
  * **Key Columns Modeled:** The model learns the relationships between user metrics, including activity (`TopicsCreated`, `LastContentDate`), social engagement (`Following`, `Followers`), and achievement (`CompetitionMedals`).
  * **PII Handling:** The `UserName` column is explicitly defined as an **ID** field in the metadata (using `metadata.set_primary_key`), ensuring the model generates unique, sequential identifiers rather than attempting to model the distribution of actual usernames.

-----

## ⚙️ Training Process

The training leverages the **SDV (Synthetic Data Vault)** library's optimized CTGAN implementation.

### 1\. Model Initialization

The `CTGANSynthesizer` is initialized with the dataset's **metadata**, which informs the model about which columns are categorical, continuous, or unique IDs. The `enforce_rounding=True` parameter ensures generated integer values remain integers (e.g., `Followers` remains a whole number).

```python
# Initialize the synthesizer
synthesizer = CTGANSynthesizer(
    metadata,
    enforce_rounding=True,
    epochs=200,
    verbose=True
)
```

### 2\. Adversarial Training

The model trains over **200 epochs**, engaging in the adversarial process:

  * The **Generator** ($G$) attempts to produce realistic user data based on the learned correlations.
  * The **Discriminator** ($D$) attempts to differentiate between the real Meta Kaggle data and the synthetic data from $G$.
  * The process converges when the Discriminator can no longer reliably distinguish the two, meaning the synthetic data is high-quality.

-----

## ✅ Evaluation Metric

The quality of the final synthetic dataset is measured using the **Overall Quality Score** provided by the SDV evaluation framework.

  * **Metric:** `sdv.evaluation.single_table.evaluate_quality`
  * **How it Works:** This metric calculates **Statistical Fidelity** (0% to 100%) by averaging the results of two key checks:
    1.  **Column Shapes:** Compares the individual statistical distributions (histograms) of every column in the real and synthetic datasets.
    2.  **Column Pairs:** Compares the correlation matrix to ensure that the relationships between variables (e.g., the link between `GoldCompetitionMedals` and `PerformanceTier`) are preserved.

A score above **80%** typically indicates data suitable for advanced analytics and machine learning tasks.

```python
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

# The result provides the final assessment of the model's success
print(f"\nOverall Quality Score: {quality_report.get_score() * 100:.2f}%")
```

-----

## ✍️ Author

**Luis Copete** — Data Engineer
* **LinkedIn:** [Luis Copete's LinkedIn Profile Link](https://www.linkedin.com/in/luiscopete)
