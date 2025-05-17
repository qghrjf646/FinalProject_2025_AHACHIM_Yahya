# KuaiRec: Hybrid Recommender System for Short Videos

## Project Overview
This project implements a modular, well-documented hybrid recommender system for short videos using the KuaiRec dataset. The system combines collaborative filtering (ALS), content-based filtering (Logistic Regression), and hybrid approaches to generate personalized recommendations. All code is organized for reproducibility, and a notebook (`project.ipynb`) illustrates each step. An additional content-based filtering recommndation was made to compare the results.

## Data location
The data needs to extracted from the zip file and the resulting `data_final_project` directory needs to be at the root of the repo.

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Feature Engineering](#feature-engineering)
3. [Model Development](#model-development)
4. [Additional model](#additional-model)
6. [Results](#results)
7. [Conclusions](#conclusions)
8. [Reproducibility](#reproducibility)

---

## 1. Data Preprocessing

**Notebook Reference:** See Section 1 and 2 in `project.ipynb`.

- **Data Loading:** All main and side data files from KuaiRec are loaded, including user-item interactions, item metadata, user features, social network, and categories.
- **Inspection:** Data is checked for missing values, duplicates, and coverage of user/item IDs.

---

## 2. Feature Engineering

**Notebook Reference:** See Section 3 in `project.ipynb`.

- **Ground Truth Definition:** A video is considered relevant if `watch_ratio > 0.9`, based on analysis of watch ratio distributions (see notebook for details).
- **Collaborative Features:** User-item interaction matrix is built for ALS.
- **Content Features:** Video catacteristics such as duration, download count, play_count, like_count... are selected and merged into a matrix for logistic regression.

---

## 3. Model Development

**Notebook Reference:** See Section 4 in `project.ipynb`.

- **Baselienes:** 2 simple baselines were established:
    + To recommend the N most watched videos of all the training set : good for cold start situations, but bad for discovery and personalization.
    + To recommend the N most watched videos of the first friend in the friend list : bad for cold start situations, but a bit better for discovery and personalization.
- **Metrics:** 4 metrics were chosen for this project:
    + Precison@K to measure how good the model is at avoiding false positives.
    + Recall@K to measure how good the model is at avoiding false negatives.
    + F1-score to combine both as in an K recommendations model, the first 2 metrics' scores are influenced by the K chosen by the evaluator (especially the recall@K).
    + NDCG@K to measure how good the model at ranking the most relevant items first, whereas the MAP@K metric wasn't used as for a short video recommender it is less important to be precise in the ranking of the each item, than to not give good recommendations first. Bc most short video consumers don't mind scrolling if a less interesting video is shown to them, as much as they would getting uninteresting ones in the beginning.
- **Collaborative Filtering:** ALS is used to learn latent user and item factors.
- **Content-Based Filtering:** Logistic Regression predicts user preferences from engineered features.
- **Hybrid Model:** Combines ALS and content-based scores (e.g., weighted sum).

---

## 4. Additional model

A content-based model that serves as a measurement of how relevant the collaborative filtering is.

- **Architecture:** The model is neural network that uses the same content features as the previous model.

---

## 5. Results

The number of videos to recommend to test the evaluation that were tested are 10 and 100, as those are 2 likely use cases for a short video recommender.

 - **For N=10:**

    + Baseline 1 (Top-10 Most Watched)
    Mean Precision: 0.1901
    Mean Recall: 0.0014
    Mean F1: 0.0028
    Mean NDCG@10: 0.1486

    + Baseline 2 (Top-10 by First Friend)
    Mean Precision: 0.1870
    Mean Recall: 0.0014
    Mean F1: 0.0028
    Mean NDCG@10: 0.1487

    + Model 1 (LR):
    Precision@10: 0.5339
    Recall@10: 0.0041
    F1@10: 0.0082
    NDCG@10: 0.5474

    + Model 2 (NN):
    Mean Precision@10: 0.3953
    Mean Recall@10: 0.0030
    Mean F1@10: 0.0060
    Mean NDCG@10: 0.3926


 - **For N=100:**

    + Baseline 1 (Top-100 Most Watched)
    Mean Precision: 0.3592
    Mean Recall: 0.0272
    Mean F1: 0.0503
    Mean NDCG@100: 0.3345

    + Baseline 2 (Top-100 by First Friend)
    Mean Precision: 0.3423
    Mean Recall: 0.0259
    Mean F1: 0.0480
    Mean NDCG@100: 0.3193

    + Model 1 (LR):
    Precision@100: 0.4135
    Recall@100: 0.0316
    F1@100: 0.0588
    NDCG@100: 0.4345

    + Model 2 (NN):
    Mean Precision@100: 0.3768
    Mean Recall@100: 0.0285
    Mean F1@100: 0.0528
    Mean NDCG@100: 0.3809
---

## 6. Conclusions

- Both models outperform (across all metrics) the 2 established baselines indicating a smarter decision making by the models, hence justifying their use.
- With both N values, the hybrid recommender outperforms the content-based recommender, even though the content-based recommender had a neural network model. This proves the utility of the user-item interactions consideration, as it outperformed a smarter model.
- **key takeway**: It's important to select features that have meaningful semnatic connection. This is proven by the ability of a relatively simple model to perform well.

---

## 7. Reproducibility

- All code is modular and well-documented.
- The notebook `project.ipynb` illustrates all steps and can be run end-to-end.
- Model training and evaluation scripts are provided for each approach.
- Data files and intermediate results are saved for transparency and reproducibility.

---