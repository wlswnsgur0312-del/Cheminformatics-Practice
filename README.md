# 🧪 Cheminformatics & CADD Practice Repository

### 🧑‍🔬 Introduction
This repository contains my personal projects and study records related to **Cheminformatics** and **Computer-Aided Drug Design (CADD)**.
As a **Pharmacy undergraduate**, I am exploring how to apply **Python** and **Artificial Intelligence** to pharmaceutical sciences, focusing on molecular modeling and property prediction.
(약학 전공자로서 파이썬과 AI를 활용해 신약 개발(CADD) 및 분자 모델링을 연구한 프로젝트 모음입니다.)

---

## 📂 Project List

### 1. [Tool] SMILES to 3D Structure Converter 🧬
> **Goal:** Automate the generation of 3D molecular structures from 1D SMILES strings.
* **Description:** A script that converts SMILES into 3D objects and performs **Geometry Optimization** using the **MMFF94** force field.
* **Key Tech:** `RDKit`, `AllChem.EmbedMolecule`, `MMFFOptimization`
* **Result:** Generates `.pdb` files ready for docking simulations.
* 🔗 **[View Code](./smiles to 3D pdb converter.py)** 

### 2. [AI] Drug Solubility Predictor (QSAR Model) 💊
> **Goal:** Predict aqueous solubility of drug candidates using Machine Learning.
* **Description:** Implemented a **QSAR (Quantitative Structure-Activity Relationship)** model using **Random Forest**.
* **Methodology:**
    1.  Extract physicochemical descriptors (LogP, TPSA, MW, Rotatable Bonds).
    2.  Train the model to classify drugs as *Soluble* or *Insoluble*.
    3.  Predict properties of new compounds (e.g., Cholesterol).
* **Key Tech:** `Scikit-Learn`, `Pandas`, `RDKit Descriptors`
* 🔗 **[View Code](./solubility_QSAR.py)**

* ### 3. [Big Data] EGFR Inhibitor Bioactivity Classifier (QSAR) 🦠
> **Goal:** Predict bioactivity of compounds against **EGFR (Lung Cancer Target)** using large-scale experimental data.
* **Description:** Automated data mining from the **ChEMBL Database** (57k+ entries) to build a classification model for **Non-Small Cell Lung Cancer (NSCLC)** drug discovery.
* **Methodology:**
    1.  **Data Mining:** Extracted real-world bioactivity data for **EGFR** (CHEMBL203).
    2.  **Preprocessing:** Cleaned data and defined 'Active' compounds (**IC50 ≤ 1000nM**).
    3.  **Modeling:** Trained **Random Forest** on physicochemical descriptors.
* **Result:** Achieved **84.32% Accuracy**, proving the feasibility of in-silico screening for kinase inhibitors.
* **Key Tech:** `ChEMBL Database`, `Pandas`, `Scikit-Learn`, `RDKit`
* 🔗 **[View Code](./EGFR_prediction_chembl_ai.py)**
---

## 🛠️ Tech Stack & Tools

| Category | Tools |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) |
| **Cheminformatics** | ![RDKit](https://img.shields.io/badge/Library-RDKit-00CC00?logo=molecule) |
| **AI / ML** | ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-F7931E?logo=scikitlearn&logoColor=white) ![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas&logoColor=white) |
| **Environment** | VS Code, Jupyter Notebook |

---

### 📫 Contact & Goals
* **Interest:** Molecular Docking, QSAR, Deep Learning in Drug Discovery.
* **Education:** Pharmaceutical science Student at Kyung Hee University.
