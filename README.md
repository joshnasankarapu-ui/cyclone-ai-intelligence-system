\# 🌀 Cyclone AI Intelligence System



\### Smart India Hackathon 2026 — Problem Statement 26070



AI/ML-based multi-source satellite intelligence system for identification, classification, and analysis of tropical cyclone patterns.



\*\*Team:\*\* CodeNova  

\*\*Theme:\*\* Disaster Management  

\*\*Category:\*\* Software



\---



\## 📌 Overview



The Cyclone AI Intelligence System is an AI/ML-based prototype designed to analyze multi-source satellite observations of tropical cyclones.



The system combines infrared and passive microwave satellite information and processes the observations through a convolutional neural network (CNN) based multi-task learning architecture.



The prototype provides:



\- Tropical cyclone development classification

\- AI-based intensity estimation

\- AI-based central pressure estimation

\- Development-class probabilities

\- Infrared satellite visualization

\- Passive microwave brightness-temperature visualization

\- Thermal signal analysis

\- Cyclone location information

\- Storm motion information

\- Satellite data quality indicators



\---



\## 🎯 Smart India Hackathon Problem Statement



\*\*Problem Statement ID:\*\* 26070



\*\*Problem Statement:\*\*



> To develop an (AI) / (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data.



\*\*Theme:\*\* Disaster Management  

\*\*Category:\*\* Software



\---



\## 🛰️ Multi-Source Satellite Data



The prototype processes multi-source satellite observations using:



\- Infrared satellite observations

\- Passive microwave observations

\- AMSR2 / GCOM-W1 data

\- ATMS / NOAA-20 data



The current extracted prototype dataset contains:



\- 6 satellite observations

\- 128 × 128 spatial resolution

\- 13 input channels

\- 1 infrared channel

\- 12 passive microwave channels



Missing microwave observations are represented using availability-aware zero filling so that observations from different satellite instruments can be processed through a common input structure.



\---



\## 🧠 AI / ML Architecture



The core model uses a convolutional neural network with:



\- Conv2D feature extraction layers

\- Batch normalization

\- Max pooling

\- Global average pooling

\- Dense feature representation

\- Dropout regularization

\- Multi-task output heads



The model produces three outputs:



\### 1. Cyclone Intensity



A regression output estimating cyclone intensity.



\### 2. Central Minimum Pressure



A regression output estimating central minimum pressure.



\### 3. Cyclone Development Classification



A classification output providing development-category probabilities.



This multi-task structure allows the prototype to analyze several cyclone-related properties from the same satellite input.



\---



\## 📊 Current Prototype



The current demonstration application provides an interactive Streamlit dashboard.



\### Application workflow



```text

Multi-Source Satellite Data

&#x20;         ↓

Data Preprocessing

&#x20;         ↓

128 × 128 × 13 Input Tensor

&#x20;         ↓

CNN Feature Extraction

&#x20;         ↓

Multi-Task AI Model

&#x20;    ↙        ↓        ↘

Intensity  Pressure  Development

&#x20;         ↓

Interactive Visualization
```text

