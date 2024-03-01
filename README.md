# Aquatic_Monitering_Using_DL
Aquatic Monitoring System - Fault Detection Repository
Welcome to the Aquatic Monitoring System Fault Detection Repository! This repository contains the code and resources for detecting sensor faults in aquatic monitoring systems using deep learning and machine learning techniques.

Overview
Aquatic environments, especially those used in aquaculture, require robust monitoring systems to maintain water quality and ensure the health of aquatic life. This project focuses on developing and comparing various algorithms to detect sensor anomalies, thus enhancing the reliability of monitoring systems.

Features
Multiple Algorithms: We employ and compare several machine learning and deep learning algorithms, including Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), Variational Autoencoders (VAE), Autoencoders, OneClass-SVM, and Isolation Forest.
Hybrid Model: A novel hybrid model combining an Autoencoder with an Isolation Forest is developed to improve detection capabilities.
Comprehensive Evaluation: The project includes a thorough evaluation using real-world data from an aquatic monitoring system, with over 83,000 entries timestamped for time-series analysis.
Performance Metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used to quantify the accuracy of each algorithm in detecting sensor faults.
Methodology
The methodology section outlines the dataset description, algorithm selection process, model development, and training procedures. It also provides insights into the development of the hybrid model and its algorithmic workflow.

Results & Discussions
The results section compares the performance of each algorithm, highlighting the superiority of CNN and the hybrid model in detecting sensor faults. It also discusses the implications of the findings and suggests future research directions.

Repository Structure
Code: Contains implementation scripts for each algorithm and the hybrid model.
Data: Includes the dataset used for training and testing the models.
Documentation: Detailed documentation on data preprocessing, model architectures, and evaluation metrics.
Results: Visualization of results and comparative analysis outputs.
Resources: Additional resources such as literature references and related studies.
Usage
To utilize the repository:

Clone the repository to your local machine.
Install the necessary dependencies listed in the documentation.
Explore the codebase and documentation to understand the implementation details.
Run the scripts to train and evaluate the models using your own datasets if desired.
