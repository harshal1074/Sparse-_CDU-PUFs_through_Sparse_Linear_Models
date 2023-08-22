# Sparse_CDU-PUFs_through_Sparse_Linear_Models
Sparse Linear Model for CDU PUF Analysis
This repository contains Python code that implements a sparse linear model to analyze and exploit vulnerabilities in Sparse Challenge-Response Pair (CRP) Physical Unclonable Functions (PUFs). The code demonstrates the feasibility of predicting responses for a given set of challenges using a sparse linear model.

Features
1. Implements a sparse linear model using both coordinate descent and projection gradient methods.
2. Utilizes numpy library for mathematical operations.
3. Demonstrates vulnerability analysis of Sparse CDU PUFs.
4. Evaluates model performance through mean absolute error calculation.

Usage
Clone the repository to your local machine:
git clone https://github.com/your-username/sparse-cdu-puf-analysis.git
cd sparse-cdu-puf-analysis

Run the submit.py file to see the implementation in action:
python submit.py

This will showcase how the sparse linear model is learned and applied to predict CDU PUF responses.

Modify the code and parameters to experiment with different settings or data.

Requirements
Python 3.x
