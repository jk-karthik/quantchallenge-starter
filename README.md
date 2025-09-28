# QuantChallenge Starter Repo

 QuantChallenge 2025.



## Directories

This repo consists of two folders: /research 
### 1. Research


- The `research` folder contains a IPython notebook `analysis.ipynb` that dives into feature analysis and engineering required to the predictions oos and has the setup required to perform the necessary parameter optimziations for the prediction models
- The `submission.py` trains the models with the selected parameters and optimizes ensemble parameters for final predictions and to make the final prediction 'submission_final.csv'


### Results

Best submission achieved a **Leaderboard Score of 0.6793**.

The final score is calculated as the average of the **Coefficient of Determination** ($\text{R}^2$) for each target variable:

$$\text{Score} = \frac{\text{R}^2_{\text{Y1}} + \text{R}^2_{\text{Y2}}}{2}$$







<!--## Questions
--If you have any lingering questions, reach out for support on Discord or email info@quantchallenge.org
-->
