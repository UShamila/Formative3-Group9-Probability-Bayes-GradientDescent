# Formative3-Group9-Probability-Bayes-GradientDescent
Group 9 Formative 3 project covering Probability Distributions, Bayesian Probability, and Gradient Descent implementation.

## Part 2: Bayesian Probability

We chose three keywords that we think show people are happy and three that show people are not happy.

The happy words are brilliant, excellent and masterpiece. The not happy words are boring, terrible and waste.

We picked brilliant and masterpiece because they are words that people do not use every day. Excellent is a common word so we used it to compare with the other two. For the not words, boring and terrible are direct complaints. Waste was used a lot in the reviews to say something was a waste of time so we included it.

We only wanted to figure out the chance that a review's happy if it has one of our keywords in it. Here is the table of results:
| Keyword     | P(Positive) | P(kw\|Positive) | P(kw)  | P(Positive\|kw) |
|-------------|-------------|-----------------|--------|-----------------|
| brilliant   | 0.50        | 0.0755          | 0.0489 | 0.7711          |
| excellent   | 0.50        | 0.1174          | 0.0725 | 0.8099          |
| masterpiece | 0.50        | 0.0377          | 0.0264 | 0.7131          |
| boring      | 0.50        | 0.0247          | 0.0623 | 0.1983          |
| terrible    | 0.50        | 0.0154          | 0.0541 | 0.1419          |
| waste       | 0.50        | 0.0145          | 0.0731 | 0.0993          |


#The chance that a review is happy is 0.50 because we have the number of happy reviews and not happy reviews.

When we look at a keyword we can tell if the review is more likely to be happy or not. For example excellent is used in 11.7 percent of happy reviews but only 7.25 percent of all reviews. So if we see excellent in a review it is more likely that the review is happy. Waste is used more in not reviews so if we see waste it is less likely that the review is happy.

Masterpiece and terrible are, like excellent and waste. Not as strong. Masterpiece is not used much so it does not make as big of a difference.


# Part 4: Writing the Gradient Descent Code

## What This Section Does
In this part of the project, we took the matrix math we calculated by hand and turned it into working Python code. We are training a linear regression model ($y = m_1x_1 + m_2x_2 + b$) to fit our data points using Gradient Descent.

Instead of hiding the math behind a black-box library like Scikit-Learn, we wrote out every single mathematical step explicitly.

---

## Key Features of Our Code

### 1. No Hidden Steps
We intentionally avoided using advanced wrappers or hidden classes. 
* Every matrix multiplication (`np.dot`) and matrix transpose (`X.T`) happens right out in the open inside our training loop.
* Every time the loop runs, the terminal prints out the exact values for the predicted outputs ($\hat{y}$), the error vector, the gradients, and the updated parameters. This makes it incredibly easy to compare the code's output line-by-line with our hand-written work.

### 2. Double-Checking Our Calculus with SciPy
Before running the loop, we wanted to make sure our manual calculus was actually correct. 
* We built a simple version of our loss function where we only change one weight ($m_1$) and freeze everything else.
* We then pass that function into `scipy.differentiate.derivative`. 
* By pulling out the answer using the `.df` tool, SciPy gives us a flawless independent baseline to verify our manual derivative before the training process even starts.

### 3. Tracking All Four Parameters Over Time
Because our data uses matrix rows, the bias ($b$) is treated as a vector with two separate parts ($b_1$ and $b_2$), just like our weights ($m_1$ and $m_2$). 
* To make sure Matplotlib prints clean lines, our code breaks these vectors down into single numbers at each step.
* This gives us two clear plots: one showing how all 4 parameters climb or drop over time, and a second one showing our overall error shrinking after every update.

---

## What We Learned from the Results
If you look closely at our terminal logs and graphs, you will see a few fascinating trends:
* **The Error Drops Instantly:** The single biggest drop in error happens right during Iteration 1. This proves our gradient math is correct and immediately guides the model toward the right answer.
* **The Weights Balance Each Other Out:** In the first iteration, weight $m_2$ takes a dramatic dip to fix a massive initial prediction error. By iterations 2 and 3, it gently climbs back up and stabilizes. This shows how multi-feature gradient descent dynamically balances different variables at the same time.
* **We Started at Step 0:** Our graphs intentionally include "Iteration 0" to capture our exact starting parameters before any math happened. This shows the full journey of the model's optimization.


