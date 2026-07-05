# Formative3-Group9-Probability-Bayes-GradientDescent
Group 9 Formative 3 project covering Probability Distributions, Bayesian Probability, and Gradient Descent implementation.

Part 1: Probability Distributions

Objective

Treat a dataset of heights as an unlabeled mixture of two Gaussian distributions and use the Expectation-Maximization (EM) algorithm to recover the two underlying groups father heights and child heights without ever being told which point belongs to which group.

Dataset

Galton Families heights dataset a classic dataset of family height measurements collected by Francis Galton.


father column → father heights
childHeight column → children's heights (both sons and daughters)
Combined into a single unlabeled array of 1,868 heights (934 fathers + 934 children)


Why not just split at the global mean?

A hard split at the mean would force every data point into one of two buckets, even points sitting in the ambiguous overlap zone between the two height populations for example, a shorter father or a taller teenage child near the boundary. That approach makes an all-or-nothing decision with no regard for genuine uncertainty.

EM instead assigns every point a soft responsibility a probability of belonging to each group based on how likely that height is under each group's current Gaussian estimate. A borderline height might get responsibilities like (0.55, 0.45) instead of being forced into one category. This lets both Gaussians' parameters get updated by every point, weighted by relevance, rather than by a rigid, potentially misclassifying cutoff.

Method

Implemented from scratch in Python/NumPy (no sklearn or other ML libraries):


Initialization μ1 = min(data), μ2 = max(data), both variances set to the global variance, π1 = π2 = 0.5
E-step compute each point's responsibility (posterior probability of belonging to each Gaussian) given current parameters
M-step re-estimate μ1, μ2, σ1², σ2², π1, π2 as responsibility-weighted statistics
Convergence check track log-likelihood each iteration; stop when the improvement falls below 1e-4 (EM guarantees log-likelihood never decreases)
Classification demo given any test height, output P(Father | height) and P(Child | height) using the final converged parameters


Results Iteration Tracking Table

Iterationμ1 (Father)μ2 (Child)σ1²σ2²π1π2Log-Likelihood0 (Initialization)56.000079.000010.963910.96390.50000.5000-11945.2214164.827970.17204.63703.63310.41180.5882-4894.5713264.919570.04635.35874.13720.40470.5953-4875.3084274 (Converged)64.072969.27695.10446.13290.25080.7492-4860.8272

Log-likelihood rose monotonically from -11,945 at initialization to -4,861 at convergence  confirming the model steadily improved its fit at every step, consistent with the EM guarantee.

Live Classification Demo (example outputs)

Test HeightP(Father | height)P(Child | height)Classification66 in0.37970.6203Child72 in0.00140.9986Child


Notes on the model

Since the "child" group includes both sons and daughters (a wider natural height spread than the all-male father group), the two recovered Gaussians approximate but won't perfectly correspond to the literal father/child labels. This is an expected property of unsupervised clustering on real-world data, not an implementation error.






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



# Part 3: Gradient Descent Manual Calculation

## Objective

This part manually computes gradient descent updates for the parameters **m** (weights) and **b** (bias) of a linear regression model, using full matrix operations rather than treating values as scalars.

Model:
```
ŷ = Xm + b
```

**Fixed inputs used throughout:**
- X = [[1, 3], [4, 10]]
- y = [5, 6]
- Learning rate (α) = 0.01
- n (number of samples) = 2
- Initial m = [-1, 2], Initial b = [1, 1]

One gradient descent update was performed per group member, using the output of the previous iteration as the input to the next so the parameters evolve continuously across the four iterations below.

## Method (derived once, reused every iteration)

1. **Predict:** ŷ = Xm + b
2. **Error:** error = ŷ − y
3. **Cost function (MSE):** J(m, b) = (1/n)·uᵀu, where u = Xm + b − y
4. **Gradients** (derived via chain rule):
   - ∂J/∂m = (2/n).Xᵀ·(ŷ − y)
   - ∂J/∂b = (2/n)·(ŷ − y)
5. **Parameter update:**
   - m_new = m_old − α·(∂J/∂m)
   - b_new = b_old − α·(∂J/∂b)

## Iteration Results

| Iteration | m (start) | b (start) | ŷ | error (ŷ−y) | m (updated) | b (updated) |
|---|---|---|---|---|---|---|
| 1 | [-1, 2] | [1, 1] | [6, 17] | [1, 11] | [-1.45, 0.87] | [0.99, 0.89] |
| 2 | [-1.45, 0.87] | [0.99, 0.89] | [2.15, 3.79] | [-2.85, -2.21]| [-1.33, 1.18]| [1.01, 0.91] |
| 3 | [-1.33, 1.18] | [1.01, 0.91] | [3.19, 7.29] | [-1.81, 1.29] | [-1.36, 1.10] | [1.03, 0.90] |
| 4 | [-1.36, 1.10] | [1.03, 0.90] | [2.97, 6.46] | [-2.03, 0.46] | [-1.36, 1.11] | [1.05, 0.90] |

*(Values rounded to 2 decimal places at each step; small rounding drift compounds slightly across iterations, but the method is consistent throughout.)*

## Observed Trend

- **m** moves from [-1, 2] toward roughly [-1.36, 1.11] the first component stabilizes around -1.3 to -1.4, and the second component drops from 2 down toward ~1.1, converging rather than diverging.
- **b** moves from [1, 1] toward roughly [1.05, 0.90] both components stay close to their starting values, with only small adjustments each round.
- The **error magnitude shrinks substantially after the first update** (from [1, 11] down to values consistently under ~3 in each component) and then oscillates in sign rather than steadily shrinking further. This is expected behavior in gradient descent with a fixed learning rate: the parameters are still moving in a direction that reduces error overall compared to the initial guess, but the fixed step size (α = 0.01) causes it to overshoot slightly and correct back and forth near the minimum rather than converging smoothly in a straight line.

## Conclusion

Across all four iterations, the parameters consistently moved away from their poorly fitting initial values and toward a region where predictions are much closer to the true y values, confirming that gradient descent is correctly reducing the cost function. More iterations would be expected to further stabilize m and b closer to the true optimal values.


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


