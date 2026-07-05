# Formative3-Group9-Probability-Bayes-GradientDescent
Group 9 Formative 3 project covering Probability Distributions, Bayesian Probability, and Gradient Descent implementation.

## Part 2: Bayesian Probability

We chose three keywords that we think show people are happy and three that show people are not happy.

The happy words are brilliant, excellent and masterpiece. The not happy words are boring, terrible and waste.

We picked brilliant and masterpiece because they are words that people do not use every day. Excellent is a common word so we used it to compare with the other two. For the not words, boring and terrible are direct complaints. Waste was used a lot in the reviews to say something was a waste of time so we included it.

We only wanted to figure out the chance that a review's happy if it has one of our keywords in it.
Here is the table of results:

| Keyword     | P(Positive) | P(kw\|Positive) | P(kw)  | P(Positive\|kw) |
|-------------|-------------|-----------------|--------|-----------------|
| brilliant   | 0.50        | 0.0755          | 0.0489 | 0.7711          |
| excellent   | 0.50        | 0.1174          | 0.0725 | 0.8099          |
| masterpiece | 0.50        | 0.0377          | 0.0264 | 0.7131          |
| boring      | 0.50        | 0.0247          | 0.0623 | 0.1983          |
| terrible    | 0.50        | 0.0154          | 0.0541 | 0.1419          |
| waste       | 0.50        | 0.0145          | 0.0731 | 0.0993          |

The chance that a review is happy is 0.50 because we have the number of happy reviews and not happy reviews.

When we look at a keyword we can tell if the review is more likely to be happy or not. For example excellent is used in 11.7 percent of happy reviews but only 7.25 percent of all reviews. So if we see excellent in a review it is more likely that the review is happy. Waste is used more in not reviews so if we see waste it is less likely that the review is happy.

Masterpiece and terrible are, like excellent and waste. Not as strong. Masterpiece is not used much so it does not make as big of a difference.
