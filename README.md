# Formative3-Group9-Probability-Bayes-GradientDescent
Group 9 Formative 3 project covering Probability Distributions, Bayesian Probability, and Gradient Descent implementation.

## Part 2: Bayesian Probability

We picked three keywords we felt signal positive sentiment and three for negative sentiment.

Positive: brilliant, excellent, masterpiece
Negative: boring, terrible, waste

We picked brilliant and masterpiece because they are strong words people don't usually use lightly. Excellent is more common and a bit more generic, so we kept it in to compare against the other two. For the negative side, boring and terrible are both direct complaints about quality, and waste came up a lot in the reviews as part of "waste of time," so we included it as its own keyword.

We decided to calculate P(Positive | keyword) only, not P(Negative | keyword).

Here is the table of results:

| Keyword     | P(Positive) | P(kw\|Positive) | P(kw)  | P(Positive\|kw) |
|-------------|-------------|-----------------|--------|-----------------|
| brilliant   | 0.50        | 0.0755          | 0.0489 | 0.7711          |
| excellent   | 0.50        | 0.1174          | 0.0725 | 0.8099          |
| masterpiece | 0.50        | 0.0377          | 0.0264 | 0.7131          |
| boring      | 0.50        | 0.0247          | 0.0623 | 0.1983          |
| terrible    | 0.50        | 0.0154          | 0.0541 | 0.1419          |
| waste       | 0.50        | 0.0145          | 0.0731 | 0.0993          |

The prior P(Positive) is 0.50 because the dataset has 25,000 positive reviews and 25,000 negative reviews, so before we even look at a keyword, there's an equal chance of either.

Once we bring in a keyword, the posterior moves depending on how much more (or less) often that word shows up in positive reviews compared to the dataset as a whole. For excellent, it shows up in about 11.7% of positive reviews but only 7.25% of all reviews, so seeing it in a review pushes our belief that the review is positive from 50% up to about 81%. Waste shows up a lot more in negative reviews than in the dataset overall, so seeing it drags the probability that the review is positive down to about 10%, meaning it's much more likely negative.

Masterpiece and terrible follow the same pattern as excellent and waste, just less extreme in masterpiece's case since it appears less often overall.
