import csv


def load_reviews(filepath):
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append({
                'review': row['review'],
                'sentiment': row['sentiment']
            })
    return reviews


def count_reviews_with_keyword(reviews, keyword):
    keyword = keyword.lower()
    count = 0
    for r in reviews:
        if keyword in r['review'].lower():
            count += 1
    return count


def keyword_breakdown(reviews, keyword):
    positive_reviews = [r for r in reviews if r['sentiment'] == 'positive']
    negative_reviews = [r for r in reviews if r['sentiment'] == 'negative']

    pos_count = count_reviews_with_keyword(positive_reviews, keyword)
    neg_count = count_reviews_with_keyword(negative_reviews, keyword)

    print(f"Keyword: '{keyword}'")
    print(f"  Appears in {pos_count} positive reviews (out of {len(positive_reviews)})")
    print(f"  Appears in {neg_count} negative reviews (out of {len(negative_reviews)})")


def compute_prior(reviews, sentiment_label):
    matching = sum(1 for r in reviews if r['sentiment'] == sentiment_label)
    return matching / len(reviews)


def compute_likelihood(reviews, keyword, sentiment_label):
    subset = [r for r in reviews if r['sentiment'] == sentiment_label]
    matches = count_reviews_with_keyword(subset, keyword)
    return matches / len(subset)


def compute_marginal(reviews, keyword):
    matches = count_reviews_with_keyword(reviews, keyword)
    return matches / len(reviews)


def bayes_posterior(prior, likelihood, marginal):
    return (likelihood * prior) / marginal


if __name__ == "__main__":
    reviews = load_reviews("data/part2-bayesian.csv")
    print("Total reviews loaded:", len(reviews))
    print()

    positive_keywords = ["brilliant", "excellent", "masterpiece"]
    negative_keywords = ["boring", "terrible", "waste"]
    all_keywords = positive_keywords + negative_keywords

    prior = compute_prior(reviews, "positive")
    print(f"Prior P(Positive) = {prior}")
    print()

    for word in all_keywords:
        likelihood = compute_likelihood(reviews, word, "positive")
        marginal = compute_marginal(reviews, word)
        posterior = bayes_posterior(prior, likelihood, marginal)

        print(f"Keyword: '{word}'")
        print(f"  P(Positive)            = {prior:.4f}")
        print(f"  P({word}|Positive)  = {likelihood:.4f}")
        print(f"  P({word})           = {marginal:.4f}")
        print(f"  P(Positive|{word}) = {posterior:.4f}")
        print()
