import copy

# spam emails:
spamails = [
    "oferta es secreto",
    "click link secreto",
    "link secreto deportes",
]

# ham emails:
hamails = [
    "practicar deportes hoy",
    "fui practicar deportes",
    "deportes secreto hoy",
    "Deporte es hoy",
    "Deportes cuesta dinero",
]


def preprocess_texts(text):
    cleaned_text = []  # A list to store the cleaned text
    for char in text:  # Iterate over each character in the text
        if char.isalpha() or char.isspace():
            cleaned_text.append(
                char.lower()
            )  # Append the char to list if it is alpha or a space
    return "".join(cleaned_text)


def tokenize(text, bow, bow_type, type, vocab):
    tokens = text.split()
    for token in tokens:
        bow.add(token)
        bow_type.append(token)
        if token in vocab:  # Fills up the vocabulary with the count of each word
            if type == 0:
                vocab[token][0] += 1
            else:
                vocab[token][1] += 1
        else:
            if type == 0:
                vocab[token] = [1, 0]
            else:
                vocab[token] = [0, 1]


def calculate_probabilities(vocab, counts, len_spam, len_ham):
    probabilities = {}

    # Calculate the probability of each word in spam and ham
    for word, probs in vocab.items():

        # Probability of word in spam
        qty = probs[0]
        vocab[word][0] = (qty + 1) / (len_spam + len(vocab))

        # Probability of word in ham
        qty = probs[1]
        vocab[word][1] = (qty + 1) / (len_ham + len(vocab))

        # Calculate the probability of each word in general
        # Stores the result in probabilities
        qty_both = counts[word][0] + counts[word][1]
        probabilities[word] = (qty_both + 1) / (len_spam + len_ham + len(vocab))

    return probabilities


def print_bow(bow):
    print(f"Len BoW: {len(bow)}.")
    for word in bow:
        print(word)


def calculate_prob(vocab, email, prob_spam, prob_ham):
    for word in email:
        if word in vocab:
            prob_spam *= vocab[word][0]
            prob_ham *= vocab[word][1]

    prob_spam = prob_spam / (prob_spam + prob_ham)
    prob_ham = prob_ham / (prob_spam + prob_ham)
    return prob_spam, prob_ham


def print_probs_vocab(vocab):
    # Print the table header
    print(f"\n{'Word':<19} {'SPAM':<9} {'HAM'}")
    print("-" * 35)

    # Print each row of the table
    for word, probabilities in vocab.items():
        spam_prob = probabilities[0] * 100
        ham_prob = probabilities[1] * 100
        print(f"{word:<15} {spam_prob:>7.2f}% {ham_prob:>7.2f}%")


def print_probs(probs):
    # Print the table header
    print(f"\n{'Word':<17} {'P(Word)'}")
    print("-" * 25)

    # Print each row of the table
    for word, p in probs.items():
        print(f"{word:<15} {p*100:>7.2f}%")


def main():
    # Bag of Words
    bow = set()  # A set stores unique elements
    bow_spam = list()
    bow_ham = list()
    vocab = {}  # A dictionary to store the proabilities (spam and ham) of each word
    counts = {}  # A dictionary to store the count of each word in the emails

    # Preprocess emails:
    spamails_cleaned = [preprocess_texts(email) for email in spamails]
    hamails_cleaned = [preprocess_texts(email) for email in hamails]
    for email in spamails_cleaned:
        tokenize(email, bow, bow_spam, 0, vocab)
    for email in hamails_cleaned:
        tokenize(email, bow, bow_ham, 1, vocab)

    counts = copy.deepcopy(vocab)
    print("Len BoW spam: ", len(vocab))
    print("Len BoW ham: ", len(bow_ham))
    print("Len Vocab: ", len(vocab))

    prob_spam = (len(spamails) + 1) / (len(spamails) + len(hamails) + 2)
    prob_ham = (len(hamails) + 1) / (len(spamails) + len(hamails) + 2)
    print(f"Probabilidad de spam: {prob_spam*100}%")
    print(f"Probabilidad de ham: {prob_ham*100}%")

    # Calculate probabilities of each word in general
    probabilities = calculate_probabilities(vocab, counts, len(bow_spam), len(bow_ham))

    print_probs(probabilities)

    print_probs_vocab(vocab)

    # Example email
    example_email = "hoy es secreto"
    print(f"\nEjemplo de email: {example_email}")
    example_email_cleaned = preprocess_texts(example_email)
    results = calculate_prob(vocab, example_email_cleaned.split(), prob_spam, prob_ham)
    print(f"Probabilidad de ser spam: {results[0]*100:>7.2f}%")
    print(f"Probabilidad de ser ham: {results[1]*100:>7.2f}%")


if __name__ == "__main__":
    main()
