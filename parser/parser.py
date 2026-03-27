import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> N V
S -> NP VP
S -> S Conj S

NP -> N
NP -> Det N
NP -> Det Adj N
NP -> NP PP
NP -> Det Adj Adj Adj N

VP -> V
VP -> V NP
VP -> V PP
VP -> V NP PP
VP -> VP Conj VP
VP -> Adv VP
VP -> V Adv
VP -> V PP Adv

PP -> P NP
PP -> P NP Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # raise NotImplementedError

    return [
        w.lower() for w in nltk.tokenize.word_tokenize(sentence) if w.isalpha()
    ]


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # raise NotImplementedError

    chunks = []
    # if 'tree' is not Tree return empty list
    if not isinstance(tree, nltk.Tree):
        return []

    # Check if current NP has other NP inside
    if tree.label() == "NP":
        has_nested_np = any(
            isinstance(sub, nltk.Tree) and sub.label() == "NP"
            for sub in tree.subtrees(lambda t: t is not tree)
        )
        if not has_nested_np:
            chunks.append(tree)

    # Check 'tree' recursively
    for child in tree:
        chunks += np_chunk(child)

    return chunks


if __name__ == "__main__":
    main()
