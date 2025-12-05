import csv
import itertools
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False if row["trait"] == "0" else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # raise NotImplementedError

    # Initialize probability prob = 1
    prob = 1

    # loop everyone person in people
    for person in people:
        # Get the number of genes the person has
        genes = get_genes(person, one_gene, two_genes)

        # Determine if the person has the trait
        has_trait = person in have_trait

        mother = people[person]["mother"]
        father = people[person]["father"]

        # Calculate parents' probability
        # The person hasn't got the parents
        if mother is None and father is None:
            p_genes = PROBS["gene"][genes]

        # The person has got the parents
        else:
            mom_genes = get_genes(mother, one_gene, two_genes)
            dad_genes = get_genes(father, one_gene, two_genes)
            p_genes = inherit_prob(genes, mom_genes, dad_genes)

        # Get the probability of appearing the trait
        p_trait = PROBS["trait"][genes][has_trait]

        # Calculate the joint probability
        prob *= p_genes * p_trait

    return prob


def get_genes(person, one_set, two_set):
    """
    Returns number of the genes the person has
    """
    ans = 0
    if person in two_set:
        ans = 2
    elif person in one_set:
        ans = 1
    return ans


def inherit_prob(child_genes, mom_genes, dad_genes):
    """
    Calculates the probability that a child will inherit child_genes copies of a gene, given the number of mom_genes and dad_genes. Takes into account the probability of mutation.
    """
    mutation = PROBS["mutation"]

    def pass_gen_prob(num_genes):
        """
        Calculates the probability that a parent's gene will be passed on to a child.
        """
        if num_genes == 2:
            return 1 - mutation
        if num_genes == 1:
            return 0.5
        return mutation

    mom_pass = pass_gen_prob(mom_genes)
    dad_pass = pass_gen_prob(dad_genes)

    if child_genes == 2:
        return mom_pass * dad_pass
    if child_genes == 1:
        return mom_pass * (1 - dad_pass) + dad_pass * (1 - mom_pass)
    return (1 - mom_pass) * (1 - dad_pass)


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # raise NotImplementedError

    # Loop every person in probabilities dictionary
    for person in probabilities:

        # Change genes distribution probability
        num_gene = 0
        if person in one_gene:
            num_gene = 1
        elif person in two_genes:
            num_gene = 2
        probabilities[person]["gene"][num_gene] = p

        # Change the trait probability
        if person in have_trait:
            probabilities[person]["trait"][True] = p
        else:
            probabilities[person]["trait"][False] = p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # raise NotImplementedError

    # Loop every person in probabilities dictionary
    for person in probabilities:
        # Normalize genes probabilities
        norm = 1 / sum(probabilities[person]["gene"].values())
        keys = probabilities[person]["gene"].keys()
        for key in keys:
            probabilities[person]["gene"][key] *= norm

        # Normalize the trait probabilities
        norm = 1 / sum(probabilities[person]["trait"].values())
        keys = probabilities[person]["trait"].keys()
        for key in keys:
            probabilities[person]["trait"][key] *= norm


if __name__ == "__main__":
    main()
# probabilities = {
#         person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
#         for person in people
#     }
