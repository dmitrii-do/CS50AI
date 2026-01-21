import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # raise NotImplementedError

    # Create a helper 'convert' to convert data
    def convert(key, value):

        to_integer = [
            "Administrative",
            "Informational",
            "ProductRelated",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
        ]

        month = "Month"
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "June",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        what_month = dict(zip(months, range(12)))

        to_float = [
            "Administrative_Duration",
            "Informational_Duration",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
        ]
        visitor_type = "VisitorType"

        weekend_and_revenue = ["Weekend", "Revenue"]

        if key in to_integer:
            return int(value)

        if key in to_float:
            return float(value)

        if key == month:
            return what_month[value]

        if key == visitor_type:
            if value == "Returning_Visitor":
                return 1
            return 0

        if key in weekend_and_revenue:
            if value == "TRUE":
                return 1
            return 0

    # Create two lists: 'evidences' and 'labels'
    evidences = []
    labels = []

    # Read csv-file to csv.Dict-object
    with open(filename, "r", encoding="UTF-8") as f:
        rows = csv.DictReader(f)

        # Get the fieldnames
        headers = rows.fieldnames

        # Convert each value and add it to the lists: 'evidences' or 'labels'
        for row in rows:
            visitor = []
            for name in headers:
                now = convert(name, row[name])
                if name != "Revenue":
                    visitor.append(now)
                else:
                    labels.append(now)
            evidences.append(visitor)

    # Return a tuple (evidences, labels)
    return (evidences, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # raise NotImplementedError
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # raise NotImplementedError

    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            if predicted == 1:
                true_positive += 1
        else:
            total_negative += 1
            if predicted == 0:
                true_negative += 1

    sensitivity = true_positive / total_positive
    specificity = true_negative / total_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
