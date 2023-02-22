import codecs
import itertools
import numpy as np
import pandas as pd

with codecs.open("multimobidity.csv", "r", "Shift-JIS", "ignore") as file1:
    data = pd.read_table(file1, delimiter=",")
    data = data.drop('その他の疾患', axis=1)
    data = data.set_index("No")
print(data)

# Load your data
symptoms = data.iloc[:, 2:-1]
anxiety = data.iloc[:, -1]

# Set up the list of all possible symptom permutations
symptom_perms = list(itertools.permutations(symptoms.columns))

# Initialize a list to store the Shapley values of each symptom
shapley_values = [0] * len(symptoms.columns)

# Loop over all possible symptom permutations
for perm in symptom_perms:

    # Initialize the contribution of each symptom in this permutation
    perm_contributions = [0] * len(symptoms.columns)

    # Loop over all subsets of symptoms up to the current permutation
    for i in range(1, len(perm)+1):
        for subset in itertools.combinations(perm, i):

            # Check if all symptoms in the subset have at least one participant
            if all(symptoms[symptom].sum() > 0 for symptom in subset):

                # Calculate the contribution of this subset to the outcome
                k6_subset = anxiety[list(symptoms[list(subset)].all(axis=1).values)]
                if len(k6_subset) == 0:
                    k6_contrib = 0
                else:
                    k6_contrib = np.mean(k6_subset)

                # Calculate the marginal contribution of each symptom in this subset
                for s in subset:
                    s_index = symptoms.columns.get_loc(s)
                    marginal_contrib = k6_contrib - perm_contributions[s_index]
                    perm_contributions[s_index] += marginal_contrib

    # Calculate the Shapley value of each symptom in this permutation
    for i, s in enumerate(perm):
        s_index = symptoms.columns.get_loc(s)
        marginal_contrib = perm_contributions[s_index]
        shapley_values[s_index] += marginal_contrib / len(symptom_perms)

# Print the Shapley values of each symptom
for i, s in enumerate(symptoms.columns):
    print(s, shapley_values[i])
