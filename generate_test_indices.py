# given indices from 0 to 6399,
# generate 5 sets of test indices of 30 in lenght of each list
# and then save it to a text file

import random

indices_range = list(range(2700))

num_sets = 10
set_length = 60

test_indices = []
for _ in range(num_sets):
    test_indices.append(random.sample(indices_range, set_length))
    
output_file = "test_indices_5.txt"
with open(output_file, "w") as f:
    for i, test_set in enumerate(test_indices):
        f.write(f"{test_set}\n")

print(f"Test indices saved to {output_file}")