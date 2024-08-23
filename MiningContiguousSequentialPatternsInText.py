from collections import defaultdict

def read_input(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().split() for line in file]

def find_contiguous_patterns(database, min_support):
    item_counts = defaultdict(int)
    for sequence in database:
        visited = set()
        #print(len(sequence))   number of words in this line

        for i in range(len(sequence)):
            for j in range(i, len(sequence)):
                subsequence = tuple(sequence[i:j + 1])
                    #print(subsequence)
                if subsequence not in visited:
                    item_counts[subsequence] += 1
                    visited.add(subsequence)

    total_sequences = len(database)
    min_absolute_support = min_support * total_sequences

    frequent_patterns = [(support, items) for items, support in item_counts.items() if support >= min_absolute_support]

    return frequent_patterns

def write_patterns_to_file(patterns, output_file):
    with open(output_file, 'w') as file:
        for support, items in patterns:
            file.write(f"{support}:{';'.join(items)}\n")


if __name__ == "__main__":
    input_file = "review_sample.txt"
    output_file = "patterns.txt"
    min_support = 0.01


    database = read_input(input_file)
    frequent_patterns = find_contiguous_patterns(database, min_support)
    print(frequent_patterns)

    write_patterns_to_file(frequent_patterns, output_file)
    print("code complete")