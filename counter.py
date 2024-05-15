def count_values_above_threshold(filename, threshold):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Filter even-indexed lines
            even_indexed_lines = [line.strip().split() for index, line in enumerate(lines) if index > 2 and index % 2 == 1]
            
            # Get the number of columns
            num_columns = len(even_indexed_lines[0])

            # Initialize counts for each column
            column_counts = [0] * num_columns
            
            # Iterate through each column and count values above threshold
            for line in even_indexed_lines:
                for i in range(num_columns):
                    if float(line[i]) > threshold:
                        column_counts[i] += 1
            
            return column_counts

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage:
filename = "DZrecord.log"  # Replace with your file name
threshold = 0.9
counts = count_values_above_threshold(filename, threshold)
if counts is not None:
    print(f"Counts of values above {threshold} at each column:")
    for i, count in enumerate(counts):
        print(f"Column {i+1}: {count}")
