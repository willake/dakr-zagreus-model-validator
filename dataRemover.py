def read_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        first_two_lines = lines[:2]
        data_lines = lines[2:]
        processed_data = []
        for i, line in enumerate(data_lines):
            if i % 2 == 0:  # Even lines (0-based index)
                elements = line.strip().split()
                elements = elements[:-3]  # Remove last three elements
                processed_data.append(' '.join(elements) + '\n')
            else:  # Odd lines
                processed_data.append(line)
    return first_two_lines + processed_data

def write_file(output_file, content):
    with open(output_file, 'w') as file:
        file.writelines(content)

def main():
    input_file = 'DZrecord.log'  # Replace with your input file path
    output_file = 'output.txt'  # Replace with your output file path

    content = read_file(input_file)
    write_file(output_file, content)

if __name__ == "__main__":
    main()
