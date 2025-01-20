import os

def copy_10_percent(input_folder, output_folder, filenames):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in filenames:
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_sample.conllu")

        with open(input_file, "r", encoding="utf-8") as file:
            total_lines = sum(1 for line in file)

        lines_to_copy = total_lines // 10000

        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            for i, line in enumerate(infile):
                if i < lines_to_copy:
                    outfile.write(line)
                else:
                    break

# Define the input and output folders and file names
folders = {
    "en": "en_sample",
    "pl": "pl_sample"
}
filenames = ["test.conllu", "train.conllu", "val.conllu"]

# Process each folder
for input_folder, output_folder in folders.items():
    copy_10_percent(input_folder, output_folder, filenames)