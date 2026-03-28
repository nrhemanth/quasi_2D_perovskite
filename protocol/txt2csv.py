import os
import re
import csv

class TxtToCsvConverter:
    def __init__(self, path_folder, path_output_folder=None):
        self.path_folder = path_folder
        self.path_output_folder = path_output_folder or path_folder  # Save to same folder if not given

        # If output folder doesn't exist, create it
        if not os.path.exists(self.path_output_folder):
            os.makedirs(self.path_output_folder)

    def txt2csv(self, txt_file, csv_file):
        path_file = os.path.join(self.path_folder, txt_file)
        with open(path_file, 'r') as file:
            data = file.readlines()

        processed_data = []
        for line in data[14:]:
            split_line = [item.strip() for item in re.split(r'\s+', line) if item]
            processed_data.append(split_line)

        # Save to output folder
        output_path = os.path.join(self.path_output_folder, csv_file)
        with open(output_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(processed_data)

    def convert(self):
        file_names = [file for file in os.listdir(self.path_folder) if file.endswith('.txt')]
        print(f"Found {len(file_names)} TXT files: {file_names}")

        for file_name in file_names:
            csv_file = f"{file_name[:-4]}.csv"
            self.txt2csv(file_name, csv_file)
    
    def rename(self, group_name, file_rename):
        '''
        group_name: the name of the group to rename
        file_rename: the name of the file to rename
        '''
        file_names = [file for file in os.listdir(self.path_folder) if file.endswith('.csv') and group_name in file]
        
        for i, file_name in enumerate(file_names):
            csv_file = f"{file_rename}{i}.csv"
            os.rename(self.path_folder+file_name,self.path_folder+csv_file)
        print(f"Renamed {file_name} to {file_rename}")
