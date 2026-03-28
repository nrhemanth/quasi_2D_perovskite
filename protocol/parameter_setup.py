import pandas as pd
import torch
import matplotlib.pyplot as plt


class MaraData: 
    def __init__(self, data_frame, BOtype='3DBO'):
        self.data_frame = data_frame
        self.BOtype = BOtype
        if BOtype == '3DBO':
            self.bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
            self.variables = ['Anneal Time','PbI2_vol', 'BAAc_vol', 'MAI_vol','R PbI2', 'R BAAc', 'R MAI']
        elif BOtype == '4DBO':
            self.bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
            self.variables = ['Anneal Time', 'Temperature', 'PbI2_vol', 'BAAc_vol', 'MAI_vol','R PbI2', 'R BAAc', 'R MAI']
        else:
            raise ValueError(f"Unsupported BO type: {BOtype}. Supported types are '3DBO' and '4DBO'.")
        
    def Convert2Volume(self, df_candidates, max_volume=100.0):
        """
        Convert the stoichiometric candidates to component volumes.
        The final volume of each mixture is scaled to a total of 200 µL.
        Assumes candidates_df has columns 'BAAc' and 'MAI'.
        The remaining fraction goes to 'PbI2' (i.e., PbI2 = 1 - BAAc - MAI).
        """
        max_volume = max_volume

        df = df_candidates.copy()

        # Compute PbI2
        df['R PbI2'] = 1.0 
        
        total = df['R PbI2'] + df['R BAAc'] + df['R MAI']
        fract = max_volume/total

        # Scale to total volume
        df['PbI2_vol'] = df['R PbI2'] * fract
        df['BAAc_vol'] = df['R BAAc'] * fract
        df['MAI_vol'] = df['R MAI'] * fract
        
        # Save to a new DataFrame
        self.volume_df = df[['Anneal Time','PbI2_vol', 'BAAc_vol', 'MAI_vol']].sort_values('Anneal Time', ascending=False).reset_index(drop=True)
        full_df = df[self.variables].sort_values('Anneal Time', ascending=False).reset_index(drop=True).round(2)
        return full_df

    def duplicate_candidates(self, count, num_repets=3, max_volume=100.0):
        self.full_df = self.Convert2Volume(self.data_frame, max_volume=max_volume)
        duplicated_conditions = pd.concat([self.full_df] * num_repets).sort_values(by=['Anneal Time', 'R MAI', 'R BAAc'], ascending=False).reset_index(drop=True)
        duplicated_conditions.insert(0, 'ID', range((count - 1) * len(self.full_df)*num_repets + 1, count * len(self.full_df)*num_repets + 1))
        return duplicated_conditions
    
    def append_conditions_to_csv(self, Round_name, counter, path_save, num_samples=6, num_repets=2, max_volume=100.0):
        self.df_save = pd.read_csv(path_save)
        # Generate the 96 well plate layout
        plate_well = [f"{chr(col)}{row}" for col in range(ord('A'), ord('H') + 1) for row in range(1, 13)]
        # Put conditions in a dictionary
        plate_capacity = 8  # Number of rows per plate
        plate_number = (counter - 1) // plate_capacity + 1
        row_in_plate = (counter - 1) % plate_capacity + 1

        start_index = (row_in_plate - 1) * num_samples * num_repets
        end_index = row_in_plate * num_samples * num_repets

        # Duplicate the rows in lhs_df by num_repets
        duplicated_df = self.duplicate_candidates(counter, num_repets=num_repets, max_volume=max_volume)
        # Assign wells to the duplicated conditions
        duplicated_df['well'] = plate_well[start_index:end_index]
        # Save a new csv file with the duplicated conditions
        duplicated_df.to_csv(f'Data/{Round_name}.csv', index=False)
        # Append the new conditions to the existing CSV

        next_condition = duplicated_df[['ID','well']+self.variables]
        next_condition.to_csv(path_save, mode= 'a', header=False, index=False)
        self.next_condition = next_condition
        print(f"Conditions appended to {path_save}")

    # def append_conditions_to_csv_4DBO(self, Round_name, counter, path_save, num_samples=6, num_repets=2, max_volume=100.0):
    #     self.df_save = pd.read_csv(path_save)
    #     # Generate the 96 well plate layout
    #     plate_well = [f"{chr(col)}{row}" for col in range(ord('A'), ord('H') + 1) for row in range(1, 13)]
    #     # Put conditions in a dictionary
    #     plate_capacity = 8  # Number of rows per plate
    #     plate_number = (counter - 1) // plate_capacity + 1
    #     row_in_plate = (counter - 1) % plate_capacity + 1

    #     start_index = (row_in_plate - 1) * num_samples * num_repets
    #     end_index = row_in_plate * num_samples * num_repets

    #     # Duplicate the rows in lhs_df by num_repets
    #     duplicated_df = self.duplicate_candidates(counter, num_repets=num_repets, max_volume=max_volume)
    #     # Assign wells to the duplicated conditions
    #     duplicated_df['well'] = plate_well[start_index:end_index]
    #     # Save a new csv file with the duplicated conditions
    #     duplicated_df.to_csv(f'Data/{Round_name}.csv', index=False)
    #     # Append the new conditions to the existing CSV
    #     next_condition = duplicated_df[['ID', 'well', 'Anneal Time', 'Temperature', 'PbI2_vol', 'BAAc_vol', 'MAI_vol','R PbI2', 'R BAAc', 'R MAI']]
    #     next_condition.to_csv(path_save, mode= 'a', header=False, index=False)
    #     self.next_condition = next_condition
    #     print(f"Conditions appended to {path_save}")

    def plot_available_conditions(self):
        # Draw the wells
        # Highlight the filled wells on the 96 well plate layout
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 9)
        # Read the wells already present in the csv_save file
        filled_wells = self.df_save["well"].values if not self.df_save.empty else []
        for col in range(1, 13):
            for row, label in enumerate("ABCDEFGH", start=1):
                well_label = f"{label}{col}"
                if well_label in self.next_condition["well"].values:
                    color = 'green'  # Highlight newly added wells in green
                elif well_label in filled_wells:
                    color = 'orange'  # Highlight wells already in csv_save in orange
                else:
                    color = 'lightblue'  # Empty wells in light blue
                ax.add_patch(plt.Circle((col, 9 - row), 0.4, color=color, ec='black'))
                ax.text(col, 9 - row, well_label, ha='center', va='center', fontsize=8)

        # Customize the plot
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("96 Well Plate Layout with Filled Wells", fontsize=14)
        plt.show()
    def get_full_df(self):
        return self.full_df
    
    def get_volume_df(self): 
        return self.volume_df