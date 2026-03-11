import pandas as pd

# df = pd.read_csv("/localdisk1/PARK/park_communications_medicine/data/validation_data/users_data_new.csv")
# df = df.drop(columns=["prolificID", "email", "us_state", "birthdate", "creation_timestamp", "last_sign_in_timestamp"])
# df.to_csv("/localdisk1/PARK/park_communications_medicine/data/validation_data/users_data_new.csv", index=False)

# df = pd.read_csv("/localdisk1/PARK/park_communications_medicine/data/validation_data/users_data.csv")
# df = df.drop(columns=["prolificID", "email", "us_state", "birthdate", "creation_timestamp", "last_sign_in_timestamp"])
# df.to_csv("/localdisk1/PARK/park_communications_medicine/data/validation_data/users_data.csv", index=False)

drop_columns = ["dob"]
file_paths = ["/localdisk1/PARK/park_communications_medicine/data/metadata_finger_tapping_left.csv", 
              "/localdisk1/PARK/park_communications_medicine/data/metadata_facial.csv",
              "/localdisk1/PARK/park_communications_medicine/data/metadata_finger_tapping_right.csv",
              "/localdisk1/PARK/park_communications_medicine/data/metadata_finger_tapping.csv",
              "/localdisk1/PARK/park_communications_medicine/data/metadata_speech.csv"]

for filename in file_paths:
    df = pd.read_csv(filename)
    df = df.drop(columns=drop_columns)
    df.to_csv(filename, index=False)