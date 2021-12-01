"""
This is a toy example that has been used to test the hi-ml package. It takes in input a dataset that contains text
files and it creates a copy of the same files with the text in upper case.

Keeping it here for quick debugging in the future. Once integration tests are in place inside hi-ml this can be removed.
"""
from pathlib import Path
import sys

if __name__ == '__main__':
    current_file = Path(__file__)
    sys.path.insert(0, str(current_file.parent/"hi-ml/src/"))
    print(f"Running {str(current_file)}")
    from health.azure.himl import submit_to_azure_if_needed
    run_info = submit_to_azure_if_needed(entry_script=current_file, 
                                         snapshot_root_directory=current_file.parent,
                                         workspace_config_path=Path(
                                             "/home/vsalvatelli/workspace/test_himl/config.json"),
                                         compute_cluster_name="training-pr-nc12",
                                         default_datastore="innereyedatasets",
                                         conda_environment_file=Path(
                                             "/home/vsalvatelli/workspace/test_himl/environment.yml"),
                                         input_datasets=["test-hi-ml"],
                                         output_datasets=["test-output-hi-ml-innereyedatasets"],
                                         )

    input_folder = Path(run_info.input_datasets[0] or Path("test_folder"))
    output_folder = Path(run_info.output_datasets[0] or Path("tmp/my_output/"))
    print(f"input folder {input_folder}")
    print(f"output folder {output_folder}")
    n = len([file for file in input_folder.rglob("*.txt")])
    print(f" Number of files {n}")
    if n > 0:
        for file in input_folder.rglob("*.txt"):
            print(f"Reading {file}")
            text_file = open(file, "r")
            contents = text_file.read()
            text_file.close()
            upper_content = contents.upper()
            output_file = output_folder / file
            with output_file.open("w") as f:
                string = f.write(upper_content)
                print(f"Output created at {output_file}")
    else:
        # If the dataset contains a single file the root folder will not be included
        print("single file detected")
        file = input_folder
        text_file = open(file, "r")
        contents = text_file.read()
        text_file.close()
        upper_content = contents.upper()
        parent_folder = file.parent
        output_file = output_folder / file.relative_to(parent_folder)
        with output_file.open("w") as f:
            string = f.write(upper_content)
            print(f"Output created at {output_file}")

