import os
from shutil import copyfile

def get_mha_files(root_dir):
    case_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.startswith("Case")]
    mha_files = []

    for case_dir in case_dirs:
        mha_dir = os.path.join(case_dir, "Mha")
        if not os.path.exists(mha_dir):
            continue

        mha_files_in_dir = [os.path.join(mha_dir, f) for f in os.listdir(mha_dir) if f.endswith(".mha")]
        mha_files.extend(mha_files_in_dir)

    return mha_files

def rename_mha_files(mha_files, output_dir):
    for mha_file in mha_files:
        # Get the case and time point numbers from the file name
        file_name = os.path.basename(mha_file)
        case_num = int(mha_file.split("Case")[1].split("Pack")[0])
        time_point_num = int(file_name.split("T")[1].split(".")[0])
        if time_point_num==0:
            temp_output_dir = output_dir+'\\Fixed\\'
        else:
            temp_output_dir = output_dir+'\\Moving\\'
        # Create the output file name
        new_file_name = f"Case{case_num}_T{time_point_num}.mha"
        output_file = os.path.join(temp_output_dir, new_file_name)

        # Rename the file
        copyfile(mha_file,output_file)
        # os.rename(mha_file, output_file)

if __name__ == "__main__":
    root_dir = "data\\Unionset\\"
    output_dir = "data\\Train"
    mha_files = get_mha_files(root_dir)
    rename_mha_files(mha_files, output_dir)
