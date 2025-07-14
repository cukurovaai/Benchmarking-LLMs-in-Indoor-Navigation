import os

def main():
    main_dir = "../../drive/vlmaps_dataset"

    for dir_name in os.listdir(main_dir):
        dir_path = os.path.join(main_dir, dir_name)
        pose_dir_path = os.path.join(dir_path, "pose")

        if os.path.isdir(dir_path):
            if not os.path.exists(pose_dir_path):
                os.mkdir(pose_dir_path)

            poses_file = os.path.join(dir_path, "poses.txt")
            output_filename = poses_file.split('/')[4]

            with open(poses_file, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                output_file = f"{output_filename}_{i+1}.txt"
                output_file_path = os.path.join(pose_dir_path, output_file)

                with open(output_file_path, 'w') as new_file:
                    new_file.write(line)

if __name__ == '__main__':
    main()
