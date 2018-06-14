import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("IDTF_dir", help="Dir of IDTFs.", type=str)
    args = parser.parse_args()

    files = os.listdir(args.IDTF_dir)
    # files = [f if f.split('.')[-1]=='features' for f in files]
    new_files = []
    for f in files:
        if f.split('.')[-1] == 'features':
            new_files.append(f)
    files = new_files
    print(files)

    for name in files:
        with open(os.path.join(args.IDTF_dir, name), 'r') as f:
            content = f.readlines()
        new_content = ''
        for line in content:
            new_line = ''
            for idx, c in enumerate(line):
                if c.isdigit():
                    new_line = line[idx:]
                    break
            new_content = new_content+new_line
        with open(os.path.join(args.IDTF_dir, name), 'w') as f:
            f.write(new_content)
        print(name, " Done.")
