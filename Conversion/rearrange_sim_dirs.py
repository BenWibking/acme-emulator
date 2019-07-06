from pathlib import Path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_root_path')
    args = parser.parse_args()

    # now iterate through subdirs of sim_root_path
    root_path = Path(args.sim_root_path)
    subdirs = [x for x in root_path.iterdir() if x.is_dir()]
    data_dirs = ['Rockstar', 'FOF', 'power']
    for data_dir in data_dirs:
        print("creating {}".format(data_dir))
        new_data_dir = root_path / data_dir
        new_data_dir.mkdir(exist_ok=True)

    for subdir in subdirs:
        if subdir.name not in data_dirs:
            for data_dir in data_dirs:
                old_dir = subdir / data_dir
                if old_dir.exists():
                    new_dir = subdir.parent / data_dir / subdir.name
                    print("{} --> {}".format(old_dir, new_dir))
                    old_dir.rename(new_dir)
            subdir.rmdir()
                
