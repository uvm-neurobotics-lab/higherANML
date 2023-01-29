"""
Scans all subfolders in the given folder for all images with the given extension(s). Writes the full set of image
filenames to disk in the given folder.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        print(f"Scanning {len(iterable)} files...")
        return iterable


def existing_dir(str_path):
    """
    This function can be used as an argument type to fully resolve a user-supplied path and ensure it exists:
        parser.add_argument(..., type=argutils.existing_path, ...)
    An exception will be raised if the path does not exist.

    Args:
        str_path: The user-supplied path.

    Returns:
        pathlib.Path: The fully-resolved path object, if it exists.
    """
    path = Path(str_path).resolve()
    if path.is_dir():
        return path
    else:
        raise argparse.ArgumentTypeError(f"{str_path} ({path}) is not a valid directory.")


def search(search_dir, exts, is_top_level=False):
    found_files = []
    # Sorted to make the ordering deterministic.
    all_files = sorted(search_dir.iterdir())

    # Add matching files from the current directory.
    curr_files = [p for p in all_files if p.is_file() and p.suffix in exts]
    found_files.extend(curr_files)

    # Search in all subdirectories.
    subdirs = [p for p in all_files if p.is_dir()]
    if is_top_level:
        print(f"Scanning for files with extensions: {exts}.")
        subdirs = tqdm(subdirs)
    for d in subdirs:
        found_files.extend(search(d, exts))

    if is_top_level:
        print(f"Found {len(found_files)} files.")
    return found_files


def write_cache(search_dir, index_name, found_files):
    outfile = search_dir / index_name
    if outfile.exists():
        print(f"WARNING: Overwriting pre-existing index at {outfile}")
    else:
        print(f"Writing index to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(found_files, f)


def read_cache(search_dir, index_name):
    outfile = search_dir / index_name
    if not outfile.exists():
        print(f"ERROR: Could not find index file: {outfile}")
        sys.exit(os.EX_NOINPUT)
    else:
        with open(outfile, "rb") as f:
            return pickle.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("search_dir", metavar="PATH", type=existing_dir, help="The directory to search.")
    parser.add_argument("-e", "--extension", metavar="EXT", nargs="+",
                        help="One or more extensions to search for, including the period ('.'). Case sensitive.")
    parser.add_argument("-r", "--read-cache", action="store_true",
                        help="Rather than building a new index, read the existing one.")
    parser.add_argument("-n", "--index-name", metavar="FILENAME", default="index.pkl",
                        help="The name for the index file to read/write. Will be a pickle file.")
    parser.add_argument("--no-write", dest="write", action="store_false",
                        help="Do not write an index file after scanning.")

    # Parse and run.
    args = parser.parse_args(argv)
    if args.read_cache:
        files = read_cache(args.search_dir, args.index_name)
        print(f"Read {len(files)} files from index. First ten files:")
        for f in files[:10]:
            print("  " + str(f))
    else:
        if args.extension:
            files = search(args.search_dir, args.extension, True)
            if args.write:
                write_cache(args.search_dir, args.index_name, files)
        else:
            parser.error(f"You must supply at least one file extension to scan (-e, --extension).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
