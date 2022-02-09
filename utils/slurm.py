"""
Utilities for launching Slurm jobs.

These utilities depend on Neuromanager (https://github.com/lfrati/neuromanager).
"""
import os
import re
import subprocess


def call_sbatch(cmd, verbose=False, dry_run=False):
    """
    Run the given command, which is assumed to be a call to `sbatch` or similar. We will expect that the console output
    contains the ID of a job which was launched.
        - In the case of success, this will output the ID of the job which was launched.
        - In the case of failure, this will print the full console log for debugging.

    Args:
        cmd: A command for `sbatch` or an `sbatch` wrapper like Neuromanager's `launcher`.
        verbose: Whether to print the console output of `cmd`, rather than swallowing it.
        dry_run: Do not actually call the command, instead just print it to the console.

    Returns:
        int: The exit code of the called process, or 0 in the case of a dry run.
    """
    if dry_run:
        print("Command that would be run:")
        print("    " + " ".join(cmd))
        return os.EX_OK

    try:
        print("Running command: " + " ".join(cmd))
        if verbose:
            # If verbose, just let the launcher output directly to console.
            stderr = None
            stdout = None
        else:
            # Normally, redirect stderr -> stdout and capture them both into stdout.
            stderr = subprocess.STDOUT
            stdout = subprocess.PIPE
        res = subprocess.run(cmd, text=True, check=True, stdout=stdout, stderr=stderr)
        # Find the Slurm job ID in the output and print it, if we captured the output.
        if not verbose:
            match = re.search(r"Submitted batch job (\d+)", res.stdout)
            if not match:
                print("WARNING: Could not find Slurm job ID in launcher output. Output of launcher:")
                print(res.stdout)
            else:
                print(match.group(0))
        return res.returncode
    except subprocess.CalledProcessError as e:
        # Print the output if we captured it, to allow for debugging.
        if not verbose:
            print("LAUNCH FAILED. Launcher output:")
            print("-" * 80)
            print(e.stdout)
            print("-" * 80)
        raise
