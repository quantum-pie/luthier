#!/usr/bin/env python3
"""
Script to collect all CUDA .so files from PyTorch's vendorized nvidia wheels
installed via rules_python/pip_install, and generate a BUILD.bazel file
for Bazel integration.
"""
import shutil
import re
from pathlib import Path
import subprocess

# Where we’ll collect the files to be used in Bazel
OUTPUT_DIR = Path("third_party/pytorch_cuda_libs/lib")
OUTPUT_BUILD_FILE = Path("third_party/pytorch_cuda_libs/BUILD.bazel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# This will run `bazel query` to list all relevant external repos with nvidia libraries
def find_nvidia_pip_packages():
    result = subprocess.run(["bazel", "query", "@pypi//..."], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to query Bazel:", result.stderr)
        return []

    packages = set()
    for line in result.stdout.splitlines():
        if "nvidia" in line:
            parts = line.split("//")
            if len(parts) > 1:
                pkg = parts[1].split(":")[0]  # e.g., nvidia_cublas_cu12
                packages.add(pkg)

    return sorted(packages)


# Create a temporary workspace directory to store the py_binary target
TEMP_WORKSPACE_DIR = Path("bazel_force_unpack")
TEMP_BUILD_FILE = TEMP_WORKSPACE_DIR / "BUILD.bazel"
TEMP_SCRIPT_FILE = TEMP_WORKSPACE_DIR / "force_unpack.py"


def generate_temp_target(packages):
    # Create dummy script that imports all modules to force Bazel unpack
    with open(TEMP_SCRIPT_FILE, "w") as py:
        py.write("import torch\n")
        py.write("torch.tensor([1.0]).cuda()\n")  # triggers lib resolution

    with open(TEMP_BUILD_FILE, "w") as f:
        f.write("py_binary(\n")
        f.write('    name = "force_cuda_unpack",\n')
        f.write('    srcs = ["force_unpack.py"],\n')
        f.write('    main = "force_unpack.py",\n')
        f.write('    deps = ["@pypi//torch"],\n')
        f.write('    visibility = ["//visibility:private"],\n')
        f.write(")\n")

    print(f"Temporary BUILD file created at {TEMP_BUILD_FILE}")

    # Build the dummy py_binary to trigger Bazel's unpacking of those packages
    print("Triggering Bazel to unpack external packages...")
    subprocess.run(["bazel", "build", f"//{TEMP_WORKSPACE_DIR.name}:force_cuda_unpack"], check=True)


# Use bazel cquery to find real .so files on disk
def collect_so_files_from_packages(_):
    print("Querying Bazel for .so file paths via cquery...")
    result = subprocess.run(["bazel", "cquery", "deps(@pypi//torch)", "--output=files"], capture_output=True, text=True)
    if result.returncode != 0:
        print("cquery failed:", result.stderr)
        return

    # Get Bazel's output_base path
    output_base_result = subprocess.run(["bazel", "info", "output_base"], capture_output=True, text=True)
    if output_base_result.returncode != 0:
        print("Failed to get output_base path:", output_base_result.stderr)
        return

    output_base = Path(output_base_result.stdout.strip())
    so_pattern = re.compile(r".*\.so(\.\d+)*$")

    seen = set()
    for line in result.stdout.splitlines():
        rel_path = Path(line)
        abs_path = output_base / rel_path
        if so_pattern.match(str(rel_path)) and abs_path.exists():
            target = OUTPUT_DIR / abs_path.name
            if target.name not in seen:
                seen.add(target.name)
                try:
                    shutil.copy2(abs_path, target)
                    print(f"Copied: {abs_path} -> {target}")
                except Exception as e:
                    print(f"Failed to copy {abs_path}: {e}")


# Generate the BUILD.bazel file to expose the .so files as a filegroup
def generate_build_file():
    with open(OUTPUT_BUILD_FILE, "w") as f:
        f.write("filegroup(\n")
        f.write('    name = "cuda_so_files",\n')
        f.write('    srcs = glob(["lib/*.so*"]),\n')
        f.write('    visibility = ["//visibility:public"],\n')
        f.write(")\n")

    print(f"BUILD.bazel written to {OUTPUT_BUILD_FILE}")


def main():
    packages = find_nvidia_pip_packages()
    print(f"Found {len(packages)} NVIDIA pip packages:", packages)

    generate_temp_target(packages)
    collect_so_files_from_packages(packages)
    generate_build_file()


if __name__ == "__main__":
    TEMP_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as e:
        print(f"Failed to collect dependencies: {e}")
    shutil.rmtree(TEMP_WORKSPACE_DIR, ignore_errors=False)
