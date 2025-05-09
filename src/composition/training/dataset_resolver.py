def get_dataset_files(dataset_name, r):
    """
    Get the list of dataset files based on the dataset name.
    """
    if dataset_name == "lakh_full":
        manifest_path = r.Rlocation("_main/datasets/lakh/lmd_full_valid_manifest.txt")
        with open(manifest_path) as f:
            return [r.Rlocation(path) for path in f.read().splitlines()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
