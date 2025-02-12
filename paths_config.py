paths_config = {
    # Temporary folder
    "temp_folder": "/tmp",

    # The folder where 10k satellite image nc files are extracted and converted to train and test data npy files, stored in the same folder
    "dataset_folder": "dataset",
    # Train and test data npy filenames (in dataset folder)
    "train_data_filename": "train_data.npy",
    "test_data_filename": "test_data.npy",

    # Train runs folder. (A subdirectory will be generated for each training run)
    "train_runs_folder": "runs",

    # Trained models folder
    "trained_models_folder": "results/trained_models",

    # Results folder
    "results_folder": "results",
    # Evaluations pickle filename (in results folder)
    "eval_results_filename": "eval_results.pickle",

    # Article graphs
    "graphs_folder": "plots"
}