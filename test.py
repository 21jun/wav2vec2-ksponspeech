import datasets

train_dataset = datasets.load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, split=data_args.train_split_name
    )