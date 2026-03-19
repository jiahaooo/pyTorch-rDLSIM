def define_Dataset(opt, phase):

    if opt['data'] in ['2d-sim']:
        from rDL_data.dataset_2d import Dataset_2d
        dataset = Dataset_2d(opt, phase)
    else:
        raise NotImplementedError

    return dataset