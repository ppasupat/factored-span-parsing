def create_dataset(config, meta):
    if config.data.name == 'top':
        from spanparser.data.top import TopDataset
        return TopDataset(config, meta)
    raise ValueError('Unknown data name: {}'.format(config.data.name))
