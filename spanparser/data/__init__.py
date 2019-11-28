def create_dataset(config, meta):
    if config.data.name == 'demo':
        from spanparser.data.demo import DemoDataset
        return DemoDataset(config, meta)
    raise ValueError('Unknown data name: {}'.format(config.data.name))
