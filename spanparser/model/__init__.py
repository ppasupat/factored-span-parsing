def create_model(config, meta):
    if config.model.name == 'span':
        from spanparser.model.span_model import SpanModel
        return SpanModel(config, meta)
    raise ValueError('Unknown model name: {}'.format(config.model.name))
