def get_model(config):
    if config.model == 'edm_da':
        from .edm_da import EDMDataAssimilation
        return EDMDataAssimilation(config)

    if config.model == 'utae':
        from .utae.model import UTAE
        return UTAE(config)
    if config.model == 'lstm':
        from .lstm.model import LSTM
        return LSTM(config)
    if config.model == 'metnet3':
        from .metnet3.model import MetNet3
        return MetNet3(config)