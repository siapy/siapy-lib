from data_loader import data_loader


def show(cfg):
    dl = data_loader.DataLoader(cfg)
    dl.load_data()


