import pandas as pd

def get_data(filename, omit):
    valida, omits = pd.DataFrame(),pd.DataFrame()

    # read all
    all = pd.read_csv(filename, encoding='iso-8859-1')
    print(all)
    # main fields to be used for training just skipping the ignores
    training_mask = list(filter(lambda h: h not in omit, all.columns))
    valida = all[training_mask]
    if omit:
        omits = all[omit]
    return valida, omits, training_mask

#def add_data_column()