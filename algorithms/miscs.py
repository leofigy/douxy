import pandas as pd


def get_data(filename, omit):
    valida, omits = pd.DataFrame(),pd.DataFrame()
    print(omit)
    headers = [*pd.read_csv(filename, nrows=1)]
    # calculating omit files
    valid_list = headers
    if omit:
        valid_list = list(filter(lambda h: h not in omit, headers))

    # filtered items for processing
    valida = pd.read_csv(filename, encoding='iso-8859-1', usecols=valid_list)
    # omitted items
    if omit:
        omits = pd.read_csv(filename, encoding='iso-8859-1', usecols=omit)
    return valida, omits, valid_list

