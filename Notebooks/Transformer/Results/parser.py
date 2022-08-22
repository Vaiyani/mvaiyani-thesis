from collections import defaultdict
import pandas as pd

data = defaultdict(list)
filename = 'result-transformers-pos.txt'
with open(filename) as file:
    for line in file:
        if line == '\n':
            continue
        if line[:3] == 'mse':
            result = line.replace(" ", "").split(',')
            data['mse'].append(result[0].split(':')[1])
            data['mae'].append(result[1].split(':')[1])
            data['rmse'].append(result[2].split(':')[1])
            data['mape'].append(result[3].split(':')[1])
            data['mspe'].append(result[4].split(':')[1])
            data['rse'].append(result[5].split(':')[1])
            data['R2'].append(result[6].split(':')[1])
        else:
            attributes = line.split(sep='_')
            model = attributes[0]
            data['lookback'].append(attributes[1].split('-')[1])
            data['future'].append(attributes[3].split('-')[1])
            data['pos'].append(attributes[5].split('-')[1])
            data['val'].append(attributes[6].split('-')[1])
            data['temp'].append(attributes[7].split('-')[1])

df = pd.DataFrame.from_dict(data)
