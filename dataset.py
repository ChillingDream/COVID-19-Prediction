import pandas as pd

boolean_features = [
    'SEX', 'PATIENT_TYPE', 'PNEUMONIA',
    'PREGNANT', 'DIABETES', 'COPD',
    'ASTHMA', 'INMSUPR', 'HIPERTENSION',
    'CARDIOVASCULAR', 'RENAL_CHRONIC',
    'OTHER_DISEASE', 'OBESITY', 'TOBACCO',
    'INTUBED', 'ICU'
]


class CovidDataset:
    def __init__(self, path):
        data = pd.read_csv(path)
        data['label'] = data['DATE_DIED'] != '9999-99-99'
        data.drop('DATE_DIED', axis=1, inplace=True)
        for name in data.columns:
            if name in boolean_features:
                data = data[data[name].isin([1, 2])]
        print(f'We get {len(data)} complete entries')

        for name in data.columns:
            if name in boolean_features:
                data[name] = data[name] == 1
            elif name == 'AGE':
                data.loc[data[name].between(1, 10), name] = 0
                data.loc[data[name].between(11, 30), name] = 1
                data.loc[data[name].between(31, 60), name] = 2
                data.loc[data[name].between(60, 90), name] = 3
                data.loc[data[name].between(90, 150), name] = 4
        data.loc[data['MEDICAL_UNIT'] >= 4, 'MEDICAL_UNIT'] = 4
        self.data = data
    
    def __getitem__(self, index):
        return self.data.iloc[index]
        return list(self.data.iloc[index])
    
    def __len__(self):
        return len(self.data)
