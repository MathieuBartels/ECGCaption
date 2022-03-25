from torch.utils.data import Dataset

class UniversalECGDataset(Dataset):
    """Universal ECG dataset in PyTorch Dataset format.
    """

    def __init__(self, dataset_type, waveform_dir, dataset, transform=None,
                 label='Label', OOD_classname=None):
        """Initializes the UMCU ECG datasets.

        Args:
            dataset_type (str): Type of dataset, options are 'umcu',
                'universal' and 'physionet'. UMCU and universal datasets
                contain numpy files, while the physionet dataset contains
                matlab files.
            waveform_dir (str): Path of the folder with the raw waveform files
            dataset (pd.DataFrame): Pandas DataFrame with the dataset your are
                using. Minimally required columns for UMCU are: PseudoID,
                TestID, SampleBase and Gain. For universal and physionet
                datasets we need Filename, SampleBase and Gain.
            transform (list): List of transformations.
            label (str): Name of the y variable in the dataset.
        """
        assert dataset_type in ['umcu', 'universal', 'physionet']

        if (('PseudoID' in dataset and 'TestID' in dataset)
           or 'Filename' in dataset):
            self.dataset = dataset
        else:
            print(('Please provide either a PseudoID/TestID combination or'
                   'a Filename in the dataset.'))
            raise

        self.waveform_dir = waveform_dir
        self.transform = transform
        self.label = label
        self.dataset_type = dataset_type
        self.OOD_classname = OOD_classname

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_id = self.get_waveform(idx)
        # Add waveform, original sample base, gain and ID to sample
        sample = {
            'waveform': waveform,
            'samplebase': int(self.dataset['SampleBase'].iloc[idx]),
            'gain': float(self.dataset['Gain'].iloc[idx]),
            'id': sample_id,
        }

        # Sometimes additional information is needed (e.g. for a median cutoff)
        possible_cols = ['AcqDate', 'POnset', 'TOffset', 'VentricularRate',
                         'QOnset', 'POffset', 'QOffset', 'start_idx',
                         'end_idx', 'Age', 'Gender']
        for col in possible_cols:
            if col in self.dataset:
                sample[col.lower()] = self.dataset[col].iloc[idx]
                
        if self.OOD_classname:
            sample['OOD_class'] = self.OOD_classname

        if self.label in self.dataset.columns.values:
            label = self.dataset[self.label].iloc[idx]
            sample['label'] = label

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def get_waveform(self, idx):
        pass 
        # 
        # return waveform
