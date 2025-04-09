# 🧠 EEG Tutorial
## 📥 EEG Dataset Downloader

This project provides a unified interface for downloading and preprocessing multiple EEG datasets for motor imagery (MI) research. It uses the [MOABB](https://github.com/NeuroTechX/moabb) (Mother of All BCI Benchmarks) library to access various datasets and processes them into a consistent format for further analysis.

### 📊 Supported Datasets

The following datasets are currently supported:

1. **BNCI2014_001**: A 4-class motor imagery dataset with 9 subjects.
- 4 classes: left hand, right hand, feet, tongue
- Trial duration: 4 seconds (2-6s)
- Rest period: 2-3 seconds

2. **BNCI2014_002**: A 2-class motor imagery dataset with 14 subjects.
- 2 classes: right hand, feet
- Trial duration: 5 seconds (3-8s)
- Rest period: 2-3 seconds
- 15 electrodes (3 Laplacian derivations at C3, Cz, C4)

3. **Lee2019_MI**: A 2-class motor imagery dataset with 54 subjects.
- 2 classes: left hand, right hand
- Trial duration: 4 seconds (3-7s)
- Rest period: 6 seconds (±1.5s)
- 62 Ag/AgCl electrodes

4. **PhysionetMI**: A 5-class motor imagery dataset with 109 subjects.
- 5 classes: rest, left hand, right hand, hands, feet
- Trial duration: Varies by run type
- Rest period: 3-4 seconds
- Includes both imagined and executed movements

5. **Schirrmeister2017**: A 4-class motor imagery dataset with 14 subjects.
- 4 classes: right hand, left hand, rest, feet
- Trial duration: 4 seconds (0-4s)
- Rest period: 3-4 seconds
- 128 electrodes (44 covering motor cortex)

### ✨ Features

**Unified Data Format**: All datasets are processed and saved in a consistent CSV format.

Each dataset is saved as CSV files with the following structure:
- **Columns**: EEG channels, time points, condition, label
- **Attributes**: Sampling rate, electrodes, reference, trial duration, rest period, etc.

**Standardized Preprocessing**: Applies the same preprocessing steps (bandpass filtering, epoch creation) to all datasets.

All datasets are processed with:
- Bandpass filtering (8-30Hz)
- Epoch creation based on task-specific time windows
- Consistent event encoding

**Comprehensive Metadata**: Includes dataset-specific information as attributes in the saved files.

Each dataset includes specific metadata such as:
- Sampling rate
- Electrode configuration
- Reference and ground locations
- Trial duration and rest periods
- Task-specific information

**Error Handling**: Implements robust error handling and retry mechanisms for reliable data processing.

The code includes:
- Automatic retry for failed downloads
- Graceful error handling for missing data
- Detailed logging of processing steps

**Modular Design**: Each dataset has its own processing function, making it easy to add new datasets.

The codebase is structured with:
- Separate processing functions for each dataset
- Consistent interface across all datasets
- Easy extensibility for adding new datasets

### 📋 Requirements

- Python 3.6+
- MOABB
- MNE
- NumPy
- Pandas

### 🚀 Installation

1. Clone this repository:
   ```
   git clone https://github.com/zyao197/EEG-Tutorial.git
   cd EEG-Tutorial
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### 💻 Usage

To download and process all datasets:

```
cd download
python download_all_datasets.py
```

The processed data will be saved in the following directories:
- `./data_bnci2014_001/`
- `./data_bnci2014_002/`
- `./data_lee2019_mi/`
- `./data_physionet_mi/`
- `./data_high_gamma/`

### 📝 Data Format

Each dataset is saved as CSV files with the following structure:

- **Columns**: EEG channels, time points, condition, label
- **Attributes**: Sampling rate, electrodes, reference, trial duration, rest period, etc.

### 📚 Dataset-Specific Information

#### BNCI2014_001
- 4 classes: left hand, right hand, feet, tongue
- Trial duration: 4 seconds (2-6s)
- Rest period: 2-3 seconds

#### BNCI2014_002
- 2 classes: right hand, feet
- Trial duration: 5 seconds (3-8s)
- Rest period: 2-3 seconds
- 15 electrodes (3 Laplacian derivations at C3, Cz, C4)

#### Lee2019_MI
- 2 classes: left hand, right hand
- Trial duration: 4 seconds (3-7s)
- Rest period: 6 seconds (±1.5s)
- 62 Ag/AgCl electrodes

#### PhysionetMI
- 5 classes: rest, left hand, right hand, hands, feet
- Trial duration: Varies by run type
- Rest period: 3-4 seconds
- Includes both imagined and executed movements

#### Schirrmeister2017
- 4 classes: right hand, left hand, rest, feet
- Trial duration: 4 seconds (0-4s)
- Rest period: 3-4 seconds
- 128 electrodes (44 covering motor cortex)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙌 Acknowledgments

- [MOABB](https://github.com/NeuroTechX/moabb) library for providing access to the datasets
- [MNE](https://mne.tools/stable/index.html) library for EEG processing capabilities
- The original dataset creators and contributors 