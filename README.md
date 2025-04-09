# 🧠 EEG Tutorial
## 📥 EEG Dataset Downloader

This project provides a unified interface for downloading and preprocessing multiple EEG datasets for motor imagery (MI) research. It uses the MOABB (Mother of All BCI Benchmarks) library to access various datasets and processes them into a consistent format for further analysis.

## 📊 Supported Datasets

The following datasets are currently supported:

<details>
<summary>1. **BNCI2014_001**: A 4-class motor imagery dataset with 9 subjects.</summary>

- 4 classes: left hand, right hand, feet, tongue
- Trial duration: 4 seconds (2-6s)
- Rest period: 2-3 seconds
</details>

<details>
<summary>2. **BNCI2014_002**: A 2-class motor imagery dataset with 14 subjects.</summary>

- 2 classes: right hand, feet
- Trial duration: 5 seconds (3-8s)
- Rest period: 2-3 seconds
- 15 electrodes (3 Laplacian derivations at C3, Cz, C4)
</details>

<details>
<summary>3. **Lee2019_MI**: A 2-class motor imagery dataset with 54 subjects.</summary>

- 2 classes: left hand, right hand
- Trial duration: 4 seconds (3-7s)
- Rest period: 6 seconds (±1.5s)
- 62 Ag/AgCl electrodes
</details>

<details>
<summary>4. **PhysionetMI**: A 5-class motor imagery dataset with 109 subjects.</summary>

- 5 classes: rest, left hand, right hand, hands, feet
- Trial duration: Varies by run type
- Rest period: 3-4 seconds
- Includes both imagined and executed movements
</details>

<details>
<summary>5. **Schirrmeister2017**: A 4-class motor imagery dataset with 14 subjects.</summary>

- 4 classes: right hand, left hand, rest, feet
- Trial duration: 4 seconds (0-4s)
- Rest period: 3-4 seconds
- 128 electrodes (44 covering motor cortex)
</details>

### ✨ Features

<details>
<summary>**Unified Data Format**: All datasets are processed and saved in a consistent CSV format.</summary>

Each dataset is saved as CSV files with the following structure:
- **Columns**: EEG channels, time points, condition, label
- **Attributes**: Sampling rate, electrodes, reference, trial duration, rest period, etc.
</details>

<details>
<summary>**Standardized Preprocessing**: Applies the same preprocessing steps (bandpass filtering, epoch creation) to all datasets.</summary>

All datasets are processed with:
- Bandpass filtering (8-30Hz)
- Epoch creation based on task-specific time windows
- Consistent event encoding
</details>

<details>
<summary>**Comprehensive Metadata**: Includes dataset-specific information as attributes in the saved files.</summary>

Each dataset includes specific metadata such as:
- Sampling rate
- Electrode configuration
- Reference and ground locations
- Trial duration and rest periods
- Task-specific information
</details>

<details>
<summary>**Error Handling**: Implements robust error handling and retry mechanisms for reliable data processing.</summary>

The code includes:
- Automatic retry for failed downloads
- Graceful error handling for missing data
- Detailed logging of processing steps
</details>

<details>
<summary>**Modular Design**: Each dataset has its own processing function, making it easy to add new datasets.</summary>

The codebase is structured with:
- Separate processing functions for each dataset
- Consistent interface across all datasets
- Easy extensibility for adding new datasets
</details>

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

<details>
<summary>#### BNCI2014_001</summary>

- 4 classes: left hand, right hand, feet, tongue
- Trial duration: 4 seconds (2-6s)
- Rest period: 2-3 seconds
</details>

<details>
<summary>#### BNCI2014_002</summary>

- 2 classes: right hand, feet
- Trial duration: 5 seconds (3-8s)
- Rest period: 2-3 seconds
- 15 electrodes (3 Laplacian derivations at C3, Cz, C4)
</details>

<details>
<summary>#### Lee2019_MI</summary>

- 2 classes: left hand, right hand
- Trial duration: 4 seconds (3-7s)
- Rest period: 6 seconds (±1.5s)
- 62 Ag/AgCl electrodes
</details>

<details>
<summary>#### PhysionetMI</summary>

- 5 classes: rest, left hand, right hand, hands, feet
- Trial duration: Varies by run type
- Rest period: 3-4 seconds
- Includes both imagined and executed movements
</details>

<details>
<summary>#### Schirrmeister2017</summary>

- 4 classes: right hand, left hand, rest, feet
- Trial duration: 4 seconds (0-4s)
- Rest period: 3-4 seconds
- 128 electrodes (44 covering motor cortex)
</details>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MOABB library for providing access to the datasets
- MNE library for EEG processing capabilities
- The original dataset creators and contributors 