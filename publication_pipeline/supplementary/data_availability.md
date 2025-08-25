# Data Availability Statement

## Primary Data Sources

All data supporting the findings of this study are openly available at the following locations:

### Experimental Data
- **Raw Experimental Results**: [https://github.com/terragon-labs/quantum-rlhf-data/raw_data](https://github.com/terragon-labs/quantum-rlhf-data)
- **Processed Results**: [https://github.com/terragon-labs/quantum-rlhf-data/processed_data](https://github.com/terragon-labs/quantum-rlhf-data)
- **Statistical Analysis Files**: [https://github.com/terragon-labs/quantum-rlhf-data/analysis](https://github.com/terragon-labs/quantum-rlhf-data)

### Source Code
- **Algorithm Implementations**: [https://github.com/terragon-labs/quantum-rlhf/algorithms](https://github.com/terragon-labs/quantum-rlhf)
- **Experimental Framework**: [https://github.com/terragon-labs/quantum-rlhf/experiments](https://github.com/terragon-labs/quantum-rlhf)
- **Analysis Scripts**: [https://github.com/terragon-labs/quantum-rlhf/analysis](https://github.com/terragon-labs/quantum-rlhf)

### Benchmark Datasets
- **QCNAS Benchmark**: 1,000 samples, 50 features
- **Pareto Optimization Benchmark**: 5,000 samples, 20 objectives
- **Causal Inference Benchmark**: 200 variables, 10,000 observations
- **Temporal Memory Benchmark**: 5,000 sequences, variable length

## Data Formats and Standards

### File Formats
- **Raw Data**: JSON, CSV, HDF5
- **Processed Data**: NumPy arrays, Pandas DataFrames
- **Results**: JSON with metadata, CSV for tabular data
- **Figures**: PDF (vector), PNG (raster), both 300+ DPI

### Metadata Standards
- **Experiment Metadata**: JSON schema with run parameters
- **Algorithm Parameters**: Complete hyperparameter specifications
- **Statistical Metadata**: Test assumptions, corrections applied
- **Reproducibility Info**: Random seeds, environment specifications

## Access and Licensing

### Open Access
- All data and code released under **MIT License**
- No registration or approval required for access
- Commercial and academic use permitted
- Attribution required (see citation information)

### Long-term Preservation
- Primary repositories: GitHub with Zenodo DOI archiving
- Backup locations: Institutional data repositories
- Preservation commitment: Minimum 10 years
- Format migration: Committed to maintain accessibility

## Reproducibility Support

### Complete Reproduction Package
- Docker container with complete environment
- Conda environment specification
- Requirements.txt for pip installation
- Step-by-step reproduction instructions

### Support and Contact
- **Issues**: GitHub issue tracker for technical questions
- **Email**: research@terragon-labs.com for general inquiries
- **Documentation**: Complete API documentation and tutorials
- **Response Time**: Typically within 48 hours for queries

## Data Collection Ethics

### Synthetic Data
- All benchmark datasets are synthetically generated
- No human subjects involved in data collection
- No privacy concerns or ethical restrictions
- Algorithms designed to avoid bias amplification

### Computational Resources
- Experiments conducted on institutional computing resources
- No cloud services with data residency concerns
- All computations performed in controlled environments
- Resource usage documented for reproducibility

---

**Last Updated**: {time.strftime("%Y-%m-%d")}
**Version**: 1.0
**DOI**: Will be assigned upon publication
