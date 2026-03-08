# dsl26

## Quick Setup

### 1. Clone the repository
```bash
git clone https://github.com/luca17baumann/dsl26.git
cd dsl26
```

### 2. Configure your API key
Rename `.env_example` to `.env` and add your API key:
```bash
mv .env_example .env
```
Then open `.env` and replace the placeholder with your key. You can find your API key at [serving.swissai.cscs.ch](https://serving.swissai.cscs.ch).

### 3. Create the conda environment
```bash
conda env create -f environment.yml
conda activate dsl26
```