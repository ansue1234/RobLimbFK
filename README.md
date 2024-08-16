# Setup on cloud instance

## Lambda Cloud
- Get GLFS 
  - `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
  - `sudo apt-get install git-lfs`
- Get conda
  - `mkdir -p ~/miniconda3`
  - `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh`
  - `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`
  - `rm -rf ~/miniconda3/miniconda.sh`
  - `~/miniconda3/bin/conda init bash`


## Frequently encountered problems
- Files are just pointers to LFS
  - `git lfs fetch`
  - `git lfs pull`

# Create dataset
- move file into `data/raw_data`
- run `notebooks/data_cleaning.ipynb`
- 