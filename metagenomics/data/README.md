# METAGENOMICS


1) Activate conda 
```
 source /home/papastrai/.bashrc
```
 2) activate virtual environment

```
conda activate idppy2
```

3) Run CAMISIM

```
python metagenomesimulation.py defaults/mini_config.ini
```

### IMPORTANT ###

change temporary output folder in mini_config.ini from /tmp to some other folder e.g.

```
temp_directory=/home/bcpl/scratch/idpmetagenome/tmp
```

4) To run plass assembler 

```
 plass assemble   ./../path/to/simulation_output_folder/anonymous_gsa_pooled.fasta.gz  protein_assembly.fas tmp

 ```
 
 
 ## CAMISIM simulation software
 
 ### Installation:
#### From Source
Download CAMISIM from the github repository into any directory and make sure the following Dependencies are fullfilled. The metagenome simulation pipeline can be downloaded using git:

    git clone https://github.com/CAMI-challenge/CAMISIM
    
Please make sure that a software is of the specified version, as outdated (or sadly sometimes too new) software can cause errors. The path to the binaries of those tools can be set in the configuration file.

#### Docker
You can also use Docker image with necessary tools installed:
```
docker pull cami/camisim:latest
```

Please see usage section for usage instructions.

* Dependencies
Operation system
The pipeline has only been tested on UNIX systems and might fail to run on any other systems (and no support for these will be provided).

Software
The following software is required for the simulation pipeline to run:

python >=3
Python is the programming language most scripts were written in.

Biopython
Python package used in the Pipeline for reading/writing sequence files.

BIOM
Python package for the handling of BIOM files

NumPy
Python package offering methods for scientific computing.

Matplotlib
Python package offering methods for plotting graphs.

Perl 5 and the library XML::Simple
Perl 5 is a programming language, used in the script that generates a perfect assembly from bam files. The perl library XML::Simple can be installed from the unix package 'libxml-simple-perl'

wgsim
Read simulator which offers error-free and uniform error rates. wgsim is also shipped within CAMISIM and does not have to be installed manually.

NanoSim
Read simulator for the generation of Oxford Nanopore Technologies (ONT) reads. Does not have a random seed in the original version, so cloning and installing the fork by abremges is advised if NanoPore reads should be simulated.

PBsim
Read simulator for generating Pacific Biosciences (PacBio) reads. Has to be cloned/installed manually if PacBio reads should be simulated.

SAMtools 1.0
Microbial ecology software.

Hardware
The simulation will be conducted primarily in the temporary folder declared in the configuration file or system tmp location by default. The results will then be moved to the output directory. Be sure to have enough space at both locations. Required hard drive space and RAM can vary a lot. Very small simulations can be run on a laptop with 4GB RAM, while realistic simulations of several hundred genomes, given a realistic metagenome size, can easily require several hundreds of gigabyte of both RAM and HD space, the chosen metagenome size being the relevant factor.

Resources
A database dump of the NCBI taxonomy is included, current versions can be downloaded from the NCBI FTP-Server.

Genomes
If the community design should be performed de novo, genomes in fasta format to be sampled from are needed. Otherwise they will be downloaded from the NCBI complete genomes.

The de novo community design needs three files to run:

A file containing, tab separated, a genome identifier and that path to the file of the genome.

A file containing, tab separated, a genome identifier and that path to the gen annotation of genome. This one is uses in case strains are simulated based on a genome

A [[meta data file|meta-data-file-format] that contains, tab separated and with header, genome identifier, novelty categorization, otu assignment and a taxonomic classification.

NCBI taxdump
At minimum the following files are required: "nodes.dmp", "merged.dmp", "names.dmp"
 
 
 ### USAGE:
from_profile:
```
 python metagenome_from_profile.py -p defaults/mini.biom
```
or de novo:
```
 python metagenomesimulation.py defaults/mini_config.ini
```
To check CAMISIM is working properly, you can perform a test run using the second command above:

It takes about one hour and creates a ~2.4GB folder "out/" in your CAMISIM directory of a small 10 samples, 24 genomes data set. The configuration file as well as the used mapping files genome_to_id and metadata are available in the defaults directory.
 
 ## PLASS ASSEMBLER 
 ### Install Plass
Plass can be install via [conda](https://github.com/conda/conda) or as statically compiled Linux version. Plass requires a 64-bit Linux/MacOS system (check with `uname -a | grep x86_64`) with at least the SSE2 instruction set.
    
     # install from bioconda
     conda install -c conda-forge -c bioconda plass 
     # static build with AVX2 (fastest)
     wget https://mmseqs.com/plass/plass-linux-avx2.tar.gz; tar xvfz plass-linux-avx2.tar.gz; export PATH=$(pwd)/plass/bin/:$PATH
     # static build with SSE4.1
     wget https://mmseqs.com/plass/plass-linux-sse41.tar.gz; tar xvfz plass-linux-sse41.tar.gz; export PATH=$(pwd)/plass/bin/:$PATH
     # static build with SSE2 (slowest, for very old systems)
     wget https://mmseqs.com/plass/plass-linux-sse2.tar.gz; tar xvfz plass-linux-sse2.tar.gz; export PATH=$(pwd)/plass/bin/:$PATH
     
 

## How to assemble
Plass can assemble both paired-end reads (FASTQ) and single reads (FASTA or FASTQ):

      # assemble paired-end reads 
      plass assemble examples/reads_1.fastq.gz examples/reads_2.fastq.gz assembly.fas tmp

      # assemble single-end reads 
      plass assemble examples/reads_1.fastq.gz assembly.fas tmp

      # assemble single-end reads using stdin
      cat examples/reads_1.fastq.gz | plass assemble stdin assembly.fas tmp


Important parameters: 

     --min-seq-id         Adjusts the overlap sequence identity threshold
     --min-length         minimum codon length for ORF prediction (default: 40)
     -e                   E-value threshold for overlaps 
     --num-iterations     Number of iterations of assembly
     --filter-proteins    Switches the neural network protein filter off/on

Modules: 

      plass assemble      Assembles proteins (i:Nucleotides -> o:Proteins)
      plass nuclassemble  Assembles nucleotides *experimental* (i:Nucleotides -> o:Nucleotides)
      
### Assemble using MPI 
Plass can be distrubted over several homogeneous computers. However the TMP folder has to be shared between all nodes (e.g. NFS). The following command assembles several nodes:

    RUNNER="mpirun -np 42" plass assemble examples/reads_1.fastq.gz examples/reads_2.fastq.gz assembly.fas tmp


### Compile from source
Compiling PLASS from source has the advantage that it will be optimized to the specific system, which should improve its performance. To compile PLASS `git`, `g++` (4.6 or higher) and `cmake` (3.0 or higher) are required. Afterwards, the PLASS binary will be located in the `build/bin` directory.

      git clone https://github.com/soedinglab/plass.git
      cd plass
      git submodule update --init
      mkdir build && cd build
      cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. ..
      make -j 4 && make install
      export PATH="$(pwd)/bin/:$PATH"
        
:exclamation: If you want to compile PLASS on macOS, please install and use `gcc` from Homebrew. The default macOS `clang` compiler does not support OpenMP and PLASS will not be able to run multithreaded. Use the following cmake call:

      CXX="$(brew --prefix)/bin/g++-8" cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. ..

#### Dependencies

When compiling from source, PLASS requires `zlib` and `bzip`.

### Use the docker image
We also provide a Docker image of Plass. You can mount the current directory containing the reads to be assembled and run plass with the following command:

      docker pull soedinglab/plass
      docker run -ti --rm -v "$(pwd):/app" -w /app plass assemble reads_1.fastq reads_2.fastq assembly.fas tmp

## Hardware requirements
Plass needs roughly 1 byte of memory per residue to work efficiently. Plass will scale its memory consumption based on the available main memory of the machine. Plass needs a CPU with at least the SSE4.1 instruction set to run. 

## Known problems 
* The assembly of Plass includes all ORFs having a start and end codon that includes even very short ORFs < 60 amino acids. Many of these short ORFs are spurious since our neural network cannot distingue them well. We would recommend to use other method to verify the coding potential of these. Assemblies above 100 amino acids are mostly genuine protein sequences. 
* Plass in default searches for ORFs of 40 amino acids or longer. This limits the read length to > 120. To assemble this protein, you need to lower the `--min-length` threshold. Be aware using short reads (< 100 length) might result in lower sensitivity.
