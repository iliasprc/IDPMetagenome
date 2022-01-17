# SEG algortihm
This directory contains C language source code for the SEG program of Wootton
and Federhen, for identifying and masking segments of low compositional
complexity in amino acid sequences.  This program is inappropriate for
masking nucleotide sequences and, in fact, may strip some nucleotide
ambiguity codes from nt. sequences as they are being read.

The SEG program can be used as a plug-in filter of query sequences used in the
NCBI BLAST programs.  See the -filter and -echofilter options described in the
BLAST software's manual page.

Input to SEG must be sequences in FASTA format.  Output can be produced in a
variety of formats, with FASTA format being one of them when the -x option is
used.  The file seg.doc includes a copy of the man page for the seg program.


References:
Wootton, J. C. and S. Federhen (1993).  Statistics of local complexity in amino
acid sequences and sequence databases.  Computers and Chemistry 17:149-163.


MODIFICATION HISTORY
10/18/94
Fixed a bug in the boundary conditions for the alphabet assignments
(colorings) calculations. This condition seems not to arise in the
current protein sequence databases, but does appear when the algorithm
is customized for the nucleic acid alphabet.

4/2/94
Fixed a bug in the reading of input sequence files.  B, Z, and U letters found
in the IUB amino acid alphabet and the NCBI standard amino acid alphabet
were being stripped.

3/30/94
WRG improved speed by about 3X (roughly 5X overall since 3/21/94), due in part
to the elimination of nearly all log() function calls, plus the removal of much
unused or unnecessary code.

3/21/94
Included support for the special characters "*" (translation stop) and "-"
(gap) which are found in some NCBI standard amino acid alphabets.

WRG replaced repetitive dynamic calls to log(2.) and log(20.) with precomputed
values, yielding a 33-50% speed improvement.

WRG added EOF checks in several places, the lack of which could produce
infinite looping.

The previous version of seg is archived beneath the archive subdirectory.

9/30/97
HMF5 plugged a memory leak.

## Usage
```python
./seg ./../../data/path/to/fastafile -x > ./../../results/seg/outputfile
```

### Arguments

```python
Usage: seg <file> <window> <locut> <hicut> <options>
         <file>   - filename containing fasta-formatted sequence(s) 
         <window> - OPTIONAL window size (default 12) 
         <locut>  - OPTIONAL low (trigger) complexity (default 2.2) 
         <hicut>  - OPTIONAL high (extension) complexity (default locut + 0.3) 
         <options> 
            -x  each input sequence is represented by a single output 
                sequence with low-complexity regions replaced by 
                strings of 'x' characters 
            -c <chars> number of sequence characters/line (default 60)
            -m <size> minimum length for a high-complexity segment 
                (default 0).  Shorter segments are merged with adjacent 
                low-complexity segments 
            -l  show only low-complexity segments (fasta format) 
            -h  show only high-complexity segments (fasta format) 
            -a  show all segments (fasta format) 
            -n  do not add complexity information to the header line 
            -o  show overlapping low-complexity segments (default merge) 
            -t <maxtrim> maximum trimming of raw segment (default 100) 
            -p  prettyprint each segmented sequence (tree format) 
            -q  prettyprint each segmented sequence (block format) 
```