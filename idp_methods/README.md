## SEG 

To run seg use the following command

```
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
### flDpnn file format 
```
-------------File Format-------------------------
line 1:  >protein ID (when the pssm matrix cannot be calculated for an input sequence,  [WARNING-DEFAULT PSSM] is added in front of the protein ID which means that the prediction is based on default/low quality PSSM which may lead to lower quality of the disorder predictions.)
line 2:  sequence
line 3:  binary disorder prediction (1 = disordered residue/amino acid; 0 ordered residue)
line 4:  disorder propensity (higher value denotes higher likelihood that a given residue is disordered)
line 5:  binary protein-binding prediction (1 = disordered protein-binding residue; 0 other disordered residue; X ordered residue)
line 6:  protein-binding propensity (higher value denotes higher likelihood that a given residue is disordered and binds proteins; X ordered residue)
line 7:  binary DNA-binding prediction (1 = disordered DNA-binding residue; 0 other disordered residue; X ordered residue)
line 8:  DNA-binding propensity (higher value denotes higher likelihood that a given residue is disordered and binds DNA; X ordered residue)
line 9:  binary RNA-binding prediction (1 = disordered RNA-binding residue; 0 other disordered residue; X ordered residue)
line 10: RNA-binding propensity (higher value denotes higher likelihood that a given residue is disordered and binds RNA; X ordered residue)
line 11: binary linker prediction (1 = disordered linker residue; 0 other disordered residue; X ordered residue)
line 12: linker propensity (higher value denotes higher likelihood that a given residue is the disordered linker; X ordered residue)```
