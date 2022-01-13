
## flDpnn file format 
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
