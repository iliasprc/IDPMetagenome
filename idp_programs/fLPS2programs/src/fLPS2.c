
/*****************************************************************************************/
/****  fLPS2: A program for rapid annotation of compositional biases in biological sequences 
 ****
 ****  Version 2.0 
 ****/ 
/****  Copyright 2017, 2021. Paul Martin Harrison. ****/ 
/****
 ****  Licensed under the 3-clause BSD license. See LICENSE.txt bundled in the fLPS package. 
 ****/ 
/****  
 ****  to compile in ./src: 
 ****   gcc -O2 -march=native -o fLPS2 fLPS2.c -lm 
 ****  OR 
 ****   make 
 ****
 ****  to run and get help: 
 ****   ./fLPS2 -h 
 ****
 ****
 ****  The latest version of this code is available at:
 ****    http://biology.mcgill.ca/faculty/harrison/flps.html 
 ****      OR 
 ****    https://github.com/pmharrison/flps 
 **** 
 ****  Citations: 
 ****    Harrison, PM. 'fLPS: fast discovery of compositional biases for the protein universe', 
 ****    (2017) BMC Bioinformatics, 18: 476. 
 ****    Harrison, PM. 'fLPS 2.0: rapid annotation of compositional biases in 
 ****    biological sequences', (2021) PeerJ, submitted. 
 ****/ 
/*****************************************************************************************/ 

#include "flps_def.h" 
#include "tables.h" 

/** structs for storing subsequences **/
/* array bounds were determined after analyses of TrEMBL database 2017 */ 
struct subsequence { 
int start; 
int end; 
int residue_types_tally; 
} stored[AA_ALPHABET_SIZE][40*DEFAULT_UNIT_STORE]; 

struct second_subsequence { 
int start; 
int end; 
int rescount; 
bool flag;
} second_stored[80*DEFAULT_UNIT_STORE]; 

struct lps { 
int start; 
int end; 
int rescount; 
int restypes[AA_ALPHABET_SIZE]; 
int numrestypes; 
double lp; 
double freq; 
bool flag;
} **tsp, **psp, third_stored[30*DEFAULT_UNIT_STORE], protein_stored[2*DEFAULT_UNIT_STORE]; 

/** sequence analysis **/ 
int alphabet_length, *iseq, *respos, rt, *whole_rescounts, sequence_length=MAXIMUM_LENGTH; 
int core_length=CORE_WINDOW, core_window_offset=CORE_OFFSET, number_of_lps;
int further_back_number_of_lps, previous_number_of_lps, number_first_stored, number_third_stored; 
int number_protein_stored, number_kept, *residue_regions_tally, number_of_sequences, number_single, number_multiple; 
char sn[MAX_SEQUENCE_NAME_SIZE+1], line[MAXIMUM_LENGTH+1], sequence_name[MAX_SEQUENCE_NAME_SIZE+1] = {"initial"};
char alphabet[AA_ALPHABET_SIZE+1] = {"ACDEFGHIKLMNPQRSTVWXY"}, nt_alphabet[NT_ALPHABET_SIZE] = {"ACGNT"}; 
char *sequence, *cleanseq, *modseq, current_info[100*DEFAULT_UNIT_STORE]; 
bool *binseq, single_or_multiple_or_whole, too_long; 

/** options/parameters **/ 
char infile[MAX_FILE_NAME_SIZE+1], outputs[4][10] = {"SINGLE", "MULTIPLE", "WHOLE"}, test_name[MAXIMUM_LENGTH+1];
char masker[3] = {"XN"}, dfo[2][15] = {"EXCISED_DOMAIN", "MASKED_DOMAIN"}, token[3], domain_name[MAXIMUM_DOMAINS][MAX_DOM_NAME_SIZE+1]; 
int minimum_window=FIXED_MIN_WINDOW, maximum_window=FIXED_MAX_WINDOW, domain_list, divisor=1; 
int number_of_domains, domain_start[MAXIMUM_DOMAINS], domain_end[MAXIMUM_DOMAINS]; 
int stepsize=DEFAULT_STEPSIZE, unit_store=DEFAULT_UNIT_STORE; 
double pvalue_threshold=DEFAULT_BASELINE_LOG_P_VALUE, baseline_log_p_value=DEFAULT_BASELINE_LOG_P_VALUE; 
double baseline_p_value=DEFAULT_BASELINE_P_VALUE, version=VERSION_NUMBER; 
bool headfoot, single_only, verbose, eop, test, trimmed_off; 
enum molecule {AA, NT} sequence_type;  
enum composn {DOMAINS, EQUAL, USER} bkgd; 
enum results_format {SHORT_FORMAT, LONG_FORMAT, MASKED, ONELINE} format; 
enum domains_format {EXCISED_DOMAIN, MASKED_DOMAIN} df; 
enum calculn {SINGLE, MULTIPLE, WHOLE} op; 
enum precision {FAST, MEDIUM, THOROUGH} z; 
char AALabel[17][18] = {"amide", "glx", "asx", "tiny_polar", "tiny_hydrophobic", "polar_aromatic", "negative", "small_polar", 
     "positive", "small_hydrophobic", "aliphatic", "aromatic", "charged", "tiny", "small", "polar", "hydrophobic"}; 

FILE *f3, *f4; 


void print_help() 
{ fprintf(stdout, "\nfLPS version %.1lf\n", version);   
  fprintf(stdout,   "================\n" 
" -h   prints help\n\n"
" -v   prints verbose runtime info\n\n"
" -d   prints header and footer information in the output files\n\n"    
" -s   calculate single-residue biases ONLY\n\n"     
" -n   specifies DNA sequence(s) (the default is AMINO ACID sequences)\n\n"             
" -o   format specified as: short (default), long, oneline or masked\n\n" 
" -m   minimum tract length (default=15), must be <= maximum and in range 5-1000\n\n"             
" -M   maximum tract length (default=500), must be >= minimum and in range 15-1000\n"
"      and > minimum tract length\n\n"              
" -t   binomial p-value threshold, must be <=0.001 (can be either decimal or exponential);\n"
"      can be <=0.01 if '-z medium' is specified, and <=0.1 of '-z thorough'\n\n"                  
" -c   if 'equal', it specifies equal amino-acid composition\n" 
"      if 'domains', it specifies domains amino-acid composition\n" 
"      else the option argument specifies the name of a user amino-acid composition file\n" 
"      (default is 'domains')\n\n" 
" -r   inputs restriction list of amino acids for bias calculation\n\n" 
" -k   does not consider un-known residues (Xs for amino acids and Ns for nucleotides). (These are\n"
"      still included in calculations of total residue amounts.)\n\n" 
" -D   indicates to read list of filtered domains from the name lines of the FASTA-format sequence input file;\n"  
"      this is only allowed for output format options '-o short' and '-o long';\n"
"      the domains should be in numeric order and not overlap.\n"
"      The format specification for the domain list is in the 'README.first'\n"
"      If '-D excised' is specified, then the domains have been excised from the sequences;\n"
"      if '-D masked' is specified, then the domains have been masked in the sequences.\n\n"
" -O   ...FILENAME_PREFIX... allows for output to be sent to a unique file prefixed with FILENAME_PREFIX.\n"
"      The parameters that are specified are included in the generated filename.\n\n"  
" -z   specifies the baseline precision of the bias calculation; there are three possibilities:\n"
"     'fast' (the default, which specifies a baseline P-value threshold of 0.001, and a windowing step size of 3)\n"
"     'medium' (the default, which specifies a baseline P-value threshold of 0.01, and a windowing step size of 2)\n"
"     'thorough' (the default, which specifies a baseline P-value threshold of 0.1, and a windowing step size of 1)\n\n"
" RECOMMENDED PARAMETERS:\n"
" -----------------------\n"
" IT IS RECOMMENDED TO USE THE DEFAULT VALUES, TO FIND MORE EXTENDED BIASED REGIONS MADE FROM A\n" 
" COMBINATION OF MILD BIASES FOR SEVERAL RESIDUES, AS WELL AS SHORTER 'LOW-COMPLEXITY' TRACTS.\n"
" THESE VALUES WILL, HOWEVER, OFTEN MERGE SHORT CONSECUTIVE LOW-COMPLEXITY REGIONS INTO ONE LONG BIASED REGION.\n\n" 
" TO FOCUS ON SHORT LOW-COMPLEXITY REGION ANNOTATION OR MASKING, THE FOLLOWING OPTIONS ARE APPROPRIATE:\n" 
"     ./fLPS2 -t1e-5 -m5 -M25 ...input_file... > ...output_file... \n\n"  
" FURTHER DETAILS CAN BE FOUND IN THE README FILE DISTRIBUTED WITH THE PROGRAM.\n\n"
"\nFurther Examples:\n" 
  "-----------------\n\n" 
"     ./fLPS2 -vm 20 -M 200 -c user_composition ...input_file... \n"
" fLPS run with verbose runtime info (-v), minimum tract length 20, maximum tract length 200\n"
" and user_composition file specified\n\n" 
"     ./fLPS2 -o long -c equal -rQNFY -t 0.000001 ...input_file... \n" 
" fLPS run with long results output, equal compositional background (each residue set =0.05), and pvalue threshold\n" 
" =0.000001. Here, bias calculations are restricted to the amino acids QNFY.\n\n" 
"     ./fLPS2 -od oneline -z medium -s  ...input_file... \n" 
" Here, '-o oneline' specifies a single-line summary output. The baseline precision of the calculation is set to\n"
" medium. Headers and footers are output (-d). Only single-residue biases are considered (-s).\n"
" Very long sequences are curtailed to 100,000 residues length.\n\n" 
"CITATIONS:\n Harrison, PM. 'fLPS: fast discovery of compositional biases for the protein universe',\n"
" (2017) BMC Bioinformatics, 18: 476. \n" 
" Harrison, PM. 'fLPS 2.0: rapid annotation of compositional biases in biological sequences', submitted.\n"
"URLs:\n http://biology.mcgill.ca/faculty/harrison/flps.html\n OR \n https://github.com/pmharrison/flps\n"); 
} /* end of print_help */ 


void print_composition_file_format()
{
int i; 
double example_aa_composition[AA_ALPHABET_SIZE] = { 
   0.0784, 0.0160, 0.0558, 0.0650, 0.0391, 0.0661, 0.0218,
   0.0493, 0.0618, 0.0920, 0.0218, 0.0411, 0.0467, 0.0433,
   0.0479, 0.0721, 0.0620, 0.0649, 0.0149, 0.0015, 0.0377
   }; 
double example_nt_composition[NT_ALPHABET_SIZE] = { 0.289, 0.210, 0.210, 0.289, 0.002 }; 

fprintf(stdout, "An example of composition file format for amino-acid sequences is shown below.\n" 
                "The file should contain all 21 single-letter amino-acid codes (including X).\n\n"); 
for(i=0;i<AA_ALPHABET_SIZE;i++) { fprintf(stdout, "%c\t%lf\n", alphabet[i], example_aa_composition[i]);} 

fprintf(stdout, "An example of composition file format for nucleotide sequences is shown below:\n" 
                "The file should contain all 5 single-letter nucleotide codes (including N).\n\n"); 
for(i=0;i<NT_ALPHABET_SIZE;i++) { fprintf(stdout, "%c\t%lf\n", nt_alphabet[i], example_nt_composition[i]);} 
} /* end of print_composition_file_format() */ 


double logbinomial(int n, int k, double expect)
{ return logfactorial[n] - logfactorial[n-k] - logfactorial[k] + ((double)k)*log10(expect) + ((double)(n-k))*log10(1.0-expect); } 


void threshold_table(double freq[], int alphabet_length, int table[][alphabet_length])
{
int i,j,k; 
double current_p_value = -9999.0, previous_p_value = -10000.0; 

for(j=alphabet_length-1;j>=0;j--)
   { for(i=MAXIMUMWINDOW;i!=ABSOLUTEMIN-1;i--) 
        { current_p_value = -9999.0; previous_p_value = -10000.0;
          for(k=1;k<=i;k++)
             {
             current_p_value = logbinomial(i,k,freq[j]); 
             if(current_p_value<baseline_log_p_value && current_p_value<previous_p_value) { break; }
             previous_p_value = current_p_value; 
             } /* end for k */ 
          table[i-1][j]=k;  
        } /* end for i */ 
   } /* end for j */ 
} /* end of threshold_table */ 


void convert_aa_sequence()
{
int a,aa,*pis; 
char *ps, *pcs; 

pis=iseq; ps=sequence; pcs=cleanseq; 
memset(whole_rescounts, 0, alphabet_length * sizeof(int)); 

for(a=0; ps<sequence+sequence_length; ps++)
    { aa=toupper(*ps); 
      switch(aa) {
          case 65 /*'A'*/: *(pis+a) = 0; *(pcs+a) = aa; whole_rescounts[0]++; a++; break; 
          case 67 /*'C'*/: *(pis+a) = 1; *(pcs+a) = aa; whole_rescounts[1]++; a++; break;
          case 68 /*'D'*/: *(pis+a) = 2; *(pcs+a) = aa; whole_rescounts[2]++; a++; break; 
          case 69 /*'E'*/: *(pis+a) = 3; *(pcs+a) = aa; whole_rescounts[3]++; a++; break;
          case 70 /*'F'*/: *(pis+a) = 4; *(pcs+a) = aa; whole_rescounts[4]++; a++; break; 
          case 71 /*'G'*/: *(pis+a) = 5; *(pcs+a) = aa; whole_rescounts[5]++; a++; break;
          case 72 /*'H'*/: *(pis+a) = 6; *(pcs+a) = aa; whole_rescounts[6]++; a++; break; 
          case 73 /*'I'*/: *(pis+a) = 7; *(pcs+a) = aa; whole_rescounts[7]++; a++; break;
          case 75 /*'K'*/: *(pis+a) = 8; *(pcs+a) = aa; whole_rescounts[8]++; a++; break; 
          case 76 /*'L'*/: *(pis+a) = 9; *(pcs+a) = aa; whole_rescounts[9]++; a++; break;
          case 77 /*'M'*/: *(pis+a) = 10; *(pcs+a) = aa; whole_rescounts[10]++; a++; break; 
          case 78 /*'N'*/: *(pis+a) = 11; *(pcs+a) = aa; a++; whole_rescounts[11]++; break;
          case 80 /*'P'*/: *(pis+a) = 12; *(pcs+a) = aa; a++; whole_rescounts[12]++; break; 
          case 81 /*'Q'*/: *(pis+a) = 13; *(pcs+a) = aa; a++; whole_rescounts[13]++; break;
          case 82 /*'R'*/: *(pis+a) = 14; *(pcs+a) = aa; a++; whole_rescounts[14]++; break; 
          case 83 /*'S'*/: *(pis+a) = 15; *(pcs+a) = aa; a++; whole_rescounts[15]++; break;
          case 84 /*'T'*/: *(pis+a) = 16; *(pcs+a) = aa; a++; whole_rescounts[16]++; break; 
          case 86 /*'V'*/: *(pis+a) = 17; *(pcs+a) = aa; a++; whole_rescounts[17]++; break;
          case 87 /*'W'*/: *(pis+a) = 18; *(pcs+a) = aa; a++; whole_rescounts[18]++; break; 
          case 89 /*'Y'*/: *(pis+a) = 20; *(pcs+a) = aa; a++; whole_rescounts[20]++; break;
          case 88 /*'X'*/: *(pis+a) = 19; *(pcs+a) = aa; a++; whole_rescounts[19]++; break; } } 
sequence_length=a; 
} /* end of convert_aa_sequence() */ 


void convert_dna_sequence()
{
int a,nt,*pis; 
char *ps, *pcs; 

pis=iseq; ps=sequence; pcs=cleanseq; 
memset(whole_rescounts, 0, alphabet_length * sizeof(int)); 

for(a=0; ps<sequence+sequence_length; ps++)
    { nt=toupper(*ps); 
      switch(nt) {
          case 65 /*'A'*/: *(pis+a) = 0; *(pcs+a) = nt; whole_rescounts[0]++; a++; break; 
          case 67 /*'C'*/: *(pis+a) = 1; *(pcs+a) = nt; whole_rescounts[1]++; a++; break;
          case 71 /*'G'*/: *(pis+a) = 2; *(pcs+a) = nt; whole_rescounts[2]++; a++; break;
          case 78 /*'N'*/: *(pis+a) = 3; *(pcs+a) = nt; a++; whole_rescounts[3]++; break;
          case 84 /*'T'*/: *(pis+a) = 4; *(pcs+a) = nt; a++; whole_rescounts[4]++; break; } }     
sequence_length=a; 
} /* end of convert_aa_sequence() */ 


char * make_label(char *s, char *label) 
{
int i, aa, labelled, signature_size, count[NUMBER_OF_AA_CLASSES]; 
char *psig; 

psig=s; 

if(sequence_type==NT) {
if(!strcmp(s,"A"))        { strcpy(label,"{A}|{T}");}
else if(!strcmp(s,"T"))   { strcpy(label,"{A}|{T}");}
else if(!strcmp(s,"G"))   { strcpy(label,"{G}|{C}");}
else if(!strcmp(s,"C"))   { strcpy(label,"{G}|{C}");}
else if(!strcmp(s,"AT"))  { strcpy(label,"{AT}");}
else if(!strcmp(s,"TA"))  { strcpy(label,"{AT}");}
else if(!strcmp(s,"AC"))  { strcpy(label,"{AC}|{GT}");}
else if(!strcmp(s,"CA"))  { strcpy(label,"{AC}|{GT}");}
else if(!strcmp(s,"AG"))  { strcpy(label,"{AG}|{CT}");}
else if(!strcmp(s,"GA"))  { strcpy(label,"{AG}|{CT}");}
else if(!strcmp(s,"TC"))  { strcpy(label,"{AG}|{CT}");}
else if(!strcmp(s,"CT"))  { strcpy(label,"{AG}|{CT}");}
else if(!strcmp(s,"TG"))  { strcpy(label,"{AC}|{GT}");}
else if(!strcmp(s,"GT"))  { strcpy(label,"{AC}|{GT}");}
else if(!strcmp(s,"GC"))  { strcpy(label,"{GC}");}
else if(!strcmp(s,"CG"))  { strcpy(label,"{GC}");}
else if(!strcmp(s,"CAT")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"CTA")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"TAC")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"TCA")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"ATC")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"ACT")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"GAT")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"GTA")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"TAG")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"TGA")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"ATG")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"AGT")) { strcpy(label,"{ATC}|{ATG}");}
else if(!strcmp(s,"GAC")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"GCA")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"CAG")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"CGA")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"AGC")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"ACG")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"GTC")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"GCT")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"CTG")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"CGT")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"TGC")) { strcpy(label,"{ACG}|{CGT}");}
else if(!strcmp(s,"TCG")) { strcpy(label,"{ACG}|{CGT}");} 
else { strcpy(label,"none"); } } /*end of if NT*/ 

else { /*sequence_type==AA*/ 
memset(count, 0, NUMBER_OF_AA_CLASSES * sizeof(int));  
signature_size = (int) strlen(s); 
if(signature_size==1) {strcpy(label,"----");}
else { /*MULTIPLE*/ 
for(;psig<s+signature_size;psig++)
   { aa = *psig; 
    switch(aa) {
          case 65 /*'A'*/: count[SMALL]++; count[TINY]++; count[HYDROPHOBIC]++; count[TINY_HYDROPHOBIC]++; 
                           count[SMALL_HYDROPHOBIC]++; break; 
          case 67 /*'C'*/: count[SMALL]++; count[TINY]++; count[HYDROPHOBIC]++; count[POLAR]++; count[TINY_POLAR]++; 
                           count[TINY_HYDROPHOBIC]++; count[SMALL_POLAR]++; count[SMALL_HYDROPHOBIC]++; break; 
          case 68 /*'D'*/: count[SMALL]++; count[NEGATIVE]++; count[CHARGED]++; count[POLAR]++; count[ASX]++; 
                           count[SMALL_POLAR]++; break; 
          case 69 /*'E'*/: count[NEGATIVE]++; count[CHARGED]++; count[POLAR]++; count[GLX]++; break; 
          case 70 /*'F'*/: count[AROMATIC]++; count[HYDROPHOBIC]++; break; 
          case 71 /*'G'*/: count[SMALL]++; count[TINY]++; count[HYDROPHOBIC]++; count[TINY_HYDROPHOBIC]++; 
                           count[SMALL_HYDROPHOBIC]++; break;
          case 72 /*'H'*/: count[AROMATIC]++; count[CHARGED]++; count[POSITIVE]++; count[POLAR]++; 
                           count[HYDROPHOBIC]++; count[POLAR_AROMATIC]++; break; 
          case 73 /*'I'*/: count[ALIPHATIC]++; count[HYDROPHOBIC]++; break;
          case 75 /*'K'*/: count[POSITIVE]++; count[CHARGED]++; count[HYDROPHOBIC]++; count[POLAR]++; break; 
          case 76 /*'L'*/: count[ALIPHATIC]++; count[HYDROPHOBIC]++; break;
          case 77 /*'M'*/: count[HYDROPHOBIC]++; break; 
          case 78 /*'N'*/: count[SMALL]++; count[POLAR]++; count[AMIDE]++; count[ASX]++; count[SMALL_POLAR]++; break;
          case 80 /*'P'*/: count[SMALL]++; break; 
          case 81 /*'Q'*/: count[POLAR]++; count[AMIDE]++; count[GLX]++; break;
          case 82 /*'R'*/: count[POSITIVE]++; count[CHARGED]++; count[POLAR]++; break; 
          case 83 /*'S'*/: count[SMALL]++; count[TINY]++; count[POLAR]++; count[SMALL_POLAR]++; count[TINY_POLAR]++; break;
          case 84 /*'T'*/: count[SMALL]++; count[POLAR]++; count[HYDROPHOBIC]++; count[SMALL_POLAR]++; 
                           count[SMALL_HYDROPHOBIC]++; break; 
          case 86 /*'V'*/: count[ALIPHATIC]++; count[HYDROPHOBIC]++; count[SMALL]++; count[SMALL_HYDROPHOBIC]++; break;
          case 87 /*'W'*/: count[HYDROPHOBIC]++; count[AROMATIC]++; count[POLAR]++; count[POLAR_AROMATIC]++; break; 
          case 89 /*'Y'*/: count[HYDROPHOBIC]++; count[AROMATIC]++; count[POLAR]++; count[POLAR_AROMATIC]++; break;
               } /*end of switch p*/ 
   } /*end for i*/ 
/* assign optimal AA set to the signature ==> a shorter class list takes precedence over a longer */ 
for(i=0,labelled=0;i<NUMBER_OF_AA_CLASSES;i++) { if(count[i]==signature_size) { strcpy(label, AALabel[i]); labelled=1; break; } } 
if(!labelled) {strcpy(label,"mixed");}
} /*end of else MULTIPLE*/ 
} /*end of sequence_type==AA*/ 

return label; 
} /*end of make_label*/ 


void merge(double freq[])
{
int i,j,k,l,m,n,p,q,Nterm,Cterm,found,N,C, add_restypes[AA_ALPHABET_SIZE], prev_bias_residue; 
int best_start,best_end,best_rescount,addl_bias_residues,test_elimit,test_N,test_C; 
bool there_already, new_bias_residue, adjust_N, adjust_C; 
double temp_composition,current_lpvalue,first_lpvalue,best_lpvalue,moveN_lpvalue,moveC_lpvalue,moveboth_lpvalue; 

for(i=0;i<number_protein_stored;i++)
   {
   if(!psp[i]->flag) 
     {
     for(j=i+1;j<number_protein_stored;j++) 
        {
        if(!psp[j]->flag && !(psp[j]->start <psp[i]->start && psp[j]->end <psp[i]->start) 
        && !(psp[i]->start <psp[j]->start && psp[i]->end <psp[j]->start) 
          ) /* ?? OVERLAP */ 
          {
            /* set termini */ 
            if(psp[i]->start<psp[j]->start) { Nterm = psp[i]->start; }
            else { Nterm = psp[j]->start; } 
            if(psp[i]->end>psp[j]->end) { Cterm = psp[i]->end; }
            else { Cterm = psp[j]->end; }             
      
            /* make an array of residue positions */ 
            for(l=Nterm,m=0;l<=Cterm;l++)
               {
               for(k=0,found=0;k<psp[i]->numrestypes;k++)
                  { if(iseq[l]==psp[i]->restypes[k]) 
                      { respos[m]=l; m++; found=1; break; } 
                  } /* end for k */ 
               if(!found)
               	 {
               	 for(k=0;k<psp[j]->numrestypes;k++)
                    { if(iseq[l]==psp[j]->restypes[k]) 
                        { respos[m]=l; m++; break; } 
                    } /* end for k */ 
                 } /* end of if !found */ 
               }/* end for l */

            /* calculate temp_composition and add_restypes */ 
            temp_composition=0.0; 
               for(k=0;k<psp[i]->numrestypes;k++)
                    { temp_composition += freq[psp[i]->restypes[k]]; } 
               for(k=0,p=0;k<psp[j]->numrestypes;k++)
                    { 
                    there_already=0; 
                    for(l=0;l<psp[i]->numrestypes;l++)
                       { if(psp[j]->restypes[k]==psp[i]->restypes[l])
                           { there_already=1; break; } 
                       } /* end for l */ 
                    if(!there_already)
                      { temp_composition += freq[psp[j]->restypes[k]];  
                        add_restypes[p] = psp[j]->restypes[k]; 
                        p++; }
                    } /* end for k */ 

            /* calculate p-value for total merged region
               and store as best */ 
            best_lpvalue = logbinomial(Cterm-Nterm+1, m, temp_composition); 
            best_start = Nterm; 
            best_end = Cterm; 
            best_rescount = m; 

            /* TRIM */
            N=1; C=m-2; /* <-- first increment/decrement is outside of loop */ 

            moveN_lpvalue=moveC_lpvalue=moveboth_lpvalue=0.0; 
            while(respos[C]-respos[N]+1 >= FIXED_MIN_WINDOW)
            	 {
                 moveN_lpvalue = logbinomial(respos[C+1]-respos[N]+1, C-N+2, temp_composition); 
                 moveC_lpvalue = logbinomial(respos[C]-respos[N-1]+1, C-N+2, temp_composition);
                 moveboth_lpvalue = logbinomial(respos[C]-respos[N]+1, C-N+1, temp_composition); 

                 if(moveboth_lpvalue<moveN_lpvalue && moveboth_lpvalue<moveC_lpvalue)
                   { if(moveboth_lpvalue<best_lpvalue)
                       { best_lpvalue = moveboth_lpvalue; 
                         best_start = respos[N]; best_end = respos[C]; best_rescount = C-N+1; 
                         } } 
                 else if(moveN_lpvalue<moveboth_lpvalue && moveN_lpvalue<moveC_lpvalue)
                      { if(moveN_lpvalue<best_lpvalue)
                        { best_lpvalue = moveN_lpvalue; 
                          best_start = respos[N]; best_end = respos[C+1]; best_rescount = C-N+2; 
                          } } 
                 else { if(moveC_lpvalue<best_lpvalue)
                        { best_lpvalue = moveC_lpvalue; 
                          best_start = respos[N-1]; best_end = respos[C]; best_rescount = C-N+2; 
                        } } 

                 N++; C--; 
                 }/* end of while moving N and C */ 

            if(best_lpvalue<psp[i]->lp)
              {
              psp[j]->flag=1; /* de-select j */ 

              test_elimit = ((int) (1.0/temp_composition)) +1; 
              addl_bias_residues=0; 

              /* check for extensions */  
                test_N=Nterm; prev_bias_residue=test_N; 
                while(test_N>=1)
                     {
                     test_N--; 
                     if(prev_bias_residue-test_N>test_elimit) { break; } 
                     new_bias_residue=0; 
                     for(k=0;k<psp[i]->numrestypes;k++)
                        { if(iseq[test_N]==psp[i]->restypes[k])
                            { addl_bias_residues++; prev_bias_residue=test_N; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { if(iseq[test_N]==add_restypes[k])
                            { addl_bias_residues++; prev_bias_residue=test_N; new_bias_residue=1; break; }
                        } /* end for k */ 
                     if(new_bias_residue==1)
                       {
                       /* calc current_lpvalue */ 
                       current_lpvalue = logbinomial(Cterm-test_N+1, m+addl_bias_residues, temp_composition);

                       if(current_lpvalue<best_lpvalue)
                         { /* update best data */ 
                         best_lpvalue = current_lpvalue; 
                         best_start = test_N; 
                         best_end = Cterm; 
                         best_rescount= m+addl_bias_residues;
                         } /* end of if current_lpvalue<best_lpvalue */ 
                       else { addl_bias_residues--; break; } /* break out of while */ 
                       } /* end of if bias_residue==1 */ 
                     } /* end of while test_N>=0 */ 
              
                test_C=Cterm; prev_bias_residue=test_C;
                while(test_C<(sequence_length-1))
                     {
                     test_C++; 
                     if(test_C-prev_bias_residue>test_elimit) { break; } 
                     new_bias_residue=0; 
                     for(k=0;k<psp[i]->numrestypes;k++)
                        { if(iseq[test_C]==psp[i]->restypes[k])
                            { addl_bias_residues++; prev_bias_residue=test_C; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { if(iseq[test_C]==add_restypes[k])
                            { addl_bias_residues++; prev_bias_residue=test_C; new_bias_residue=1; break; }
                        } /* end for k */ 
                     if(new_bias_residue==1)
                       {
                       if(best_start>Nterm)
                       	 { /* calc current_lpvalue */ 
                       current_lpvalue = logbinomial(test_C-Nterm+1, m+addl_bias_residues, temp_composition);

                       if(current_lpvalue<best_lpvalue)
                         { /* update best data */ 
                         best_lpvalue = current_lpvalue; 
                         best_start = Nterm; 
                         best_end = test_C; 
                         best_rescount= m+addl_bias_residues; 
                         } /* end of if current_lpvalue<best_lpvalue */ 
                       else { break; } /* break out of while */ 
                         } /* end of if best_end>Cterm */ 

                       else { 
                       current_lpvalue = logbinomial(test_C-best_start+1, m+addl_bias_residues, temp_composition);

                       if(current_lpvalue<best_lpvalue)
                         { /* update best data */ 
                         best_lpvalue = current_lpvalue; 
                         best_end = test_C; 
                         best_rescount= m+addl_bias_residues; 
                         } /* end of if current_lpvalue<best_lpvalue */ 
                       else { break; } /* break out of while */ 
                            } /* end of else */ 
                       } /* end of if bias_residue==1 */ 
                     } /* end of while test_C<sequence_length */ 

              /* last end check */ 
               test_N=best_start; addl_bias_residues=0; adjust_N=0; 
                while(test_N>=1)
                     {
                     test_N--; 
                     new_bias_residue=0; 
                     for(k=0;k<psp[i]->numrestypes;k++)
                        { if(iseq[test_N]==psp[i]->restypes[k])
                            { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { if(iseq[test_N]==add_restypes[k])
                            { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     if(!new_bias_residue) { test_N++; break; }
                     } /* end of while test_N>=1 */ 
                     if(addl_bias_residues>0)
                       { best_start = test_N; best_rescount+=addl_bias_residues; adjust_N=1; }
             
                test_C=best_end; addl_bias_residues=0; adjust_C=0; 
                while(test_C<(sequence_length-1))
                     {
                     test_C++; 
                     new_bias_residue=0; 
                     for(k=0;k<psp[i]->numrestypes;k++)
                        { if(iseq[test_C]==psp[i]->restypes[k])
                            { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { if(iseq[test_C]==add_restypes[k])
                            { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     if(!new_bias_residue) { test_C--; break; } 
                     } /* end of while test_C<sequence_length */ 
                     if(addl_bias_residues>0)
                       { best_end = test_C; best_rescount+=addl_bias_residues; adjust_C=1; } 
                if(adjust_N||adjust_C)
                  { best_lpvalue = logbinomial(best_end-best_start+1, best_rescount, temp_composition); } 

              psp[i]->lp = best_lpvalue; 
              psp[i]->start = best_start; 
              psp[i]->end = best_end;               
              memcpy((psp[i]->restypes)+psp[i]->numrestypes, add_restypes, p*(sizeof(int))); 
              psp[i]->numrestypes += p; 
              psp[i]->freq += psp[j]->freq;
              psp[i]->rescount = best_rescount; 
              } /* end of if best_lpvalue<psp[i]->lp */ 
          } /* end of if !flag j  && OVERLAP */ 
        } /* end for j */ 
     } /* end of if !flag i  */ 
   } /* end for i */ 
} /* end of merge () */ 
 

int compare(const void * a, const void * b) /* for qsort */ 
{
struct lps *lpsA = (struct lps *)a; 
struct lps *lpsB = (struct lps *)b; 
return ( (int) (100000.0 * (lpsA->lp - lpsB->lp)) ); 
}/* end of compare() */ 


int whole(double freq[], char *alph, enum calculn c)
{
int i,a;
double lp,f;  

for(i=0,a=0;i<alphabet_length;i++)
   { lp = logbinomial(sequence_length, whole_rescounts[i], freq[i]); 
     f = ((double) whole_rescounts[i]) / ((double) sequence_length); 
     if(lp<pvalue_threshold && f>=freq[i])
       { protein_stored[a].lp = lp; 
         protein_stored[a].restypes[0] = i; 
         protein_stored[a].rescount = whole_rescounts[i]; 
         a++; } /* end of if lp < baseline_log_p_value */ 
   } /* end of for i */ 

if(a>0)
  { if(a>1) { qsort(protein_stored, a, sizeof(struct lps), compare); 
    for(i=0;i<a;i++)
       { fprintf(f4, "%s\t%d\t%s\t%d\t1\t%d\t%d\t%.3e\t{%c}\n", sequence_name, sequence_length, outputs[c], i+1, sequence_length, 
         protein_stored[i].rescount, pow(10.0,protein_stored[i].lp), alph[protein_stored[i].restypes[0]]); } 
            } /* end of if a>1 */ 
    else { fprintf(f4, "%s\t%d\t%s\t%d\t1\t%d\t%d\t%.3e\t{%c}\n", sequence_name, sequence_length, outputs[c], 1, sequence_length, 
           protein_stored[0].rescount, pow(10.0,protein_stored[0].lp), alph[protein_stored[0].restypes[0]]); }
  }/* end of if a>0 */ 

return a; 
} /* end of whole () */ 


void trim_minimum_LPS(int i)
{ while(iseq[protein_stored[i].start] != protein_stored[i].restypes[0]) { protein_stored[i].start++; trimmed_off++; } 
  while(iseq[protein_stored[i].end]   != protein_stored[i].restypes[0]) { protein_stored[i].end--;   trimmed_off++; } } 


void output(enum calculn c, double freq[], char *alph) 
{
int i,j,k,l,a,b,len,output_number_whole=0, *pis;
int core_start=-1,N_offset,C_offset,centre,best_total,best_core_start,centre_dist,current_centre_dist,current_total; 
char *psn=sequence_name, *pcs, *pss, info_append[100], signature[AA_ALPHABET_SIZE], label[18]; 
bool *pbs; 
double p,lp,f,enr; 

pbs=binseq; pcs=cleanseq; pis=iseq; pss=sequence; 

if(format==MASKED)  
{
if(c==SINGLE) { sprintf(current_info,"%s\t%d\t", psn, sequence_length); }

for(i=0,a=0;i<number_protein_stored;i++) 
   {
   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { /* make name line cumulatively */   
   for(j=0,b=0;j<psp[i]->numrestypes;j++) { signature[b]=alph[psp[i]->restypes[j]]; b++; }; signature[psp[i]->numrestypes]='\0'; 
   sprintf(info_append, "%d-%d,%d,{%s},%.1e\t", (psp[i]->start)+1, (psp[i]->end)+1, psp[i]->rescount, signature, pow(10.0, psp[i]->lp)); 
   strcat(current_info, info_append); a++; 

     /* mask cleanseq cumulatively */ 
     for(j=psp[i]->start;j<=psp[i]->end;j++)
        { if(*(pcs+j) != masker[sequence_type] /*X or N*/)
            { for(k=0;k<psp[i]->numrestypes;k++)
                 { if(psp[i]->restypes[k] == *(pis+j)) { *(pcs+j)=masker[sequence_type]; break; } } 
            } /* end of if cleanseq[j]!=X or N */ 
        } /* end of for j */ 
      } /* end of if p<=pvalue_threshold */ 
   } /* end for i*/ 

if(single_only||c==MULTIPLE)
{ fprintf(f4, ">%s\n", current_info); 
  for(i=0;i<=sequence_length-OUTPUT_LINE_LENGTH;i+=OUTPUT_LINE_LENGTH)
     { fprintf(f4,"%.*s\n", OUTPUT_LINE_LENGTH, pcs+i); 
     } fprintf(f4,"%.*s\n", sequence_length-i, pcs+i); 
} /* end of if single_only || c==MULTIPLE */ 

if(c==SINGLE) { number_single=a; } else { number_multiple=a; }
} /* end of if format==MASKED */ 

else if(format==ONELINE)
{
if(c==SINGLE) { sprintf(current_info,"%s\t%d\t", psn, sequence_length); }

for(i=0,a=0;i<number_protein_stored;i++) 
   {
   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { /* make oneline cumulatively */ 
   for(j=0,b=0;j<psp[i]->numrestypes;j++) { signature[b]=alph[psp[i]->restypes[j]]; b++; }; signature[psp[i]->numrestypes]='\0'; 
   sprintf(info_append, "%d-%d,%d,{%s},%.1e\t", (psp[i]->start)+1, (psp[i]->end)+1, psp[i]->rescount, signature, pow(10.0, psp[i]->lp)); 
   strcat(current_info, info_append); a++; 
     } /* end of if p<=pvalue_threshold */ 
   } /* end for i*/ 

if((single_only||c==MULTIPLE) && number_single+a>0) { fprintf(f4, "%s\n", current_info); } 

if(c==SINGLE) { number_single=a; } else { number_multiple=a; }
}/* end of if format==ONELINE */ 

else if(format==SHORT_FORMAT)
{
if(c==WHOLE) { output_number_whole=whole(freq, alph, c); } 

else { /* not whole */ 
for(i=0,a=0;i<number_protein_stored;i++) 
   {
   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { 
     len = psp[i]->end - psp[i]->start + 1; 
     enr = ((double)(psp[i]->rescount)/(double)len) / psp[i]->freq; /* enrichment */ 
     fprintf(f4, "%s\t%d\t%s\t%d\t%d\t%d\t%d\t%.3e\t{", 
      psn, sequence_length, outputs[c], a+1, (psp[i]->start)+1, (psp[i]->end)+1, psp[i]->rescount, pow(10.0, psp[i]->lp) ); 
     for(j=0;j<psp[i]->numrestypes;j++) { signature[j]=alph[psp[i]->restypes[j]]; } ; signature[psp[i]->numrestypes]='\0'; 
     fprintf(f4,"%s}\t%s\t%.3lf\n", signature, make_label(signature, label), enr); 
     a++; 
     } /* end of if p<=pvalue_threshold */ 
   } /* end for i*/ 

if(c==SINGLE) { number_single=a; } else { number_multiple=a; } 
     } /* end of else not whole */ 
} /* end of if format==SHORT_FORMAT */ 

else { /* LONG_FORMAT */ 
if(c==WHOLE) { output_number_whole=whole(freq, alph, c); } 

else { /* not whole */ 
for(i=0,a=0;i<number_protein_stored;i++) 
   { 
   len = psp[i]->end - psp[i]->start + 1; 

   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { 
     core_length=minimum_window; 
     if(len>core_length)
     {
     /* fill binseq for core calculation */ 
       for(j=psp[i]->start;j<=psp[i]->end;j++)
          { 
          for(k=0,*(pbs+j)=0;k<psp[i]->numrestypes;k++)
             { if(psp[i]->restypes[k]== *(pis+j)) { *(pbs+j)=1; break; } } 
          }/* end of for j */ 

     /** calculate core **/ 
     centre = (int) (len/2); 

     /* deal with first window */ 
     centre_dist = abs(centre - psp[i]->start - CORE_OFFSET); 
     best_core_start=psp[i]->start; 
     for(j=psp[i]->start,best_total=0,centre_dist=0; j<(psp[i]->start)+core_length; j++)
        { best_total += *(pbs+j); }

     /* check subsequent windows */ 
     current_centre_dist = centre_dist; 
     for(j=(psp[i]->start)+1,current_total=best_total; j<=(psp[i]->end)-core_length+1; j++)
        {
        current_total += *(pbs+j+core_length-1) - *(pbs+j-1); 
        current_centre_dist = abs(centre - j - CORE_OFFSET); 
        if(current_total>best_total) 
          { 
          best_total=current_total; 
          best_core_start=j; 
          centre_dist = abs(centre - j - CORE_OFFSET); 
          } /* end of if */ 
        else if(current_total==best_total && current_centre_dist<centre_dist)
               { best_core_start=j; 
                 centre_dist = current_centre_dist; } 
        }/* end of for j */ 
      } /* end of if len>core_length */ 
      else { best_core_start = psp[i]->start; core_length=len; } 

     /* calculate N- and C-term context lengths */ 
     if(psp[i]->start < 10) { N_offset=psp[i]->start; }
     else { N_offset=10; }
     if(psp[i]->end > (sequence_length-10)) { C_offset=sequence_length-(psp[i]->end)-1; }
     else { C_offset=10; }    

     enr = ((double)(psp[i]->rescount)/(double)len) / psp[i]->freq; /* enrichment */ 
     fprintf(f4, "%s\t%d\t%s\t%d\t%d\t%d\t%d\t%.3e\t{", 
        psn, sequence_length, outputs[c], a+1, (psp[i]->start)+1, (psp[i]->end)+1, psp[i]->rescount, pow(10.0, psp[i]->lp)); 
     for(j=0;j<psp[i]->numrestypes;j++) { signature[j]=alph[psp[i]->restypes[j]]; } ; signature[psp[i]->numrestypes]='\0';
     fprintf(f4,"%s}\t%s\t%.3lf\t%d\t%d\t%.*s\t%.*s|\t%.*s\t|%.*s\n", signature, make_label(signature, label),  
      	enr, best_core_start+1, best_core_start+core_length, core_length, pcs+best_core_start, N_offset, 
          (pcs+psp[i]->start)-N_offset, psp[i]->end-psp[i]->start+1, pcs+psp[i]->start, C_offset, (pcs+psp[i]->end)+1); 
     a++; 
     } /* end of if*/ 
   } /* end for i*/ 

 } /* end of else not whole */ 

/* individual footer & assigning number_single, number_multiple */ 
if(!single_only)
{
if(headfoot)
{
if(c==SINGLE) 
  { number_single=a; number_multiple=0; if(a>0){ sprintf(current_info, "<%s\tlength=%d\t#%s=%d\t", psn, sequence_length, outputs[c], a); 
    a=0; single_or_multiple_or_whole=1; } 
  } 
else if(c==MULTIPLE) 
	   { number_multiple=a; if(a>0){ sprintf(info_append, "#%s=%d\t", outputs[c], a); strcat(current_info, info_append); 
       a=0; single_or_multiple_or_whole=1; } 
     }  
else if(c==WHOLE)  
  { 
   if(output_number_whole>0) {
      sprintf(info_append, "#%s=%d\t", outputs[c], output_number_whole); 
    strcat(current_info, info_append); single_or_multiple_or_whole=1; } 
   if(single_or_multiple_or_whole) { fprintf(f4, "%s\n", current_info); } 
   single_or_multiple_or_whole=0; 
      } 

} /* end of if headfoot */ 
else { /*!headfoot*/ if(c==SINGLE){number_single=a;} else if(c==MULTIPLE){number_multiple=a;} }
} /* end of if !single_only */ 
else { /* single_only */ 
number_single=a; 
if(headfoot && a>0){ sprintf(current_info, "<%s\tlength=%d\t#%s=%d\t", psn, sequence_length, outputs[c], a); 
                     a=0; fprintf(f4, "%s\n", current_info); }  
} /* end of single_only */ 

} /* end of LONG_FORMAT */ 

} /* end of output() */ 

#ifdef DEVELOP 
void extra_technical_output()
{ 
int i,j,k,N_windows; 

/* output N_windows for each residue type and calculate & output maxima */ 
fprintf(f3,"%s\t1\t", sequence_name); 
for(i=0,N_windows=0;i<alphabet_length;i++) 
   { fprintf(f3, "%d\t", residue_regions_tally[i]); N_windows += residue_regions_tally[i]; } 

if(number_third_stored==0) { number_third_stored=number_protein_stored; }
fprintf(f3,"\n%s\t2\t%d\t%d\t%d\t%d\t%d\t%d\n", sequence_name, sequence_length, N_windows, 
  number_of_lps, number_third_stored, number_single, number_multiple); 
} /* end of extra_technical_output() */ 
#endif 

void inner_bias_filter(struct subsequence *kept, int alphabet_length, int table[][alphabet_length])
{ 
int g,h,i,j,k,mincheck, range, temp_end, temp_maximum, removed_position, halfseq; 
int current_count, search_start, search_end, first_stored_for_window_size, previous_first_stored_for_window_size; 
double fraction_g, fraction_h, denom_g, denom_h; 
int *pis; 
 
pis=iseq;
temp_end=removed_position=current_count=first_stored_for_window_size=0; 
previous_first_stored_for_window_size=0; 

search_end = kept->end + stepsize;     if(search_end>sequence_length-1) { search_end=sequence_length-1; }
search_start = kept->start - stepsize; if(search_start<0) { search_start=0; } 

range = kept->end - kept->start + 1;  /* range is now measured as plus 1 */
if(range>sequence_length) { temp_maximum=sequence_length; } else { temp_maximum=range; } 
if(temp_maximum>MAXIMUM_LENGTH) { temp_maximum=MAXIMUM_LENGTH; }

halfseq = (int) (  ((double) sequence_length)/2.0  ) +2; 

for(i=temp_maximum; i>=minimum_window; i--)
   { 
   first_stored_for_window_size=0;
   current_count=0; 
   /* count up residue_type amounts for first window*/ 
   for(j=search_start;j<(search_start+i);j++)
      { if(*(pis+j) == rt) { current_count++; } } 
 
   if(current_count >= table[i-1][rt])  
     { 
     number_of_lps++;
     first_stored_for_window_size=1; 
     second_stored[number_of_lps].start = search_start; 
     second_stored[number_of_lps].end = search_start+i-1; 
     second_stored[number_of_lps].rescount = current_count; 
     second_stored[number_of_lps].flag = 0; 
     
     goto first_window_stored; 
     } /* end of if current_count >= . . . etc. */ 
 
   /* analyse subsequent windows until first_window_stored is found */ 
   for(k=search_start+1;k<=(search_end-i+1);k++) 
      { 
      removed_position = k-1; 
      temp_end = k+i-1; 
      if( *(pis+removed_position)==rt) { current_count--; } 
      if( *(pis+temp_end)==rt) { current_count++; } 
      
      if(current_count >= table[i-1][rt])
        { 
        first_stored_for_window_size=1; 
        number_of_lps++;  
        second_stored[number_of_lps].rescount = current_count; 
        second_stored[number_of_lps].start = removed_position+1; 
        second_stored[number_of_lps].end = temp_end; 
        second_stored[number_of_lps].flag = 0; 

        goto first_window_stored; 
        } /* end of if current_count>previous_count && ... */ 
      }/* end for k . . . start positions for windows */ 

first_window_stored: /* GOTO DESTINATION */ 
   if(first_stored_for_window_size==1)
   { 
   /* analyse subsequent windows */ 
   for(k=second_stored[number_of_lps].start+1; k<=(search_end-i+1); k++) 
      { 
      removed_position = k-1; 
      temp_end = k+i-1; 
      if(*(pis+removed_position)==rt) { current_count--; } 
      if(*(pis+temp_end)==rt) { current_count++; } 
      
      if(current_count>second_stored[number_of_lps].rescount)
        { 
        if(removed_position>=second_stored[number_of_lps].end)
          { number_of_lps++; } 
        second_stored[number_of_lps].rescount = current_count; 
        second_stored[number_of_lps].start = removed_position+1; 
        second_stored[number_of_lps].end = temp_end; 
        second_stored[number_of_lps].flag = 0; 
        } /* end of if current_count> ... */ 
      } /* end for k . . . start positions for windows */ 

      /* compare to previous window length */ 
      if(previous_first_stored_for_window_size==1)
        {
        for(g=number_of_lps;g>previous_number_of_lps;g--)
           {
denom_g = ((double)(second_stored[g].end-second_stored[g].start+1)); 
fraction_g = ((double)second_stored[g].rescount)/ denom_g ; 

           for(h=previous_number_of_lps;h>further_back_number_of_lps;h--)
              {
              denom_h = ((double)(second_stored[h].end-second_stored[h].start+1));

              if(  /*BOTH LARGER THAN HALF*/ 
                  (denom_g>halfseq && denom_h>halfseq)
                || /* OVERLAP? */ 
                (  (second_stored[g].end>=second_stored[h].start && second_stored[g].start<=second_stored[h].start)
                || (second_stored[g].end>=second_stored[h].end && second_stored[g].start<=second_stored[h].end) 
                ))
                {
                if(!second_stored[h].flag)
                  {
                  fraction_h = ((double)second_stored[h].rescount)/ denom_h;  
                  if(fraction_h>=fraction_g) { second_stored[g].flag=1; break; } 
                  else { second_stored[h].flag=1; second_stored[g].flag=0; } 
                  } /* end of if flag == 0 */ 
                } /* end of if OVERLAP? */             
              } /* end for h */ 
           } /* end for g */ 
        } /* end of if previous_first_stored_for_window_size */
      } /* end of if first_stored_for_window_size==1 */ 

   further_back_number_of_lps = previous_number_of_lps; 
   previous_number_of_lps = number_of_lps; 
   previous_first_stored_for_window_size = first_stored_for_window_size; 
   }/* end for i . . . window sizes */ 
} /* end of inner_bias_filter() */ 


void check_ends(int i)
{
while(second_stored[i].end<sequence_length-1) 
     { second_stored[i].end++; 
       if(iseq[second_stored[i].end]==rt) 
         { second_stored[i].rescount++; } 
       else { second_stored[i].end--; break; } } 
while(second_stored[i].start>0) 
     { second_stored[i].start--; 
       if(iseq[second_stored[i].start]==rt) 
         { second_stored[i].rescount++; } 
       else { second_stored[i].start++; break; } }
} /* end of check_ends() */ 


void outer_bias_filter(int alphabet_length, int table[][alphabet_length], double freq[], char *alph)
{
int i,j,k,ii,jj,kk,p=0,q=0,r=0,s=0,t=0,within=0, temp_maximum_window, *residue_types_tally;  
struct subsequence *kept; 

residue_types_tally = (int *)calloc(alphabet_length,sizeof(int)); 
memset(residue_regions_tally, 0, alphabet_length * sizeof(int)); 
if(sequence_length<maximum_window) { temp_maximum_window = sequence_length; }
else { temp_maximum_window = maximum_window; }
number_first_stored=0; 

      /* analyze sequence */       
      for(ii=temp_maximum_window-1;ii>=minimum_window-1;ii--) 
        { 
        memset(residue_types_tally, 0, alphabet_length * sizeof(int));  

        /* count up residue_type amounts for first window*/ 
        for(kk=0;kk<(ii+1);kk++)
           { residue_types_tally[ iseq[kk] ]++; }  
        for(k=0;k<alphabet_length;k++)
           {
           if(residue_types_tally[k]>=table[ii][k]) 
             {
             stored[k][residue_regions_tally[k]].start = 0; 
             stored[k][residue_regions_tally[k]].end = ii; 
             stored[k][residue_regions_tally[k]].residue_types_tally = residue_types_tally[k]; 
             residue_regions_tally[k]++; 
             } /* end of if residue_types_tally[k] */ 
           } /* end for k */ 

        for(jj=stepsize;jj<(sequence_length-ii-1);jj+=stepsize)
           {
           /* update residue_type amounts --> shift by stepsize */ 
           for(kk=jj-stepsize;kk<=jj-1;kk++) { residue_types_tally[ iseq[kk] ]--; }
           for(kk=ii+jj;kk>=ii+jj-stepsize+1;kk--) { residue_types_tally[ iseq[kk] ]++; } 

           /* store tallies if over a threshold */ 
           for(k=0;k<alphabet_length;k++)
              {
              if(residue_types_tally[k]>=table[ii][k]) 
                {
                within=0; 
                for(q=0;q<residue_regions_tally[k];q++)
                   {
                   if(jj+ii<=stored[k][q].end && jj>=stored[k][q].start) 
                     {
                     within=1; 
                     if(stored[k][q].residue_types_tally==residue_types_tally[k]) 
                       { stored[k][q].start = jj; 
                         stored[k][q].end = jj+ii; } /* end of if */ 
                         break;
                       } /* end of if */ 
                    } /* end for q */ 
                if(!within)
                  { stored[k][residue_regions_tally[k]].start = jj; 
                    stored[k][residue_regions_tally[k]].end = jj+ii; 
                    stored[k][residue_regions_tally[k]].residue_types_tally = residue_types_tally[k]; 
                    residue_regions_tally[k]++; } 
                } /* end of if residue_types_tally */ 
              } /* end of for k */ 
            } /* end for jj */ 
        } /* end for ii */ 
 
      /** merge any regions of the same bias that overlap **/  
      for(k=0;k<alphabet_length;k++)
      {
      for(r=0;r<residue_regions_tally[k];r++) 
         { 
         for(q=0;q<r;q++) 
            { 
            if(stored[k][q].residue_types_tally!=99999 &&  
              ((stored[k][r].end>=stored[k][q].start && stored[k][r].start<=stored[k][q].start) || 
              (stored[k][q].end>=stored[k][r].start && stored[k][q].start<=stored[k][r].start)))
              { 
              stored[k][q].residue_types_tally=99999; 
              if(stored[k][q].start<stored[k][r].start) { stored[k][r].start = stored[k][q].start; } 
              if(stored[k][q].end>stored[k][r].end) { stored[k][r].end = stored[k][q].end; } 
              } /* end of if .... */ 
            } /* end for q */ 
         } /* end for r */ 
         } /* end for k */ 
      
      /** inner filter processing  **/  
      further_back_number_of_lps=previous_number_of_lps=number_protein_stored=0;  
      for(rt=0,s=0;rt<alphabet_length;rt++)
      {
      for(r=0;r<residue_regions_tally[rt];r++) 
         { 
           if(stored[rt][r].residue_types_tally!=99999) /* if good first stored */ 
             { 
              kept=&stored[rt][r]; 
              s++; 
              number_of_lps=number_third_stored=0; 
              /* second_stored is filled from array address 1 */ 
              inner_bias_filter(kept, alphabet_length, table); 

              if(number_of_lps==0) { continue; }
              else if(number_of_lps==1) 
                     { 
                     check_ends(1); 

                     /* fill protein_stored array */ 
                     protein_stored[number_protein_stored].start = second_stored[1].start; 
                     protein_stored[number_protein_stored].end = second_stored[1].end;    
                     protein_stored[number_protein_stored].rescount = second_stored[1].rescount; 
                     protein_stored[number_protein_stored].restypes[0] =  rt ; 
                     protein_stored[number_protein_stored].numrestypes = 1; 
                     if(protein_stored[number_protein_stored].end - protein_stored[number_protein_stored].start + 1 == minimum_window)
                       { trimmed_off=0; trim_minimum_LPS(number_protein_stored); } /* MIN TRIM HERE */ 
                     protein_stored[number_protein_stored].lp = 
logbinomial((protein_stored[number_protein_stored].end-protein_stored[number_protein_stored].start)+1,protein_stored[number_protein_stored].rescount,freq[rt]); 
                     protein_stored[number_protein_stored].freq = freq[rt]; 
                     protein_stored[number_protein_stored].flag = 0; 

                     number_protein_stored++;
                     } /* end of if number_of_lps==1 */ 
              else { /* number_of_lps>1 */ 
                   for(i=1;i<=number_of_lps;i++)
                      { 
                      if(!second_stored[i].flag)
                        {
                        check_ends(i); 

                        /* fill third_stored while calculating pvalue */ 
                        third_stored[number_third_stored].start = second_stored[i].start; 
                        third_stored[number_third_stored].end = second_stored[i].end;
                        third_stored[number_third_stored].rescount = second_stored[i].rescount; 
                        third_stored[number_third_stored].restypes[0] = rt ; 
                        third_stored[number_third_stored].numrestypes = 1; 
                        third_stored[number_third_stored].lp = 
                          logbinomial((second_stored[i].end-second_stored[i].start)+1,second_stored[i].rescount,freq[rt]); 
                        third_stored[number_third_stored].freq = freq[rt]; 
                        third_stored[number_third_stored].flag = 0; 

                        number_third_stored++;
                        } /* end of if !second_stored[i].flag */ 
                      } /* end for i */ 

                   if(number_third_stored==0) { continue; }
                   else if(number_third_stored==1)
                          { /*fill protein_stored */ 
                          protein_stored[number_protein_stored].start = third_stored[0].start; 
                          protein_stored[number_protein_stored].end = third_stored[0].end;
                          protein_stored[number_protein_stored].rescount = third_stored[0].rescount; 
                          protein_stored[number_protein_stored].restypes[0] = third_stored[0].restypes[0]; 
                          protein_stored[number_protein_stored].numrestypes = 1; 
                          if(protein_stored[number_protein_stored].end - protein_stored[number_protein_stored].start + 1 == minimum_window)
                            { trimmed_off=0; trim_minimum_LPS(number_protein_stored); 
                              if(trimmed_off) { 
                              protein_stored[number_protein_stored].lp = 
logbinomial((protein_stored[number_protein_stored].end-protein_stored[number_protein_stored].start)+1,protein_stored[number_protein_stored].rescount,freq[rt]); 
                                              } else { protein_stored[number_protein_stored].lp = third_stored[0].lp; } }  
                          /* MIN TRIM HERE */ 
                          else { protein_stored[number_protein_stored].lp = third_stored[0].lp; }
                          protein_stored[number_protein_stored].freq = freq[rt]; 
                          protein_stored[number_protein_stored].flag = 0;  
                          
                          number_protein_stored++; 
                          } /* end of if number_third_stored==1 */ 
                   else { /* number_third_stored>1 */ 
                        qsort(third_stored, number_third_stored, sizeof(struct lps), compare); 

                        /**** reduce list for overlap ****/ 
                        for(i=0;i<number_third_stored;i++)
                           { 
                           if(!tsp[i]->flag)
                             { 
                             for(j=i+1;j<number_third_stored;j++)
                                { if( !tsp[j]->flag && !(tsp[j]->start<tsp[i]->start &&
	                                     tsp[j]->end  <tsp[i]->start) 
                                    && !(tsp[i]->start<tsp[j]->start &&
	                                     tsp[i]->end  <tsp[j]->start) ) /* OVERLAP? */
                                    { tsp[j]->flag=1; }  
                                }/* end of for j */ 
                             }/* end of if !flag i */ 
                           }/* end of for i */ 

                           for(i=0;i<number_third_stored;i++)
                              { if(!tsp[i]->flag) { 
                                protein_stored[number_protein_stored] = *tsp[i] ; 
                                if(protein_stored[number_protein_stored].end - protein_stored[number_protein_stored].start + 1 == minimum_window)
                                  { trimmed_off=0; trim_minimum_LPS(number_protein_stored); 
                                    if(trimmed_off) { 
                                    protein_stored[number_protein_stored].lp = 
logbinomial((protein_stored[number_protein_stored].end-protein_stored[number_protein_stored].start)+1,protein_stored[number_protein_stored].rescount,freq[rt]); 
                                                    }
                                  } /* MIN TRIM HERE */ 
                                number_protein_stored++; } 
                              }/* end of for i */
                           /* END OF REDUCE FOR OVERLAP */ 
                        } /* end of if number_third_stored>1 */ 
                   } /* end of if number_of_lps>1 */
              } /* end of if good first_stored */ 
    } /* end for r */ 
    } /* end for p */ 

number_kept=s; 

/* sort, output, merge, output */ 
if(number_protein_stored>1) { qsort(protein_stored, number_protein_stored, sizeof(struct lps), compare); } 

if(!single_only)
{
if(format!=MASKED)
  { if(number_protein_stored>0)
      { op=SINGLE; 
        number_single=number_multiple=0; 
        output(op, freq, alph); 
        op=MULTIPLE; 
        merge(freq); 
        output(op, freq, alph); } 
#ifdef DEVELOP
    if(eop) { extra_technical_output(); } 
#endif 
    op=WHOLE; 
    output(op, freq, alph); } 
else { op=SINGLE; 
       number_single=number_multiple=0; 
       output(op, freq, alph); 
       op=MULTIPLE; 
       merge(freq); 
       output(op, freq, alph); 
#ifdef DEVELOP 
       if(eop) { extra_technical_output(); } 
#endif 
     }
} /* end of if !single_only */ 
else { /* single_only */ 
if(format!=MASKED)
  { if(number_protein_stored>0)
      { op=SINGLE; 
        number_single=0; 
        output(op, freq, alph); } 
  } 
else { op=SINGLE; 
       number_single=0; 
       output(op, freq, alph); }
#ifdef DEVELOP
if(eop) { extra_technical_output(); } 
#endif 
} /* end of else single_only */ 
free(residue_types_tally); 
} /* end of outer_bias_filter() */ 


void read_domains(char *nameline)
{
int i; 
char *position, interval[51]; 
char *shift_format, *domain_format; 

/* read number of domains */ 
if(sscanf(nameline,"%*s %d",&number_of_domains)==1) 
  { 
  if(number_of_domains>MAXIMUM_DOMAINS) { 
    fprintf(stderr, 
     "Sequence %s has too many domains to read (%d) for the -D option. Only the first %d will be read.\n\n", 
       sequence_name, number_of_domains, MAXIMUM_DOMAINS); number_of_domains=MAXIMUM_DOMAINS; } 
 
  shift_format = (char *)malloc((8*(number_of_domains+2)+8) * sizeof(char)); 
  domain_format = (char *)malloc((8*(number_of_domains+2)+8) * sizeof(char)); 
  strcpy(shift_format,"%*s %*s ");    
  strcpy(domain_format,"%*s %*s %s %s "); 

/* read the domain intervals progressively and split them using strtok() */ 
/********* for df=EXCISED_DOMAIN, domain_start is actually the excision_point, and domain_end is the domain_length *********/ 
for(i=1;i<=number_of_domains;i++) 
{ domain_name[i][0]='\0'; test_name[0]='\0'; interval[0]='\0'; 
sscanf(nameline,domain_format,interval,test_name); strncpy(domain_name[i],test_name,MAX_DOM_NAME_SIZE);
strcat(shift_format,domain_format); strcpy(domain_format,shift_format); 
strcpy(shift_format, "%*s %*s "); 
position = strtok(interval,token); sscanf(position,"%d",&domain_start[i]); 
position = strtok(0,token); sscanf(position,"%d",&domain_end[i]); 
if(domain_start[i]<=sequence_length)
  { fprintf(f4, "%s\t%s\t%d\t%d\t%d\t%s\n", sequence_name, dfo[df], i, domain_start[i], domain_end[i], domain_name[i]); } 
} /* end for i */ 
free(shift_format); free(domain_format); 
} /* end of if sscanf for number */ 
else { number_of_domains=0; } 

} /* end of read_domains() */ 


void read_sequence_line(char *sb)
{
int test_length, current_length; 

test_length=strlen(sb);
current_length=strlen(sequence); 
if(test_length+current_length > MAXIMUM_LENGTH)
  { strncat(sequence,sb,MAXIMUM_LENGTH-current_length-1); 
    if(verbose && !too_long) { fprintf(stderr, 
"the length of the sequence %s is too great, only the first %d residues will be analyzed\n", sequence_name, MAXIMUM_LENGTH); } 
    too_long=1; 
  } /* end of if test_length+strlen(sequence) > MAXIMUM_LENGTH */ 
else{ strcat(sequence,sb); }
} /* end of read_sequence_line() */ 


void analyse_aa(FILE *fi, double composn[], int alphabet_length, int table[][alphabet_length])
{
bool past_first_name_line=0; 

while(fgets(line,MAXIMUM_LENGTH-1,fi)!=NULL)
{ 
if(!strncmp(line,">",1))
  {
  if(past_first_name_line)
    { 
    sequence_length = strlen(sequence); 
    convert_aa_sequence(); 
    if(verbose) { fprintf(stderr, "analyzing %s  #%d sequence...\n", sequence_name, number_of_sequences); }
    outer_bias_filter(alphabet_length, table, composn, alphabet); 
    sequence[0]='\0'; sequence_name[0]='\0'; 
    } /* end of if past_first_name_line */ 
  else { past_first_name_line=1; }
  sscanf(line,"%s",sn); strcpy(sequence_name,sn+1); 
  if(domain_list){ read_domains(line); } 
  number_of_sequences++; too_long=0; 
  }/*** end of if !strncmp(line,">",1) ***/ 
else { read_sequence_line(line); } 
} /* end of while fgets(line,MAXIMUM_LENGTH-1,f1)!=NULL */ 
if(feof(fi))
  { /* handle last sequence */ 
  sequence_length = strlen(sequence); 
  convert_aa_sequence(); 
  if(verbose) { fprintf(stderr, "analyzing %s  #%d (last) sequence...\n", sequence_name, number_of_sequences); }
  outer_bias_filter(alphabet_length, table, composn, alphabet); 
  if(verbose) { fprintf(stderr, "Finished analysis of database %s.\n", infile); } 
  } else { fprintf(stderr, "error in reading database %s, exiting ...\n", infile); exit(1); }
} /* end of analyse_aa() */ 


void analyse_nt(FILE *fi, double composn[], int alphabet_length, int table[][alphabet_length])
{
bool past_first_name_line=0; 

while(fgets(line,MAXIMUM_LENGTH-1,fi)!=NULL)
{ 
if(!strncmp(line,">",1))
  {
  if(past_first_name_line)
    { 
    sequence_length = strlen(sequence); 
    convert_dna_sequence(); 
    if(verbose) { fprintf(stderr, "analyzing %s  #%d sequence...\n", sequence_name, number_of_sequences); }
    outer_bias_filter(alphabet_length, table, composn, nt_alphabet); 
    sequence[0]='\0'; sequence_name[0]='\0'; 
    } /* end of if past_first_name_line */ 
  else { past_first_name_line=1; }
  sscanf(line,"%s",sn); strcpy(sequence_name,sn+1); 
  if(domain_list){ read_domains(line); } 
  number_of_sequences++; too_long=0; 
  }/*** end of if !strncmp(line,">",1) ***/ 
else { read_sequence_line(line); } 
} /* end of while fgets(line,MAXIMUM_LENGTH-1,f1)!=NULL */ 
if(feof(fi))
  { /* handle last sequence */ 
  sequence_length = strlen(sequence); 
  convert_dna_sequence(); 
  if(verbose) { fprintf(stderr, "analyzing %s  #%d (last) sequence...\n", sequence_name, number_of_sequences); }
  outer_bias_filter(alphabet_length, table, composn, nt_alphabet); 
  if(verbose) { fprintf(stderr, "Finished analysis of database %s.\n", infile); } 
  } else { fprintf(stderr, "error in reading database %s, exiting ...\n", infile); exit(1); }
} /* end of analyse_nt() */ 


size_t getFilesize(const char* filename) {
struct stat st;
if(stat(filename, &st)) { return 0; }
return st.st_size;   
} /* end of getFilesize */ 


int file_exist(char *filename) { struct stat sa; return (stat (filename, &sa) == 0); } 


int main(int argc, char **argv)
  { 
  int i,j; 
  char residue, sh[50]; 
  FILE *f1,*f2,*f5; 
  double tempcomp, temp_pvalue;   

  /* for options */ 
  int c, errflg=0, run_id=0, restriction_flag=0, known_only=0, oop=0, opened=0, pvt_set=0, z_reset=0, m_set=0, M_set=0; 
  extern char *optarg;
  extern int optind, optopt; 
  char user_array[MAX_FILE_NAME_SIZE+1] = {"domains"}, e_prefix[MAX_FILE_NAME_SIZE+1], prefix[MAX_FILE_NAME_SIZE+1]; 
  char efile[MAX_FILE_NAME_SIZE+1], outfile[MAX_FILE_NAME_SIZE+1], outcomp[MAX_FILE_NAME_SIZE+1]; 
  char tempopt[50]="", optlist[50]="", restriction_list[AA_ALPHABET_SIZE+1] = {"O"}; 
  char startrealtime[30]; 
  char D_opt[30]={"excised"}, O_opt[30]={"short"}, z_option[30]={"fast"}; 
  time_t t; 
  size_t filesize; 

  double *pc, *pnc, composition[AA_ALPHABET_SIZE] = { /* re-sets to user composition if that is the option */ 
   /* the default 'domains' composition is derived from the astralscop-2.06 40% sequence identity threshold data set */ 
   0.0822, 0.0137, 0.0581, 0.0690, 0.0406, 0.0719, 0.0230, 0.0589, 0.0586, 0.0943, 0.0220, 
   0.0415, 0.0455, 0.0369, 0.0517, 0.0587, 0.0535, 0.0721, 0.0135, 0.0001, 0.0343 }; 
  double equal_composition[AA_ALPHABET_SIZE] = { 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 }; 
  double nt_composition[NT_ALPHABET_SIZE] = { 0.25, 0.25, 0.25, 0.25, 0.25 }; 
  
  strcpy(startrealtime,__TIME__); 
  srand(time(&t)); 

  bkgd=DOMAINS; 
  format=SHORT_FORMAT; 
  sequence_type=AA; alphabet_length=AA_ALPHABET_SIZE; 
  z=FAST; 
  
/*  *  *  *  PROCESS COMMAND-LINE OPTIONS  *  *  *  */ 
while((c = getopt(argc, argv, "hvdkD:sz:e:nyo:m:M:t:r:c:O:u")) != -1) {
     switch(c) {
     /* output */ 
     case 'h': print_help(); exit(0);   
     case 'v': verbose=1;     sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break;
     case 'd': headfoot=1;    sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break; 
/* #ifdef DEVELOP
     case 'e': eop=1; strcpy(e_prefix,optarg); sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break; 
#endif */ 
     case 'O': oop=1; strcpy(prefix,optarg);   sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break; 
     case 'D': domain_list=1; if(!strcmp(optarg, "masked")) { df=MASKED_DOMAIN; known_only=1; strcpy(token,"-"); strcpy(D_opt,optarg); } 
               else { df=EXCISED_DOMAIN; strcpy(token,","); } 
               sprintf(tempopt,"%c%s",optopt,D_opt); strcat(optlist, tempopt); break;
     case 'o': if(!strcmp(optarg, "long")) { format=LONG_FORMAT; strcpy(O_opt,optarg); }
               else if(!strcmp(optarg, "short")) { format=SHORT_FORMAT; } 
               else if(!strcmp(optarg, "masked")) { format=MASKED; known_only=1; strcpy(O_opt,optarg); } 
               else if(!strcmp(optarg, "oneline")) { format=ONELINE; strcpy(O_opt,optarg); } 
               else { fprintf(stderr, " -o value is not valid, using default short format\n"); 
                      format=SHORT_FORMAT; }
               sprintf(tempopt,"%c%s",optopt,O_opt); strcat(optlist, tempopt); break; 
     /* bias calculation */ 
     case 's': single_only=1; sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break; 
     case 'k': known_only=1;  sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break; 
     case 'z': strcpy(z_option,optarg);  
               if(!strcmp(z_option, "medium")) { z=MEDIUM; sprintf(tempopt,"%c%s",optopt,z_option); strcat(optlist, tempopt); } 
               else if(!strcmp(z_option, "thorough")) 
                      { z=THOROUGH; sprintf(tempopt,"%c%s",optopt,z_option); strcat(optlist, tempopt); }
               else { z=FAST; }; break; 
     case 'n': sequence_type=NT; alphabet_length=NT_ALPHABET_SIZE; sprintf(tempopt,"%c",optopt); strcat(optlist, tempopt); break; 
     case 'm': sscanf(optarg,"%d", &minimum_window); m_set=1; break; 
     case 'M': sscanf(optarg,"%d", &maximum_window); M_set=1; break; 
     case 't': sscanf(optarg,"%lf", &pvalue_threshold); pvt_set=1; pvalue_threshold = log10(pvalue_threshold); break; 
     case 'c': strcpy(user_array,optarg);  
               if(!strcmp(user_array, "equal")) 
                 { bkgd=EQUAL; sprintf(tempopt,"%c%s",optopt,optarg); strcat(optlist, tempopt); } 
               else if(!strcmp(user_array, "domains")) 
                      { bkgd=DOMAINS; sprintf(tempopt,"%c%s",optopt,optarg); strcat(optlist, tempopt); }
               else { bkgd=USER; sprintf(tempopt,"%cuser",optopt); strcat(optlist, tempopt); } break; 
     case 'r': restriction_flag=1; sprintf(tempopt,"%c%s",optopt,optarg); strcat(optlist, tempopt); 
               i=0; while(optarg[i]) { restriction_list[i]=toupper(optarg[i]); i++; } break; 
     /* errors */           
     case 'y': test=1; errflg++; break; 
     case ':': fprintf(stderr, "option -%c requires a value\n", optopt); errflg++; break;
     case '?': fprintf(stderr, "unrecognized option: -%c\n", optopt); errflg++;
} /* end of switch */ 
} /* end of while() getopt */ 
if (errflg) { print_help(); exit(1); } 

if(z==MEDIUM) { stepsize=2; baseline_log_p_value = -2.0; baseline_p_value = 0.01; if(!pvt_set) { pvalue_threshold = -2.0; } } 
else if(z==THOROUGH) { stepsize=1; baseline_log_p_value = -1.0; baseline_p_value = 0.1; 
                       if(!pvt_set) { pvalue_threshold = -1.0; } } 

if(minimum_window>ABSOLUTEMAX || minimum_window<ABSOLUTEMIN) { minimum_window=FIXED_MIN_WINDOW; } 
if(maximum_window>ABSOLUTEMAX || maximum_window<ABSOLUTEMIN) { maximum_window=FIXED_MAX_WINDOW; } 
if(minimum_window>maximum_window) { minimum_window=FIXED_MIN_WINDOW; maximum_window=FIXED_MAX_WINDOW; } 
if(!m_set && format==MASKED) { minimum_window=5; } 
if(!M_set && format==MASKED) { maximum_window=25; } 

if(pvalue_threshold>baseline_log_p_value) { pvalue_threshold=baseline_log_p_value; } 
if(pvalue_threshold==baseline_log_p_value && format==MASKED) { pvalue_threshold=DEFAULT_MASKING_LOG_PVALUE; }

if(oop) { sprintf(tempopt,"m%dM%dt%.2le",minimum_window,maximum_window,pow(10.0,pvalue_threshold)); strcat(optlist, tempopt); } 
#ifdef DEVELOP
if(eop && !oop) { sprintf(tempopt,"m%dM%dt%.2le",minimum_window,maximum_window,pow(10.0,pvalue_threshold)); strcat(optlist, tempopt); } 
#endif 

if(format==LONG_FORMAT) { if(minimum_window<CORE_WINDOW) { core_length=minimum_window; } }

if(known_only) { restriction_flag=1; if(sequence_type==NT) { strcpy(restriction_list, "ACGT"); } 
                                     else { strcpy(restriction_list,"ACDEFGHIKLMNPQRSTVWY"); } } 

if(z==THOROUGH && sequence_type==NT) { z=MEDIUM; z_reset=1; }

/*  *  *  *  OPEN SEQUENCE FILE  *  *  *  */ 
if((f1 = fopen(argv[optind],"r"))==NULL)
  { fprintf(stderr, "There is no sequence file. Please supply one in FASTA format.\n");
    exit(1); }   
else { strcpy(infile, argv[optind]); } 

/*  *  *  *  PROCESS COMPOSITION FILE AND RESTRICTION LIST, AND CALCULATE THRESHOLD TABLES  *  *  *  */ 

if(z!=FAST && !restriction_flag && bkgd!=USER) 
  { if(sequence_type==AA) { if(bkgd==EQUAL) { threshold_table(equal_composition, alphabet_length, equal_threshold_table); }
                            else { /*bkgd==DOMAINS*/ threshold_table(composition, alphabet_length, default_threshold_table); } } 
    else { /* NT */ threshold_table(nt_composition, alphabet_length, default_nt_threshold_table); } 
  } /* end of if z!=FAST && !restriction_flag && bkgd!=USER */  

  if(bkgd==USER) 
    {
    if((f2 = fopen(user_array, "r"))==NULL)   
      { fprintf(stderr, "The specified COMPOSITION FILE is not there.\n\n");
        print_composition_file_format(); 
        exit(1); }
    else { /* if open composition file successful */ 
         while(fgets(sh,49,f2)!=NULL)
              { 
              sscanf(sh,"%c %lf", &residue, &tempcomp); 
              if(sequence_type==AA)
                { 
              if(restriction_flag)
                {
                for(i=0;i<alphabet_length;i++) 
                   { if(alphabet[i]==toupper(residue)) 
                 	   { if(strchr(restriction_list, toupper(residue))!=NULL) { composition[i]=tempcomp; break; } 
                 	     else { composition[i]=0.999999; break; }
                 	   } /* end of if alphabet */   
                   } /* end for i */ 
                 } /* end of if restriction_flag */ 
              else { /*!restriction_flag */ 
                   for(i=0;i<alphabet_length;i++) 
                      { if(alphabet[i]==toupper(residue)) { composition[i]=tempcomp; break; } } 
                   } /* end of if !restriction_flag */

                 threshold_table(composition, alphabet_length, default_threshold_table);
                 } /* end of sequence_type==AA */ 
              else{ /* sequence_type==NT*/ 
                  if(restriction_flag)
                    {
                    for(i=0;i<alphabet_length;i++) 
                       { if(nt_alphabet[i]==toupper(residue)) 
                    	   { if(strchr(restriction_list, toupper(residue))!=NULL) { nt_composition[i]=tempcomp; break; } 
                 	         else { nt_composition[i]=0.999999; break; }
                   	       } /* end of if alphabet */   
                       } /* end for i */ 
                    } /* end of if restriction_flag */ 
              else { /* !restriction_flag */ 
                   for(i=0;i<alphabet_length;i++) 
                      { if(nt_alphabet[i]==toupper(residue)) { nt_composition[i]=tempcomp; break; } } 
                   } /* end of if !restriction_flag */ 

                  threshold_table(nt_composition, alphabet_length, default_nt_threshold_table); 
                  } /* end of else sequence_type==NT*/ 
              } /* end of while */ 
         if(!feof(f2)){ fprintf(stderr, 
            "Error reading user COMPOSITION FILE, check the format and try again...\n\n"); 
            print_composition_file_format(); 
            exit(1); }    
         } /* end of if open composition file successful */ 
    } /* end of if bkgd USER  */ 
   else{ /* bkgd!=USER */ 
       if(restriction_flag) 
         { 
         if(sequence_type==AA)
           { 
           if(bkgd==EQUAL)
             {  
             for(i=0;i<alphabet_length;i++) 
           	    { if(strchr(restriction_list, toupper(alphabet[i]))==NULL) { equal_composition[i]=0.999; } } 
             threshold_table(equal_composition, alphabet_length, equal_threshold_table); 
             } /* end of bkgd==EQUAL */ 
           else { /* bkgd==DOMAINS */ 
                for(i=0;i<alphabet_length;i++) 
                   { if(strchr(restriction_list, toupper(alphabet[i]))==NULL) { composition[i]=0.999; } } 
                threshold_table(composition, alphabet_length, default_threshold_table); 
                } /* end of bkgd==DOMAINS */ 
           } /* end of if sequence_type==AA */ 
         else { /* sequence_type==NT */ 
              for(i=0;i<alphabet_length;i++) 
            	   { if(strchr(restriction_list, toupper(nt_alphabet[i]))==NULL) { nt_composition[i]=0.999; } } 
              threshold_table(nt_composition, alphabet_length, default_nt_threshold_table); 
              } /* end of sequence_type==NT*/ 
         } /* end of if restriction_flag */ 
       } /* end of else bkgd!=USER*/ 

/*  *  *  *  HANDLE OUTPUT FILE STREAM(S)  *  *  *  */ 
if(strlen(optlist)==0) { strcpy(optlist,"default"); }
if(oop) { opened=0;  
while(!opened) { run_id = (int) floor( rand()/100000); 
                 sprintf(outfile,"%s.%d.%s.fLPS.out", prefix, run_id, optlist); 
                 if(!file_exist(outfile)) { f4=fopen(outfile, "w"); opened=1; }  
#ifdef DEVELOP
                 if(eop) { sprintf(efile,"%s.%d.%s.fLPS.extra.out", e_prefix, run_id, optlist); f3=fopen(efile, "w"); } 
#endif 
               } /*end of while !opened*/ 
} /* end if oop */ 
else { f4 = stdout; 
#ifdef DEVELOP 
if(eop) { opened=0;  
while(!opened) { run_id = (int) floor( rand()/100000); 
                 sprintf(efile,"%s.%d.%s.fLPS.extra.out", e_prefix, run_id, optlist); 
                 if(!file_exist(efile)) { f3=fopen(efile, "w"); opened=1; }
               } /*end of while !opened*/ 
} /* end if eop */ 
#endif                
} /* end else */ 

/*  *  *  *  OUTPUT OPTIONS/PARAMETERS, IF VERBOSE  *  *  *  */ 
if(verbose)
{
fprintf(stderr, "%s %s %s\n", argv[0], __DATE__, startrealtime); 
for(i=0;i<argc;i++) { fprintf(stderr, "%s ", argv[i]); } 
fprintf(stderr, "\nUsing fLPS version %.1lf\n\nOptions/parameters:\n------------------- \n", VERSION_NUMBER); 

if(sequence_type==NT) { fprintf(stderr, "Sequence_type is DNA (Option -n). RNA should be converted to DNA before program use.\n"); } 
else { fprintf(stderr, "Sequence_type is AMINO_ACID (default)\n\n"); }
if(format==LONG_FORMAT)       { fprintf(stderr, "Output columnar long format (Option -o long)\n"); } 
else if(format==SHORT_FORMAT) { fprintf(stderr, "Output columnar short format (Option -o short)\n"); } 
else if(format==MASKED)       { fprintf(stderr, "Output masked FASTA-format sequences (Option -o masked)\n"); } 
else if(format==ONELINE)      { fprintf(stderr, "Output one-line summary for each sequence (Option -o oneline)\n\n"); } 

fprintf(stderr, "minimum_window (m) =%d\nmaximum_window (M) =%d\n", minimum_window, maximum_window); 
fprintf(stderr, "pvalue_threshold (t) =%le\n\n", pow(10.0,pvalue_threshold)); 

if(format==MASKED && !pvt_set) { fprintf(stderr, "Default p-value threshold (0.00001) for masking is used.\n"); } 
if(format==MASKED && !m_set) { fprintf(stderr, "Default m=5 for masking is used.\n"); } 
if(format==MASKED && !M_set) { fprintf(stderr, "Default M=25 for masking is used.\n"); } 

if(sequence_type==AA)
{ if(z==FAST)        { fprintf(stderr, "Default base-line precision used (-z fast)\n"); } 
else if(z==MEDIUM)   { fprintf(stderr, "Medium base-line precision used (-z medium)\n"); } 
else if(z==THOROUGH) { fprintf(stderr, "Thorough base-line precision used (-z thorough)\n"); } 
if(bkgd==USER)           { fprintf(stderr, "background composition = %s (Option -c %s)\n", user_array, user_array); } 
else if(bkgd==EQUAL)     { fprintf(stderr, "background composition = equal (0.05, Option -c equal)\n"); } 
else /*DOMAINS*/         { fprintf(stderr, "background composition = domains (default, Option -c domains)\n\n"); } } /*end if AA*/
else { /* sequence_type==NT */ 
if(z==FAST)        { fprintf(stderr, "Default base-line precision used (-z fast)\n"); }
else if(z==MEDIUM) { fprintf(stderr, "Medium base-line precision used (-z medium)\n"); 
                     if(z_reset) {fprintf(stderr, "....Re-set from -z thorough, as thorough option is not available for DNA\n");} }
if(bkgd==USER)  { fprintf(stderr, "background composition = %s (Option -c %s)\n", user_array, user_array); } 
else { fprintf(stderr, "background composition = equal (0.25, default or Option -c equal)\n\n"); } } /*end if NT */ 

if(headfoot)    { fprintf(stderr, "Headers will be appear in all files as described in the README (Option -d).\n"); 
                  if(format==LONG_FORMAT) { fprintf(stderr, 
                    "Footers will be appear in the output (Option -d with -o long).\n");}; } 
if(single_only) { fprintf(stderr, "Only SINGLE-residue biases will be output (Option -s).\n"); } 
if(domain_list) { fprintf(stderr, "Lists of excised or masked domains will be read from the FASTA '>' name lines (Option -D)\n\n"); 
if(df==MASKED_DOMAIN) 
      { fprintf(stderr,"Unknown residues ('Xs' in amino-acid, and 'Ns' in DNA sequence) will be interpreted as already masked..\n"); }  
      if(format==ONELINE||format==MASKED) { fprintf(stderr, 
      "For the '-D' option, only the '-o long' and '-o short' output options are allowed.\nDomain lists will not be output.....\n"); 
      domain_list=0; }  } 
if(restriction_flag && !known_only) { fprintf(stderr, "Biases restricted to %s (Option -r)\n", restriction_list); } 
if(known_only) { fprintf(stderr, 
                 "Unknown residues will not be considered (Ns for nucleotide and Xs for amino-acid sequences) (Option -k)\n"); } 
if(oop) { fprintf(stderr, "Output will be to file %s (Option -O)\n", outfile); } else {fprintf(stderr, "Output will be to stdout");}
#ifdef DEVELOP
if(eop) { fprintf(stderr, "Extra technical output will be to file %s (Option -e)\n", efile); } 
#endif 
fprintf(stderr, "If parameter values were out of bounds, they have been set to defaults\n------------------- \n"); 
} /* end of output parameters if verbose */ 

/* output user tables, if user_composition specified */ 
if(verbose && bkgd==USER)
{
fprintf(stderr, "User composition file %s used.\n\n", user_array); 
if(sequence_type==AA)
{ 
fprintf(stderr,"COMPOSITION:\n"); 
for(i=0;i<alphabet_length;i++) { fprintf(stderr, "%c\t%lf\n", alphabet[i], composition[i]);} 
fprintf(stderr,"THRESHOLD TABLE\n"); 
for(i=0;i<MAXIMUMWINDOW;i++) { for(j=0;j<alphabet_length;j++) { 
     fprintf(stderr, "%d,",default_threshold_table[i][j]); } 
     fprintf(stderr, "\n"); 
   } fprintf(stderr, "\n\n"); } /* end if AA*/ 
else{ /* if NT*/ 
fprintf(stderr,"COMPOSITION:\n"); 
for(i=0;i<alphabet_length;i++) { fprintf(stderr, "%c\t%lf\n", nt_alphabet[i], nt_composition[i]);} 
fprintf(stderr,"THRESHOLD TABLE\n"); 
for(i=0;i<MAXIMUMWINDOW;i++) { for(j=0;j<alphabet_length;j++) { 
     fprintf(stderr, "%d,",default_nt_threshold_table[i][j]); } 
     fprintf(stderr, "\n"); 
   } fprintf(stderr, "\n\n"); }/* end if NT*/
} /* end of user table output if verbose */ 

/*  *  *  *  ALLOCATE GLOBAL ARRAYS *  *  *  */ 
sequence = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 
cleanseq = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 
if(format==LONG_FORMAT) { binseq = (bool *)calloc(MAXIMUM_LENGTH,sizeof(bool)); } 
iseq = (int *)calloc(MAXIMUM_LENGTH,sizeof(int)); 
if(!single_only) { respos = (int *)calloc(MAXIMUM_LENGTH,sizeof(int)); } 
tsp = (struct lps**)calloc(30*DEFAULT_UNIT_STORE, sizeof(struct lps*)); 
psp = (struct lps**)calloc(2*DEFAULT_UNIT_STORE, sizeof(struct lps*)); 
whole_rescounts = (int *)calloc(alphabet_length,sizeof(int)); 
residue_regions_tally = (int *)calloc(alphabet_length,sizeof(int));

if(!sequence || !iseq || !tsp || !psp)
  { fprintf(stderr, "memory allocation error, decrease MAXIMUM_LENGTH and UNIT_STORE in code," 
    " and/or use a smaller maximum_window (-M option)...exiting...\n"); 
    exit(1); }

for(i=0;i<30*DEFAULT_UNIT_STORE;i++) { tsp[i] = &third_stored[i]; } 
for(i=0;i<2*DEFAULT_UNIT_STORE;i++)  { psp[i] = &protein_stored[i]; } 

/*  *  *  * INITIAL HEADER OUTPUT *  *  *  */ 
if(headfoot)
{
if(format==SHORT_FORMAT)
{ fprintf(f4, "## short format\n## SEQUENCE_NAME\tSEQUENCE_LENGTH\tBIAS_TYPE\tLPS#\tSTART\tEND\t"
              "RESIDUE_COUNT\tBINOMIALP\tSIGNATURE\tCLASS\tENRICHMENT\n"); }
else if(format==LONG_FORMAT) { fprintf(f4, "## long format\n## footer begins < and lists numbers of each bias type\n##\n"
       "## SEQUENCE_NAME\tSEQUENCE_LENGTH\tBIAS_TYPE\tLPS#\tSTART\tEND\tRESIDUE_COUNT\tBINOMIALP\tSIGNATURE\tCLASS\tENRICHMENT\t"
       "CORE_START\tCORE_END\tCORE_SEQUENCE\tNTERM_CONTEXT\tLPS_SEQUENCE\tCTERM_CONTEXT\n"); } 
else if(format==ONELINE) { fprintf(f4, "## ONELINE format:\n## SEQUENCE_NAME\tSEQUENCE_LENGTH\tcomma-delimited list of biased regions:\n" 
       "## start-end,bias_residue_count,{bias_signature},binomial_P\n"); } 
else { /* MASKED */ fprintf(f4, "## FASTA format with regions masked (with 'X' for amino-acids, 'N' for nucleotides)\n"
       "## LPSs listed in > line in comma-delimited format after SEQUENCE_NAME and SEQUENCE_LENGTH:\n"
       "## start-end,bias_residue_count,{bias_signature},binomial_P\n"); } 
if((format==SHORT_FORMAT||format==LONG_FORMAT) && domain_list) { 
  if(df==MASKED_DOMAIN) { fprintf(f4, "## or ##\n## SEQUENCE_NAME\tMASKED_DOMAIN\tDOMAIN#\tSTART\tEND\tDOMAIN_NAME\n"); }
  else /*df==EXCISED*/  { fprintf(f4, "## or ##\n## SEQUENCE_NAME\tEXCISED_DOMAIN\tDOMAIN#\tEXCISION_POINT\tDOMAIN_LENGTH\tDOMAIN_NAME\n"); } } 
fprintf(f4,"##\n##\n"); 
} /* end of if headfoot */ 

#ifdef DEVELOP
if(eop)
{
fprintf(f3, "# %s %s %s\n", argv[0], __DATE__, startrealtime); 
for(i=0;i<argc;i++) { fprintf(f3, "%s ", argv[i]); } 
fprintf(f3, "#\n# Using fLPS version %.1lf\n#\n# EXTRA TECHNICAL OUTPUT\n# ----------------------\n#\n", VERSION_NUMBER); 
fprintf(f3, "# STEPSIZE\t%d\n# MAXIMUM_LENGTH\t%d\n# BASELINE_P_VALUE\t%lf\n# UNIT_STORE\t%d\n# MAXIMUM_CORE_WINDOW\t%d\n", 
  stepsize, MAXIMUM_LENGTH, baseline_p_value, DEFAULT_UNIT_STORE, CORE_WINDOW); 
fprintf(f3, "# alphabet_length\t%d\n#\n# The format per sequence is as follows:\n", alphabet_length); 
fprintf(f3, "# LINE #1:  sequence_name followed by N_windows for each sequence for each residue type\n"); 
if(sequence_type==AA) { for(i=0;i<alphabet_length;i++) { fprintf(f3, " %c", alphabet[i]); } } 
else { for(i=0;i<alphabet_length;i++) { fprintf(f3, "\t%c", nt_alphabet[i]); } } 
fprintf(f3, "\n# LINE #2: sequence_name followed by these parameters that count the number of windows or \n" 
            "# subsequences that are considered at different points in the program:\n" 
            "# sequence_name\tL\tN_window\tN_contig\tN_lps_filtered\tN_SINGLE\tN_MULTIPLE\n"); 
} /* end of if eop */
#endif 

/*  *  *  *  ANALYZE SEQUENCES *  *  *  */ 
sequence_name[0]='\0'; sequence[0]='\0'; 
if(sequence_type==NT) { analyse_nt(f1, nt_composition, alphabet_length, default_nt_threshold_table); }
else { /* sequence_type==AA */ 
     if(bkgd==EQUAL) { analyse_aa(f1, equal_composition, alphabet_length, equal_threshold_table); }
     else { analyse_aa(f1, composition, alphabet_length, default_threshold_table); } } 

if(!verbose) { fprintf(stderr, "fLPS has finished analysis of %d sequences in database %s.\n", number_of_sequences, infile); } 
exit(0); 
}/******** end of main() ********/ 

/********* END OF CODE FILE *********/ 



