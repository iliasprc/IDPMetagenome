
/*****************************************************************************************/
/****  fLPS: fast discovery of compositional biases for the protein universe 
 ****/ 
/****  Copyright 2017. Paul Martin Harrison. ****/ 
/****
 ****  Licensed under the 3-clause BSD license. See LICENSE.txt bundled in the fLPS package. 
 ****/ 
/****  
 ****  to compile: 
 ****   gcc -O2 -march=native -o fLPS fLPS.c -lm 
 ****
 ****  to run and get help: 
 ****   ./fLPS -h 
 ****
 ****  The latest version of this code is likely to be available at:
 ****    http://biology.mcgill.ca/faculty/harrison/flps.html 
 **** 
 ****  Citation: 
 ****    Harrison, PM. 'fLPS: fast discovery of compositional biases for the protein universe', 
 ****    (2017) BMC Bioinformatics, under revision. 
 ****
 ****/ 
/*****************************************************************************************/

#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <math.h> 
#include <ctype.h> 
#include <sys/types.h>
#include <sys/stat.h> 
#include <time.h> 
#include <unistd.h>
#include <sys/time.h> 
#include <sys/resource.h> 
#include <stdbool.h>

#define MAXIMUM_LENGTH 20000
#define FIXED_MIN_WINDOW 15 
#define FIXED_MAX_WINDOW 500  
#define ABSOLUTEMIN 5 
#define ABSOLUTEMAX 1000 
#define UNIT_STORE 1000
#define MAXIMUMWINDOW 1001 
#define BASELINE_P_VALUE 0.001 
#define BASELINE_LOG_P_VALUE -3.0 
#define DEFAULT_MASKING_LPVALUE -3.0 /* change to use different default masking threshold */ 
#define CORE_WINDOW 15 
#define CORE_OFFSET 7 
#define OUTPUT_LINE_LENGTH 80 
#define STEPSIZE 3 /* modify code at SHIFT_BY_STEPSIZE, if change this */ 

#include "tables.h" 

/* structs for storing subsequences */
/* array bounds were determined after analyses of TrEMBL database */ 
struct subsequence { 
int start; 
int end; 
int residue_types_tally; 
} stored[21][30*UNIT_STORE]; 

struct second_subsequence { 
int start; 
int end; 
int rescount; 
bool flag;
} second_stored[60*UNIT_STORE]; 

struct lps { 
int start; 
int end; 
int rescount; 
int restypes[21]; 
int numrestypes; 
double lp; 
double sumlp; 
bool flag;
} **tsp, **psp, third_stored[30*UNIT_STORE], protein_stored[2*UNIT_STORE]; 

/* sequence analysis */ 
int alphabet_length=21; 
int input_sequence_length=MAXIMUM_LENGTH; 
int sequence_length=MAXIMUM_LENGTH, core_length=CORE_WINDOW;
int number_of_lps, maxnumlps;  
int further_back_number_of_lps, previous_number_of_lps; 
int number_first_stored, number_third_stored; 
int number_protein_stored, number_kept; 
int number_of_sequences, maxfirststored, maxthirdstored, maxrespos; 
char alphabet[25] = {"ACDEFGHIKLMNPQRSTVWXY"}; 
char sequence_name[250] = {"initial"}; 
char *sequence, *cleanseq; 
bool *binseq; 
int *iseq, *respos, rt, whole_rescounts[21], elimits[21]; 

/* options/parameters */ 
int minimum_window=FIXED_MIN_WINDOW, maximum_window=FIXED_MAX_WINDOW; 
double pvalue_threshold=BASELINE_LOG_P_VALUE; 
bool headfoot, single_only; 
enum composn {DOMAINS, EQUAL, USER} bkgd; 
enum results {SHORT_FORMAT, LONG_FORMAT, MASKED} format; 
enum calculn {SINGLE, MULTIPLE, WHOLE} op; 
char outputs[3][10] = {"SINGLE", "MULTIPLE", "WHOLE"}; 

/* CPU usage*/ 
struct rusage measureusage; 
struct timeval startu, previousu, currentu; 
struct timeval starts, previouss, currents; 
FILE *f1;


size_t getFilesize(const char* filename) {
    struct stat st;
    if(stat(filename, &st)) {
        return 0;
    }
    return st.st_size;   
} /* end of getFilesize */ 


int file_exist(char *filename)
{
struct stat sa;   
return (stat (filename, &sa) == 0);
} /* end of file_exist() */ 


void print_help() 
{
fprintf(stdout, "\nfLPS\n"); 
fprintf(stdout,   "====\n"); 
fprintf(stdout, " -h   prints help\n"); 
fprintf(stdout, " -v   prints runtime information\n"); 
fprintf(stdout, " -d   prints header and footer information in the output files\n");    
fprintf(stdout, " -s   calculate single-residue biases ONLY\n");                      
fprintf(stdout, " -o   format specified as: short (default), long or masked\n"); 
fprintf(stdout, " -m   minimum tract length (default=15), must be <= maximum and in range 5-1000\n");              
fprintf(stdout, " -M   maximum tract length (default=500), must be >= minimum and in range 15-1000"
                " and > minimum tract length\n");              
fprintf(stdout, " -t   binomial p-value threshold, must be <=0.001 (can be either decimal or exponential)\n");                   
fprintf(stdout, " -c   if 'equal', it specifies equal amino-acid composition\n"); 
fprintf(stdout, "      if 'domains', it specifies domains amino-acid composition\n"); 
fprintf(stdout, "      else it specifies the name of a user amino-acid composition file\n"); 
fprintf(stdout, "      (default is 'domains')\n"); 
fprintf(stdout, "\nExamples:\n"); 
fprintf(stdout,   "---------\n\n"); 
fprintf(stdout, "./fLPS -v -m 20 -M 200 -c user_composition ...input_file... \n"); 
fprintf(stdout, " fLPS run with verbose runtime info, minimum tract length 20, maximum tract length 200\n");
fprintf(stdout, " and user_composition file specified\n\n"); 
fprintf(stdout, "./fLPS -vo long -t 0.000001 ...input_file... \n"); 
fprintf(stdout, " fLPS run with verbose runtime info, long output for results and pvalue threshold \n"); 
fprintf(stdout, " =0.000001\n\n"); 
fprintf(stdout, " IT IS RECOMMENDED TO USE DEFAULT VALUES, EXCEPT WHERE OTHER -m, -M AND -t VALUES\n"); 
fprintf(stdout, " MIGHT BE USEFUL IN COMBINATION WITH THE -o MASKED OPTION\n\n"); 
fprintf(stdout, " Very long sequences are curtailed to 20,000 residues length.\n"); 
fprintf(stdout, "CITATION:\n Harrison, PM. 'fLPS: fast discovery of compositional biases for the protein universe',\n(2017) BMC Bioinformatics, under revision. \n"); 
fprintf(stdout, "URL:\n http://biology.mcgill.ca/faculty/harrison/flps.html\n"); 
} /* end of print_help */ 


double logbinomial(int n, int k, double expect)
{ return logfactorial[n] - logfactorial[n-k] - logfactorial[k] + ((double)k)*log10(expect) + ((double)(n-k))*log10(1.0-expect); } 


void calculate_threshold_table(double freq[])
{
int i,j,k; 
double current_p_value = -9999.0, previous_p_value = -10000.0; 

for(j=20;j>=0;j--)
   {
   for(i=MAXIMUMWINDOW;i!=ABSOLUTEMIN-1;i--) 
      {
      current_p_value = -9999.0; previous_p_value = -10000.0;
      for(k=1;k<=i;k++)
         {
         current_p_value = logbinomial(i,k,freq[j]); 
         if(current_p_value<BASELINE_LOG_P_VALUE && current_p_value<previous_p_value)
           { break; }
         previous_p_value = current_p_value; 
         } /* end for k */ 
      default_threshold_table[i-1][j]=k;  
      } /* end for i */ 
    } /* end for j */ 
} /* end of calculate_threshold_table */ 


void convert_sequence()
{
int a,i,j,k,aa; 
static int *pis; 
static char *ps, *pcs; 

pis=iseq; 
ps=sequence; 
pcs=cleanseq; 

memset(whole_rescounts, 0, alphabet_length * sizeof(int)); 

for(a=0; ps<sequence+sequence_length; ps++)
    {
    aa=toupper(*ps); 
    switch(aa) 
          {
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
          case 88 /*'X'*/: *(pis+a) = 19; *(pcs+a) = aa; a++; whole_rescounts[19]++; break;
          } /* end of switch */ 
    } /* end for pis */  

input_sequence_length=sequence_length;            
sequence_length=a; 
} /* end of convert_sequence() */ 


void merge(double freq[])
{
int i,j,k,l,m,n,p,q; 
int Nterm,Cterm,found,N,C; 
int add_restypes[21], prev_bias_residue; 
int best_start,best_end,best_rescount,addl_bias_residues,test_elimit,test_N,test_C; 
bool there_already, new_bias_residue, adjust_N, adjust_C; 
double temp_composition,current_lpvalue; 
double first_lpvalue,best_lpvalue,moveN_lpvalue,moveC_lpvalue,moveboth_lpvalue; 

for(i=0;i<number_protein_stored;i++)
   {
   if(psp[i]->flag==0) 
     {
     for(j=i+1;j<number_protein_stored;j++) 
        {
        if(psp[j]->flag==0 && !(psp[j]->start <psp[i]->start && psp[j]->end <psp[i]->start) 
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
                  {
                  if(iseq[l]==psp[i]->restypes[k]) 
                    { respos[m]=l; m++; found=1; break; } 
                  } /* end for k */ 
               if(found==0)
               	 {
               	 for(k=0;k<psp[j]->numrestypes;k++)
                    {
                    if(iseq[l]==psp[j]->restypes[k]) 
                      { respos[m]=l; m++; break; } 
                    } /* end for k */ 
                 } /* end of if found==0 */ 
               }/* end for l */
            if(m>maxrespos) { maxrespos=m; }

            /* calculate temp_composition and add_restypes */ 
            temp_composition=0.0; 
               for(k=0;k<psp[i]->numrestypes;k++)
                    { temp_composition += freq[psp[i]->restypes[k]]; } 
               for(k=0,p=0;k<psp[j]->numrestypes;k++)
                    { 
                    there_already=0; 
                    for(l=0;l<psp[i]->numrestypes;l++)
                       {
                       if(psp[j]->restypes[k]==psp[i]->restypes[l])
                         { there_already=1; break; } 
                       } /* end for l */ 
                    if(there_already==0)
                      { 
                      temp_composition += freq[psp[j]->restypes[k]];  
                      add_restypes[p] = psp[j]->restypes[k]; 
                      p++; 
                      }
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
                       {
                       best_lpvalue = moveboth_lpvalue; 
                       best_start = respos[N]; best_end = respos[C]; best_rescount = C-N+1; 
                       } } 
                 else if(moveN_lpvalue<moveboth_lpvalue && moveN_lpvalue<moveC_lpvalue)
                      { if(moveN_lpvalue<best_lpvalue)
                        { 
                        best_lpvalue = moveN_lpvalue; 
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
                        { 
                        if(iseq[test_N]==psp[i]->restypes[k])
                          { addl_bias_residues++; prev_bias_residue=test_N; new_bias_residue=1; 
                            break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { 
                        if(iseq[test_N]==add_restypes[k])
                          { addl_bias_residues++; prev_bias_residue=test_N; new_bias_residue=1; 
                            break; }
                        } /* end for k */ 
                     if(new_bias_residue==1)
                       {
                       /* calc current_lpvalue */ 
                       current_lpvalue = 
                          logbinomial(Cterm-test_N+1, m+addl_bias_residues, temp_composition);

                       if(current_lpvalue<best_lpvalue)
                         {
                         /* update best data */ 
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
                        { 
                        if(iseq[test_C]==psp[i]->restypes[k])
                          { addl_bias_residues++; prev_bias_residue=test_C; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { 
                        if(iseq[test_C]==add_restypes[k])
                          { addl_bias_residues++; prev_bias_residue=test_C; new_bias_residue=1; break; }
                        } /* end for k */ 
                     if(new_bias_residue==1)
                       {
                       if(best_start>Nterm)
                       	 {/* calc current_lpvalue */ 
                       current_lpvalue = 
                          logbinomial(test_C-Nterm+1, m+addl_bias_residues, temp_composition);

                       if(current_lpvalue<best_lpvalue)
                         {
                         /* update best data */ 
                         best_lpvalue = current_lpvalue; 
                         best_start = Nterm; 
                         best_end = test_C; 
                         best_rescount= m+addl_bias_residues; 
                         } /* end of if current_lpvalue<best_lpvalue */ 
                       else { break; } /* break out of while */ 
                         } /* end of if best_end>Cterm */ 

                       else { 
                       current_lpvalue = 
                          logbinomial(test_C-best_start+1, m+addl_bias_residues, temp_composition);

                       if(current_lpvalue<best_lpvalue)
                         {
                         /* update best data */ 
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
                        { 
                        if(iseq[test_N]==psp[i]->restypes[k])
                          { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { 
                        if(iseq[test_N]==add_restypes[k])
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
                        { 
                        if(iseq[test_C]==psp[i]->restypes[k])
                          { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     for(k=0;k<p;k++)
                        { 
                        if(iseq[test_C]==add_restypes[k])
                          { addl_bias_residues++; new_bias_residue=1; break; }
                        } /* end for k */ 
                     if(!new_bias_residue) { test_C--; break; } 
                     } /* end of while test_C<sequence_length */ 
                     if(addl_bias_residues>0)
                       { best_end = test_C; best_rescount+=addl_bias_residues; adjust_C=1; } 
                if(adjust_N||adjust_C)
                  { best_lpvalue = 
                      logbinomial(best_end-best_start+1, best_rescount, temp_composition); } 

              psp[i]->lp = best_lpvalue; 
              psp[i]->start = best_start; 
              psp[i]->end = best_end;               
              memcpy((psp[i]->restypes)+psp[i]->numrestypes, add_restypes, p*(sizeof(int))); 
              psp[i]->numrestypes += p; 
              psp[i]->sumlp += psp[j]->sumlp; 
              psp[i]->rescount = best_rescount; 
              } /* end of if best_lpvalue<psp[i]->lp */ 
          } /* end of if flag j ==0 && OVERLAP */ 
        } /* end for j */ 
     } /* end of if flag i ==0 */ 
   } /* end for i */ 
} /* end of merge () */ 
 

int compare(const void * a, const void * b) /* for qsort */ 
{
struct lps *lpsA = (struct lps *)a; 
struct lps *lpsB = (struct lps *)b; 

return ( (int) (100000.0 * (lpsA->lp - lpsB->lp)) ); 
}/* end of compare() */ 


int output(enum calculn c, double freq[]) 
{
int i,j,k,l,a; 
int core_start=-1,N_offset,C_offset,centre; 
int best_total, best_core_start, centre_dist, current_centre_dist, current_total; 
static char *psn=sequence_name, *pcs;
static bool *pbs, single_or_multiple; 
static int *pis; 
static char current_info[25*UNIT_STORE];
char info_append[50], signature[21]; 
double p,lp,f; 
int output_number_whole=0;

pbs=binseq; 
pcs=cleanseq; 
pis=iseq; 

if(format==MASKED)
{
if(c==SINGLE) { strcpy(current_info,psn); strcat(current_info, "\t"); }

for(i=0;i<number_protein_stored;i++) 
   {
   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { 
     /* make name line cumulatively */ 
     memset(signature, 0, alphabet_length * sizeof(char));   
     for(j=0,a=0;j<psp[i]->numrestypes;j++) { signature[a]=alphabet[psp[i]->restypes[j]]; a++; } 
     sprintf(info_append, "%d{%s}%d=%.3e ", (psp[i]->start)+1, signature, 
     	(psp[i]->end)+1, pow(10.0, psp[i]->lp)); 
     strcat(current_info, info_append); 

     /* mask cleanseq cumulatively */ 
     for(j=psp[i]->start;j<=psp[i]->end;j++)
        { if(*(pcs+j) != 88 /*X*/)
            { for(k=0;k<psp[i]->numrestypes;k++)
                 { if(psp[i]->restypes[k] == *(pis+j)) { *(pcs+j)=88; break; } } 
            } /* end of if cleanseq[j]!=X */ 
        } /* end of for j */ 
      } /* end of if p<=pvalue_threshold */ 
   } /* end for i*/ 

if(single_only==1||c==MULTIPLE)
{
fprintf(stdout, ">%s\n", current_info); 
for(i=0;i<=sequence_length-OUTPUT_LINE_LENGTH;i+=OUTPUT_LINE_LENGTH)
   {
   fprintf(stdout,"%.*s\n", OUTPUT_LINE_LENGTH, pcs+i); 
   }fprintf(stdout,"%.*s\n", sequence_length-i, pcs+i); 
} /* end of if single_only==1||c==MULTIPLE */ 

} /* end of if format==MASKED */ 

else if(format==SHORT_FORMAT)
{
if(c==WHOLE)
  {
    for(i=0,a=0;i<21;i++)
     {
     lp = logbinomial(sequence_length, whole_rescounts[i], freq[i]); 
     f = ((double) whole_rescounts[i]) / ((double) sequence_length); 
     if(lp<pvalue_threshold && f>=freq[i])
       {
       protein_stored[a].lp = lp; 
       protein_stored[a].restypes[0] = i; 
       protein_stored[a].rescount = whole_rescounts[i]; 
       a++; 
       }/* end of if lp<BASELINE_LOG_P_VALUE */ 
     } /* end of for i */ 

if(a>0)
  {   
    if(a>1) { qsort(protein_stored, a, sizeof(struct lps), compare); 
    for(i=0;i<a;i++)
       { fprintf(stdout, "%s\t%s\t%d\t1\t%d\t%d\t%.3e\t{%c}\n", 
          psn, outputs[c], i+1, sequence_length, protein_stored[i].rescount, 
           pow(10.0,protein_stored[i].lp), alphabet[protein_stored[i].restypes[0]]); } 
            }
    else { fprintf(stdout, "%s\t%s\t%d\t1\t%d\t%d\t%.3e\t{%c}\n", 
           psn, outputs[c], 1, sequence_length, protein_stored[0].rescount, 
           pow(10.0,protein_stored[0].lp), alphabet[protein_stored[0].restypes[0]]); }
  }/* end of if a>0 */ 
  } /* end of if(c==WHOLE) */

else { /* not whole */ 
for(i=0,a=0;i<number_protein_stored;i++) 
   {
   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { 
     fprintf(stdout, "%s\t%s\t%d\t%d\t%d\t%d\t%.3e\t{", 
        psn, outputs[c], a+1, (psp[i]->start)+1, (psp[i]->end)+1, psp[i]->rescount, 
           pow(10.0, psp[i]->lp) ); 
     for(j=0;j<psp[i]->numrestypes;j++) { fputc(alphabet[psp[i]->restypes[j]],stdout); }
     fprintf(stdout,"}\n"); 
     a++; 
     } /* end of if p<=pvalue_threshold */ 
   } /* end for i*/ 
     } /* end of else not whole */ 
} /* end of if format==SHORT_FORMAT */ 

else { /* LONG_FORMAT */ 
if(c==WHOLE)
  {
    for(i=0,a=0;i<21;i++)
     {
     lp = logbinomial(sequence_length, whole_rescounts[i], freq[i]); 
     f = ((double) whole_rescounts[i]) / ((double) sequence_length); 
     if(lp<pvalue_threshold && f>=freq[i])
       {
       protein_stored[a].lp = lp; 
       protein_stored[a].restypes[0] = i; 
       protein_stored[a].rescount = whole_rescounts[i]; 
       a++; 
       }/* end of if lp<BASELINE_LOG_P_VALUE */ 
     } /* end of for i */ 

if(a>0)
  { 
    if(a>1) { qsort(protein_stored, a, sizeof(struct lps), compare); 
    for(i=0;i<a;i++)
       { fprintf(stdout, "%s\t%s\t%d\t1\t%d\t%d\t%.3e\t{%c}\n", 
          psn, outputs[c], i+1, sequence_length, protein_stored[i].rescount, 
           pow(10.0,protein_stored[i].lp), alphabet[protein_stored[i].restypes[0]]); } 
            }
    else { 
         fprintf(stdout, "%s\t%s\t%d\t1\t%d\t%d\t%.3e\t{%c}\n", 
          psn, outputs[c], 1, sequence_length, protein_stored[0].rescount, 
           pow(10.0,protein_stored[0].lp), alphabet[protein_stored[0].restypes[0]]);
         }
  }/* end of if a>0 */ 
  output_number_whole=a; 
  } /* end of if(c==WHOLE) */

else { /* not whole */ 
for(i=0,a=0;i<number_protein_stored;i++) 
   { 
   if(c==MULTIPLE && psp[i]->numrestypes==1) { continue; } 

   if(psp[i]->lp<=pvalue_threshold && !psp[i]->flag)
     { 
     if(psp[i]->end - psp[i]->start > core_length-1)
     {
     /* fill binseq for core calculation */ 
       for(j=psp[i]->start;j<=psp[i]->end;j++)
          { 
          for(k=0,*(pbs+j)=0;k<psp[i]->numrestypes;k++)
             {
             if(psp[i]->restypes[k]== *(pis+j))
               { *(pbs+j)=1; break; } 
             } /* end of for k */ 
          }/* end of for j */ 

     /** calculate core **/ 
     centre = (int) ((psp[i]->end - psp[i]->start +1)/2); 

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
      } /* end of if psp[i]->end - psp[i]->start > core_length-1 */ 
      else { best_core_start = psp[i]->start; } 

     /* calculate N- and C-term context lengths */ 
     if(psp[i]->start < 10)
       { N_offset=psp[i]->start; }
     else { N_offset=10; }
     if(psp[i]->end > (sequence_length-10))
       { C_offset=sequence_length-(psp[i]->end)-1; }
     else { C_offset=10; }    

     fprintf(stdout, "%s\t%s\t%d\t%d\t%d\t%d\t%.3e\t{", 
        psn, outputs[c], a+1, (psp[i]->start)+1, (psp[i]->end)+1, psp[i]->rescount, 
           pow(10.0, psp[i]->lp)); 
     for(j=0;j<psp[i]->numrestypes;j++) { fputc(alphabet[psp[i]->restypes[j]],stdout); }
     fprintf(stdout,"}\t%.3lf\t%d\t%d\t%.*s\t%.*s|\t%.*s\t|%.*s\n", 
     	psp[i]->sumlp, best_core_start+1, best_core_start+core_length, core_length, 
     	  pcs+best_core_start, N_offset, (pcs+psp[i]->start)-N_offset, 
     	     psp[i]->end-psp[i]->start+1, pcs+psp[i]->start, C_offset, 
     	        (pcs+psp[i]->end)+1); 
     a++; 
     } /* end of if*/ 
   } /* end for i*/ 
 } /* end of else not whole */ 

/* individual footer */ 
if(single_only==0)
{ 
if(headfoot==1)
{
if(c==SINGLE) 
  { if(a>0){ sprintf(current_info, "<%s\tlength=%d\t#%s=%d\t", 
                  psn, sequence_length, outputs[c], a); a=0; single_or_multiple=1; }
  } 
else if(c==MULTIPLE) 
	   { if(a>0){ sprintf(info_append, "#%s=%d\t", outputs[c], a); 
                       strcat(current_info, info_append); a=0; single_or_multiple=1; 
     } }  
else if(c==WHOLE && output_number_whole>0) 
  { 
  sprintf(info_append, "#%s=%d\t", outputs[c], output_number_whole); 
  strcat(current_info, info_append); 
  fprintf(stdout, "%s\n", current_info); output_number_whole=0; single_or_multiple=0; 
  } 
else if(c==WHOLE && output_number_whole==0 && single_or_multiple==1)
       { fprintf(stdout, "%s\n", current_info); single_or_multiple=0; } 
} /* end of if headfoot */ 
} /* end of if single_only==0 */ 
else { /* single_only==1 */ 
if(headfoot==1 && a>0){ sprintf(current_info, "<%s\tlength=%d\t#%s=%d\t", 
                  psn, sequence_length, outputs[c], a); a=0; 
         fprintf(stdout, "%s\n", current_info); }  
} /* end of single_only==1 */ 

} /* end of LONG_FORMAT */ 

return 0; 
} /* end of output() */ 


void inner_bias_filter(struct subsequence *kept, int table[][21])
{ 
int g,h,i,j,k; 
int mincheck, range, temp_end, temp_maximum, removed_position, halfseq; 
int current_count, search_start, search_end; 
int first_stored_for_window_size, previous_first_stored_for_window_size; 
double fraction_g, fraction_h, denom_g, denom_h; 
static int *pis; 
 
pis=iseq;
temp_end=removed_position=current_count=first_stored_for_window_size=0; 
previous_first_stored_for_window_size=0; 

search_end = kept->end + STEPSIZE;   
if(search_end>sequence_length-1) { search_end=sequence_length-1; }

search_start = kept->start - STEPSIZE; 
if(search_start<0) { search_start=0; } 

range = kept->end - kept->start + 1;  /* range is now measured as plus 1 */
if(range>sequence_length) { temp_maximum=sequence_length; } 
else { temp_maximum=range; } 
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
                if(second_stored[h].flag==0)
                  {
                  fraction_h = ((double)second_stored[h].rescount)/ denom_h;  
                  if(fraction_h>=fraction_g)
                    { second_stored[g].flag=1; break; } 
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

if(number_of_lps>maxnumlps) { maxnumlps=number_of_lps; }
} /* end of inner_bias_filter() */ 
/* * * * * * * * * * * * * * * * */ 


void outer_bias_filter(int table[][21], double freq[])
{
  int i,j,k,ii,jj,kk;
  int p=0,q=0,r=0,s=0,t=0,within=0, temp_maximum_window; 
  int residue_types_tally[21];  
  int residue_regions_tally[21]; 
  struct subsequence *kept; 

if(sequence_length<maximum_window) { temp_maximum_window = sequence_length; }
else { temp_maximum_window = maximum_window; }
memset(residue_regions_tally, 0, alphabet_length * sizeof(int));  
maxnumlps=maxthirdstored=0; 
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

        for(jj=STEPSIZE;jj<(sequence_length-ii-1);jj+=STEPSIZE)
           {
           /* update residue_type amounts */ 
           /* SHIFT_BY_STEPSIZE */ 
           residue_types_tally[ iseq[jj-3] ]--;  
           residue_types_tally[ iseq[jj-2] ]--; 
           residue_types_tally[ iseq[jj-1] ]--;   
           residue_types_tally[ iseq[ii+jj] ]++; 
           residue_types_tally[ iseq[ii+jj-1] ]++; 
           residue_types_tally[ iseq[ii+jj-2] ]++;

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
                               { 
                               stored[k][q].start = jj; 
                               stored[k][q].end = jj+ii; 
                               } /* end of if */ 
                             break;
                             } /* end of if */ 
                         } /* end for q */ 

                       if(within==0)
                         {
                         stored[k][residue_regions_tally[k]].start = jj; 
                         stored[k][residue_regions_tally[k]].end = jj+ii; 
                         stored[k][residue_regions_tally[k]].residue_types_tally = residue_types_tally[k]; 
                         residue_regions_tally[k]++; 
                         } /* end of if within==0 */ 
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
      further_back_number_of_lps=previous_number_of_lps=number_protein_stored=maxfirststored=0;  
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
              inner_bias_filter(kept, table); 

              if(number_of_lps==0) { continue; }
              else if(number_of_lps==1) 
                     { 
                     /* check ends */ 
                     while(second_stored[1].end<sequence_length-1) 
                          { second_stored[1].end++; 
                            if(iseq[second_stored[1].end]==rt) 
                              { second_stored[1].rescount++; } 
                            else { second_stored[1].end--; break; } } 
                     while(second_stored[1].start>0) 
                          { second_stored[1].start--; 
                            if(iseq[second_stored[1].start]==rt) 
                              { second_stored[1].rescount++; } 
                            else { second_stored[1].start++; break; } }

                     /* fill protein_stored array */ 
                     protein_stored[number_protein_stored].start = second_stored[1].start; 
                     protein_stored[number_protein_stored].end = second_stored[1].end;    
                     protein_stored[number_protein_stored].rescount = second_stored[1].rescount; 
                     protein_stored[number_protein_stored].restypes[0] =  rt ; 
                     protein_stored[number_protein_stored].numrestypes = 1; 
                     protein_stored[number_protein_stored].lp = 
logbinomial((second_stored[1].end-second_stored[1].start)+1,second_stored[1].rescount,freq[rt]); 
                     protein_stored[number_protein_stored].sumlp = protein_stored[number_protein_stored].lp; 
                     protein_stored[number_protein_stored].flag = 0; 

                     number_protein_stored++;
                     } /* end of if number_of_lps==1 */ 
              else { /* number_of_lps>1 */ 
                   for(i=1;i<=number_of_lps;i++)
                      { 
                      if(second_stored[i].flag==0)
                        {
                        /* check ends */ 
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

                        /* fill third_stored while calcing pvalue */ 
                        third_stored[number_third_stored].start = second_stored[i].start; 
                        third_stored[number_third_stored].end = second_stored[i].end;
                        third_stored[number_third_stored].rescount = second_stored[i].rescount; 
                        third_stored[number_third_stored].restypes[0] = rt ; 
                        third_stored[number_third_stored].numrestypes = 1; 
                        third_stored[number_third_stored].lp = 
     logbinomial((second_stored[i].end-second_stored[i].start)+1,second_stored[i].rescount,freq[rt]); 
                        third_stored[number_third_stored].sumlp = third_stored[number_third_stored].lp;
                        third_stored[number_third_stored].flag = 0; 

                        number_third_stored++;
                        } /* end of if second_stored[i].flag==0 */ 
                      } /* end for i */ 

                   if(number_third_stored==0) { continue; }
                   else if(number_third_stored==1)
                          { /*fill protein_stored */ 
                          protein_stored[number_protein_stored].start = third_stored[0].start; 
                          protein_stored[number_protein_stored].end = third_stored[0].end;
                          protein_stored[number_protein_stored].rescount = third_stored[0].rescount; 
                          protein_stored[number_protein_stored].restypes[0] = third_stored[0].restypes[0]; 
                          protein_stored[number_protein_stored].numrestypes = 1; 
                          protein_stored[number_protein_stored].lp = third_stored[0].lp; 
                          protein_stored[number_protein_stored].sumlp = third_stored[0].sumlp; 
                          protein_stored[number_protein_stored].flag = 0;  

                          number_protein_stored++; 
                          } /* end of if number_third_stored==1 */ 
                   else { /* number_third_stored>1 */ 

                        qsort(third_stored, number_third_stored, sizeof(struct lps), compare); 

                        /**** reduce list for overlap ****/ 
                        for(i=0;i<number_third_stored;i++)
                           { 
                           if(tsp[i]->flag==0)
                             { 
                             for(j=i+1;j<number_third_stored;j++)
                                { 
                                  if( tsp[j]->flag==0 && !(tsp[j]->start<tsp[i]->start &&
	                                     tsp[j]->end  <tsp[i]->start) 
                                    && !(tsp[i]->start<tsp[j]->start &&
	                                     tsp[i]->end  <tsp[j]->start) ) /* OVERLAP? */
                                    { tsp[j]->flag=1; 
                                    }  /* end of if flag j ==0  && OVERLAP */
                                }/* end of for j */ 
                             }/* end of if flag i ==0 */ 
                           }/* end of for i */ 

                           for(i=0;i<number_third_stored;i++)
                              {
                              if(tsp[i]->flag==0) 
                                { 
                                protein_stored[number_protein_stored] = *tsp[i] ; 
                                number_protein_stored++; 
                                } /* end of if */ 
                               }/* end of for i */
                           /* END OF REDUCE FOR OVERLAP */ 
                        } /* end of if number_third_stored>1 */ 
                   } /* end of if number_of_lps>1 */
              if(number_third_stored>maxthirdstored) { maxthirdstored=number_third_stored; }
              } /* end of if good first_stored */ 
    } /* end for r */ 
    } /* end for p */ 

/* sort, output, merge, output */ 
if(number_protein_stored>1)
  { qsort(protein_stored, number_protein_stored, sizeof(struct lps), compare); } 

if(single_only==0)
{
if(format!=MASKED)
  { if(number_protein_stored>0)
      { op=SINGLE; 
        output(op, freq); 
        maxrespos=0; 
        op=MULTIPLE; 
        merge(freq); 
        output(op, freq); } 
    op=WHOLE; 
    output(op, freq); } 
else { op=SINGLE; 
       output(op, freq); 
       maxrespos=0; 
       op=MULTIPLE; 
       merge(freq); 
       output(op, freq); }
} /* end of if single_only==0 */ 
else { /* single_only==1 */ 
if(format!=MASKED)
  { if(number_protein_stored>0)
      { op=SINGLE; 
        output(op, freq); 
        maxrespos=0; 
      } 
  } 
else { op=SINGLE; 
       output(op, freq); 
       maxrespos=0; 
     }
} /* end of else single_only==1 */ 

number_kept=s; 
} /* end of outer_bias_filter() */ 


int main(argc,argv)
int argc; 
char **argv; 
  { 
  int i,j; 
  char sa[250],sb[MAXIMUM_LENGTH],se[250],sh[50]; 

  /* files */ 
  char infile[250], logfile[250], masked_output[250]; 
  FILE *f2,*f3; 

  /* for analyzing sequences */ 
  int first_name_line=0, test_length, current_length; 

  /* for read in composition */ 
  double tempcomp, temp_pvalue; 
  char residue; 

  double composition[21] = { 
   0.0784, 0.0160, 0.0556, 0.0650, 0.0391, 0.0663, 0.0218,
   0.0493, 0.0618, 0.0920, 0.0218, 0.0431, 0.0467, 0.0433,
   0.0479, 0.0721, 0.0620, 0.0649, 0.0149, 0.0005, 0.0377
   }; /* re-sets to user composition if that is the option */ 

  double equal_composition[21] = { 
   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 
   }; 

  /* for options */ 
  int c;
  int errflg=0;
  extern char *optarg;
  extern int optind, optopt; 
  int run_id=0; 
  char scale_file[250], user_array[250] = {"domains"}; 
  char optionstring[1000]; 
  bool verbose=0; 

  char startrealtime[30]; 
  time_t t; 

  strcpy(startrealtime,__TIME__); 
  srand(time(&t)); 

  getrusage(RUSAGE_SELF, &measureusage); 
  currentu = measureusage.ru_utime; 
  currents = measureusage.ru_stime; 
  startu = currentu; 
  starts = currents; 
  previouss = starts; 
  previousu = startu; 

  bkgd=DOMAINS; 
  format=SHORT_FORMAT; 
  optionstring[0]='\0'; 


/*  *  *  *  PROCESS COMMAND-LINE OPTIONS  *  *  *  */ 
while((c = getopt(argc, argv, "hvdso:m:M:t:c:")) != -1) {
     switch(c) {
     case 'h': print_help(); exit(0);   
     case 'v': verbose=1; break;
     case 'd': headfoot=1; break; 
     case 's': single_only=1; break; 
     case 'o': if(!strcmp(optarg, "long")) { format=LONG_FORMAT; } 
               else if(!strcmp(optarg, "short")) { format=SHORT_FORMAT; } 
               else if(!strcmp(optarg, "masked")) { format=MASKED; } 
               else { fprintf(stderr, " -o value is not valid, using default short format\n"); 
                      format=SHORT_FORMAT; }
               break; 
     case 'm': sscanf(optarg,"%d", &minimum_window); break; 
     case 'M': sscanf(optarg,"%d", &maximum_window); break; 
     case 't': sscanf(optarg,"%lf", &pvalue_threshold); 
               pvalue_threshold = log10(pvalue_threshold); 
               break; 
     case 'c': strcpy(user_array,optarg); 
               if(!strcmp(user_array, "equal")) { bkgd=EQUAL; } 
               else if(!strcmp(user_array, "domains")) { bkgd=DOMAINS; }
               else { bkgd=USER; } 
               break; 
     case ':': fprintf(stderr, "option -%c requires a value\n", optopt); errflg++; break;
     case '?': fprintf(stderr, "unrecognized option: -%c\n", optopt); errflg++;
} /* end of switch */ 
} /* end of while() getopt */ 
if (errflg) { print_help(); exit(1); }

for(i=0;i<argc;i++)
   { strcat(optionstring, " "); strcat(optionstring, argv[i]); } 

if(minimum_window>ABSOLUTEMAX || minimum_window<ABSOLUTEMIN) 
  { minimum_window=FIXED_MIN_WINDOW; }

if(maximum_window>ABSOLUTEMAX || maximum_window<ABSOLUTEMIN) 
  { maximum_window=FIXED_MAX_WINDOW; } 

if(minimum_window>maximum_window) 
  { minimum_window=FIXED_MIN_WINDOW; maximum_window=FIXED_MAX_WINDOW; }

if(pvalue_threshold>BASELINE_LOG_P_VALUE) { pvalue_threshold=BASELINE_LOG_P_VALUE; } 
if(pvalue_threshold==BASELINE_LOG_P_VALUE && format==MASKED)
  { pvalue_threshold=DEFAULT_MASKING_LPVALUE; }

if(format==LONG_FORMAT) 
  { if(minimum_window<CORE_WINDOW) { core_length=minimum_window; } }

/*  *  *  *  PROCESS COMPOSITION FILE, IF ANY  *  *  *  */ 
  if(bkgd==USER)
    {
    if((f3 = fopen(user_array, "r"))==NULL)   
      { fprintf(stderr, "Specified composition file is not there.\n");
        exit(1); }
    else { /* if open composition file successful */ 
         while(fgets(sh,49,f3)!=NULL)
              { 
              sscanf(sh,"%c %lf", &residue, &tempcomp); 
              for(i=0;i<21;i++) 
              { if(alphabet[i]==residue) { composition[i]=tempcomp; break; } } 
              } /* end of while */ 
         if(!feof(f3)){ fprintf(stderr, 
            "error reading user composition file, check format and try again...\n"); 
                        exit(1); } 
         calculate_threshold_table(composition);
         } /* end of if open composition file successful */ 
    } /* end of if bkgd USER  */ 


/*  *  *  *  OUTPUT OPTIONS/PARAMETERS, IF VERBOSE  *  *  *  */ 
if(verbose==1)
{
fprintf(stderr, " %s  %s %s\n", argv[0], __DATE__, startrealtime); 
fprintf(stderr, "options/parameters: \n"); 
if(format==LONG_FORMAT) { fprintf(stderr, "output long format\n"); } 
else if(format==SHORT_FORMAT) { fprintf(stderr, "output short format\n"); } 
else if(format==MASKED) { fprintf(stderr, "output masked\n"); } 

fprintf(stderr, "minimum_window=%d\nmaximum_window=%d\n", minimum_window, maximum_window); 
fprintf(stderr, "pvalue_threshold=%le\n", pow(10.0,pvalue_threshold)); 

if(bkgd==USER) { fprintf(stderr, "bkgd composition = %s\n", user_array); } 

else if(bkgd==EQUAL) { fprintf(stderr, "bkgd composition = equal (0.05)\n"); }
else /*DOMAINS*/ { fprintf(stderr, "bkgd composition = domains\n"); }

fprintf(stderr, "if parameter values were out of bounds, they have been set to defaults\n\n"); 
} /* end of output parameters */ 

/* output user tables, if user_composition specified */ 
if(bkgd==USER && verbose==1)  
{
fprintf(stderr,"USER COMPOSITION:\n"); 
for(i=0;i<21;i++) { fprintf(stderr, "%c\t%lf\n", alphabet[i], composition[i]);} 

fprintf(stderr,"USER THRESHOLD TABLE\n"); 
for(i=0;i<MAXIMUMWINDOW;i++) { for(j=0;j<21;j++) { 
     fprintf(stderr, "%d,",default_threshold_table[i][j]); } 
     fprintf(stderr, "\n"); 
   } fprintf(stderr, "\n\n"); 
} /* end of if bkgd USER && verbose==1 */ 


/*  *  *  *  OPEN SEQUENCE FILE  *  *  *  */ 
if((f2 = fopen(argv[optind],"r"))==NULL)
    { fprintf(stderr, "There is no sequence file. Please supply one.\n");
      exit(1); }   
else { strcpy(infile, argv[optind]); }


/*  *  *  *  ALLOCATE MEMORY *  *  *  */ 
sequence = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 
cleanseq = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 
binseq = (bool *)calloc(MAXIMUM_LENGTH,sizeof(bool));
iseq = (int *) calloc(MAXIMUM_LENGTH,sizeof(int));
respos = (int *) calloc(MAXIMUM_LENGTH,sizeof(int)); 
tsp = (struct lps**) calloc(30*UNIT_STORE, sizeof(struct lps*));
psp = (struct lps**) calloc(2*UNIT_STORE, sizeof(struct lps*));

if(!sequence || !iseq || !tsp || !psp)
  { fprintf(stderr,
"memory allocation error, decrease MAXIMUM_LENGTH and"
" UNIT_STORE in code, and use a smaller maximum_window"
" (-M option)...exiting...\n"); 
    exit(1); }

for(i=0;i<30*UNIT_STORE;i++)
   { tsp[i] = &third_stored[i]; } 
for(i=0;i<2*UNIT_STORE;i++)
   { psp[i] = &protein_stored[i]; }   


/*  *  *  * INITIAL HEADER OUTPUT *  *  *  */ 
if(headfoot==1)
{
if(format==SHORT_FORMAT)
{
fprintf(stdout, "## short format\n##SEQUENCE\tBIAS_TYPE\tLPS#\tSTART\tEND\tRESIDUE_COUNT\t"
  "BINOMIALP\tSIGNATURE\n"); 
fprintf(stdout, "##\n##\n"); 
} /* end of format==SHORT_FORMAT */ 
else if(format==LONG_FORMAT) { 
fprintf(stdout, "## long format\n## footer begins < and lists numbers of each bias type\n##\n"); 
fprintf(stdout, "##SEQUENCE_NAME\tBIAS_TYPE\tLPS#\tSTART\tEND\tRESIDUE_COUNT\t"
  "BINOMIALP\tSIGNATURE\tSUMLOGP\tCORE_START\tCORE_END\tCORE_SEQUENCE\tNTERM_CONTEXT\tLPS_SEQUENCE\tCTERM_CONTEXT\n"); 
} /* end of format==LONG_FORMAT */ 
else { /* MASKED */ 
fprintf(stdout, "## FASTA format with masked residues = 'X'\n## LPSs in > line in format:"
  " start {bias_signature} end = binomial_P\n"); 
fprintf(stdout, "##\n##\n");
} /* end of MASKED */ 
} /* end of if headfoot */ 


/*  *  *  *  ANALYZE SEQUENCES *  *  *  */ 
sequence_name[0]='\0'; 
sequence[0]='\0'; 
while(fgets(sb,MAXIMUM_LENGTH-1,f2)!=NULL)
{
/***READ sequence_name and sequence ***/ 
if(strncmp(sb,">",1)==0)
  {
  if(first_name_line==1)
    { 
    sequence_length = strlen(sequence); 
    convert_sequence(); 
    if(verbose==1) 
      { fprintf(stderr, "analyzing %s  %d...\n", sequence_name, number_of_sequences); }
    if(bkgd==EQUAL)
      { outer_bias_filter(equal_threshold_table, equal_composition); }
    else { outer_bias_filter(default_threshold_table, composition); }
    sequence[0]='\0';   
    sequence_name[0]='\0';  
    } /* end of if first_name_line==1 */ 
  else { first_name_line=1; }
  sscanf(sb,"%s",se);
  strcpy(sequence_name,se+1);
  number_of_sequences++; 
  }/* end of if strncmp(sb,">",1)==0 */ 
else { /* sequence line */ 
     test_length=strlen(sb);
     current_length=strlen(sequence); 
     if(test_length+current_length > MAXIMUM_LENGTH)
       { strncat(sequence,sb,MAXIMUM_LENGTH-current_length-1); 
         if(verbose==1) { fprintf(stderr, 
"length of sequence %s is too large, only the first 20,000 residues will be analyzed\n",
         sequence_name); } /* end of if verbose */ 
       } /* end of if test_length+strlen(sequence) > MAXIMUM_LENGTH */ 
     else{ strcat(sequence,sb); }
     } /* end of else sequence line */ 
} /* end of while feof(f2) */ 
if(feof(f2))
  { /* handle last sequence */ 
  sequence_length = strlen(sequence); 
  convert_sequence(); 
  if(verbose==1) 
    { fprintf(stderr, "analyzing %s  %d...\n", sequence_name, number_of_sequences); }
  if(bkgd==EQUAL)
    { outer_bias_filter(equal_threshold_table, equal_composition); }
  else { outer_bias_filter(default_threshold_table, composition); }
  if(verbose==1) { fprintf(stderr, "finished analysis of database.\n"); } 
  }
else { fprintf(stderr, "error in database reading, exiting ...\n"); exit(1); }

exit(0); 
}/******** end of main() ********/ 

/********* END OF CODE FILE *********/ 
