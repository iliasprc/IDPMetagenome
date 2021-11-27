/**** 
 **** CompositionMaker.c 
 **** 
 **** This program is part of the fLPS package version 2.0 
 ****
 ****/ 
/****  Copyright 2017, 2021. Paul Martin Harrison. ****/ 
/****
 ****  Licensed under the 3-clause BSD license. See LICENSE.txt bundled in the fLPS package. 
 ****/ 
/****  
 ****  This programs calculate the residue composition of a protein or DNA file.  
 **** 
 ****  to compile: 
 ****   gcc -O2 -march=native -o CompositionMaker CompositionMaker.c -lm 
 ****    OR 
 ****   make 
 ****
 ****  The header file "flps_def.h" is required 
 ****
 ****  to run and get help: 
 ****   ./CompositionMaker -h 
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

enum molecule {AA, NT} sequence_type;  
char line[MAXIMUM_LENGTH+1], alphabet[AA_ALPHABET_SIZE+1] = {"ACDEFGHIKLMNPQRSTVWXY"}, nt_alphabet[NT_ALPHABET_SIZE] = {"ACGNT"}; 
bool verbose, complete=1; 
int divisor=1; 


void print_help()
{
fprintf(stderr, "\ncomposition program , part of fLPS package version %.1f\n", VERSION_NUMBER);   
fprintf(stderr,   "=======================================================\n\n" 
"This programs calculates the residue composition of a FASTA-format sequence file.\n\nThe program options are:\n"
" -h   prints help\n"
" -v   prints runtime information\n" 
" -s   use a sample of the file (applies only to very large files greater than the current size of the TrEMBL database, 2021);\n" 
"      the file size is sampled to reduce it to the maximum allowed for this option;\n"
"      by default, the complete file is used.\n"
" -n   specifies DNA sequence (default is protein)\n" 
" This is an example of running the program is:\n"
"        ./composition -vn input.fasta > results.out\n\n"
" Here, verbose information is printed while running, and the input file is a DNA file.\n\n"
" The residue composition is output to a file named from the input file with the suffix '.COMPOSITION'.\n\n"
"CITATION:\n Harrison, PM. 'fLPS: fast discovery of compositional biases for the protein universe',\n"
" (2017) BMC Bioinformatics, 18: 476. \n" 
"URLs:\n http://biology.mcgill.ca/faculty/harrison/flps.html\n OR \nhttps://github.com/pmharrison/flps\n"); 
} /* end of print_help() */ 


double * calculate_composition(FILE *fi, FILE *fo, char *in, char *out)
{
int i, c=0; 
unsigned long total=0, count[AA_ALPHABET_SIZE];  
double *freq; 

freq = (double *)calloc(AA_ALPHABET_SIZE,sizeof(double)); 
memset(count, 0, AA_ALPHABET_SIZE * sizeof(unsigned long)); 

/* open composition output file */ 
strcpy(out,in); sprintf(out, "%s.COMPOSITION", in); fo=fopen(out,"w"); 

if(sequence_type==NT) { 
while(fgets(line,MAXIMUM_LENGTH-1,fi)!=NULL)
{
if(strncmp(line,">",1) && strcmp(line,""))
  { 
  c++; 
  if(c%divisor==0)
    { for(i=0; i<strlen(line); i++) 
         { switch(toupper(line[i])) { 
               case 65 /*'A'*/: count[0]++; break; 
               case 67 /*'C'*/: count[1]++; break;
               case 71 /*'G'*/: count[2]++; break;
               case 78 /*'N'*/: count[3]++; break;
               case 84 /*'T'*/: count[4]++; break; } } 
    } /* end of if c%div==0 */ 
  } /* end of if sequence line */  
} /* end of while */ 
for(i=0;i<NT_ALPHABET_SIZE;i++) { total+=count[i]; } 
for(i=0;i<NT_ALPHABET_SIZE;i++) { freq[i] = ((double) count[i])/((double) total); 
                                  fprintf(fo,"%c\t%.9lf\n", nt_alphabet[i], freq[i]); } 
} /*end of if NT*/

else { /*sequence_type==AA*/ 
while(fgets(line,MAXIMUM_LENGTH-1,fi)!=NULL)
{
if(strncmp(line,">",1) && strcmp(line,""))
  {
  c++; 
  if(c%divisor==0)
    { for(i=0; i<strlen(line); i++) 
         { switch(toupper(line[i])) { 
               case 65 /*'A'*/: count[0]++;  break; 
               case 67 /*'C'*/: count[1]++;  break;
               case 68 /*'D'*/: count[2]++;  break; 
               case 69 /*'E'*/: count[3]++;  break;
               case 70 /*'F'*/: count[4]++;  break; 
               case 71 /*'G'*/: count[5]++;  break;
               case 72 /*'H'*/: count[6]++;  break; 
               case 73 /*'I'*/: count[7]++;  break;
               case 75 /*'K'*/: count[8]++;  break; 
               case 76 /*'L'*/: count[9]++;  break;
               case 77 /*'M'*/: count[10]++;  break; 
               case 78 /*'N'*/: count[11]++;  break;
               case 80 /*'P'*/: count[12]++;  break; 
               case 81 /*'Q'*/: count[13]++;  break;
               case 82 /*'R'*/: count[14]++;  break; 
               case 83 /*'S'*/: count[15]++;  break;
               case 84 /*'T'*/: count[16]++;  break; 
               case 86 /*'V'*/: count[17]++;  break;
               case 87 /*'W'*/: count[18]++;  break; 
               case 89 /*'Y'*/: count[20]++;  break;
               case 88 /*'X'*/: count[19]++;  break; } }  
    } /* end of if c%div==0 */ 
  } /* end of if sequence line */  
} /* end of while */ 
for(i=0;i<AA_ALPHABET_SIZE;i++) { total+=count[i]; }
for(i=0;i<AA_ALPHABET_SIZE;i++) { freq[i] = ((long double) count[i])/((long double) total); 
                                  fprintf(fo,"%c\t%.9lf\n", alphabet[i], freq[i]); } 
} /* end of else sequence_type==AA*/ 
return freq; 
} /* end of calculate_composition() */ 


size_t getFilesize(const char* filename) {
struct stat st;
if(stat(filename, &st)) { return 0; }
return st.st_size;   
} /* end of getFilesize */ 


int main(int argc, char **argv)
{
FILE *fin /* input sequence file */, *fout /* output composition file */; 
char infile[MAX_FILE_NAME_SIZE]="", outfile[MAX_FILE_NAME_SIZE]=""; 
int i, c, errflg=0;
extern char *optarg;
extern int optind, optopt; 
size_t filesize; 

char startrealtime[30]; 
time_t t; 

strcpy(startrealtime,__TIME__); 
srand(time(&t)); 

sequence_type=AA; 

/*  *  *  *  PROCESS COMMAND-LINE OPTIONS  *  *  *  */ 
while((c = getopt(argc, argv, "hvns")) != -1) { 
     switch(c) {
     case 'h': print_help(); exit(0);   
     case 'v': verbose=1; break; 
     case 'n': sequence_type=NT; break; 
     case 's': complete=0; break; 
     case ':': fprintf(stderr, "option -%c requires a value...\n", optopt); errflg++; break; 
     case '?': fprintf(stderr, "unrecognized option: -%c ...\n", optopt); errflg++; 
} /* end of switch */ 
} /* end of while() getopt */ 
if (errflg) { print_help(); exit(1); } 

/*  *  *  * OPEN SEQUENCE FILE *  *  *  */ 
if((fin = fopen(argv[optind],"r"))==NULL)
    { fprintf(stderr, "There is no sequence file. Please supply one in FASTA format.\n");
      exit(1); }   
else { strcpy(infile, argv[optind]); } 

/* make divisor if !complete */ 
if(!complete)
{ filesize = getFilesize(infile); 
  if(filesize>MAX_FILESIZE4CCC) { divisor = (int) floor(filesize/MAX_FILESIZE4CCC); } }

/*  *  *  * INITIAL VERBOSE OUTPUT *  *  *  */ 
if(verbose)
{
fprintf(stderr, "%s %s %s\n", argv[0], __DATE__, startrealtime); 
for(i=0;i<argc;i++) { fprintf(stderr, "%s ", argv[i]); } 
fprintf(stderr, "\nUsing the composition program, part of the fLPS version %.1lf package\n\n", VERSION_NUMBER);
if(sequence_type==NT) { fprintf(stderr, "Sequence type is DNA, only the characters ACGTN will be counted...\n\n"); }
else { /* sequence_type==AA */ fprintf(stderr, 
       "Sequence type is protein, only the characters ACDEFGHIKLMNPQRSTVWXY will be counted...\n\n"); } 
if(complete) { fprintf(stderr, "The whole file will be used for the calculation.\n"); } 
else { fprintf(stderr, "Option -s was specified, \n");
       if(filesize>MAX_FILESIZE4CCC) 
         { fprintf(stderr, " filesize %zu is > than the maximum for complete calculation of residue composition,\n" 
           " so %lf of the file will be sampled\n\n", filesize, 1.0 / ((double) divisor)); } 
       else { fprintf(stderr, " but file does not exceed the maximum size above which file sampling occurs...\n"); }
      } /* end of else */ 
} /* end of initial verbose output */ 

/*  *  *  * ANALYSE COMPOSITION *  *  *  */
calculate_composition(fin, fout, infile, outfile);  

exit(0); 
} /* end of main() */ 

/******** END OF CODE FILE ********/ 

