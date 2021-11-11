/**** 
 **** 
 **** DomainFilter.c 
 **** 
 ****
 **** This program is part of the fLPS package version 2.0 
 ****
 ****/ 
/****  Copyright 2017, 2021. Paul Martin Harrison. ****/ 
/****
 ****  Licensed under the 3-clause BSD license. See LICENSE.txt bundled in the fLPS package. 
 ****/ 
/****  
 ****  This programs filters a FASTA-format sequence file with user-defined lists of domains and outputs the 
 **** sequences in either of two formats, in which the domains have either been excised or masked. 
 **** 
 ****  to compile in ./src: 
 ****   gcc -O2 -march=native -o DomainFilter DomainFilter.c -lm 
 ****    OR 
 ****   make 
 ****
 ****  The header file "flps_def.h" is required 
 ****
 ****  to run and get help: 
 ****   ./DomainFilter -h 
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

/* globals */ 
int verbose, header, sequence_length=MAXIMUM_LENGTH, number_of_sequences; 
bool too_long; 
enum results {EXCISED, MASKED} format; 
enum molecule {AA, NT} sequence_type;  
char masker[3] = {"XN"}, sequence_name[250] = {"initial"}; 
char *sequence, *cleanseq, *modseq; 
char line[MAXIMUM_LENGTH+1], nline[MAXIMUM_LENGTH+1], sn[MAXIMUM_LENGTH+1]; 
char current_info[(MAXIMUM_DOMAINS+1)*25], infile[250]; 
char info_append[(MAXIMUM_DOMAINS+1)*25]; 
float version=VERSION_NUMBER;
FILE *f2; 

/* for domain reading */ 
int number_of_domains, domain_start[MAXIMUM_DOMAINS], domain_end[MAXIMUM_DOMAINS]; 
int excision_point[MAXIMUM_DOMAINS][2], total_domain_length; 
char domain_name[MAXIMUM_DOMAINS][MAX_DOM_NAME_SIZE+1], test_name[MAXIMUM_LENGTH+1]; 


void print_help() 
{ fprintf(stdout, "\nDomainFilter, part of fLPS package version %.1f\n", version);   
  fprintf(stdout,   "==============================================\n\n" 
"This programs filters a FASTA-format sequence file with user-defined lists of domains and outputs the \n" 
"sequences in either of two formats, in which the domains have either been excised or masked.\n\n"
"The format specification for the domain list is in the 'README.first'\n"
"The domains should be sorted in numerical order and not overlap each other.\n"
"The program options are:\n"
" -h   prints help\n\n"
" -v   prints runtime information\n\n"
" -d   prints header information in the output files\n\n"    
" -D   output specified with: 'excised' (default), or 'masked'\n\n" 
" -O   ...FILENAME_PREFIX... allows for output to be sent to a unique file prefixed with FILENAME_PREFIX\n\n"
" -n   specifies DNA sequence (only necessary for the '-o masked' option, the default is protein)\n\n"
"This program is part of the fLPS package version 2.0 .\n"
"CITATIONS:\n Harrison, PM. 'fLPS: fast discovery of compositional biases for the protein universe',\n"
" (2017) BMC Bioinformatics, 18: 476. \n" 
" Harrison, PM. 'fLPS 2.0: rapid annotation of compositional biases in biological sequences', submitted.\n"
"URLs:\n http://biology.mcgill.ca/faculty/harrison/flps.html\n OR \n https://github.com/pmharrison/flps\n"); 
} /* end of print_help */ 


void output() 
{
int i,j,a,b; 
char *psn=sequence_name, *pms=modseq, *pcs=cleanseq; 

if(format==EXCISED)
{
if(number_of_domains==0) { pms=cleanseq; }

sprintf(current_info, "%s %d ", psn, number_of_domains); 

for(i=1,a=0;i<=number_of_domains;i++) 
   { sprintf(info_append, "%d,%d %s ", excision_point[i][0]+1, excision_point[i][1], domain_name[i]); 
     strcat(current_info, info_append); a++; } 

fprintf(f2, ">%s\n", current_info); 

for(i=0;i<=sequence_length-OUTPUT_LINE_LENGTH;i+=OUTPUT_LINE_LENGTH)
   { fprintf(f2,"%.*s\n", OUTPUT_LINE_LENGTH, pms+i); 
   } fprintf(f2,"%.*s\n", sequence_length-i, pms+i); 
} /* end of format EXCISED */ 

else { /* MASKED */ 
fputs(nline, f2); 

for(i=1,a=0;i<=number_of_domains;i++) 
   { for(j=domain_start[i];j<=domain_end[i];j++) 
        { *(pcs+j) = masker[sequence_type]; } 
   } /* end for i*/ 

for(i=0;i<=sequence_length-OUTPUT_LINE_LENGTH;i+=OUTPUT_LINE_LENGTH) 
   { fprintf(f2,"%.*s\n", OUTPUT_LINE_LENGTH, pcs+i); 
   } fprintf(f2,"%.*s\n", sequence_length-i, pcs+i); 
} /* end of format MASKED */ 

}/* end of output() */ 


void modify_sequence()
{
int i,j,k; 
char *pcs, *pms; 

/* use domain starts and ends to make the modified 
   sequence from subsequences; also make list of excisions (points and lengths) */  
pcs=cleanseq; pms=modseq; 

modseq[0]='\0'; 
if(domain_start[1]>0){ strncpy(modseq,cleanseq,domain_start[1]); } 
modseq[domain_start[1]]='\0'; 

domain_start[number_of_domains+1] = sequence_length-1; 
domain_end[number_of_domains+1] = 0; 

total_domain_length=0; 
for(i=1;i<number_of_domains;i++) 
   { 
   excision_point[i][0] = domain_start[i] - total_domain_length; 
   excision_point[i][1] = domain_end[i]-domain_start[i]+1; 
   total_domain_length += excision_point[i][1]; 

   strncat(pms,pcs+domain_end[i]+1,domain_start[i+1]-domain_end[i]-1); 
   } /* end for i */ 

/* last one is for the sequence stretch beyond the last domain */ 
   excision_point[number_of_domains][0] = domain_start[number_of_domains] - total_domain_length; 
   excision_point[number_of_domains][1] = domain_end[number_of_domains]-domain_start[number_of_domains]+1; 
   total_domain_length += excision_point[number_of_domains][1];  

   strncat(pms,pcs+domain_end[number_of_domains]+1,domain_start[number_of_domains+1]-domain_end[number_of_domains]); 

sequence_length -= total_domain_length; 
} /* end of modify_sequence() */ 


void read_domains(char *nameline)
{ 
int i; 
char format[8*(MAXIMUM_DOMAINS+1)+8]="%*s %*s %s %s ", *position; 
char shift_format[8*(MAXIMUM_DOMAINS+1)+8]="%*s %*s ", interval[25], token[2] = "-"; 

/* read number of domains */ 
if(sscanf(nameline,"%*s %d",&number_of_domains)==1) 
  {
  if(number_of_domains>MAXIMUM_DOMAINS) { 
    fprintf(stderr, 
     "Sequence %s has too many domains to read (%d) for the -D option. Only the first %d will be read.\n\n", 
       sequence_name, number_of_domains, MAXIMUM_DOMAINS); number_of_domains=MAXIMUM_DOMAINS; }

/* read the domain intervals progressively and split them using strtok() */ 
for(i=1;i<=number_of_domains;i++) 
{ domain_name[i][0]='\0'; test_name[0]='\0'; 
sscanf(nameline,format,interval,test_name); strncpy(domain_name[i],test_name,MAX_DOM_NAME_SIZE); 
strcat(shift_format,format); strcpy(format,shift_format); strcpy(shift_format, "%*s %*s "); 
position = strtok(interval,token); sscanf(position,"%d",&domain_start[i]); 
position = strtok(0,token); sscanf(position,"%d",&domain_end[i]); 
domain_start[i]--; 
domain_end[i]--; 
} /* end for i */ 

} /* end of if sscanf for number */ 
else { number_of_domains=0; } 

} /* end of read_domains() */ 


void clean_sequence()
{
int a,aa; 
char *ps, *pcs; 

ps=sequence; pcs=cleanseq; 

for(a=0; ps<sequence+sequence_length; ps++)
    { aa=toupper(*ps); 
      switch(aa) {
          case 65 /*'A'*/: *(pcs+a) = aa; a++; break; 
          case 67 /*'C'*/: *(pcs+a) = aa; a++; break;
          case 68 /*'D'*/: *(pcs+a) = aa; a++; break; 
          case 69 /*'E'*/: *(pcs+a) = aa; a++; break;
          case 70 /*'F'*/: *(pcs+a) = aa; a++; break; 
          case 71 /*'G'*/: *(pcs+a) = aa; a++; break;
          case 72 /*'H'*/: *(pcs+a) = aa; a++; break; 
          case 73 /*'I'*/: *(pcs+a) = aa; a++; break;
          case 75 /*'K'*/: *(pcs+a) = aa; a++; break; 
          case 76 /*'L'*/: *(pcs+a) = aa; a++; break;
          case 77 /*'M'*/: *(pcs+a) = aa; a++; break; 
          case 78 /*'N'*/: *(pcs+a) = aa; a++; break;
          case 80 /*'P'*/: *(pcs+a) = aa; a++; break; 
          case 81 /*'Q'*/: *(pcs+a) = aa; a++; break;
          case 82 /*'R'*/: *(pcs+a) = aa; a++; break; 
          case 83 /*'S'*/: *(pcs+a) = aa; a++; break;
          case 84 /*'T'*/: *(pcs+a) = aa; a++; break; 
          case 86 /*'V'*/: *(pcs+a) = aa; a++; break;
          case 87 /*'W'*/: *(pcs+a) = aa; a++; break; 
          case 89 /*'Y'*/: *(pcs+a) = aa; a++; break;
          case 88 /*'X'*/: *(pcs+a) = aa; a++; break; } } 
sequence_length=a; 
} /* end of clean_sequence() */ 


void read_sequence_line(char *sb)
{
int test_length, current_length; 

test_length=strlen(sb);
current_length=strlen(sequence); 
if(test_length+current_length > MAXIMUM_LENGTH)
  { strncat(sequence,sb,MAXIMUM_LENGTH-current_length-1); 
         if(verbose && !too_long) { fprintf(stderr, 
"length of sequence %s is too large, only the first %d residues will be analyzed\n",
         sequence_name, MAXIMUM_LENGTH); } /* end of if verbose */ 
    too_long=1; 
  } /* end of if test_length+strlen(sequence) > MAXIMUM_LENGTH */ 
else{ strcat(sequence,sb); }
} /* end of read_sequence_line() */ 


void analyse(FILE *fi)
{
bool past_first_name_line=0; 

nline[0]='\0'; 

while(fgets(line,MAXIMUM_LENGTH-1,fi)!=NULL)
{
/*** READ sequence_name and sequence ***/ 
if(!strncmp(line,">",1))
  {
  if(past_first_name_line)
    { 
    sequence_length = strlen(sequence); 
    if(verbose) { fprintf(stderr, "processing %s  #%d sequence...\n", sequence_name, number_of_sequences); }
    clean_sequence(); 
    if(number_of_domains>0 && format==EXCISED) { modify_sequence(); }
    output(); 
    sequence[0]='\0'; modseq[0]='\0'; sequence_name[0]='\0'; 
    } /* end of if past_first_name_line */ 
  else { past_first_name_line=1; } 
  strcpy(nline, line); 
  sscanf(line,"%s",sn); strcpy(sequence_name,sn+1); 
  read_domains(line); 
  number_of_sequences++; 
  }/* end of if !strncmp(line,">",1) */ 
else { read_sequence_line(line); } 
} /* end of while fgets(line,MAXIMUM_LENGTH-1,f1)!=NULL */ 
if(feof(fi))
  { /* handle last sequence */ 
  sequence_length = strlen(sequence); 
  if(verbose) { fprintf(stderr, "processing %s  #%d (last) sequence...\n", sequence_name, number_of_sequences); }
  clean_sequence(); 
  if(number_of_domains>0 && format==EXCISED) { modify_sequence(); }
  output(); 
  if(verbose) { fprintf(stderr, "Finished processing database %s.\n", infile); } 
  } else { fprintf(stderr, "error in reading database %s, exiting ...\n", infile); exit(1); }
} /* end of analyse_aa() */ 


int file_exist(char *filename) { struct stat sa; return (stat (filename, &sa) == 0); } 


int main(int argc, char **argv)
{
FILE *f1; 

int i, c, oop=0, errflg=0, opened=0, run_id=0;
extern char *optarg;
extern int optind, optopt; 
char outform[50]="excised", outfile[MAX_FILE_NAME_SIZE], prefix[MAX_FILE_NAME_SIZE]; 
char startrealtime[30]; 
time_t t; 

strcpy(startrealtime,__TIME__); 
srand(time(&t)); 

sequence_type=AA; 

/*  *  *  *  PROCESS COMMAND-LINE OPTIONS  *  *  *  */ 
while((c = getopt(argc, argv, "hvndD:O:")) != -1) { 
     switch(c) {
     case 'h': print_help(); exit(0);   
     case 'v': verbose=1; break;
     case 'n': sequence_type=NT; break; 
     case 'd': header=1; break; 
     case 'D': if(!strcmp(optarg, "excised")) { format=EXCISED; } 
               else if(!strcmp(optarg, "masked")) { format=MASKED; strcpy(outform,"masked"); } 
               else { fprintf(stderr, " -D value is not valid, using default 'excised' output format\n"); 
                      format=EXCISED; }
               break; 
     case 'O': oop=1; strcpy(prefix,optarg); break; 
     case ':': fprintf(stderr, "option -%c requires a value\n", optopt); errflg++; break; 
     case '?': fprintf(stderr, "unrecognized option: -%c\n", optopt); errflg++; 
} /* end of switch */ 
} /* end of while() getopt */ 
if (errflg) { print_help(); exit(1); } 

/*  *  *  *  OPEN SEQUENCE FILE  *  *  *  */ 
if((f1 = fopen(argv[optind],"r"))==NULL)
    { fprintf(stderr, "There is no sequence file. Please supply one in FASTA format.\n");
      exit(1); }   
else { strcpy(infile, argv[optind]); } 

/*  *  *  *  HANDLE OUTPUT FILE STREAM, IF -O OPTION  *  *  *  */ 
if(oop) { opened=0;  
while(!opened) { run_id = (int) floor( rand()/100000);  
                 sprintf(outfile,"%s.%d.%s.fasta", prefix, run_id, outform); 
                 if(!file_exist(outfile)) { f2=fopen(outfile, "w"); opened=1; }  
               } /*end of while !opened*/ 
} /* end if oop */ 
else { f2 = stdout; } 

/*  *  *  *  ALLOCATE MEMORY *  *  *  */ 
sequence = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 
cleanseq = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 
modseq = (char *)calloc(MAXIMUM_LENGTH,sizeof(char)); 

/*  *  *  * INITIAL VERBOSE OUTPUT *  *  *  */ 
if(verbose)
{
fprintf(stderr, "%s %s %s\n", argv[0], __DATE__, startrealtime); 
for(i=0;i<argc;i++) { fprintf(stderr, "%s ", argv[i]); } 
fprintf(stderr, "\nUsing domain-filter part of the fLPS version %.1lf package\n\n"
  "Options/parameters:\n------------------- \n", VERSION_NUMBER);
if(format==MASKED)       { fprintf(stderr, "Output sequences with domains masked by letter 'X' if protein and 'N' if DNA (Option -o masked)\n"); } 
else { fprintf(stderr, "Output sequences with domains excised (Option -o excised)\n"); } 
if(header) { fprintf(stderr, "Headers will appear at the top of output files (Option -d).\n"); } 
} /* end of initial verbose output */ 

/*  *  *  * INITIAL HEADER OUTPUT *  *  *  */ 
if(header)
{
if(format==MASKED)
{ fprintf(f2, "## FASTA format with user-listed domains masked (with the letter 'X')\n"
       "## The domains are listed on the > line, thus: number of domains followed by a space-delimited list in the format 'x-y name', where x is the\n"
       "## start residue and y is the end residue and 'name' is the domain name"); }
else { /* EXCISED */ fprintf(f2, "## excised format:  sequence file in FASTA format, with excised domains listed after the name on the > line, thus:\n"
  "## number of domains followed by a space-delimited list of domain excision points 'x,y name' where x is the residue after the excision\n"
  "## and y is the number of residues excised, and 'name' is the domain name\n"); } 
fprintf(f2,"##\n##\n"); 
} /* end of if header */ 

/*  *  *  *  ANALYSE *  *  *  */ 
analyse(f1); 

exit(0); 
} /* end of main() */ 


