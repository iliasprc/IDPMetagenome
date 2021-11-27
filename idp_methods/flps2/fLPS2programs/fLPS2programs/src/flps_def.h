
/*****************************************************************************************/
/****  flps_def.h 
 ****
 ****  This header is part of the fLPS package version 2.0 
 ****/ 
/****  Copyright 2017, 2021. Paul Martin Harrison. ****/ 
/****
 ****  Licensed under the 3-clause BSD license. See LICENSE.txt bundled in the fLPS package. 
 ****/ 
/****  
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

#define MAXIMUM_LENGTH 100000
#define MAXIMUM_DOMAINS 200 
#define FIXED_MIN_WINDOW 15 
#define FIXED_MAX_WINDOW 500 
#define ABSOLUTEMIN 5 
#define ABSOLUTEMAX 1000 
#define DEFAULT_UNIT_STORE 10000 
#define MAXIMUMWINDOW 1001 
#define DEFAULT_BASELINE_P_VALUE 0.001 
#define DEFAULT_BASELINE_LOG_P_VALUE -3.0 
#define DEFAULT_MASKING_LOG_PVALUE -5.0 /* change to use different default masking threshold */ 
#define CORE_WINDOW 15 
#define CORE_OFFSET 7 
#define OUTPUT_LINE_LENGTH 60 
#define DEFAULT_STEPSIZE 3 
#define AA_ALPHABET_SIZE 21 
#define NT_ALPHABET_SIZE 5 
#define VERSION_NUMBER 2.0 
#define MAX_SEQUENCE_NAME_SIZE 250 
#define MAX_FILE_NAME_SIZE 500 
#define MAX_DOM_NAME_SIZE 50
#define MAX_LONG_FORMAT_LINE_LENGTH 101000 
#define MAX_FILESIZE4CCC 150000000000 
#define PLACEHOLDER -11.1111 

#define NUMBER_OF_AA_CLASSES 17 /* physico-chemical classes of amino acids */ 
#define AMIDE 0 
#define GLX 1 
#define ASX 2 
#define TINY_POLAR 3 
#define TINY_HYDROPHOBIC 4 
#define POLAR_AROMATIC 5 
#define NEGATIVE 6 
#define SMALL_POLAR 7 
#define POSITIVE 8 
#define SMALL_HYDROPHOBIC 9 
#define ALIPHATIC 10 
#define AROMATIC 11 
#define CHARGED 12 
#define TINY 13 
#define SMALL 14 
#define POLAR 15 
#define HYDROPHOBIC 16 

/* #define DEVELOP 1 */ 

