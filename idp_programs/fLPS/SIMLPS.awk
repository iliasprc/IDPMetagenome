# 
# SIMLPS.awk
# 
# To pick out similar biases from fLPS output for a database
# 
# To use: 
#   awk -f SIMLPS.awk -v signature=... score=... permute=... INPUT > OUTPUT 
#  
# INPUT should be long-format output of fLPS.  
# The signature should be specified in the style "{xyz}". 
# The score is a SUMLPS score ($8 of the fLPS long format). 
# The flag permute=0 for an exact match and =1 to allow permutation of any of the  
# subsidiary biases (for a signature {xyz}, x is the main bias and yz are the   
# subsidiary biases). 
# 
# OUTPUT is a long-format list of LPSs that match the query signature, but with an 
# additional $1 field for ∆SUMLPS = abs(SUMLPSquery - SUMLPSmatch).  
# The matches can be sorted on ∆SUMLPS, e.g. using the unix sort command: 
#   sort -n -k 1 ...
# 

BEGIN{
first = substr(signature,2,1); 
rest = substr(signature,3, length(signature)-3); 
} # end of BEGIN 

{
if(permute==0)
  { 
  if( match($8,signature)!=0 && length($8)==length(signature))
    {
       if((score-$9)<0){ print 0-score+$9 "\t" $0; }  
       else { print score-$9 "\t" $0; }
    } # end of if 
  }
else { # permute==1 
   if( substr($8,2,1)== first && match($8,"[" rest "]{" length(rest) "}")!=0 && length($8)==length(signature))
       {
       if((score-$9)<0){ print 0-score+$9 "\t" $0; }  
       else { print score-$9 "\t" $0; }
       } # end of if 
     } 
} 

