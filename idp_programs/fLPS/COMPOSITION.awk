#
# COMPOSITION.awk 
# 
# To make user amino-acid composition file from database 
# 
# To use: 
#  awk -f COMPOSITION.awk -v sample=... INPUT > OUTPUT 
# 
#  INPUT is a database in FASTA format. 
# 
#  The flag sample should be a decimal in the range 0.0-1.0, 
#  e.g. 0.1 to sample one tenth of the database. 
#  The default is 1.0. 
# 
#  The OUTPUT composition file can be used by the fLPS 
#  program, with the -c option. 
# 

BEGIN { 
FS=""; 

if(sample<=0.0||sample>1.0) { sample=1.0; }
divider=int(1.0/sample); 

alphabet="ACDEFGHIKLMNPQRSTVWXY"; 
alphabet_length=21; 
a=1; 

for(i=1;i<=21;i++)
   { 
   res_count[i]=0; 
   } # end for i  

} # end of BEGIN 

substr($1,1,1)!=">"&&$0!="" { 

a++; 
if(a%divider==0)
{
for(i=1;i<=NF;i++) 
   { 
   for(j=1;j<=alphabet_length;j++) 
      { 
      if($i==substr(alphabet,j,1))
        { 
        res_count[j]++; 
        break; 
        } # end of if 
      } # end for j 
   } # end for i 
}

} # end substr($1,1,1)!=">"&&$0!="" 


END { 

for(i=1;i<=alphabet_length;i++)
   { 
   total_count += res_count[i] ; 
   } # end for i 

for(i=1;i<=alphabet_length;i++)
   { 
   printf "%s\t%.4f\n", substr(alphabet,i,1), res_count[i]/total_count; 
   } # end for i 

} # end of END  
