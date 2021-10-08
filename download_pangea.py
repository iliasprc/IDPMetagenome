username, password = 'papastrai@csd.auth.gr', 'vyiNvYCD6g2v6B9'

# from pangea_api import Knex, User, Organization
#
# knex = Knex()
# User(knex, "papastrai@csd.auth.gr", password).login()
# org = Organization(knex, "MetaSUB Consortium").idem()
# grp = org.sample_group("MetaSUB Doha").idem()
# for sample in grp.get_samples(cache=False):
#     for ar in sample.get_analysis_results(cache=False):
#         if ar.module_name != 'raw::raw_reads':
#             continue
#         for field in ar.get_fields(cache=False):
#             field.download_file(filename=filename)

# pangea-api download sample-results -e 'papastrai@csd.auth.gr' -p 'vyiNvYCD6g2v6B9' --module-name "raw::raw_reads"
# 'MetaSUB Consortium' 'MetaSUB Doha'
# pangea-api download sample-results -e 'papastrai@csd.auth.gr' -p vyiNvYCD6g2v6B9 --module-name 'raw::raw_reads' 'MetaSUB Consortium' 'MetaSUB Doha'
# pangea-api download metadata -e 'papastrai@csd.auth.gr' -p vyiNvYCD6g2v6B9 'MetaSUB Consortium' 'MetaSUB Doha'

pangea-api download sample-results -e 'papastrai@csd.auth.gr' -p vyiNvYCD6g2v6B9 --module-name "raw::raw_reads" "MetaSUB Consortium" "MetaSUB Doha"
import pandas as pd
from pangea_api import Knex, User, Organization

knex = Knex()
User(knex, "papastrai@csd.auth.gr", password).login()
print('sdsds')
org = Organization(knex, "MetaSUB Consortium").idem()
grp = org.sample_group("MetaSUB Doha").idem()

for sample in grp.get_samples(cache=False):
    print(sample)
    if sample_names and sample.name not in sample_names:
        continue
    metadata[sample.name] = sample.metadata
metadata = pd.DataFrame.from_dict(metadata, orient='index')
metadata.to_csv("MetaSUB Doha_metadata.csv")