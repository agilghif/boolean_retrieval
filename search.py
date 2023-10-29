from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_path = 'collections', \
                          postings_encoding = VBEPostings, \
                          output_path = 'index')

queries = ["pupil mata", "aktor", "batu permata"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for doc in BSBI_instance.boolean_retrieve(query):
        print(doc, BSBI_instance.doc_id_map[doc])
    print()