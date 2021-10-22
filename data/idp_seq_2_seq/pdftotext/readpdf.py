#
# # importing required modules
# import PyPDF2
#
# # creating a pdf file object
# pdfFileObj = open('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/LDR_train_dataset.pdf', 'rb')
# from tika import parser # pip install tika
#
# # raw = parser.from_file('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/LDR_train_dataset.pdf')
# # print(raw['content'])
# # exit()
# # creating a pdf reader object
# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
#
# # printing number of pages in pdf file
# print(pdfReader.numPages)
#
# # creating a page object
# pageObj = pdfReader.getPage(0)
#
# # extracting text from page
# print(pageObj.extractText().encode('utf-8'))
#
# # closing the pdf file object
# pdfFileObj.close()


import docx


def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
        print(para.text)
    return '\n'.join(fullText)

txt = getText('/data/idp_seq_2_seq/pdftotext/SDR_train_dataset.docx')
print(txt)