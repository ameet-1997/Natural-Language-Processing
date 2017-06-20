import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

y_pos = np.arange(len(objects))
# numbers_of_documents is a list which contains the number of documents per categories
# labels are the label names that can be obtained from RCV1 dataset
plt.bar(y_pos, numbers_of_documents, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Number of documents')
plt.title('Categories in RCV1 Dataset')
 
plt.show()
