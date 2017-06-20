import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = labels
y_pos = np.arange(len(objects))
performance = number_of_documents
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of documents')
plt.title('Categories in RCV1 Dataset')
 
plt.show()