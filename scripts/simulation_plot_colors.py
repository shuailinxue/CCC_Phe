"""Hungarian color alignment for visualization only; predictions remain untouched."""
import numpy as np
from scipy.optimize import linear_sum_assignment
TRUTH_COLORS=['#dcdfe3','#e98088','#dc9c3b','#8e69b1','#398d93']
COARSE_COLORS=['#dcdfe3','#e98088','#c19651','#398d93']
def display_color_map(reference,prediction,n_reference):
 values=np.unique(prediction);counts=np.array([[np.sum((reference==k)&(prediction==v)) for v in values] for k in range(n_reference)])
 row,col=linear_sum_assignment(-counts)
 return {int(values[j]):int(i) for i,j in zip(row,col)}
def matched_palette(reference,prediction,palette):
 match=display_color_map(reference,prediction,len(palette))
 return [palette[match[k]] if k in match else '#444444' for k in range(int(np.max(prediction))+1)]
