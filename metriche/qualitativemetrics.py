TP = 
FP = 
FN = 
TN = 

accuracy = (TP + TN) / (TP + TN + FP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1score = (2 * TP) / (2 * TP + FP + FN)
TP_total = 
FN_total = 
accuracy_total = TP_total/(TP_total+FN_total)
print("tot accuracy : %.4f" %accuracy_total)
print("accuracy: %.4f" % accuracy)
print("precision: %.2f" % precision)
print("recall:  %.2f" % recall)
print("specificity: %.4f" % specificity)
print("f1score: %.2f" % F1score)
