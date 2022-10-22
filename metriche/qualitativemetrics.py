TP = 5
FP = 0
FN = 0
TN = 76

# accuracy = (TP + TN) / (TP + TN + FP + FN)
# specificity = TN / (TN + FP)
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# F1score = (2 * TP) / (2 * TP + FP + FN)
TP_total = 5+3+3+2+5+26+10+3
FN_total = 1+2+6+3+7+4
accuracy_total = TP_total/(TP_total+FN_total)
print("tot accuracy : %.4f" %accuracy_total)
# print("accuracy: %.4f" % accuracy)
# print("precision: %.2f" % precision)
# print("recall:  %.2f" % recall)
# print("specificity: %.4f" % specificity)
# print("f1score: %.2f" % F1score)
# melo = ((10*0.9)+(5*0.44)+(3*1)+(4*0.4)+(0.43*8)+(0.89*32)+(0.81*14)+(1*5))/81
# print(melo)