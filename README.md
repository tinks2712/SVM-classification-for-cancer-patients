# SVM-classification-for-cancer-patients
Perform SVM classification on cancer patient data using the 'sklearn' library. Use the 'rbf' kernel for nonlinear analysis. Evaluate model performance metrics including TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, PT, TS, ACC, BA, F1-Score, MCC, FM, BM, MK from confusion matrix.
import numpy as np
import matplotlib.pyplot as plt
      
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, matthews_corrcoef
from matplotlib.colors import ListedColormap
# Loading the dataset
modified_data = pd.read_csv('Patient_Cancer_SVM.csv', index_col=0, header=0) X_modified = modified_data.iloc[:, [1, 2]].values
Y_modified = modified_data.iloc[:, 3].values
# here, Splitig the dataset into training and testing sets
X_train_modified, X_test_modified, Y_train_modified, Y_test_modified = train_test_split(X_modified, Y_modified, test_size=0.5, random_state=0)
scaler_modified = StandardScaler()
X_train_scaled_modified = scaler_modified.fit_transform(X_train_modified) X_test_scaled_modified = scaler_modified.transform(X_test_modified)
# Train the SVM model with 'rbf' kernel
svm_model_modified = SVC(kernel='rbf', random_state=0) svm_model_modified.fit(X_train_scaled_modified, Y_train_modified)
# Predictions trial on the test set
Y_pred_modified = svm_model_modified.predict(X_test_scaled_modified)
#Modifying Confusion Matrix
cm_test_modified = confusion_matrix(Y_test_modified, Y_pred_modified)

tn_modified, fp_modified, fn_modified, tp_modified = cm_test_modified.ravel()
# Calculating the evaluation criteria
TPR_modified = tp_modified / (tp_modified + fn_modified)
TNR_modified = tn_modified / (tn_modified + fp_modified)
PPV_modified = tp_modified / (tp_modified + fp_modified)
NPV_modified = tn_modified / (tn_modified + fn_modified)
FNR_modified = fn_modified / (fn_modified + tp_modified)
FPR_modified = fp_modified / (fp_modified + tn_modified)
FDR_modified = fp_modified / (fp_modified + tp_modified)
FOR_modified = fn_modified / (fn_modified + tn_modified)
PT_modified = tp_modified + fn_modified
TS_modified = tp_modified / (tp_modified + FNR_modified)
ACC_modified = (tp_modified + tn_modified) / (tp_modified + tn_modified + fp_modified + fn_modified)
BA_modified = (TPR_modified + TNR_modified) / 2
F1_modified = f1_score(Y_test_modified, Y_pred_modified) MCC_modified = matthews_corrcoef(Y_test_modified, Y_pred_modified) FM_modified = np.sqrt(TPR_modified * PPV_modified)
BM_modified = np.sqrt(TNR_modified * NPV_modified)
MK_modified = np.sqrt(PPV_modified * NPV_modified)
# printing the results
print("Modified Confusion Matrix:") print(cm_test_modified)
print("\nTrue Positive Rate (TPR):", TPR_modified) print("True Negative Rate (TNR):", TNR_modified) print("Positive Predictive Value (PPV):", PPV_modified)

print("Negative Predictive Value (NPV):", NPV_modified) print("False Negative Rate (FNR):", FNR_modified) print("False Positive Rate (FPR):", FPR_modified) print("False Discovery Rate (FDR):", FDR_modified) print("False Omission Rate (FOR):", FOR_modified) print("Prevalence Threshold (PT):", PT_modified) print("Threat Score (TS):", TS_modified)
print("Accuracy (ACC):", ACC_modified)
print("Balanced Accuracy (BA):", BA_modified)
print("F1-Score (F1):", F1_modified)
print("Matthews Correlation Coefficient (MCC):", MCC_modified) print("Fowlkes-Mallows Index (FM):", FM_modified) print("Bookmaker Informedness (BM):", BM_modified) print("Markedness (MK):", MK_modified)
# Plotting results for the training set
plot_train_modified = plt.figure(1)
X_set_train_modified, Y_set_train_modified = X_train_scaled_modified, Y_train_modified
X1_train_modified, X2_train_modified = np.meshgrid(np.arange(start=X_set_train_modified[:, 0].min() - 1, stop=X_set_train_modified[:, 0].max() + 1, step=0.01),
np.arange(start=X_set_train_modified[:, 1].min() - 1, stop=X_set_train_modified[:, 1].max() + 1, step=0.01))
plt.contourf(X1_train_modified, X2_train_modified, svm_model_modified.predict(np.array([X1_train_modified.ravel(), X2_train_modified.ravel()]).T).reshape(X1_train_modified.shape),
alpha=0.75, cmap=ListedColormap(('white', 'gray'))) plt.xlim(X1_train_modified.min(), X1_train_modified.max()) plt.ylim(X2_train_modified.min(), X2_train_modified.max())
for i_train_modified, j_train_modified in enumerate(np.unique(Y_set_train_modified)):

plt.scatter(X_set_train_modified[Y_set_train_modified == j_train_modified, 0], X_set_train_modified[Y_set_train_modified == j_train_modified, 1],
cmap=ListedColormap(('blue', 'purple'))(i_train_modified), label=j_train_modified) plt.title('Support Vector Machine (Training set)')
plt.xlabel('Age')
plt.ylabel('Marker Gene Level')
plt.legend()
# Plotting results for the testing set
plot_test_modified = plt.figure(2)
X_set_test_modified, Y_set_test_modified = X_test_scaled_modified, Y_test_modified
X1_test_modified, X2_test_modified = np.meshgrid(np.arange(start=X_set_test_modified[:, 0].min() - 1, stop=X_set_test_modified[:, 0].max() + 1, step=0.01),
np.arange(start=X_set_test_modified[:, 1].min() - 1, stop=X_set_test_modified[:, 1].max() + 1, step=0.01))
plt.contourf(X1_test_modified, X2_test_modified, svm_model_modified.predict(np.array([X1_test_modified.ravel(), X2_test_modified.ravel()]).T).reshape(X1_test_modified.shape),
alpha=0.75, cmap=ListedColormap(('white', 'gray'))) plt.xlim(X1_test_modified.min(), X1_test_modified.max()) plt.ylim(X2_test_modified.min(), X2_test_modified.max())
for i_test_modified, j_test_modified in enumerate(np.unique(Y_set_test_modified)):
plt.scatter(X_set_test_modified[Y_set_test_modified == j_test_modified, 0], X_set_test_modified[Y_set_test_modified == j_test_modified, 1],
cmap=ListedColormap(('blue', 'purple'))(i_test_modified), label=j_test_modified) plt.title('Support Vector Machine (Test set)')
plt.xlabel('Age')
plt.ylabel('Marker Gene Level')
plt.legend()

plt.show()
