import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    # 初期化してファイルに書き込む
    with open('log.txt', 'w') as file:
        file.write("h,c,acc\n")


    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Determine hyperparameters
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for h, gram_matrix in enumerate(all_matrices):
                #Gram matrix for train
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                #Gram matrix for evaluation
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                #class of train datasets
                c_train = classes[train_index]
                #class of evaluation datasets
                c_val = classes[val_index]
                
                
                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0
                    print(f"h={h+1},c={c},acc={val_acc}")
                    with open('log.txt', 'a') as file:
                        file.write(f"{h+1},{c},{val_acc}\n")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix
                        
            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]

            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())


