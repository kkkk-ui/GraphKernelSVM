# -*- coding: utf-8 -*-
import numpy as np
import GraphKernelFunc as kf
import kernel_evaluation as kb
import data

#-------------------
# Make data
myData = data.classification(negLabel=-1.0,posLabel=1.0)
myData.makeData(dataType=int(input("data type =>")))
#-------------------

#-------------------
# Learning and evaluation.
# Number of repetitions of 10-CV.
num_reps = 10
results = []

# 1-WL kernel, number of iterations in [1:6].
all_matrices = []

print("""choose kernel type 
[0]:Shortest Path kernel 
[1]:GraphletSampling kernel
[2]:Weisfeiler-Lehman kernel
[3]:VertexHistogram kernel
[4]:EdgeHistogram kernel""")
kernelType = int(input("=>"))
myKernel = kf.GraghkernelFunc(kernelType, kernelParam=1)

# カーネルの設定
kernelFunc = myKernel
if kernelType == 0:
    print('sp-kernel mode')
    # gram matrix
    gm = kernelFunc.createMatrix(myData.graphs)
    # normalize gram matrix
    gm_n = kf.GraghkernelFunc.normalize_gram_matrix(gm)
    
    all_matrices.append(gm_n)
    classes = myData.classes
    acc, s_1, s_2 = kb.kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print("c15" + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
elif kernelType == 1:
    print('gs-kernel mode')
    # gram matrix
    gm = kernelFunc.createMatrix(myData.graphs)
    # normalize gram matrix
    gm_n = kf.GraghkernelFunc.normalize_gram_matrix(gm)
    
    all_matrices.append(gm_n)
    classes = myData.classes
    acc, s_1, s_2 = kb.kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print("c15" + " " + "GS " + str(acc) + " " + str(s_1) + " " + str(s_2))
elif kernelType == 2:
    print('wl-kernel mode')
    for i in range(1, 5):
        # gram matrix
        gm = kernelFunc.createMatrix(myData.graphs, i)
        # normalize gram matrix
        gm_n = kf.GraghkernelFunc.normalize_gram_matrix(gm)
    
        all_matrices.append(gm_n)
    classes = myData.classes
    acc, s_1, s_2 = kb.kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print("c15" + " " + "WL " + str(acc) + " " + str(s_1) + " " + str(s_2))
elif kernelType == 3:
    print('vh-kernel mode')
    # gram matrix
    gm = kernelFunc.createMatrix(myData.graphs)
    # normalize gram matrix
    gm_n = kf.GraghkernelFunc.normalize_gram_matrix(gm)
    
    all_matrices.append(gm_n)
    classes = myData.classes
    acc, s_1, s_2 = kb.kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print("c15" + " " + "VH " + str(acc) + " " + str(s_1) + " " + str(s_2))
elif kernelType == 4:
    print('eh-kernel mode')
    # gram matrix
    gm = kernelFunc.createMatrix(myData.graphs)
    # normalize gram matrix
    gm_n = kf.GraghkernelFunc.normalize_gram_matrix(gm)
    
    all_matrices.append(gm_n)
    classes = myData.classes
    acc, s_1, s_2 = kb.kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print("c15" + " " + "EH " + str(acc) + " " + str(s_1) + " " + str(s_2))
#-------------------


