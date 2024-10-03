import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)



    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Test validity of validation ratio
        if args.val_ratio < 0 or args.val_ratio > 1:
            raise argparse.ArgumentTypeError(f"{args.val_ratio} is not a valid ratio; it must be between 0 and 1")
        
        # Calculate the number of validation samples
        total_samples = xtrain.shape[0]
        num_v = int(total_samples * args.val_ratio)

        # Shuffle indices for creating validation set
        indices = np.arange(xtrain.shape[0])
        np.random.shuffle(indices)
        
        validation_indices = indices[:num_v]
        train_indices = indices[num_v:]

        # Create validation set
        xtest, ctest, ytest = xtrain[validation_indices], ctrain[validation_indices], ytrain[validation_indices]
        xtrain, ctrain, ytrain = xtrain[train_indices], ctrain[train_indices], ytrain[train_indices]
    
    #Normalize the training and test data
    means = xtrain.mean(0,keepdims=True)
    stds  = xtrain.std(0,keepdims=True)
    xtrain_norm = normalize_fn(xtrain, means, stds)
    xtest_norm = normalize_fn(xtest, means, stds)

    # Add bias to xtrain and xtest
    xtrain_bias = np.hstack([np.ones((xtrain.shape[0], 1)), xtrain])
    xtest_bias = np.hstack([np.ones((xtest.shape[0], 1)), xtest])

    ### WRITE YOUR CODE HERE to do any other data processing
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        if args.task == "breed_identifying":
            method_obj = KNN(k=args.K, task_kind="classification")
        elif args.task == "center_locating":
            method_obj = KNN(k=args.K, task_kind="regression")
    
    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda=args.lmda)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    else:  ### WRITE YOUR CODE HERE
        pass


    ## 4. Train and evaluate the method

    if args.task == "center_locating":
        if args.method == "knn":
            #Use the normalized sets
            preds_train = method_obj.fit(xtrain_norm, ctrain)
            train_pred = method_obj.predict(xtrain_norm)
            preds = method_obj.predict(xtest_norm)
            
        elif args.method == "linear_regression":
            # Fit parameters on training data with added bias
            preds_train = method_obj.fit(xtrain_bias, ctrain)
    
            # Perform inference for training and test data
            train_pred = method_obj.predict(xtrain_bias)
            preds = method_obj.predict(xtest_bias)
            
        ## Report results: performance on train and valid/test sets
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.3f} - Test loss = {loss:.3f}")

    elif args.task == "breed_identifying":
        if args.method == "knn":
            #Use the normalized data
            preds_train = method_obj.fit(xtrain_norm, ytrain)
            preds = method_obj.predict(xtest_norm)

        else:
            # Fit (:=train) the method on the training data for classification task
            preds_train = method_obj.fit(xtrain_norm, ytrain)
    
            # Predict on unseen data
            preds = method_obj.predict(xtest_norm)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    if args.method == "knn" and args.plt:
        k_values = np.arange(1, args.K + 1)

        if args.task == "center_locating":
            train_loss_array = np.empty(args.K)
            test_loss_array = np.empty(args.K)

            for k in range(1, args.K + 1):
                new_obj = KNN(k=k, task_kind="regression")
                preds_train = new_obj.fit(xtrain_norm, ctrain)
                train_pred = new_obj.predict(xtrain_norm)
                preds = new_obj.predict(xtest_norm)

                train_loss_array[k-1] = mse_fn(train_pred, ctrain)
                test_loss_array[k-1] = mse_fn(preds, ctest)
            
            plt.figure()
            plt.plot(k_values, train_loss_array, label='Train Loss', marker='o', color='blue')
            plt.plot(k_values, test_loss_array, label='Test Loss', marker='o', color='red')
            plt.xlabel('k Values in Range [1, '+ str(args.K) +']')
            plt.ylabel('Loss (MSE)')
            plt.title('Train and Test Losses')
            plt.legend()
            plt.grid(True)
            plt.show()

        elif args.task == "breed_identifying":
            train_acc_array = np.empty(args.K)
            test_acc_array = np.empty(args.K)

            train_f1_array = np.empty(args.K)
            test_f1_array = np.empty(args.K)

            for k in range(1, args.K + 1):
                new_obj = KNN(k=k, task_kind="classification")
                preds_train = new_obj.fit(xtrain_norm, ytrain)
                preds = new_obj.predict(xtest_norm)

                train_acc_array[k-1] = accuracy_fn(preds_train, ytrain)
                test_acc_array[k-1] = accuracy_fn(preds, ytest)

                train_f1_array[k-1] = macrof1_fn(preds_train, ytrain)
                test_f1_array[k-1] = macrof1_fn(preds, ytest)
            
            fig, axs = plt.subplots(1, 2)

            axs[0].plot(k_values, train_acc_array, label='Train Accuracy', marker='o', color='blue')
            axs[0].plot(k_values, test_acc_array, label='Test Accuracy', marker='o', color='red')
            axs[0].set_xlabel('k Values in Range [1, '+ str(args.K) +']')
            axs[0].set_ylabel('Accuracy (%)')
            axs[0].set_title('Train and Test Accuracies')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(k_values, train_f1_array, label='Train F1-Score', marker='o', color='green')
            axs[1].plot(k_values, test_f1_array, label='Test F1-Score', marker='o', color='brown')
            axs[1].set_xlabel('k Values in Range [1, '+ str(args.K) +']')
            axs[1].set_ylabel('F1-Score')
            axs[1].set_title('Train and Test F1-Scores')
            axs[1].legend()
            axs[1].grid(True)

            plt.show()  

    elif args.method == "linear_regression" and args.plt:
        if args.task == "center_locating":
            lmda_values = np.logspace(-2, 10, num=50)   # Lambda values in a logarithmic scale from 0.01 to 10^10

            train_loss_array = []
            test_loss_array = []

            for lmda in lmda_values:
                method_obj = LinearRegression(lmda=lmda)

                # Fit parameters on training data with added bias
                preds_train = method_obj.fit(xtrain_bias, ctrain)

                # Perform inference for training and test data
                train_pred = method_obj.predict(xtrain_bias)
                test_pred = method_obj.predict(xtest_bias)

                # Calculate training and test loss
                train_loss = mse_fn(train_pred, ctrain)
                test_loss = mse_fn(test_pred, ctest)

                train_loss_array.append(train_loss)
                test_loss_array.append(test_loss)

            # Create subplots with shared y-axis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

            # Plotting training loss
            ax1.plot(lmda_values, train_loss_array, label='Train Loss', marker='o', color='blue')
            ax1.set_xlabel('Lambda Values')
            ax1.set_ylabel('Training Loss (MSE)')
            ax1.set_title('Training Loss')
            ax1.set_xscale('log')  # Set logarithmic scale for x-axis
            ax1.grid(True)
            ax1.legend()

            # Plotting test loss
            ax2.plot(lmda_values, test_loss_array, label='Test Loss', marker='o', color='red')
            ax2.set_xlabel('Lambda Values')
            ax2.set_ylabel('Test Loss (MSE)')
            ax2.set_title('Test Loss')
            ax2.set_xscale('log')  # Set logarithmic scale for x-axis
            ax2.grid(True)
            ax2.legend()

            plt.show()

    elif args.method == "logistic_regression" and args.plt:
        lr_values = np.logspace(-5, -1, num=20)   # LR values in a logarithmic scale from 0.01 to 10^10

        train_acc_array = []
        test_acc_array = []
        train_f1_array = []
        test_f1_array = []       

        for lr in lr_values:
            method_obj = LogisticRegression(lr=lr, max_iters = 4000)

            # Fit parameters on training data with added bias
            preds_train = method_obj.fit(xtrain_norm, ytrain)

            # Perform inference for training and test data
            train_pred = method_obj.predict(xtrain_norm)
            test_pred = method_obj.predict(xtest_norm)

            # Calculate training and test loss
            train_acc = accuracy_fn(train_pred, ytrain)
            test_acc = accuracy_fn(test_pred, ytest)

            train_f1 = macrof1_fn(train_pred, ytrain)
            test_f1 = macrof1_fn(test_pred, ytest)

            train_acc_array.append(train_acc)
            test_acc_array.append(test_acc)

            train_f1_array.append(train_f1)
            test_f1_array.append(test_f1)            

        

        best_train_acc_index = np.argmax(train_acc_array)
        best_test_acc_index = np.argmax(test_acc_array)
        best_train_acc = train_acc_array[best_train_acc_index]
        best_test_acc = test_acc_array[best_test_acc_index]
        best_train_lr = lr_values[best_train_acc_index]
        best_test_lr = lr_values[best_test_acc_index]

        best_train_f1_index = np.argmax(train_f1_array)
        best_test_f1_index = np.argmax(test_f1_array)
        best_train_f1 = train_f1_array[best_train_f1_index]
        best_test_f1 = test_f1_array[best_test_f1_index]
        best_train_lr = lr_values[best_train_f1_index]
        best_test_lr = lr_values[best_test_f1_index]

        # Plotting acc

        plt.subplot(1, 2, 1)
        plt.plot(lr_values, train_acc_array, label='Train Accuracy', marker='o', color='blue')
        plt.plot(lr_values, test_acc_array, label='Test Accuracy', marker='o', color='red')
        plt.xlabel('LR Values')
        plt.ylabel('Accuracies')
        plt.title('LR vs Accuracies')
        plt.xscale('log')  # Set logarithmic scale for x-axis
        plt.grid(True)
        plt.legend()

        # Plotting f1s
        plt.subplot(1, 2, 2)
        plt.plot(lr_values, train_f1_array, label='Train F1', marker='o', color='green')
        plt.plot(lr_values, test_f1_array, label='Test F1', marker='o', color='black')
        plt.xlabel('LR Values')
        plt.ylabel('F1-score')
        plt.title('LR versus F1 score')
        plt.xscale('log')  # Set logarithmic scale for x-axis
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!
    parser.add_argument('--plt', action="store_true", help="Plot the accuracy and MSE over different hyper parameters")
    parser.add_argument('--val_ratio', type=float, default=0.3, help="ratio of the data for validation set")

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
