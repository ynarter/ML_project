import argparse

import numpy as np
from torchinfo import summary
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import time


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data_path)
    print(f"xtrain: {xtrain.shape}, ytrain: {ytrain.shape}")
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    print(f"xtrain: {xtrain.shape}, ytrain: {ytrain.shape}")
    print(f"xtest: {xtest.shape}")

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.



    # Make a validation set
    if not args.test:
    ### WRITE YOUR CODE HERE
        if args.val_ratio < 0 or args.val_ratio > 1:
            raise argparse.ArgumentTypeError(f"{args.val_ratio} is not a valid ratio; it must be between 0 and 1")
        
        """
        if args.use_pca:
            print("Using PCA")
            pca_obj = PCA(d=args.pca_d)
            exvar = pca_obj.find_principal_components(xtrain)  # Fit PCA on training data only
            xtrain = pca_obj.reduce_dimension(xtrain) # Transform training data   
            ytrain = pca_obj.reduce_dimension(ytrain)
            print(f"Explained variance by top {args.pca_d} components: {exvar:.2f}%")
        """
        
        # Shuffle indices
        indices = np.arange(len(xtrain))
        np.random.shuffle(indices)

        split_index = int(args.val_ratio * len(xtrain))
        val_indices, train_indices = indices[:split_index], indices[split_index:]
        xtrain, xtest = xtrain[train_indices], xtrain[val_indices]
        ytrain, ytest = ytrain[train_indices], ytrain[val_indices]
        print(f"xtrain: {xtrain.shape}, ytrain: {ytrain.shape}")
        print(f"xtest: {xtest.shape}, ytest: {ytest.shape}")
        


    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        time_start_pca = time.time()
        
        exvar = pca_obj.find_principal_components(xtrain)
        xtrain_reduced = pca_obj.reduce_dimension(xtrain)
        xtest_reduced = pca_obj.reduce_dimension(xtest)
        
        time_end_pca = time.time()
        elapsed_time_pca = time_end_pca - time_start_pca 
        
        print(f"Explained variance by top {args.pca_d} components: {exvar:.2f}%")
        print(f"Elapsed time for PCA: {elapsed_time_pca:.4f} seconds")
        
        
        if args.nn_type == "mlp": #use the PCA for only MLP
            xtrain = xtrain_reduced
            xtest = xtest_reduced


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    input_size = xtrain.shape[1]
    if args.nn_type == "mlp":
        model = MLP(input_size=input_size, n_classes=n_classes, 
                    hidden_dim = args.hidden_dim, num_layers = args.num_layers, 
                    activation_fn= args.activation_fn, use_bias= args.use_bias)

    elif args.nn_type == "cnn":
        xtrain = np.expand_dims(xtrain.reshape(-1, 28, 28), axis=0).transpose((1, 0, 2, 3)) / 255
        xtest = np.expand_dims(xtest.reshape(-1, 28, 28), axis=0).transpose((1, 0, 2, 3)) / 255
        model = CNN(input_channels = 1, n_classes = 10, filters = (args.filter1, args.filter2, args.filter3), dropouts = (args.dropout1, args.dropout2, args.dropout3))   
        summary(model)    

    elif args.nn_type == "transformer":
        xtrain = np.expand_dims(xtrain.reshape(-1, 28, 28), axis=0).transpose((1, 0, 2, 3)) / 255
        xtest = np.expand_dims(xtest.reshape(-1, 28, 28), axis=0).transpose((1, 0, 2, 3)) / 255
        chw = xtrain.shape
        model = MyViT(chw=chw, n_patches=args.n_patches, n_blocks=args.n_blocks, hidden_d=args.hidden_d, n_heads=args.n_heads,
                      position_type=args.position_type, dropout=args.dropout)
        if (args.verbose):
            print(f"lr: {args.lr}, epochs: {args.max_iters}, n_patches: {args.n_patches}, n_blocks: {args.n_blocks}, hidden_d: {args.hidden_d}, n_heads: {args.n_heads}")
    else:
        raise ValueError(f"Unsupported network architecture, choose one of the following: 'mlp', 'transformer', 'cnn'")    
    
    # summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, verbose=args.verbose)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    time_start_tr = time.time()
    
    preds_train = method_obj.fit(xtrain, ytrain)
    
    time_end_tr = time.time()

    # Predict on unseen data
    time_start_pr = time.time()
    
    preds = method_obj.predict(xtest)

    time_end_pr = time.time()

    if args.test:
        print(f"preds = {preds.shape}")
        np.save("predictions", preds) 

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    if not args.test:
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    
    elapsed_time_tr = time_end_tr - time_start_tr
    elapsed_time_pr = time_end_pr - time_start_pr
    
    print(f"Elapsed time for training: {elapsed_time_tr:.4f} seconds")
    print(f"Elapsed time for prediction: {elapsed_time_pr:.4f} seconds")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    # Visualization
    # if args.nn_type == "transformer" and args.plt:
    #     lr_values = np.logspace(-5, -2, num=10)  # Learning rate values in a logarithmic scale from 1e-5 to 1e-2

    #     train_acc_array = []
    #     test_acc_array = []

    #     for lr in lr_values:
    #         model = MyViT(
    #             chw=(1, 28, 28), 
    #             n_patches=args.n_patches, 
    #             n_blocks=args.n_blocks, 
    #             hidden_d=args.hidden_d, 
    #             n_heads=args.n_heads,
    #             position_type=args.position_type
    #         )
    #         method_obj = Trainer(model=model, lr=lr, epochs=args.max_iters, batch_size=args.nn_batch_size, verbose=args.verbose)
            
    #         # Fit parameters on training data with added bias
    #         preds_train = method_obj.fit(xtrain, ytrain)

    #         # Perform inference for training and test data
    #         train_pred = method_obj.predict(xtrain)
    #         test_pred = method_obj.predict(xtest)

    #         # Calculate training and test accuracy
    #         train_acc = accuracy_fn(train_pred, ytrain)
    #         test_acc = accuracy_fn(test_pred, ytest)

    #         train_acc_array.append(train_acc)
    #         test_acc_array.append(test_acc)

    #     # Create subplots with shared y-axis
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    #     # Plotting training accuracy
    #     ax1.plot(lr_values, train_acc_array, label='Train Accuracy', marker='o', color='blue')
    #     ax1.set_xlabel('Learning Rate')
    #     ax1.set_ylabel('Accuracy')
    #     ax1.set_title('Training Accuracy')
    #     ax1.set_xscale('log')  # Set logarithmic scale for x-axis
    #     ax1.grid(True)
    #     ax1.legend()

    #     # Plotting test accuracy
    #     ax2.plot(lr_values, test_acc_array, label='Test Accuracy', marker='o', color='red')
    #     ax2.set_xlabel('Learning Rate')
    #     ax2.set_ylabel('Accuracy')
    #     ax2.set_title('Test Accuracy')
    #     ax2.set_xscale('log')  # Set logarithmic scale for x-axis
    #     ax2.grid(True)
    #     ax2.legend()

    #     plt.show()

    

    if args.nn_type == "cnn" and args.plt and args.epoch_plt:
        epochs = range(1, args.max_iters + 1)
            
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, method_obj.train_accuracies, label='Train Accuracy', marker='o', color='red')
        plt.plot(epochs, method_obj.test_accuracies, label='Test Accuracy', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy and Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        plt.show()

    # if args.nn_type == "mlp" and args.plt:
    #     lr_values = np.logspace(-5, -2, num=10)  # Learning rate values in a logarithmic scale from 1e-5 to 1e-2

    #     train_acc_array = []
    #     test_acc_array = []

    #     for lr in lr_values:
    #         model = MLP(input_size=input_size, n_classes=n_classes, 
    #                 hidden_dim = args.hidden_dim, num_layers = args.num_layers, 
    #                 activation_fn= args.activation_fn, use_bias= args.use_bias)
    #         method_obj = Trainer(model=model, lr=lr, epochs=args.max_iters, batch_size=args.nn_batch_size, verbose=args.verbose)
            
    #         # Fit parameters on training data with added bias
    #         preds_train = method_obj.fit(xtrain, ytrain)

    #         # Perform inference for training and test data
    #         train_pred = method_obj.predict(xtrain)
    #         test_pred = method_obj.predict(xtest)

    #         # Calculate training and test accuracy
    #         train_acc = accuracy_fn(train_pred, ytrain)
    #         test_acc = accuracy_fn(test_pred, ytest)

    #         train_acc_array.append(train_acc)
    #         test_acc_array.append(test_acc)

    #     # Create subplots with shared y-axis
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    #     # Plotting training accuracy
    #     ax1.plot(lr_values, train_acc_array, label='Train Accuracy', marker='o', color='blue')
    #     ax1.set_xlabel('Learning Rate')
    #     ax1.set_ylabel('Accuracy')
    #     ax1.set_title('Training Accuracy')
    #     ax1.set_xscale('log')  # Set logarithmic scale for x-axis
    #     ax1.grid(True)
    #     ax1.legend()

    #     # Plotting test accuracy
    #     ax2.plot(lr_values, test_acc_array, label='Test Accuracy', marker='o', color='red')
    #     ax2.set_xlabel('Learning Rate')
    #     ax2.set_ylabel('Accuracy')
    #     ax2.set_title('Test Accuracy')
    #     ax2.set_xscale('log')  # Set logarithmic scale for x-axis
    #     ax2.grid(True)
    #     ax2.legend()

    #     plt.show()

    # if args.nn_type == "cnn" and args.plt:
    #     lr_values = np.logspace(-5, -2, num=10)  # Learning rate values in a logarithmic scale from 1e-5 to 1e-2

    #     train_acc_array = []
    #     test_acc_array = []

    #     for lr in lr_values:
    #         model = CNN(input_channels = 1, n_classes = 10, filters = (args.filter1, args.filter2, args.filter3), dropouts = (args.dropout1, args.dropout2, args.dropout3))   

    #         method_obj = Trainer(model=model, lr=lr, epochs=args.max_iters, batch_size=args.nn_batch_size, verbose=args.verbose)
            
    #         # Fit parameters on training data with added bias
    #         preds_train = method_obj.fit(xtrain, ytrain)

    #         # Perform inference for training and test data
    #         train_pred = method_obj.predict(xtrain)
    #         test_pred = method_obj.predict(xtest)

    #         # Calculate training and test accuracy
    #         train_acc = accuracy_fn(train_pred, ytrain)
    #         test_acc = accuracy_fn(test_pred, ytest)

    #         train_acc_array.append(train_acc)
    #         test_acc_array.append(test_acc)

    #     # Create subplots with shared y-axis
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    #     # Plotting training accuracy
    #     ax1.plot(lr_values, train_acc_array, label='Train Accuracy', marker='o', color='blue')
    #     ax1.set_xlabel('Learning Rate')
    #     ax1.set_ylabel('Accuracy')
    #     ax1.set_title('Training Accuracy')
    #     ax1.set_xscale('log')  # Set logarithmic scale for x-axis
    #     ax1.grid(True)
    #     ax1.legend()

    #     # Plotting test accuracy
    #     ax2.plot(lr_values, test_acc_array, label='Test Accuracy', marker='o', color='red')
    #     ax2.set_xlabel('Learning Rate')
    #     ax2.set_ylabel('Accuracy')
    #     ax2.set_title('Test Accuracy')
    #     ax2.set_xscale('log')  # Set logarithmic scale for x-axis
    #     ax2.grid(True)
    #     ax2.legend()

    #     plt.show()


def percent_formatter(x, pos):
    return f'{x:.2f}%'
if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data_path', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    
    parser.add_argument('--epoch_plt', action="store_true", help="Enable detailed output during program execution")
    parser.add_argument('--verbose', action="store_true", help="Enable detailed output during program execution")
    parser.add_argument('--plt', action="store_true", help="Plot the accuracy and MSE over different hyper parameters")
    parser.add_argument('--val_ratio', type=float, default=0.3, help="ratio of the data for validation set")

    #ADDED FOR MLP
    parser.add_argument('--hidden_dim', type=int, default=512, help="size of hidden layers for MLP (default: 512)")
    parser.add_argument('--num_layers', type=int, default=5, help="number of hidden layers for MLP (default: 5)")
    parser.add_argument('--activation_fn', type=str, default="relu", help="activation function to be used for MLP (default: 'relu', other choices: 'sigmoid', 'tanh', 'elu')")
    parser.add_argument('--use_bias', type=bool, default=True, help="whether to use bias in linear layers in MLP (default: True)")
    
    #ADDED FOR VISION TRANSFORMER
    parser.add_argument('--n_patches', type=int, default=4, help="number of patches for ViT (default: 4)")
    parser.add_argument('--n_blocks', type=int, default=8, help="number of blocks for ViT (default: 8)")
    parser.add_argument('--hidden_d', type=int, default=64, help="hidden dimension size for ViT (default: 64)")
    parser.add_argument('--n_heads', type=int, default=4, help="number of head for attention in ViT (default: 4)")

    #ADDED FOR CNN
    parser.add_argument('--filter1', type=int, default=32, help="filter size for first convolutional layer of CNN (default: 32)")
    parser.add_argument('--filter2', type=int, default=64, help="filter size for second convolutional layer of CNN (default: 64)")
    parser.add_argument('--filter3', type=int, default=128, help="filter size for third convolutional layer of CNN (default: 128)")
    parser.add_argument('--dropout1', type = float, default=0.3, help="dropout ratio for first dropout layer of CNN (default: 0.3)")
    parser.add_argument('--dropout2', type = float, default=0.4, help="dropout ratio for second dropout layer of CNN (default: 0.4)")
    parser.add_argument('--dropout3', type = float, default=0.25, help="dropout ratio for third dropout layer of CNN (default: 0.25)")


    parser.add_argument('--position_type', default="learnable", help="which positional embedding to use, it can be 'learnable' | 'trigonometric'")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout ratio (default: 0.2)")
    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
