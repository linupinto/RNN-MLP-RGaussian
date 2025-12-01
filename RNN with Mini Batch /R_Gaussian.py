import numpy as np
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score,f1_score,classification_report,accuracy_score
import math
import csv
import seaborn as sns
import tracemalloc

base_path=r"C:\Users\HP\Desktop\D folder\RNN\RNN pgm\output_rgauss"
tracemalloc.start()  #  START memory tracking

# import data using pandas
path=r"C:\Users\HP\Desktop\D folder\RNN\RNN pgm\HAR_RNN_ready.csv"
dataset = pd.read_csv(path)


# Select the input features as x output features as y
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values[:, np.newaxis] #(N,1)


# Each time step has 9 features
T = X.shape[1] // 9
# Reshape to (N, T, 9) for RNN input
X = X.reshape(X.shape[0], T, 9)  # (N,T,9)
T_feature=X.shape[2]
n_samples = X.shape[0] # Total Number of samples
CLS=6 # No of classes

# One hot encoding to NXC matrix
Y_flat = Y.flatten()
Y_flat=Y_flat.astype(int)
Y=np.eye(CLS)[Y_flat] 



# ------------- Shuffle the data------------------
indices = np.random.permutation(n_samples)
X_shuffled = X[indices,:,:]
y_shuffled = Y[indices]

# 2. Compute split points
train_end=int(0.7*n_samples)
val_end=int(.85*n_samples)


# 3. Split
train_X, train_Y = X_shuffled[:train_end,:,:], y_shuffled[:train_end]
val_X, val_Y = X_shuffled[train_end:val_end,:,:], y_shuffled[train_end:val_end]
test_X, test_Y = X_shuffled[val_end:,:,:], y_shuffled[val_end:]

# -------Parameters------------
layer_dims=[T_feature,64,32,CLS] # input dimension (features per time step),  hidden state size, output dimension (e.g., for classification)
L= len(layer_dims)-2      # Total hidden layers here 3

# -------------Random initialization -----------
params={}
act={}
for l in range(1, L+1): # 1 to 3
    #params[f'Wxh{l}']= np.random.randn(layer_dims[l], layer_dims[l - 1])   # input → hidden
    #params[f'Whh{l}']= np.random.randn(layer_dims[l], layer_dims[l])  # hidden → hidden
    params[f'Wxh{l}']=np.random.randn(layer_dims[l], layer_dims[l - 1])* np.sqrt(1. / layer_dims[l - 1])  # input → hidden
    W_hh_1_raw = np.random.randn(layer_dims[l], layer_dims[l])
    W_hh_1, _ = np.linalg.qr(W_hh_1_raw)
    params[f'Whh{l}']=W_hh_1    
    params[f'bh{l}']= np.zeros(( 1,layer_dims[l]))
      
#params['Why']= np.random.randn(layer_dims[L+1], layer_dims[L ])  
params['Why']= np.random.randn(layer_dims[L+1], layer_dims[L ])* np.sqrt(1. /layer_dims[L ])
params['by']=np.zeros((1,layer_dims[L+1]))



#---------------Mini Batches-------------------------
def create_minibatches(X, Y, batch_size):
    N = X.shape[0]
    minibatches = []

    for i in range(0, N, batch_size):
        X_batch = X[i:i + batch_size]  # shape: (batch_size, T, D)
        Y_batch = Y[i:i + batch_size]  # shape: (batch_size, C)
        minibatches.append((X_batch, Y_batch))

    return minibatches

#-----------------Activation and derivatives --------------------------
# Activation function
def r_gauss(Z):
    M=np.matrix(np.max(Z, axis=0))
    m=np.matrix(np.min(Z, axis=0))
    C = M - m
    C1 = 2*np.square(C)
    N = -1*np.square(Z-M)
    A1 = np.exp(N/C1)
    return A1

def r_gaus_der(Z):
    M=np.matrix(np.max(Z, axis=0))
    m=np.matrix(np.min(Z, axis=0))
    C = M - m
    A=r_gauss(Z)
    B=np.multiply(A,-(Z-M))/np.square(C)
    return B  

# Softmax and loss
def softmax(Z):
    Z = np.asarray(Z)
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e_Z / np.sum(e_Z, axis=1, keepdims=True)
   
#--------------- Forward Propagation ------------------------

act= {}
output={}
# Initialize H-1 for all layers
def forwardpropagation(X,params):
    N = X.shape[0] # No. smaples
    for l in range(1, L+1):
        H0 = np.zeros((N, params[f'Whh{l}'].shape[0]))
        act[f'H{0}_{l}'] = H0
        
    for t in range(1,T+1): # 1 to 999
        x_t = X[:,t-1,:]  # (N, D)
        
        for l in range(1, L+1):
            if l == 1:
                z = x_t @ params[f'Wxh{l}'].T + act[f'H{t-1}_{l}'] @ params[f'Whh{l}'].T + params[f'bh{l}']
            else:       
                z =  act[f'H{t}_{l-1}'] @ params[f'Wxh{l}'].T + act[f'H{t-1}_{l}'] @ params[f'Whh{l}'].T + params[f'bh{l}']

            act[f'Z{t}_{l}'] = z
            act[f'H{t}_{l}'] = r_gauss(z)

    # Final output layer (to C classes)
    
    last_h =  act[f'H{T}_{L}']  # top layer at final time step
    z_out = last_h @ params['Why'].T + params['by']
    act[f'Z{T}_{L+1}'] = z_out
    act[f'H{T}_{L+1}'] = softmax(act[f'Z{T}_{L+1}'])  # shape: (N, C)
    return act
#------------------------------------------------

def compute_cross_entropy(predictions, Y_hot):
    epsilon=1e-12
    # Prevent log(0)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)   
    # Only compute log probs for the true class in each row
    log_preds = np.log(predictions)   
    # Cross-entropy loss for each sample
    loss_per_sample = -np.sum(Y_hot * log_preds, axis=1)  # shape: (N,)    
    # Average over samples
    return np.mean(loss_per_sample)


#----------- BACKPROPAGATION-----------------
grads={}
for l in range(1, L+1):  # for hidden layers
    grads[f'dWxh{l}'] = np.zeros_like(params[f'Wxh{l}'])
    grads[f'dWhh{l}'] = np.zeros_like(params[f'Whh{l}'])
    grads[f'dbh{l}'] = np.zeros_like(params[f'bh{l}'])
delta={}
def backwardpropagation(X,Y, act, params):
    E=(act[f'H{T}_{L+1}']-Y)
    for l in range(1,L):
        delta[f'd{T+1}_{l}'] = np.zeros(( X.shape[0], layer_dims[l]))
   
    for t in range(T, 0, -1):
        if t==T:
            delta[f'd{t}_{L}']=(E @ params['Why']) * np.asarray(r_gaus_der( act[f'Z{t}_{L}']))
        else:
            delta[f'd{t}_{L}']=(delta[f'd{t+1}_{L}']@ params[f'Whh{L}'])*np.asarray(r_gaus_der(act[f'Z{t}_{L}']))

    for l in range(L-1,0,-1):
        for t in range(T,0,-1):            
            delta[f'd{t}_{l}']=(delta[f'd{t+1}_{l}']@ params[f'Whh{l}']+delta[f'd{t}_{l+1}'] 
                                @  params[f'Wxh{l+1}'])* np.asarray(r_gaus_der(act[f'Z{t}_{l}']))
    # Gradients        
    for l in range(1,L+1):
        for t in range(1,T+1):
            x_t = X[:,t-1,:]  # (N, D)
            h_prev = act[f'H{t-1}_{l}'] 
            grads[f'dWhh{l}'] += delta[f'd{t}_{l}'].T @ h_prev 
            if l==1:
                grads[f'dWxh{l}'] += delta[f'd{t}_{l}'].T @ x_t
            else:
                grads[f'dWxh{l}'] += delta[f'd{t}_{l}'].T @ act[f'H{t}_{l-1}']
            grads[f'dbh{l}'] += np.sum(delta[f'd{t}_{l}'].T, axis=1)

    # Output layer
    grads['dWhy'] = E.T @ act[f'H{T}_{L}']
    grads['dby'] = np.sum(E, axis=0, keepdims=True)
    return grads

   
#--------------------Update Prams-----------------------------------
learning_rate=0.001
def updateparams(grads,learning_rate,params,L):
    for l in range(1, L + 1):
        params[f'Wxh{l}'] -= learning_rate * grads[f'dWxh{l}']
        params[f'Whh{l}'] -= learning_rate * grads[f'dWhh{l}']
        params[f'bh{l}']  -= learning_rate * grads[f'dbh{l}']

    params['Why'] -= learning_rate * grads['dWhy']
    params['by']  -= learning_rate * grads['dby']    
    return params

#---------------------Training ------------------------------
epoch =100
min_delta = 1e-4  # Minimum change in loss to count as an improvement    
batch_size =64
    
Max, Min, C ={},{},{}
C_avg, Avg, train_losses, val_losses, grad_norm  = [],[],[],[],[]

# ===  Training Loop ===
best_loss = float('inf')          # Start with a very high loss
patience = 10                     # Number of epochs to tolerate no improvement
convergence_start_epoch = None    # The epoch where convergence starts
convergence_reached = False       # Flag to check if convergence is already achieved
start_time = time.time() 
t = 0    

for i in range(1,epoch+1):    
    Avg=[]      
    flattened_grads = []
    train_array=[]
    mini_batches = create_minibatches(train_X, train_Y,batch_size)
    
    for X_batch, Y_batch in mini_batches:              
        act = forwardpropagation(X_batch, params) 
                
        # Compute training loss for this mini-batch
        train_loss=compute_cross_entropy(np.array(act[f'H{T}_{L+1}']), Y_batch)
        train_array+=[train_loss]          # Saving values of all mini batches
                
        grads = backwardpropagation(X_batch, Y_batch, act, params)
        params = updateparams(grads, learning_rate,params,L) 
    
    train_avg=np.mean(train_array)         # Avg of all mini batches     
    # Compute validation loss after each epoch
    
    # === Check for Convergence if it hasn't been reached yet ===
    if not convergence_reached:            # Only check convergence if it hasn't been reached
       if best_loss - train_avg > min_delta:
          best_loss = train_avg
          patience_counter = 0
       else:
          patience_counter += 1
       # Check if convergence is detected   
       if patience_counter == patience and convergence_start_epoch is None:
             convergence_start_epoch = i - patience + 1
             converged_time = time.time() - start_time
             convergence_reached = True  # Set flag to stop further convergence checks
             print(f"Convergence detected at epoch {convergence_start_epoch}")
             
    Y_val_pred = forwardpropagation(val_X,params)
    val_loss = compute_cross_entropy(np.array(Y_val_pred[f'H{T}_{L+1}']), val_Y)
        
        
    # Store loss values
    train_losses.append(train_avg)  # You can use last mini-batch loss or average
    val_losses.append(val_loss)
    
    #----------Print Progress---------
    with open(f"{base_path}\\output.txt", "a") as f:
       f.write(f"Epoch {i}/{epoch}  | Training Losses: {train_avg:.4f} | Validation Loss: {val_loss:.4f}\n")
    print(f"Epoch {i}/{epoch}  | Training Losses: {train_avg:.4f} | Validation Loss: {val_loss:.4f}")
    
    for l in range(1, L+2): 
        Max[f"M{l}"]=np.max(act[f"Z{T}_{l}"],axis=0)  
        Min[f"m{l}"]=np.min(act[f"Z{T}_{l}"],axis=0)
        C[f"C{l}"]=Max[f"M{l}"]-Min[f"m{l}"]
        average=np.mean(C[f"C{l}"])
        Avg.append(average.reshape(1,-1))
        
        if l<L+1:
            flattened_grads.append(grads[f'dWhh{l}'].flatten().T) # Flatten the gradient and add it to the list
            flattened_grads.append(grads[f'dWxh{l}'].flatten().T)
            flattened_grads.append(grads[f'dbh{l}'].flatten())
        else:
            flattened_grads.append(np.asarray(grads['dWhy']).flatten())# Flatten the gradient and add it to the list
            flattened_grads.append(grads['dby'].flatten())
            
    concatenated_C= np.concatenate(Avg)
    C_avg+=[ concatenated_C.T]
    concatenated_grads = np.concatenate(flattened_grads) 
    grad_norm += [np.linalg.norm(concatenated_grads)/len(concatenated_grads)]


with open(f"{base_path}\\val_grad.csv", "w", newline='') as file:
    writer = csv.writer(file)    
    writer.writerow(["epoch","train_loss", "val_loss", "grad_norm"])   # Write header
    for i in range(len(val_losses)):
        writer.writerow([i+1,train_losses[i], val_losses[i], grad_norm[i]])  # Write rows with epoch starting from 1        

#---------- Saving C values ------------------
C_array = np.squeeze(np.array(C_avg))                      # Convert to a (100, 3) array, shape becomes (100, 3)
headers = ["Epoch"] + [f"{i + 1} Layer" for i in range(L)] # Generate headers for saving C avg values
rows_with_epoch = [[epoch + 1] + list(C_array[epoch]) for epoch in range(epoch)] # Add epoch numbers as first column
# Write to CSV
with open(f"{base_path}\\cvalues.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)           # Write headers
    writer.writerows(rows_with_epoch)  # Write data rows       

#------Calculate total time after all epochs--------
end_time_total=time.time()                 # End time for total training
total_time=end_time_total-start_time       # Total time for all epochs

#------ Converegnce Result for Train Loss------
print(f"\nFinal Train Loss: {train_losses[-1]:.4f}")
print(f"\nFinal Vali. Loss: {val_losses[-1]:.4f}")
if convergence_reached:
    print(f"\nConverged at Epoch: {convergence_start_epoch}")
    print(f"\nTime to Convergence: {converged_time:.2f} seconds")
else:
    print("\nTrain loss did not converge within given patience.")
print(f"\nTotal Training Time: {total_time:.2f} seconds")

# ----To print the converegence epoch C Values-----
if convergence_reached is True:
    print("Convege epoch Averge C values of all Layers",C_avg[convergence_start_epoch-1])
print(f"Gradient Norm after the last epoch: {grad_norm[-1]:.4f}")   

#--------------Testing---------------
act=forwardpropagation(test_X,params)

# --Convert to Class Labels
Y_pred_classes = np.argmax(act[f'H{T}_{L+1}'], axis=1)  # predicted labels
Y_true_classes = np.argmax(test_Y, axis=1)        # actual labels

accuracy = np.mean(Y_pred_classes == Y_true_classes)
test_loss = compute_cross_entropy(act[f'H{T}_{L+1}'], test_Y)


# Macro-average
# Macro-average with zero_division set to 1 (handle undefined cases)
precision_macro = precision_score(Y_true_classes, Y_pred_classes, average='macro',zero_division=0)
recall_macro = recall_score(Y_true_classes, Y_pred_classes, average='macro',zero_division=0)

# Micro-average
# Micro-average with zero_division set to 1 (handle undefined cases)
precision_micro = precision_score(Y_true_classes, Y_pred_classes, average='micro',zero_division=0)
recall_micro = recall_score(Y_true_classes, Y_pred_classes, average='micro',zero_division=0)

# Print results
print(f"Macro Precision: {precision_macro}")
print(f"Macro Recall: {recall_macro}")
print(f"Micro Precision: {precision_micro}")
print(f"Micro Recall: {recall_micro}")       
#-------------------------------------------------------
# Per-class metrics and F1 scores
print("\nClassification Report:")
print(classification_report(Y_true_classes, Y_pred_classes, digits=4,zero_division=0))
# F1 Macro
f1_macro = f1_score(Y_true_classes, Y_pred_classes, average='macro',zero_division=0)

# F1 Micro
f1_micro = f1_score(Y_true_classes,Y_pred_classes, average='micro',zero_division=0)

print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")    

f1_weighted = f1_score(Y_true_classes,Y_pred_classes, average='weighted',zero_division=0) 

#-----Confusion matrix ------
cm = confusion_matrix(Y_true_classes, Y_pred_classes, labels=range(CLS))
class_labels = [f'class {i}' for i in range(CLS)]
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

plt.figure(1,figsize=(20,10))
ax=sns.heatmap(cm_df, annot=True, fmt='d',cmap='Reds',cbar=True, annot_kws={"size": 16, "weight": "bold"})
plt.title('Confusion Matrix',fontsize=16)
plt.ylabel('Actual Values',fontsize=16)
plt.xlabel('Predicted Values',fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=13)
plt.savefig(f"{base_path}\\confmat.jpeg", dpi =300)
plt.show()

# MEMORY REPORT
current, peak = tracemalloc.get_traced_memory()
print(f"\n🧠 Current memory usage: {current / 10**6:.2f} MB")
print(f"🚀 Peak memory usage:    {peak / 10**6:.2f} MB")


#-------------- Save everything in one file txt file------------------ 
with open(f"{base_path}\\results.txt", "w") as f:
    f.write(f"Layer dimensions:,{layer_dims}\n")
    f.write(f"Learning Rate:{learning_rate}\n")
    f.write(f"Macro Precision: {precision_macro:.4f}\n")
    f.write(f"Macro Recall:    {recall_macro:.4f}\n")
    f.write(f"Micro Precision: {precision_micro:.4f}\n")
    f.write(f"Micro Recall:    {recall_micro:.4f}\n")
    f.write(f"F1 Macro:        {f1_macro:.4f}\n")
    f.write(f"F1 Micro:        {f1_micro:.4f}\n")

    f.write(f"\nFinal Train Loss: {train_losses[-1]:.4f}\n")
    f.write(f"\nFinal Val Loss: {val_losses[-1]:.4f}\n")
    
    f.write(f"\n Current memory usage: {current / 10**6:.2f} MB\n")
    f.write(f" Peak memory usage:    {peak / 10**6:.2f} MB\n")
   

    if convergence_reached:
        f.write(f"\nConverged at Epoch: {convergence_start_epoch}\n")
        f.write(f"Time to Convergence: {converged_time:.2f} sec\n")
        f.write(f"C Avg Values at Convergence: {C_avg[convergence_start_epoch - 1]}\n")
    else:
        f.write("\nTraining did not converge within patience.\n")

    f.write(f"Total Training Time: {total_time:.2f} sec\n")
    f.write(f"Gradient Norm (last epoch): {grad_norm[-1]:.4f}\n")
    

    # Classification Report
    report = classification_report( Y_true_classes,Y_pred_classes, digits=4, zero_division=0)
    f.write("\n Classification Report:\n")
    f.write(report)
    



