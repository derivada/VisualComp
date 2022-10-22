from cProfile import run
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer
from PIL import Image
torch.random.manual_seed(0)
np.random.seed(0)


# Load the dataset and print some stats
batch_size = 64

image_transform_normalized = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = 0.5, std = 0.5), # Add image normalization
])

image_transform_unnormalized = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform_normalized)
example_images, _ = next(iter(dataloader_train))
print(f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}", sep="\n\t")


def create_model():
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.Linear(28*28*1, 10)
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model

def create_model_hidden_layer():
    """
        Initializes the model with a hidden layer of 64 nodes in the network and ReLU as its activation function
    """
    model = nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.Linear(28*28*1, 64), # 28x28 input, 64 inputs to hidden layer 
        nn.ReLU(), # ReLU for next layer
        nn.Linear(64, 10) # Hidden layer with 64 inputs and 10 outputs
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model = create_model()

# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"


# Hyperparameters
learning_rate = .0192
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)

trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict, test_loss_dict = trainer.train(num_epochs)

# Task 4b)
weight = list(model.children())[1].weight.cpu().data # Get the weights data
for i in range(10):
    data = weight[i].reshape([28, 28]).numpy() # Convert the weights of a digit to a 28x28 numpy ndarray
    normalized_data = (((data - data.min()) / (data.max() - data.min())) * 255.9).astype(np.uint8) # Nrmalize the data to fit the 8-bit range of grayscale images
    img = Image.fromarray(normalized_data).convert("L") # Create the image using PIL and convert it to grayscale mode
    img.save("image_solutions/task_4b_weights{}.png".format(i)) # Save the image to the output folder

# We can now plot the training loss with our utility script
'''
# Plot loss
utils.plot_loss(train_loss_dict, label="Train Loss")
utils.plot_loss(test_loss_dict, label="Test Loss")
# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4a.png")

plt.show()
'''

torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")

# Comparison with other models: tasks 4a, 4c and 4d
def run_task(task):
    print('---------------------')
    print('   Running task 4{}   '.format(task))
    print('---------------------')

    # Reset random seed
    torch.random.manual_seed(0)
    np.random.seed(0)
    
    # Task 4a variation: unnormalized image
    if task == 'a':
        dataloader_train, dataloader_test = dataloaders.load_dataset(batch_size, image_transform_unnormalized)
    else:
        dataloader_train, dataloader_test = dataloaders.load_dataset(batch_size, image_transform_normalized)
    
    # Task 4c variation: high learning rate of 1.0
    if task == 'c':
        learning_rate = 1.0
    else:
        learning_rate = .0192 # Original lr

    # Task 4d variation: model with hidden layer
    if task == 'd':
        model = create_model_hidden_layer() # Task 4d model
    else:
        model = create_model() # Original model


    num_epochs = 5

    # Redefine optimizer, as we have a new model.
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        batch_size=batch_size,
        loss_function=loss_function,
        optimizer=optimizer
    )

    train_loss_dict_new, test_loss_dict_new = trainer.train(num_epochs)

    # We can now plot the two models against eachother
    def plot_model_comparison(new_model_name, output_file_name, ylim):
        # Plot loss
        utils.plot_loss(train_loss_dict_new,
        label="Train Loss - {}".format(new_model_name))
        utils.plot_loss(test_loss_dict_new,
        label="Test Loss - {}".format(new_model_name))

        utils.plot_loss(train_loss_dict, label="Train Loss - Original model")
        utils.plot_loss(test_loss_dict, label="Test Loss - Original model")

        # Limit the y-axis of the plot (The range should not be increased!)
        plt.ylim(ylim)
        plt.legend()
        plt.xlabel("Global Training Step")
        plt.ylabel("Cross Entropy Loss")

        plt.savefig("image_solutions/{}.png".format(output_file_name))
        # plt.show()

        torch.save(model.state_dict(), "saved_model_{}.torch".format(output_file_name))
        final_loss, final_acc = utils.compute_loss_and_accuracy(
            dataloader_test, model, loss_function)
        print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")

    # Generate task 4a, 4c and 4d outputs
    if task == 'a':
        plot_model_comparison('Model with unnormalized image', 'task_4a', [0, 1])
    elif task == 'c':
        plot_model_comparison('Model with learning rate = 1.0', 'task_4c', [0, 10])
    elif task == 'd':
        plot_model_comparison('Model with ReLU hidden layer', 'task_4d', [0, 1])
    print('---------------------')
    print('    Task 4{} done    '.format(task))
    print('---------------------')

# run_task('a')
# run_task('c')
run_task('d')