import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast


def plot_melody():
    # File path
    coop = True

    if coop:
        file_path = "results/data/melody/coop/training_data100_2.json"
    else:
        file_path = "results/data/melody/non_coop/training_data100_2.json"

    title = "Melody Agent coop - Loss" if coop else "Melody Agent non coop - Loss"

    # Reading the data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extracting training and validation loss
    loss_list = data["loss_list"]
    val_loss_list = data["val_loss_list"]

    epochs = range(1, len(loss_list) + 1)

    # Plotting with thicker lines
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, color="skyblue", linewidth=2, label="Training loss")
    plt.plot(
        epochs,
        val_loss_list,
        color="lightgreen",
        linewidth=2,
        label="Validation loss",
    )
    plt.title(title, fontsize=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    sns.despine()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_chord():
    # file_path = "results/data/chord/non/training_data50.json"
    file_path = "results/data/chord/non_coop/training_data50.json"

    loss_list = []
    val_loss_list = []

    with open(file_path, "r") as file:
        data_string = file.read()
        # Splitting the string into two parts and stripping the square brackets
        parts = data_string.strip("[]").split("][")
        # Converting string representations of lists into actual Python lists
        loss_list = ast.literal_eval("[" + parts[0] + "]")
        val_loss_list = ast.literal_eval("[" + parts[1] + "]")

    loss_list = loss_list[2:]
    val_loss_list = val_loss_list[2:]
    # Data for plotting
    epochs = range(1, len(loss_list) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, color="skyblue", linewidth=2, label="Training loss")
    plt.plot(
        epochs,
        val_loss_list,
        color="lightgreen",
        linewidth=2,
        label="Validation loss",
    )
    plt.title("Training and validation loss", fontsize=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_bass():
    file_path = "results/data/bass/training_data50.json"

    loss_list = []
    val_loss_list = []

    with open(file_path, "r") as file:
        data_string = file.read()
        parts = data_string.strip("[]").split("][")
        loss_list = ast.literal_eval("[" + parts[0] + "]")
        val_loss_list = ast.literal_eval("[" + parts[1] + "]")

    # Data for plotting
    epochs = range(1, len(loss_list) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, color="skyblue", linewidth=2, label="Training loss")
    plt.plot(
        epochs,
        val_loss_list,
        color="lightgreen",
        linewidth=2,
        label="Validation loss",
    )
    plt.title("Training and validation loss", fontsize=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


plot_melody()
