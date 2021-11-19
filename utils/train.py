import torch
from model.sequencetosequence import SequenceToSequenceModel


def train(model: SequenceToSequenceModel, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer) -> int:
    """
    Train the model through a single epoch

    Params:
    model (SequenceToSequence): Sequence to Sequence model to train
    dataloader (torch.utils.data.DataLoader): The loader containing training data

    Returns:
    loss (int): Average batch loss
    """
    total_batch_loss = 0

    # Run through the example
    for batch_index, (source_sentences, target_sentences, source_lengths) in enumerate(dataloader):
        # get the batch size
        batch_size = len(source_lengths)

        # Zero out the gradient
        optimizer.zero_grad()

        # get the loss for each sentences in the batch
        # losses is a tensor with size (batch_size,)
        losses = model(source_sentences, target_sentences, source_lengths)

        # sum up the loss to get batch_loss
        batch_loss = losses.sum()

        # divide by the number of sentences to find batch_size
        loss = batch_loss / batch_size
        loss.backward() # Perform gradient descent

        # add to total batch_loss
        total_batch_loss += batch_loss.item()

        # run optimizer
        optimizer.step()

        # Display the loss
        print(f"Batch {batch_index}, Batch loss: {batch_loss.item()}")

    # Divide total batch loss with number of batch to get average batch loss
    avg_batch_loss = total_batch_loss / len(dataloader)

    return avg_batch_loss


def evaluate(model: SequenceToSequenceModel, dataloader: torch.utils.data.DataLoader) -> int:
    """
    Evaluate the model through a evaluation dataset

    Params:
    model (SequenceToSequence): Sequence to Sequence model to train
    dataloader (torch.utils.data.DataLoader): The loader containing evaluation data

    Returns:
    loss (int): Average batch loss for evaluation data
    """
    total_batch_loss = 0

    # Run through the example
    with torch.no_grad():
        for batch_index, (source_sentences, target_sentences, source_lengths) in enumerate(dataloader):
            # losses is a tensor with size (batch_size,)
            losses = model(source_sentences, target_sentences, source_lengths)

            # sum up the loss to get batch_loss
            batch_loss = losses.sum()

            # add to total batch_loss
            total_batch_loss += batch_loss.item()

    # Divide total batch loss with number of batch to get average batch loss
    avg_batch_loss = total_batch_loss / len(dataloader)

    return avg_batch_loss


def train_model(model: SequenceToSequenceModel, 
    train_dataloader: torch.utils.data.DataLoader, 
    eval_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    initial_learning_rate: float) -> SequenceToSequenceModel:
    """
    Train a sequence to sequence model through epochs

    Params:
    model (SequenceToSequenceModel): sequence to sequence model to train
    train_dataloader (DataLoader): containing training data
    eval_dataloader (DataLoader): containing evaluation data
    epochs (int): the number of epochs to train
    initial_learning_rate (float): the initial learning rate of the model

    Returns:
    model (SequenceToSequenceMode): the trained model
    """
    # Createe SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate)

    # Start training loop
    for epoch in range(epochs):
        # Calculate the train and eval loss for this epoch
        avg_train_loss = train(model, train_dataloader, optimizer)
        avg_eval_loss = evaluate(model, eval_dataloader)

        # Display the loss
        print(f"Epoch {epoch}, Training loss {avg_train_loss}, Evaluation loss {avg_eval_loss}")

        # Save the weights
        torch.save(model.state_dict(), "model.bin")

    return model