import random
import torch
import os
import torch.nn as nn
import torch.optim as optim
import heapq
import argparse
import matplotlib.pyplot as plt

# contextual - 2 layers
# projection - 3 layers

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, attr_dim=0, contextual_attention=False):
        super(NeuralNetwork, self).__init__()
        if contextual_attention:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim + attr_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, 1)
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim + attr_dim, input_dim + attr_dim),
                nn.LeakyReLU(),
                nn.Linear(input_dim + attr_dim, input_dim),
                nn.LeakyReLU(),
                nn.Linear(input_dim, input_dim)
            )

    def forward(self, x):
        return self.linear_relu_stack(x)


def compare(L1, L2, p_confusion):
    with torch.no_grad():
        u = torch.ones_like(L1).float() * (1 - p_confusion)
        strided_vect = torch.bernoulli(u)
        sim_vect = 2 * (L1 == L2) - 1
        feedback_vector = sim_vect * strided_vect
        return feedback_vector


def getAttribute(dataset, image):
    path = "../Datasets/" + dataset + "/labels/" + str(image)
    return torch.load(path)


def getRepresentation(dataset, image):
    path = "../Datasets/" + dataset + "/resnet_representations/" + str(image)
    return torch.load(path)


def distance(a, b):
    return torch.mean((a - b) ** 2)


# def penalty_function(iteration_number, desired_iterations, penalty_factor = None):  #Will fill in the penalty factor in later
#     """
#     This function calculates the penalty function for the
#     iteration process.
#     :param iteration_number: The current iteration number.
#     :param desired_iterations: The desired number of iterations.
#     :param penalty_factor: The penalty factor (1 gives linear and radicalization increases with the number).
#     """
#     if iteration_number > desired_iterations:
#         return iteration_number * (penalty_factor ** (iteration_number - desired_iterations))
#     else:
#         return (-1 * iteration_number * (penalty_factor ** (iteration_number - desired_iterations))) + (
#                 2 * desired_iterations)

def penalty_function(iteration_number, desired_iterations):
    if iteration_number < desired_iterations:
        return 1 + 2 * iteration_number / desired_iterations
    else:
        return 1 + 2 * (iteration_number / desired_iterations) ** 3


def get_initial_images(database, num_images):
    return random.sample(database, num_images)


def contextual_attention(context_vec, image_vectors, contextual_attention_net):
    total_input = torch.cat([image_vectors, context_vec.repeat(image_vectors.size(0), 1)], axis = 1)
    scores = contextual_attention_net(total_input)
    weights = nn.Softmax(dim=0)(scores)
    return torch.sum(image_vectors * weights.repeat(1, image_vectors.size(1)), axis=0, keepdim = True)


def pairwise_triplet_loss(anchor, positive, negatives):
    triplet_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    total_loss = 0
    for negative in negatives:
        total_loss += triplet_margin_loss(anchor, positive, negative)
    return total_loss / len(negatives)


def knn(query_vector, k, dataset, representations_images):
    # Discrete KNN Case
    scores = [(-1 * distance(query_vector, representations_images[i]), i) for i in dataset]
    if len(scores) > k:
        scores = heapq.nlargest(k, scores)
    return [a[1] for a in scores]


def train(dataset_name: str, max_count: int, num_images: int, desired_iteration: int, epochs_per_curricula: int):
    full_dataset = os.listdir(f"../Datasets/{dataset_name}/labels")
    representations_images = {}
    labels_images = {}
    for img_name in full_dataset:
        representations_images[img_name] = getRepresentation(dataset_name, img_name).unsqueeze(0)
        representations_images[img_name].requires_grad = False
        labels_images[img_name] = getAttribute(dataset_name, img_name).unsqueeze(0)
        labels_images[img_name].requires_grad = False

    representation_dim = representations_images[list(representations_images.keys())[0]].size(1)
    attribute_dim = labels_images[list(labels_images.keys())[0]].size(1)

    p_confusion_curriculum = [0, 0.01, 0.05, 0.1, 0.175, 0.25, 0.5, 0.75]

    query_projection = NeuralNetwork(representation_dim, attribute_dim)
    contextual_attention_net = NeuralNetwork(2 * representation_dim, 0, contextual_attention=True)
    optimizer = optim.Adam(list(contextual_attention_net.parameters()) + list(query_projection.parameters()))

    D = penalty_function(max_count, desired_iteration)

    iterations_over_training = []

    for p_confusion in p_confusion_curriculum:
        for epoch in range(epochs_per_curricula):
            for target_image in full_dataset:
                print(f"p_confusion {p_confusion} Epoch {epoch} Target {target_image}")
                target_rep = representations_images[target_image]
                recomm_images = get_initial_images(full_dataset, num_images)
                queries = {}
                queries[0] = torch.mean(torch.cat([representations_images[image] for image in recomm_images], axis = 0), axis = 0)
                iter_counter = 0
                curr_dataset = set(full_dataset)
                while target_image not in recomm_images and iter_counter < max_count:
                    iter_counter += 1
                    query_vec_part = {}
                    similarity_vec = {}
                    for candidate_image in recomm_images:
                        representation_image = representations_images[candidate_image]
                        similarity_vec[candidate_image] = compare(labels_images[candidate_image],
                                                                labels_images[target_image],
                                                                p_confusion)
                        query_vec_part[candidate_image] = query_projection(torch.cat([representation_image, similarity_vec[candidate_image]], axis = 1))
                    queries[iter_counter - 1].requires_grad = False
                    queries[iter_counter] = contextual_attention(queries[iter_counter - 1],
                                                                    torch.cat([query_vec_part[candidate_image] for
                                                                    candidate_image in
                                                                    recomm_images], axis = 0), contextual_attention_net)
                    loss = penalty_function(iter_counter + 1, desired_iteration)/D * pairwise_triplet_loss(queries[iter_counter], target_rep, [representations_images[image] for image in recomm_images])
                    optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    optimizer.step()
                    curr_dataset = curr_dataset.difference(set(recomm_images))
                    recomm_images = knn(queries[iter_counter], num_images, curr_dataset, representations_images)
                    torch.save(query_projection.state_dict(), "./saved_models/query_projection.pt")
                    torch.save(contextual_attention_net.state_dict(), "./saved_models/contextual_attention.pt")
                iterations_over_training.append(iter_counter)

    plt.plot(list(range(len(iterations_over_training))), iterations_over_training)
    plt.xlabel("Training Iteration Number")
    plt.ylabel("Target Image Found in Rounds")
    plt.savefig("./saved_models/iteration_vs_rounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--dataset_name', type = str, default = "CelebA", help = "Name of the dataset")
    parser.add_argument('--desired_iterations', type = int, default = 5, help = "Iterations in which most of the images should be found")
    parser.add_argument('--max_count', type = int, default = 20, help="Max number of iterations till we train")
    parser.add_argument('--num_images', type = int, default = 16, help = "Number of images recommended per iteration")
    parser.add_argument('--epochs_per_curricula', type = int, default = 5, help = "Epochs for each value of p curriculum")

    FLAGS = parser.parse_args()

    train(FLAGS.dataset_name, FLAGS.max_count, FLAGS.num_images, FLAGS.desired_iterations, FLAGS.epochs_per_curricula)