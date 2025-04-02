# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift # type: ignore[import]
from cyvlfeat.kmeans import kmeans # type: ignore[import]
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []
    for path in tqdm(img_paths):
        img = Image.open(path)
        # Make it grayscale
        img = img.convert('L')
        # Resize to 16x16
        img = img.resize((16, 16), Image.LANCZOS)
        # Convert to numpy array and flatten
        img = np.array(img).flatten()
        # Normalize the image
        img_1D = img - np.mean(img)
        norm = np.linalg.norm(img_1D)
        if norm > 0:
            img_1D /= norm
        # Append to the list of tiny image features
        tiny_img_feats.append(img_1D)  
        
    # Convert to numpy array
    tiny_img_feats = np.array(tiny_img_feats)
        
    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################


###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    step_sample = 1
    descriptors_list = []
    
    for path in tqdm(img_paths):
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
        
        # Convert to grayscale
        gray_img = img.convert('L')
        # Convert to numpy array
        gray_np = np.array(gray_img, dtype=np.uint8)

        # Apply dsift to get SIFT features
        _, descriptors = dsift(gray_np, step=[step_sample, step_sample], fast=True)
        
        # If descriptors are not empty, append them to the list
        if descriptors is not None and len(descriptors) > 0:
            descriptors_list.append(descriptors)
            
    if len(descriptors_list) == 0:
        raise ValueError("No descriptors found. Check your image paths or SIFT extraction.")
        
    all_descriptors = np.vstack(descriptors_list) if descriptors_list else None
    vocab = kmeans(all_descriptors, num_centers=vocab_size)

    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab


###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    
    step_sample = 1
    vocab_size = vocab.shape[0]
    
    for path in tqdm(img_paths):
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
    
        # convert to grayscale
        gray_img = img.convert('L')
        
        # convert to numpy array
        gray_np = np.array(gray_img, dtype=np.uint8)
        
        # Apply dsift to get SIFT features
        _, descriptors = dsift(gray_np, step=[step_sample, step_sample], fast=True)
        
        if descriptors is not None and len(descriptors) > 0:
            # Calculate the distances between descriptors and vocab
            distances = cdist(descriptors, vocab, metric='minkowski', p=2)
            # Assign each local feature to its nearest cluster center
            nearest_clusters = np.argmin(distances, axis=1)
            # Accordingly, build a histogram indicating how many times each cluster presents
            histogram, _ = np.histogram(nearest_clusters, bins=np.arange(vocab_size + 1)) # We need vocab_size bins, so we use vocab_size + 1
            # Normalize the histogram by number of features
            if len(descriptors) > 0:
                histogram = histogram / len(descriptors)
        else:
            histogram = np.zeros(vocab_size)
        
        img_feats.append(histogram)
        
    img_feats = np.array(img_feats)

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################
    k = 5  # Number of nearest neighbors to consider
    test_predicts = []
    
    distances = cdist(test_img_feats, train_img_feats, metric='minkowski', p=2)
    
    # Sort the distances and get the indices of the k nearest neighbors
    sorted_indices = np.argsort(distances, axis=1)
    k_nearest_indices = sorted_indices[:, :k]
    
    '''
    distances = [
    [2.3, 1.5, 3.0],   # 測試影像1 與三個訓練影像的距離
    [0.9, 1.2, 0.8]    # 測試影像2 與三個訓練影像的距離
    ]
    sorted_indices = [
    [1, 0, 2],
    [2, 0, 1]
    ]
    '''
    
    for i in range(len(test_img_feats)):
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [train_labels[idx] for idx in k_nearest_indices[i]]
        
        # Count the occurrences of each label
        label_counts = np.bincount([CAT2ID[label] for label in k_nearest_labels], minlength=len(CAT))
        
        # Get the label with the highest count (vote)
        predicted_label_id = np.argmax(label_counts)
        
        # Convert back to string using CAT
        predicted_label = CAT[predicted_label_id]
        
        test_predicts.append(predicted_label)

    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts
