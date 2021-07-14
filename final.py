
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
import functions.KNN_predict as knn
from functions.Standardize import center
import itertools as itertools
import multiprocessing
from scipy.spatial import KDTree

def main(train_data_location,test_data_location,k,number_of_pcs,knn_method, distance_method = "euclidean"):

    # loading data:
    train_labels, train_values = load_the_pickle(train_data_location)
    test_labels, test_values = load_the_pickle(test_data_location)

    # standardization and PCA:
    train_values_centered, train_mean = center(train_values)
    train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

    test_values_centered, test_mean = center(test_values, Y=train_values)
    test_values_pca, _ = PCA_func(test_values_centered,test_mean, number_of_pcs, train_evs=train_evs)

    # kNN:
    hit = 0
    miss = 0
    if knn_method == "kdtree":
        tree = KDTree(train_values_pca)

    if __name__ == '__main__':
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            if knn_method == "kdtree":
                result = p.starmap(knn.kdtree_knn, zip(test_values_pca[range(10000), :], itertools.repeat(k),itertools.repeat(train_labels), itertools.repeat(tree), itertools.repeat(distance_method)),chunksize=500)
            elif knn_method == "weighted_knn":
                result = p.starmap(knn.weighted_knn, zip(itertools.repeat(distance_method), itertools.repeat(train_values_pca),itertools.repeat(train_labels),test_values_pca[range(10000),:],itertools.repeat(k)),chunksize=500)
            elif knn_method == "traditional":
                result = p.starmap(knn.knn, zip(itertools.repeat(distance_method), itertools.repeat(train_values_pca),itertools.repeat(train_labels),test_values_pca[range(10000),:],itertools.repeat(k)),chunksize=500)
            for sample in range(10000):
                if result[sample] == test_labels[sample]:
                    hit += 1
                else:
                    miss +=1
            print(hit, "vs", miss)


main("data/train_points.p","data/test_points.p",4,51,"weighted_knn")



