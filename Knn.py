from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from analysis import acquisition_func
import math

"""Find and calculate neighbors"""
def Findneigh(pointcloud, pred, num_neighbors, inner_num_neighbors, alg):
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm=alg).fit(pointcloud)
    distances, indices = nbrs.kneighbors(pointcloud)#邻域距离与编号
    choice = np.random.choice(60, 60, replace=False)
    indices.T[0:60,:][choice,:].T
    pred_neigh=pred[indices.T[0:60,:][choice,:].T]#邻域标签
    pred_neigh_sum=pred_neigh[:,0:60].reshape([pred_neigh.shape[0],10,6])
    pred_vars=np.array([np.var([np.argmax(np.bincount(pred_neigh_sum[i, j, :])) for j in range(pred_neigh_sum.shape[1])]) for i in range(pred_neigh_sum.shape[0])])
    return np.floor(100 * pred_vars / (pred_vars.max() - pred_vars.min())).astype('int64')

def Findneigh_withoutpart(pointcloud, output, num_neighbors, alg):
    NumClass=np.shape(output)[1]
    output_mean = output * 0  # for pred_label
    output_square = output * 0  # for unct_square
    entropy_mean = output.mean(1) * 0
    points = pointcloud[:, 0:3]
    output_s = output
    Num= num_neighbors if points.shape[0]>=num_neighbors else points.shape[0]
    #print(str(i) + ';' + str(points.shape[0])+ ';'+str(Num))
    if Num is not 0:
        nbrs = NearestNeighbors(n_neighbors=Num, algorithm=alg, n_jobs=-1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        Tempt=output_s[indices.reshape(1,-1)].reshape(-1,Num,NumClass)
        output_mean = torch.sum(Tempt, 1)/Num
        output_square = torch.sum(Tempt**2, 1)/Num
        entropy_mean = torch.sum(acquisition_func('e1', Tempt), 1)/Num
    return output_mean, output_square, entropy_mean

def Findneigh_withpart(pointcloud, output, partlabel,num_neighbors, alg):
    NumClass=np.shape(output)[1]
    output_mean = output * 0  # for pred_label
    output_square = output * 0  # for unct_square
    entropy_mean = output.mean(1) * 0
    datamax=np.max(partlabel).astype(int)
    for i in range(datamax+1):
        rowi = np.argwhere(partlabel == i).flatten()
        points = pointcloud[rowi, 0:6]
        output_s = output[rowi,:]
        Num= num_neighbors if points.shape[0]>=num_neighbors else points.shape[0]
        #print(str(i) + ';' + str(points.shape[0])+ ';'+str(Num))
        if Num is not 0:
            nbrs = NearestNeighbors(n_neighbors=Num, algorithm=alg, n_jobs=-1).fit(points)
            distances, indices = nbrs.kneighbors(points)
            Tempt=output_s[indices.reshape(1,-1)].reshape(-1,Num,NumClass)
            output_mean[rowi] = torch.sum(Tempt, 1)/Num
            output_square[rowi] = torch.sum(Tempt**2, 1)/Num
            entropy_mean[rowi] = torch.sum(acquisition_func('e1', Tempt), 1)/Num
    return output_mean, output_square, entropy_mean

def Findneigh_withsimilar(pointcloud, output, num_neighbors, alg):
    NumClass=np.shape(output)[1]
    NumFea = 3
    output_mean = output * 0  # for pred_label
    output_square = output * 0  # for unct_square
    entropy_mean = output.mean(1) * 0
    points = pointcloud[:, 0:3]
    #points = (pointcloud - np.min(pointcloud,1).reshape(-1,1))/(np.max(pointcloud,1) - np.min(pointcloud,1)).reshape(-1,1)
    output_s = output
    Num= num_neighbors if points.shape[0]>=num_neighbors else points.shape[0]
    if Num is not 0:
        nbrs = NearestNeighbors(n_neighbors=Num, algorithm=alg, n_jobs=-1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        Tempt=output_s[indices.reshape(1,-1)].reshape(-1,Num,NumClass)
        Tempt2 = points[indices.reshape(1,-1)].reshape(-1,Num,NumFea)
        Temptfz = np.sum(Tempt2 * (Tempt2[:,0,:].reshape(-1,1,NumFea)),2)
        Temptfm = np.sqrt(np.sum((Tempt2[:,0,:].reshape(-1,1,NumFea))**2,2)) * np.sqrt(np.sum(Tempt2**2,2))
        similar=Temptfz/Temptfm
        similar = 0.5 * ((similar - np.min(similar, 1).reshape(-1, 1)) / (
                    np.max(similar, 1).reshape(-1, 1) - np.min(similar, 1).reshape(-1, 1)) + 1)
        similar=similar/np.sum(similar,1).reshape(-1,1)
        Tempt=Tempt*torch.tensor(similar.reshape(-1,Num,1))
        output_mean = torch.sum(Tempt, 1)/Num
        output_square = torch.sum(Tempt**2, 1)/Num
        entropy_mean = torch.sum(acquisition_func('e1', Tempt), 1)/Num
    return output_mean, output_square, entropy_mean


def Findneigh_withpartpro(pointcloud, output, partlabel,num_neighbors, alg):
    NumClass=np.shape(output)[1]
    output_mean = output * 0  # for pred_label
    output_square = output * 0  # for unct_square
    entropy_mean = output.mean(1) * 0
    datamax=np.max(partlabel).astype(int)
    for i in range(datamax+1):
        rowi = np.argwhere(partlabel == i).flatten()
        points = pointcloud[rowi, 0:6]
        output_s = output[rowi,:]
        Num= num_neighbors if points.shape[0]>=num_neighbors else points.shape[0]
        #print(str(i) + ';' + str(points.shape[0])+ ';'+str(Num))
        if Num is not 0:
            nbrs = NearestNeighbors(n_neighbors=Num, algorithm=alg).fit(points)
            distances, indices = nbrs.kneighbors(points)
            Tempt = output_s[indices.reshape(1,-1)].reshape(-1,Num,NumClass)
            Temptpro = output_s.mean(0)
            output_mean[rowi] = (torch.sum(Tempt, 1)/Num + Temptpro)/2
            output_square[rowi] = (torch.sum(Tempt**2, 1)/Num + Temptpro**2)/2
            entropy_mean[rowi] = (torch.sum(acquisition_func('e1', Tempt), 1)/Num + acquisition_func('e', output_s))/2
    return output_mean, output_square, entropy_mean

def uncertoutput_withpart(pointcloud, output, partlabel):
    output_mean = output * 0  # for pred_label
    output_square = output * 0  # for unct_square
    entropy_mean = output.mean(1) * 0
    datamax=np.max(partlabel).astype(int)
    for i in range(datamax+1):
        rowi = np.argwhere(partlabel == i).flatten()
        points = pointcloud[rowi, 0:6]
        output_s = output[rowi,:]
        Num = points.shape[0]
        if Num is not 0:
            output_mean[rowi] = torch.sum(output_s,0)/Num
            output_square[rowi] = torch.sum(output_s ** 2, 0) / Num
            entropy_mean[rowi] = torch.sum(acquisition_func('e', output_s), 0) / Num
    return output_mean, output_square, entropy_mean


"""Calculate neighbors_Knn"""
def Calneigh(pred):
    classCount = {}
    for i in range(len(pred)):
        classCount[pred[i]] = classCount.get(pred[i], 0) + 1
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            label = key
    return label

if __name__ == '__main__':
    points = np.random.randint(0, 10, size=7)
    for i in range(2999):
        points=np.vstack((points, np.random.randint(0,10,size=7)))
    output = np.random.randint(0, 10, size=13)
    for i in range(2999):
        output=np.vstack((output, np.random.randint(0,10,size=13)))
    pred = np.random.randint(0, 13, size=3000)
    Findneigh_withpart(points,output, pred, 10, 'ball_tree')