import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils import save_data, load_data


class CamerasCorregistrator():
    def __init__(self, config):
        self.cfg = config
        self.matx_2d_combined = None
        self.errors = None
        self.save_file_name = "corregistration/corregistration_params"

    def _find_corresponding_points(self, points_ref, points_mov, points_ordered=False):
        points_ref = np.array(points_ref)
        points_mov = np.array(points_mov)

        # if points are ordered, search of correspondances is irrelevant
        if points_ordered:
            return points_ref, points_mov

        # calculate distances between reverence and moving points
        idx_pair = - np.ones((points_ref.shape[0],1), dtype='int32')
        idx_dist = np.ones((points_ref.shape[0], points_mov.shape[0]))
        for i in range(points_ref.shape[0]):
            for j in range(points_mov.shape[0]):
                idx_dist[i,j] = np.sum((points_ref[i,:2] - points_mov[j,:2])**2)

        # Bijective transformation
        while not np.all(idx_dist == np.inf):
            i,j = np.where(idx_dist == np.min(idx_dist))
            idx_pair[i[0]] = j[0]
            idx_dist[i[0],:] = np.inf
            idx_dist[:,j[0]] = np.inf

        # assign pairs of points
        idx_valid, idx_not_valid = np.where(idx_pair>=0)
        idx_valid = np.array(idx_valid)
        points_ref_corr = points_ref[idx_valid,:]
        points_mov_corr = points_mov[idx_pair[idx_valid].flatten(),:]
        return points_ref_corr, points_mov_corr

    def _map_affine_approx_2d(self, points_ref, points_mov):
        """Afina aproksimacijska poravnava"""
        points_ref = np.matrix(points_ref)
        points_mov = np.matrix(points_mov)
        points_ref = points_ref.transpose()
        points_mov = points_mov.transpose()
        #U = T*X -> T = U*X'(X*X')^-1
        # matx_2d = np.dot(points_ref, np.linalg.pinv(points_mov))
        matx_2d = points_ref*points_mov.transpose()* np.linalg.inv(points_mov * points_mov.transpose())
        return matx_2d

    def _affine_matx_2d(self, iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
        """Create arbitrary affine transformation matrix"""
        iRot = iRot * np.pi/180
        matx_scale = np.array( ((iScale[0],0,0),(0, iScale[1], 0), (0,0,1) ))
        matx_trans = np.array(((1,0,iTrans[0]), (0,1,iTrans[1]), (0,0,1)))
        matx_rot = np.array(((np.cos(iRot), -np.sin(iRot), 0), (np.sin(iRot), np.cos(iRot), 0), (0,0,1)))
        matx_shear = np.array(((1,iShear[0], 0), (iShear[1], 1, 0), (0,0,1)))
        matx_2d = np.dot(matx_trans,np.dot(matx_shear, np.dot(matx_rot, matx_scale))) # matrix multiplication
        return matx_2d

    def align(self, points_ref, points_mov, eps=1e-6, max_iter=50, plot_progress=False, points_ordered=False):
        """Align interactive corresponding points"""
        points_ref = points_ref.to_numpy()
        points_mov = points_mov.to_numpy()

        matrices = []; errors = []; idx = 0
        if plot_progress:
            points_mov_orig = points_mov
            fig = plt.figure()
            ax = fig.add_subplot(111)

        while True:
            points_ref_corr, points_mov_corr = self._find_corresponding_points(points_ref, points_mov, points_ordered)
            matx_2d_combined = self._map_affine_approx_2d(points_ref_corr, points_mov_corr)
            points_mov = np.dot(points_mov, matx_2d_combined.transpose())

            matrices.append(matx_2d_combined)
            errors.append(np.sqrt(np.sum((points_ref_corr[:,:2] - points_mov_corr[:,:2])**2)))
            idx = idx + 1

            # check for convergence
            matx_diff = np.abs(matx_2d_combined - self._affine_matx_2d())
            if idx > max_iter or np.all(matx_diff < eps):
                break

        matx_2d_combined = self._affine_matx_2d() #initialize with identity matrix
        for matx_2d in matrices:
            if plot_progress:
                points_mov_corr = np.dot(points_mov_orig, matx_2d_combined.transpose())
                ax.clear()
                ax.plot(points_ref[:,0], points_ref[:,1], 'ob')
                ax.plot(points_mov_corr[:,0], points_mov_corr[:,1],'om')
                fig.canvas.draw()
                plt.pause(1)

            # multiply all matrices to get the final transformation
            matx_2d_combined = np.dot(matx_2d, matx_2d_combined)

        self.matx_2d_combined = matx_2d_combined
        self.errors = errors

    def transform(self, points):
        """Transform points"""
        points_transformed = np.dot(points.to_numpy(), self.matx_2d_combined.transpose())
        points_transformed = pd.DataFrame(points_transformed.astype("int"), columns=["x","y","z"])
        return points_transformed

    def save_params(self):
        params = {"matx_2d": self.matx_2d_combined,
                  "errors": self.errors}
        save_data(self.cfg, data=params, data_file_name=self.save_file_name)

    def load_params(self):
        params = load_data(self.cfg, data_file_name=self.save_file_name)
        self.matx_2d_combined = params.matx_2d
        self.errors = params.errors



