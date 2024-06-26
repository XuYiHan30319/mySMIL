import spams
import numpy as np
import cv2
import time
import os
from tqdm import tqdm


class vahadane(object):

    def __init__(
        self,
        STAIN_NUM=2,
        THRESH=0.9,
        LAMBDA1=0.01,
        LAMBDA2=0.01,
        ITER=100,
        fast_mode=0,
        getH_mode=0,
    ):
        self.STAIN_NUM = STAIN_NUM
        self.THRESH = THRESH
        self.LAMBDA1 = LAMBDA1
        self.LAMBDA2 = LAMBDA2
        self.ITER = ITER
        self.fast_mode = fast_mode  # 0: normal; 1: fast
        self.getH_mode = getH_mode  # 0: spams.lasso; 1: pinv;

    def show_config(self):
        print("STAIN_NUM =", self.STAIN_NUM)
        print("THRESH =", self.THRESH)
        print("LAMBDA1 =", self.LAMBDA1)
        print("LAMBDA2 =", self.LAMBDA2)
        print("ITER =", self.ITER)
        print("fast_mode =", self.fast_mode)
        print("getH_mode =", self.getH_mode)

    def getV(self, img):
        I0 = img.reshape((-1, 3)).T
        I0[I0 == 0] = 1
        V0 = np.log(255 / I0)
        img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mask = (img_LAB[:, :, 0] / 255 < self.THRESH).astype(bool)
        I = img[mask].reshape((-1, 3)).T
        I[I == 0] = 1
        V = np.log(255 / I)
        return V0, V

    def getW(self, V):
        W = spams.trainDL(
            np.asfortranarray(V),
            K=self.STAIN_NUM,
            lambda1=self.LAMBDA1,
            iter=self.ITER,
            mode=2,
            modeD=0,
            posAlpha=True,
            posD=True,
            verbose=False,
        )
        W = W / np.linalg.norm(W, axis=0)[None, :]
        if W[0, 0] < W[0, 1]:
            W = W[:, [1, 0]]
        return W

    def getH(self, V, W):
        if self.getH_mode == 0:
            H = spams.lasso(
                np.asfortranarray(V),
                np.asfortranarray(W),
                mode=2,
                lambda1=self.LAMBDA2,
                pos=True,
                verbose=False,
            ).toarray()
        elif self.getH_mode == 1:
            H = np.linalg.pinv(W).dot(V)
            H[H < 0] = 0
        else:
            H = 0
        return H

    def stain_separate(self, img):
        start = time.time()
        if self.fast_mode == 0:
            V0, V = self.getV(img)
            W = self.getW(V)
            H = self.getH(V0, W)
        elif self.fast_mode == 1:
            m = img.shape[0]
            n = img.shape[1]
            grid_size_m = int(m / 5)
            lenm = int(m / 20)
            grid_size_n = int(n / 5)
            lenn = int(n / 20)
            W = np.zeros((81, 3, self.STAIN_NUM)).astype(np.float64)
            for i in range(0, 4):
                for j in range(0, 4):
                    px = (i + 1) * grid_size_m
                    py = (j + 1) * grid_size_n
                    patch = img[px - lenm : px + lenm, py - lenn : py + lenn, :]
                    V0, V = self.getV(patch)
                    W[i * 9 + j] = self.getW(V)
            W = np.mean(W, axis=0)
            V0, V = self.getV(img)
            H = self.getH(V0, W)
        # print("stain separation time:", time.time() - start, "s")
        return W, H

    def SPCN(self, img, Ws, Hs, Wt, Ht):
        Hs_RM = np.percentile(Hs, 99)
        Ht_RM = np.percentile(Ht, 99)
        Hs_norm = Hs * Ht_RM / Hs_RM
        Vs_norm = np.dot(Wt, Hs_norm)
        Is_norm = 255 * np.exp(-1 * Vs_norm)
        I = Is_norm.T.reshape(img.shape).astype(np.uint8)
        return I


if __name__ == "__main__":
    path = "../../data/pathology_img_data"
    target_image = cv2.imread("../../data/target.png")
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # 先收集所有文件
    all_files = []
    for folder_name, _, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(folder_name, file))

    # 处理所有文件并显示整体进度
    for img_path in tqdm(all_files, desc="Processing images"):
        # 初始化 vahadane 对象
        vahadane_obj = vahadane()

        source_image = cv2.imread(img_path)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        # 分离源图片和目标图片的染色基矩阵和染色浓度矩阵
        Ws, Hs = vahadane_obj.stain_separate(source_image)
        Wt, Ht = vahadane_obj.stain_separate(target_image)

        # 将源图片的染色浓度矩阵和目标图片的染色基矩阵进行匹配
        normalized_image = vahadane_obj.SPCN(source_image, Ws, Hs, Wt, Ht)

        # 保存匹配后的图片
        normalized_image_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR)
        # 覆盖原图片
        cv2.imwrite(img_path, normalized_image_bgr)
